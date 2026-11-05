# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HF VLM shaping for image-conditioned clipped GRPO and same-token OPD.

This module deliberately implements only Posttrain's exercised online-RL surface. It accepts text
or conversation prompts, renders them with ``AutoProcessor``, attaches image media to vLLM
requests, and reconstructs the same processor inputs for policy scoring. Video, SFT, DPO, and
base64 helpers from the original Cosmos-RL implementation are intentionally absent until a
Posttrain entrypoint owns and tests those contracts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from cosmos_framework.posttrain.data.packer.base import (
    DataPacker,
    collapse_contiguous_token_runs,
)
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput
from cosmos_framework.posttrain.data.packer.forward_routing import add_forward_routing
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256, qualified_type
from cosmos_framework.posttrain.utils.log import logger

_VLLM_TOKEN_PROMPT_IMAGE_EXPANSION_MODEL_TYPES = frozenset({"qwen3_vl", "qwen3_vl_moe"})


def _images_to_pil(conversation: Any) -> Any:
    """Copy a conversation and convert ndarray image payloads to ``PIL.Image`` objects."""
    if not isinstance(conversation, list):
        return conversation
    copied: list[Any] = []
    for message in conversation:
        if not isinstance(message, dict):
            copied.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            copied.append(dict(message))
            continue
        copied_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                copied_content.append(part)
                continue
            image = part.get("image")
            if isinstance(image, np.ndarray):
                copied_content.append({**part, "image": Image.fromarray(image.astype(np.uint8))})
            else:
                copied_content.append(dict(part))
        copied.append({**message, "content": copied_content})
    return copied


def _normalize_conversation(value: Any) -> list[dict[str, Any]]:
    """Return one canonical HF image-chat representation without mutating the caller."""
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("HF VLM prompt must be a non-empty string or conversation list")

    messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(value):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError(f"HF VLM conversation message {message_index} must contain a string role")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            raise ValueError(f"HF VLM conversation message {message_index} has unsupported content {content!r}")

        parts: list[dict[str, Any]] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise ValueError(f"HF VLM conversation message {message_index} content {part_index} must be a mapping")
            part_type = part.get("type", "text")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError(
                        f"HF VLM conversation message {message_index} text content {part_index} needs 'text'"
                    )
            elif part_type == "image":
                if "image" not in part:
                    raise ValueError(
                        f"HF VLM conversation message {message_index} image content {part_index} needs 'image'"
                    )
            else:
                raise ValueError(f"HFVLMDataPacker supports text and image content only, got type={part_type!r}")
            parts.append(dict(part))
        messages.append({**message, "content": parts})
    return _images_to_pil(messages)


def _as_conversation(sample: Any) -> list[dict[str, Any]]:
    """Extract and normalize an ``ObsBundle`` prompt or a bare supported prompt value."""
    representations = getattr(sample, "representations", None)
    conversation = sample if representations is None else representations.get("prompt")
    if representations is not None and conversation is None:
        raise ValueError(
            f"VLM packer needs obs.representations['prompt'] (the conversation); got keys {sorted(representations)}"
        )
    return _normalize_conversation(conversation)


def _conversation_images(conversation: list[dict[str, Any]]) -> list[Any]:
    """Return image payloads in chat-template order."""
    return [part["image"] for message in conversation for part in message["content"] if part.get("type") == "image"]


def _first_present(*values: Any) -> Any:
    """Return the first value that is not ``None``."""
    return next((value for value in values if value is not None), None)


class HFVLMDataPacker(DataPacker):
    """Shape HF image-text models for clipped GRPO and same-token OPD requests."""

    def setup(self, name_or_path: str, trust_remote_code: bool = False) -> None:
        """Build the tokenizer, processor, and image-token metadata for one HF model."""
        super().setup(name_or_path, trust_remote_code=trust_remote_code)
        self.hf_processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        self.hf_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)

        vision_config = getattr(self.hf_config, "vision_config", None)
        self.image_token_id = _first_present(
            getattr(self.hf_config, "image_token_id", None),
            getattr(vision_config, "image_token_id", None),
            getattr(self.hf_config, "img_context_token_id", None),
            getattr(self.hf_config, "image_token_index", None),
            getattr(vision_config, "image_token_index", None),
        )
        if self.image_token_id is None:
            raise ValueError(f"Cannot find image token id in hf_config={self.hf_config!r}")
        self.image_token = _first_present(
            getattr(self.hf_processor, "image_token", None),
            getattr(self.hf_config, "img_context_token", None),
        )
        self.model_type = str(getattr(self.hf_config, "model_type", ""))
        logger.info(
            f"Initialized HFVLMDataPacker with image_token_id={self.image_token_id}, model_type={self.model_type}"
        )

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Bind the processor/config fields that determine prompt and image shaping."""

        def _component_sha256(component: Any, *, context: str) -> str | None:
            if component is None:
                return None
            to_dict = getattr(component, "to_dict", None)
            if not callable(to_dict):
                raise TypeError(f"{context} exposes no to_dict()")
            return canonical_json_sha256(to_dict(), context=context)

        config_to_dict = getattr(self.hf_config, "to_dict", None)
        if not callable(config_to_dict):
            raise TypeError("HF VLM model config exposes no to_dict()")
        return {
            "version": 2,
            "processor_type": qualified_type(self.hf_processor),
            "processor_chat_template_sha256": canonical_json_sha256(
                getattr(self.hf_processor, "chat_template", None),
                context="HF VLM processor chat template",
            ),
            "image_processor_sha256": _component_sha256(
                getattr(self.hf_processor, "image_processor", None),
                context="HF VLM image processor config",
            ),
            "hf_config_sha256": canonical_json_sha256(config_to_dict(), context="HF VLM model config"),
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
            "model_type": self.model_type,
        }

    def _prepare_conversation(self, sample: Any) -> list[dict[str, Any]]:
        """Normalize a prompt and remove duplicated legacy image-token text."""
        conversation = _as_conversation(sample)
        if self.image_token is None:
            return conversation
        for message in conversation:
            for part in message["content"]:
                if part.get("type") == "text":
                    part["text"] = part["text"].replace(self.image_token, "")
        return conversation

    def _processor_prompt(
        self,
        conversation: list[dict[str, Any]],
        images: list[Any],
    ) -> tuple[list[int], dict[str, Any]]:
        """Return the processor-owned prompt ids and model inputs for one normalized conversation."""
        text = self.hf_processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        processed = dict(self.hf_processor(**processor_kwargs))
        raw_input_ids = processed.pop("input_ids")  # [1,P]
        first_input_ids = raw_input_ids[0]  # [P]
        prompt_ids = first_input_ids.tolist() if hasattr(first_input_ids, "tolist") else list(first_input_ids)
        processed.pop("attention_mask", None)
        return [int(token_id) for token_id in prompt_ids], processed

    def _rollout_request(
        self,
        sample: Any,
        *,
        prompt_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Build a rendered or exact-token vLLM request with ordered image identity."""
        conversation = self._prepare_conversation(sample)
        if prompt_token_ids is None:
            prompt = self.hf_processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            request: dict[str, Any] = {"prompt": prompt}
        else:
            request = {"prompt_token_ids": [int(token_id) for token_id in prompt_token_ids]}

        images = _conversation_images(conversation)
        if images:
            request["multi_modal_data"] = {"image": images}
            request["multi_modal_uuids"] = {"image": self._image_content_uuids(images)}
        return request

    def get_rollout_input(self, sample: Any) -> dict[str, Any]:
        """Render a prompt and attach its images for vLLM generation."""
        return self._rollout_request(sample)

    def get_pretokenized_rollout_input(self, item: Any, token_ids: list[int]) -> dict[str, Any]:
        """Attach images to a same-token OPD request without changing its realized sequence.

        Qwen3-VL serving re-expands image markers on token prompts, so its processor-owned prompt
        prefix carries one logical marker per image; the completion remains verbatim. Other HF VLMs
        keep the exact ids they received.
        """
        exact_ids = [int(token_id) for token_id in token_ids]
        conversation = self._prepare_conversation(item)
        images = _conversation_images(conversation)
        if not images or getattr(self, "model_type", "") not in _VLLM_TOKEN_PROMPT_IMAGE_EXPANSION_MODEL_TYPES:
            return self._rollout_request(item, prompt_token_ids=exact_ids)
        processor_prompt_ids, _model_inputs = self._processor_prompt(conversation, images)
        if exact_ids[: len(processor_prompt_ids)] != processor_prompt_ids:
            raise ValueError("pretokenized HF Qwen3-VL request does not begin with the processor's exact prompt IDs")
        request_ids = [
            *collapse_contiguous_token_runs(processor_prompt_ids, self.image_token_id),
            *exact_ids[len(processor_prompt_ids) :],
        ]
        return self._rollout_request(item, prompt_token_ids=request_ids)

    @staticmethod
    def _image_content_uuids(images: list[Any]) -> list[str]:
        """Return ordered, content-addressed image ids for cross-rank request validation."""
        uuids: list[str] = []
        for index, item in enumerate(images):
            if hasattr(item, "tobytes"):
                payload = item.tobytes()
                shape = getattr(item, "shape", None)
                if shape is None:
                    shape = getattr(item, "size", None)
                geometry = repr(tuple(shape)) if shape is not None else ""
                mode = str(getattr(item, "mode", "") or "")
                dtype = str(getattr(item, "dtype", "") or "")
                content = hashlib.sha256(payload + geometry.encode() + mode.encode() + dtype.encode()).hexdigest()
            elif isinstance(item, torch.Tensor):
                contiguous = item.detach().contiguous().cpu()
                payload = contiguous.view(torch.uint8).numpy().tobytes()
                metadata = f"{tuple(contiguous.shape)}|{contiguous.dtype}".encode()
                content = hashlib.sha256(payload + metadata).hexdigest()
            else:
                content = hashlib.sha256(repr(item).encode()).hexdigest()
            seed = f"hf-vlm-image-v1\0{index}\0{content}".encode()
            uuids.append(hashlib.sha256(seed).hexdigest())
        return uuids

    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str | list[int] | torch.Tensor,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> RLPolicyInput:
        """Build an image-aware policy input and verify stored rollout prompt ids."""
        if n_ignore_prefix_tokens < 0:
            raise ValueError(f"n_ignore_prefix_tokens must be >= 0, got {n_ignore_prefix_tokens}")
        conversation = self._prepare_conversation(sample)
        images = _conversation_images(conversation)
        processor_prompt_ids, processed = self._processor_prompt(conversation, images)

        if prompt_ids is not None:
            exact_prompt_ids = [int(token_id) for token_id in prompt_ids]
            if exact_prompt_ids != processor_prompt_ids:
                mismatch = next(
                    (
                        index
                        for index, (stored, shaped) in enumerate(zip(exact_prompt_ids, processor_prompt_ids))
                        if stored != shaped
                    ),
                    min(len(exact_prompt_ids), len(processor_prompt_ids)),
                )
                raise ValueError(
                    "stored rollout prompt_ids differ from the VLM processor prefix at index "
                    f"{mismatch}: stored_len={len(exact_prompt_ids)}, "
                    f"processor_len={len(processor_prompt_ids)}"
                )
            policy_prompt_ids = exact_prompt_ids
        else:
            policy_prompt_ids = [int(token_id) for token_id in processor_prompt_ids]
        if not policy_prompt_ids:
            raise ValueError("HF VLM processor produced an empty prompt")

        if isinstance(rollout_output, str):
            completion_ids = self.tokenizer(rollout_output, add_special_tokens=False).input_ids
        elif hasattr(rollout_output, "tolist"):
            completion_ids = rollout_output.tolist()
        else:
            completion_ids = list(rollout_output)
        completion_ids = [int(token_id) for token_id in completion_ids]
        if n_ignore_prefix_tokens > len(completion_ids):
            raise ValueError(
                f"n_ignore_prefix_tokens={n_ignore_prefix_tokens} exceeds completion length {len(completion_ids)}"
            )
        input_ids = policy_prompt_ids + completion_ids
        return RLPolicyInput(
            input_ids=input_ids,
            logprob_masks=self._logprob_masks(
                len(policy_prompt_ids),
                len(completion_ids),
                n_ignore_prefix_tokens,
            ),
            mm_kwargs={key: value for key, value in processed.items() if value is not None} or None,
        )

    @staticmethod
    def _logprob_masks(prompt_len: int, completion_len: int, n_ignore_prefix_tokens: int) -> list[int]:
        """Return next-token-aligned completion masks."""
        return [0] * (prompt_len - 1 + n_ignore_prefix_tokens) + [1] * (completion_len - n_ignore_prefix_tokens) + [0]

    def policy_compute_max_len(self, processed_samples: list[RLPolicyInput]) -> int:
        """Return the longest token sequence in a non-empty policy batch."""
        if not processed_samples:
            raise ValueError("policy_compute_max_len got 0 HF VLM samples")
        return max(len(item.input_ids) for item in processed_samples)

    def policy_collate_fn(
        self,
        processed_samples: list[RLPolicyInput],
        computed_max_len: int,
    ) -> dict[str, Any]:
        """Pad tokens and concatenate per-sample HF image-processor outputs."""
        if not processed_samples:
            raise ValueError("policy_collate_fn got 0 HF VLM samples")
        pad_token_id = int(self.tokenizer.pad_token_id)
        lengths = [min(len(sample.input_ids), computed_max_len) for sample in processed_samples]
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(
                [
                    sample.input_ids[:computed_max_len]
                    + [pad_token_id] * max(0, computed_max_len - len(sample.input_ids))
                    for sample in processed_samples
                ],
                dtype=torch.long,
            ),
            "logprob_masks": torch.tensor(
                [
                    sample.logprob_masks[:computed_max_len] + [0] * max(0, computed_max_len - len(sample.logprob_masks))
                    for sample in processed_samples
                ],
                dtype=torch.bool,
            ),
            "attention_mask": torch.tensor(
                [[position < length for position in range(computed_max_len)] for length in lengths],
                dtype=torch.bool,
            ),
        }

        mm_kwargs = [sample.mm_kwargs or {} for sample in processed_samples]
        side_input_keys = set().union(*(set(values) for values in mm_kwargs))
        for key in sorted(side_input_keys):
            per_sample_values = [sample.get(key) for sample in mm_kwargs]
            values = [value for value in per_sample_values if value is not None]
            if not values:
                continue
            if key == "mm_token_type_ids":
                if len(values) != len(processed_samples):
                    raise ValueError("mm_token_type_ids must be present for every sample or omitted for the batch")
                rows: list[list[int]] = []
                for value in values:
                    ids = value.tolist() if isinstance(value, torch.Tensor) else list(value)
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    rows.append(ids[:computed_max_len] + [0] * max(0, computed_max_len - len(ids)))
                batch[key] = torch.tensor(rows, dtype=torch.long)
            elif all(isinstance(value, torch.Tensor) for value in values):
                batch[key] = torch.cat(values, dim=0)
            else:
                raise TypeError(f"HF VLM processor output {key!r} must contain tensors, got {values!r}")
        add_forward_routing(batch, processed_samples, computed_max_len)
        return batch
