# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_framework.data.generator.processors.base import (
    BaseVLMProcessor,
    convert_string_content_to_list_content,
    maybe_get_max_pixels_from_images_kwargs,
    maybe_parse_video_content,
)
from cosmos_framework.utils.generator.structured_conversation import (
    REASONING_PREFILL_KEY,
    assistant_body_spans,
    restore_agentic_messages,
    restrict_assistant_loss,
)

# Set by the reasoner video decoder on media (and copied onto chat content) whose frames it already
# smart-resized to the sample's video token budget.
VIDEO_BUDGET_RESIZED_KEY = "budget_resized"


def _is_local_qwen35_snapshot(path: str) -> bool:
    """Recognize Qwen3.5 metadata independently of a snapshot's directory name."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open() as config_file:
        config = json.load(config_file)
    return config.get("model_type") in {"qwen3_5", "qwen3_5_moe"}


class Qwen3VLProcessor(BaseVLMProcessor):
    """Wrapper around the HuggingFace ``AutoProcessor`` for Qwen3-VL models."""

    # Qwen3-VL does not expose a single vision-end token in its vocabulary, so
    # leave ``vision_end_id`` unset (None) rather than the legacy ``</img>``,
    # which silently resolved to the UNK token id.
    VISION_END_TOKEN: Optional[str] = None
    USES_SOURCE_VIDEO_TIMESTAMPS: bool = True
    USES_TEMPORAL_PATCH_TIMESTAMPS: bool = True

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-2B-Init",
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "bucket4",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name=name, credentials=credentials, bucket=bucket, cache_dir=cache_dir)
        # Helper attributes consumed by the dataloader video decoding path.
        shortest_edge = self.processor.image_processor.size["shortest_edge"]
        self.min_height_width = int(np.sqrt(shortest_edge))
        self.patch_size = self.processor.video_processor.patch_size
        self.temporal_patch_size = self.processor.video_processor.temporal_patch_size
        self.merge_size = self.processor.video_processor.merge_size
        self.use_smart_resize = True
        normalized_name = f"{name} {type(self.processor).__name__}".lower()
        self.retain_mm_token_type_ids: bool = "qwen3.5" in normalized_name or "qwen3_5" in normalized_name or _is_local_qwen35_snapshot(name)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True,
        **kwargs,
    ):
        """
        Return:
            inputs: dict
                input_ids: torch.Tensor, shape: (N_token)
                attention_mask: torch.Tensor, shape: (N_token)
                texts: str, the raw text
                image_sizes: torch.Tensor, shape (N_img, 2)
                pixel_values: torch.Tensor, shape (N_img_patch, 3, 224, 224)
        """
        assert tokenize, "tokenize must be True"
        assert return_tensors == "pt", "return_tensors must be pt"
        # Note: this tokenizer does not support "content": str, it always expect "content" entry to be a list of dicts
        messages, stored_tools = restore_agentic_messages(messages)
        messages = convert_string_content_to_list_content(messages)
        tools = kwargs.pop("tools", stored_tools)
        if stored_tools is not None and tools != stored_tools:
            raise ValueError("Explicit tools disagree with the stored tool menu")
        # ``add_vision_id`` is a chat-template variable ("Picture N: " / "Video N: " labels). TokenizeData requests it
        # only when its opt-in ``add_vision_id`` is set, so existing checkpoints keep seeing unlabeled media; it is
        # forwarded only when set because HF warns about it on calls without media.
        template_kwargs = {"add_vision_id": True} if kwargs.pop("add_vision_id", False) else {}
        max_pixels, min_pixels = maybe_get_max_pixels_from_images_kwargs(messages)

        if max_pixels is not None:
            min_pixels = min_pixels if min_pixels is not None else self.processor.image_processor.size["shortest_edge"]
            kwargs["images_kwargs"] = {"max_pixels": max_pixels, "min_pixels": min_pixels}

        num_video, video_fps, video_total_num_frames, video_frames_indices = maybe_parse_video_content(messages)
        if num_video > 0:
            # Here we add the args to avoid the error:
            # File "/invalid_dir", line 321, in _decode_and_sample_videos
            #     raise ValueError(
            # ValueError: Sampling frames from a list of images is not supported! Set `do_sample_frames=False`.
            video_metadata = [
                dict(fps=fps, total_num_frames=total_num_frames, frames_indices=frames_indices)
                for fps, total_num_frames, frames_indices in zip(
                    video_fps,
                    video_total_num_frames,
                    video_frames_indices,
                )
            ]
            kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": video_metadata[0] if num_video == 1 else video_metadata,
            }
            if self._videos_are_budget_resized(messages):
                # The HF resize would re-apply video_processor.size["longest_edge"] (12,288 tokens per video for the
                # released Qwen3-VL / Qwen3.5 configs) and shrink frames the decoder already sized to the budget.
                kwargs["videos_kwargs"]["do_resize"] = False
        # Both Transformers 4.x and 5.x forward modality kwargs directly to
        # the processor. Nesting them in ``processor_kwargs`` silently loses
        # frame indices, FPS and the request to keep already-sampled frames.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors=return_tensors,
            tools=tools,
            **template_kwargs,
            **kwargs,
        )

        # Convert batch features into single features
        # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
        sequence_keys = ["input_ids", "attention_mask"]
        if getattr(self, "retain_mm_token_type_ids", False):
            sequence_keys.append("mm_token_type_ids")
        else:
            inputs.pop("mm_token_type_ids", None)
        for key in sequence_keys:
            if key in inputs:
                inputs[key] = inputs[key][0]  # [N_token]
        if getattr(self, "retain_mm_token_type_ids", False) and (
            "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape
        ):
            # Rebuild after the HF processor expands media placeholders into tokens.
            inputs["mm_token_type_ids"] = self.build_mm_token_type_ids(inputs["input_ids"])
        return inputs

    def _videos_are_budget_resized(self, messages: list[dict]) -> bool:
        """Whether the reasoner video decoder already sized every video of the conversation to its token budget.

        Videos from any other producer (e.g. raw clips passed by the VFM model) keep the HF resize.

        Raises:
            ValueError: Only some videos are decoder-sized. HF takes one resize setting per call, so it would either
                re-cap the decoder-sized videos below their token budget or leave the others unresized.
        """
        flags = [
            isinstance(content.get("video"), list) and bool(content.get(VIDEO_BUDGET_RESIZED_KEY, False))
            for message in messages
            if isinstance(message["content"], list)
            for content in message["content"]
            if content.get("type", "") == "video"
        ]
        if not any(flags):
            return False
        if not all(flags):
            raise ValueError("Either every video of a conversation is sized by the reasoner decoder or none is")
        return True

    def build_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Identify text, image and video tokens in the final expanded sequence."""
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        if self.image_token_id is not None:
            mm_token_type_ids[input_ids == self.image_token_id] = 1
        if self.video_token_id is not None:
            mm_token_type_ids[input_ids == self.video_token_id] = 2
        return mm_token_type_ids

    def apply_supervision_overrides(
        self,
        tokens: torch.Tensor | list[int],
        mask: torch.Tensor | list[bool],
        messages: list[dict[str, Any]],
    ) -> torch.Tensor | list[bool]:
        """Preserve plain-message masks while honoring structured source supervision."""
        structured = any(
            any(key in message for key in ("tools", "tool_calls", "loss", REASONING_PREFILL_KEY))
            or message.get("role") == "tool"
            for message in messages
        )
        if not structured:
            return mask
        values = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
        # Structured messages may merge source-leading whitespace into the header
        # token. Repair that boundary without replacing processor-specific masks
        # (in particular, Nemo Chat's generation-supplied thinking prefix).
        losses = (
            mask.detach().cpu().numpy().copy()
            if isinstance(mask, torch.Tensor)
            else np.asarray(mask, dtype=bool).copy()
        )
        tokenizer = self.processor.tokenizer
        frame_starts = np.flatnonzero(values == tokenizer.convert_tokens_to_ids("<|im_start|>"))
        header_size = 1 + len(tokenizer.encode("assistant\n", add_special_tokens=False))
        for start, end in assistant_body_spans(values, tokenizer):
            frame_start = int(frame_starts[np.searchsorted(frame_starts, start) - 1])
            losses[start : min(frame_start + header_size, end)] = True
        restricted = restrict_assistant_loss(values, losses, messages, self.processor.tokenizer)
        return torch.from_numpy(restricted) if isinstance(tokens, torch.Tensor) else restricted.tolist()

    def add_assistant_tokens_mask(self, tokens: list[int] | torch.Tensor) -> list[bool] | torch.Tensor:
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack(
                [self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])]
            )  # [B,N_token]
            assert mask.shape == tokens.shape
            return mask
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1

        # Constants defining the ChatML boundaries.
        BOS_TOKEN = "<|im_start|>"
        EOS_TOKEN = "<|im_end|>"
        ROLE = "assistant"
        END_OFFSET = 1

        # Match tokenizer-rendered headers instead of assuming that the role is
        # one token. Some supported tokenizers render ``assistant`` as several
        # BPE tokens followed by a newline token.
        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        role_id = self.processor.tokenizer.convert_tokens_to_ids(ROLE)
        full_role_header_ids = self.processor.tokenizer.encode(f"{BOS_TOKEN}{ROLE}\n", add_special_tokens=False)
        role_header_ids = self.processor.tokenizer.encode(f"{ROLE}\n", add_special_tokens=False)
        role_ids = self.processor.tokenizer.encode(ROLE, add_special_tokens=False)

        # Locate all positions where the start and end markers appear.
        start_indices = np.where(np_tokens == bos_token_id)[0]
        end_indices = np.where(np_tokens == eos_token_id)[0]

        # Initialize the mask with False values.
        masks = np.zeros_like(np_tokens, dtype=bool)
        assert len(start_indices) == len(end_indices)
        # For each pair of bos/eos, check if the role is assistant and apply
        # the mask to the response plus the closing boundary.
        for start, end in zip(start_indices, end_indices):
            content_start = None
            full_rendered_header = np.asarray(full_role_header_ids, dtype=np_tokens.dtype)
            header_end = start + len(full_rendered_header)
            if header_end <= end and np.array_equal(np_tokens[start:header_end], full_rendered_header):
                content_start = header_end
            else:
                rendered_header = np.asarray(role_header_ids, dtype=np_tokens.dtype)
                header_end = start + 1 + len(rendered_header)
                if header_end <= end and np.array_equal(np_tokens[start + 1 : header_end], rendered_header):
                    content_start = header_end
            if content_start is None and start + 1 < end and np_tokens[start + 1] == role_id:
                content_start = start + 3
            elif content_start is None:
                rendered_role = np.asarray(role_ids, dtype=np_tokens.dtype)
                role_end = start + 1 + len(rendered_role)
                if role_end < end and np.array_equal(np_tokens[start + 1 : role_end], rendered_role):
                    content_start = role_end + 1
            if content_start is None or content_start > end:
                continue
            masks[content_start : end + END_OFFSET] = True

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()


if __name__ == "__main__":
    """
    PYTHONPATH=. python3 cosmos_framework/data/generator/processors/qwen3vl_processor.py

    For image, expected output:
        input_ids: type: <class 'torch.Tensor'>      shape: torch.Size([2772])
        attention_mask: type: <class 'torch.Tensor'> shape: torch.Size([2772])
        pixel_values: type: <class 'torch.Tensor'>   shape: torch.Size([11008, 1536])
        image_grid_thw: tensor([[  1,  86, 128]])
        num_image_token_id_tokens: 2752
        decoded_assistant_tokens: Paris<|im_end|>

    For video, expected output:
        input_ids: type: <class 'torch.Tensor'>      shape: torch.Size([5538])
        pixel_values_videos: type: <class 'torch.Tensor'> shape: torch.Size([22016, 1536])
        video_grid_thw: tensor([[  2,  86, 128]])
        num_video_token_id_tokens: 5504
        decoded_assistant_tokens: Paris<|im_end|>
    """
    processor = Qwen3VLProcessor("Qwen/Qwen3-VL-2B-Init")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": ["https://invalid_url"] * 4,
                    "fps": 12,
                },
                {"type": "text", "text": "What is the capital of France?"},
            ],
        },
        {"role": "assistant", "content": "Paris"},
    ]
    inputs = processor.apply_chat_template(messages)
    input_ids = inputs["input_ids"]
    decoded_text = processor.decode(input_ids, skip_special_tokens=True)
    print(decoded_text)
    print(inputs.keys())
    for k, v in inputs.items():
        print(f"{k}: type: {type(v)}")
        if isinstance(v, torch.Tensor):
            print(f"shape: {v.shape}")
        if "grid" in k:
            print(f"{k}: {v}")
    num_image_token_id_tokens = inputs["input_ids"] == processor.image_token_id
    print(f"num_image_token_id_tokens: {num_image_token_id_tokens.sum()}")
    num_video_token_id_tokens = inputs["input_ids"] == processor.video_token_id
    print(f"num_video_token_id_tokens: {num_video_token_id_tokens.sum()}")

    assistant_tokens_mask = processor.add_assistant_tokens_mask(inputs["input_ids"])
    print(f"assistant_tokens_mask: {assistant_tokens_mask.sum()}")
    assistant_tokens = inputs["input_ids"][assistant_tokens_mask]
    print(f"assistant_tokens: {assistant_tokens}")
    decoded_assistant_tokens = processor.decode(assistant_tokens, skip_special_tokens=False)
    print(f"decoded_assistant_tokens: {decoded_assistant_tokens}")

    print("done")
