# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Weight-free CPU shaping for i4-native Qwen3-VL dense and MoE policies."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, unquote_to_bytes, urlparse

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from cosmos_framework.posttrain.data.packer.base import (
    DataPacker,
    collapse_contiguous_token_runs,
)
from cosmos_framework.posttrain.data.packer.forward_routing import add_forward_routing
from cosmos_framework.posttrain.schema import ObsBundle
from cosmos_framework.posttrain.utils.identity import (
    canonical_json_sha256,
    qualified_type,
)

_MEDIA_KEYS: tuple[str, ...] = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)


@dataclass
class Qwen3VLPolicyInput:
    """One exact rollout prefix/completion plus its processed media and, when the rollout recorded
    it, the expert routing of the sequence (`data/packer/forward_routing.py`)."""

    input_ids: list[int]
    attention_mask: list[bool]
    logprob_masks: list[int]
    model_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    forward_routing: torch.Tensor | None = None  # [T-1,L,K] integer expert ids | None


@dataclass(frozen=True)
class Qwen3VLCrossTokenTeacherInput:
    """One exact image-only vLLM request plus compact validation metadata."""

    request: dict[str, Any]
    image_count: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    expected_image_marker_widths: tuple[int, ...]
    image_uuids: tuple[str, ...]


def _clone_message_value(value: Any, *, media_field: bool = False) -> Any:
    """Clone message containers without copying heavyweight media handle objects."""
    if media_field:
        return value
    if isinstance(value, dict):
        return {
            key: _clone_message_value(
                item,
                media_field=key in {"image", "image_url", "url", "path", "base64", "video"},
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_message_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_message_value(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 -- opaque media-adjacent metadata may be handle-like
        return value


def _token_ids(value: Any, *, field_name: str) -> list[int]:
    raw = value.tolist() if hasattr(value, "tolist") else list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise ValueError(f"{field_name} must contain one sequence, got batch size {len(raw)}")
        raw = raw[0]
    return [int(token_id) for token_id in raw]


def _to_tensor(value: Any, *, field_name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)  # [*]
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()  # [*]
    if field_name.endswith("grid_thw"):
        tensor = tensor.to(dtype=torch.long)  # [N_media,3]
    return tensor


class Qwen3VLDataPacker(DataPacker):
    """Qwen3-VL chat/media shaping shared by dense and MoE policies."""

    def __init__(
        self,
        *,
        processor: Any | None = None,
        model_config: Any | None = None,
    ) -> None:
        self.processor = processor
        self.model_config = model_config
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.vision_start_token_id: int | None = None
        self.vision_end_token_id: int | None = None
        self.spatial_merge_size: int = 1
        self._processor_identity: dict[str, Any] | None = None

    def setup(self, name_or_path: str, trust_remote_code: bool = False) -> None:
        """Load one processor/config pair, or finish setup of injected test doubles."""
        if not name_or_path and self.processor is None:
            raise ValueError("Qwen3VLDataPacker.setup: empty name_or_path — cannot build a processor")
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        if self.model_config is None:
            from transformers import AutoConfig

            self.model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise TypeError("Qwen3-VL processor exposes no tokenizer")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = self.model_config
        vision_config = getattr(config, "vision_config", None)
        self.image_token_id = getattr(config, "image_token_id", None) or getattr(vision_config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None) or getattr(vision_config, "video_token_id", None)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(config, "vision_end_token_id", None) or getattr(
            vision_config, "vision_end_token_id", None
        )
        self.spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 1))
        if self.image_token_id is None:
            raise ValueError("Qwen3-VL config exposes no image_token_id")
        if self.spatial_merge_size < 1:
            raise ValueError(f"spatial_merge_size must be positive, got {self.spatial_merge_size}")
        self._processor_identity = None

    def _messages_and_media(self, item: ObsBundle) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
        representations = item.representations
        payload = representations.get("messages", representations.get("prompt"))
        if isinstance(payload, Mapping) and "messages" in payload:
            payload = payload["messages"]
        if isinstance(payload, str) and payload:
            # A text prompt is exactly one user turn. Without this, no text source can reach a
            # Qwen3-VL model at all: the shared projection passes a string prompt column through as
            # a string (`roles/env/dataset.py::project_prompt_items`) and the built-in arithmetic
            # source emits strings, so every text dataset died here. Wrapping also routes the prompt
            # through `apply_chat_template` below, which is what an Instruct checkpoint expects. A
            # conversation list (the projection's other accepted shape) arrives here as the messages.
            payload = [{"role": "user", "content": payload}]
        if not isinstance(payload, list) or not all(isinstance(message, Mapping) for message in payload):
            raise ValueError("Qwen3-VL ObsBundle requires representations['messages'] (or list-valued 'prompt')")
        messages = [_clone_message_value(dict(message)) for message in payload]

        image_blocks: list[dict[str, Any]] = []
        video_blocks: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
                message["content"] = content
            if not isinstance(content, list):
                raise ValueError(f"Qwen3-VL message content must be str or list, got {type(content).__name__}")
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError(f"Qwen3-VL content block must be a mapping, got {type(block).__name__}")
                modality = block.get("type")
                if modality in {"image", "image_url"}:
                    image_blocks.append(block)
                elif modality in {"video", "video_frames"}:
                    video_blocks.append(block)

        separate_images = representations.get("images")
        if separate_images is None and "image" in representations:
            separate_images = [representations["image"]]
        separate_videos = representations.get("videos")
        if separate_videos is None and "video" in representations:
            separate_videos = [representations["video"]]
        if separate_images is not None:
            images = list(separate_images)
            if len(images) != len(image_blocks):
                raise ValueError(
                    f"ObsBundle has {len(images)} image handle(s) but {len(image_blocks)} image content block(s)"
                )
            for block, handle in zip(image_blocks, images, strict=True):
                block.pop("image_url", None)
                block.pop("url", None)
                block.pop("path", None)
                block.pop("base64", None)
                block["type"] = "image"
                block["image"] = handle
        else:
            images = []
            for block in image_blocks:
                images.append(self._image_block_handle(block))
        if separate_videos is not None:
            videos = list(separate_videos)
            if len(videos) != len(video_blocks):
                raise ValueError(
                    f"ObsBundle has {len(videos)} video handle(s) but {len(video_blocks)} video content block(s)"
                )
            for block, handle in zip(video_blocks, videos, strict=True):
                block["type"] = "video"
                block["video"] = handle
        else:
            videos = []
            for block in video_blocks:
                if "video" not in block:
                    raise ValueError("video content block carries no video handle")
                videos.append(block["video"])
        return messages, images, videos

    @staticmethod
    def _image_block_handle(block: dict[str, Any]) -> Any:
        for field_name in ("image", "image_url", "url", "path"):
            if field_name in block:
                return block[field_name]
        if "base64" in block:
            encoded = block["base64"]
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("image content block base64 handle must be a non-empty string")
            if encoded.startswith("data:"):
                return encoded
            media_type = block.get("mime_type", block.get("media_type", "image/png"))
            if not isinstance(media_type, str) or not media_type.startswith("image/"):
                raise ValueError("image content block base64 media type must be an image/* string")
            return f"data:{media_type};base64,{encoded}"
        raise ValueError("image content block carries no image/image_url/url/path/base64 handle")

    @staticmethod
    def _processor_kwargs(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Derive one processor contract shared by policy shaping and vLLM rollout requests."""
        # The generic posttrain packer registry is imported by text-only CPU/Ray entrypoints. Keep
        # the optional Cosmos3 vision stack lazy so those imports do not require torchvision; the
        # Qwen packer resolves these helpers only when it actually shapes a Qwen request.
        from cosmos_framework.data.generator.processors.base import (
            maybe_get_max_pixels_from_images_kwargs,
            maybe_parse_video_content,
        )

        kwargs: dict[str, Any] = {}
        max_pixels, min_pixels = maybe_get_max_pixels_from_images_kwargs(messages)
        if max_pixels is not None:
            images_kwargs: dict[str, Any] = {"max_pixels": max_pixels}
            if min_pixels is not None:
                images_kwargs["min_pixels"] = min_pixels
            kwargs["images_kwargs"] = images_kwargs
        num_video, video_fps, total_frames, frame_indices = maybe_parse_video_content(messages)
        if num_video:
            metadata = [
                {"fps": fps, "total_num_frames": total, "frames_indices": indices}
                for fps, total, indices in zip(video_fps, total_frames, frame_indices, strict=True)
            ]
            kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": metadata[0] if num_video == 1 else metadata,
            }
        return kwargs

    @classmethod
    def _vllm_media(
        cls,
        messages: list[dict[str, Any]],
        images: list[Any],
        videos: list[Any],
    ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        """Translate HF chat-template kwargs into vLLM's flat multimodal request contract."""
        hf_kwargs = cls._processor_kwargs(messages)
        vllm_kwargs: dict[str, Any] = dict(hf_kwargs.get("images_kwargs", {}))
        video_kwargs = dict(hf_kwargs.get("videos_kwargs", {}))
        raw_metadata = video_kwargs.pop("video_metadata", None)
        vllm_kwargs.update(video_kwargs)

        video_items: list[Any] = list(videos)
        if raw_metadata is not None:
            metadata = raw_metadata if isinstance(raw_metadata, list) else [raw_metadata]
            if len(metadata) != len(videos):
                raise ValueError(
                    f"Qwen3-VL video metadata has {len(metadata)} row(s) for {len(videos)} video handle(s)"
                )
            # vLLM's Qwen3-VL processor consumes ``(video, metadata)`` items and flat sampling
            # kwargs; HF ``apply_chat_template`` consumes nested ``videos_kwargs`` instead.
            video_items = list(zip(videos, metadata, strict=True))
        elif videos:
            video_blocks = [
                block
                for message in messages
                for block in message["content"]
                if block.get("type") in {"video", "video_frames"}
            ]
            if len(video_blocks) != len(videos):
                raise ValueError(
                    f"Qwen3-VL found {len(video_blocks)} video message block(s) for {len(videos)} handle(s)"
                )
            # Path/URL videos are decoded independently by HF and vLLM. Forward every explicit
            # sampling override from the canonical message so they select the same frames and
            # therefore the same timestamp/placeholder expansion.
            for field_name in ("fps", "num_frames", "min_frames", "max_frames"):
                values = [block.get(field_name) for block in video_blocks]
                if not any(value is not None for value in values):
                    continue
                if any(value is None for value in values):
                    raise ValueError(f"Qwen3-VL multi-video {field_name} must be present on every video block or none")
                vllm_kwargs[field_name] = values[0] if len(values) == 1 else values

        multi_modal_data: dict[str, list[Any]] = {}
        if images:
            multi_modal_data["image"] = images
        if video_items:
            multi_modal_data["video"] = video_items
        return multi_modal_data, vllm_kwargs

    def _resolved_chat_template(self) -> str:
        processor_template = getattr(self.processor, "chat_template", None)
        if isinstance(processor_template, str) and processor_template:
            return processor_template
        tokenizer_template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(tokenizer_template, str) and tokenizer_template:
            return tokenizer_template
        raise TypeError("Qwen3-VL teacher processor/tokenizer exposes no non-empty string chat template")

    def processor_semantics_identity(self) -> dict[str, Any]:
        """Return the immutable processor contract (``PackerProcessorIdentity``).

        Consumed by the OPD cross-token teacher identity and, through
        ``validation_semantics_identity``, by held-out validation. It binds the video processor as
        well as the image processor, so identities persisted by either consumer before that binding
        refuse to resume on Qwen3-VL: two live processors differing only in video shaping must never
        share an identity.
        """
        if self._processor_identity is not None:
            return copy.deepcopy(self._processor_identity)
        if self.processor is None or self.model_config is None or self.tokenizer is None:
            raise TypeError("Qwen3-VL processor identity requires a setup-complete processor and model config")
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise TypeError("Qwen3-VL processor tokenizer exposes no get_vocab()")
        vocab = get_vocab()
        if not isinstance(vocab, dict) or any(
            not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
            for token, token_id in vocab.items()
        ):
            raise TypeError("Qwen3-VL processor tokenizer.get_vocab() must return dict[str, int]")
        resolved_chat_template = self._resolved_chat_template()
        image_processor = getattr(self.processor, "image_processor", None)
        image_processor_config = None
        if image_processor is not None:
            to_dict = getattr(image_processor, "to_dict", None)
            if not callable(to_dict):
                raise TypeError("Qwen3-VL image processor exposes no to_dict()")
            image_processor_config = canonical_json_sha256(
                to_dict(),
                context="Qwen3-VL image processor config",
            )
        # The Qwen processor exposes its video processor as a component separate from the image
        # processor, and frame sampling/resize configuration lives there. Without binding it, two
        # live processors that differ only in video shaping declare the same identity and a pending
        # held-out validation can resume under changed model inputs. Text-only processors expose
        # no video processor; both fields then stay ``None`` so their identity remains stable.
        video_processor = getattr(self.processor, "video_processor", None)
        video_processor_type = None
        video_processor_config = None
        if video_processor is not None:
            to_dict = getattr(video_processor, "to_dict", None)
            if not callable(to_dict):
                raise TypeError("Qwen3-VL video processor exposes no to_dict()")
            video_processor_type = qualified_type(video_processor)
            video_processor_config = canonical_json_sha256(
                to_dict(),
                context="Qwen3-VL video processor config",
            )
        self._processor_identity = {
            "processor_type": qualified_type(self.processor),
            "tokenizer_type": qualified_type(self.tokenizer),
            "tokenizer_name_or_path": str(getattr(self.tokenizer, "name_or_path", "")),
            "tokenizer_vocab_sha256": canonical_json_sha256(vocab, context="Qwen3-VL teacher tokenizer vocab"),
            "chat_template_sha256": hashlib.sha256(resolved_chat_template.encode("utf-8")).hexdigest(),
            "image_processor_sha256": image_processor_config,
            "video_processor_type": video_processor_type,
            "video_processor_sha256": video_processor_config,
            "model_type": str(getattr(self.model_config, "model_type", "")),
            "image_token_id": self.image_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        }
        return copy.deepcopy(self._processor_identity)

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Declare the generation semantics of every configured packer field.

        Held-out validation (roles/rollout/validation.py, ``_packer_semantics_identity``)
        refuses a packer whose configured instance fields carry undeclared generation
        semantics. ``processor_semantics_identity`` already binds ``processor``,
        ``model_config``, and the ``_processor_identity`` cache through the processor and
        tokenizer types, the resolved chat template hash, the image processor config hash,
        the video processor type and config hash (``None`` for text-only processors), the
        model type, and the image/vision-start/vision-end token IDs. The two shaping
        fields it does not cover, ``video_token_id`` and ``spatial_merge_size``, are
        declared explicitly. The tokenizer artifact itself is bound separately by the
        validation framework.
        """
        return {
            "processor": self.processor_semantics_identity(),
            "video_token_id": self.video_token_id,
            "spatial_merge_size": self.spatial_merge_size,
        }

    @staticmethod
    def _data_uri_payload(value: str) -> tuple[str, bytes] | None:
        if not value.startswith("data:"):
            return None
        header, separator, encoded = value[5:].partition(",")
        if not separator:
            raise ValueError("teacher image data URI has no comma separator")
        parts = header.split(";")
        media_type = parts[0] or "text/plain"
        try:
            payload = base64.b64decode(encoded, validate=True) if "base64" in parts[1:] else unquote_to_bytes(encoded)
        except (binascii.Error, ValueError) as error:
            raise ValueError("teacher image data URI payload is malformed") from error
        return media_type, payload

    @staticmethod
    def _snapshot_pil_image(image: Image.Image, *, location: str) -> Image.Image:
        """Apply Transformers path-loading semantics and detach pixels from the source buffer."""
        try:
            transposed = ImageOps.exif_transpose(image)
            rgb = transposed.convert("RGB")
            rgb.load()
            return rgb.copy()
        except (OSError, ValueError) as error:
            raise ValueError(f"{location} could not be normalized to one RGB image") from error

    @classmethod
    def _decode_image_bytes(cls, payload: bytes, *, location: str) -> Image.Image:
        """Decode one already-read payload without leaving a lazy BytesIO-backed image."""
        try:
            with Image.open(io.BytesIO(payload)) as source:
                return cls._snapshot_pil_image(source, location=location)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"{location} bytes do not decode as a supported image") from error

    @staticmethod
    def _read_local_image_bytes(path: str, *, location: str) -> bytes:
        """Read one local image before passing an immutable snapshot to vLLM."""
        expanded = os.path.realpath(os.path.expanduser(path))
        try:
            with open(expanded, "rb") as handle:
                return handle.read()
        except OSError as error:
            raise ValueError(f"{location} local media file is not readable: {path!r}") from error

    def _materialize_vllm_image(self, value: Any, *, location: str) -> tuple[Any, str]:
        """Snapshot one image and return the content digest used only for vLLM cache safety."""
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            data_uri = self._data_uri_payload(value)
            if data_uri is not None:
                media_type, payload = data_uri
                if not media_type.startswith("image/"):
                    raise ValueError(f"{location} data URI media type {media_type!r} is not image/*")
            else:
                parsed = urlparse(value)
                if parsed.scheme == "file":
                    if parsed.netloc not in ("", "localhost"):
                        raise ValueError(f"{location} non-local file URI host {parsed.netloc!r} is unsupported")
                    local_path = unquote(parsed.path)
                elif parsed.scheme:
                    raise ValueError(
                        f"{location} remote URI scheme {parsed.scheme!r} has no verifiable raw-media content hash; "
                        "stage it as a local file or data URI"
                    )
                else:
                    local_path = value
                payload = self._read_local_image_bytes(local_path, location=location)
            image = self._decode_image_bytes(payload, location=location)
            return image, hashlib.sha256(payload).hexdigest()
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
            image = self._decode_image_bytes(payload, location=location)
            return image, hashlib.sha256(payload).hexdigest()
        if isinstance(value, Image.Image):
            snapshot = self._snapshot_pil_image(value, location=location)
            header = f"PIL:RGB:{snapshot.width}x{snapshot.height}:".encode("utf-8")
            payload = header + snapshot.tobytes()
        elif isinstance(value, np.ndarray):
            snapshot = np.ascontiguousarray(value).copy()
            header = f"numpy:{snapshot.dtype}:{snapshot.shape}:".encode("utf-8")
            payload = header + snapshot.tobytes()
        elif isinstance(value, torch.Tensor):
            snapshot = value.detach().resolve_conj().resolve_neg().cpu().contiguous().clone()  # [*]
            byte_tensor = snapshot.view(torch.uint8).reshape(-1)  # [N_byte]
            header = f"torch:{snapshot.dtype}:{tuple(snapshot.shape)}:".encode("utf-8")
            payload = header + byte_tensor.numpy().tobytes()
        else:
            raise TypeError(
                f"{location} must be local/data-URI/bytes/PIL/numpy/torch image data, got {qualified_type(value)}"
            )
        return snapshot, hashlib.sha256(payload).hexdigest()

    def _materialize_vllm_images(self, images: list[Any]) -> tuple[list[Any], tuple[str, ...]]:
        payloads: list[Any] = []
        image_uuids: list[str] = []
        for index, image in enumerate(images):
            payload, content_sha256 = self._materialize_vllm_image(image, location=f"vLLM image[{index}]")
            cache_seed = f"qwen3-vl-image-v1\0{index}\0{content_sha256}".encode("utf-8")
            payloads.append(payload)
            image_uuids.append(hashlib.sha256(cache_seed).hexdigest())
        return payloads, tuple(image_uuids)

    @staticmethod
    def _bind_image_payloads(messages: list[dict[str, Any]], images: list[Any]) -> None:
        """Replace every message reference with the exact snapshotted object sent to vLLM."""
        image_index = 0
        for message in messages:
            for block in message["content"]:
                if block.get("type") not in {"image", "image_url"}:
                    continue
                if image_index >= len(images):
                    raise ValueError("Qwen3-VL message/image cardinality changed while binding teacher payloads")
                for field_name in ("image_url", "url", "path", "base64"):
                    block.pop(field_name, None)
                block["type"] = "image"
                block["image"] = images[image_index]
                image_index += 1
        if image_index != len(images):
            raise ValueError("Qwen3-VL message/image cardinality changed while binding teacher payloads")

    def render_cross_token_multimodal_prompt(self, item: ObsBundle, *, enable_thinking: bool) -> str:
        """Render canonical image/text messages with this teacher's own processor template."""
        messages, _images, videos = self._messages_and_media(item)
        if videos:
            raise ValueError("cross-token multimodal teacher supports ordered images only; video is unsupported")
        chat_template = self._resolved_chat_template()
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            chat_template=chat_template,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TypeError("Qwen3-VL teacher processor chat template must return a non-empty string")
        return rendered

    def get_cross_token_multimodal_teacher_input(
        self,
        item: ObsBundle,
        token_ids: list[int],
    ) -> Qwen3VLCrossTokenTeacherInput:
        """Attach ordered image snapshots to exact teacher-tokenizer IDs without re-tokenizing them."""
        messages, images, videos = self._messages_and_media(item)
        if videos:
            raise ValueError("cross-token multimodal teacher supports ordered images only; video is unsupported")
        if self.image_token_id is None or self.vision_start_token_id is None or self.vision_end_token_id is None:
            raise ValueError("cross-token multimodal teacher requires image, vision-start, and vision-end token IDs")
        exact_ids = [int(token_id) for token_id in token_ids]
        logical_widths = self._image_marker_widths(exact_ids)
        if logical_widths != (1,) * len(images):
            raise ValueError(
                "cross-token multimodal teacher requires exactly one logical image marker per image before vLLM"
            )
        materialized_images, image_uuids = self._materialize_vllm_images(images)
        self._bind_image_payloads(messages, materialized_images)
        expected_image_marker_widths = self._expected_image_marker_widths(messages, materialized_images)
        multi_modal_data, vllm_kwargs = self._vllm_media(messages, materialized_images, [])
        if set(multi_modal_data) - {"image"}:
            raise ValueError("cross-token multimodal teacher request may contain image media only")
        request: dict[str, Any] = {"prompt_token_ids": exact_ids}
        if multi_modal_data:
            request["multi_modal_data"] = multi_modal_data
            request["multi_modal_uuids"] = {"image": list(image_uuids)}
        if vllm_kwargs:
            request["mm_processor_kwargs"] = vllm_kwargs
        return Qwen3VLCrossTokenTeacherInput(
            request=request,
            image_count=len(materialized_images),
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            expected_image_marker_widths=expected_image_marker_widths,
            image_uuids=image_uuids,
        )

    def _request(self, item: ObsBundle, *, prompt_token_ids: list[int] | None = None) -> dict[str, Any]:
        messages, images, videos = self._messages_and_media(item)
        materialized_images, image_uuids = self._materialize_vllm_images(images)
        self._bind_image_payloads(messages, materialized_images)
        if prompt_token_ids is None:
            chat_template = self._resolved_chat_template()
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
            request: dict[str, Any] = {"prompt": prompt}
        else:
            exact_ids = [int(token_id) for token_id in prompt_token_ids]
            processed_prompt_ids, _attention, model_inputs = self._processor_inputs_from_messages(
                messages,
                materialized_images,
                videos,
            )
            self._validate_processed_media(
                prompt_ids=exact_ids,
                images=materialized_images,
                videos=videos,
                model_inputs=model_inputs,
            )
            if exact_ids[: len(processed_prompt_ids)] != processed_prompt_ids:
                raise ValueError("pretokenized Qwen3-VL request does not begin with the processor's exact prompt IDs")
            # vLLM re-applies the Qwen3-VL placeholder replacement to token prompts: the i-th
            # ``<|image_pad|>`` match becomes the i-th image's full grid-derived run while every other
            # pad stays, so an already-expanded prompt comes back ``N - 1`` tokens longer per image.
            # Submit one logical marker per image instead; the engine expands it back to exactly
            # ``exact_ids`` and the same-token teacher proves that before scoring. Only the
            # processor's own prompt region is collapsed; the completion tail is sent verbatim.
            request = {
                "prompt_token_ids": [
                    *collapse_contiguous_token_runs(processed_prompt_ids, self.image_token_id),
                    *exact_ids[len(processed_prompt_ids) :],
                ]
            }
        multi_modal_data, vllm_kwargs = self._vllm_media(messages, materialized_images, videos)
        if multi_modal_data:
            request["multi_modal_data"] = multi_modal_data
        if materialized_images:
            request["multi_modal_uuids"] = {"image": list(image_uuids)}
        if vllm_kwargs:
            request["mm_processor_kwargs"] = vllm_kwargs
        return request

    def get_rollout_input(self, item: ObsBundle) -> dict[str, Any]:
        return self._request(item)

    def get_pretokenized_rollout_input(self, item: ObsBundle, token_ids: list[int]) -> dict[str, Any]:
        """Attach original media to exact ids without re-rendering the conversation.

        The returned ``prompt_token_ids`` carry each image placeholder run of the prompt collapsed to
        one marker and the completion verbatim; vLLM's own placeholder expansion restores exactly
        ``token_ids``.
        """
        return self._request(item, prompt_token_ids=token_ids)

    def _validate_media_markers(
        self,
        prompt_ids: list[int],
        num_images: int,
        num_videos: int,
        *,
        expected_video_markers: int | None = None,
    ) -> None:
        image_items = 0
        video_items = 0
        if self.vision_start_token_id is not None:
            for index, token_id in enumerate(prompt_ids[:-1]):
                if token_id != self.vision_start_token_id:
                    continue
                following = prompt_ids[index + 1]
                image_items += int(following == self.image_token_id)
                video_items += int(self.video_token_id is not None and following == self.video_token_id)
        else:
            # Fallback for a stub/custom config: each contiguous placeholder run
            # represents one media item.
            image_items = self._count_token_runs(prompt_ids, self.image_token_id)
            video_items = self._count_token_runs(prompt_ids, self.video_token_id)
        video_expected = num_videos if expected_video_markers is None else expected_video_markers
        if image_items != num_images or video_items != video_expected:
            raise ValueError(
                "Qwen3-VL prompt/media cardinality mismatch: "
                f"prompt has image={image_items}, video={video_items}; "
                f"expected image={num_images}, video markers={video_expected} for {num_videos} video handle(s)"
            )

    @staticmethod
    def _count_token_runs(prompt_ids: list[int], token_id: int | None) -> int:
        if token_id is None:
            return 0
        return sum(
            1
            for index, value in enumerate(prompt_ids)
            if value == token_id and (index == 0 or prompt_ids[index - 1] != token_id)
        )

    def _image_marker_widths(self, prompt_ids: list[int]) -> tuple[int, ...]:
        """Return ordered image-run widths, requiring exact vision start/end boundaries."""
        if self.image_token_id is None or self.vision_start_token_id is None or self.vision_end_token_id is None:
            raise ValueError("cross-token multimodal teacher requires image marker boundary IDs")
        widths: list[int] = []
        index = 0
        while index < len(prompt_ids):
            token_id = prompt_ids[index]
            if token_id == self.image_token_id:
                raise ValueError("cross-token multimodal teacher prompt contains a stray image marker")
            if token_id == self.vision_end_token_id:
                raise ValueError("cross-token multimodal teacher prompt contains a stray vision-end marker")
            if token_id != self.vision_start_token_id:
                index += 1
                continue
            index += 1
            if index >= len(prompt_ids) or prompt_ids[index] != self.image_token_id:
                raise ValueError("cross-token multimodal teacher vision-start is not followed by an image marker")
            end = index
            while end < len(prompt_ids) and prompt_ids[end] == self.image_token_id:
                end += 1
            widths.append(end - index)
            if end >= len(prompt_ids) or prompt_ids[end] != self.vision_end_token_id:
                raise ValueError("cross-token multimodal teacher image run is not followed by vision-end")
            index = end + 1
        return tuple(widths)

    def _grid_marker_widths(self, grid: torch.Tensor, *, modality: str) -> tuple[tuple[int, ...], int]:
        """Return ordered post-merge marker widths and the raw patch count for one media grid."""
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"Qwen3-VL {modality} grid must be [N,3], got {tuple(grid.shape)}")
        grid_rows = [[int(value) for value in row] for row in grid.tolist()]
        if any(value < 1 for row in grid_rows for value in row):
            raise ValueError(f"Qwen3-VL {modality} grid must contain positive [t,h,w] rows, got {grid_rows}")
        raw_widths = tuple(t * h * w for t, h, w in grid_rows)
        merge_area = self.spatial_merge_size**2
        if any(raw_width % merge_area for raw_width in raw_widths):
            raise ValueError(f"Qwen3-VL {modality} grid is not divisible by spatial merge area {merge_area}")
        return tuple(raw_width // merge_area for raw_width in raw_widths), sum(raw_widths)

    def _expected_image_marker_widths(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
    ) -> tuple[int, ...]:
        """Compute each realized image width from this teacher's HF processor grid."""
        if not images:
            return ()
        processed_prompt_ids, _attention, model_inputs = self._processor_inputs_from_messages(messages, images, [])
        image_grid = model_inputs.get("image_grid_thw")  # [N_image,3]
        if image_grid is None:
            raise ValueError("cross-token multimodal teacher processor returned no image_grid_thw")
        widths, _raw_patch_count = self._grid_marker_widths(image_grid, modality="image")
        processor_widths = self._image_marker_widths(processed_prompt_ids)
        if processor_widths != widths:
            raise ValueError(
                "cross-token multimodal teacher processor marker runs do not match ordered image grids: "
                f"markers={processor_widths}, grids={widths}"
            )
        if len(widths) != len(images) or any(width < 1 for width in widths):
            raise ValueError(
                "cross-token multimodal teacher processor returned invalid ordered image marker widths "
                f"{widths} for {len(images)} image(s)"
            )
        return widths

    def _processor_inputs(self, item: ObsBundle) -> tuple[list[int], list[bool], dict[str, torch.Tensor]]:
        messages, images, videos = self._messages_and_media(item)
        return self._processor_inputs_from_messages(messages, images, videos)

    def _processor_inputs_from_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        videos: list[Any],
    ) -> tuple[list[int], list[bool], dict[str, torch.Tensor]]:
        """Process already-resolved media and message bytes once for policy input."""
        chat_template = self._resolved_chat_template()
        processed = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            chat_template=chat_template,
            **self._processor_kwargs(messages),
        )
        if not isinstance(processed, Mapping):
            items = getattr(processed, "items", None)
            if not callable(items):
                raise TypeError(f"Qwen3-VL processor returned {type(processed).__name__}, expected a mapping")
            processed = dict(items())
        if "input_ids" not in processed:
            raise ValueError("Qwen3-VL processor returned no input_ids")
        prompt_ids = _token_ids(processed["input_ids"], field_name="processor input_ids")
        if not prompt_ids:
            raise ValueError("Qwen3-VL processor produced an empty prompt")
        raw_attention = processed.get("attention_mask")
        attention_mask = (
            [True] * len(prompt_ids)
            if raw_attention is None
            else [bool(value) for value in _token_ids(raw_attention, field_name="processor attention_mask")]
        )
        if len(attention_mask) != len(prompt_ids):
            raise ValueError(
                f"processor attention_mask length {len(attention_mask)} != prompt length {len(prompt_ids)}"
            )
        model_inputs = {
            key: _to_tensor(value, field_name=key)
            for key, value in processed.items()
            if key in _MEDIA_KEYS and value is not None
        }
        self._validate_processed_media(
            prompt_ids=prompt_ids,
            images=images,
            videos=videos,
            model_inputs=model_inputs,
        )
        return prompt_ids, attention_mask, model_inputs

    def _validate_processed_media(
        self,
        *,
        prompt_ids: list[int],
        images: list[Any],
        videos: list[Any],
        model_inputs: dict[str, torch.Tensor],
    ) -> None:
        video_grid = model_inputs.get("video_grid_thw")
        expected_video_markers = None if video_grid is None else sum(int(row[0]) for row in video_grid.tolist())
        self._validate_media_markers(
            prompt_ids,
            len(images),
            len(videos),
            expected_video_markers=expected_video_markers,
        )
        self._validate_modality_tensors(
            modality="image",
            handles=images,
            token_id=self.image_token_id,
            pixels=model_inputs.get("pixel_values"),
            grid=model_inputs.get("image_grid_thw"),
            prompt_ids=prompt_ids,
        )
        self._validate_modality_tensors(
            modality="video",
            handles=videos,
            token_id=self.video_token_id,
            pixels=model_inputs.get("pixel_values_videos"),
            grid=model_inputs.get("video_grid_thw"),
            prompt_ids=prompt_ids,
        )

    def _validate_modality_tensors(
        self,
        *,
        modality: str,
        handles: list[Any],
        token_id: int | None,
        pixels: torch.Tensor | None,
        grid: torch.Tensor | None,
        prompt_ids: list[int],
    ) -> None:
        if (pixels is None) != (grid is None):
            raise ValueError(f"Qwen3-VL {modality} pixel/grid tensors must be present together")
        if not handles:
            if pixels is not None or (token_id is not None and token_id in prompt_ids):
                raise ValueError(f"Qwen3-VL {modality} tensors/placeholders exist without media handles")
            return
        if pixels is None or grid is None:
            raise ValueError(f"Qwen3-VL {modality} handles require processor pixel/grid tensors")
        marker_widths, raw_patch_count = self._grid_marker_widths(grid, modality=modality)
        if int(grid.shape[0]) != len(handles):
            raise ValueError(f"Qwen3-VL {modality} grid has {grid.shape[0]} row(s) for {len(handles)} handle(s)")
        if pixels.ndim < 1 or int(pixels.shape[0]) != raw_patch_count:
            raise ValueError(
                f"Qwen3-VL {modality} pixels have {pixels.shape[0] if pixels.ndim else 0} patch row(s), "
                f"grid requires {raw_patch_count}"
            )
        expected_placeholders = sum(marker_widths)
        actual_placeholders = 0 if token_id is None else prompt_ids.count(token_id)
        if actual_placeholders != expected_placeholders:
            raise ValueError(
                f"Qwen3-VL {modality} placeholder count {actual_placeholders} != grid-derived "
                f"visual token count {expected_placeholders}"
            )

    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> Qwen3VLPolicyInput:
        if n_ignore_prefix_tokens < 0:
            raise ValueError(f"n_ignore_prefix_tokens must be >= 0, got {n_ignore_prefix_tokens}")
        processed_prompt_ids, prompt_attention, model_inputs = self._processor_inputs(sample)
        if prompt_ids is None:
            exact_prompt_ids = processed_prompt_ids
        else:
            exact_prompt_ids = [int(token_id) for token_id in prompt_ids]
            if exact_prompt_ids != processed_prompt_ids:
                mismatch = next(
                    (
                        index
                        for index, (actual, expected) in enumerate(
                            zip(exact_prompt_ids, processed_prompt_ids, strict=False)
                        )
                        if actual != expected
                    ),
                    min(len(exact_prompt_ids), len(processed_prompt_ids)),
                )
                raise ValueError(
                    "stored rollout prompt_ids differ from the Qwen3-VL processor prefix at "
                    f"index {mismatch}: stored_len={len(exact_prompt_ids)}, "
                    f"processor_len={len(processed_prompt_ids)}"
                )
        if isinstance(rollout_output, str):
            completion_ids = [
                int(token_id) for token_id in self.tokenizer(rollout_output, add_special_tokens=False).input_ids
            ]
        elif rollout_output is None:
            completion_ids = []
        else:
            completion_ids = _token_ids(rollout_output, field_name="rollout completion ids")
        if n_ignore_prefix_tokens > len(completion_ids):
            raise ValueError(
                f"n_ignore_prefix_tokens={n_ignore_prefix_tokens} exceeds completion length {len(completion_ids)}"
            )
        input_ids = exact_prompt_ids + completion_ids
        attention_mask = prompt_attention + [True] * len(completion_ids)
        logprob_masks = (
            [0] * (len(exact_prompt_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0]
        )
        return Qwen3VLPolicyInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logprob_masks=logprob_masks,
            model_inputs=model_inputs,
        )

    def policy_compute_max_len(self, processed_samples: list[Qwen3VLPolicyInput]) -> int:
        if not processed_samples:
            raise ValueError("policy_compute_max_len got 0 Qwen3-VL samples")
        return max(len(sample.input_ids) for sample in processed_samples)

    def policy_collate_fn(
        self,
        processed_samples: list[Qwen3VLPolicyInput],
        computed_max_len: int,
    ) -> dict[str, Any]:
        if not processed_samples:
            raise ValueError("policy_collate_fn got 0 Qwen3-VL samples")
        required_len = max(len(sample.input_ids) for sample in processed_samples)
        if computed_max_len < required_len:
            raise ValueError(
                f"computed_max_len={computed_max_len} would truncate a Qwen3-VL sample of length {required_len}"
            )
        pad = int(self.tokenizer.pad_token_id)
        input_ids = torch.tensor(  # [B,L]
            [sample.input_ids + [pad] * (computed_max_len - len(sample.input_ids)) for sample in processed_samples],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(  # [B,L]
            [
                sample.attention_mask + [False] * (computed_max_len - len(sample.attention_mask))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        logprob_masks = torch.tensor(  # [B,L]
            [
                sample.logprob_masks + [0] * (computed_max_len - len(sample.logprob_masks))
                for sample in processed_samples
            ],
            dtype=torch.bool,
        )
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logprob_masks": logprob_masks,
        }
        for key in _MEDIA_KEYS:
            values = [sample.model_inputs[key] for sample in processed_samples if key in sample.model_inputs]
            if values:
                batch[key] = torch.cat(values, dim=0)  # [N_total,...]
        add_forward_routing(batch, processed_samples, computed_max_len)
        return batch
