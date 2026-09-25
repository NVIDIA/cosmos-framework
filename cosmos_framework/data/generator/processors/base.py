# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Base class and shared helpers for VFM/VLM processor wrappers.

Each concrete processor wraps a HuggingFace ``AutoProcessor`` for a specific
model family and exposes a small surface used by dataloaders and the training
model:

* ``apply_chat_template`` -- model-specific message templating (per subclass)
* ``add_assistant_tokens_mask`` -- model-specific loss mask construction
* ``tokenizer`` -- the underlying HF tokenizer (uniform property)
* ``tokenize_text`` / ``encode`` / ``decode`` -- simple delegations

This module hosts the parts that were truly common across subclasses so the
concrete files only contain model-specific logic.
"""

import json
import os
from typing import Dict, List, Optional

from transformers.models.auto.processing_auto import AutoProcessor

from cosmos_framework.utils import log
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
    build_cosmos3_edge_processor,
    is_cosmos3_edge_native_snapshot,
)
from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import maybe_download_hf_model_from_s3
from cosmos_framework.utils.generator.video_source_metadata import (
    VIDEO_METADATA_KEY,
    validate_source_video_metadata,
)


def convert_string_content_to_list_content(messages: List[Dict]) -> List[Dict]:
    """Normalize chat messages so ``content`` is always a list of typed dicts.

    Many HF processors do not accept ``"content": str``; they expect each
    message's content to be a list of ``{"type": ..., ...}`` entries. This
    helper rewrites bare-string contents into a single ``{"type": "text", ...}``
    entry in place and returns the same list for convenience.
    """
    for i, message in enumerate(messages):
        if isinstance(message["content"], str):
            messages[i]["content"] = [{"type": "text", "text": message["content"]}]
    return messages


def maybe_parse_video_content(
    messages: List[Dict],
) -> tuple[int, Optional[list[float]], Optional[list[int]], Optional[list[list[int]]]]:
    """Scan messages for video entries and return their decoding metadata.

    Returns ``(num_video, fps_per_video, total_frames_per_video, frame_indices_per_video)``.
    Source metadata takes precedence over sampled FPS. Without source metadata,
    use the legacy local frame-sequence clock and warn when ``fps`` is omitted.
    """
    num_video = 0
    video_fps: list[float] = []
    video_total_num_frames: list[int] = []
    video_frames_indices: list[list[int]] = []
    for message in messages:
        if isinstance(message["content"], list):
            for sub_content in message["content"]:
                if sub_content.get("type", "") == "video" and isinstance(sub_content["video"], list):
                    num_video += 1
                    if VIDEO_METADATA_KEY in sub_content:
                        metadata = validate_source_video_metadata(
                            sub_content[VIDEO_METADATA_KEY], len(sub_content["video"])
                        )
                        video_fps.append(metadata["fps"])
                        video_total_num_frames.append(metadata["total_num_frames"])
                        video_frames_indices.append(metadata["frames_indices"])
                        continue
                    fps = sub_content.get("fps", None)
                    if fps is None:
                        log.critical(
                            f"fps is None for video {sub_content}. Better to set the fps explicitly", rank0_only=False
                        )
                    frames_indices = sub_content.get("frames_indices")
                    if frames_indices is None:
                        frames_indices = list(range(len(sub_content["video"])))
                    if len(frames_indices) != len(sub_content["video"]):
                        raise ValueError(
                            "Video frame metadata must contain one index per provided frame: "
                            f"{len(frames_indices)} indices for {len(sub_content['video'])} frames"
                        )
                    video_fps.append(fps)
                    video_total_num_frames.append(sub_content.get("total_num_frames", len(sub_content["video"])))
                    video_frames_indices.append(frames_indices)
    return num_video, video_fps, video_total_num_frames, video_frames_indices


def maybe_get_max_pixels_from_images_kwargs(messages: List[Dict]) -> tuple[Optional[int], Optional[int]]:
    """Return ``(max_pixels, min_pixels)`` from the first image entry that sets ``max_pixels``."""
    for message in messages:
        if isinstance(message["content"], list):
            for sub_content in message["content"]:
                if sub_content.get("type", "") == "image" and sub_content.get("max_pixels", None) is not None:
                    return sub_content["max_pixels"], sub_content.get("min_pixels", None)
    return None, None


class BaseVLMProcessor:
    """Shared skeleton for VFM/VLM processor wrappers.

    Subclasses inherit the S3-or-local model resolution, the
    ``AutoProcessor`` load, and the extraction of common token IDs. They
    are responsible only for:

    * the chat templating logic (``apply_chat_template``);
    * the loss-mask construction (``add_assistant_tokens_mask``);
    * any model-specific dataloader helper fields (e.g. ``patch_size``,
      ``merge_size``, ``use_smart_resize``).

    A subclass that needs a different pad-id resolution (e.g. NemotronVL's
    ``<SPECIAL_999>`` convention) overrides :py:meth:`_resolve_pad_id`. A
    subclass that needs a different vision-end marker sets the
    ``VISION_END_TOKEN`` class attribute; the default ``None`` skips that
    lookup entirely (used for tokenizers that lack a single-token marker).
    """

    # Override on subclasses to the model's vision-end token (e.g. ``"</img>"``).
    # Leave as None when the tokenizer does not expose a single-token marker —
    # ``vision_end_id`` will then be set to None and downstream consumers
    # (e.g. ``debug_data_qwen.py``) will skip the check.
    VISION_END_TOKEN: Optional[str] = None
    USES_SOURCE_VIDEO_TIMESTAMPS: bool = False

    def __init__(
        self,
        name: str,
        credentials: str = "./credentials/s3_training.secret",
        bucket: str = "bucket4",
        cache_dir: Optional[str] = None,
        use_native_edge_processor: bool = False,
    ) -> None:
        if not isinstance(use_native_edge_processor, bool):
            raise TypeError("use_native_edge_processor must be a bool")
        if use_native_edge_processor:
            if not os.path.isdir(name):
                raise ValueError("Explicit native Edge processing requires staged local processor metadata")
            with open(os.path.join(name, "config.json")) as stream:
                model_type = json.load(stream).get("model_type")
            with open(os.path.join(name, "preprocessor_config.json")) as stream:
                processor_class = json.load(stream).get("processor_class")
            if model_type not in {"cosmos3_edge", "nemotron_siglip2"} or processor_class not in {
                "Cosmos3EdgeProcessor",
                "NemotronNanoV3BridgeProcessor",
            }:
                raise ValueError("Explicit native Edge processing requires compatible Cosmos3 Edge metadata")
        self.name = name
        if os.path.isdir(name):
            model_name_or_path_local = name
        else:
            model_name_or_path_local = maybe_download_hf_model_from_s3(
                name, credentials, bucket, include_model_weights=False, cache_dir=cache_dir
            )

        if use_native_edge_processor or is_cosmos3_edge_native_snapshot(model_name_or_path_local):
            # AutoProcessor on transformers 4.x silently degrades to a bare tokenizer
            # for renewed (no remote code) Cosmos3-Edge snapshots; use the native port.
            self.processor = build_cosmos3_edge_processor(model_name_or_path_local)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path_local, trust_remote_code=True)
        log.info("Successfully loaded processor from local cache")

        self.image_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            if hasattr(self.processor, "image_token")
            else None
        )
        self.video_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
            if hasattr(self.processor, "video_token")
            else None
        )
        self.eos_id = self.processor.tokenizer.eos_token_id
        self.pad_id = self._resolve_pad_id()
        self.vision_end_id = (
            self.processor.tokenizer.convert_tokens_to_ids(self.VISION_END_TOKEN)
            if self.VISION_END_TOKEN is not None
            else None
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _resolve_pad_id(self):
        """Return the pad token id. Default: ``pad_token_id`` falling back to ``eos_id``.

        Override on subclasses whose model uses a non-standard pad token (e.g.
        NemotronVL uses ``<SPECIAL_999>``).
        """
        pad = self.processor.tokenizer.pad_token_id
        return pad if pad is not None else self.eos_id

    # ------------------------------------------------------------------
    # Shared interfaces
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Expose the underlying HF tokenizer uniformly.

        Lets model and test code call ``proc.tokenizer`` regardless of which
        concrete processor wrapper they received.
        """
        return self.processor.tokenizer

    def tokenize_text(
        self,
        caption: str,
        is_video: bool = False,
        use_system_prompt: bool = False,
        system_prompt: Optional[str] = None,
    ) -> list[int]:
        """Tokenize a text caption via the shared ``tokenize_caption`` helper.

        Keeps VFM diffusion augmentors and VLM dataloaders on the same code
        path so a single processor instance serves both.
        """
        return tokenize_caption(
            caption,
            self.processor.tokenizer,
            is_video=is_video,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
        )

    def encode(self, *args, **kwargs):
        return self.processor.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.tokenizer.decode(*args, **kwargs)
