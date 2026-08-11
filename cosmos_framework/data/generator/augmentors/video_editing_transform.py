# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Conversation parsing and aligned decoding for paired video editing data."""

from __future__ import annotations

import io
import json
import math
import pickle
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as transforms_F

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils.generator.image_resize import get_max_pixels_resized_size
from cosmos_framework.utils.generator.torchcodec_video import VideoMetadata, decode_frames_tchw_uint8, probe_video


def valid_video_frame_count(available_frames: int, max_num_frames: int) -> int:
    """Return the largest positive Wan-compatible ``4N+1`` frame count."""
    capped = min(available_frames, max_num_frames)
    if capped < 1:
        return 0
    return 1 + 4 * ((capped - 1) // 4)


def aligned_frame_indices(metadata: VideoMetadata, target_fps: float, num_frames: int) -> list[int]:
    """Map a shared target-FPS timeline to nearest source-frame indices."""
    if target_fps <= 0 or metadata.average_fps <= 0:
        raise ValueError("FPS values must be positive")
    indices = [round(frame_index * metadata.average_fps / target_fps) for frame_index in range(num_frames)]
    return [min(max(index, 0), metadata.num_frames - 1) for index in indices]


def _decode_json_payload(payload: object) -> object:
    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode("utf-8")
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _conversation_messages(payload: object) -> list[dict[str, Any]]:
    decoded = _decode_json_payload(payload)
    if isinstance(decoded, dict):
        decoded = decoded.get("content")
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("conversation must be a non-empty list")
    messages = decoded[0] if isinstance(decoded[0], list) else decoded
    if not isinstance(messages, list) or len(messages) != 2 or not all(isinstance(item, dict) for item in messages):
        raise ValueError("video editing conversation must contain exactly one user/assistant round")
    return messages


def parse_video_editing_conversation_with_references(payload: object) -> tuple[str, str, list[str], str]:
    """Return source, target, ordered reference-image keys, and instruction."""
    user_message, assistant_message = _conversation_messages(payload)
    if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
        raise ValueError("conversation roles must be user then assistant")

    user_content = user_message.get("content")
    assistant_content = assistant_message.get("content")
    if not isinstance(user_content, list) or not isinstance(assistant_content, list):
        raise ValueError("conversation content must be a list")

    source_keys = [item.get("video") for item in user_content if isinstance(item, dict) and item.get("type") == "video"]
    target_keys = [
        item.get("video") for item in assistant_content if isinstance(item, dict) and item.get("type") == "video"
    ]
    reference_keys = [
        item.get("image") for item in user_content if isinstance(item, dict) and item.get("type") == "image"
    ]
    text_parts = [
        str(item.get("text", "")).strip()
        for item in user_content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    instruction = " ".join(part for part in text_parts if part).strip()
    if len(source_keys) != 1 or not isinstance(source_keys[0], str):
        raise ValueError("user message must contain exactly one source video")
    if len(target_keys) != 1 or not isinstance(target_keys[0], str):
        raise ValueError("assistant message must contain exactly one target video")
    if source_keys[0] == target_keys[0]:
        raise ValueError("source and target video keys must differ")
    if not all(isinstance(key, str) for key in reference_keys):
        raise ValueError("user message contains an invalid reference image key")
    if len(reference_keys) != len(set(reference_keys)):
        raise ValueError("user message contains duplicate reference image keys")
    if not instruction:
        raise ValueError("editing instruction is empty")
    return source_keys[0], target_keys[0], reference_keys, instruction


def parse_video_editing_conversation(payload: object) -> tuple[str, str, str]:
    """Return the legacy source, target, and instruction tuple."""
    source_key, target_key, _, instruction = parse_video_editing_conversation_with_references(payload)
    return source_key, target_key, instruction


class PairedVideoEditingToTrainingFormat(Augmentor):
    """Decode an aligned source/target video pair and optional reference images.

    Source and target frames are sampled on the same ``target_fps`` timeline and
    resized to the source aspect ratio. Their encoded durations may differ by at
    most one target-frame interval plus ``duration_ratio_tolerance`` of the longer
    clip. Reference images are emitted between source and target as independent
    one-frame vision items.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        *,
        media_key: str = "media",
        conversation_key: str = "texts",
        target_fps: float = 16.0,
        max_num_frames: int = 93,
        max_pixels: int = 512 * 512,
        padding_divisor: int = 16,
        duration_ratio_tolerance: float = 0.0,
        aspect_ratio_tolerance: float = 0.05,
        mean: float = 0.5,
        std: float = 0.5,
        num_decode_threads: int = 1,
        max_reference_images: int = 4,
        min_reference_image_side: int = 128,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], None, args)
        if target_fps <= 0 or max_num_frames < 1:
            raise ValueError("target_fps and max_num_frames must be positive")
        if max_reference_images < 0:
            raise ValueError("max_reference_images must be non-negative")
        if min_reference_image_side <= 0:
            raise ValueError("min_reference_image_side must be positive")
        if not 0 <= duration_ratio_tolerance < 1 or not 0 <= aspect_ratio_tolerance < 1:
            raise ValueError("alignment tolerances must be in [0, 1)")
        self.media_key = media_key
        self.conversation_key = conversation_key
        self.target_fps = target_fps
        self.max_num_frames = max_num_frames
        self.max_pixels = max_pixels
        self.padding_divisor = padding_divisor
        self.duration_ratio_tolerance = duration_ratio_tolerance
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.mean = mean
        self.std = std
        self.num_decode_threads = num_decode_threads
        self.max_reference_images = max_reference_images
        self.min_reference_image_side = min_reference_image_side

    @staticmethod
    def _media_bundle(payload: object) -> dict[str, bytes]:
        decoded = pickle.loads(payload) if isinstance(payload, (bytes, bytearray)) else payload
        if not isinstance(decoded, dict):
            raise ValueError("media payload must decode to a dictionary")
        if not all(isinstance(key, str) and isinstance(value, bytes) for key, value in decoded.items()):
            raise ValueError("media payload must contain only string-to-bytes entries")
        return decoded

    def _decode_reference_image(self, image_bytes: bytes) -> tuple[torch.Tensor, int, int]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            image = image.convert("RGB")
            original_width, original_height = image.size
            if min(original_width, original_height) < self.min_reference_image_side:
                raise ValueError(
                    "reference_image_too_small:"
                    f"{original_width}x{original_height}:min_side={self.min_reference_image_side}"
                )
            width, height = get_max_pixels_resized_size(
                original_width,
                original_height,
                max_pixels=self.max_pixels,
                padding_constant=self.padding_divisor,
            )
            if min(width, height) < self.min_reference_image_side:
                raise ValueError(
                    f"reference_image_too_small_after_resize:{width}x{height}:min_side={self.min_reference_image_side}"
                )
            # Follow still-image preprocessing with bicubic resampling; source and
            # target videos use faster batched bilinear interpolation below.
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
            tensor = transforms_F.to_tensor(image)
        tensor = transforms_F.normalize(tensor, mean=[self.mean] * 3, std=[self.std] * 3)
        return tensor.unsqueeze(1), width, height  # [C,1,H,W]

    @staticmethod
    def _duration(metadata: VideoMetadata) -> float:
        return metadata.num_frames / metadata.average_fps

    def _validate_pair(self, source: VideoMetadata, target: VideoMetadata) -> tuple[float, int, int]:
        if source.height is None or source.width is None or target.height is None or target.width is None:
            raise ValueError("video dimensions are unavailable")
        source_duration = self._duration(source)
        target_duration = self._duration(target)
        longer_duration = max(source_duration, target_duration)
        if longer_duration <= 0:
            raise ValueError("video duration is not positive")
        duration_difference = abs(source_duration - target_duration)
        # Editing pairs should be temporally aligned. Permit one target-frame interval
        # for container/FPS metadata rounding, plus an explicitly configured ratio.
        duration_tolerance = 1.0 / self.target_fps + self.duration_ratio_tolerance * longer_duration
        if duration_difference > duration_tolerance:
            raise ValueError(f"duration_mismatch:{source_duration:.3f}:{target_duration:.3f}")

        source_aspect = source.width / source.height
        target_aspect = target.width / target.height
        aspect_delta = abs(source_aspect - target_aspect) / max(source_aspect, target_aspect)
        if aspect_delta > self.aspect_ratio_tolerance:
            raise ValueError(f"aspect_ratio_mismatch:{source_aspect:.3f}:{target_aspect:.3f}")

        common_duration = min(source_duration, target_duration)
        available_frames = max(1, math.floor(common_duration * self.target_fps))
        num_frames = valid_video_frame_count(available_frames, self.max_num_frames)
        if num_frames < 1:
            raise ValueError("pair has no decodable common frames")
        width, height = get_max_pixels_resized_size(
            source.width,
            source.height,
            max_pixels=self.max_pixels,
            padding_constant=self.padding_divisor,
        )
        return common_duration, width, height

    def _decode_pair(
        self, source_bytes: bytes, target_bytes: bytes
    ) -> tuple[torch.Tensor, torch.Tensor, VideoMetadata, int, int]:
        source_metadata = probe_video(source_bytes, include_dimensions=True, num_threads=self.num_decode_threads)
        target_metadata = probe_video(target_bytes, include_dimensions=True, num_threads=self.num_decode_threads)
        common_duration, width, height = self._validate_pair(source_metadata, target_metadata)
        available_frames = max(1, math.floor(common_duration * self.target_fps))
        num_frames = valid_video_frame_count(available_frames, self.max_num_frames)
        source_indices = aligned_frame_indices(source_metadata, self.target_fps, num_frames)
        target_indices = aligned_frame_indices(target_metadata, self.target_fps, num_frames)
        source_tchw, _ = decode_frames_tchw_uint8(  # [T,C,Hs,Ws]
            source_bytes, source_indices, num_threads=self.num_decode_threads
        )
        target_tchw, _ = decode_frames_tchw_uint8(  # [T,C,Ht,Wt]
            target_bytes, target_indices, num_threads=self.num_decode_threads
        )
        # Batched bilinear resizing keeps video preprocessing inexpensive. Both sides
        # take the identical path, preserving source/target spatial alignment.
        source_tchw = F.interpolate(
            source_tchw.float(), size=(height, width), mode="bilinear", align_corners=False
        )  # [T,C,H,W]
        target_tchw = F.interpolate(
            target_tchw.float(), size=(height, width), mode="bilinear", align_corners=False
        )  # [T,C,H,W]
        source_cthw = source_tchw.permute(1, 0, 2, 3).contiguous() / 255.0  # [C,T,H,W]
        target_cthw = target_tchw.permute(1, 0, 2, 3).contiguous() / 255.0  # [C,T,H,W]
        source_cthw = (source_cthw - self.mean) / self.std  # [C,T,H,W]
        target_cthw = (target_cthw - self.mean) / self.std  # [C,T,H,W]
        return source_cthw, target_cthw, source_metadata, width, height

    def __call__(self, data_dict: dict) -> dict | None:
        sample_key = data_dict.get("__key__", "unknown")
        try:
            source_key, target_key, reference_keys, instruction = parse_video_editing_conversation_with_references(
                data_dict[self.conversation_key]
            )
            if len(reference_keys) > self.max_reference_images:
                raise ValueError(
                    f"sample has {len(reference_keys)} reference images; maximum is {self.max_reference_images}"
                )
            media = self._media_bundle(data_dict[self.media_key])
            required_keys = [source_key, target_key, *reference_keys]
            missing_keys = [key for key in required_keys if key not in media]
            if missing_keys:
                raise ValueError(f"conversation references media keys absent from the bundle: {missing_keys}")
            source_cthw, target_cthw, _, width, height = self._decode_pair(media[source_key], media[target_key])
            references = [self._decode_reference_image(media[key]) for key in reference_keys]
            reference_tensors = [tensor for tensor, _, _ in references]
            num_frames = source_cthw.shape[1]
            data_dict["video"] = [source_cthw, *reference_tensors, target_cthw]
            data_dict["ai_caption"] = instruction
            data_dict["selected_caption_type"] = "editing_instruction"
            data_dict["fps"] = self.target_fps
            data_dict["num_frames"] = num_frames
            data_dict["image_size"] = [
                torch.tensor([height, width, height, width], dtype=torch.float),
                *[
                    torch.tensor([reference_height, reference_width] * 2, dtype=torch.float)
                    for _, reference_width, reference_height in references
                ],
                torch.tensor([height, width, height, width], dtype=torch.float),
            ]
            data_dict.setdefault("dataset_name", "video_editing")
            data_dict["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],
                # Source and target videos are frame-aligned. Reference images remain
                # independent because their one-frame temporal grid differs from the videos.
                share_vision_temporal_positions=not reference_tensors,
                vision_temporal_position_groups=[0, *([None] * len(reference_tensors)), 0]
                if reference_tensors
                else None,
            )
            return data_dict
        except Exception as error:
            log.warning(f"Rejecting video editing sample {sample_key}: {error}", rank0_only=False)
            return None
