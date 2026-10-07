# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Collect CPU-side per-sample metadata for parquet data-stat logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cosmos_framework.data.generator.reasoner.video_decoder_qwen import pixels_to_token, token_to_pixels


def _mean(values: list[float] | list[int]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def system_prompt_from(conversation: list[dict[str, Any]]) -> str:
    """Return the effective leading system prompt as plain text."""
    if not conversation or conversation[0].get("role") != "system":
        return ""
    content = conversation[0].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


@dataclass
class SampleDataStats:
    """Accumulate source-media facts while TokenizeData builds one model sample."""

    num_images: int
    num_videos: int
    raw_image_tokens: int = 0
    raw_video_tokens: int = 0
    raw_image_pixels: int = 0
    raw_video_pixels: int = 0
    video_native_fps: list[float] = field(default_factory=list)
    video_native_num_frames: int = 0
    video_sampled_num_frames: int = 0
    video_pixels_per_frame: list[int] = field(default_factory=list)
    video_duration_sec: list[float] = field(default_factory=list)
    effective_max_video_token_length: list[int] = field(default_factory=list)
    video_min_pixels: list[int] = field(default_factory=list)
    video_max_pixels: list[int] = field(default_factory=list)

    @classmethod
    def from_conversation(cls, conversation: list[dict[str, Any]]) -> "SampleDataStats":
        """Count raw media references in the selected conversation before subsampling."""
        num_images = 0
        num_videos = 0
        for message in conversation:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                continue
            num_images += sum(item.get("type") == "image" for item in content)
            num_videos += sum(item.get("type") == "video" for item in content)
        return cls(num_images=num_images, num_videos=num_videos)

    def add_image(self, height: int, width: int, processor: Any) -> None:
        """Record native size for an image retained by model-specific filtering."""
        pixels = int(height) * int(width)
        self.raw_image_pixels += pixels
        self.raw_image_tokens += pixels_to_token(
            pixels,
            patch_size=processor.patch_size,
            temporal_patch_size=1,
            merge_size=processor.merge_size,
        )

    def add_video(self, media: dict[str, Any], processor: Any, temporal_patch_size: int) -> None:
        """Record native and effective decoder metadata for one video."""
        native_frames = int(media.get("native_num_frames", 0))
        pixels_per_frame = int(media.get("native_height", 0)) * int(media.get("native_width", 0))
        pixels = pixels_per_frame * native_frames
        self.raw_video_pixels += pixels
        self.raw_video_tokens += pixels_to_token(
            pixels,
            patch_size=processor.patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=processor.merge_size,
        )
        if media.get("native_fps") is not None:
            self.video_native_fps.append(float(media["native_fps"]))
        self.video_native_num_frames += native_frames
        self.video_sampled_num_frames += int(media.get("sampled_num_frames", len(media["videos"])))
        self.video_pixels_per_frame.append(pixels_per_frame)
        if media.get("native_duration_sec") is not None:
            self.video_duration_sec.append(float(media["native_duration_sec"]))
        if media.get("effective_max_video_token_length") is not None:
            self.effective_max_video_token_length.append(int(media["effective_max_video_token_length"]))
        if media.get("budget_min_pixels") is not None:
            self.video_min_pixels.append(int(media["budget_min_pixels"]))
        if media.get("budget_max_pixels") is not None:
            self.video_max_pixels.append(int(media["budget_max_pixels"]))

    def finalize(
        self,
        *,
        input_ids: torch.Tensor,
        processor: Any,
        system_prompt: str,
        is_thinking_stripped: bool,
        max_image_token_length: int,
        max_video_token_length: int,
    ) -> dict[str, Any]:
        """Return the flat metadata payload preserved by the collate function."""
        image_token_id = getattr(processor, "image_token_id", None)
        video_token_id = getattr(processor, "video_token_id", None)
        seq_image_tokens = int((input_ids == image_token_id).sum()) if image_token_id is not None else 0
        seq_video_tokens = int((input_ids == video_token_id).sum()) if video_token_id is not None else 0

        image_min_pixels: int | None = None
        image_max_pixels: int | None = None
        if self.num_images:
            image_max_pixels = token_to_pixels(
                max_image_token_length,
                patch_size=processor.patch_size,
                temporal_patch_size=1,
                merge_size=processor.merge_size,
            )
            if getattr(processor, "use_smart_resize", False):
                image_min_pixels = int(processor.processor.image_processor.size["shortest_edge"]) * self.num_images

        return {
            "is_thinking_stripped": is_thinking_stripped,
            "system_prompt": system_prompt,
            "num_images": self.num_images,
            "num_videos": self.num_videos,
            "raw_image_tokens": self.raw_image_tokens,
            "raw_video_tokens": self.raw_video_tokens,
            "video_native_fps": _mean(self.video_native_fps),
            "video_native_num_frames": self.video_native_num_frames if self.num_videos else None,
            "video_sampled_num_frames": self.video_sampled_num_frames if self.num_videos else None,
            "video_pixels_per_frame": (
                int(_mean(self.video_pixels_per_frame)) if self.video_pixels_per_frame else None
            ),
            "video_duration_sec": _mean(self.video_duration_sec),
            "max_image_token_length": int(max_image_token_length),
            "max_video_token_length": int(max_video_token_length),
            "effective_max_video_token_length": (
                int(_mean(self.effective_max_video_token_length)) if self.effective_max_video_token_length else None
            ),
            "raw_image_pixels": self.raw_image_pixels,
            "raw_video_pixels": self.raw_video_pixels,
            "image_min_pixels": image_min_pixels,
            "image_max_pixels": image_max_pixels,
            "video_min_pixels": sum(self.video_min_pixels) if self.video_min_pixels else None,
            "video_max_pixels": sum(self.video_max_pixels) if self.video_max_pixels else None,
            "seq_image_tokens": seq_image_tokens,
            "seq_video_tokens": seq_video_tokens,
            "raw_text_tokens": int(input_ids.numel()) - seq_image_tokens - seq_video_tokens,
        }
