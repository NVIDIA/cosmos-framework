# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build teacher negative captions with the sampled camera clip's actual metadata."""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo


def _sample_scalars(batch: dict[str, Any], key: str, batch_size: int) -> list[float]:
    values = batch.get(key)
    if isinstance(values, torch.Tensor):
        values = values.reshape(-1).tolist()  # [B] -> list[float]
    if not isinstance(values, (list, tuple)) or len(values) != batch_size:
        raise ValueError(f"Negative metadata requires one {key} value per sample.")
    result: list[float] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Negative metadata requires scalar {key} values.")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Negative metadata requires numeric {key} values.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Negative metadata requires positive finite {key} values.")
        result.append(float(value))
    return result


def multiview_negative_captions(
    batch: dict[str, Any],
    batch_size: int,
    negative_prompt: str,
    *,
    video_key: str = "video",
) -> list[str]:
    """Match the positive MADS metadata without multiplying duration by camera count.

    Called before context-parallel owner rotation and before rollout truncation.
    Each logical sample owns a control and target, each in camera-major layout.
    Use the same augmentors as the dataset, including whole-second duration and
    punctuation, rather than the single-view transfer recipe's fractional rule.
    """
    if not isinstance(negative_prompt, str):
        raise TypeError("The teacher negative prompt must be a string.")
    videos = batch.get(video_key)
    counts = batch.get("num_vision_items_per_sample")
    if (
        not isinstance(videos, (list, tuple))
        or not isinstance(counts, (list, tuple))
        or len(counts) != batch_size
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts)
        or sum(counts) != len(videos)
    ):
        raise ValueError("Negative metadata requires aligned RGB item counts and target pixels.")
    view_counts = _sample_scalars(batch, "sample_n_views", batch_size)
    frame_counts = _sample_scalars(batch, "num_video_frames_per_view", batch_size)
    frame_rates = _sample_scalars(batch, "conditioning_fps", batch_size)
    captions: list[str] = []
    offset = 0
    for count, views, frames, fps in zip(counts, view_counts, frame_counts, frame_rates, strict=True):
        target = videos[offset + count - 1]  # [1,C,V*T,H,W]
        offset += count
        if not isinstance(target, torch.Tensor) or target.ndim != 5 or target.shape[0] != 1:
            raise ValueError("Negative metadata requires target pixels shaped [1,C,V*T,H,W].")
        if not views.is_integer() or not frames.is_integer() or target.shape[2] != int(views * frames):
            raise ValueError("Camera count and per-view frame count must match the actual target pixels.")
        if min(target.shape[1:]) < 1:
            raise ValueError("Negative metadata requires nonempty target pixels.")
        urls = batch.get("__url__")
        source_url = urls[len(captions)] if isinstance(urls, (list, tuple)) and len(urls) == batch_size else None
        source_options = getattr(getattr(source_url, "meta", None), "opts", {})
        captions.append(
            format_multiview_negative_caption(
                negative_prompt,
                frames=int(frames),
                fps=fps,
                height=target.shape[-2],
                width=target.shape[-1],
                aspect_ratio=source_options.get("aspect_ratio"),
            )
        )
    return captions


def format_multiview_negative_caption(
    negative_prompt: str, *, frames: int, fps: float, height: int, width: int, aspect_ratio: str | None = None
) -> str:
    """Use the training augmentors for both training and opt-in sim-auto inference."""
    if not isinstance(negative_prompt, str):
        raise TypeError("The teacher negative prompt must be a string.")
    if min(frames, height, width) < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Negative metadata requires positive finite camera geometry and FPS.")
    # The positive dataset formats metadata before transfer-item flattening.
    # A nonempty sentinel lets the same augmentors format the empty-text arm.
    sentinel = "SIM_AUTO_EMPTY_CAPTION"
    caption_value: str | dict = negative_prompt.strip() or sentinel
    try:
        structured = json.loads(negative_prompt)
    except json.JSONDecodeError:
        structured = None
    if isinstance(structured, dict):
        caption_value = structured
    divisor = math.gcd(height, width)
    sample: dict[str, Any] = {
        "ai_caption": caption_value,
        "video": None,
        "conditioning_fps": fps,
        "num_video_frames_per_view": frames,
        "image_size": (height, width),
        "aspect_ratio": aspect_ratio or f"{width // divisor},{height // divisor}",
    }
    temporal = DurationFPSTextTimeStamps(args={"num_frames_key": "num_video_frames_per_view", "skip_on_error": False})
    timed = temporal(sample)
    if timed is None:
        raise ValueError("Could not format teacher negative duration metadata.")
    augmented = ResolutionTextInfo()(timed)
    if augmented is None or not isinstance(augmented["ai_caption"], (str, dict)):
        raise ValueError("Could not format teacher negative metadata.")
    value = augmented["ai_caption"]
    caption = json.dumps(value) if isinstance(value, dict) else value
    if not negative_prompt.strip():
        caption = caption.removeprefix(sentinel).lstrip(". ")
    return caption
