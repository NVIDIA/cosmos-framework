# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

_DEFAULT_FPS = 2.0
_DEFAULT_MIN_FRAMES = 4
_DEFAULT_MAX_FRAMES = 768


def _round_by_factor(number: float, factor: int) -> int:
    """Return the closest integer divisible by ``factor``."""
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    """Return the smallest integer divisible by ``factor`` above ``number``."""
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    """Return the largest integer divisible by ``factor`` below ``number``."""
    return math.floor(number / factor) * factor


def smart_nframes_with_factor(
    element: dict[str, float | int],
    *,
    total_frames: int,
    video_fps: float,
    frame_factor: int,
) -> int:
    """Mirror qwen-vl-utils frame sampling with an explicit frame factor."""
    if frame_factor < 1:
        raise ValueError(f"frame_factor must be positive, got {frame_factor}")
    if "fps" in element and "nframes" in element:
        raise ValueError("Only accept either `fps` or `nframes`")

    if "nframes" in element:
        nframes = _round_by_factor(float(element["nframes"]), frame_factor)
    else:
        fps = float(element.get("fps", _DEFAULT_FPS))
        min_frames = _ceil_by_factor(float(element.get("min_frames", _DEFAULT_MIN_FRAMES)), frame_factor)
        requested_max_frames = float(element.get("max_frames", min(_DEFAULT_MAX_FRAMES, total_frames)))
        max_frames = _floor_by_factor(requested_max_frames, frame_factor)
        requested_frames = total_frames / video_fps * fps
        bounded_frames = min(min(max(requested_frames, min_frames), max_frames), total_frames)
        nframes = _floor_by_factor(bounded_frames, frame_factor)

    if not (frame_factor <= nframes <= total_frames):
        raise ValueError(f"nframes should be in [{frame_factor}, {total_frames}], got {nframes}")
    return nframes
