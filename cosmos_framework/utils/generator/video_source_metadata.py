# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Source frame metadata for Qwen's index/FPS video timestamp convention."""

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict

VIDEO_METADATA_KEY: str = "video_metadata"


class VideoSourceMetadata(TypedDict):
    """Indices refer to the original encoded video, including for cropped inputs."""

    fps: float
    total_num_frames: int
    frames_indices: list[int]


def validate_source_video_metadata(metadata: object, num_frames: int) -> VideoSourceMetadata:
    """Validate alignment and return a copy safe from processor padding in place."""
    if not isinstance(metadata, Mapping):
        raise ValueError("video_metadata must contain source fps, total_num_frames, and frames_indices")
    fps = metadata.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video_metadata fps must be positive and finite")
    total = metadata.get("total_num_frames")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("video_metadata total_num_frames must be a positive integer")
    indices = metadata.get("frames_indices")
    if not isinstance(indices, list) or num_frames < 1 or len(indices) != num_frames:
        raise ValueError("video_metadata requires one source frame index per selected frame")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total for index in indices):
        raise ValueError("video_metadata frame indices must be integers within the original video")
    if any(left > right for left, right in zip(indices, indices[1:])):
        raise ValueError("video_metadata frame indices must retain source order")
    return {"fps": float(fps), "total_num_frames": total, "frames_indices": indices.copy()}


def calculate_video_timestamps(frame_indices: Sequence[int], fps: float, temporal_patch_size: int) -> list[float]:
    """Match Qwen's first/last-frame midpoint per temporal patch, without rebasing."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if isinstance(temporal_patch_size, bool) or not isinstance(temporal_patch_size, int) or temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be a positive integer")
    indices = list(frame_indices)
    if not indices or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("frame_indices must contain nonnegative source frame indices")
    if any(left > right for left, right in zip(indices, indices[1:])):
        raise ValueError("frame_indices must retain source order")
    if remainder := len(indices) % temporal_patch_size:
        indices.extend([indices[-1]] * (temporal_patch_size - remainder))
    times = [index / fps for index in indices]
    return [
        (times[index] + times[index + temporal_patch_size - 1]) / 2
        for index in range(0, len(times), temporal_patch_size)
    ]
