# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validated per-stream presentation times for the opt-in Edge Reasoner path."""

import math
from collections.abc import Mapping
from typing import TypedDict

import torch

SOURCE_VIDEO_TIMING_KEY = "source_video_timing"
VIDEO_TIMESTAMP_MODES = ("qwen_index", "legacy_fps", "source_pts")


class SourceVideoTiming(TypedDict):
    clock: str
    time_unit: str
    origin: str
    origin_pts_seconds: float
    source_sha256: str
    frame_indices: list[int]
    pts_seconds: list[float]
    duration_seconds: list[float]
    pts_dtype: str
    duration_dtype: str


def validate_video_timestamp_mode(mode: str) -> None:
    if mode not in VIDEO_TIMESTAMP_MODES:
        raise ValueError(f"video_timestamp_mode must be one of {VIDEO_TIMESTAMP_MODES}, got {mode!r}")


def require_source_pts_processor(processor: object) -> None:
    # Local imports avoid a cycle: the native Edge renderer also consumes timing records.
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import NemotronNanoV3BridgeProcessor
    from cosmos_framework.data.generator.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor

    if not isinstance(processor, Nemotron3DenseVLProcessor) or not isinstance(
        processor.processor, NemotronNanoV3BridgeProcessor
    ):
        raise ValueError("source_pts requires the native Cosmos3-Edge processor and Nemotron3DenseVL wrapper")
    if processor.temporal_patch_size != 1:
        raise ValueError("source_pts currently requires Edge temporal_patch_size=1")


def build_source_video_timing(
    frame_indices: list[int],
    pts_seconds: torch.Tensor,  # [T], native floating dtype, CPU
    duration_seconds: torch.Tensor,  # [T], native floating dtype, CPU
    source_sha256: str,
) -> SourceVideoTiming:
    """Keep native floating-point precision provenance; conversion cannot recover lost precision."""
    for name, tensor in (("pts_seconds", pts_seconds), ("duration_seconds", duration_seconds)):
        if tensor.device.type != "cpu" or not tensor.is_floating_point() or tensor.shape != (len(frame_indices),):
            raise ValueError(f"source_pts {name} must be a CPU floating tensor of shape [{len(frame_indices)}]")
    pts = pts_seconds.to(dtype=torch.float64).tolist()  # [T] -> list[float]
    durations = duration_seconds.to(dtype=torch.float64).tolist()  # [T] -> list[float]
    if not pts:
        raise ValueError("source_pts requires at least one selected frame")
    record: SourceVideoTiming = {
        "clock": "stream_presentation_timestamps",
        "time_unit": "seconds",
        "origin": "first_selected_frame",
        "origin_pts_seconds": pts[0],
        "source_sha256": source_sha256,
        "frame_indices": list(frame_indices),
        "pts_seconds": pts,
        "duration_seconds": durations,
        "pts_dtype": str(pts_seconds.dtype),
        "duration_dtype": str(duration_seconds.dtype),
    }
    validate_source_video_timing(record, len(frame_indices))
    return record


def _finite_numbers(value: object, name: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"source_pts {name} must have exactly {count} entries")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
        raise ValueError(f"source_pts {name} must contain finite numbers")
    return [float(x) for x in value]


def validate_source_video_timing(record: object, num_frames: int) -> list[float]:
    """Return clip-relative seconds, preserving repeated selections and source order."""
    if not isinstance(record, Mapping) or num_frames < 1:
        raise ValueError("source_pts requires a timing record for every nonempty video")
    expected = {
        "clock": "stream_presentation_timestamps",
        "time_unit": "seconds",
        "origin": "first_selected_frame",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("source_pts requires declared stream presentation seconds and first-selected-frame origin")
    for field in ("pts_dtype", "duration_dtype"):
        if record.get(field) not in {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}:
            raise ValueError(f"source_pts requires observed floating dtype metadata: {field}")
    source_sha256 = record.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_pts requires the encoded source SHA256")
    indices = record.get("frame_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != num_frames
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise ValueError("source_pts requires one nonnegative source frame index per frame")
    pts = _finite_numbers(record.get("pts_seconds"), "pts_seconds", num_frames)
    durations = _finite_numbers(record.get("duration_seconds"), "duration_seconds", num_frames)
    if any(duration < 0 for duration in durations):
        raise ValueError("source_pts frame durations must be nonnegative")
    origin = _finite_numbers([record.get("origin_pts_seconds")], "origin_pts_seconds", 1)[0]
    if origin != pts[0]:
        raise ValueError("source_pts origin must equal the first selected presentation timestamp")
    for index in range(1, num_frames):
        if indices[index] < indices[index - 1] or pts[index] < pts[index - 1]:
            raise ValueError("source_pts requires nondecreasing frame indices and presentation timestamps")
        if indices[index] == indices[index - 1] and (
            pts[index] != pts[index - 1] or durations[index] != durations[index - 1]
        ):
            raise ValueError("source_pts repeated source frames must retain identical timing")
    return [timestamp - origin for timestamp in pts]


def reject_source_pts_temporal_augmentation(media: Mapping[str, object]) -> None:
    """Reject legacy overlay/label snapping only when that augmentor actually selects a sample."""
    if any(isinstance(value, Mapping) and SOURCE_VIDEO_TIMING_KEY in value for value in media.values()):
        raise ValueError("source_pts does not support legacy temporal-localization overlay or label augmentation")
