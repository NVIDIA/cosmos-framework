# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmos_framework.data.generator.action.action_binary import (
    decode_action_frames,
    decode_action_per_clip,
    get_action_binary_spec,
)
from cosmos_framework.data.generator.local_simulation.core import LocalByteRangeBackend as ByteRangeBackend
from cosmos_framework.utils.easy_io.transient_retry import retry_on_transient_error


@dataclass(frozen=True)
class ACT2RangeReadStats:
    """Byte ranges and request counts for one ACT2 window read."""

    frame_start: int
    frame_end: int
    frame_bytes: int
    frame_requests: int
    per_clip_start: int
    per_clip_end: int
    per_clip_bytes: int
    per_clip_requests: int


@dataclass(frozen=True)
class ACT2Window:
    """Selected per-frame records and the clip-wide ACT2 fields."""

    per_frame: dict[str, np.ndarray]
    per_clip: dict[str, np.ndarray]
    stats: ACT2RangeReadStats


def _read_exact_range(backend: ByteRangeBackend, uri: str, start: int, end: int, *, purpose: str) -> bytes:
    """Download and validate one half-open object byte range."""
    if start < 0 or end <= start:
        raise ValueError(f"Invalid {purpose} range [{start}, {end}).")
    expected_size = end - start

    def read_once() -> bytes:
        data = backend.get(filepath=uri, offset=start, size=expected_size)
        if not isinstance(data, bytes):
            raise TypeError(f"Object-store range read for {uri!r} returned {type(data).__name__}, expected bytes.")
        if len(data) != expected_size:
            raise OSError(f"Short {purpose} read for {uri!r}: expected {expected_size} bytes, received {len(data)}.")
        return data

    return retry_on_transient_error(
        read_once,
        operation=f"{purpose}({uri}, {start}:{end})",
    )


def read_act2_window(
    backend: ByteRangeBackend,
    uri: str,
    frame_indices: np.ndarray,
    *,
    source_dataset: str,
    num_frames: int,
) -> ACT2Window:
    """Range-read ACT2 frame records and, when present, the per-clip tail."""
    if frame_indices.ndim != 1 or not frame_indices.size:
        raise ValueError(f"frame_indices must be a non-empty one-dimensional array, got {frame_indices.shape}.")
    if np.any(frame_indices < 0) or np.any(frame_indices >= num_frames):
        raise IndexError(f"Action frame indices must be within [0, {num_frames}).")

    # The code-level dataset contract makes the ACT2 frame range computable without reading its header.
    spec = get_action_binary_spec(source_dataset)
    header_size_bytes = spec.header_size_bytes
    first_frame = int(frame_indices.min())
    last_frame = int(frame_indices.max())
    frame_start = header_size_bytes + first_frame * spec.frame_record_size_bytes
    frame_end = header_size_bytes + (last_frame + 1) * spec.frame_record_size_bytes
    frame_bytes = _read_exact_range(backend, uri, frame_start, frame_end, purpose="action_frames")
    enclosing_frames = decode_action_frames(frame_bytes, source_dataset)
    relative_indices = frame_indices - first_frame  # [T]
    selected_frames = {
        name: np.ascontiguousarray(values[relative_indices]) for name, values in enclosing_frames.items()
    }

    # Per-clip fields live in a fixed-size tail. Libero has none, so its hot path performs only the frame request.
    per_clip_start = header_size_bytes + num_frames * spec.frame_record_size_bytes
    per_clip_end = per_clip_start + spec.per_clip_size_bytes
    per_clip_data: dict[str, np.ndarray] = {}
    per_clip_requests = 0
    if spec.per_clip_size_bytes:
        per_clip_bytes = _read_exact_range(backend, uri, per_clip_start, per_clip_end, purpose="action_per_clip")
        per_clip_data = decode_action_per_clip(per_clip_bytes, source_dataset)
        per_clip_requests = 1

    stats = ACT2RangeReadStats(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_bytes=len(frame_bytes),
        frame_requests=1,
        per_clip_start=per_clip_start,
        per_clip_end=per_clip_end,
        per_clip_bytes=spec.per_clip_size_bytes,
        per_clip_requests=per_clip_requests,
    )
    return ACT2Window(per_frame=selected_frames, per_clip=per_clip_data, stats=stats)
