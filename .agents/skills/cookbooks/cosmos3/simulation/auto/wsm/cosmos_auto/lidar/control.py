# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export geometry controls in the physical, unnormalized runtime input format."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from safetensors.numpy import save_file


def save_control(
    ranges: np.ndarray, frame_ids: Sequence[int], path: Path, clip_id: str
) -> None:
    """Pool nearest nonzero returns to 128×1800; preserve metres, not model padding."""
    ranges = np.asarray(ranges, dtype=np.float32)
    if ranges.ndim != 3 or ranges.shape[0] < 1 or ranges.shape[1] != 128:
        raise ValueError("Expected nonempty ranges[T,128,W]")
    if ranges.shape[0] != len(frame_ids) or ranges.shape[2] not in (1800, 3600):
        raise ValueError("Expected aligned frame IDs and a 1800 or 3600 column lattice")
    if not np.isfinite(ranges).all() or (ranges < 0).any():
        raise ValueError("Ranges must be finite, nonnegative metres")
    # Match the Framework/cookbook sparse-control conversion. No interpolation
    # across empty rays; intensity is a geometry placeholder, never reflectance.
    groups = ranges.reshape(*ranges.shape[:2], 1800, ranges.shape[2] // 1800)
    nearest = np.where(groups > 0, groups, np.inf).min(axis=-1)
    nearest = np.where(np.isfinite(nearest), nearest, 0).astype(np.float32)
    occupied = (nearest > 0).astype(np.float32)
    frames = np.stack([nearest, occupied, occupied])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    save_file(
        {"frames": frames},
        str(temporary),
        metadata={
            "source": json.dumps(
                {
                    "clip": clip_id,
                    "camera_frame_ids": list(frame_ids),
                    "kind": "geometry_hdmap_control",
                    "range_units": "metres",
                }
            )
        },
    )
    temporary.replace(path)
