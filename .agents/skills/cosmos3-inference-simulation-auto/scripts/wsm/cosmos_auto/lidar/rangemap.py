# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Point cloud <-> range map projection and the sparse WebDataset tar layout.

Port of ``utils/lidar_rangemap_utils.py`` from the Cosmos-Drive-Dreams toolkit.
``torch_scatter.scatter_min`` is replaced by ``torch.Tensor.scatter_reduce_``
(same nearest-point-wins result, stable argmin so intensity stays aligned).
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from cosmos_auto.lidar.sensor_model import SpinningLidarModel
from cosmos_auto.wm.wds_utils import get_sample, write_to_tar

DEFAULT_MAX_RANGE_M = 105.0
# Points whose elevation falls outside the ring span by more than this are dropped
# before the nearest-ring lookup (toolkit value).
ELEVATION_MARGIN_DEG = 0.1


def points_to_range_map(
    points: torch.Tensor,
    model: SpinningLidarModel,
    max_range: float = DEFAULT_MAX_RANGE_M,
    intensity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project sensor-frame points to a range map, keeping the nearest return per cell.

    Column binning follows the toolkit: ``azimuth = -atan2(y, x) + pi`` so column 0
    is the rear and the forward direction is the centre column; the row is the
    nearest ring elevation; the column index is the truncated azimuth bin. Per-row
    azimuth offsets are *not* applied here, so the map inverts exactly with
    :meth:`SpinningLidarModel.ray_directions`, whose rays point at the cell centres
    (truncation then returns every unprojected cell to its own column).

    Args:
        points: ``[N, 3]`` sensor-frame points.
        model: lattice (``n_rows``, ``n_columns``, ring elevations).
        max_range: returns at or beyond this range (metres) are dropped.
        intensity: optional ``[N]`` per-point intensity.

    Returns:
        ``(range_map, intensity_map)`` each ``[n_rows, n_columns]`` float32 on the
        points' device; ``0`` marks an empty cell. ``intensity_map`` is all zeros
        when ``intensity`` is ``None``.
    """
    n_rows, n_cols = model.shape
    device = points.device
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {tuple(points.shape)}")
    points = points.float()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    range_values = torch.sqrt(x**2 + y**2 + z**2)
    valid_range = (range_values < max_range) & (range_values > 0)

    azimuth = -torch.atan2(y, x) + torch.pi
    col_idx = ((azimuth / (2 * torch.pi)) * n_cols).to(torch.int64)
    col_idx = torch.clamp(col_idx, 0, n_cols - 1)

    elevation_deg = torch.rad2deg(
        torch.asin(torch.clamp(z / torch.clamp(range_values, min=1e-12), -1.0, 1.0))
    )
    rows_deg = torch.as_tensor(
        model.row_elevations_deg, dtype=torch.float32, device=device
    )
    valid_elevation = (elevation_deg >= rows_deg.min() - ELEVATION_MARGIN_DEG) & (
        elevation_deg <= rows_deg.max() + ELEVATION_MARGIN_DEG
    )
    valid = valid_range & valid_elevation

    elevation_deg = elevation_deg[valid]
    col_idx = col_idx[valid]
    range_values = range_values[valid]
    intensity_values = (
        torch.zeros_like(range_values)
        if intensity is None
        else intensity.float()[valid]
    )
    row_idx = model.nearest_rows(torch.deg2rad(elevation_deg))

    flat_idx = row_idx * n_cols + col_idx
    out_size = n_rows * n_cols
    min_range = torch.full((out_size,), torch.inf, dtype=torch.float32, device=device)
    min_range.scatter_reduce_(
        0, flat_idx, range_values, reduce="amin", include_self=True
    )

    # Stable argmin so the stored intensity belongs to the stored range.
    source_positions = torch.arange(
        range_values.numel(), dtype=torch.long, device=device
    )
    is_min = range_values == min_range[flat_idx]
    argmin = torch.full(
        (out_size,), range_values.numel(), dtype=torch.long, device=device
    )
    argmin.scatter_reduce_(
        0, flat_idx[is_min], source_positions[is_min], reduce="amin", include_self=True
    )
    hit = argmin != range_values.numel()
    chosen = argmin[hit]

    range_map = torch.zeros((out_size,), dtype=torch.float32, device=device)
    intensity_map = torch.zeros((out_size,), dtype=torch.float32, device=device)
    range_map[hit] = range_values[chosen]
    intensity_map[hit] = intensity_values[chosen]
    return range_map.view(n_rows, n_cols), intensity_map.view(n_rows, n_cols)


def project_frames_to_range_maps(
    frames: Sequence[torch.Tensor],
    model: SpinningLidarModel,
    device: torch.device | str,
    intensities: Sequence[torch.Tensor | None] | None = None,
    max_range: float = DEFAULT_MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project a list of per-frame sensor-frame point clouds.

    Returns ``(range_maps [T, H, W], intensity_maps [T, H, W] | None)`` as float32
    numpy arrays; ``intensity_maps`` is ``None`` when no intensities were given.
    """
    intensity_list: Sequence[torch.Tensor | None] = (
        intensities if intensities is not None else [None] * len(frames)
    )
    if len(intensity_list) != len(frames):
        raise ValueError("intensities must align with frames")
    range_maps: list[np.ndarray] = []
    intensity_maps: list[np.ndarray] = []
    for idx, (frame_points, frame_intensity) in enumerate(zip(frames, intensity_list)):
        range_map, intensity_map = points_to_range_map(
            frame_points.to(device),
            model,
            max_range=max_range,
            intensity=None if frame_intensity is None else frame_intensity.to(device),
        )
        range_maps.append(range_map.cpu().numpy())
        intensity_maps.append(intensity_map.cpu().numpy())
        if idx % 100 == 99:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    stacked_range = (
        np.stack(range_maps, axis=0)
        if range_maps
        else np.zeros((0, *model.shape), dtype=np.float32)
    )
    stacked_intensity = (
        np.stack(intensity_maps, axis=0)
        if (intensities is not None and intensity_maps)
        else None
    )
    return stacked_range, stacked_intensity


def range_map_to_points(
    range_map: torch.Tensor, ray_directions: torch.Tensor
) -> torch.Tensor:
    """Unproject the occupied cells of ``range_map [H, W]`` to ``[M, 3]`` sensor-frame points."""
    occupied = range_map > 0
    return range_map[occupied][:, None] * ray_directions[occupied]


# --------------------------------------------------------------------------- tar I/O
def save_range_maps_to_tar(
    range_maps: np.ndarray,
    frame_ids: Sequence[int | str],
    clip_id: str,
    tar_path: str | Path,
    intensity_maps: np.ndarray | None = None,
) -> None:
    """Write sparse range maps as one WebDataset sample (toolkit layout).

    Each frame stores the occupied pixels as ``<frame>.lidar_row.npz`` (uint8),
    ``<frame>.lidar_col.npz`` (uint16), ``<frame>.lidar_range.npz`` (float16) and,
    when given, ``<frame>.lidar_intensity.npz`` (float16). Frame ids are
    zero-padded to six digits when given as integers. ``lidar_metadata.json``
    records the source lattice dimensions for readers that support format v2.
    """
    if range_maps.ndim != 3 or any(size < 1 for size in range_maps.shape):
        raise ValueError("Expected nonempty range_maps[T, H, W]")
    if range_maps.shape[0] != len(frame_ids):
        raise ValueError("frame_ids must align with range_maps")
    if intensity_maps is not None and intensity_maps.shape != range_maps.shape:
        raise ValueError("intensity_maps must align with range_maps")
    if (
        range_maps.shape[1] > np.iinfo(np.uint8).max + 1
        or range_maps.shape[2] > np.iinfo(np.uint16).max + 1
    ):
        raise ValueError(
            "range map too large for the uint8 row / uint16 column tar layout"
        )
    sample: dict[str, object] = {
        "__key__": clip_id,
        "lidar_metadata.json": {
            "format_version": 2,
            "n_rows": int(range_maps.shape[1]),
            "n_columns": int(range_maps.shape[2]),
        },
    }
    for idx, frame_id in enumerate(frame_ids):
        key = _frame_key(frame_id)
        frame_map = range_maps[idx]
        rows, cols = np.nonzero(frame_map > 0)
        sample[f"{key}.lidar_row.npz"] = {"arr_0": rows.astype(np.uint8)}
        sample[f"{key}.lidar_col.npz"] = {"arr_0": cols.astype(np.uint16)}
        sample[f"{key}.lidar_range.npz"] = {
            "arr_0": frame_map[rows, cols].astype(np.float16)
        }
        if intensity_maps is not None:
            sample[f"{key}.lidar_intensity.npz"] = {
                "arr_0": intensity_maps[idx][rows, cols].astype(np.float16)
            }
    write_to_tar(sample, str(tar_path))


def load_range_maps_from_tar(
    tar_path: str | Path, n_rows: int, n_cols: int
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Inverse of :func:`save_range_maps_to_tar`.

    Returns ``{frame_key: (range_map [H, W] float32, intensity_map | None)}``.
    """
    sample = get_sample(str(tar_path))
    if "lidar_metadata.json" in sample:
        metadata = sample["lidar_metadata.json"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != 2
            or metadata.get("n_rows") != n_rows
            or metadata.get("n_columns") != n_cols
        ):
            raise ValueError("LiDAR metadata does not match the requested lattice")
    frames: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key in sorted(k for k in sample if k.endswith(".lidar_range.npz")):
        frame_key = key.split(".")[0]
        rows = sample[f"{frame_key}.lidar_row.npz"]["arr_0"].astype(np.int64)
        cols = sample[f"{frame_key}.lidar_col.npz"]["arr_0"].astype(np.int64)
        range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        range_map[rows, cols] = sample[key]["arr_0"].astype(np.float32)
        intensity_map: np.ndarray | None = None
        intensity_key = f"{frame_key}.lidar_intensity.npz"
        if intensity_key in sample:
            intensity_map = np.zeros((n_rows, n_cols), dtype=np.float32)
            intensity_map[rows, cols] = sample[intensity_key]["arr_0"].astype(
                np.float32
            )
        frames[frame_key] = (range_map, intensity_map)
    return frames


def _frame_key(frame_id: int | str) -> str:
    return (
        f"{frame_id:06d}" if isinstance(frame_id, (int, np.integer)) else str(frame_id)
    )
