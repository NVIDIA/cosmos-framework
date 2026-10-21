# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Recorded ``lidar_raw`` sweeps -> as-captured sensor-frame points.

RDS-HQ publishes every sweep motion-compensated to the sweep-start sensor frame
(``xyz``) together with the ring (``row``) and firing column (``column``) of each
return. To render the measured range map on the same lattice as the HD-map
range map, the compensation is undone with the sweep's own start/end poses
(toolkit behaviour, ``MadsLidarDataset._load_mads_lidar_without_motion_compensation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cosmos_auto.lidar.pose_source import SweepPoses, sweep_poses_from_lidar_raw
from cosmos_auto.lidar.sensor_model import SpinningLidarModel
from cosmos_auto.lidar.shutter import undo_motion_compensation
from cosmos_auto.wm.wds_utils import get_sample


@dataclass
class RawSweep:
    """One recorded sweep's returns, filtered to the model lattice.

    Only what :func:`uncompensated_sweep_points` needs is kept; the sweep's pose
    and timestamp come from :func:`~cosmos_auto.lidar.pose_source.sweep_poses_from_lidar_raw`.
    """

    frame_id: int
    points: np.ndarray  # [N, 3] float32, sweep-start sensor frame (as published)
    intensity: np.ndarray  # [N] float32
    column: np.ndarray  # [N] int64 firing column


def load_lidar_raw(input_root: str | Path, clip_id: str) -> dict:
    """Decoded ``lidar_raw/<clip>.tar`` sample."""
    return get_sample(str(Path(input_root) / "lidar_raw" / f"{clip_id}.tar"))


def raw_sweeps(
    lidar_sample: dict, model: SpinningLidarModel, frame_ids: list[int] | None = None
) -> list[RawSweep]:
    """Extract per-frame sweeps, dropping returns outside the ``[n_rows, n_columns]`` lattice."""
    sweeps: list[RawSweep] = []
    wanted = None if frame_ids is None else set(frame_ids)
    for key in sorted(k for k in lidar_sample if k.endswith(".lidar_raw.npz")):
        frame_id = int(key.split(".")[0])
        if wanted is not None and frame_id not in wanted:
            continue
        raw = lidar_sample[key]
        row = raw["row"].astype(np.int64)
        column = raw["column"].astype(np.int64)
        keep = (
            (row >= 0)
            & (row < model.n_rows)
            & (column >= 0)
            & (column < model.n_columns)
        )
        sweeps.append(
            RawSweep(
                frame_id=frame_id,
                points=raw["xyz"].astype(np.float32)[keep],
                intensity=raw["intensity"].astype(np.float32)[keep],
                column=column[keep],
            )
        )
    return sweeps


def uncompensated_sweep_points(
    sweeps: list[RawSweep],
    sweep_poses: SweepPoses,
    model: SpinningLidarModel,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Undo motion compensation for every sweep that has bracketing poses.

    Each return fired at relative time ``column / (n_columns - 1)``; the sweep-start
    frame is moved to the sensor frame at that time using the recorded start and
    end poses. Returns ``(points, intensities)`` aligned with ``sweep_poses.frame_ids``.
    """
    by_frame = {sweep.frame_id: sweep for sweep in sweeps}
    points_out: list[torch.Tensor] = []
    intensity_out: list[torch.Tensor] = []
    for idx, frame_id in enumerate(sweep_poses.frame_ids):
        sweep = by_frame[frame_id]
        start = torch.as_tensor(
            sweep_poses.lidar_to_world_start[idx], dtype=torch.float64, device=device
        )
        end = torch.as_tensor(
            sweep_poses.lidar_to_world_end[idx], dtype=torch.float64, device=device
        )
        sensor_end_from_start = torch.linalg.inv(end) @ start
        points = torch.as_tensor(sweep.points, dtype=torch.float64, device=device)
        relative_time = torch.as_tensor(
            sweep.column, dtype=torch.float64, device=device
        ) / (model.n_columns - 1)
        points_out.append(
            undo_motion_compensation(
                points, sensor_end_from_start, relative_time
            ).float()
        )
        intensity_out.append(
            torch.as_tensor(sweep.intensity, dtype=torch.float32, device=device)
        )
    return points_out, intensity_out


def sweep_poses_and_points(
    lidar_sample: dict, model: SpinningLidarModel, device: torch.device | str = "cpu"
) -> tuple[SweepPoses, list[torch.Tensor], list[torch.Tensor]]:
    """Convenience: recorded sweep poses plus uncompensated points/intensities."""
    sweep_poses = sweep_poses_from_lidar_raw(lidar_sample)
    sweeps = raw_sweeps(lidar_sample, model, sweep_poses.frame_ids)
    points, intensities = uncompensated_sweep_points(sweeps, sweep_poses, model, device)
    return sweep_poses, points, intensities
