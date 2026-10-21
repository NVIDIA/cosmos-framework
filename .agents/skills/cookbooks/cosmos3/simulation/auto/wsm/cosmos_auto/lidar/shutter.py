# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rolling-shutter (spinning head) motion model for one LiDAR sweep.

A sweep lasts from ``t_start`` to ``t_end`` while the ego vehicle moves. The
head fires each column at a different time, so a static world point is observed
from the sensor pose *at its firing time*. This module provides

* :func:`interpolate_rigid` - slerp/lerp between two rigid transforms,
* :func:`apply_rigid` / :func:`apply_rigid_inverse` - per-point application of
  interpolated transforms (sensor -> world and world -> sensor), and
* :func:`world_points_to_shutter_sensor_frame` - the fixed-point solver that
  finds, for every world point, the firing time consistent with the sensor
  pose at that time (port of NCORE's
  ``world_points_to_sensor_angles_shutter_pose``), returning the points in the
  moving sensor frame exactly as a real sweep would record them.

The solver interpolates the *forward* sensor poses (``lidar_to_world`` at sweep
start and end) and then expresses each world point in the interpolated frame,
``p_sensor = R(t)^T (p_world - t(t))``. Interpolating the inverse transforms
instead would lerp ``-R^T t``, which is not the inverse of the lerped forward
pose and makes the result depend on the ego's distance from the clip world
origin; the forward form is translation-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cosmos_auto.lidar.sensor_model import SpinningLidarModel


def _quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Unit quaternion ``[w, x, y, z]`` of a single ``[3, 3]`` rotation matrix."""
    xyzw = Rotation.from_matrix(rotation.detach().cpu().double().numpy()).as_quat()
    wxyz = np.roll(xyzw, 1)
    return torch.as_tensor(wxyz, dtype=rotation.dtype, device=rotation.device)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation between two unit quaternions ``[4]`` at ``t [N]`` -> ``[N, 4]``."""
    dot = torch.dot(q0, q1)
    if dot < 0:  # take the short arc
        q1 = -q1
        dot = -dot
    dot = torch.clamp(dot, -1.0, 1.0)
    if float(dot) > 0.9995:  # nearly identical: normalised lerp
        out = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
        return out / torch.linalg.norm(out, dim=1, keepdim=True)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    return w0[:, None] * q0[None, :] + w1[:, None] * q1[None, :]


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """``[N, 4]`` unit quaternions ``[w, x, y, z]`` -> ``[N, 3, 3]`` rotation matrices."""
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        [
            torch.stack(
                [
                    1 - two * (y * y + z * z),
                    two * (x * y - z * w),
                    two * (x * z + y * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    two * (x * y + z * w),
                    1 - two * (x * x + z * z),
                    two * (y * z - x * w),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    two * (x * z - y * w),
                    two * (y * z + x * w),
                    1 - two * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def interpolate_rigid(
    transform_start: torch.Tensor,
    transform_end: torch.Tensor,
    relative_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate two ``[4, 4]`` rigid transforms at ``relative_time [N]`` in ``[0, 1]``.

    Rotation uses quaternion slerp, translation linear interpolation (the NCORE
    ``PoseInterpolator`` convention). Returns ``(rotations [N, 3, 3], translations [N, 3])``.
    """
    q0 = _quaternion_wxyz(transform_start[:3, :3])
    q1 = _quaternion_wxyz(transform_end[:3, :3])
    t = relative_time.to(transform_start.dtype)
    rotations = _quaternion_to_matrix(_slerp(q0, q1, t))
    translations = (1.0 - t)[:, None] * transform_start[:3, 3][None, :] + t[
        :, None
    ] * transform_end[:3, 3][None, :]
    return rotations, translations


def apply_rigid(
    rotations: torch.Tensor, translations: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Apply per-point rigid transforms ``[N, 3, 3]``, ``[N, 3]`` to ``points [N, 3]``."""
    return torch.einsum("nij,nj->ni", rotations, points) + translations


def apply_rigid_inverse(
    rotations: torch.Tensor, translations: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Apply the inverse of per-point rigid transforms to ``points [N, 3]``: ``R^T (p - t)``.

    With ``(rotations, translations)`` being sensor -> world poses this expresses
    world points in the per-point sensor frame.
    """
    return torch.einsum("nji,nj->ni", rotations, points - translations)


def undo_motion_compensation(
    points_sweep_start: torch.Tensor,
    sensor_end_from_sensor_start: torch.Tensor,
    relative_time: torch.Tensor,
) -> torch.Tensor:
    """Move motion-compensated points back into the sensor frame at their firing time.

    Recorded sweeps are published compensated to the sweep-start sensor frame.
    Given the rigid transform from that frame to the sweep-end sensor frame and
    each point's relative firing time, this returns the raw (uncompensated)
    coordinates, i.e. what the spinning sensor measured.
    """
    identity = torch.eye(
        4, dtype=points_sweep_start.dtype, device=points_sweep_start.device
    )
    rotations, translations = interpolate_rigid(
        identity, sensor_end_from_sensor_start.to(points_sweep_start), relative_time
    )
    return apply_rigid(rotations, translations, points_sweep_start)


@dataclass
class ShutterProjection:
    """World points re-expressed in the moving sensor frame at their firing times."""

    points_sensor: torch.Tensor  # [M, 3]
    relative_time: torch.Tensor  # [M] in [0, 1]
    valid_indices: torch.Tensor  # [M] indices into the input world points


@torch.no_grad()
def world_points_to_shutter_sensor_frame(
    model: SpinningLidarModel,
    world_points: torch.Tensor,
    lidar_to_world_start: torch.Tensor | np.ndarray,
    lidar_to_world_end: torch.Tensor | np.ndarray,
    max_iterations: int = 10,
    stop_mean_relative_time_error: float = 1e-4,
    stop_delta_mean_relative_time_error: float = 1e-6,
) -> ShutterProjection:
    """Find each world point's firing time and its coordinates in the sensor frame then.

    Fixed-point iteration (NCORE's ``world_points_to_sensor_angles_shutter_pose``):
    start from the sweep-start pose (or the sweep-end pose for points only
    visible there), read the firing time from the azimuth the point projects to,
    move the sensor to that time (slerp/lerp of the forward ``lidar_to_world``
    poses), re-express the point in that sensor frame, and repeat until the mean
    change in relative time is below ``stop_mean_relative_time_error`` or stops
    improving. Points that leave the vertical field of view are dropped.

    Args:
        model: lattice used for the azimuth -> firing-time lookup.
        world_points: ``[N, 3]`` points in the clip world frame.
        lidar_to_world_start / lidar_to_world_end: sensor poses at sweep start/end.

    Returns:
        :class:`ShutterProjection` with the surviving points in the *moving* sensor
        frame (ready for :func:`cosmos_auto.lidar.rangemap.points_to_range_map`).
    """
    device = world_points.device
    dtype = world_points.dtype
    pose_start = torch.as_tensor(lidar_to_world_start, dtype=dtype, device=device)
    pose_end = torch.as_tensor(lidar_to_world_end, dtype=dtype, device=device)

    # R^T (p - t) for a single pose: ``(p - t) @ R`` equals ``R^T (p - t)`` row-wise.
    points_start = (world_points - pose_start[:3, 3]) @ pose_start[:3, :3]
    points_end = (world_points - pose_end[:3, 3]) @ pose_end[:3, :3]
    _, _, valid_start = model.sensor_angles(points_start)
    _, _, valid_end = model.sensor_angles(points_end)
    valid = valid_start | valid_end
    if not bool(valid.any()):
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return ShutterProjection(
            points_sensor=torch.empty((0, 3), dtype=dtype, device=device),
            relative_time=torch.empty((0,), dtype=dtype, device=device),
            valid_indices=empty,
        )

    candidate_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    candidates = world_points[candidate_indices]
    # Prefer the sweep-start projection as the initial guess (NCORE behaviour).
    previous_points = torch.where(
        valid_start[candidate_indices, None],
        points_start[candidate_indices],
        points_end[candidate_indices],
    )
    previous_time = torch.where(
        valid_start[candidate_indices],
        torch.zeros_like(previous_points[:, 0]),
        torch.ones_like(previous_points[:, 0]),
    )
    # NCORE seeds this with 0.5, so a first update that happens to average 0.5
    # (e.g. one point per column around a full ring) passed the stagnation test
    # after a single iteration. Only compare once a previous iteration exists.
    previous_error: float | None = None

    relative_time = previous_time
    current_points = previous_points
    current_valid = torch.ones_like(previous_time, dtype=torch.bool)
    for _ in range(max_iterations):
        relative_time, _ = model.relative_frame_times(previous_points)
        # Interpolate the forward poses, then express each world point in its own sensor frame.
        rotations, translations = interpolate_rigid(pose_start, pose_end, relative_time)
        current_points = apply_rigid_inverse(rotations, translations, candidates)
        _, _, current_valid = model.sensor_angles(current_points)

        if bool(current_valid.any()):
            error = float(
                (relative_time[current_valid] - previous_time[current_valid])
                .abs()
                .mean()
            )
        else:
            error = 0.0
        converged = (
            previous_error is not None
            and abs(previous_error - error) <= stop_delta_mean_relative_time_error
        )
        previous_error = error
        if converged or error <= stop_mean_relative_time_error:
            break
        previous_points = torch.where(
            current_valid[:, None], current_points, previous_points
        )
        previous_time = torch.where(current_valid, relative_time, previous_time)

    return ShutterProjection(
        points_sensor=current_points[current_valid],
        relative_time=relative_time[current_valid],
        valid_indices=candidate_indices[current_valid],
    )
