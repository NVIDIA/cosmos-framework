# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
GPU-accelerated collision detection between ego vehicle and dynamic obstacles.

Uses pure PyTorch ops on CUDA tensors — no custom .cu kernels needed.
All data stays on GPU; the scene loader already places ego + obstacle
tensors in device memory.

Algorithm:
  1. Match obstacle pose timestamps to nearest ego pose via searchsorted.
  2. Extract 2D yaw from z-only quaternions: yaw = 2 * atan2(qz, qw).
  3. Vectorized 2D OBB-OBB separating-axis-theorem (SAT) test across
     all obstacle poses simultaneously.

Includes a minimal "collision-only" loader that extracts only
egomotion_estimate + object_fused from tar files, skipping all map
geometry for ~5-10x faster I/O than full scene loading.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ._ops import PRIM_OBSTACLE, CubePool
from .clipgt import ClipgtGpuScene, EgoTrackData

# True ego vehicle half-extents in meters (not the BEV-inflated values).
EGO_HALF_SIZE_XY = (4.5 / 2.0, 2.0 / 2.0)

# Maximum time gap (microseconds) for matching ego ↔ obstacle timestamps.
DEFAULT_TIME_TOLERANCE_US = 100_000  # 100 ms


@dataclass
class CollisionEvent:
    """Single ego-obstacle overlap at one timestamp."""

    timestamp_us: int
    track_idx: int
    distance_m: float


@dataclass
class CollisionResult:
    """Collision detection output for one scene."""

    has_collision: bool
    events: List[CollisionEvent] = field(default_factory=list)
    skipped: bool = False


def _quat_to_yaw(q: Tensor) -> Tensor:
    """Extract yaw angle from z-only quaternions [N, 4] (x, y, z, w)."""
    return 2.0 * torch.atan2(q[:, 2], q[:, 3])


def _obb_overlap_2d(
    pos_a: Tensor,  # [N, 2]
    yaw_a: Tensor,  # [N]
    half_a: Tensor,  # [N, 2] or [2]
    pos_b: Tensor,  # [N, 2]
    yaw_b: Tensor,  # [N]
    half_b: Tensor,  # [N, 2] or [2]
) -> Tensor:
    """Vectorised 2D OBB-OBB SAT overlap test.

    Returns a boolean tensor [N] — True where boxes overlap.
    All 4 separating axes are evaluated in a single batched matmul
    to avoid Python-level loops.
    """
    N = pos_a.shape[0]
    cos_a = torch.cos(yaw_a)
    sin_a = torch.sin(yaw_a)
    cos_b = torch.cos(yaw_b)
    sin_b = torch.sin(yaw_b)

    d = pos_b - pos_a  # [N, 2]

    # Stack all 4 axis directions into [N, 4, 2]:
    #   axis 0: a's local x  ( cos_a,  sin_a)
    #   axis 1: a's local y  (-sin_a,  cos_a)
    #   axis 2: b's local x  ( cos_b,  sin_b)
    #   axis 3: b's local y  (-sin_b,  cos_b)
    axes = torch.stack(
        [
            torch.stack([cos_a, sin_a], dim=1),
            torch.stack([-sin_a, cos_a], dim=1),
            torch.stack([cos_b, sin_b], dim=1),
            torch.stack([-sin_b, cos_b], dim=1),
        ],
        dim=1,
    )  # [N, 4, 2]

    # Project separation vector onto each axis: [N, 4]
    proj_d = (axes * d.unsqueeze(1)).sum(dim=2)

    # 2x2 rotation matrices for a and b
    rot_a = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=1).view(N, 2, 2)
    rot_b = torch.stack([cos_b, -sin_b, sin_b, cos_b], dim=1).view(N, 2, 2)

    # Project rotation columns onto axes: |axis · rot_col| → [N, 4, 2]
    # For box a: each column of rot_a dotted with each axis
    abs_a = torch.abs(torch.bmm(axes, rot_a))  # [N, 4, 2]
    abs_b = torch.abs(torch.bmm(axes, rot_b))  # [N, 4, 2]

    # Half-extent projections: sum(|axis·col_i| * half_i) for each box
    if half_a.dim() == 1:
        half_a = half_a.unsqueeze(0)  # [1, 2]
    if half_b.dim() == 1:
        half_b = half_b.unsqueeze(0)  # [1, 2]

    proj_a = (abs_a * half_a.unsqueeze(1)).sum(dim=2)  # [N, 4]
    proj_b = (abs_b * half_b.unsqueeze(1)).sum(dim=2)  # [N, 4]

    # SAT: overlap iff no separating axis exists
    separated = torch.abs(proj_d) > (proj_a + proj_b)  # [N, 4]
    return ~separated.any(dim=1)  # [N]


def detect_collisions_gpu(
    ego_track: EgoTrackData,
    cube_pools: List[CubePool],
    ego_half_size_xy: Tuple[float, float] = EGO_HALF_SIZE_XY,
    time_tolerance_us: int = DEFAULT_TIME_TOLERANCE_US,
) -> CollisionResult:
    """Detect ego-obstacle collisions using 2D OBB overlap on GPU.

    Args:
        ego_track: Ego vehicle trajectory (timestamps + poses on CUDA).
        cube_pools: List of CubePool from TimestampedScene.cube_pools.
        ego_half_size_xy: (half_length, half_width) of the ego vehicle in metres.
        time_tolerance_us: Max time gap in microseconds for timestamp matching.

    Returns:
        CollisionResult with boolean flag and per-event metadata.
    """
    # Find the dynamic obstacle pool
    obs_pool: Optional[CubePool] = None
    for pool in cube_pools or []:
        if pool.prim_type_id == PRIM_OBSTACLE:
            obs_pool = pool
            break

    if obs_pool is None or obs_pool.translations.shape[0] == 0:
        return CollisionResult(has_collision=False)

    device = ego_track.timestamps.device
    ego_ts = ego_track.timestamps  # [n_ego] int64, sorted
    ego_pos = ego_track.translations  # [n_ego, 3]
    ego_quat = ego_track.quaternions  # [n_ego, 4]

    obs_ts = obs_pool.track_timestamps_us  # [n_obs_poses] int64
    obs_pos = obs_pool.translations  # [n_obs_poses, 3]
    obs_quat = obs_pool.quaternions  # [n_obs_poses, 4]

    n_ego = ego_ts.shape[0]
    n_obs = obs_ts.shape[0]
    if n_ego == 0 or n_obs == 0:
        return CollisionResult(has_collision=False)

    # --- 1. Timestamp matching via searchsorted ---
    idx = torch.searchsorted(ego_ts, obs_ts).clamp(0, n_ego - 1)

    # Check neighbour to find true nearest
    idx_lo = (idx - 1).clamp(0, n_ego - 1)
    diff_hi = torch.abs(ego_ts[idx] - obs_ts)
    diff_lo = torch.abs(ego_ts[idx_lo] - obs_ts)
    use_lo = diff_lo < diff_hi
    nearest_idx = torch.where(use_lo, idx_lo, idx)
    time_gap = torch.where(use_lo, diff_lo, diff_hi)

    valid_time = time_gap <= time_tolerance_us
    if not valid_time.any():
        return CollisionResult(has_collision=False)

    # --- 2. Gather ego poses at matched timestamps ---
    ego_xy = ego_pos[nearest_idx, :2]  # [n_obs, 2]
    ego_q = ego_quat[nearest_idx]  # [n_obs, 4]
    ego_yaw = _quat_to_yaw(ego_q)  # [n_obs]

    obs_xy = obs_pos[:, :2]  # [n_obs, 2]
    obs_yaw = _quat_to_yaw(obs_quat)  # [n_obs]

    # --- 3. Per-pose obstacle half-extents (expand from per-track scales) ---
    # cube_ts_prefix_sum is cumulative track lengths [n_tracks].
    prefix = obs_pool.cube_ts_prefix_sum  # [n_tracks] int32
    scales = obs_pool.scales  # [n_tracks, 3] (full extents)

    # Map each pose index → track index
    track_idx = torch.searchsorted(prefix, torch.arange(1, n_obs + 1, device=device, dtype=prefix.dtype))
    obs_half_xy = scales[track_idx, :2] * 0.5  # [n_obs, 2]

    ego_half = torch.tensor(
        [ego_half_size_xy[0], ego_half_size_xy[1]],
        dtype=torch.float32,
        device=device,
    )

    # --- 4. 2D OBB-OBB SAT test ---
    overlap = _obb_overlap_2d(ego_xy, ego_yaw, ego_half, obs_xy, obs_yaw, obs_half_xy)
    overlap &= valid_time

    if not overlap.any():
        return CollisionResult(has_collision=False)

    # --- 5. Build collision events ---
    hit_indices = overlap.nonzero(as_tuple=False).squeeze(1)
    hit_ts = obs_ts[hit_indices]
    hit_track = track_idx[hit_indices]
    hit_dist = torch.norm(ego_xy[hit_indices] - obs_xy[hit_indices], dim=1)

    # Transfer small result tensors to CPU
    hit_ts_cpu = hit_ts.cpu().tolist()
    hit_track_cpu = hit_track.cpu().tolist()
    hit_dist_cpu = hit_dist.cpu().tolist()

    events = [
        CollisionEvent(timestamp_us=int(t), track_idx=int(tr), distance_m=float(d))
        for t, tr, d in zip(hit_ts_cpu, hit_track_cpu, hit_dist_cpu)
    ]

    return CollisionResult(has_collision=True, events=events)


def detect_collisions_from_scene(
    scene: ClipgtGpuScene,
    ego_half_size_xy: Tuple[float, float] = EGO_HALF_SIZE_XY,
    time_tolerance_us: int = DEFAULT_TIME_TOLERANCE_US,
) -> CollisionResult:
    """Convenience wrapper: detect collisions from a loaded ClipgtGpuScene."""
    cube_pools = scene.timestamped_scene.cube_pools or []
    return detect_collisions_gpu(
        scene.ego_track,
        cube_pools,
        ego_half_size_xy=ego_half_size_xy,
        time_tolerance_us=time_tolerance_us,
    )


# ---------------------------------------------------------------------------
# Minimal CPU-only collision pipeline (no GPU, no full scene load)
# ---------------------------------------------------------------------------

import io
import os

import numpy as np


def _np_obb_overlap_2d(
    pos_a: np.ndarray,
    yaw_a: np.ndarray,
    half_a: np.ndarray,
    pos_b: np.ndarray,
    yaw_b: np.ndarray,
    half_b: np.ndarray,
) -> np.ndarray:
    """Vectorized 2D OBB-SAT on numpy arrays. Returns bool [N]."""
    cos_a, sin_a = np.cos(yaw_a), np.sin(yaw_a)
    cos_b, sin_b = np.cos(yaw_b), np.sin(yaw_b)
    d = pos_b - pos_a  # [N, 2]

    axes = np.stack(
        [
            np.stack([cos_a, sin_a], axis=1),
            np.stack([-sin_a, cos_a], axis=1),
            np.stack([cos_b, sin_b], axis=1),
            np.stack([-sin_b, cos_b], axis=1),
        ],
        axis=1,
    )  # [N, 4, 2]

    proj_d = np.sum(axes * d[:, None, :], axis=2)  # [N, 4]

    N = len(yaw_a)
    rot_a = np.stack([cos_a, -sin_a, sin_a, cos_a], axis=1).reshape(N, 2, 2)
    rot_b = np.stack([cos_b, -sin_b, sin_b, cos_b], axis=1).reshape(N, 2, 2)

    abs_a = np.abs(np.einsum("nij,njk->nik", axes, rot_a))  # [N, 4, 2]
    abs_b = np.abs(np.einsum("nij,njk->nik", axes, rot_b))  # [N, 4, 2]

    if half_a.ndim == 1:
        half_a = half_a[None, :]
    if half_b.ndim == 1:
        half_b = half_b[None, :]

    proj_a = np.sum(abs_a * half_a[:, None, :], axis=2)  # [N, 4]
    proj_b = np.sum(abs_b * half_b[:, None, :], axis=2)  # [N, 4]

    separated = np.abs(proj_d) > (proj_a + proj_b)  # [N, 4]
    return ~np.any(separated, axis=1)  # [N]


def _parse_ego_numpy(parquet_bytes: bytes):
    """Parse egomotion parquet → (timestamps_i64, positions_f32[N,2], quats_f32[N,4])."""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(parquet_bytes), columns=["key", "egomotion_estimate"])
    if len(table) == 0:
        return None
    key_col = table.column("key").combine_chunks()
    ego_col = table.column("egomotion_estimate").combine_chunks()

    ts = key_col.field("timestamp_micros").to_numpy().astype(np.int64)
    loc = ego_col.field("location")
    ori = ego_col.field("orientation")

    pos = np.column_stack(
        [
            loc.field("x").to_numpy(zero_copy_only=False),
            loc.field("y").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32)

    quat = np.column_stack(
        [
            ori.field("x").to_numpy(zero_copy_only=False),
            ori.field("y").to_numpy(zero_copy_only=False),
            ori.field("z").to_numpy(zero_copy_only=False),
            ori.field("w").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32)

    return ts, pos, quat


def _parse_obs_flat_numpy(parquet_bytes: bytes):
    """Parse obstacle parquet → flat numpy arrays (no per-track grouping).

    Returns (timestamps_i64, pos_f32[N,2], yaw_f32[N], half_xy_f32[N,2])
    or None if empty.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(parquet_bytes), columns=["key", "object_fused"])
    if len(table) == 0:
        return None
    obj_col = table.column("object_fused").combine_chunks()
    key_col = table.column("key").combine_chunks()

    ts = key_col.field("timestamp_micros").to_numpy().astype(np.int64)

    center = obj_col.field("cuboid_3D_center")
    pos = np.column_stack(
        [
            center.field("x").to_numpy(zero_copy_only=False),
            center.field("y").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32)

    direction = obj_col.field("obstacle_direction")
    dx = direction.field("x").to_numpy(zero_copy_only=False)
    dy = direction.field("y").to_numpy(zero_copy_only=False)
    yaw = np.arctan2(dy, dx).astype(np.float32)

    half_axis = obj_col.field("cuboid_3D_halfAxisXYZ")
    half_xy = np.column_stack(
        [
            half_axis.field("x").to_numpy(zero_copy_only=False),
            half_axis.field("y").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32)

    # Filter rows with valid obstacle_id
    obstacle_ids = obj_col.field("obstacle_id").to_numpy(zero_copy_only=False)
    valid = ~np.isnan(obstacle_ids.astype(np.float64))
    if not valid.all():
        idx = np.where(valid)[0]
        ts, pos, yaw, half_xy = ts[idx], pos[idx], yaw[idx], half_xy[idx]

    return ts, pos, yaw, half_xy


def detect_collisions_cpu(
    tar_path: str,
    ego_half_size_xy: Tuple[float, float] = EGO_HALF_SIZE_XY,
    time_tolerance_us: int = DEFAULT_TIME_TOLERANCE_US,
) -> CollisionResult:
    """CPU-only collision detection — single bulk tar read, direct numpy parsing.

    Reads the entire tar into memory in one syscall (Lustre-friendly), then
    extracts only egomotion_estimate + object_fused parquets from the in-memory
    buffer. All parsing produces flat numpy arrays directly — no torch tensors,
    no per-track grouping.
    """
    import tarfile

    if not os.path.exists(tar_path):
        return CollisionResult(has_collision=False, skipped=True)

    with open(tar_path, "rb") as fh:
        raw = fh.read()
    tf = tarfile.open(fileobj=io.BytesIO(raw))

    ego_bytes = obs_bytes = None
    while True:
        m = tf.next()
        if m is None:
            break
        name = m.name.rsplit("/", 1)[-1] if "/" in m.name else m.name
        if name == "egomotion_estimate.parquet":
            ego_bytes = tf.extractfile(m).read()
        elif name == "object_fused.parquet":
            obs_bytes = tf.extractfile(m).read()
        if ego_bytes is not None and obs_bytes is not None:
            break
    tf.close()

    if ego_bytes is None or obs_bytes is None:
        return CollisionResult(has_collision=False)

    ego_parsed = _parse_ego_numpy(ego_bytes)
    obs_parsed = _parse_obs_flat_numpy(obs_bytes)
    if ego_parsed is None or obs_parsed is None:
        return CollisionResult(has_collision=False)

    ego_ts, ego_pos, ego_quat = ego_parsed
    obs_ts, obs_pos, obs_yaw, obs_half_xy = obs_parsed

    n_ego = len(ego_ts)
    n_obs = len(obs_ts)
    if n_ego == 0 or n_obs == 0:
        return CollisionResult(has_collision=False)

    # Timestamp matching: find nearest ego pose for each obstacle pose
    idx = np.searchsorted(ego_ts, obs_ts).clip(0, n_ego - 1)
    idx_lo = (idx - 1).clip(0, n_ego - 1)
    diff_hi = np.abs(ego_ts[idx] - obs_ts)
    diff_lo = np.abs(ego_ts[idx_lo] - obs_ts)
    use_lo = diff_lo < diff_hi
    nearest_idx = np.where(use_lo, idx_lo, idx)
    time_gap = np.where(use_lo, diff_lo, diff_hi)

    valid_time = time_gap <= time_tolerance_us
    if not np.any(valid_time):
        return CollisionResult(has_collision=False)

    ego_xy = ego_pos[nearest_idx]
    ego_yaw = 2.0 * np.arctan2(ego_quat[nearest_idx, 2], ego_quat[nearest_idx, 3])
    ego_half = np.array(ego_half_size_xy, dtype=np.float32)

    overlap = _np_obb_overlap_2d(ego_xy, ego_yaw, ego_half, obs_pos, obs_yaw, obs_half_xy)
    overlap &= valid_time

    if not np.any(overlap):
        return CollisionResult(has_collision=False)

    hit = np.where(overlap)[0]
    events = [
        CollisionEvent(
            timestamp_us=int(obs_ts[i]),
            track_idx=int(i),
            distance_m=float(np.linalg.norm(ego_xy[i] - obs_pos[i])),
        )
        for i in hit
    ]
    return CollisionResult(has_collision=True, events=events)
