# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scene augmentation utilities for extending scenes via plane reflection.

Given a loaded ClipgtGpuScene, creates an extended scene by iteratively:
1. Walking along the nearest lane/boundary polyline to find the mirror point
2. Placing the mirror plane perpendicular to the lane at that point
3. Reflecting the time-reversed scene about the mirror plane
4. Clipping each segment at its boundary planes to eliminate element overlap

The mirror plane is derived entirely from road geometry -- the nearest lane
or road boundary polyline is followed forward by a configurable distance,
and the lane tangent at that point becomes the plane normal.  This guarantees
C1 continuity of lane lines at segment boundaries.  Ego heading in augmented
segments is computed from the trajectory tangent so the vehicle always faces
its direction of travel.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ._ops import (
    PRIM_EGO_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_LANE_BOUNDARY,
    PRIM_LANE_LINE,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_OBSTACLE,
    PRIM_ROAD_BOUNDARY,
    CubePool,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
)
from .clipgt import (
    ClipgtGpuScene,
    EgoTrackData,
    _ego_obstacle_to_pool,
    _ego_trajectory_to_pool,
)

_ROAD_PRIM_TYPES = frozenset(
    {
        PRIM_ROAD_BOUNDARY,
        PRIM_LANE_LINE,
        PRIM_LANE_BOUNDARY,
        PRIM_LANE_LINE_WHITE_SOLID,
        PRIM_LANE_LINE_WHITE_DASHED,
        PRIM_LANE_LINE_YELLOW_SOLID,
        PRIM_LANE_LINE_YELLOW_DASHED,
    }
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _forward_from_quat(q: Tensor) -> Tensor:
    """Rotate [1, 0, 0] by a quaternion in xyzw format to get the forward direction.

    Args:
        q: Quaternion(s) with shape ``[..., 4]`` in ``(x, y, z, w)`` order.

    Returns:
        Forward direction(s) with shape ``[..., 3]``.
    """
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy + qw * qz),
            2 * (qx * qz - qw * qy),
        ],
        dim=-1,
    )


def _reflect_positions(positions: Tensor, center: Tensor, normal: Tensor) -> Tensor:
    """Reflect positions about a plane defined by *center* and *normal*.

    ``p' = p - 2 * dot(p - center, n_hat) * n_hat``

    Args:
        positions: ``[N, 3]`` float32.
        center: ``[3]`` float32, a point on the mirror plane.
        normal: ``[3]`` float32, unit normal of the mirror plane.

    Returns:
        Reflected positions ``[N, 3]`` float32.
    """
    diff = positions - center
    dots = (diff * normal).sum(dim=-1, keepdim=True)
    return positions - 2 * dots * normal


def _reflect_quaternions(quaternions: Tensor, normal: Tensor) -> Tensor:
    """Reflect orientation quaternions about a plane with the given *normal*.

    A plane reflection is an improper rotation (det = -1) and cannot be
    represented directly as a quaternion.  Instead we reflect the forward
    direction vector ``R * [1,0,0]`` about the plane, then convert the
    reflected heading to a yaw-only quaternion.  This is correct for
    ground-plane objects where pitch and roll are negligible.

    Args:
        quaternions: ``[..., 4]`` float32 in ``(x, y, z, w)`` order.
        normal: ``[3]`` float32, unit normal of the mirror plane.

    Returns:
        Reflected quaternions ``[..., 4]`` float32.
    """
    fwd = _forward_from_quat(quaternions)
    dots = (fwd * normal).sum(dim=-1, keepdim=True)
    fwd_ref = fwd - 2 * dots * normal

    yaw = torch.atan2(fwd_ref[..., 1], fwd_ref[..., 0])
    half_yaw = yaw * 0.5

    result = torch.zeros_like(quaternions)
    result[..., 2] = torch.sin(half_yaw)
    result[..., 3] = torch.cos(half_yaw)
    return result


# ---------------------------------------------------------------------------
# Rigid-transform helpers (yaw rotation + translation, det=+1)
# ---------------------------------------------------------------------------


def _yaw_from_positions(positions: Tensor, n: int = 5) -> float:
    """Extract yaw angle (radians) from the tangent of the last *n* positions."""
    if len(positions) < 2:
        return 0.0
    n = min(n, len(positions))
    d = positions[-1] - positions[-n]
    if d[:2].norm().item() < 1e-6:
        return 0.0
    return float(torch.atan2(d[1], d[0]).item())


def _rigid_transform_positions(
    positions: Tensor,
    yaw: float,
    pivot: Tensor,
    offset: Tensor,
) -> Tensor:
    """Rotate *positions* by *yaw* around *pivot* on the ground plane, then
    translate so that *pivot* maps to *offset*.

    ``p' = R(yaw) * (p - pivot) + offset``
    """
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    rel = positions - pivot
    out = positions.clone()
    out[..., 0] = c * rel[..., 0] - s * rel[..., 1] + offset[0]
    out[..., 1] = s * rel[..., 0] + c * rel[..., 1] + offset[1]
    out[..., 2] = rel[..., 2] + offset[2]
    return out


def _yaw_rotate_quaternions(quaternions: Tensor, yaw: float) -> Tensor:
    """Pre-multiply quaternions (xyzw) by a yaw-only rotation."""
    half = yaw * 0.5
    s = float(torch.sin(torch.tensor(half)))
    c = float(torch.cos(torch.tensor(half)))
    x, y, z, w = (quaternions[..., i] for i in range(4))
    out = torch.empty_like(quaternions)
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    out[..., 2] = s * w + c * z
    out[..., 3] = c * w - s * z
    return out


def _rigid_transform_polyline_pool(
    pool: TimestampedPolylinePool,
    yaw: float,
    pivot: Tensor,
    offset: Tensor,
) -> TimestampedPolylinePool:
    """Apply a rigid ground-plane transform to a polyline pool."""
    return TimestampedPolylinePool(
        timestamps_us=pool.timestamps_us.clone(),
        timestamped_varrays_prefix_sum=pool.timestamped_varrays_prefix_sum.clone(),
        varrays_prefix_sum=pool.varrays_prefix_sum.clone(),
        vertices=_rigid_transform_positions(pool.vertices, yaw, pivot, offset),
        prim_type_id=pool.prim_type_id,
    )


def _rigid_transform_polygon_pool(
    pool: TimestampedPolygonPool,
    yaw: float,
    pivot: Tensor,
    offset: Tensor,
) -> TimestampedPolygonPool:
    """Apply a rigid ground-plane transform to a polygon pool (det=+1, no winding swap)."""
    return TimestampedPolygonPool(
        timestamps_us=pool.timestamps_us.clone(),
        timestamped_varrays_prefix_sum=pool.timestamped_varrays_prefix_sum.clone(),
        varrays_prefix_sum=pool.varrays_prefix_sum.clone(),
        triangle_prefix_sum=pool.triangle_prefix_sum.clone(),
        vertices=_rigid_transform_positions(pool.vertices, yaw, pivot, offset),
        triangles=pool.triangles.clone(),
        prim_type_id=pool.prim_type_id,
    )


def _rigid_transform_cube_pool(
    pool: CubePool,
    yaw: float,
    pivot: Tensor,
    offset: Tensor,
    time_offset_us: int,
) -> CubePool:
    """Apply a rigid ground-plane transform + timestamp shift to a cube pool."""
    return CubePool(
        timestamps_us=pool.timestamps_us.clone() + time_offset_us,
        cube_ts_prefix_sum=pool.cube_ts_prefix_sum.clone(),
        track_timestamps_us=pool.track_timestamps_us.clone() + time_offset_us,
        translations=_rigid_transform_positions(pool.translations, yaw, pivot, offset),
        quaternions=_yaw_rotate_quaternions(pool.quaternions, yaw),
        scales=pool.scales.clone(),
        colors=pool.colors.clone(),
        prim_type_id=pool.prim_type_id,
        render_flags=pool.render_flags,
    )


def _rigid_transform_ego_track(
    ego: EgoTrackData,
    yaw: float,
    pivot: Tensor,
    offset: Tensor,
    time_offset_us: int,
) -> EgoTrackData:
    """Apply a rigid ground-plane transform + timestamp shift to an ego track."""
    new_pos = _rigid_transform_positions(ego.translations, yaw, pivot, offset)
    new_quat = _yaw_rotate_quaternions(ego.quaternions, yaw)
    return EgoTrackData(
        timestamps=ego.timestamps + time_offset_us,
        poses_tquat=torch.cat([new_pos, new_quat], dim=-1),
    )


def _lane_based_mirror_plane(
    polyline_pools: List[TimestampedPolylinePool],
    query_point: Tensor,
    forward_hint: Tensor,
    walk_distance: float,
) -> Tuple[Tensor, Tensor]:
    """Determine a mirror plane by walking along the nearest lane polyline.

    1. Finds the nearest road polyline segment to *query_point*.
    2. Orients the polyline so its traversal direction matches *forward_hint*.
    3. Walks *walk_distance* metres along the polyline from the closest point.
    4. Returns ``(center, normal)`` -- the reached point and the lane tangent
       at that point (unit vector).

    If the polyline ends before *walk_distance* is exhausted, the walk
    continues linearly along the last segment's tangent.

    Falls back to ``query_point + forward_hint * walk_distance`` when no
    road geometry is found.
    """
    dev = query_point.device
    best_dist = float("inf")
    best_polyline: Optional[Tensor] = None
    best_seg_idx = 0
    best_t = 0.0

    for pool in polyline_pools:
        if pool.prim_type_id not in _ROAD_PRIM_TYPES:
            continue
        verts = pool.vertices.to(dev)
        vps = pool.varrays_prefix_sum

        poly_start = 0
        for j in range(len(vps)):
            poly_end = vps[j].item()
            if poly_end - poly_start < 2:
                poly_start = poly_end
                continue
            pv = verts[poly_start:poly_end]

            seg_a = pv[:-1]
            seg_b = pv[1:]
            ab = seg_b - seg_a
            ap = query_point.unsqueeze(0) - seg_a
            ab_len_sq = (ab * ab).sum(dim=-1).clamp(min=1e-12)
            t_param = ((ap * ab).sum(dim=-1) / ab_len_sq).clamp(0.0, 1.0)
            closest = seg_a + t_param.unsqueeze(-1) * ab
            dists = (closest - query_point.unsqueeze(0)).norm(dim=-1)

            min_idx = dists.argmin().item()
            d = dists[min_idx].item()
            if d < best_dist:
                best_dist = d
                best_polyline = pv
                best_seg_idx = min_idx
                best_t = t_param[min_idx].item()

            poly_start = poly_end

    if best_polyline is None:
        fb = forward_hint.to(dev)
        fb = fb / fb.norm().clamp(min=1e-9)
        return query_point + fb * walk_distance, fb

    pv = best_polyline
    n_segs = len(pv) - 1
    seg_dir = pv[best_seg_idx + 1] - pv[best_seg_idx]
    if (seg_dir * forward_hint.to(dev)).sum() < 0:
        pv = pv.flip(0)
        best_seg_idx = n_segs - 1 - best_seg_idx
        best_t = 1.0 - best_t

    start_pt = pv[best_seg_idx] + best_t * (pv[best_seg_idx + 1] - pv[best_seg_idx])
    remaining = walk_distance

    rest_of_seg = (pv[best_seg_idx + 1] - start_pt).norm().item()
    if rest_of_seg >= remaining:
        direction = pv[best_seg_idx + 1] - start_pt
        direction = direction / direction.norm().clamp(min=1e-9)
        center = start_pt + direction * remaining
        return center, direction

    remaining -= rest_of_seg
    cur_seg = best_seg_idx + 1

    while cur_seg < n_segs and remaining > 0:
        s, e = pv[cur_seg], pv[cur_seg + 1]
        seg_len = (e - s).norm().item()
        if seg_len >= remaining:
            d = (e - s) / max(seg_len, 1e-9)
            center = s + d * remaining
            return center, d
        remaining -= seg_len
        cur_seg += 1

    last_tang = pv[-1] - pv[-2]
    last_tang = last_tang / last_tang.norm().clamp(min=1e-9)
    center = pv[-1] + last_tang * remaining
    return center, last_tang


def _heading_from_trajectory(translations: Tensor, min_step: float = 0.01) -> Tensor:
    """Compute yaw-only quaternions from position tangents (xyzw format).

    Uses forward differences.  When consecutive positions are closer than
    *min_step* metres (stationary ego), the heading from the last valid
    tangent is carried forward to avoid noise.

    Args:
        translations: ``[N, 3]`` positions.
        min_step: Minimum displacement (metres) to accept a tangent as valid.

    Returns:
        ``[N, 4]`` quaternions in ``(x, y, z, w)`` order.
    """
    n = len(translations)
    if n == 0:
        return torch.zeros(0, 4, device=translations.device, dtype=translations.dtype)

    tangents = torch.zeros(n, 3, device=translations.device, dtype=translations.dtype)
    if n >= 2:
        tangents[:-1] = translations[1:] - translations[:-1]
        tangents[-1] = tangents[-2]
    else:
        tangents[0] = torch.tensor([1.0, 0.0, 0.0], device=translations.device)

    norms = tangents[:, :2].norm(dim=-1)
    valid = norms > min_step

    # Carry forward the last valid tangent through stationary sections.
    # Find first valid tangent as initial seed; fall back to +x.
    last_valid_yaw = torch.tensor(0.0, device=translations.device, dtype=translations.dtype)
    yaw = torch.zeros(n, device=translations.device, dtype=translations.dtype)
    for i in range(n):
        if valid[i]:
            last_valid_yaw = torch.atan2(tangents[i, 1], tangents[i, 0])
        yaw[i] = last_valid_yaw

    half_yaw = yaw * 0.5
    quats = torch.zeros(n, 4, device=translations.device, dtype=translations.dtype)
    quats[:, 2] = torch.sin(half_yaw)
    quats[:, 3] = torch.cos(half_yaw)
    return quats


def _slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between quaternions (xyzw format).

    Args:
        q0: ``[4]`` start quaternion.
        q1: ``[4]`` end quaternion.
        t: ``[N]`` interpolation parameter in ``[0, 1]``.

    Returns:
        ``[N, 4]`` interpolated quaternions.
    """
    dot = (q0 * q1).sum().clamp(-1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0.unsqueeze(0) + t.unsqueeze(1) * (q1 - q0).unsqueeze(0)
        return result / result.norm(dim=-1, keepdim=True)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return s0.unsqueeze(1) * q0.unsqueeze(0) + s1.unsqueeze(1) * q1.unsqueeze(0)


def _concat_ego_tracks(*tracks: EgoTrackData) -> EgoTrackData:
    """Concatenate multiple :class:`EgoTrackData` along the time axis."""
    ts_parts = [t.timestamps for t in tracks if len(t.timestamps) > 0]
    tq_parts = [t.poses_tquat for t in tracks if len(t.poses_tquat) > 0]
    if not ts_parts:
        return EgoTrackData(
            timestamps=torch.zeros(0, dtype=torch.int64),
            poses_tquat=torch.zeros(0, 7, dtype=torch.float32),
        )
    return EgoTrackData(
        timestamps=torch.cat(ts_parts),
        poses_tquat=torch.cat(tq_parts),
    )


# ---------------------------------------------------------------------------
# Ego trajectory helpers
# ---------------------------------------------------------------------------


def _extrapolate_ego_track(
    ego_track: EgoTrackData,
    distance_m: float,
    forward_override: Optional[Tensor] = None,
) -> EgoTrackData:
    """Linearly extrapolate the ego trajectory forward by *distance_m* metres.

    The extension uses constant velocity (derived from the overall trajectory)
    and a fixed forward direction.

    Args:
        ego_track: Source ego track.
        distance_m: How far to extrapolate (metres).
        forward_override: If provided, use this unit vector as the
            extrapolation direction instead of deriving it from the last
            quaternion.  Useful for following the road direction.

    Returns:
        A **new** :class:`EgoTrackData` with the extrapolated poses appended.
    """
    n_poses = len(ego_track.timestamps)
    if distance_m <= 0 or n_poses < 2:
        return ego_track

    device = ego_track.timestamps.device

    # Speed from overall trajectory (robust to ego stopping at the end)
    total_dist = (ego_track.translations[1:] - ego_track.translations[:-1]).norm(dim=-1).sum().item()
    total_dt_s = (ego_track.timestamps[-1] - ego_track.timestamps[0]).float().item() * 1e-6
    speed = max(total_dist / max(total_dt_s, 1e-9), 1.0)

    # Average timestep from last few poses
    n_avg = min(10, n_poses)
    avg_dt_us = (ego_track.timestamps[-1] - ego_track.timestamps[-n_avg]).float().item() / (n_avg - 1)
    dt_us = max(1, int(round(avg_dt_us)))

    if forward_override is not None:
        forward_unit = forward_override.to(device)
        forward_unit = forward_unit / forward_unit.norm().clamp(min=1e-9)
    else:
        forward = _forward_from_quat(ego_track.quaternions[-1])
        forward_unit = forward / forward.norm().clamp(min=1e-9)

    # Number of extra poses -- evenly divide distance_m, capped to stay reasonable
    step_dist = speed * dt_us * 1e-6
    if step_dist < 1e-6:
        n_extra = 1
        step_dist = distance_m
    else:
        n_extra = max(1, round(distance_m / step_dist))
        n_extra = min(n_extra, max(n_poses, 500))
        step_dist = distance_m / n_extra

    last_ts = ego_track.timestamps[-1].item()
    last_pos = ego_track.translations[-1]
    last_quat = ego_track.quaternions[-1]

    indices = torch.arange(1, n_extra + 1, device=device, dtype=torch.float32)
    extra_pos = last_pos.unsqueeze(0) + forward_unit.unsqueeze(0) * (indices.unsqueeze(1) * step_dist)
    extra_quat = last_quat.unsqueeze(0).expand(n_extra, -1)
    extra_tquat = torch.cat([extra_pos, extra_quat], dim=-1)
    extra_ts = (last_ts + indices.long() * dt_us).to(torch.int64)

    return EgoTrackData(
        timestamps=torch.cat([ego_track.timestamps, extra_ts]),
        poses_tquat=torch.cat([ego_track.poses_tquat, extra_tquat]),
    )


def _reflect_ego_track(
    ego_track: EgoTrackData,
    center: Tensor,
    normal: Tensor,
    t_pivot_us: int,
) -> EgoTrackData:
    """Time-reverse an ego track and reflect about a mirror plane.

    Both positions and quaternions are reflected so that the ego continues
    forward with correct heading after the combined time-reversal + reflection.

    Timestamps are remapped as ``t' = 2 * t_pivot - t_original`` so they
    continue forward from *t_pivot_us*.

    Returns:
        A **new** :class:`EgoTrackData`.
    """
    reversed_tquat = ego_track.poses_tquat.flip(0)
    reversed_ts = ego_track.timestamps.flip(0)

    reflected_pos = _reflect_positions(reversed_tquat[:, :3], center, normal)
    reflected_quat = _reflect_quaternions(reversed_tquat[:, 3:], normal)
    reflected_tquat = torch.cat([reflected_pos, reflected_quat], dim=-1)

    new_ts = 2 * t_pivot_us - reversed_ts

    return EgoTrackData(timestamps=new_ts, poses_tquat=reflected_tquat)


# ---------------------------------------------------------------------------
# Pool reflection helpers
# ---------------------------------------------------------------------------


def _reflect_polyline_pool(
    pool: TimestampedPolylinePool,
    center: Tensor,
    normal: Tensor,
) -> TimestampedPolylinePool:
    """Return a reflected copy of a polyline pool."""
    return TimestampedPolylinePool(
        timestamps_us=pool.timestamps_us.clone(),
        timestamped_varrays_prefix_sum=pool.timestamped_varrays_prefix_sum.clone(),
        varrays_prefix_sum=pool.varrays_prefix_sum.clone(),
        vertices=_reflect_positions(pool.vertices, center, normal),
        prim_type_id=pool.prim_type_id,
    )


def _reflect_polygon_pool(
    pool: TimestampedPolygonPool,
    center: Tensor,
    normal: Tensor,
) -> TimestampedPolygonPool:
    """Return a reflected copy of a polygon pool.

    Reflection is improper (det = -1) so triangle winding flips.  Swap
    indices 1 and 2 in every triangle to restore correct winding.
    """
    return TimestampedPolygonPool(
        timestamps_us=pool.timestamps_us.clone(),
        timestamped_varrays_prefix_sum=pool.timestamped_varrays_prefix_sum.clone(),
        varrays_prefix_sum=pool.varrays_prefix_sum.clone(),
        triangle_prefix_sum=pool.triangle_prefix_sum.clone(),
        vertices=_reflect_positions(pool.vertices, center, normal),
        triangles=pool.triangles[:, [0, 2, 1]].clone(),
        prim_type_id=pool.prim_type_id,
    )


def _reflect_cube_pool(
    pool: CubePool,
    center: Tensor,
    normal: Tensor,
    t_pivot_us: int,
    time_reverse: bool = True,
) -> CubePool:
    """Return a reflected copy of a cube pool.

    Args:
        pool: The cube pool to reflect.
        center: ``[3]`` point on the mirror plane.
        normal: ``[3]`` unit normal of the mirror plane.
        t_pivot_us: Pivot timestamp for remapping (``t' = 2*t_pivot - t``).
        time_reverse: If True, reverse the pose order within each track and
            remap timestamps (for dynamic obstacles). If False, only reflect
            positions/quaternions and leave timestamps unchanged (for static
            cubes like traffic lights / signs).
    """
    new_translations = pool.translations.clone()
    new_quaternions = pool.quaternions.clone()
    new_track_ts = pool.track_timestamps_us.clone()

    if time_reverse:
        n_cubes = len(pool.cube_ts_prefix_sum)
        starts = torch.zeros(n_cubes, dtype=torch.int32, device=pool.cube_ts_prefix_sum.device)
        if n_cubes > 1:
            starts[1:] = pool.cube_ts_prefix_sum[:-1]

        for i in range(n_cubes):
            s, e = starts[i].item(), pool.cube_ts_prefix_sum[i].item()
            new_translations[s:e] = pool.translations[s:e].flip(0)
            new_quaternions[s:e] = pool.quaternions[s:e].flip(0)
            new_track_ts[s:e] = pool.track_timestamps_us[s:e].flip(0)

    new_translations = _reflect_positions(new_translations, center, normal)
    new_quaternions = _reflect_quaternions(new_quaternions, normal)

    if time_reverse:
        new_track_ts = 2 * t_pivot_us - new_track_ts
        new_global_ts = (2 * t_pivot_us - pool.timestamps_us).flip(0)
    else:
        new_global_ts = pool.timestamps_us.clone()

    return CubePool(
        timestamps_us=new_global_ts,
        cube_ts_prefix_sum=pool.cube_ts_prefix_sum.clone(),
        track_timestamps_us=new_track_ts,
        translations=new_translations,
        quaternions=new_quaternions,
        scales=pool.scales.clone(),
        colors=pool.colors.clone(),
        prim_type_id=pool.prim_type_id,
        render_flags=pool.render_flags,
    )


# ---------------------------------------------------------------------------
# Spatial clipping helpers
# ---------------------------------------------------------------------------

MirrorPlane = Tuple[Tensor, Tensor]  # (center [3], normal [3])


def _signed_distance(vertices: Tensor, plane_point: Tensor, plane_normal: Tensor) -> Tensor:
    """Signed distance of each vertex from a plane.  Positive = same side as normal."""
    return ((vertices - plane_point) * plane_normal).sum(dim=-1)


def _clip_polyline_pool(
    pool: TimestampedPolylinePool,
    plane_point: Tensor,
    plane_normal: Tensor,
    keep_positive: bool,
) -> Optional[TimestampedPolylinePool]:
    """Clip a polyline pool at a plane, keeping the specified side.

    Polylines that cross the plane are split at the intersection.  Polylines
    fully on the discard side are dropped.  Returns ``None`` if nothing
    survives.
    """
    verts = pool.vertices
    vps = pool.varrays_prefix_sum
    n_polylines = len(vps)

    sd = _signed_distance(verts, plane_point, plane_normal)
    keep_mask = sd >= 0 if keep_positive else sd <= 0

    new_verts_list: List[Tensor] = []
    new_vps_list: List[int] = []
    vert_count = 0

    poly_start = 0
    for j in range(n_polylines):
        poly_end = vps[j].item()
        n_v = poly_end - poly_start
        if n_v < 2:
            poly_start = poly_end
            continue

        pv = verts[poly_start:poly_end]
        pk = keep_mask[poly_start:poly_end]
        psd = sd[poly_start:poly_end]

        seg_verts: List[Tensor] = []
        for vi in range(n_v):
            on_keep = pk[vi].item()
            if vi > 0:
                prev_on_keep = pk[vi - 1].item()
                if on_keep != prev_on_keep:
                    d0, d1 = psd[vi - 1].item(), psd[vi].item()
                    t = d0 / (d0 - d1)
                    interp = pv[vi - 1] + t * (pv[vi] - pv[vi - 1])
                    if on_keep:
                        seg_verts = [interp]
                    else:
                        seg_verts.append(interp)
                        if len(seg_verts) >= 2:
                            chunk = torch.stack(seg_verts)
                            new_verts_list.append(chunk)
                            vert_count += len(chunk)
                            new_vps_list.append(vert_count)
                        seg_verts = []
            if on_keep:
                seg_verts.append(pv[vi])

        if len(seg_verts) >= 2:
            chunk = torch.stack(seg_verts)
            new_verts_list.append(chunk)
            vert_count += len(chunk)
            new_vps_list.append(vert_count)

        poly_start = poly_end

    if not new_verts_list:
        return None

    new_vertices = torch.cat(new_verts_list)
    new_varrays_ps = torch.tensor(new_vps_list, dtype=torch.int32, device=verts.device)
    n_new_polylines = len(new_vps_list)

    # Rebuild timestamped prefix sums: assign all output polylines to the
    # first timestamp (static map elements typically have one timestamp).
    new_tvps = torch.tensor([n_new_polylines], dtype=torch.int32, device=verts.device)
    ts = pool.timestamps_us[:1].clone()

    return TimestampedPolylinePool(
        timestamps_us=ts,
        timestamped_varrays_prefix_sum=new_tvps,
        varrays_prefix_sum=new_varrays_ps,
        vertices=new_vertices,
        prim_type_id=pool.prim_type_id,
    )


def _clip_polygon_pool(
    pool: TimestampedPolygonPool,
    plane_point: Tensor,
    plane_normal: Tensor,
    keep_positive: bool,
) -> Optional[TimestampedPolygonPool]:
    """Drop entire polygons whose centroid is on the discard side of the plane.

    Returns ``None`` if nothing survives.
    """
    verts = pool.vertices
    vps = pool.varrays_prefix_sum
    tps = pool.triangle_prefix_sum
    n_polys = len(vps)

    keep_indices: List[int] = []
    poly_start_v = 0
    for j in range(n_polys):
        poly_end_v = vps[j].item()
        centroid = verts[poly_start_v:poly_end_v].mean(dim=0)
        d = _signed_distance(centroid.unsqueeze(0), plane_point, plane_normal).item()
        if (d >= 0) == keep_positive:
            keep_indices.append(j)
        poly_start_v = poly_end_v

    if not keep_indices:
        return None
    if len(keep_indices) == n_polys:
        return pool  # no change

    new_verts_list: List[Tensor] = []
    new_tris_list: List[Tensor] = []
    new_vps: List[int] = []
    new_tps: List[int] = []
    vert_offset = 0
    tri_offset = 0

    for j in keep_indices:
        vs = vps[j - 1].item() if j > 0 else 0
        ve = vps[j].item()
        ts_start = tps[j - 1].item() if j > 0 else 0
        ts_end = tps[j].item()

        poly_verts = verts[vs:ve]
        poly_tris = pool.triangles[ts_start:ts_end]

        new_verts_list.append(poly_verts)
        new_tris_list.append(poly_tris)

        vert_offset += ve - vs
        tri_offset += ts_end - ts_start
        new_vps.append(vert_offset)
        new_tps.append(tri_offset)

    new_vertices = torch.cat(new_verts_list)
    new_triangles = torch.cat(new_tris_list)
    n_kept = len(keep_indices)

    new_tvps = torch.tensor([n_kept], dtype=torch.int32, device=verts.device)
    ts = pool.timestamps_us[:1].clone()

    return TimestampedPolygonPool(
        timestamps_us=ts,
        timestamped_varrays_prefix_sum=new_tvps,
        varrays_prefix_sum=torch.tensor(new_vps, dtype=torch.int32, device=verts.device),
        triangle_prefix_sum=torch.tensor(new_tps, dtype=torch.int32, device=verts.device),
        vertices=new_vertices,
        triangles=new_triangles,
        prim_type_id=pool.prim_type_id,
    )


def _clip_cube_pool(
    pool: CubePool,
    plane_point: Tensor,
    plane_normal: Tensor,
    keep_positive: bool,
) -> Optional[CubePool]:
    """Drop cubes whose mean track position is on the discard side.

    Returns ``None`` if nothing survives.
    """
    n_cubes = len(pool.cube_ts_prefix_sum)
    starts = torch.zeros(n_cubes, dtype=torch.int32, device=pool.cube_ts_prefix_sum.device)
    if n_cubes > 1:
        starts[1:] = pool.cube_ts_prefix_sum[:-1]

    keep_indices: List[int] = []
    for i in range(n_cubes):
        s, e = starts[i].item(), pool.cube_ts_prefix_sum[i].item()
        mean_pos = pool.translations[s:e].mean(dim=0)
        d = _signed_distance(mean_pos.unsqueeze(0), plane_point, plane_normal).item()
        if (d >= 0) == keep_positive:
            keep_indices.append(i)

    if not keep_indices:
        return None
    if len(keep_indices) == n_cubes:
        return pool

    new_trans_list: List[Tensor] = []
    new_quat_list: List[Tensor] = []
    new_track_ts_list: List[Tensor] = []
    new_cube_ts_ps: List[int] = []
    track_offset = 0

    for i in keep_indices:
        s, e = starts[i].item(), pool.cube_ts_prefix_sum[i].item()
        new_trans_list.append(pool.translations[s:e])
        new_quat_list.append(pool.quaternions[s:e])
        new_track_ts_list.append(pool.track_timestamps_us[s:e])
        track_offset += e - s
        new_cube_ts_ps.append(track_offset)

    dev = pool.cube_ts_prefix_sum.device
    ki = torch.tensor(keep_indices, dtype=torch.long, device=dev)

    return CubePool(
        timestamps_us=pool.timestamps_us.clone(),
        cube_ts_prefix_sum=torch.tensor(new_cube_ts_ps, dtype=torch.int32, device=dev),
        track_timestamps_us=torch.cat(new_track_ts_list),
        translations=torch.cat(new_trans_list),
        quaternions=torch.cat(new_quat_list),
        scales=pool.scales[ki],
        colors=pool.colors[ki],
        prim_type_id=pool.prim_type_id,
        render_flags=pool.render_flags,
    )


def _clip_pools(
    polyline_pools: List[TimestampedPolylinePool],
    polygon_pools: List[TimestampedPolygonPool],
    cube_pools: List[CubePool],
    entry_plane: Optional[MirrorPlane],
    exit_plane: Optional[MirrorPlane],
) -> Tuple[List[TimestampedPolylinePool], List[TimestampedPolygonPool], List[CubePool]]:
    """Clip all pools at entry (keep positive) and exit (keep negative) planes."""
    pl, pg, cb = list(polyline_pools), list(polygon_pools), list(cube_pools)

    for plane, keep_pos in [(entry_plane, True), (exit_plane, False)]:
        if plane is None:
            continue
        center, normal = plane
        pl = [r for p in pl if (r := _clip_polyline_pool(p, center, normal, keep_pos)) is not None]
        pg = [r for p in pg if (r := _clip_polygon_pool(p, center, normal, keep_pos)) is not None]
        cb = [r for p in cb if (r := _clip_cube_pool(p, center, normal, keep_pos)) is not None]

    return pl, pg, cb


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """Intermediate container for one segment's pools and ego track."""

    polyline_pools: List[TimestampedPolylinePool]
    polygon_pools: List[TimestampedPolygonPool]
    cube_pools: List[CubePool]
    ego: EgoTrackData
    extended_ego: EgoTrackData  # includes extrapolation (for ego assembly)


def mirror_augment_scene(
    scene: ClipgtGpuScene,
    n_mirrors: int = 1,
    lookahead_m: float = 50.0,
) -> ClipgtGpuScene:
    """Extend a scene by tiling two canonical tiles (original + mirror).

    Two canonical tiles are computed once:

    * **tile_fwd** -- the original scene pools.
    * **tile_bwd** -- the original reflected about a mirror plane P0 at the
      scene exit, with time-reversed ego.

    Segments alternate ``[fwd, bwd, fwd, bwd, ...]``.  The first two
    segments (k=0, k=1) use identity placement.  All subsequent segments
    are placed via a 2D rigid body transform (yaw rotation + translation)
    that aligns the source tile's entry frame to the previous segment's
    exit frame.  This prevents the rotational drift that would arise from
    compounding reflections on curved roads.

    Args:
        scene: The source :class:`ClipgtGpuScene`.
        n_mirrors: Number of augmentation iterations (total segments =
            ``n_mirrors + 1``).
        lookahead_m: Distance (metres) to extrapolate the ego trajectory
            before placing the first mirror plane.

    Returns:
        A **new** :class:`ClipgtGpuScene` with the extended ego track and
        tiled scene elements.
    """
    if n_mirrors <= 0:
        return scene

    device = scene.device
    ts_scene = scene.timestamped_scene

    has_ego_traj = any(p.prim_type_id == PRIM_EGO_TRAJECTORY for p in ts_scene.polyline_pools)
    has_ego_obs = any(p.prim_type_id == PRIM_EGO_OBSTACLE for p in (ts_scene.cube_pools or []))

    map_polyline_pools = [p for p in ts_scene.polyline_pools if p.prim_type_id != PRIM_EGO_TRAJECTORY]
    map_polygon_pools = list(ts_scene.polygon_pools)
    other_cube_pools = [p for p in (ts_scene.cube_pools or []) if p.prim_type_id != PRIM_EGO_OBSTACLE]

    # ------------------------------------------------------------------
    # Phase 0: build two canonical tiles (fwd = original, bwd = mirror)
    # ------------------------------------------------------------------
    seg_ego = EgoTrackData(
        timestamps=scene.ego_track.timestamps.clone(),
        poses_tquat=scene.ego_track.poses_tquat.clone(),
    )

    # Compute mirror plane P0 from original lane geometry
    ego_last_pos = seg_ego.translations[-1]
    n_poses = len(seg_ego.translations)
    if n_poses >= 2:
        traj_dir = seg_ego.translations[-1] - seg_ego.translations[-min(10, n_poses)]
        if traj_dir.norm().item() > 0.1:
            ego_fwd = traj_dir / traj_dir.norm()
        else:
            ego_fwd = _forward_from_quat(seg_ego.quaternions[-1])
    else:
        ego_fwd = _forward_from_quat(seg_ego.quaternions[-1])

    center_cpu, normal_cpu = _lane_based_mirror_plane(
        map_polyline_pools,
        ego_last_pos,
        ego_fwd,
        lookahead_m,
    )

    extrap_vec = center_cpu - ego_last_pos
    extrap_dist = extrap_vec.norm().item()
    extrap_dir = extrap_vec / extrap_vec.norm().clamp(min=1e-9)

    fwd_extended_ego = _extrapolate_ego_track(
        seg_ego,
        extrap_dist,
        forward_override=extrap_dir,
    )

    center_gpu = center_cpu.to(device)
    normal_gpu = normal_cpu.to(device)
    t_pivot = fwd_extended_ego.timestamps[-1].item()

    p0_plane: MirrorPlane = (center_gpu, normal_gpu)

    # tile_bwd: reflect original about P0 + time-reverse
    bwd_ego = _reflect_ego_track(fwd_extended_ego, center_cpu, normal_cpu, t_pivot)
    bwd_pl = [_reflect_polyline_pool(p, center_gpu, normal_gpu) for p in map_polyline_pools]
    bwd_pg = [_reflect_polygon_pool(p, center_gpu, normal_gpu) for p in map_polygon_pools]
    bwd_cb = [
        _reflect_cube_pool(p, center_gpu, normal_gpu, t_pivot, time_reverse=(p.prim_type_id == PRIM_OBSTACLE))
        for p in other_cube_pools
    ]

    # Reference frames (position + yaw) for each tile
    fwd_entry_pos = seg_ego.translations[0]
    fwd_entry_yaw = _yaw_from_positions(seg_ego.translations[:10], n=10)
    fwd_exit_pos = fwd_extended_ego.translations[-1]
    fwd_exit_yaw = _yaw_from_positions(fwd_extended_ego.translations, n=10)

    bwd_entry_pos = bwd_ego.translations[0]
    bwd_entry_yaw = _yaw_from_positions(bwd_ego.translations[:10], n=10)
    bwd_exit_pos = bwd_ego.translations[-1]
    bwd_exit_yaw = _yaw_from_positions(bwd_ego.translations, n=10)

    # ------------------------------------------------------------------
    # Phase 1: place segments via rigid transforms
    # ------------------------------------------------------------------
    seg0 = _Segment(
        polyline_pools=map_polyline_pools,
        polygon_pools=map_polygon_pools,
        cube_pools=other_cube_pools,
        ego=seg_ego,
        extended_ego=fwd_extended_ego,
    )
    seg1 = _Segment(
        polyline_pools=bwd_pl,
        polygon_pools=bwd_pg,
        cube_pools=bwd_cb,
        ego=bwd_ego,
        extended_ego=bwd_ego,
    )

    segments: List[_Segment] = [seg0, seg1]
    boundary_planes: List[MirrorPlane] = [p0_plane]

    # Track exit frame of each segment for chaining
    prev_exit_pos = bwd_exit_pos
    prev_exit_yaw = bwd_exit_yaw

    for k in range(2, n_mirrors + 1):
        is_fwd = k % 2 == 0

        if is_fwd:
            src_pl, src_pg, src_cb = map_polyline_pools, map_polygon_pools, other_cube_pools
            src_ego = fwd_extended_ego
            src_entry_pos, src_entry_yaw = fwd_entry_pos, fwd_entry_yaw
            src_exit_pos, src_exit_yaw = fwd_exit_pos, fwd_exit_yaw
        else:
            src_pl, src_pg, src_cb = bwd_pl, bwd_pg, bwd_cb
            src_ego = bwd_ego
            src_entry_pos, src_entry_yaw = bwd_entry_pos, bwd_entry_yaw
            src_exit_pos, src_exit_yaw = bwd_exit_pos, bwd_exit_yaw

        theta = prev_exit_yaw - src_entry_yaw
        pivot = src_entry_pos
        offset = prev_exit_pos

        # Timestamp offset: continue from previous segment's last timestamp
        prev_last_ts = segments[-1].ego.timestamps[-1].item()
        src_first_ts = src_ego.timestamps[0].item()
        time_offset_us = prev_last_ts - src_first_ts

        new_pl = [_rigid_transform_polyline_pool(p, theta, pivot, offset) for p in src_pl]
        new_pg = [_rigid_transform_polygon_pool(p, theta, pivot, offset) for p in src_pg]
        new_cb = [_rigid_transform_cube_pool(p, theta, pivot, offset, time_offset_us) for p in src_cb]
        new_ego = _rigid_transform_ego_track(src_ego, theta, pivot, offset, time_offset_us)

        # Boundary plane at junction: position = previous exit, normal = heading vector
        heading_vec = torch.tensor(
            [float(torch.cos(torch.tensor(prev_exit_yaw))), float(torch.sin(torch.tensor(prev_exit_yaw))), 0.0],
            device=device,
        )
        boundary_planes.append((prev_exit_pos.to(device), heading_vec))

        segments.append(
            _Segment(
                polyline_pools=new_pl,
                polygon_pools=new_pg,
                cube_pools=new_cb,
                ego=new_ego,
                extended_ego=new_ego,
            )
        )

        # Update exit frame for the next iteration: transform the source's
        # exit through the same rigid transform
        new_exit = _rigid_transform_positions(
            src_exit_pos.unsqueeze(0),
            theta,
            pivot,
            offset,
        ).squeeze(0)
        prev_exit_pos = new_exit
        prev_exit_yaw = src_exit_yaw + theta

    # ------------------------------------------------------------------
    # Phase 2: clip each segment at its boundary planes and assemble
    # ------------------------------------------------------------------
    all_polyline_pools: List[TimestampedPolylinePool] = []
    all_polygon_pools: List[TimestampedPolygonPool] = []
    all_cube_pools: List[CubePool] = []

    for i, seg in enumerate(segments):
        entry = boundary_planes[i - 1] if i > 0 else None
        exit_ = boundary_planes[i] if i < len(boundary_planes) else None

        pl, pg, cb = _clip_pools(
            seg.polyline_pools,
            seg.polygon_pools,
            seg.cube_pools,
            entry_plane=entry,
            exit_plane=exit_,
        )
        all_polyline_pools.extend(pl)
        all_polygon_pools.extend(pg)
        all_cube_pools.extend(cb)

    # ------------------------------------------------------------------
    # Assemble ego track from all segments
    # ------------------------------------------------------------------
    accumulated_ego = segments[0].ego

    for i in range(len(boundary_planes)):
        seg_prev = segments[i]
        seg_next = segments[i + 1]

        n_orig = len(seg_prev.ego.timestamps)
        extrap_pos = seg_prev.extended_ego.translations[n_orig:]
        extrap_ts = seg_prev.extended_ego.timestamps[n_orig:]

        next_pos = seg_next.ego.translations[1:]
        next_ts = seg_next.ego.timestamps[1:]

        aug_positions = torch.cat([extrap_pos, next_pos], dim=0)
        aug_quats = _heading_from_trajectory(aug_positions)
        n_extrap = len(extrap_ts)

        if n_extrap > 0:
            extrap_portion = EgoTrackData(
                timestamps=extrap_ts,
                poses_tquat=torch.cat([aug_positions[:n_extrap], aug_quats[:n_extrap]], dim=-1),
            )
            accumulated_ego = _concat_ego_tracks(accumulated_ego, extrap_portion)

        if len(next_ts) > 0:
            next_portion = EgoTrackData(
                timestamps=next_ts,
                poses_tquat=torch.cat([aug_positions[n_extrap:], aug_quats[n_extrap:]], dim=-1),
            )
            accumulated_ego = _concat_ego_tracks(accumulated_ego, next_portion)

    # --- Reconstruct ego-specific pools from the full accumulated track ---
    if has_ego_traj:
        ego_traj_pool = _ego_trajectory_to_pool(accumulated_ego, device)
        if ego_traj_pool is not None:
            all_polyline_pools.append(ego_traj_pool)

    if has_ego_obs:
        ego_obs_pool = _ego_obstacle_to_pool(accumulated_ego, device)
        if ego_obs_pool is not None:
            all_cube_pools.append(ego_obs_pool)

    # --- Assemble the new scene ---
    new_ts_scene = TimestampedScene(
        polyline_pools=all_polyline_pools,
        polygon_pools=all_polygon_pools,
        cube_pools=all_cube_pools,
    )

    return ClipgtGpuScene(
        timestamped_scene=new_ts_scene,
        cameras=scene.cameras,
        camera_name_to_id=dict(scene.camera_name_to_id),
        sensor_to_rig=dict(scene.sensor_to_rig),
        ego_track=accumulated_ego,
        device=device,
    )
