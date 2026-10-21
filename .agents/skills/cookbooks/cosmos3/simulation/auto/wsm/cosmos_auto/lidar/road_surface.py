# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional drivable-surface fill between road boundaries (toolkit ``--fill_in_road``).

Port of ``utils/fill_in_road.py`` from the Cosmos-Drive-Dreams toolkit. The map is
tiled into blocks; inside each block the road boundaries are rasterised onto a
coarse BEV grid, the connected component touching the lane lines is taken as
road, a plane is fitted to the boundary + lane points, and the road cells are
up-sampled onto that plane. Differences from the toolkit: the device is a
parameter (CPU works), connected components use ``scipy.ndimage.label`` instead
of ``skimage.measure.label`` and the plane fit uses an SVD instead of
``skspatial`` (both are the same algorithms), so no extra dependencies are needed.
"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import torch
from scipy import ndimage


def _fit_plane(points: np.ndarray) -> np.ndarray:
    """Least-squares plane ``[a, b, c, d]`` with ``a x + b y + c z + d = 0`` (unit normal)."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[2]
    if normal[2] < 0:
        normal = -normal
    return np.array(
        [normal[0], normal[1], normal[2], -float(normal @ centroid)], dtype=np.float64
    )


def upsample_plane_points(
    vv: torch.Tensor,
    uu: torch.Tensor,
    plane_params: torch.Tensor,
    block_x_start: torch.Tensor,
    block_y_start: torch.Tensor,
    coarse_voxel_sizes: list[float],
    fine_voxel_sizes: list[float],
) -> torch.Tensor:
    """Subdivide each coarse road cell into fine cells and lift them onto the plane."""
    device = vv.device
    nx = max(1, int(coarse_voxel_sizes[0] / fine_voxel_sizes[0]))
    ny = max(1, int(coarse_voxel_sizes[1] / fine_voxel_sizes[1]))
    offset_u = torch.linspace(0, 1, nx + 1)[:-1] + 1 / (2 * nx)
    offset_v = torch.linspace(0, 1, ny + 1)[:-1] + 1 / (2 * ny)
    offset_u, offset_v = torch.meshgrid(offset_u, offset_v, indexing="xy")
    offset_u = offset_u.reshape(-1).to(device) - 0.5
    offset_v = offset_v.reshape(-1).to(device) - 0.5
    uu_flat = (uu.unsqueeze(1) + offset_u.unsqueeze(0)).reshape(-1)
    vv_flat = (vv.unsqueeze(1) + offset_v.unsqueeze(0)).reshape(-1)
    x = uu_flat * coarse_voxel_sizes[0] + block_x_start
    y = vv_flat * coarse_voxel_sizes[1] + block_y_start
    z = -(plane_params[0] * x + plane_params[1] * y + plane_params[3]) / plane_params[2]
    return torch.stack([x, y, z], dim=1)


def estimate_road_surface_in_grid(
    road_edge_full: torch.Tensor,
    lane_full: torch.Tensor,
    block_x_idx: int,
    block_y_idx: int,
    blocks_x_start: torch.Tensor,
    blocks_x_end: torch.Tensor,
    blocks_y_start: torch.Tensor,
    blocks_y_end: torch.Tensor,
    voxel_sizes: list[float],
    fine_voxel_sizes: list[float],
    verbose: bool = False,
) -> torch.Tensor:
    """Road-surface points for one block (see module docstring)."""
    device = road_edge_full.device
    block_x_start = blocks_x_start[block_x_idx]
    block_x_end = blocks_x_end[block_x_idx]
    block_y_start = blocks_y_start[block_y_idx]
    block_y_end = blocks_y_end[block_y_idx]
    block_size_x = block_x_end - block_x_start
    block_size_y = block_y_end - block_y_start

    def _inside(
        points: torch.Tensor, margin_x: float = 0.0, margin_y: float = 0.0
    ) -> torch.Tensor:
        return (
            (points[:, 0] >= block_x_start - margin_x)
            & (points[:, 0] <= block_x_end + margin_x)
            & (points[:, 1] >= block_y_start - margin_y)
            & (points[:, 1] <= block_y_end + margin_y)
        )

    road_edge = road_edge_full[_inside(road_edge_full)]
    lane = lane_full[_inside(lane_full)]
    road_edge_full = road_edge_full[
        _inside(road_edge_full, voxel_sizes[0], voxel_sizes[1])
    ]
    lane_full = lane_full[_inside(lane_full, voxel_sizes[0], voxel_sizes[1])]
    if road_edge.shape[0] < 3 or lane.shape[0] < 3:
        if verbose:
            print(f"block {block_x_idx}_{block_y_idx} has too few points")
        return torch.zeros((0, 3), device=device)

    bev_w = round(block_size_x.item() / voxel_sizes[0])
    bev_h = round(block_size_y.item() / voxel_sizes[1])
    bev_rasterize_map = torch.zeros((bev_h, bev_w), dtype=torch.uint8, device=device)

    def _cells(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = ((points[:, 0] - block_x_start) // voxel_sizes[0]).long()
        v = ((points[:, 1] - block_y_start) // voxel_sizes[1]).long()
        keep = (u >= 0) & (u < bev_w) & (v >= 0) & (v < bev_h)
        uv = torch.unique(torch.stack([u[keep], v[keep]], dim=1), dim=0)
        return uv[:, 0], uv[:, 1]

    road_edge_u, road_edge_v = _cells(road_edge)
    lane_u, lane_v = _cells(lane)
    if len(lane_u) < 3 or len(road_edge_u) < 3:
        if verbose:
            print(
                f"block {block_x_idx}_{block_y_idx} has too few valid points after bounds checking"
            )
        return torch.zeros((0, 3), device=device)

    bev_rasterize_map[road_edge_v, road_edge_u] = 255
    # 4-connected components of the non-boundary cells (boundary cells get label 0).
    label_result: Any = ndimage.label(bev_rasterize_map.cpu().numpy() != 255)
    labels = np.asarray(label_result[0])
    cc_image = torch.from_numpy(labels.astype(np.int64)).to(device)
    lane_cc_unique = torch.unique(cc_image[lane_v, lane_u])
    lane_cc_mask = torch.isin(cc_image, lane_cc_unique)

    # A component covering most of the block means the boundaries have gaps and the fill
    # escapes; fall back to a nearest-feature test per sub-block.
    if lane_cc_mask.float().mean() > 0.7:
        subdivide_num = 4
        # Sub-block cell bounds as plain ints (same values as the toolkit's linspace().long()).
        sub_w = [
            int(b) for b in torch.linspace(0, bev_w, subdivide_num + 1).long().tolist()
        ]
        sub_h = [
            int(b) for b in torch.linspace(0, bev_h, subdivide_num + 1).long().tolist()
        ]
        for j in range(subdivide_num):
            for i in range(subdivide_num):
                w0, w1 = sub_w[i], sub_w[i + 1]
                h0, h1 = sub_h[j], sub_h[j + 1]
                lane_in = (
                    (lane_u >= w0) & (lane_u < w1) & (lane_v >= h0) & (lane_v < h1)
                )
                edge_in = (
                    (road_edge_u >= w0)
                    & (road_edge_u < w1)
                    & (road_edge_v >= h0)
                    & (road_edge_v < h1)
                )
                sub_mask = lane_cc_mask[h0:h1, w0:w1]
                if (lane_in.sum() > 0 or edge_in.sum() > 0) and sub_mask.any():
                    sub_v = torch.arange(h0, h1, device=device)
                    sub_u = torch.arange(w0, w1, device=device)
                    grid_x, grid_y = torch.meshgrid(
                        sub_u * voxel_sizes[0] + block_x_start,
                        sub_v * voxel_sizes[1] + block_y_start,
                        indexing="xy",
                    )
                    sub_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
                    lane_dist = torch.cdist(sub_xy, lane_full[:, :2]).min(dim=1)[0]
                    edge_dist = torch.cdist(sub_xy, road_edge_full[:, :2]).min(dim=1)[0]
                    mask = (lane_dist < edge_dist).view(h1 - h0, w1 - w0)
                    lane_cc_mask[h0:h1, w0:w1] &= mask
                else:
                    lane_cc_mask[h0:h1, w0:w1] = False

    n_road_edge = min(1500, road_edge.shape[0])
    n_lane = min(1500, lane.shape[0])
    sample = torch.cat(
        [
            road_edge[torch.randperm(road_edge.shape[0], device=device)[:n_road_edge]],
            lane[torch.randperm(lane.shape[0], device=device)[:n_lane]],
        ],
        dim=0,
    )
    plane_params = torch.tensor(
        _fit_plane(sample.cpu().double().numpy()), device=device, dtype=torch.float32
    )
    vv, uu = torch.where(lane_cc_mask)
    return upsample_plane_points(
        vv,
        uu,
        plane_params,
        block_x_start,
        block_y_start,
        voxel_sizes,
        fine_voxel_sizes,
    )


def estimate_road_surface_in_world(
    road_edge: torch.Tensor,
    lane: torch.Tensor,
    block_size: list[float] | None = None,
    voxel_sizes: list[float] | None = None,
    fine_voxel_sizes: list[float] | None = None,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Estimate drivable-surface points over the whole map from boundary and lane points.

    Args:
        road_edge: ``[N, 3]`` road-boundary points (world frame).
        lane: ``[M, 3]`` lane-line points (world frame); the toolkit also prepends
            the ego positions so the driven road is always covered.
        block_size: tile size in metres (toolkit range-map defaults: ``[100, 100]``).
        voxel_sizes / fine_voxel_sizes: coarse BEV cell and fine up-sampling cell.
        device: defaults to ``road_edge.device``.

    Returns:
        ``[K, 3]`` world-frame road-surface points (float32).
    """
    block_size = block_size or [40.0, 40.0]
    voxel_sizes = voxel_sizes or [0.4, 0.4, 0.2]
    fine_voxel_sizes = fine_voxel_sizes or [0.02, 0.02, 0.2]
    device = torch.device(device) if device is not None else road_edge.device
    road_edge = torch.as_tensor(road_edge, device=device).float()
    lane = torch.as_tensor(lane, device=device).float()
    if lane.shape[0] == 0 or road_edge.shape[0] == 0:
        return torch.zeros((0, 3), device=device)

    map_x_min, map_x_max = lane[:, 0].min(), lane[:, 0].max()
    map_y_min, map_y_max = lane[:, 1].min(), lane[:, 1].max()
    block_x_num = max(1, ceil((map_x_max.item() - map_x_min.item()) / block_size[0]))
    block_y_num = max(1, ceil((map_y_max.item() - map_y_min.item()) / block_size[1]))
    blocks_x_start = (
        torch.arange(block_x_num, device=device) * block_size[0] + map_x_min
    )
    blocks_y_start = (
        torch.arange(block_y_num, device=device) * block_size[1] + map_y_min
    )
    blocks_x_end = blocks_x_start + block_size[0]
    blocks_y_end = blocks_y_start + block_size[1]
    if verbose:
        print(f"block_x_num: {block_x_num}, block_y_num: {block_y_num}")

    road_surface_points = []
    for j in range(block_y_num):
        for i in range(block_x_num):
            in_lane = (
                (lane[:, 0] >= blocks_x_start[i])
                & (lane[:, 0] < blocks_x_end[i])
                & (lane[:, 1] >= blocks_y_start[j])
                & (lane[:, 1] < blocks_y_end[j])
            )
            in_edge = (
                (road_edge[:, 0] >= blocks_x_start[i])
                & (road_edge[:, 0] < blocks_x_end[i])
                & (road_edge[:, 1] >= blocks_y_start[j])
                & (road_edge[:, 1] < blocks_y_end[j])
            )
            if in_lane.sum() == 0 or in_edge.sum() == 0:
                continue
            points = estimate_road_surface_in_grid(
                road_edge,
                lane,
                i,
                j,
                blocks_x_start,
                blocks_x_end,
                blocks_y_start,
                blocks_y_end,
                voxel_sizes,
                fine_voxel_sizes,
                verbose,
            )
            if points.shape[0] > 0:
                road_surface_points.append(points)
    if not road_surface_points:
        return torch.zeros((0, 3), device=device)
    return torch.cat(road_surface_points, dim=0).float()
