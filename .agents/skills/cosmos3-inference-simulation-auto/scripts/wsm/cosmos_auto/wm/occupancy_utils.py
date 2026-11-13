# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import io
import tarfile

import numpy as np
import torch
from scipy.spatial import cKDTree

from cosmos_auto.wm.graphics_utils import (
    BELOW_LEVEL_BASE_MARGIN_M,
    BELOW_LEVEL_FAR_MARGIN_M,
    BELOW_LEVEL_LOCAL_RANGE_M,
    BELOW_LEVEL_MARGIN_SLOPE,
    BELOW_LEVEL_MAX_MARGIN_M,
    BELOW_LEVEL_TRANSITION_M,
    InstancedVoxels,
)
from cosmos_auto.wm.wds_utils import get_sample

STATIC_OCCUPANCY_COLOR_RGB = [255, 128, 0]  # Orange

# Semantic class IDs to EXCLUDE when computing column heights from pc_voxelsize_01.
# Everything not listed is kept as a static obstacle.
#   1  = road / driveable surface
#   16 = static (parked) vehicle
#   18 = dynamic (moving) vehicle
#   37 = sidewalk (ground-like, low flat surface flanking the road — not an obstacle)
#   40 = curb (near road level; road boundaries are already conveyed by lane lines)
PC_VOXELSIZE_NON_STATIC_CLASS_IDS = frozenset([1, 16, 18, 37, 40])

GROUND_LEVEL_MINIMAP_NAMES = frozenset(
    [
        "lanelines",
        "road_boundaries",
        "crosswalks",
        "wait_lines",
        "road_markings",
    ]
)


def build_ground_elevation_tree(minimap_name_to_minimap_data, processed_lanelines=None):
    """
    Build a KD-tree of ground-level 3D points from HD map polylines and lane lines.

    These points carry accurate world-space elevation (z) and are used to estimate
    the ground height beneath occupancy voxels.

    Args:
        minimap_name_to_minimap_data: dict[str, list[np.ndarray]], mapping from minimap
            name to lists of polyline/polygon arrays, each of shape [N, 3].
        processed_lanelines: list[dict] or None, output of prepare_laneline_type_geometry_data.
            Each dict has key 'pattern_segments_list' (list of [N, 2, 3] arrays).

    Returns:
        ground_tree: scipy.spatial.cKDTree built on (x, y) columns, or None if no points.
        ground_z: np.ndarray of shape [M], the z (elevation) of each point in the tree.
    """
    all_points = []

    if minimap_name_to_minimap_data is not None:
        for minimap_name, minimap_data in minimap_name_to_minimap_data.items():
            if minimap_name not in GROUND_LEVEL_MINIMAP_NAMES:
                continue
            for geom in minimap_data:
                if isinstance(geom, np.ndarray) and geom.ndim == 2 and geom.shape[1] == 3:
                    all_points.append(geom)

    if processed_lanelines is not None:
        for laneline_info in processed_lanelines:
            for segments in laneline_info["pattern_segments_list"]:
                if len(segments) == 0:
                    continue
                pts = segments.reshape(-1, 3)
                all_points.append(pts)

    if len(all_points) == 0:
        return None, None

    all_points = np.concatenate(all_points, axis=0).astype(np.float64)
    ground_tree = cKDTree(all_points[:, :2])
    ground_z = all_points[:, 2].astype(np.float32)
    return ground_tree, ground_z


def estimate_ground_elevation(voxel_xy, ground_tree, ground_z, k=4, max_radius=25.0):
    """
    Estimate ground elevation for a batch of (x, y) positions using a KD-tree of
    ground-level HD map points.

    For each query point, up to *k* nearest neighbors within *max_radius* meters are
    retrieved and their z values are averaged (inverse-distance weighted).  If no
    neighbor is found within the radius the elevation defaults to 0.

    Args:
        voxel_xy: np.ndarray, shape [N, 2], (x, y) world positions of voxels.
        ground_tree: scipy.spatial.cKDTree built on (x, y).
        ground_z: np.ndarray, shape [M], z values corresponding to tree points.
        k: int, number of nearest neighbors to average.
        max_radius: float, maximum query radius in meters.

    Returns:
        elevation: np.ndarray, shape [N], estimated ground z for each voxel.
    """
    dists, idxs = ground_tree.query(voxel_xy.astype(np.float64), k=k)

    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    valid = (idxs < len(ground_z)) & (dists <= max_radius) & np.isfinite(dists)
    safe_idxs = np.where(valid, idxs, 0)
    neighbor_z = ground_z[safe_idxs]

    weights = np.where(valid, 1.0 / np.maximum(dists, 1e-6), 0.0)
    weight_sum = weights.sum(axis=1)

    has_neighbors = weight_sum > 0.0
    elevation = np.zeros(len(voxel_xy), dtype=np.float32)
    elevation[has_neighbors] = (weights[has_neighbors] * neighbor_z[has_neighbors]).sum(axis=1) / weight_sum[
        has_neighbors
    ]
    return elevation


def extract_ground_plane_from_pose(camera_pose, camera_pose_init):
    """
    Derive a local ground plane from a single camera pose.

    The vehicle's up vector is obtained by rotating world-up ([0,0,1]) through the
    relative rotation between the initial and current camera poses.  This makes the
    result independent of the camera mounting convention.  The ground plane passes
    through the estimated ground point directly below the camera and is oriented
    according to this up vector.

    Args:
        camera_pose: np.ndarray, shape (4, 4), current camera-to-world transform.
        camera_pose_init: np.ndarray, shape (4, 4), initial camera-to-world transform
            (defines the reference ground elevation z=0).

    Returns:
        ground_plane: dict with keys 'normal' (unit up vector, shape [3]) and
            'ground_point' (point on the plane directly below the camera, shape [3]).
    """
    R_cur = camera_pose[:3, :3].astype(np.float64)
    R_init = camera_pose_init[:3, :3].astype(np.float64)
    P_cur = camera_pose[:3, 3].astype(np.float64)
    P_init = camera_pose_init[:3, 3].astype(np.float64)

    up = R_cur @ R_init.T @ np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)

    camera_height = P_init[2]
    ground_point = P_cur - camera_height * up

    return {"normal": up, "ground_point": ground_point}


def precompute_ground_planes(pose_all_frames, camera_pose_init, smooth_window=10):
    """
    Precompute smoothed ground planes for every frame in the clip.

    For each frame, the raw up vector and ground point are computed from the camera
    pose via extract_ground_plane_from_pose, then averaged over a symmetric sliding
    window of +/- *smooth_window* frames to suppress high-frequency noise from bumps
    and suspension bounce while preserving gradual terrain slope.

    Args:
        pose_all_frames: np.ndarray, shape (T, 4, 4), camera-to-world transforms.
        camera_pose_init: np.ndarray, shape (4, 4), reference pose (usually frame 0).
        smooth_window: int, half-width of the averaging window in frames.
            Total window size is 2 * smooth_window + 1.  Set to 0 to disable smoothing.

    Returns:
        ground_planes: list[dict], length T.  Each element has keys 'normal' and
            'ground_point', ready to pass to apply_ground_plane_elevation.
    """
    T = len(pose_all_frames)
    R_init = camera_pose_init[:3, :3].astype(np.float64)
    R_init_T = R_init.T
    P_init = camera_pose_init[:3, 3].astype(np.float64)
    camera_height = P_init[2]
    world_up = np.array([0.0, 0.0, 1.0])

    raw_ups = np.empty((T, 3), dtype=np.float64)
    raw_points = np.empty((T, 3), dtype=np.float64)

    for i in range(T):
        R_cur = pose_all_frames[i, :3, :3].astype(np.float64)
        P_cur = pose_all_frames[i, :3, 3].astype(np.float64)
        up = R_cur @ R_init_T @ world_up
        up /= np.linalg.norm(up)
        raw_ups[i] = up
        raw_points[i] = P_cur - camera_height * up

    if smooth_window <= 0 or T <= 1:
        return [{"normal": raw_ups[i], "ground_point": raw_points[i]} for i in range(T)]

    kernel_size = 2 * smooth_window + 1
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size

    smoothed_ups = np.empty_like(raw_ups)
    smoothed_points = np.empty_like(raw_points)
    for axis in range(3):
        smoothed_ups[:, axis] = np.convolve(np.pad(raw_ups[:, axis], smooth_window, mode="edge"), kernel, mode="valid")
        smoothed_points[:, axis] = np.convolve(
            np.pad(raw_points[:, axis], smooth_window, mode="edge"), kernel, mode="valid"
        )

    norms = np.linalg.norm(smoothed_ups, axis=1, keepdims=True)
    smoothed_ups /= norms

    return [{"normal": smoothed_ups[i], "ground_point": smoothed_points[i]} for i in range(T)]


def points_below_ground_plane(
    points,
    ground_plane,
    base_margin=BELOW_LEVEL_BASE_MARGIN_M,
    margin_slope=BELOW_LEVEL_MARGIN_SLOPE,
    max_margin=BELOW_LEVEL_MAX_MARGIN_M,
    local_range=BELOW_LEVEL_LOCAL_RANGE_M,
    far_margin=BELOW_LEVEL_FAR_MARGIN_M,
    transition=BELOW_LEVEL_TRANSITION_M,
):
    """Return a per-point mask for geometry below the pose-derived road level."""

    points = np.asarray(points, dtype=np.float64)
    normal = np.array(ground_plane["normal"], dtype=np.float64, copy=True)
    normal /= np.linalg.norm(normal)
    relative = points - np.asarray(ground_plane["ground_point"], dtype=np.float64)
    height = relative @ normal
    tangent = relative - height[:, None] * normal
    distance = np.linalg.norm(tangent, axis=1)
    margin = np.minimum(
        max_margin,
        base_margin + distance * margin_slope,
    )
    blend = np.clip((distance - local_range) / transition, 0.0, 1.0)
    threshold = margin + blend * (far_margin - margin)
    return height < -threshold


def apply_ground_plane_elevation(voxel_xy, ground_plane):
    """
    Compute ground elevation at a batch of (x, y) positions from a tilted plane.

    The plane is defined by a point and a unit normal (from
    extract_ground_plane_from_pose).  For each voxel the elevation is the z value
    where the vertical line through (x, y) intersects the plane.

    Args:
        voxel_xy: np.ndarray, shape [N, 2], (x, y) world positions.
        ground_plane: dict with 'normal' and 'ground_point'.

    Returns:
        elevation: np.ndarray, shape [N], ground z at each (x, y).
    """
    n = ground_plane["normal"]
    g = ground_plane["ground_point"]
    dx = voxel_xy[:, 0] - g[0]
    dy = voxel_xy[:, 1] - g[1]
    return (g[2] - (n[0] * dx + n[1] * dy) / n[2]).astype(np.float32)


def depth_to_color(depth_values, depth_min=0.0, depth_max=75.0):
    """
    Convert depth values to grayscale RGB colors.

    Args:
        depth_values: np.ndarray, shape [N], depth values in meters
        depth_min: float, minimum depth for color mapping
        depth_max: float, maximum depth for color mapping

    Returns:
        colors: np.ndarray, shape [N, 3], RGB colors in range [0, 1]
    """
    normalized_depth = np.clip((depth_values - depth_min) / (depth_max - depth_min), 0.0, 1.0)

    # Grayscale: closer = brighter, farther = darker
    gray = 1.0 - normalized_depth
    colors = np.stack([gray, gray, gray], axis=1)

    return colors.astype(np.float32)


def _build_ground_z_grid(
    road_pts,
    gx_min,
    gy_min,
    grid_w,
    grid_h,
    cell_resolution,
    ground_resolution=1.0,
    k=32,
    idw_power=2.0,
    query_cells=None,
):
    """
    Estimate ground elevation at every cell of the output voxel grid.

    Road voxels (class 1, already extracted) are binned into a coarse grid at
    `ground_resolution` metres.  The minimum z per coarse cell is the ground
    estimate — the road surface is always at the bottom of any accumulated voxel
    stack.  The surface is then carried onto every output-grid cell by
    inverse-distance weighting over the *k* nearest coarse road samples.

    IDW (rather than a Delaunay linear interpolation) is deliberate: on a
    switchback hill the upper and lower road segments are xy-adjacent but metres
    apart in z, and a triangulation bridges them — injecting the upper level's
    height into the strip beside the lower road and producing a spike/discontinuity
    right at the road edge.  With IDW the nearest same-level road samples dominate
    the weight, so the adjacent level wins smoothly and the cross-level spike
    collapses, while off-road cells still extrapolate the nearest road level
    instead of jumping.

    Note: no neighbourhood (e.g. median) smoothing is applied to the coarse grid.
    Such a filter corrupts multi-level boundaries — where the upper segment
    dominates a switchback neighbourhood it overwrites the isolated lower-road
    cells with the upper level, reintroducing exactly the beside-road spike this
    routine avoids.  Per-cell min-z is already robust for the clean road class.

    Args:
        road_pts:  np.ndarray [M, 3], world-space positions of class-1 voxels.
        gx_min, gy_min: int, minimum grid indices of the full output grid.
        grid_w, grid_h: int, dimensions of the output grid.
        cell_resolution: float, output grid cell size in metres (e.g. 0.2).
        ground_resolution: float, coarse grid cell size for ground fitting (e.g. 1.0).
        k: int, number of nearest road samples averaged per output cell.  Larger
            k widens the far-field averaging window, spreading the transition
            where an off-road cell sits between two road levels (e.g. the wedge
            radiating diagonally out from a switchback corner) over more distance.
            On the road the nearest sample dominates via 1/d^idw_power, so near-road
            accuracy is essentially unaffected by k.
        idw_power: float, inverse-distance weighting exponent (higher = more local).
        query_cells: np.ndarray [K, 2] int or None.  When None (default) the
            ground surface is evaluated at every output-grid cell (dense surface,
            used by dump_ground_surface_ply.py).  When provided, only those (i, j)
            cells are evaluated and the rest of the returned grid stays 0.0.  The
            IDW value at a cell is a function of that cell's xy and the fixed
            road-sample tree alone — independent of which other cells are in the
            batch — so restricting the query is bit-identical at the evaluated
            cells, just cheaper (evaluate only the cells the caller will read).

    Returns:
        ground_z_grid: np.ndarray [grid_w, grid_h] float32, ground elevation at
            each output cell (only the query_cells entries are populated when
            query_cells is given; unevaluated cells are 0.0).
    """
    if len(road_pts) == 0:
        return np.zeros((grid_w, grid_h), dtype=np.float32)
    if query_cells is not None and len(query_cells) == 0:
        return np.zeros((grid_w, grid_h), dtype=np.float32)

    # Bin road voxels to coarse grid.
    cgx = np.round(road_pts[:, 0] / ground_resolution).astype(np.int64)
    cgy = np.round(road_pts[:, 1] / ground_resolution).astype(np.int64)
    cgz = road_pts[:, 2].astype(np.float32)

    cgx_min, cgy_min = cgx.min(), cgy.min()
    cgx_rel = (cgx - cgx_min).astype(np.int64)
    cgy_rel = (cgy - cgy_min).astype(np.int64)
    cw = int(cgx_rel.max()) + 1
    ch = int(cgy_rel.max()) + 1

    coarse_min_z = np.full((cw, ch), np.inf, dtype=np.float32)
    np.minimum.at(coarse_min_z, (cgx_rel, cgy_rel), cgz)

    # Scatter ground samples: world-space (x, y, z) of each occupied coarse cell.
    ci, cj = np.where(np.isfinite(coarse_min_z))
    sample_xy = np.column_stack(
        [
            (ci + cgx_min) * ground_resolution,
            (cj + cgy_min) * ground_resolution,
        ]
    ).astype(np.float64)
    sample_z = coarse_min_z[ci, cj].astype(np.float64)

    # World-space centres of the cells to evaluate.  Default: every output grid
    # cell (dense surface).  If query_cells is given, only those cells.
    if query_cells is None:
        all_i = np.arange(grid_w, dtype=np.float64)
        all_j = np.arange(grid_h, dtype=np.float64)
        qi, qj = np.meshgrid(all_i, all_j, indexing="ij")
        qi = qi.ravel()
        qj = qj.ravel()
    else:
        qi = query_cells[:, 0].astype(np.float64)
        qj = query_cells[:, 1].astype(np.float64)
    query_xy = np.column_stack(
        [
            (qi + gx_min) * cell_resolution,
            (qj + gy_min) * cell_resolution,
        ]
    )

    # Inverse-distance weighting over the k nearest coarse road samples.
    k = int(min(k, len(sample_xy)))
    tree = cKDTree(sample_xy)
    dists, idxs = tree.query(query_xy, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / np.maximum(dists, 1e-6) ** idw_power
    neighbor_z = sample_z[idxs]
    ground_flat = (weights * neighbor_z).sum(axis=1) / weights.sum(axis=1)

    # A query sitting exactly on a sample gets that sample's z (avoids 1e-6 bias).
    exact = dists[:, 0] < 1e-6
    if exact.any():
        ground_flat[exact] = neighbor_z[exact, 0]

    if query_cells is None:
        return ground_flat.reshape(grid_w, grid_h).astype(np.float32)
    # Scatter the evaluated cells back into a full-size grid; unevaluated cells
    # stay 0.0 and are never read by the caller (which indexes obstacle cells).
    out = np.zeros((grid_w, grid_h), dtype=np.float32)
    out[query_cells[:, 0], query_cells[:, 1]] = ground_flat.astype(np.float32)
    return out


def derive_occupancy_columns_from_pc_voxelsize(
    pc_voxelsize_tar_path,
    max_height=None,
    color_rgb=None,
    cell_resolution=0.2,
    ground_below_tol=1.0,
):
    """
    Build occupancy column data entirely from raw pc_voxelsize_01 LiDAR voxels,
    without touching the pre-converted 3d_occupancy_static data.

    Algorithm:
      1. Fit a ground surface from road (class 1) voxels (see _build_ground_z_grid).
      2. Extract static obstacle voxels (drop road, parked/moving vehicles).
      3. Keep only obstacle voxels within [ground - ground_below_tol,
         ground + max_height].  The upper bound removes overpass decks; the lower
         bound removes stray LiDAR outliers below the road that would otherwise
         drag a column's base far below the surface.
      4. Per cell, round the surviving voxels' z extent (max - min) to a whole
         number of cell_resolution voxels (round half up).  An extent under half a
         voxel rounds to 0 and the cell is dropped; otherwise the column is an
         integer stack of cubes based at floor(min_z) on the global z-lattice.
         Rounding is required because pc_voxelsize_01 stores continuous point
         positions, not grid-snapped voxel centers.

    Args:
        pc_voxelsize_tar_path: str, path to the pc_voxelsize_01 <clip>.tar file.
        max_height: float or None, maximum column height above ground in meters;
            voxels above ground + max_height are discarded.
        color_rgb: list/tuple of 3 ints [0-255] or None, voxel RGB color.
        cell_resolution: float, output grid cell size in meters (default 0.2).
        ground_below_tol: float, discard obstacle voxels more than this many
            meters below the fitted ground surface (outlier rejection).

    Returns:
        dict with keys 'base_centers', 'heights', 'colors', 'cell_resolution'
        in the same format as process_static_occupancy_data(), or None if the
        tar file cannot be read or yields no valid columns.
    """
    try:
        with tarfile.open(pc_voxelsize_tar_path) as t:
            member = t.getmembers()[0]
            pth_data = torch.load(io.BytesIO(t.extractfile(member).read()), weights_only=False)
    except Exception:
        return None

    pts = pth_data["points"].numpy().astype(np.float64)  # [M, 3]
    sem = pth_data["semantics"].numpy().astype(np.int32)  # [M]
    T = pth_data["pc_to_world"].numpy()
    if not np.allclose(T, np.eye(4)):
        pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
        pts = (T @ pts_h.T).T[:, :3]

    # Establish the output grid extent from all voxels.
    gx = np.round(pts[:, 0] / cell_resolution).astype(np.int64)
    gy = np.round(pts[:, 1] / cell_resolution).astype(np.int64)
    gx_min, gy_min = gx.min(), gy.min()
    grid_w = int(gx.max() - gx_min) + 1
    grid_h = int(gy.max() - gy_min) + 1

    # Step 1: extract static obstacle voxels (drop road, parked/moving vehicles).
    # Done before the ground fit so the surface is only evaluated where it is
    # consumed — at obstacle cells (the sole reader is `cell_ground` below).
    static_mask = ~np.isin(sem, list(PC_VOXELSIZE_NON_STATIC_CLASS_IDS))
    obs_pts = pts[static_mask]
    obs_gx = gx[static_mask] - gx_min
    obs_gy = gy[static_mask] - gy_min
    obs_gz = obs_pts[:, 2].astype(np.float32)

    # Step 2: fit a smooth ground surface from road (class 1) voxels.
    # These are the LiDAR ground returns from the ego's drive — a better
    # reference than laneline geometry, and naturally filtered to the ego's
    # level by the LiDAR viewpoint.  The surface is evaluated only at the unique
    # obstacle cells: the IDW value at a cell is independent of the rest of the
    # query set, so this is bit-identical to evaluating the full grid and
    # indexing it at obstacle cells — but evaluates ~18% of the cells (~6x
    # faster, ~85% less transient memory) since 82% of the grid is never read.
    road_pts = pts[sem == 1]
    if len(obs_gx):
        # Unique obstacle cells via a 1-D integer key (grid_w*grid_h fits int64):
        # a 1-D unique is far cheaper than np.unique(axis=0)'s 2-D lexsort. Order
        # is irrelevant — each cell scatters to its own grid slot downstream.
        cell_key = obs_gx.astype(np.int64) * np.int64(grid_h) + obs_gy.astype(np.int64)
        uniq_key = np.unique(cell_key)
        obs_cells = np.stack([uniq_key // grid_h, uniq_key % grid_h], axis=1)
    else:
        obs_cells = np.empty((0, 2), dtype=np.int64)
    ground_z_grid = _build_ground_z_grid(
        road_pts,
        gx_min,
        gy_min,
        grid_w,
        grid_h,
        cell_resolution,
        query_cells=obs_cells,
    )

    # Step 3: keep obstacle voxels within [ground - ground_below_tol,
    # ground + max_height]. The upper bound removes overpass decks; the lower
    # bound rejects stray voxels below the road that would otherwise drag a
    # column's base far below the surface.
    cell_ground = ground_z_grid[obs_gx, obs_gy]
    keep = obs_gz >= cell_ground - ground_below_tol
    if max_height is not None:
        keep &= obs_gz <= cell_ground + max_height
    obs_gx = obs_gx[keep]
    obs_gy = obs_gy[keep]
    obs_gz = obs_gz[keep]

    # Step 4: per output cell, compute base (min z) and top (max z) of remaining
    # obstacle voxels.
    min_z_grid = np.full((grid_w, grid_h), np.inf, dtype=np.float32)
    max_z_grid = np.full((grid_w, grid_h), -np.inf, dtype=np.float32)
    np.minimum.at(min_z_grid, (obs_gx, obs_gy), obs_gz)
    np.maximum.at(max_z_grid, (obs_gx, obs_gy), obs_gz)

    occupied = np.isfinite(min_z_grid) & np.isfinite(max_z_grid)
    occ_gx, occ_gy = np.where(occupied)
    raw_base = min_z_grid[occ_gx, occ_gy]
    raw_top = max_z_grid[occ_gx, occ_gy]

    # Round the vertical extent to a whole number of voxels (round half up): an
    # extent under half a voxel rounds to 0 and the cell is dropped (no column);
    # >= half a voxel rounds up to at least one cube. The pc_voxelsize_01 cloud
    # stores continuous point positions (thinned to ~1 point per cell, not snapped
    # to voxel centers), so this rounding is what imposes the voxel structure.
    n_voxels = np.floor((raw_top - raw_base) / cell_resolution + 0.5).astype(np.int64)
    valid = n_voxels >= 1
    occ_gx = occ_gx[valid]
    occ_gy = occ_gy[valid]
    raw_base = raw_base[valid]
    n_voxels = n_voxels[valid]

    if len(raw_base) == 0:
        return None

    # Snap base down to the voxel z-lattice and stack n_voxels cubes on top, so
    # every column is an integer stack aligned to a global grid.
    base_z = (np.floor(raw_base / cell_resolution) * cell_resolution).astype(np.float32)
    heights = (n_voxels * cell_resolution).astype(np.float32)

    world_x = ((occ_gx + gx_min) * cell_resolution).astype(np.float32)
    world_y = ((occ_gy + gy_min) * cell_resolution).astype(np.float32)
    base_centers = np.stack([world_x, world_y, base_z + heights / 2.0], axis=1)

    rgb = color_rgb if color_rgb is not None else STATIC_OCCUPANCY_COLOR_RGB
    static_color = np.array(rgb, dtype=np.float32) / 255.0
    colors = np.tile(static_color, (len(heights), 1)).astype(np.float32)

    return {
        "base_centers": base_centers,
        "heights": heights,
        "colors": colors,
        "cell_resolution": cell_resolution,
    }


def expand_columns_to_unit_cubes(processed_occupancy):
    """Split each occupancy column into its stacked unit (cell_resolution) cubes.

    A column of height n*cell becomes n separate cube instances of height cell
    stacked contiguously.  The faces stay flush (the solid column looks the same),
    but the edge pass now draws horizontal cube-boundary lines at every voxel
    level — giving explicit height/depth cues rather than one tall unbroken box.

    Args:
        processed_occupancy: dict with 'base_centers' [N,3], 'heights' [N],
            'colors' [N,3], 'cell_resolution'.  Returned unchanged if None/empty.

    Returns:
        A new processed_occupancy dict with one entry per unit cube.
    """
    if processed_occupancy is None:
        return processed_occupancy
    centers = processed_occupancy["base_centers"]
    heights = processed_occupancy["heights"]
    colors = processed_occupancy["colors"]
    cell = float(processed_occupancy["cell_resolution"])
    if len(centers) == 0:
        return processed_occupancy

    n = np.maximum(np.round(heights / cell).astype(np.int64), 1)
    total = int(n.sum())
    rep = np.repeat(np.arange(len(n)), n)  # source column per output cube
    k = np.arange(total) - np.repeat(np.cumsum(n) - n, n)  # 0..n_i-1 within column

    bottom = centers[rep, 2] - heights[rep] / 2.0
    cz = bottom + (k + 0.5) * cell
    new_centers = np.column_stack([centers[rep, 0], centers[rep, 1], cz]).astype(np.float32)
    new_heights = np.full(total, cell, dtype=np.float32)
    new_colors = colors[rep].astype(np.float32)

    return {
        "base_centers": new_centers,
        "heights": new_heights,
        "colors": new_colors,
        "cell_resolution": cell,
    }


def _cuboid_vertices_from_labels(label_list):
    """Collect (8, 3) world cuboid vertices from a sign/traffic-light label list.

    Both traffic_signs.json and traffic_lights.json store each element as
    labelData.shape3d.cuboid3d.vertices.  Returns an [M, 8, 3] array (empty if
    the list is absent or is the RDS-HQ single-empty-label "none" sentinel).
    """
    if label_list is None or (len(label_list) == 1 and label_list[0].get("labelFamily") == "EMPTY"):
        return np.empty((0, 8, 3), dtype=np.float64)
    cuboids = []
    for label in label_list:
        try:
            v = label["labelData"]["shape3d"]["cuboid3d"]["vertices"]
        except (KeyError, TypeError):
            continue
        v = np.asarray(v, dtype=np.float64)
        if v.shape == (8, 3):
            cuboids.append(v)
    if not cuboids:
        return np.empty((0, 8, 3), dtype=np.float64)
    return np.stack(cuboids, axis=0)


def cull_voxels_intersecting_world_model(
    processed_occupancy,
    sign_label_list=None,
    tl_position_list=None,
    pole_polylines=None,
    margin=0.2,
):
    """Delete voxel columns whose 3D extent overlaps priority world-model geometry.

    WMv4.0-and-earlier elements take visual priority over WMv4.1 occupancy voxels:
    where a voxel column would intersect one of these elements it is removed, so
    the element renders unobscured while realistic occlusion is preserved
    everywhere else.  This handles the STATIC elements (signs, traffic lights,
    poles), which are fixed in world space and can be culled once per clip.
    Dynamic bounding boxes move per frame and are culled separately in the render
    loop.

    Overlap is a 3D axis-aligned test: a column (cell_resolution footprint, z-span
    from base to base+height) is deleted when its inflated AABB overlaps a sign/TL
    cuboid AABB, or when its footprint comes within `margin` of a pole segment in
    xy and their z-spans overlap.  The z check matters — a short voxel directly
    under a high sign must survive.

    Args:
        processed_occupancy: dict with 'base_centers' [N,3], 'heights' [N],
            'colors' [N,3], 'cell_resolution'.  Returned unchanged if None/empty.
        sign_label_list: traffic-sign label list (from prepare_traffic_sign_data_v4).
        tl_position_list: traffic-light position list (from prepare_traffic_light_status_data).
        pole_polylines: list of [P,3] world polylines (minimap_data["poles"]).
        margin: float, extra clearance in metres applied around every element.

    Returns:
        A new processed_occupancy dict with intersecting columns removed (or the
        original object if nothing was culled / no elements were supplied).
    """
    if processed_occupancy is None:
        return processed_occupancy
    centers = processed_occupancy["base_centers"]
    heights = processed_occupancy["heights"]
    n = len(centers)
    if n == 0:
        return processed_occupancy

    half = float(processed_occupancy["cell_resolution"]) / 2.0
    # Column AABBs (inflated later per-element by margin).
    cxmin = centers[:, 0] - half
    cxmax = centers[:, 0] + half
    cymin = centers[:, 1] - half
    cymax = centers[:, 1] + half
    czmin = centers[:, 2] - heights / 2.0
    czmax = centers[:, 2] + heights / 2.0

    delete = np.zeros(n, dtype=bool)

    # --- signs + traffic lights: AABB-vs-AABB overlap ---
    cuboids = np.concatenate(
        [
            _cuboid_vertices_from_labels(sign_label_list),
            _cuboid_vertices_from_labels(tl_position_list),
        ],
        axis=0,
    )
    if len(cuboids) > 0:
        emin = cuboids.min(axis=1) - margin  # [M, 3]
        emax = cuboids.max(axis=1) + margin
        # Broadcast column AABBs [N,1] against element AABBs [1,M].
        ov = (
            (cxmin[:, None] <= emax[None, :, 0])
            & (cxmax[:, None] >= emin[None, :, 0])
            & (cymin[:, None] <= emax[None, :, 1])
            & (cymax[:, None] >= emin[None, :, 1])
            & (czmin[:, None] <= emax[None, :, 2])
            & (czmax[:, None] >= emin[None, :, 2])
        )
        delete |= ov.any(axis=1)

    # --- poles: footprint within `margin` of a segment in xy, z-spans overlap ---
    if pole_polylines is not None:
        cxy = centers[:, :2].astype(np.float64)
        radius = margin + half
        for poly in pole_polylines:
            poly = np.asarray(poly, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 2:
                continue
            for a, b in zip(poly[:-1], poly[1:]):
                ab = b[:2] - a[:2]
                seg_len2 = float(ab @ ab)
                if seg_len2 < 1e-12:
                    d2 = ((cxy - a[:2]) ** 2).sum(axis=1)
                else:
                    t = np.clip(((cxy - a[:2]) @ ab) / seg_len2, 0.0, 1.0)
                    proj = a[:2] + t[:, None] * ab
                    d2 = ((cxy - proj) ** 2).sum(axis=1)
                seg_zmin = min(a[2], b[2]) - margin
                seg_zmax = max(a[2], b[2]) + margin
                near = (d2 <= radius * radius) & (czmin <= seg_zmax) & (czmax >= seg_zmin)
                delete |= near

    if not delete.any():
        return processed_occupancy

    keep = ~delete
    return {
        "base_centers": centers[keep],
        "heights": heights[keep],
        "colors": processed_occupancy["colors"][keep],
        "cell_resolution": processed_occupancy["cell_resolution"],
    }


def prepare_static_occupancy_data(static_occupancy_wds_file):
    """
    Load static occupancy data from tar file containing accumulated occupancy.

    Args:
        static_occupancy_wds_file: str, path to the webdataset file containing static occupancy data

    Returns:
        static_occupancy_data: dict with keys 'x', 'y', 'z', 'cell_resolution', etc.
                               Returns None if no static occupancy data exists
    """
    try:
        occupancy_sample = get_sample(static_occupancy_wds_file)

        static_occ_key = None
        for key in occupancy_sample.keys():
            if "static_occupancy.json" in key:
                static_occ_key = key
                break

        if static_occ_key is None:
            return None

        return occupancy_sample[static_occ_key]
    except Exception:
        return None


def process_static_occupancy_data(
    static_occupancy_data,
    render_mode="classification",
    camera_pose=None,
    ground_elevation_tree=None,
    ground_elevation_z=None,
    max_height=None,
    base_elevation=None,
    color_rgb=None,
):
    """
    Process static occupancy data into format suitable for rendering as height map towers.

    The static occupancy is a 2.5D height map where each (x, y, z) position represents
    a tower of height z.  The tower base can be offset from z=0 to follow terrain by
    providing a KD-tree of HD map ground points (*ground_elevation_tree*), or by
    supplying explicit per-column base elevations (*base_elevation*).
    For per-frame elevation methods (e.g. tilted_plane) use the *ground_plane* parameter
    on the InstancedVoxels object via update_centers() instead.

    Args:
        static_occupancy_data: dict with keys 'x', 'y', 'z', 'cell_resolution', etc.
        render_mode: str, either 'classification' or 'depth'
        camera_pose: np.ndarray, shape (4, 4), camera pose in world coordinates (required for depth mode)
        ground_elevation_tree: scipy.spatial.cKDTree or None, KD-tree on (x, y) of ground-level
            HD map points (from build_ground_elevation_tree).
        ground_elevation_z: np.ndarray or None, z values for the tree points.
        max_height: float or None, maximum tower height in meters. Towers taller than this
            are clamped. None means no limit.
        base_elevation: np.ndarray or None, shape [N], per-column base z read directly from
            occupancy data.  When provided, towers span from base_elevation to
            base_elevation + height, bypassing any other elevation estimation.
        color_rgb: list/tuple of 3 ints [0-255] or None, RGB color for classification mode.
            Falls back to STATIC_OCCUPANCY_COLOR_RGB when None.

    Returns:
        processed_occupancy: dict with keys:
            - 'base_centers': np.ndarray, shape [N, 3], world coordinates at center of tower
            - 'heights': np.ndarray, shape [N], height of each tower (the z value)
            - 'colors': np.ndarray, shape [N, 3], RGB colors for each tower
            - 'cell_resolution': float, width/depth of each tower
            Returns None if no valid data
    """
    if static_occupancy_data is None:
        return None

    x = np.array(static_occupancy_data["x"], dtype=np.float32)
    y = np.array(static_occupancy_data["y"], dtype=np.float32)
    z = np.array(static_occupancy_data["z"], dtype=np.float32)
    cell_resolution = static_occupancy_data["cell_resolution"]

    if len(x) == 0:
        return None

    heights = z
    if max_height is not None:
        heights = np.minimum(heights, max_height)

    if base_elevation is not None:
        base_centers = np.stack([x, y, base_elevation + heights / 2.0], axis=1)
    elif ground_elevation_tree is not None and ground_elevation_z is not None:
        voxel_xy = np.stack([x, y], axis=1)
        ground_z = estimate_ground_elevation(voxel_xy, ground_elevation_tree, ground_elevation_z)
        base_centers = np.stack([x, y, ground_z + heights / 2.0], axis=1)
    else:
        base_centers = np.stack([x, y, heights / 2.0], axis=1)

    if render_mode == "depth":
        world_to_camera = np.linalg.inv(camera_pose)
        base_centers_homogeneous = np.concatenate([base_centers, np.ones((base_centers.shape[0], 1))], axis=1)
        camera_space_positions = (world_to_camera @ base_centers_homogeneous.T).T
        depth_values = camera_space_positions[:, 2]
        colors = depth_to_color(depth_values, depth_min=0.0, depth_max=75.0)
    elif render_mode == "classification":
        rgb = color_rgb if color_rgb is not None else STATIC_OCCUPANCY_COLOR_RGB
        static_color = np.array(rgb, dtype=np.float32) / 255.0
        colors = np.tile(static_color, (len(x), 1)).astype(np.float32)
    else:
        raise ValueError(f"Invalid render mode: {render_mode}")

    return {
        "base_centers": base_centers,
        "heights": heights,
        "colors": colors,
        "cell_resolution": cell_resolution,
    }


def create_instanced_voxels(
    processed_static_occupancy,
    camera_model,
    max_distance=None,
    max_lateral=None,
    draw_edges=True,
    edge_color=(0, 0, 0),
    edge_width=0.5,
):
    """
    Build a persistent GPU-instanced geometry object for static occupancy data.

    The returned InstancedVoxels object caches GPU buffers across frames.
    Call set_camera_pose() each frame before rendering.  If per-frame ground
    plane adjustment is needed, call update_centers() with adjusted centers.

    Args:
        processed_static_occupancy: dict with keys 'base_centers', 'heights', 'colors', 'cell_resolution'
        camera_model: CameraModel, ftheta or pinhole camera model
        max_distance: float or None, GPU-side forward culling distance.
        max_lateral: float or None, GPU-side lateral culling distance.
        draw_edges: bool, draw black (or edge_color) cube edges so voxels read as cubes.
        edge_color: RGB in [0-255] or [0-1] for the cube edges.
        edge_width: float, edge line width.

    Returns:
        InstancedVoxels or None if no valid data.
    """
    base_centers = processed_static_occupancy["base_centers"]
    heights = processed_static_occupancy["heights"]
    colors = processed_static_occupancy["colors"]
    cell_resolution = processed_static_occupancy["cell_resolution"]

    if len(base_centers) == 0:
        return None

    edge_color = np.asarray(edge_color, dtype=np.float32)
    if edge_color.max() > 1.0:
        edge_color = edge_color / 255.0

    return InstancedVoxels(
        centers=base_centers,
        heights=heights,
        colors=colors,
        cell_resolution=cell_resolution,
        camera_model=camera_model,
        max_distance=max_distance,
        max_lateral=max_lateral,
        line_width=edge_width,
        draw_edges=draw_edges,
        edge_color=edge_color,
    )
