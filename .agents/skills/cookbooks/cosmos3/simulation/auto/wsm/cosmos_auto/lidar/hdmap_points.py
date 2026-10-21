# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense point sampling of RDS-HQ map geometry and object boxes.

Port of ``utils/minimap_utils_for_lidar.py`` from the Cosmos-Drive-Dreams toolkit.
Polylines and cuboid edges become thin point cylinders, polygons are filled with
a planar grid, so that the range-map projection sees surfaces rather than
infinitely thin lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cosmos_auto.wm.bbox_utils import build_cuboid_bounding_box, fix_static_objects
from cosmos_auto.wm.minimap_utils import (
    cuboid3d_to_polyline,
    get_type_from_name,
    simplify_minimap,
)
from cosmos_auto.wm.wds_utils import get_sample

DEFAULT_MINIMAP_TYPES: tuple[str, ...] = (
    "lanelines",
    "road_boundaries",
    "crosswalks",
    "road_markings",
    "wait_lines",
    "poles",
    "traffic_signs",
    "traffic_lights",
)
DEFAULT_DENSITY_M = 0.02
DEFAULT_RADIUS_M = 0.05

_EMPTY = np.zeros((0, 3), dtype=np.float64)


def sample_points_along_line(
    start: np.ndarray, end: np.ndarray, density: float = DEFAULT_DENSITY_M
) -> np.ndarray:
    """``[N, 3]`` points spaced ``density`` metres along a segment (empty for segments < 1 cm)."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-2:
        return _EMPTY
    num_points = max(2, int(np.ceil(length / density)))
    t = np.linspace(0.0, 1.0, num_points)
    return start[None, :] + t[:, None] * direction[None, :]


def sample_points_in_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
) -> np.ndarray:
    """Points on the surface of a cylinder of ``radius`` around a segment, ``density`` apart."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-2:
        return _EMPTY
    unit = direction / length
    num_axial = max(2, int(np.ceil(length / density)))
    num_radial = max(4, int(np.ceil(2 * np.pi * radius / density)))
    axial = (
        start[None, :] + np.linspace(0.0, 1.0, num_axial)[:, None] * direction[None, :]
    )
    perpendicular = (
        np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    v1 = np.cross(unit, perpendicular)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(unit, v1)
    angles = np.linspace(0.0, 2 * np.pi, num_radial, endpoint=False)
    offsets = radius * (
        np.cos(angles)[:, None] * v1[None, :] + np.sin(angles)[:, None] * v2[None, :]
    )
    return (axial[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def points_from_polylines(
    polylines: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
) -> np.ndarray:
    """Cylinder-sample every segment of every polyline.

    Chunks are converted to ``dtype`` as they are produced, so a layer with tens of
    millions of points never holds a float64 copy of itself when float32 is requested.
    """
    chunks = []
    for polyline in polylines:
        polyline = np.asarray(polyline, dtype=np.float64)
        if len(polyline) < 2:
            continue
        for i in range(len(polyline) - 1):
            chunks.append(
                sample_points_in_cylinder(
                    polyline[i], polyline[i + 1], density, radius
                ).astype(dtype, copy=False)
            )
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=dtype)


def _grid_inside_polygon(
    xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """``[len(ys), len(xs)]`` mask of grid points inside ``polygon`` by the even-odd rule.

    One scanline per grid row: a point is inside when an odd number of polygon
    edges cross the row to its left.
    """
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for row, y in enumerate(ys):
        straddles = (y0 > y) != (y1 > y)
        ax, ay, bx, by = x0[straddles], y0[straddles], x1[straddles], y1[straddles]
        crossings = np.sort(ax + (y - ay) * (bx - ax) / (by - ay))
        mask[row] = np.searchsorted(crossings, xs, side="right") % 2 == 1
    return mask


def points_from_polygons(
    hulls: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    dtype: type = np.float64,
) -> np.ndarray:
    """Sample polygon boundaries and fill the interior with a planar grid.

    The interior is filled on the best-fit plane (SVD) with grid points inside
    the projected outline. The toolkit kept points inside a Delaunay
    triangulation of the vertices, i.e. the convex hull, which also filled the
    notches of concave polygons such as some crosswalks.
    """
    chunks = []
    for hull in hulls:
        hull = np.asarray(hull, dtype=np.float64)
        if len(hull) < 3:
            continue
        for i in range(len(hull)):
            chunks.append(
                sample_points_along_line(
                    hull[i], hull[(i + 1) % len(hull)], density
                ).astype(dtype, copy=False)
            )
        centroid = hull.mean(axis=0)
        centered = hull - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        v1 = (
            np.array([1.0, 0.0, 0.0])
            if abs(normal[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        v1 = v1 - np.dot(v1, normal) * normal
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        hull_2d = np.stack([centered @ v1, centered @ v2], axis=1)
        lo, hi = hull_2d.min(axis=0), hull_2d.max(axis=0)
        xs = np.arange(lo[0], hi[0], density)
        ys = np.arange(lo[1], hi[1], density)
        if xs.size == 0 or ys.size == 0:
            continue
        xx, yy = np.meshgrid(xs, ys)
        # A degenerate (collinear) outline encloses no area: boundary samples only.
        inside = _grid_inside_polygon(xs, ys, hull_2d).ravel()
        grid = np.stack([xx.ravel()[inside], yy.ravel()[inside]], axis=1)
        chunks.append(
            (centroid + grid[:, 0:1] * v1[None, :] + grid[:, 1:2] * v2[None, :]).astype(
                dtype, copy=False
            )
        )
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=dtype)


def points_from_closed_polylines(
    hulls: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
) -> np.ndarray:
    """Cylinder-sample the closed edge loop of each hull (used for cuboid outlines)."""
    chunks = []
    for hull in hulls:
        hull = np.asarray(hull, dtype=np.float64)
        if len(hull) < 3:
            continue
        for i in range(len(hull)):
            chunks.append(
                sample_points_in_cylinder(
                    hull[i], hull[(i + 1) % len(hull)], density, radius
                ).astype(dtype, copy=False)
            )
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=dtype)


def minimap_points(
    minimap_name: str,
    elements: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
) -> np.ndarray:
    """Sample one map layer (output of :func:`simplify_minimap`) into world-frame points of ``dtype``."""
    minimap_type = get_type_from_name(minimap_name)
    if minimap_type == "polygon":
        return points_from_polygons(elements, density, dtype)
    if minimap_type == "polyline":
        return points_from_polylines(elements, density, radius, dtype)
    if minimap_type == "cuboid3d":
        return points_from_closed_polylines(elements, density, radius, dtype)
    raise ValueError(f"unsupported minimap type {minimap_type!r} for {minimap_name!r}")


def load_static_map_points(
    input_root: str | Path,
    clip_id: str,
    minimap_types: Sequence[str] = DEFAULT_MINIMAP_TYPES,
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    missing_ok: bool = False,
) -> dict[str, np.ndarray]:
    """``{layer_name: [N, 3] float32 world points}`` for every ``3d_<layer>/<clip>.tar`` map component.

    Wait-line labels flagged ``is_implicit`` (virtual stop lines) are not rendered; the
    Cosmos-Drive-Dreams toolkit draws them, this renderer keeps explicit paint only.
    Points are produced in float32, the renderer's working precision, which halves the peak
    memory of a dense layer compared with converting after concatenation.
    """
    layers: dict[str, np.ndarray] = {}
    for minimap_type in minimap_types:
        tar_path = Path(input_root) / f"3d_{minimap_type}" / f"{clip_id}.tar"
        if not tar_path.exists():
            if missing_ok:
                layers[minimap_type] = np.zeros((0, 3), dtype=np.float32)
                continue
            raise FileNotFoundError(f"map layer missing: {tar_path}")
        elements, name = simplify_minimap(str(tar_path))
        layers[name] = minimap_points(name, elements, density, radius, dtype=np.float32)
    return layers


def load_object_info(input_root: str | Path, clip_id: str) -> dict:
    """Decoded ``all_object_info/<clip>.tar`` with static-object poses/sizes averaged over the clip."""
    sample = get_sample(str(Path(input_root) / "all_object_info" / f"{clip_id}.tar"))
    return fix_static_objects(sample)


def bbox_points_for_frames(
    all_object_info: dict,
    frame_ids: Sequence[int],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
) -> list[np.ndarray]:
    """Per-frame world points on the edges of every tracked object cuboid.

    Frames without an ``all_object_info`` record yield an empty array so the
    static map and LiDAR geometry are still rendered.
    """
    result: list[np.ndarray] = []
    for frame_id in frame_ids:
        key = (
            f"{frame_id:06d}.all_object_info.json"
            if isinstance(frame_id, (int, np.integer))
            else f"{frame_id}.all_object_info.json"
        )
        objects = all_object_info.get(key) or {}
        polylines = []
        for tracking_id in sorted(objects):
            info = objects[tracking_id]
            lwh = np.asarray(info["object_lwh"], dtype=np.float64)
            corners = build_cuboid_bounding_box(
                lwh[0],
                lwh[1],
                lwh[2],
                np.asarray(info["object_to_world"], dtype=np.float64),
            )
            polylines.append(cuboid3d_to_polyline(corners))
        result.append(points_from_polylines(polylines, density, radius))
    return result
