# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Primitive data types and packing functions for Ludus renderer."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import torch

# -----------------------------------------------------------------------------
# Primitive Type Constants (must match shader constants)
# -----------------------------------------------------------------------------

PRIM_ROAD_BOUNDARY = 0
PRIM_LANE_LINE = 1
PRIM_CROSSWALK = 2
PRIM_STATIC_OBSTACLE = 3
PRIM_EGO_TRAJECTORY = 4
PRIM_OBSTACLE = 5
PRIM_EGO_OBSTACLE = 6
PRIM_WAIT_LINE = 7
PRIM_POLE = 8
PRIM_ROAD_MARKING = 9
PRIM_LANE_BOUNDARY = 10
PRIM_TRAFFIC_LIGHT = 11
PRIM_TRAFFIC_SIGN = 12
PRIM_INTERSECTION = 13
PRIM_ROAD_ISLAND = 14
PRIM_BUFFER_ZONE = 15
PRIM_LANE_LINE_WHITE_SOLID = 16
PRIM_LANE_LINE_WHITE_DASHED = 17
PRIM_LANE_LINE_YELLOW_SOLID = 18
PRIM_LANE_LINE_YELLOW_DASHED = 19
PRIM_DOT_YELLOW = 20
PRIM_DOT_WHITE = 21
PRIM_TYPE_COUNT = 22

# Camera type IDs
CAMERA_TYPE_REGULAR = 0
CAMERA_TYPE_BEV = 1

# Cube render flags
CUBE_FLAG_WIREFRAME = 1


# -----------------------------------------------------------------------------
# Primitive Data Classes
# -----------------------------------------------------------------------------


class CapStyle(IntEnum):
    """Polyline end cap style."""

    NONE = 0
    FLAT = 1
    ROUND = 2


@dataclass
class Polyline:
    """Open thick line strip with solid color."""

    points: torch.Tensor  # [N, 3] float32 world positions
    color: torch.Tensor  # [3] float32 RGB
    width: float = 2.0  # Screen-space pixels
    cap_style: CapStyle = CapStyle.ROUND


@dataclass
class Polygon:
    """Filled polygon with solid color."""

    vertices: torch.Tensor  # [N, 3] float32 boundary vertices (CCW winding)
    color: torch.Tensor  # [3] float32 RGB


@dataclass
class Cube:
    """Oriented box defined by 9-DOF transform (translate + rotate + scale)."""

    translation: torch.Tensor  # [3] float32 world position (center)
    scale: torch.Tensor  # [3] float32 half-extents
    rotation: torch.Tensor  # [3] float32 axis-angle (Rodrigues)
    front_color: torch.Tensor  # [3] float32 RGB for front faces
    back_color: torch.Tensor  # [3] float32 RGB for back faces


@dataclass
class FThetaCamera:
    """F-theta fisheye camera intrinsics."""

    principal_point: torch.Tensor  # [2] cx, cy in pixels
    image_size: torch.Tensor  # [2] width, height in pixels
    fw_poly: torch.Tensor  # [6] forward polynomial (θ → r)
    max_ray_angle: float  # Max FOV in radians
    linear_distortion: Optional[torch.Tensor] = None  # [2, 2] affine correction
    depth_max: float = 100.0  # Max depth for z-buffer


# -----------------------------------------------------------------------------
# Timestamped Pools
# -----------------------------------------------------------------------------


@dataclass
class TimestampedPolylinePool:
    """Pool of timestamped polylines for one element type (e.g., road_boundary).

    Style (color, width) is looked up at render time via prim_type_id.
    Data is stored as flat arrays with temporal indexing via prefix sums.
    """

    timestamps_us: torch.Tensor  # [n_timestamps] int64, sorted observation times
    timestamped_varrays_prefix_sum: torch.Tensor  # [n_timestamps] int32, cumulative polyline count
    varrays_prefix_sum: torch.Tensor  # [n_varrays] int32, cumulative vertex count
    vertices: torch.Tensor  # [n_vertices, 3] float32, all vertices
    prim_type_id: int  # Index into PrimitiveStyle lookup table


@dataclass
class TimestampedPolygonPool:
    """Pool of timestamped polygons for one element type (e.g., crosswalk).

    Style (color) is looked up at render time via prim_type_id.
    Pre-triangulated with indices.
    """

    timestamps_us: torch.Tensor  # [n_timestamps] int64
    timestamped_varrays_prefix_sum: torch.Tensor  # [n_timestamps] int32
    varrays_prefix_sum: torch.Tensor  # [n_varrays] int32, cumulative vertex count
    triangle_prefix_sum: torch.Tensor  # [n_varrays] int32, cumulative triangle count
    vertices: torch.Tensor  # [n_vertices, 3] float32
    triangles: torch.Tensor  # [n_triangles, 3] int32, local indices
    prim_type_id: int  # Index into PrimitiveStyle lookup table


@dataclass
class CubePool:
    """Pool of oriented boxes (cubes) with time-varying poses.

    Used for: dynamic obstacles (cars), ego vehicle, traffic lights, etc.
    Each cube has a trajectory (track) of poses over time.
    """

    timestamps_us: torch.Tensor  # [n_global_timestamps] int64
    cube_ts_prefix_sum: torch.Tensor  # [n_cubes] int32, cumulative track length
    track_timestamps_us: torch.Tensor  # [n_track_poses] int64, timestamp per track pose
    translations: torch.Tensor  # [n_track_poses, 3] float32
    quaternions: torch.Tensor  # [n_track_poses, 4] float32, (x,y,z,w)
    scales: torch.Tensor  # [n_cubes, 3] float32
    colors: torch.Tensor  # [n_cubes, 6] float32, (front_rgb, back_rgb)
    prim_type_id: int = PRIM_OBSTACLE
    render_flags: int = 0


# Backward compatibility alias
ObstaclePool = CubePool


@dataclass
class TimestampedScene:
    """A scene with timestamped map elements and cube objects.

    Uploaded once to GPU, then rendered at any timestamp.
    """

    polyline_pools: List[TimestampedPolylinePool]
    polygon_pools: List[TimestampedPolygonPool]
    cube_pools: List[CubePool] = None

    @property
    def obstacle_pool(self) -> Optional[CubePool]:
        """Backward compat: returns first cube pool or None."""
        if self.cube_pools and len(self.cube_pools) > 0:
            return self.cube_pools[0]
        return None

    def __post_init__(self):
        if self.cube_pools is None:
            self.cube_pools = []


# -----------------------------------------------------------------------------
# Packing Functions
# -----------------------------------------------------------------------------


def _pack_cubes(cubes: List[Cube], device) -> torch.Tensor:
    """Pack list of Cubes into tensor [N, 16]."""
    if not cubes:
        return torch.empty((0, 16), dtype=torch.float32, device=device)

    packed = []
    for obs in cubes:
        row = torch.cat(
            [
                obs.translation.flatten()[:3],
                obs.scale.flatten()[:3],
                obs.rotation.flatten()[:3],
                torch.zeros(1),
                obs.front_color.flatten()[:3],
                obs.back_color.flatten()[:3],
            ]
        )
        packed.append(row)
    return torch.stack(packed).to(device=device, dtype=torch.float32)


def _pack_polylines(polylines: List[Polyline], device):
    """Pack polylines into header tensor and vertex tensor."""
    if not polylines:
        headers = torch.empty((0, 8), dtype=torch.float32, device=device)
        vertices = torch.empty((0, 4), dtype=torch.float32, device=device)
        return headers, vertices

    headers = []
    all_verts = []
    vertex_offset = 0

    for pl in polylines:
        pts = pl.points.float()
        num_verts = pts.shape[0]

        header_uint = torch.zeros(8, dtype=torch.uint32)
        header_uint[0] = vertex_offset
        header_uint[1] = num_verts
        header_uint[6] = int(pl.cap_style)
        header_uint[7] = 0

        header_float = header_uint.view(torch.float32)
        header_float[2] = pl.width
        header_float[3] = pl.color[0].item()
        header_float[4] = pl.color[1].item()
        header_float[5] = pl.color[2].item()

        headers.append(header_uint.clone().view(torch.float32))

        padded = torch.zeros((num_verts, 4), dtype=torch.float32)
        padded[:, :3] = pts[:, :3]
        all_verts.append(padded)

        vertex_offset += num_verts

    headers_tensor = torch.stack(headers).to(device=device)
    vertices_tensor = (
        torch.cat(all_verts).to(device=device) if all_verts else torch.empty((0, 4), dtype=torch.float32, device=device)
    )

    return headers_tensor, vertices_tensor


def _triangulate_polygon_ear_clipping(vertices: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Triangulate a polygon using ear clipping algorithm."""
    n = vertices.shape[0]
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    verts_2d = vertices[:, :2].cpu().numpy()

    def cross_2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def signed_area():
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += verts_2d[i][0] * verts_2d[j][1]
            area -= verts_2d[j][0] * verts_2d[i][1]
        return area / 2.0

    polygon_area = signed_area()
    is_ccw = polygon_area > 0

    def is_convex_vertex(prev_v, curr_v, next_v):
        cross = cross_2d(prev_v, curr_v, next_v)
        return cross > 0 if is_ccw else cross < 0

    def point_in_triangle(p, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def is_ear(prev_idx, curr_idx, next_idx, remaining_indices):
        a = verts_2d[prev_idx]
        b = verts_2d[curr_idx]
        c = verts_2d[next_idx]
        if not is_convex_vertex(a, b, c):
            return False
        for idx in remaining_indices:
            if idx in (prev_idx, curr_idx, next_idx):
                continue
            if point_in_triangle(verts_2d[idx], a, b, c):
                return False
        return True

    indices = list(range(n))
    triangles = []
    safety_counter = 0
    max_iterations = n * n

    while len(indices) > 3 and safety_counter < max_iterations:
        safety_counter += 1
        ear_found = False

        for i in range(len(indices)):
            prev_i = (i - 1) % len(indices)
            next_i = (i + 1) % len(indices)
            prev_idx = indices[prev_i]
            curr_idx = indices[i]
            next_idx = indices[next_i]

            if is_ear(prev_idx, curr_idx, next_idx, indices):
                if is_ccw:
                    triangles.append((prev_idx, curr_idx, next_idx))
                else:
                    triangles.append((prev_idx, next_idx, curr_idx))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            for i in range(1, len(indices) - 1):
                triangles.append((indices[0], indices[i], indices[i + 1]))
            break

    if len(indices) == 3:
        if is_ccw:
            triangles.append((indices[0], indices[1], indices[2]))
        else:
            triangles.append((indices[0], indices[2], indices[1]))

    return triangles


def _pack_polygons(polygons: List[Polygon], vertex_offset: int, device):
    """Pack polygons into header tensor, vertex tensor, and triangle tensor."""
    if not polygons:
        headers = torch.empty((0, 8), dtype=torch.float32, device=device)
        vertices = torch.empty((0, 4), dtype=torch.float32, device=device)
        triangles = torch.empty((0, 4), dtype=torch.int32, device=device)
        return headers, vertices, triangles

    headers = []
    all_verts = []
    all_tris = []
    current_vertex_offset = vertex_offset
    current_tri_offset = 0

    for pg in polygons:
        verts = pg.vertices.float()
        num_verts = verts.shape[0]
        tri_indices = _triangulate_polygon_ear_clipping(verts)
        num_tris = len(tri_indices)

        header_uint = torch.zeros(8, dtype=torch.uint32)
        header_uint[0] = current_vertex_offset
        header_uint[1] = num_verts
        header_uint[2] = current_tri_offset
        header_uint[3] = num_tris
        header_uint[7] = 0

        header_float = header_uint.view(torch.float32)
        header_float[4] = pg.color[0].item()
        header_float[5] = pg.color[1].item()
        header_float[6] = pg.color[2].item()

        headers.append(header_uint.clone().view(torch.float32))

        padded = torch.zeros((num_verts, 4), dtype=torch.float32)
        padded[:, :3] = verts[:, :3]
        all_verts.append(padded)

        for tri in tri_indices:
            all_tris.append(torch.tensor([tri[0], tri[1], tri[2], 0], dtype=torch.int32))

        current_vertex_offset += num_verts
        current_tri_offset += num_tris

    headers_tensor = torch.stack(headers).to(device=device)
    vertices_tensor = (
        torch.cat(all_verts).to(device=device) if all_verts else torch.empty((0, 4), dtype=torch.float32, device=device)
    )
    triangles_tensor = (
        torch.stack(all_tris).to(device=device) if all_tris else torch.empty((0, 3), dtype=torch.int32, device=device)
    )

    return headers_tensor, vertices_tensor, triangles_tensor


def _pack_cameras(cameras: List[FThetaCamera], device) -> torch.Tensor:
    """Pack list of FThetaCamera into tensor [P, 18]."""
    if not cameras:
        return torch.empty((0, 18), dtype=torch.float32, device=device)

    packed = []
    for cam in cameras:
        if cam.linear_distortion is None:
            ld = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        else:
            ld = cam.linear_distortion.cpu().flatten()[:4]

        alpha = cam.max_ray_angle
        poly = cam.fw_poly.cpu().flatten()[:6]
        max_val = (
            poly[0]
            + poly[1] * alpha
            + poly[2] * alpha**2
            + poly[3] * alpha**3
            + poly[4] * alpha**4
            + poly[5] * alpha**5
        )
        max_dval = (
            poly[1] + 2 * poly[2] * alpha + 3 * poly[3] * alpha**2 + 4 * poly[4] * alpha**3 + 5 * poly[5] * alpha**4
        )

        row = torch.cat(
            [
                cam.principal_point.cpu().flatten()[:2],
                cam.image_size.cpu().flatten()[:2],
                poly,
                torch.tensor([cam.max_ray_angle, max_val.item(), max_dval.item(), cam.depth_max], dtype=torch.float32),
                ld,
            ]
        )
        packed.append(row)
    return torch.stack(packed).to(device=device, dtype=torch.float32)
