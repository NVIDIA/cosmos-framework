# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np
from pycg import Isometry
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from cosmos_auto.wm.graphics_utils import (
    ALL_DIAGONAL_INDICES,
    BACK_DIAGONAL_INDICES,
    EDGE_INDICES,
    FRONT_DIAGONAL_INDICES,
    FthetaCurvedEdges,
    TriangleList2D,
    render_geometries,
)

# Face quads: (p00, p10, p01, p11) corner indices for bilinear interpolation
# p(u,v) = (1-u)(1-v)*p00 + u(1-v)*p10 + (1-u)v*p01 + uv*p11
_FACE_QUADS = {
    "front": [(1, 0, 5, 4)],
    "back": [(3, 2, 7, 6)],
    "front_and_back": [(1, 0, 5, 4), (3, 2, 7, 6)],
    "all": [(1, 0, 5, 4), (3, 2, 7, 6), (0, 3, 4, 7), (2, 1, 6, 5), (1, 0, 2, 3), (5, 4, 6, 7)],
}
_DARK = np.array([0.25, 0.25, 0.25])


def _build_face_fills(corners_cam, per_vertex_color, fill_face, fill_face_style, camera_model, n_steps):
    """
    Build face fill geometry by walking the curved boundary edges (same math as edge renderer)
    and fanning triangles from the face center. All faces are batched into ONE TriangleList2D
    (one VBO/draw-call per bbox) and fill boundaries exactly match the drawn edge lines.
    """
    if fill_face == "none":
        return []

    face_quads = _FACE_QUADS.get(fill_face, _FACE_QUADS["all"])
    # n_steps points per edge, exclude shared endpoint to form a clean closed loop
    t = np.linspace(0.0, 1.0, n_steps + 1)[:-1]  # (n_steps,)

    all_tri_verts = []
    all_tri_colors = []

    for i00, i10, i01, i11 in face_quads:
        p00, p10 = corners_cam[i00], corners_cam[i10]
        p01, p11 = corners_cam[i01], corners_cam[i11]

        # Boundary loop in camera space: walk 4 edges CCW (same interpolation as edge renderer)
        boundary_3d = np.concatenate(
            [
                p00 + t[:, None] * (p10 - p00),  # i00 → i10
                p10 + t[:, None] * (p11 - p10),  # i10 → i11
                p11 + t[:, None] * (p01 - p11),  # i11 → i01
                p01 + t[:, None] * (p00 - p01),  # i01 → i00
            ]
        )  # (4*n_steps, 3)

        center_3d = (p00 + p10 + p01 + p11) / 4.0
        all_pts = np.vstack([center_3d, boundary_3d])  # index 0 = center

        if np.all(all_pts[:, 2] <= 0):
            continue

        xy = camera_model.ray2pixel(all_pts)
        xyd = np.hstack([xy, all_pts[:, 2:3]]).astype("f4")

        if fill_face_style == "diagonal":
            c00 = c10 = c01 = c11 = _DARK
        else:
            c00, c10 = per_vertex_color[i00], per_vertex_color[i10]
            c01, c11 = per_vertex_color[i01], per_vertex_color[i11]

        center_color = (c00 + c10 + c01 + c11) / 4.0
        boundary_colors = np.concatenate(
            [
                c00 + t[:, None] * (c10 - c00),
                c10 + t[:, None] * (c11 - c10),
                c11 + t[:, None] * (c01 - c11),
                c01 + t[:, None] * (c00 - c01),
            ]
        )  # (4*n_steps, 3)
        all_colors = np.vstack([center_color, boundary_colors]).astype("f4")

        # Fan: (center, boundary[i], boundary[(i+1) % N])
        N = 4 * n_steps
        idx = np.arange(N)
        tri_verts = np.stack(
            [
                np.tile(xyd[0], (N, 1)),
                xyd[1:][idx],
                xyd[1:][(idx + 1) % N],
            ],
            axis=1,
        )  # (N, 3, 3)
        tri_colors = np.stack(
            [
                np.tile(all_colors[0], (N, 1)),
                all_colors[1:][idx],
                all_colors[1:][(idx + 1) % N],
            ],
            axis=1,
        )  # (N, 3, 3)

        all_tri_verts.append(tri_verts)
        all_tri_colors.append(tri_colors)

    if not all_tri_verts:
        return []

    verts = np.concatenate(all_tri_verts)
    colors = np.concatenate(all_tri_colors)
    return [TriangleList2D(verts, colors.reshape(-1, 3))]  # one VBO for all faces


def _build_face_fills_batched(all_corners_cam, per_vertex_color, fill_face, fill_face_style, camera_model, n_steps):
    """Vectorized equivalent of _build_face_fills over MANY bboxes at once.

    Produces the identical boundary-fan triangle geometry as calling
    _build_face_fills per bbox, but batches the ftheta ray2pixel projection and
    the fan assembly across every bbox and every face — one ray2pixel call and
    one TriangleList2D per object type instead of ~6 per bbox. Bit-identical:
    ray2pixel is per-point, and the triangle order (bbox-major, then face, then
    fan index) matches the per-bbox path.

    Args:
        all_corners_cam: (B, 8, 3) camera-space cuboid corners for B bboxes.
    Returns:
        list[TriangleList2D]: 0 or 1 batched triangle list.
    """
    if fill_face == "none":
        return []
    corners = np.asarray(all_corners_cam)
    B = corners.shape[0]
    if B == 0:
        return []

    face_quads = _FACE_QUADS.get(fill_face, _FACE_QUADS["all"])
    fq = np.asarray(face_quads)  # (F, 4) corner indices
    F = fq.shape[0]
    N = 4 * n_steps
    t = np.linspace(0.0, 1.0, n_steps + 1)[:-1]  # (n_steps,)

    # Face corner points, camera space: (B, F, 3) each of the 4 quad corners.
    p = corners[:, fq, :]  # (B, F, 4, 3)
    p00, p10, p01, p11 = p[:, :, 0], p[:, :, 1], p[:, :, 2], p[:, :, 3]

    # Boundary loop, CCW, same interpolation as the edge renderer: (B, F, N, 3).
    tt = t[None, None, :, None]
    boundary = np.concatenate(
        [
            p00[:, :, None] + tt * (p10 - p00)[:, :, None],
            p10[:, :, None] + tt * (p11 - p10)[:, :, None],
            p11[:, :, None] + tt * (p01 - p11)[:, :, None],
            p01[:, :, None] + tt * (p00 - p01)[:, :, None],
        ],
        axis=2,
    )  # (B, F, N, 3)
    center = ((p00 + p10 + p01 + p11) / 4.0)[:, :, None]  # (B, F, 1, 3)
    all_pts = np.concatenate([center, boundary], axis=2)  # (B, F, 1+N, 3)

    # Per-face cull: drop faces entirely behind the camera (all z <= 0).
    keep = ~np.all(all_pts[..., 2] <= 0, axis=2).reshape(B * F)  # (B*F,)
    pts_flat = all_pts.reshape(B * F, 1 + N, 3)[keep]  # (K, 1+N, 3)
    K = pts_flat.shape[0]
    if K == 0:
        return []

    # One ftheta projection for every kept vertex.
    xy = camera_model.ray2pixel(pts_flat.reshape(-1, 3))  # (K*(1+N), 2)
    depth = pts_flat.reshape(-1, 3)[:, 2:3]
    xyd = np.hstack([xy, depth]).astype("f4").reshape(K, 1 + N, 3)

    # Per-face colors depend only on the face index (same per_vertex_color for
    # all bboxes of this type), so build once for F faces then gather per kept.
    if fill_face_style == "diagonal":
        all_colors_f = np.broadcast_to(_DARK, (F, 1 + N, 3)).astype("f4")
    else:
        cq = per_vertex_color[fq]  # (F, 4, 3)
        c00, c10, c01, c11 = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]
        tc = t[None, :, None]
        boundary_c = np.concatenate(
            [
                c00[:, None] + tc * (c10 - c00)[:, None],
                c10[:, None] + tc * (c11 - c10)[:, None],
                c11[:, None] + tc * (c01 - c11)[:, None],
                c01[:, None] + tc * (c00 - c01)[:, None],
            ],
            axis=1,
        )  # (F, N, 3)
        center_c = ((c00 + c10 + c01 + c11) / 4.0)[:, None]  # (F, 1, 3)
        all_colors_f = np.concatenate([center_c, boundary_c], axis=1).astype("f4")  # (F,1+N,3)

    face_idx = np.nonzero(keep)[0] % F  # (K,) face index of each kept face
    colors_k = all_colors_f[face_idx]  # (K, 1+N, 3)

    # Fan triangles: (center, boundary[i], boundary[(i+1) % N]) for i in [0, N).
    idx = np.arange(N)
    nxt = (idx + 1) % N
    bnd_v = xyd[:, 1:, :]
    tri_verts = np.stack(
        [
            np.broadcast_to(xyd[:, 0:1, :], (K, N, 3)),
            bnd_v[:, idx, :],
            bnd_v[:, nxt, :],
        ],
        axis=2,
    )  # (K, N, 3, 3)
    bnd_c = colors_k[:, 1:, :]
    tri_colors = np.stack(
        [
            np.broadcast_to(colors_k[:, 0:1, :], (K, N, 3)),
            bnd_c[:, idx, :],
            bnd_c[:, nxt, :],
        ],
        axis=2,
    )  # (K, N, 3, 3)

    verts = np.ascontiguousarray(tri_verts.reshape(-1, 3, 3))
    colors = np.ascontiguousarray(tri_colors.reshape(-1, 3))
    return [TriangleList2D(verts, colors)]  # one VBO for all faces/bboxes


def _build_edge_endpoints_batched(all_corners_cam, edge_pairs, per_vertex_color, edge_color_arr, eps=1e-3):
    """Vectorized z-near clip of a fixed set of cuboid edges over MANY bboxes.

    Equivalent to the per-bbox/per-edge Python loop: for each edge, drop it if
    both endpoints are behind the camera, otherwise clip the behind endpoint to
    the near plane. The clip parameter s = (eps - z0) / (z1 - z0) is applied to
    the start (t0) when p0 is behind and to the end (t1) when p1 is behind — the
    same if/elif math, computed for every (bbox, edge) at once. Ordering is
    bbox-major then edge-order, matching the scalar path.

    Args:
        all_corners_cam: (B, 8, 3) camera-space cuboid corners.
        edge_pairs: sequence of (i0, i1) corner-index pairs.
    Returns:
        (endpoints (M, 2, 3) float32, colors (M, 3)) or (None, None) if empty.
    """
    corners = np.asarray(all_corners_cam)
    pairs = np.asarray(edge_pairs)
    if corners.shape[0] == 0 or pairs.size == 0:
        return None, None

    i0, i1 = pairs[:, 0], pairs[:, 1]
    P0 = corners[:, i0, :]  # (B, E, 3)
    P1 = corners[:, i1, :]
    z0, z1 = P0[..., 2], P1[..., 2]  # (B, E)
    keep = ~((z0 <= 0) & (z1 <= 0))  # edge has >=1 vertex in front

    with np.errstate(divide="ignore", invalid="ignore"):
        s = (eps - z0) / (z1 - z0)  # only used where exactly one behind
    t0 = np.where(z0 <= 0, s, 0.0)
    t1 = np.where((z0 > 0) & (z1 <= 0), s, 1.0)

    d = P1 - P0
    P0c = (P0 + t0[..., None] * d).astype("f4")
    P1c = (P0 + t1[..., None] * d).astype("f4")
    endpoints = np.stack([P0c, P1c], axis=2)  # (B, E, 2, 3)

    if edge_color_arr is not None:
        colors_e = np.broadcast_to(np.asarray(edge_color_arr, dtype=float), (pairs.shape[0], 3))
    else:
        colors_e = (per_vertex_color[i0] + per_vertex_color[i1]) / 2.0  # (E, 3)
    colors = np.broadcast_to(colors_e[None], (corners.shape[0], pairs.shape[0], 3))

    return endpoints[keep], colors[keep]  # boolean index flattens bbox-major


from cosmos_auto.wm.minimap_utils import cuboid3d_to_polyline


def load_bbox_colors(version="Uniform Color"):
    """
    Load bbox colors based on the specified version.

    Args:
        version: str, version key from config_color_bbox.json
                Available versions: 'Uniform Color', 'Gradient Color', 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_scheme" / "config_color_bbox.json"
    with open(config_path, "r") as f:
        bbox_config = json.load(f)

    if version not in bbox_config:
        available_versions = list(bbox_config.keys())
        raise ValueError(f"Version '{version}' not found in bbox config. Available versions: {available_versions}")

    return bbox_config[version]


# Default colors for backward compatibility
CLASS_COLORS = load_bbox_colors("Uniform Color")
GRADIENT_CLASS_COLORS = load_bbox_colors("Gradient Color")


def simplify_type_in_object_info(object_info):
    if object_info["object_type"] not in CLASS_COLORS:
        # labels from category v1
        if object_info["object_type"] == "Bus":
            object_info["object_type"] = "Truck"
        # labels from category v1
        elif object_info["object_type"] == "Vehicle":
            object_info["object_type"] = "Car"

        # labels from category v2
        elif (
            object_info["object_type"] == "Heavy_truck"
            or object_info["object_type"] == "Train_or_tram_car"
            or object_info["object_type"] == "Trolley_bus"
            or object_info["object_type"] == "Trailer"
        ):
            object_info["object_type"] = "Truck"
        # labels from category v2
        elif object_info["object_type"] == "Automobile" or object_info["object_type"] == "Other_vehicle":
            object_info["object_type"] = "Car"
        # labels from category v2
        elif object_info["object_type"] == "Person":
            object_info["object_type"] = "Pedestrian"
        # labels from category v2
        elif object_info["object_type"] == "Rider":
            object_info["object_type"] = "Cyclist"
        else:
            object_info["object_type"] = "Others"

    return object_info


def _camera_inside_object_box(object_info, camera_pose):
    """Return whether the camera center lies inside the oriented object box."""
    object_to_world = np.asarray(object_info["object_to_world"])
    half_lwh = np.asarray(object_info["object_lwh"]) / 2.0
    camera_in_object = object_to_world[:3, :3].T @ (np.asarray(camera_pose)[:3, 3] - object_to_world[:3, 3])
    return bool(np.all(np.abs(camera_in_object) <= half_lwh + 1e-6))


def create_bbox_projection(
    all_object_info, all_camera_poses, valid_frame_ids, camera_model, color_version="Uniform Color"
):
    """
    Create a projection of bounding boxes on the minimap.
    Args:
        all_object_info: dict, containing all object info
        all_camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, including all camera poses
        camera_model: CameraModel, camera model
        valid_frame_ids: list[int], valid frame ids
        color_version: str, version key for bbox colors from config_color_bbox.json

    Returns:
        np.ndarray, shape (N, H, W, 3), dtype=np.uint8, projected bounding boxes on canvas
    """
    # Load colors based on the specified version
    class_colors = load_bbox_colors(color_version)

    bbox_projections = []

    for i in valid_frame_ids:
        current_object_info = all_object_info[f"{i:06d}.all_object_info.json"]

        object_type_to_polylines = {
            "Car": [],
            "Truck": [],
            "Pedestrian": [],
            "Cyclist": [],
            "Others": [],
        }

        tracking_ids = list(current_object_info.keys())
        tracking_ids.sort()

        for tracking_id in tracking_ids:
            object_info = current_object_info[tracking_id]
            if _camera_inside_object_box(object_info, all_camera_poses[i]):
                continue
            object_info = simplify_type_in_object_info(object_info)

            object_to_world = np.array(object_info["object_to_world"])
            object_lwh = np.array(object_info["object_lwh"])
            cuboid_eight_vertices = build_cuboid_bounding_box(
                object_lwh[0], object_lwh[1], object_lwh[2], object_to_world
            )
            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            # draw by the object type
            if object_info["object_type"] in ["Car", "Truck", "Pedestrian", "Cyclist"]:
                object_type_to_polylines[object_info["object_type"]].append(polyline)
            else:
                object_type_to_polylines["Others"].append(polyline)

        cars_bbox_projection = camera_model.draw_line_depth(
            all_camera_poses[i], object_type_to_polylines["Car"], radius=5, colors=np.array(class_colors["Car"])
        )
        trucks_bbox_projection = camera_model.draw_line_depth(
            all_camera_poses[i], object_type_to_polylines["Truck"], radius=5, colors=np.array(class_colors["Truck"])
        )
        pedestrians_bbox_projection = camera_model.draw_line_depth(
            all_camera_poses[i],
            object_type_to_polylines["Pedestrian"],
            radius=5,
            colors=np.array(class_colors["Pedestrian"]),
        )
        cyclists_bbox_projection = camera_model.draw_line_depth(
            all_camera_poses[i], object_type_to_polylines["Cyclist"], radius=5, colors=np.array(class_colors["Cyclist"])
        )
        others_bbox_projection = camera_model.draw_line_depth(
            all_camera_poses[i], object_type_to_polylines["Others"], radius=5, colors=np.array(class_colors["Others"])
        )

        # combine the dynamic and static bbox projection
        bbox_projection = np.maximum.reduce(
            [
                cars_bbox_projection,
                trucks_bbox_projection,
                pedestrians_bbox_projection,
                cyclists_bbox_projection,
                others_bbox_projection,
            ]
        )
        bbox_projections.append(bbox_projection)

    return np.concatenate(bbox_projections, axis=0)


def create_bbox_projection_gl(
    all_object_info,
    all_camera_poses,
    valid_frame_ids,
    camera_model,
    depth_max,
    fill_face="front_and_back",
    fill_face_style="diagonal",
    color_version="Gradient Color",
    line_width=9,
):
    """
    Drawing bounding boxes using moderngl api.

    Args:
        all_object_info: dict, containing all object info
        all_camera_poses: np.ndarray, camera poses
        valid_frame_ids: list[int], valid frame ids
        camera_model: CameraModel, camera model
        depth_max: float, maximum depth for rendering
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        color_version: str, version key for bbox colors from config_color_bbox.json
        line_width: int, line width for rendering
    """

    # Build per-vertex color map
    object_type_to_per_vertex_color = build_per_vertex_color_map(color_version)

    bbox_projections = []

    for i in valid_frame_ids:
        current_camera_pose = all_camera_poses[i]  # (4, 4)
        current_object_info = all_object_info[f"{i:06d}.all_object_info.json"]

        # Build geometry objects for this frame
        bounding_box_geometry_objects = create_bbox_geometry_objects_for_frame(
            current_object_info,
            current_camera_pose,
            camera_model,
            fill_face=fill_face,
            fill_face_style=fill_face_style,
            object_type_to_per_vertex_color=object_type_to_per_vertex_color,
            color_version=color_version,
            line_width=line_width,
        )

        # draw the bbox projection
        bbox_projection = render_geometries(
            bounding_box_geometry_objects,
            camera_model.height,
            camera_model.width,
            depth_max=depth_max,
            depth_gradient=True,
        )
        bbox_projections.append(bbox_projection)

    return np.stack(bbox_projections, axis=0)


def create_bbox_geometry_objects_for_frame(
    current_object_info,
    current_camera_pose,
    camera_model,
    fill_face="front_and_back",
    fill_face_style="diagonal",
    object_type_to_per_vertex_color=None,
    color_version="Gradient Color",
    line_width=9,
    edge_color=None,
    n_edge_steps=4,
    curve_subdivide=10,
    allowed_object_types=None,
):
    """
    Build BoundingBox2D geometry objects for a single frame.
    Args:
        current_object_info: dict, containing all object info for the current frame
        current_camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, camera model
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        object_type_to_per_vertex_color: dict, mapping from object type to per-vertex color
        color_version: str, version key for bbox colors from config_color_bbox.json
        line_width: int, line width for rendering
        edge_color: np.ndarray, shape (3,), dtype=np.float32, optional edge color

    Returns:
        list[BoundingBox2D], geometry objects for the current frame
    """
    # Prepare per-vertex colors if not supplied
    if object_type_to_per_vertex_color is None:
        object_type_to_per_vertex_color = build_per_vertex_color_map(color_version)
    if edge_color is not None:
        edge_color = np.array(edge_color) / 255.0

    # Store the 8 corner vertices of each object type
    object_type_to_corner_vertices = {"Car": [], "Truck": [], "Pedestrian": [], "Cyclist": [], "Others": []}

    tracking_ids = list(current_object_info.keys())
    tracking_ids.sort()

    for tracking_id in tracking_ids:
        object_info = current_object_info[tracking_id]
        if _camera_inside_object_box(object_info, current_camera_pose):
            continue
        object_info = simplify_type_in_object_info(object_info)

        object_to_world = np.array(object_info["object_to_world"])
        object_lwh = np.array(object_info["object_lwh"])
        cuboid_eight_vertices = build_cuboid_bounding_box(
            object_lwh[0], object_lwh[1], object_lwh[2], object_to_world
        )  # (8, 3)

        # Cull objects entirely behind camera
        if np.all(np.dot(cuboid_eight_vertices - current_camera_pose[:3, 3], current_camera_pose[:3, 2]) < 0):
            continue

        if object_info["object_type"] in ["Car", "Truck", "Pedestrian", "Cyclist"]:
            object_type_to_corner_vertices[object_info["object_type"]].append(cuboid_eight_vertices)
        else:
            object_type_to_corner_vertices["Others"].append(cuboid_eight_vertices)

    # Filter to only the requested object types.
    if allowed_object_types:
        object_type_to_corner_vertices = {
            k: v for k, v in object_type_to_corner_vertices.items() if k in allowed_object_types
        }

    # Determine which diagonal line indices to draw based on fill style.
    if fill_face_style == "diagonal" and fill_face not in ("none",):
        if fill_face == "front":
            diagonal_indices = FRONT_DIAGONAL_INDICES
        elif fill_face == "front_and_back":
            diagonal_indices = FRONT_DIAGONAL_INDICES + BACK_DIAGONAL_INDICES
        elif fill_face == "all":
            diagonal_indices = ALL_DIAGONAL_INDICES
        else:
            diagonal_indices = []
    else:
        diagonal_indices = []

    n_edge_steps = max(1, min(10, curve_subdivide))

    # ftheta params for the GPU geometry shader
    fw_coef = camera_model._fw_poly.coef  # (6,) forward poly coefficients
    cx, cy = camera_model._center
    lc, ld, le = camera_model.linear_cde
    ftheta_params = {
        "c": fw_coef.tolist(),
        "cx": float(cx),
        "cy": float(cy),
        "lc": float(lc),
        "ld": float(ld),
        "le": float(le),
        "n_steps": n_edge_steps,
    }

    # Accumulate all edge endpoints across all bboxes — one batched VBO per call
    all_edge_endpoints = []
    all_edge_colors = []
    all_diag_endpoints = []
    all_diag_colors = []

    # draw the bbox projection. xy are pixel coordinate, depth is in meters.
    geometry_objects = []
    for object_type, all_corner_vertices in object_type_to_corner_vertices.items():
        if len(all_corner_vertices) == 0:
            continue

        # all_corner_vertices is [n, 8, 3]
        n_objects = len(all_corner_vertices)
        all_corner_vertices_flatten = np.array(all_corner_vertices).reshape(-1, 3)
        all_points_in_cam = camera_model.transform_points(
            all_corner_vertices_flatten, np.linalg.inv(current_camera_pose)
        )
        all_depth = all_points_in_cam[:, 2:3]  # (n * 8, 1)
        all_xy = camera_model.ray2pixel(all_points_in_cam)  # (n * 8, 2)
        all_xy_and_depth = np.hstack([all_xy, all_depth]).reshape(n_objects, 8, 3)  # (n, 8, 3)
        all_corners_in_cam = all_points_in_cam.reshape(n_objects, 8, 3)  # (n, 8, 3)

        # valid corner: (1) 0 <= x <= width, (2) 0 <= y <= height, (3) depth > 0
        valid_x_mask = (all_xy_and_depth[:, :, 0] >= 0) & (all_xy_and_depth[:, :, 0] < camera_model.width)
        valid_y_mask = (all_xy_and_depth[:, :, 1] >= 0) & (all_xy_and_depth[:, :, 1] < camera_model.height)
        valid_depth_mask = all_xy_and_depth[:, :, 2] > 0
        not_valid_vertex_mask = ~valid_x_mask | ~valid_y_mask | ~valid_depth_mask
        not_valid_object_mask = np.all(not_valid_vertex_mask, axis=1)
        valid_object_mask = ~not_valid_object_mask

        all_xy_and_depth = all_xy_and_depth[valid_object_mask]
        all_corners_in_cam = all_corners_in_cam[valid_object_mask]

        per_vertex_color = object_type_to_per_vertex_color[object_type]
        edge_color_arr = np.array(edge_color) if edge_color is not None else None

        # Face fill: boundary-edge fan — batched across all bboxes of this type
        # into one VBO (vectorized ray2pixel + fan assembly).
        geometry_objects.extend(
            _build_face_fills_batched(
                all_corners_in_cam, per_vertex_color, fill_face, fill_face_style, camera_model, n_edge_steps
            )
        )

        # Edge endpoints for batched GPU geometry shader rendering, z-near
        # clipped on the CPU (all projection/subdivision runs on the GPU).
        # Vectorized across all bboxes of this type; regular and diagonal edges
        # go to separate FthetaCurvedEdges objects (different depth bias).
        reg_ep, reg_co = _build_edge_endpoints_batched(
            all_corners_in_cam, EDGE_INDICES, per_vertex_color, edge_color_arr
        )
        if reg_ep is not None and len(reg_ep):
            all_edge_endpoints.append(reg_ep)
            all_edge_colors.append(reg_co)
        if diagonal_indices:
            dg_ep, dg_co = _build_edge_endpoints_batched(
                all_corners_in_cam, diagonal_indices, per_vertex_color, edge_color_arr
            )
            if dg_ep is not None and len(dg_ep):
                all_diag_endpoints.append(dg_ep)
                all_diag_colors.append(dg_co)

    # Emit one FthetaCurvedEdges object for all regular edges and one for diagonals
    if all_edge_endpoints:
        geometry_objects.append(
            FthetaCurvedEdges(
                np.concatenate(all_edge_endpoints), np.concatenate(all_edge_colors), line_width, ftheta_params
            )
        )
    if all_diag_endpoints:
        geometry_objects.append(
            FthetaCurvedEdges(
                np.concatenate(all_diag_endpoints),
                np.concatenate(all_diag_colors),
                line_width,
                ftheta_params,
                depth_bias=-1e-4,
            )
        )

    return geometry_objects


def build_per_vertex_color_map(color_version="Gradient Color"):
    """
    Build per-vertex color arrays for each object class according to the
    bbox color configuration version.

    Returns:
        dict[str, np.ndarray(8,3)]: mapping from class name to per-vertex RGB
    """
    gradient_class_colors = load_bbox_colors(color_version)
    object_type_to_per_vertex_color = {}
    for object_type, colors in gradient_class_colors.items():
        per_vertex_color = np.zeros((8, 3))
        per_vertex_color[[0, 1, 4, 5]] = np.array(colors[0]) / 255.0
        per_vertex_color[[2, 3, 6, 7]] = np.array(colors[1]) / 255.0
        object_type_to_per_vertex_color[object_type] = per_vertex_color
    return object_type_to_per_vertex_color


def interpolate_pose(prev_pose, next_pose, t):
    """
    new pose = (1 - t) * prev_pose + t * next_pose.
    - linear interpolation for translation
    - slerp interpolation for rotation

    Args:
        prev_pose: np.ndarray, shape (4, 4), dtype=np.float32, previous pose
        next_pose: np.ndarray, shape (4, 4), dtype=np.float32, next pose
        t: float, interpolation factor

    Returns:
        np.ndarray, shape (4, 4), dtype=np.float32, interpolated pose

    Note:
        if input is list, also return list.
    """
    input_is_list = isinstance(prev_pose, list)
    prev_pose = np.array(prev_pose)
    next_pose = np.array(next_pose)

    prev_translation = prev_pose[:3, 3]
    next_translation = next_pose[:3, 3]
    translation = (1 - t) * prev_translation + t * next_translation

    prev_rotation = R.from_matrix(prev_pose[:3, :3])
    next_rotation = R.from_matrix(next_pose[:3, :3])

    times = [0, 1]
    rotations = R.from_quat([prev_rotation.as_quat(), next_rotation.as_quat()])
    rotation = Slerp(times, rotations)(t)

    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation.as_matrix()
    new_pose[:3, 3] = translation

    if input_is_list:
        return new_pose.tolist()
    else:
        return new_pose


def interpolate_bbox(all_object_info, valid_frame_ids):
    """
    Interpolate bbox from 10Hz to 30Hz.
    Args:
        all_object_info: dict, containing all object info. Keys will be
            {frame_id.06d}.all_object_info.json, where frame_id has a interval of 3.
            For example, 000000.all_object_info.json, 000003.all_object_info.json, 000006.all_object_info.json, etc.

            For one key, the value is a dict, containing all object info for that frame.
            "000000.all_object_info.json": {
                "1": {
                    "object_to_world": 4x4 matrix,
                    "object_lwh": 3-length array,
                    "object_is_moving": bool,
                    "object_type": str,
                },
                "2": {
                    ...
                },
            }

            Here "1" is the tracking id, and the value is the object info for that frame.

        valid_frame_ids: list[int], valid frame ids

    Returns:
        dict, containing interpolated object info
    """
    interpolated_all_object_info = {}

    for frame_id in valid_frame_ids:
        # no need to interpolate
        if f"{frame_id:06d}.all_object_info.json" in all_object_info:
            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = all_object_info[
                f"{frame_id:06d}.all_object_info.json"
            ]
        else:
            min_frame_id_in_all_object_info = min(
                int(key.split(".")[0]) for key in all_object_info.keys() if key.endswith(".all_object_info.json")
            )
            max_frame_id_in_all_object_info = max(
                int(key.split(".")[0]) for key in all_object_info.keys() if key.endswith(".all_object_info.json")
            )
            # find the nearest frame with object info
            prev_frame_id = frame_id
            next_frame_id = frame_id

            while f"{prev_frame_id:06d}.all_object_info.json" not in all_object_info:
                prev_frame_id -= 1
                if prev_frame_id <= 0:
                    break

            while f"{next_frame_id:06d}.all_object_info.json" not in all_object_info:
                next_frame_id += 1
                if next_frame_id >= max_frame_id_in_all_object_info:
                    break

            if prev_frame_id < min_frame_id_in_all_object_info:
                prev_frame_id = min_frame_id_in_all_object_info

            if next_frame_id > max_frame_id_in_all_object_info:
                next_frame_id = max_frame_id_in_all_object_info

            # interpolate the object info from the previous and next frame
            prev_object_info = all_object_info[f"{prev_frame_id:06d}.all_object_info.json"]
            next_object_info = all_object_info[f"{next_frame_id:06d}.all_object_info.json"]

            # tracking ids in the previous and next frame
            prev_tracking_ids = set(prev_object_info.keys())
            next_tracking_ids = set(next_object_info.keys())

            # common tracking ids in the previous and next frame
            common_tracking_ids = prev_tracking_ids & next_tracking_ids

            if next_frame_id - prev_frame_id == 0:
                t = 1
            else:
                t = (frame_id - prev_frame_id) / (next_frame_id - prev_frame_id)

            interpolated_object_info = {}
            # interpolate the object info from the previous and next frame
            for tracking_id in common_tracking_ids:
                prev_pose = np.array(prev_object_info[tracking_id]["object_to_world"])
                next_pose = np.array(next_object_info[tracking_id]["object_to_world"])
                interpolated_pose = interpolate_pose(prev_pose, next_pose, t)
                interpolated_object_info[tracking_id] = {}
                interpolated_object_info[tracking_id]["object_to_world"] = interpolated_pose.tolist()

                prev_lwh = np.array(prev_object_info[tracking_id]["object_lwh"])
                next_lwh = np.array(next_object_info[tracking_id]["object_lwh"])
                interpolated_lwh = (1 - t) * prev_lwh + t * next_lwh
                interpolated_object_info[tracking_id]["object_lwh"] = interpolated_lwh.tolist()

                interpolated_object_info[tracking_id]["object_is_moving"] = prev_object_info[tracking_id][
                    "object_is_moving"
                ]
                interpolated_object_info[tracking_id]["object_type"] = prev_object_info[tracking_id]["object_type"]

            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = interpolated_object_info

    return interpolated_all_object_info


def fix_static_objects(all_object_info):
    ############## 1. fix object_lwh ##############
    # record the lwh of static objects
    static_tracking_id_to_lwhs = {}

    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith("__"):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info["object_is_moving"]:
                if tracking_id not in static_tracking_id_to_lwhs:
                    static_tracking_id_to_lwhs[tracking_id] = []
                static_tracking_id_to_lwhs[tracking_id].append(object_info["object_lwh"])

    static_tracking_id_to_mean_lwh = {}

    for tracking_id, lwhs in static_tracking_id_to_lwhs.items():
        static_tracking_id_to_mean_lwh[tracking_id] = np.mean(lwhs, axis=0)

    # update the lwh of static objects
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith("__"):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info["object_is_moving"]:
                object_info["object_lwh"] = static_tracking_id_to_mean_lwh[tracking_id].tolist()

    ############## 2. fix object_to_world ##############
    # record the object_to_world of static objects
    static_tracking_id_to_tfms = {}
    static_tracking_id_to_headings = {}
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith("__"):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info["object_is_moving"]:
                if tracking_id not in static_tracking_id_to_tfms:
                    static_tracking_id_to_tfms[tracking_id] = []
                    static_tracking_id_to_headings[tracking_id] = []

                static_tracking_id_to_tfms[tracking_id].append(np.array(object_info["object_to_world"]))
                static_tracking_id_to_headings[tracking_id].append(
                    object_tfm_to_heading(np.array(object_info["object_to_world"]))
                )

    # compute mean heading of static objects (used to remove outlier)
    static_tracking_id_to_mean_heading = {}
    for tracking_id, headings in static_tracking_id_to_headings.items():
        static_tracking_id_to_mean_heading[tracking_id] = np.mean(headings, axis=0)
        static_tracking_id_to_mean_heading[tracking_id] /= np.linalg.norm(
            static_tracking_id_to_mean_heading[tracking_id]
        )

    # remove outlier
    threshold = 0.7
    static_tracking_id_to_tfms_remove_outlier = {}
    for tracking_id, tfms in static_tracking_id_to_tfms.items():
        for tfm in tfms:
            heading = object_tfm_to_heading(tfm)
            if np.dot(heading, static_tracking_id_to_mean_heading[tracking_id]) > threshold:
                if tracking_id not in static_tracking_id_to_tfms_remove_outlier:
                    static_tracking_id_to_tfms_remove_outlier[tracking_id] = []

                tfm_orthogoal = Isometry.from_matrix(tfm, ortho=True).matrix
                static_tracking_id_to_tfms_remove_outlier[tracking_id].append(tfm_orthogoal)

    # get the mean tfm of static objects (separate translation and rotation, use quaternion mean for rotation)
    static_tracking_id_to_mean_tfm = {}
    for tracking_id, tfms in static_tracking_id_to_tfms_remove_outlier.items():
        translation_mean = np.mean([tfm[:3, 3] for tfm in tfms], axis=0)

        front_dir_mean = np.mean([tfm[:3, 0] for tfm in tfms], axis=0)
        front_dir_mean /= np.linalg.norm(front_dir_mean)
        up_dir_mean = np.mean([tfm[:3, 2] for tfm in tfms], axis=0)
        up_dir_mean /= np.linalg.norm(up_dir_mean)

        left_dir_mean = np.cross(up_dir_mean, front_dir_mean)
        left_dir_mean /= np.linalg.norm(left_dir_mean)
        rotation_mean = np.stack([front_dir_mean, left_dir_mean, up_dir_mean], axis=1)

        static_tracking_id_to_mean_tfm[tracking_id] = np.eye(4)
        static_tracking_id_to_mean_tfm[tracking_id][:3, 3] = translation_mean
        static_tracking_id_to_mean_tfm[tracking_id][:3, :3] = rotation_mean

    # update the object_to_world of static objects
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith("__"):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info["object_is_moving"] and tracking_id in static_tracking_id_to_mean_tfm:
                object_info["object_to_world"] = static_tracking_id_to_mean_tfm[tracking_id].tolist()

    return all_object_info


def build_cuboid_bounding_box(dimXMeters, dimYMeters, dimZMeters, cuboid_transform=np.eye(4)):
    """
    Args
        dimXMeters, dimYMeters, dimZMeters: float, the dimensions of the cuboid
        cuboid_transform: 4x4 numpy array, the transformation matrix from the cuboid coordinate to the other coordinate

        z
        ^
        |   y
        | /
        |/
        o----------> x  (heading)

           3 ---------------- 0
          /|                 /|
         / |                / |
        2 ---------------- 1  |
        |  |               |  |
        |  7 ------------- |- 4
        | /                | /
        6 ---------------- 5

    Returns
        cuboid: numpy array, shape [8, 3]
            the 8 vertices of the cuboid
    """
    # Build the cuboid bounding box
    cuboid = np.array(
        [
            [dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
            [dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
            [-dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
            [-dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
            [dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
            [dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
            [-dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
            [-dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
        ]
    )
    cuboid = np.hstack([cuboid, np.ones((8, 1))])  # [8, 4]
    cuboid = np.dot(cuboid_transform, cuboid.T).T
    return cuboid[:, :3]


def object_tfm_to_heading(tfm):
    """
    Args:
        tfm: 4x4 numpy array, the transformation matrix
    Returns:
        heading_vector: [3,] numpy array, the heading of the object
    """
    if isinstance(tfm, list):
        tfm = np.array(tfm)

    heading_vector = tfm[:3, 0]
    heading_vector = heading_vector / np.linalg.norm(heading_vector)
    return heading_vector
