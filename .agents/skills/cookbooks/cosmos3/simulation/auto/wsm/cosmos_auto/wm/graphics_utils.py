import os
from typing import List, Optional, Tuple, Union

try:
    import moderngl
except ImportError:
    moderngl = None  # type: ignore[assignment]
import numpy as np

try:
    import ray
except ImportError:
    ray = None  # type: ignore[assignment]

from cosmos_auto.wm.shader_utils import (
    depth_only_fragment_shader_code,
    depth_only_vertex_shader_code,
    fragment_shader_code,
    ftheta_geometry_shader_code,
    ftheta_vertex_shader_code,
    geometry_shader_code,
    instanced_edge_fragment_shader_code,
    instanced_ftheta_vertex_shader_code,
    instanced_pinhole_vertex_shader_code,
    vertex_shader_constant_color_code,
    vertex_shader_graident_color_code,
    worldspace_ftheta_vertex_shader_code,
    worldspace_geometry_shader_code,
    worldspace_pinhole_vertex_shader_code,
    worldspace_triangle_geometry_shader_code,
)

BELOW_LEVEL_BASE_MARGIN_M = 0.6
BELOW_LEVEL_MARGIN_SLOPE = float(np.tan(np.deg2rad(2.0)))
BELOW_LEVEL_MAX_MARGIN_M = 1.5
BELOW_LEVEL_LOCAL_RANGE_M = 75.0
BELOW_LEVEL_FAR_MARGIN_M = 3.0
BELOW_LEVEL_TRANSITION_M = 15.0

# Cube edge and face indices
EDGE_INDICES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

FRONT_DIAGONAL_INDICES = [(0, 5), (1, 4)]

BACK_DIAGONAL_INDICES = [
    (2, 7),
    (3, 6),
]

ALL_DIAGONAL_INDICES = [
    (0, 5),
    (1, 4),
    (2, 7),
    (3, 6),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 7),
    (3, 4),
    (1, 6),
    (2, 5),
]

ALL_FACE_INDICES = [
    (0, 1, 5),
    (0, 4, 5),
    (2, 3, 7),
    (2, 6, 7),
    (0, 2, 3),
    (0, 1, 2),
    (4, 6, 7),
    (4, 5, 6),
    (0, 4, 7),
    (0, 3, 7),
    (1, 5, 6),
    (1, 2, 6),
]

FRONT_FACE_INDICES = [
    (0, 1, 5),
    (5, 4, 0),
]

BACK_FACE_INDICES = [
    (2, 3, 7),
    (2, 6, 7),
]


def get_remaining_face_indices(all_face_indices, exclude_face_indices):
    """
    Get the remaining face indices from the all face indices by excluding the exclude_face_indices.

    Args:
        all_face_indices: List[Tuple[int, int, int]]
            the full face indices
        exclude_face_indices: List[Tuple[int, int, int]]
            the face indices to exclude

    Returns:
        remaining_face_indices: List[Tuple[int, int, int]]
            the remaining face indices
    """
    remaining_face_indices = []
    for face_indices in all_face_indices:
        if face_indices not in exclude_face_indices:
            remaining_face_indices.append(face_indices)
    return remaining_face_indices


class Geometry2D:
    def __init__(self):
        raise NotImplementedError("Derived class must implement __init__ method")

    def render(self):
        raise NotImplementedError("Derived class must implement render method")


class LineSegment2D(Geometry2D):
    """
    This object includes N line segments, each line segment is defined by two vertices.

    Args:
        xy_and_depth: np.ndarray, [N, 2, 3]
            pixel coordinate and depth of N line segments (two vertices each) after projection
        base_color: np.ndarray, [N, 3]
            base color of the N line segments, it will faded according to the depth
        line_width: float
            line width of the N line segments
    """

    def __init__(
        self, xy_and_depth: np.ndarray, base_color: Union[Tuple[float, float, float], np.ndarray], line_width: float
    ):
        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color
        self.line_width = line_width

    def render(self, ctx, program, **kwargs):
        """
        Render the line segment.

        Args:
            ctx: moderngl.Context
                the opengl context to render the line segment
            program: moderngl.Program
                the shader program to render the line segment
            **kwargs: dict
                additional arguments, including:
                    - image_width: int
                        the width of the image
        """
        program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]

        # prepare vertex position
        vertex_position = self.xy_and_depth.reshape(-1, 3).astype("f4")

        # prepare vertex color
        vertex_base_color = np.tile(self.base_color, (vertex_position.shape[0], 1)).astype("f4")

        # prepare vbo
        vbo = ctx.buffer(vertex_position.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


class PolyLine2D(Geometry2D):
    """
    After projecting 3D vertices to 2D, the vertices are stored in this class.

    Args:
        xy_and_depth: np.ndarray, [N, 3]
            pixel coordinate and depth of N vertices of the polyline after projection
        base_color: np.ndarray, [3]
            base color of the polyline, it will faded according to the depth
        line_width: float
            line width of the polyline
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color: Union[Tuple[float, float, float], np.ndarray],
        line_width: float,
    ):
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color
        self.line_width = line_width

    def render(self, ctx, program, **kwargs):
        """
        Render the polyline.

        Args:
            ctx: moderngl.Context
                the opengl context to render the polyline
            program: moderngl.Program
                the shader program to render the polyline
            **kwargs: dict
                additional arguments, including:
                    - image_width: int
                        the width of the image
        """
        program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]

        # prepare vertex position
        xy_and_depth = self.xy_and_depth.astype("f4")
        start_vertex_position = xy_and_depth[:-1]  # [N, 3]
        end_vertex_position = xy_and_depth[1:]  # [N, 3]
        vertex_position = np.stack([start_vertex_position, end_vertex_position], axis=-2)  # [N, 2, 3]
        vertex_position = vertex_position.reshape(-1, 3)  # [2N, 3]

        # prepare vertex color
        vertex_base_color = np.tile(self.base_color, (vertex_position.shape[0], 1)).astype("f4")

        # prepare vbo
        vbo = ctx.buffer(vertex_position.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


class TriangleList2D(Geometry2D):
    """
    Render a list of triangles. Useful for rendering triangulated polygons.

    Args:
        triangles: np.ndarray, [M, 3, 3]
            M triangles, each with 3 vertices of shape [x, y, depth]
        base_color: np.ndarray, [3] or [M, 3]
            base color(s) for the triangles, will be faded according to depth
    """

    def __init__(
        self,
        triangles: np.ndarray,
        base_color: Union[Tuple[float, float, float], np.ndarray],
    ):
        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if (base_color > 1.0).any():
            raise ValueError("color must be in the range [0, 1]")

        self.triangles = triangles
        self.base_color = base_color

    def render(self, ctx, program, **kwargs):
        """
        Render the triangles.

        Args:
            ctx: moderngl.Context
                the opengl context to render the triangles
            program: moderngl.Program
                the shader program to render the triangles
            **kwargs: dict
                not used.
        """
        # Flatten triangles to vertices: [M, 3, 3] -> [M*3, 3]
        vertices = self.triangles.reshape(-1, 3).astype("f4")

        # Prepare colors for each vertex
        if self.base_color.ndim == 1:  # uniform: [3]
            vertex_colors = np.tile(self.base_color, (len(vertices), 1)).astype("f4")
        elif len(self.base_color) == len(vertices):  # per-vertex: [M*3, 3]
            vertex_colors = self.base_color.astype("f4")
        else:  # per-triangle: [M, 3] -> [M*3, 3]
            vertex_colors = np.repeat(self.base_color, 3, axis=0).astype("f4")

        vbo = ctx.buffer(vertices.tobytes())
        vbo_base_color = ctx.buffer(vertex_colors.tobytes())

        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLES
        )
        vao.render()


class FthetaCurvedEdges(Geometry2D):
    """
    All bbox edges batched into a single VBO for the ftheta geometry shader.
    CPU sends 2 raw camera-space endpoints per edge; GPU subdivides and projects.

    Args:
        edge_endpoints: np.ndarray, (N, 2, 3) — camera-space 3D endpoints per edge
        colors: np.ndarray, (N, 3) — one color per edge
        line_width: float
        ftheta_params: dict with keys c (list[6]), cx, cy, lc, ld, le, n_steps
        depth_bias: float — subtracted from z_ndc (use -1e-4 for diagonal lines on top)
    """

    def __init__(self, edge_endpoints, colors, line_width, ftheta_params, depth_bias=0.0):
        self.edge_endpoints = np.asarray(edge_endpoints, dtype="f4")
        self.colors = np.asarray(colors, dtype="f4")
        self.line_width = line_width
        self.ftheta_params = ftheta_params
        self.depth_bias = depth_bias

    def render(self, ctx, program, **kwargs):
        fp = self.ftheta_params
        c = fp["c"]
        program["c0"].value = float(c[0])
        program["c1"].value = float(c[1])
        program["c2"].value = float(c[2])
        program["c3"].value = float(c[3])
        program["c4"].value = float(c[4])
        program["c5"].value = float(c[5])
        program["cam_cx"].value = float(fp["cx"])
        program["cam_cy"].value = float(fp["cy"])
        program["cam_lc"].value = float(fp["lc"])
        program["cam_ld"].value = float(fp["ld"])
        program["cam_le"].value = float(fp["le"])
        program["image_width"].value = float(kwargs["image_width"])
        program["image_height"].value = float(kwargs["image_height"])
        program["depth_max"].value = float(kwargs["depth_max"])
        program["n_steps"].value = int(fp["n_steps"])
        program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]

        # Pack all edges as GL_LINES pairs: (N*2, 3)
        verts = self.edge_endpoints.reshape(-1, 3)
        cols = np.repeat(self.colors, 2, axis=0)  # one color per edge → both endpoints

        vbo = ctx.buffer(verts.tobytes())
        vbo_color = ctx.buffer(cols.tobytes())
        vao = ctx.vertex_array(program, [(vbo, "3f", "in_cam"), (vbo_color, "3f", "in_color")], mode=moderngl.LINES)
        vao.render()


class Polygon2D(Geometry2D):
    """
    After projecting 3D vertices to 2D, the vertices are stored in this class.

    Args:
        xy_and_depth: np.ndarray, [N, 3]
            pixel coordinate and depth of N vertices of the polygon after projection
        base_color: np.ndarray, [3]
            base color of the polygon, it will faded according to the depth
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color: Union[Tuple[float, float, float], np.ndarray],
    ):
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        # if first vertex is not the same as the last vertex, add the first vertex to the end
        if not np.allclose(xy_and_depth[0], xy_and_depth[-1]):
            xy_and_depth = np.concatenate([xy_and_depth, [xy_and_depth[0]]], axis=0)

        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color

    def render(self, ctx, program, **kwargs):
        """
        Render the polygon.

        Args:
            ctx: moderngl.Context
                the opengl context to render the polygon
            program: moderngl.Program
                the shader program to render the polygon
            **kwargs: dict
                not used.
        """
        # use all vertices as a triangle fan (ensure the vertex order is correct and closed)
        triangles = self.xy_and_depth[:-1].astype("f4")  # last vertex is the same as the first vertex
        vertex_base_color = np.tile(self.base_color, (len(triangles), 1)).astype("f4")

        vbo = ctx.buffer(triangles.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())

        # Use the TRIANGLE_FAN mode (more suitable for polygon filling)
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLE_FAN
        )
        vao.render()


class _WorldSpaceBase(Geometry2D):
    """Shared camera-parameter storage for WorldSpaceLines / WorldSpaceTriangles."""

    def _init_camera(self, camera_model):
        from cosmos_auto.wm.camera.ftheta import FThetaCamera
        from cosmos_auto.wm.camera.pinhole import PinholeCamera

        self._w2c_col_major = None
        self._level_ground_point = None
        self._level_ground_normal = None

        if isinstance(camera_model, FThetaCamera):
            self.camera_type = "ftheta"
            self._cam_params = {
                "optical_center": camera_model._center.astype("f4"),
                "fw_poly": camera_model._fw_poly.coef.astype("f4"),
                "max_ray_angle": float(camera_model._max_ray_angle),
                "max_ray_distortion": camera_model._max_ray_distortion.astype("f4"),
                "A": camera_model.A.astype("f4"),
            }
        elif isinstance(camera_model, PinholeCamera):
            self.camera_type = "pinhole"
            self._cam_params = {
                "fx": float(camera_model.fx),
                "fy": float(camera_model.fy),
                "cx": float(camera_model.cx),
                "cy": float(camera_model.cy),
            }
        else:
            raise ValueError(f"Unsupported camera model: {type(camera_model)}")

    def set_camera_pose(self, camera_pose: np.ndarray):
        """Set per-frame camera pose (camera-to-world). Computes w2c internally."""
        w2c = np.linalg.inv(camera_pose).astype(np.float32)
        self._w2c_col_major = np.ascontiguousarray(w2c.T, dtype="f4")

    def set_world_to_camera(self, world_to_camera: np.ndarray):
        """Set pre-computed world-to-camera matrix (avoids redundant inversions)."""
        self._w2c_col_major = np.ascontiguousarray(world_to_camera.astype(np.float32).T, dtype="f4")

    def set_level_plane(self, ground_plane: dict):
        """Set the pose-derived road-level plane used by the next draw."""

        point = np.asarray(ground_plane["ground_point"], dtype="f4")
        normal = np.asarray(ground_plane["normal"], dtype="f4")
        normal /= np.linalg.norm(normal)
        self._level_ground_point = np.ascontiguousarray(point)
        self._level_ground_normal = np.ascontiguousarray(normal)

    def _set_camera_uniforms(self, program):
        program["u_world_to_camera"].write(self._w2c_col_major.tobytes())
        if self.camera_type == "pinhole":
            program["u_fx"].value = self._cam_params["fx"]
            program["u_fy"].value = self._cam_params["fy"]
            program["u_cx"].value = self._cam_params["cx"]
            program["u_cy"].value = self._cam_params["cy"]
        else:
            program["u_optical_center"].write(self._cam_params["optical_center"].tobytes())
            fw_poly = self._cam_params["fw_poly"]
            padded = np.zeros(6, dtype="f4")
            padded[: len(fw_poly)] = fw_poly
            program["u_fw_poly"].write(padded.tobytes())
            program["u_max_ray_angle"].value = self._cam_params["max_ray_angle"]
            program["u_max_ray_distortion"].write(self._cam_params["max_ray_distortion"].tobytes())
            A_col_major = np.ascontiguousarray(self._cam_params["A"].T, dtype="f4")
            program["u_A"].write(A_col_major.tobytes())

    def _set_level_filter_uniforms(self, program):
        enabled = bool(self.below_level_filter)
        program["u_below_level_filter"].value = 1.0 if enabled else 0.0
        program["u_camera_depth_bias_m"].value = self.camera_depth_bias_m
        if not enabled:
            return
        if self._level_ground_point is None:
            raise RuntimeError("set_level_plane() must be called before filtered rendering")
        program["u_level_ground_point"].write(self._level_ground_point.tobytes())
        program["u_level_ground_normal"].write(self._level_ground_normal.tobytes())
        program["u_level_base_margin"].value = BELOW_LEVEL_BASE_MARGIN_M
        program["u_level_margin_slope"].value = BELOW_LEVEL_MARGIN_SLOPE
        program["u_level_max_margin"].value = BELOW_LEVEL_MAX_MARGIN_M
        program["u_level_local_range"].value = BELOW_LEVEL_LOCAL_RANGE_M
        program["u_level_far_margin"].value = BELOW_LEVEL_FAR_MARGIN_M
        program["u_level_transition"].value = BELOW_LEVEL_TRANSITION_M


class WorldSpaceLines(_WorldSpaceBase):
    """
    GPU-projected world-space line segments with per-vertex colors.

    World-space geometry is uploaded to the GPU once and projected by the
    vertex shader each frame.  Only the camera pose uniform needs updating
    per frame, eliminating expensive per-frame CPU projection.

    Args:
        segments: np.ndarray [N, 2, 3], world-space line segment endpoints.
        colors:   np.ndarray [N*2, 3], per-vertex RGB in [0, 1].
        line_width: float, pixel width passed to the geometry shader.
        camera_model: FThetaCamera or PinholeCamera.
    """

    def __init__(
        self,
        segments: np.ndarray,
        colors: np.ndarray,
        line_width: float,
        camera_model,
        below_level_filter: bool = False,
        level_positions: Optional[np.ndarray] = None,
    ):
        self._init_camera(camera_model)
        self.num_segments = len(segments)
        self.line_width = line_width
        self.below_level_filter = below_level_filter
        self.camera_depth_bias_m = 0.0
        self._vertex_positions = np.ascontiguousarray(segments.reshape(-1, 3).astype("f4"))
        self._level_positions = None
        if level_positions is not None:
            self._level_positions = np.ascontiguousarray(np.asarray(level_positions).reshape(-1, 3).astype("f4"))
        if self._level_positions is not None and self._level_positions.shape != self._vertex_positions.shape:
            raise ValueError("level_positions must match the line vertex shape")
        self._vertex_colors = np.ascontiguousarray(colors.astype("f4"))
        self._pos_vbo = None
        self._color_vbo = None
        self._level_vbo = None

    def _ensure_buffers(self, ctx):
        if self._pos_vbo is None:
            self._pos_vbo = ctx.buffer(self._vertex_positions.tobytes())
            self._color_vbo = ctx.buffer(self._vertex_colors.tobytes())
            if self._level_positions is not None:
                self._level_vbo = ctx.buffer(self._level_positions.tobytes())

    def render(self, ctx, program, **kwargs):
        if self.num_segments == 0:
            return
        self._set_camera_uniforms(program)
        self._set_level_filter_uniforms(program)
        program["u_line_width"].value = 2.0 * self.line_width / kwargs["image_width"]
        self._ensure_buffers(ctx)
        vao = ctx.vertex_array(
            program,
            [
                (self._pos_vbo, "3f", "in_pos"),
                (self._color_vbo, "3f", "in_color"),
                (
                    self._level_vbo if self._level_vbo is not None else self._pos_vbo,
                    "3f",
                    "in_level_pos",
                ),
            ],
            mode=moderngl.LINES,
        )
        vao.render()
        vao.release()

    def release_buffers(self):
        for attr in ("_pos_vbo", "_color_vbo", "_level_vbo"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.release()
                setattr(self, attr, None)


class WorldSpaceTriangles(_WorldSpaceBase):
    """
    GPU-projected world-space triangle list with per-vertex colors.

    Same pattern as WorldSpaceLines but renders triangle primitives.

    Args:
        vertices: np.ndarray [M*3, 3], triangle vertices (groups of 3) in world space.
        colors:   np.ndarray [M*3, 3], per-vertex RGB in [0, 1].
        camera_model: FThetaCamera or PinholeCamera.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        camera_model,
        below_level_filter: bool = False,
        level_positions: Optional[np.ndarray] = None,
        camera_depth_bias_m: float = 0.0,
    ):
        self._init_camera(camera_model)
        self.num_vertices = len(vertices)
        self.below_level_filter = below_level_filter
        self.camera_depth_bias_m = camera_depth_bias_m
        self._vertex_positions = np.ascontiguousarray(vertices.astype("f4"))
        self._level_positions = None
        if level_positions is not None:
            self._level_positions = np.ascontiguousarray(np.asarray(level_positions).reshape(-1, 3).astype("f4"))
        if self._level_positions is not None and self._level_positions.shape != self._vertex_positions.shape:
            raise ValueError("level_positions must match the triangle vertex shape")
        self._vertex_colors = np.ascontiguousarray(colors.astype("f4"))
        self._pos_vbo = None
        self._color_vbo = None
        self._level_vbo = None

    def _ensure_buffers(self, ctx):
        if self._pos_vbo is None:
            self._pos_vbo = ctx.buffer(self._vertex_positions.tobytes())
            self._color_vbo = ctx.buffer(self._vertex_colors.tobytes())
            if self._level_positions is not None:
                self._level_vbo = ctx.buffer(self._level_positions.tobytes())

    def render(self, ctx, program, **kwargs):
        if self.num_vertices == 0:
            return
        self._set_camera_uniforms(program)
        self._set_level_filter_uniforms(program)
        self._ensure_buffers(ctx)
        vao = ctx.vertex_array(
            program,
            [
                (self._pos_vbo, "3f", "in_pos"),
                (self._color_vbo, "3f", "in_color"),
                (
                    self._level_vbo if self._level_vbo is not None else self._pos_vbo,
                    "3f",
                    "in_level_pos",
                ),
            ],
            mode=moderngl.TRIANGLES,
        )
        vao.render()
        vao.release()

    def release_buffers(self):
        for attr in ("_pos_vbo", "_color_vbo", "_level_vbo"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.release()
                setattr(self, attr, None)


class BoundingBox2D(Geometry2D):
    """
    3D Bounding box in 2D. xy_and_depth refers to 8 vertices of the bounding box.

    The order of the vertices is:

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

    Args:
        xy_and_depth: np.ndarray, [8, 3]
            pixel coordinate and depth of 8 vertices of the bounding box after projection
        base_color_or_per_vertex_color: tuple or np.ndarray
            - if a tuple, (r, g, b), this is base color shared by all vertices
            - if a list of tuples, [(r, g, b), (r, g, b), ...], this is the color of each vertex
            - if a np.ndarray, [3], this is the base color shared by all vertices
            - if a np.ndarray, [8, 3], this is the color of each vertex

        fill_face: str = 'front', # none, front, all or front_and_back
            fill the front face, back face or both faces
        fill_face_style: str = 'solid',
            - solid: fill the face with color
            - diagonal: connect the diagonal of the face
        line_width: float = 4.0
            line width of the bounding box
        force_all_edge_on_top: bool = False
            if true, make all edges (including diagonal edges) on top of rendered face polygons
        edge_color: np.ndarray, shape (3,), dtype=np.float32, optional edge color
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color_or_per_vertex_color: Union[Tuple[float, float, float], List[Tuple[float, float, float]], np.ndarray],
        fill_face: str = "front",  # none, front or all, front_and_back
        fill_face_style: str = "solid",
        line_width: float = 4.0,
        force_all_edge_on_top: bool = False,
        edge_color: Union[Tuple[float, float, float], np.ndarray, None] = None,
        draw_edges: bool = True,
    ):
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        if isinstance(base_color_or_per_vertex_color, tuple) or isinstance(base_color_or_per_vertex_color, list):
            base_color_or_per_vertex_color = np.array(base_color_or_per_vertex_color)

        if (base_color_or_per_vertex_color > 1.0).any():
            raise ValueError("color must be in the range [0, 1]")

        if base_color_or_per_vertex_color.shape == (3,):
            self.per_vertex_color = np.tile(base_color_or_per_vertex_color, (8, 1))
        elif base_color_or_per_vertex_color.shape == (8, 3):
            self.per_vertex_color = base_color_or_per_vertex_color
        else:
            raise ValueError(f"Invalid base_color_or_per_vertex_color: {base_color_or_per_vertex_color.shape}")

        self.xy_and_depth = xy_and_depth
        self.fill_face = fill_face
        self.fill_face_style = fill_face_style
        self.edge_color = edge_color
        self.line_width = line_width
        self.force_all_edge_on_top = force_all_edge_on_top
        self.draw_edges = draw_edges

    def render(self, ctx, polyline_program, polygon_program, **kwargs):
        polyline_program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]

        # if fill_face_style is solid (not diagonal), draw the solid face
        if self.fill_face == "front" and self.fill_face_style == "solid":
            solid_face_indices = FRONT_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, FRONT_FACE_INDICES)
        elif self.fill_face == "back" and self.fill_face_style == "solid":
            solid_face_indices = BACK_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, BACK_FACE_INDICES)
        elif self.fill_face == "front_and_back" and self.fill_face_style == "solid":
            solid_face_indices = FRONT_FACE_INDICES + BACK_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, solid_face_indices)
        elif self.fill_face == "all" and self.fill_face_style == "solid":
            solid_face_indices = ALL_FACE_INDICES
            black_face_indices = None
        elif self.fill_face == "none" and self.fill_face_style == "solid":
            solid_face_indices = None
            black_face_indices = ALL_FACE_INDICES
        elif self.fill_face_style == "diagonal":
            solid_face_indices = None
            black_face_indices = ALL_FACE_INDICES  # make all faces black
        else:
            raise ValueError(
                f"Invalid fill_face_style x fill_face combination: {self.fill_face_style} x {self.fill_face}"
            )

        draw_triangles_vertices = []
        draw_triangles_colors = []

        if solid_face_indices is not None:
            for i0, i1, i2 in solid_face_indices:
                draw_triangles_vertices.append(self.xy_and_depth[i0])
                draw_triangles_vertices.append(self.xy_and_depth[i1])
                draw_triangles_vertices.append(self.xy_and_depth[i2])
                draw_triangles_colors.append(self.per_vertex_color[i0])
                draw_triangles_colors.append(self.per_vertex_color[i1])
                draw_triangles_colors.append(self.per_vertex_color[i2])

        if black_face_indices is not None:
            for i0, i1, i2 in black_face_indices:
                draw_triangles_vertices.append(self.xy_and_depth[i0])
                draw_triangles_vertices.append(self.xy_and_depth[i1])
                draw_triangles_vertices.append(self.xy_and_depth[i2])
                draw_triangles_colors.append(np.ones(3) * 0.25)
                draw_triangles_colors.append(np.ones(3) * 0.25)
                draw_triangles_colors.append(np.ones(3) * 0.25)

        if len(draw_triangles_vertices) > 0:
            draw_triangles_vertices = np.concatenate(draw_triangles_vertices, axis=0).astype("f4")
            draw_triangles_colors = np.concatenate(draw_triangles_colors, axis=0).astype("f4")

            vbo = ctx.buffer(draw_triangles_vertices.tobytes())
            vbo_base_color = ctx.buffer(draw_triangles_colors.tobytes())
            vao = ctx.vertex_array(
                polygon_program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLES
            )
            vao.render()

        if self.draw_edges:
            # draw edges, make small depth shift to make sure edges are on top of the faces
            line_vertices = []
            line_colors = []

            for i0, i1 in EDGE_INDICES:
                line_vertices.append(self.xy_and_depth[i0])
                line_vertices.append(self.xy_and_depth[i1])
                if self.edge_color is not None:
                    line_colors.append(self.edge_color)
                    line_colors.append(self.edge_color)
                else:
                    line_colors.append(self.per_vertex_color[i0])
                    line_colors.append(self.per_vertex_color[i1])

            # draw diagonal lines if needed
            if self.fill_face_style == "diagonal" and self.fill_face != "none":
                if self.fill_face == "front":
                    diagonal_line_indices = FRONT_DIAGONAL_INDICES
                elif self.fill_face == "front_and_back":
                    diagonal_line_indices = FRONT_DIAGONAL_INDICES + BACK_DIAGONAL_INDICES
                elif self.fill_face == "all":
                    diagonal_line_indices = ALL_DIAGONAL_INDICES
                else:
                    raise ValueError(f"Invalid fill_face: {self.fill_face}")

                for i0, i1 in diagonal_line_indices:
                    line_vertices.append(self.xy_and_depth[i0])
                    line_vertices.append(self.xy_and_depth[i1])
                    line_colors.append(self.per_vertex_color[i0])
                    line_colors.append(self.per_vertex_color[i1])

                    line_vertices[-1][2] -= 1e-4  # make sure diagonal lines are on top of the faces
                    line_vertices[-2][2] -= 1e-4  # make sure diagonal lines are on top of the faces

            line_vertices = np.array(line_vertices, dtype="f4")
            if self.force_all_edge_on_top:
                line_vertices[:, 2] = np.min(self.xy_and_depth[:, 2]) - 1e-4

            line_colors = np.array(line_colors, dtype="f4")

            vbo = ctx.buffer(line_vertices.tobytes())
            vbo_base_color = ctx.buffer(line_colors.tobytes())
            vao = ctx.vertex_array(
                polyline_program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
            )
            vao.render()


class InstancedVoxels(Geometry2D):
    """
    WMv4.1 GPU-instanced rendering of axis-aligned rectangular prisms (voxel towers)
    for static occupancy.

    Instead of generating N*8 corners on the CPU and projecting them through the
    camera model, this class uploads a single unit-cube template (8 vertices) and
    per-instance data (center, height, color), then lets the vertex shader handle
    scaling, world-to-camera transformation, and camera projection (pinhole or
    ftheta) entirely on the GPU.

    Each instanced draw call renders all N towers with a single glDrawElementsInstanced,
    dramatically reducing CPU work and driver overhead.

    Args:
        centers: np.ndarray [N, 3], world-space center of each tower.
        heights: np.ndarray [N], height of each tower in meters.
        colors: np.ndarray [N, 3], RGB color for each tower (range 0-1).
        cell_resolution: float, width and depth of each tower in meters.
        camera_model: CameraModel (FThetaCamera or PinholeCamera).
        max_distance: float or None, GPU-side forward culling distance.
        max_lateral: float or None, GPU-side lateral culling distance.
        line_width: float, edge line width.
    """

    _CUBE_VERTS = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ],
        dtype="f4",
    )

    _FACE_IBO = np.array([idx for tri in ALL_FACE_INDICES for idx in tri], dtype="i4")
    _EDGE_IBO = np.array([idx for edge in EDGE_INDICES for idx in edge], dtype="i4")

    def __init__(
        self,
        centers: np.ndarray,
        heights: np.ndarray,
        colors: np.ndarray,
        cell_resolution: float,
        camera_model,
        max_distance: Optional[float] = None,
        max_lateral: Optional[float] = None,
        line_width: float = 0.5,
        draw_edges: bool = True,
        edge_color: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
    ):
        from cosmos_auto.wm.camera.ftheta import FThetaCamera
        from cosmos_auto.wm.camera.pinhole import PinholeCamera

        self.centers = centers.astype("f4")
        self.heights = heights.astype("f4")
        self.colors = colors.astype("f4")
        self.cell_resolution = float(cell_resolution)
        self.max_distance = float(max_distance) if max_distance is not None else -1.0
        self.max_lateral = float(max_lateral) if max_lateral is not None else -1.0
        self.line_width = line_width
        self.draw_edges = draw_edges
        self.edge_color = np.asarray(edge_color, dtype="f4")
        self.num_instances = len(centers)

        self._w2c_col_major = None

        if isinstance(camera_model, FThetaCamera):
            self.camera_type = "ftheta"
            self._cam_params = {
                "optical_center": camera_model._center.astype("f4"),
                "fw_poly": camera_model._fw_poly.coef.astype("f4"),
                "max_ray_angle": float(camera_model._max_ray_angle),
                "max_ray_distortion": camera_model._max_ray_distortion.astype("f4"),
                "A": camera_model.A.astype("f4"),
            }
        elif isinstance(camera_model, PinholeCamera):
            self.camera_type = "pinhole"
            self._cam_params = {
                "fx": float(camera_model.fx),
                "fy": float(camera_model.fy),
                "cx": float(camera_model.cx),
                "cy": float(camera_model.cy),
            }
        else:
            raise ValueError(f"Unsupported camera model type: {type(camera_model)}")

        self._cube_vbo = None
        self._instance_vbo = None
        self._face_ibo = None
        self._edge_ibo = None
        self._instance_dirty = True

    def set_camera_pose(self, camera_pose: np.ndarray):
        """Set the per-frame camera pose (camera-to-world transform)."""
        world_to_camera = np.linalg.inv(camera_pose).astype(np.float32)
        self._w2c_col_major = np.ascontiguousarray(world_to_camera.T, dtype="f4")

    def update_centers(self, centers: np.ndarray):
        """Replace instance centers and mark GPU buffer for re-upload."""
        self.centers = centers.astype("f4")
        self.num_instances = len(centers)
        self._instance_dirty = True

    def update_colors(self, colors: np.ndarray):
        """Replace instance colors and mark GPU buffer for re-upload."""
        self.colors = colors.astype("f4")
        self._instance_dirty = True

    def _ensure_buffers(self, ctx):
        """Allocate or update GPU buffers, skipping work when data hasn't changed."""
        if self._cube_vbo is None:
            self._cube_vbo = ctx.buffer(self._CUBE_VERTS.tobytes())
            self._face_ibo = ctx.buffer(self._FACE_IBO.tobytes())
            self._edge_ibo = ctx.buffer(self._EDGE_IBO.tobytes())

        if self._instance_dirty:
            instance_data = np.empty((self.num_instances, 7), dtype="f4")
            instance_data[:, :3] = self.centers
            instance_data[:, 3] = self.heights
            instance_data[:, 4:7] = self.colors
            data_bytes = instance_data.tobytes()

            if self._instance_vbo is not None and self._instance_vbo.size == len(data_bytes):
                self._instance_vbo.write(data_bytes)
            else:
                if self._instance_vbo is not None:
                    self._instance_vbo.release()
                self._instance_vbo = ctx.buffer(data_bytes)
            self._instance_dirty = False

    def release_buffers(self):
        """Release cached GPU buffers."""
        for attr in ("_cube_vbo", "_instance_vbo", "_face_ibo", "_edge_ibo"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.release()
                setattr(self, attr, None)

    def _set_uniforms(self, program, image_width):
        program["u_world_to_camera"].write(self._w2c_col_major.tobytes())
        program["u_box_size"].value = self.cell_resolution
        program["u_max_distance"].value = self.max_distance
        program["u_max_lateral"].value = self.max_lateral

        if self.camera_type == "pinhole":
            program["u_fx"].value = self._cam_params["fx"]
            program["u_fy"].value = self._cam_params["fy"]
            program["u_cx"].value = self._cam_params["cx"]
            program["u_cy"].value = self._cam_params["cy"]
        else:
            program["u_optical_center"].write(self._cam_params["optical_center"].tobytes())
            fw_poly = self._cam_params["fw_poly"]
            padded = np.zeros(6, dtype="f4")
            padded[: len(fw_poly)] = fw_poly
            program["u_fw_poly"].write(padded.tobytes())
            program["u_max_ray_angle"].value = self._cam_params["max_ray_angle"]
            program["u_max_ray_distortion"].write(self._cam_params["max_ray_distortion"].tobytes())
            A_col_major = np.ascontiguousarray(self._cam_params["A"].T, dtype="f4")
            program["u_A"].write(A_col_major.tobytes())

    def render(self, ctx, face_program, edge_program, **kwargs):
        if self.num_instances == 0:
            return

        image_width = kwargs["image_width"]
        self._set_uniforms(face_program, image_width)

        self._ensure_buffers(ctx)

        face_vao = ctx.vertex_array(
            face_program,
            [
                (self._cube_vbo, "3f", "in_vert"),
                (self._instance_vbo, "3f 1f 3f/i", "in_center", "in_height", "in_color"),
            ],
            index_buffer=self._face_ibo,
        )
        face_vao.render(mode=moderngl.TRIANGLES, instances=self.num_instances)
        face_vao.release()

        if not self.draw_edges:
            return

        self._set_uniforms(edge_program, image_width)
        edge_program["u_line_width"].value = 2.0 * self.line_width / image_width
        edge_program["u_edge_color"].value = tuple(self.edge_color)

        edge_vao = ctx.vertex_array(
            edge_program,
            [
                (self._cube_vbo, "3f", "in_vert"),
                (self._instance_vbo, "3f 1f 3f/i", "in_center", "in_height", "in_color"),
            ],
            index_buffer=self._edge_ibo,
        )
        edge_vao.render(mode=moderngl.LINES, instances=self.num_instances)
        edge_vao.release()


def create_polyline_program(ctx, depth_gradient):
    """
    Create a program for rendering polylines.

    Args:
        ctx: moderngl.Context
            the context to create the program
        depth_gradient: bool
            whether to use variable color according to the depth

    Returns:
        program: moderngl.Program
            the program for rendering polyline
    """
    vertex_shader = vertex_shader_graident_color_code if depth_gradient else vertex_shader_constant_color_code
    geometry_shader = geometry_shader_code
    fragment_shader = fragment_shader_code.replace("v_color", "g_color")
    return ctx.program(vertex_shader=vertex_shader, geometry_shader=geometry_shader, fragment_shader=fragment_shader)


def create_polygon_program(ctx, depth_gradient):
    """
    Args:
        ctx: moderngl.Context
            the context to create the program
        depth_gradient: bool
            whether to use variable color according to the depth

    Returns:
        program: moderngl.Program
            the program for rendering polygon
    """
    vertex_shader = vertex_shader_graident_color_code if depth_gradient else vertex_shader_constant_color_code
    fragment_shader = fragment_shader_code
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def create_polygon_depth_only_program(ctx):
    """
    Create a program for updating depth buffer for polygons (without updating color),

    Args:
        ctx: moderngl.Context
            the context to create the program

    Returns:
        program: moderngl.Program
            the program for updating depth buffer for polygons (without updating color)
    """
    vertex_shader = depth_only_vertex_shader_code
    fragment_shader = depth_only_fragment_shader_code
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def _ensure_instanced_programs(ctx, cached_item, geometries, depth_gradient):
    """Lazily create and cache instanced shader programs when InstancedVoxels are present."""
    for geom in geometries:
        if not isinstance(geom, InstancedVoxels):
            continue
        key = f"instanced_{geom.camera_type}"
        if key + "_face" not in cached_item:
            if geom.camera_type == "pinhole":
                vs = instanced_pinhole_vertex_shader_code
            else:
                vs = instanced_ftheta_vertex_shader_code

            face_prog = ctx.program(vertex_shader=vs, fragment_shader=fragment_shader_code)
            gs = geometry_shader_code
            edge_prog = ctx.program(
                vertex_shader=vs,
                geometry_shader=gs,
                fragment_shader=instanced_edge_fragment_shader_code,
            )

            cached_item[key + "_face"] = face_prog
            cached_item[key + "_edge"] = edge_prog

        value = 1.0 if depth_gradient else 0.0
        cached_item[key + "_face"]["use_depth_gradient"].value = value
        cached_item[key + "_edge"]["use_depth_gradient"].value = value


def _ensure_worldspace_programs(ctx, cached_item, camera_type):
    """Lazily create and cache worldspace shader programs for GPU-projected geometry."""
    key_prefix = f"worldspace_{camera_type}"
    if key_prefix + "_line" in cached_item:
        return

    if camera_type == "ftheta":
        vs = worldspace_ftheta_vertex_shader_code
    else:
        vs = worldspace_pinhole_vertex_shader_code

    fs_line = fragment_shader_code.replace("v_color", "g_color")

    cached_item[key_prefix + "_line"] = ctx.program(
        vertex_shader=vs,
        geometry_shader=worldspace_geometry_shader_code,
        fragment_shader=fs_line,
    )
    cached_item[key_prefix + "_polygon"] = ctx.program(
        vertex_shader=vs,
        geometry_shader=worldspace_triangle_geometry_shader_code,
        fragment_shader=fs_line,
    )


def create_ftheta_program(ctx, depth_gradient):
    vertex_shader = ftheta_vertex_shader_code
    geometry_shader = ftheta_geometry_shader_code
    fragment_shader = fragment_shader_code.replace("v_color", "g_color")
    return ctx.program(vertex_shader=vertex_shader, geometry_shader=geometry_shader, fragment_shader=fragment_shader)


_rendering_cache = {}


def render_geometries(
    geometries: list[Geometry2D],
    image_height: int,
    image_width: int,
    depth_max: float,
    depth_gradient: bool = True,
    multi_sample: int = 4,
    use_cache: bool = True,
) -> np.ndarray:
    """
    A unified function to render geometries.

    Args:
        geometries: List[Geometry2D]
            a list of geometries to render, can be PolyLine2D, Polygon2D, BoundingBox2D
        image_height: int
            the height of the image
        image_width: int
            the width of the image
        depth_max: float
            the maximum depth of the image
        depth_gradient: bool
            whether to use variable color according to the depth
        use_cache: bool
            whether to use cache for rendering context and programs

    Returns:
        image: np.ndarray, [H, W, 3]
            the rendered image, type: np.uint8
    """
    # get visible GPU from ray's api
    if ray is not None and ray.is_initialized():
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) == 1, (
            f"Only one GPU is needed, but got {len(gpu_ids)} GPUs. Set num_gpus <= 1 in ray_remote decorator."
        )
        gpu_index = gpu_ids[0]
    else:
        gpu_index = 0

    pid = os.getpid()
    cache_key = (pid, gpu_index, depth_gradient, multi_sample, image_width, image_height)

    cached_item = _rendering_cache.get(cache_key) if use_cache else None

    if cached_item:
        ctx = cached_item["ctx"]
        polyline_program = cached_item["polyline_program"]
        polygon_program = cached_item["polygon_program"]
        ftheta_program = cached_item["ftheta_program"]
        if multi_sample > 1:
            msaa_fbo = cached_item["msaa_fbo"]
            resolve_fbo = cached_item["resolve_fbo"]
            msaa_fbo.use()
            msaa_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        else:
            fbo = cached_item["fbo"]
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
    else:
        ctx = moderngl.create_context(
            standalone=True,
            backend="egl",
            libgl="libGL.so.1",
            libegl="libEGL.so.1",
            device_index=gpu_index,
        )
        ctx.enable(moderngl.DEPTH_TEST)

        cached_item = {"ctx": ctx}

        if multi_sample > 1:
            color_rb = ctx.renderbuffer((image_width, image_height), samples=multi_sample)
            depth_rb = ctx.depth_renderbuffer((image_width, image_height), samples=multi_sample)
            msaa_fbo = ctx.framebuffer(color_attachments=[color_rb], depth_attachment=depth_rb)
            msaa_fbo.use()
            msaa_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

            resolve_color_rb = ctx.renderbuffer((image_width, image_height))
            resolve_fbo = ctx.framebuffer(color_attachments=[resolve_color_rb])
            cached_item.update(
                {
                    "msaa_fbo": msaa_fbo,
                    "resolve_fbo": resolve_fbo,
                    "color_rb": color_rb,
                    "depth_rb": depth_rb,
                    "resolve_color_rb": resolve_color_rb,
                }
            )
        else:
            color_rb = ctx.renderbuffer((image_width, image_height))
            depth_rb = ctx.depth_renderbuffer((image_width, image_height))
            fbo = ctx.framebuffer(color_attachments=[color_rb], depth_attachment=depth_rb)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
            cached_item.update({"fbo": fbo, "color_rb": color_rb, "depth_rb": depth_rb})

        polyline_program = create_polyline_program(ctx, depth_gradient)
        polygon_program = create_polygon_program(ctx, depth_gradient)
        ftheta_program = create_ftheta_program(ctx, depth_gradient)
        cached_item.update(
            {
                "polyline_program": polyline_program,
                "polygon_program": polygon_program,
                "ftheta_program": ftheta_program,
            }
        )
        if use_cache:
            _rendering_cache[cache_key] = cached_item

    polyline_program["image_height"].value = image_height
    polyline_program["image_width"].value = image_width
    polyline_program["depth_max"].value = depth_max
    polygon_program["image_height"].value = image_height
    polygon_program["image_width"].value = image_width
    polygon_program["depth_max"].value = depth_max

    _ensure_instanced_programs(ctx, cached_item, geometries, depth_gradient)

    for geometry in geometries:
        if isinstance(geometry, PolyLine2D) or isinstance(geometry, LineSegment2D):
            geometry.render(ctx, polyline_program, image_width=image_width)

        elif isinstance(geometry, FthetaCurvedEdges):
            geometry.render(
                ctx, ftheta_program, image_width=image_width, image_height=image_height, depth_max=depth_max
            )

        elif isinstance(geometry, Polygon2D):
            geometry.render(ctx, polygon_program)

        elif isinstance(geometry, TriangleList2D):
            geometry.render(ctx, polygon_program)

        elif isinstance(geometry, BoundingBox2D):
            geometry.render(ctx, polyline_program, polygon_program, image_width=image_width)

        elif isinstance(geometry, InstancedVoxels):
            key = f"instanced_{geometry.camera_type}"
            face_prog = cached_item[key + "_face"]
            edge_prog = cached_item[key + "_edge"]
            for prog in (face_prog, edge_prog):
                prog["image_height"].value = image_height
                prog["image_width"].value = image_width
                prog["depth_max"].value = depth_max
            geometry.render(ctx, face_prog, edge_prog, image_width=image_width)

        elif isinstance(geometry, WorldSpaceLines):
            _ensure_worldspace_programs(ctx, cached_item, geometry.camera_type)
            prog = cached_item[f"worldspace_{geometry.camera_type}_line"]
            prog["image_height"].value = image_height
            prog["image_width"].value = image_width
            prog["depth_max"].value = depth_max
            prog["use_depth_gradient"].value = 1.0 if depth_gradient else 0.0
            geometry.render(ctx, prog, image_width=image_width)

        elif isinstance(geometry, WorldSpaceTriangles):
            _ensure_worldspace_programs(ctx, cached_item, geometry.camera_type)
            prog = cached_item[f"worldspace_{geometry.camera_type}_polygon"]
            prog["image_height"].value = image_height
            prog["image_width"].value = image_width
            prog["depth_max"].value = depth_max
            prog["use_depth_gradient"].value = 1.0 if depth_gradient else 0.0
            geometry.render(ctx, prog)

    # Read the color buffer data and convert it to a NumPy array
    if multi_sample > 1:
        resolve_fbo = cached_item["resolve_fbo"]
        msaa_fbo = cached_item["msaa_fbo"]
        ctx.copy_framebuffer(resolve_fbo, msaa_fbo)
        data = resolve_fbo.read(components=3, alignment=1)
    else:
        fbo = cached_item["fbo"]
        data = fbo.read(components=3, alignment=1)

    image = np.frombuffer(data, dtype=np.uint8).reshape((image_height, image_width, 3))
    image = np.flipud(image)  # Convert OpenGL coordinates to NumPy image coordinates

    return image
