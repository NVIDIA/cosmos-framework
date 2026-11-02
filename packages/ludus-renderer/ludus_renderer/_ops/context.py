# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ludus rendering contexts - LudusGLContext and LudusTimestampedContext."""

import struct
from typing import List, Optional, Tuple, Union

import torch

from ._plugin import _get_plugin
from .primitives import (
    CAMERA_TYPE_REGULAR,
    PRIM_TYPE_COUNT,
    Cube,
    FThetaCamera,
    Polygon,
    Polyline,
    TimestampedScene,
    _pack_cameras,
    _pack_cubes,
    _pack_polygons,
    _pack_polylines,
)


class LudusGLContext:
    """OpenGL context for Ludus f-theta mesh shader rendering."""

    def __init__(self, device=None):
        """Create a new Ludus GL context.

        Args:
            device: CUDA device for the context. If None, uses current device.
        """
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cpp_wrapper = _get_plugin(gl=True).LudusGLStateWrapper(cuda_device_idx)
        self.cuda_device_idx = cuda_device_idx

    def set_msaa_samples(self, samples: int) -> None:
        """Set MSAA sample count for antialiasing.

        Args:
            samples: Number of samples (0=disabled, 2, 4, or 8)
        """
        self.cpp_wrapper.set_msaa_samples(samples)


def ludus_render(
    glctx: LudusGLContext,
    cameras: List[FThetaCamera],
    camera_poses: torch.Tensor,
    resolution: tuple,
    cubes: Optional[List[Cube]] = None,
    polylines: Optional[List[Polyline]] = None,
    polygons: Optional[List[Polygon]] = None,
    tessellation_threshold: float = 0.0,
) -> torch.Tensor:
    """Render scene with f-theta fisheye cameras using mesh shaders.

    Args:
        glctx: LudusGLContext created for rendering.
        cameras: List of FThetaCamera intrinsics.
        camera_poses: [P, 4, 4] float32 world-to-camera transforms.
        resolution: (H, W) output image resolution.
        cubes: Optional list of Cube objects.
        polylines: Optional list of Polyline objects.
        polygons: Optional list of Polygon objects.
        tessellation_threshold: Pixel error threshold for adaptive tessellation.

    Returns:
        [P, H, W, 4] float32 RGBA images for all cameras.
    """
    assert isinstance(glctx, LudusGLContext), "ludus_render requires a LudusGLContext"

    device = camera_poses.device
    resolution = tuple(resolution)

    cubes = cubes or []
    polylines = polylines or []
    polygons = polygons or []

    # Pack data
    cubes_tensor = _pack_cubes(cubes, device)
    polyline_headers, polyline_vertices = _pack_polylines(polylines, device)
    num_polyline_verts = polyline_vertices.shape[0]
    polygon_headers, polygon_vertices, polygon_triangles = _pack_polygons(polygons, num_polyline_verts, device)

    # Merge vertex buffers
    if num_polyline_verts > 0 and polygon_vertices.shape[0] > 0:
        all_vertices = torch.cat([polyline_vertices, polygon_vertices], dim=0)
    elif num_polyline_verts > 0:
        all_vertices = polyline_vertices
    elif polygon_vertices.shape[0] > 0:
        all_vertices = polygon_vertices
    else:
        all_vertices = torch.empty((0, 4), dtype=torch.float32, device=device)

    camera_intrinsics = _pack_cameras(cameras, device)

    num_cameras = len(cameras)
    assert camera_poses.shape == (num_cameras, 4, 4), (
        f"camera_poses must have shape [{num_cameras}, 4, 4], got {camera_poses.shape}"
    )

    # Ensure contiguous
    cubes_tensor = cubes_tensor.contiguous()
    polyline_headers = polyline_headers.contiguous()
    polygon_headers = polygon_headers.contiguous()
    all_vertices = all_vertices.contiguous()
    polygon_triangles = polygon_triangles.contiguous()
    camera_intrinsics = camera_intrinsics.contiguous()
    camera_poses = camera_poses.contiguous()

    out_rgba = _get_plugin(gl=True).ludus_render_fwd_gl(
        glctx.cpp_wrapper,
        polyline_headers,
        polygon_headers,
        cubes_tensor,
        all_vertices,
        polygon_triangles,
        camera_intrinsics,
        camera_poses,
        resolution,
        tessellation_threshold,
    )

    return out_rgba


def _compute_element_aabbs(vertices: torch.Tensor, prefix_sum: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute per-element AABBs from vertices and prefix sum.

    Returns a flat ``[n_elements * 6]`` float32 tensor with
    ``(min_x, min_y, min_z, max_x, max_y, max_z)`` per element.
    """
    n_elem = len(prefix_sum)
    if n_elem == 0 or len(vertices) == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    verts = vertices.to(device, dtype=torch.float32)
    if verts.ndim == 2 and verts.shape[1] > 3:
        verts = verts[:, :3]
    ps = prefix_sum.to(device, dtype=torch.int64).contiguous()

    vid = torch.arange(len(verts), device=device, dtype=torch.int64)
    elem_id = torch.searchsorted(ps, vid, side="right")

    e_min = torch.full((n_elem, 3), float("inf"), device=device)
    e_max = torch.full((n_elem, 3), float("-inf"), device=device)
    idx_exp = elem_id.unsqueeze(-1).expand(-1, 3)
    e_min.scatter_reduce_(0, idx_exp, verts, reduce="amin")
    e_max.scatter_reduce_(0, idx_exp, verts, reduce="amax")

    return torch.cat([e_min, e_max], dim=-1).reshape(-1)


class LudusCudaTimestampedContext:
    """CUDA-only context for timestamped scene rendering.

    Drop-in replacement for ``LudusTimestampedContext`` with no OpenGL
    dependency.  All timestamp search, element extraction, color/width
    lookup, and geometry generation happen on the GPU via CUDA kernels.
    """

    def __init__(self, device=None):
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cuda_device_idx = cuda_device_idx
        self.cpp_wrapper = _get_plugin(gl=False).LudusCudaStateWrapper(cuda_device_idx)
        self._tessellation_threshold = 1.0
        self._max_extrapolation_us = 500_000

        self._cameras: List[FThetaCamera] = []
        self._camera_intrinsics: Optional[torch.Tensor] = None
        self.needs_vflip = False  # CUDA renders top-down (standard image convention)

        # Per-scene flat buffers
        self._scenes: List[dict] = []

    @property
    def max_batch_size(self) -> int:
        """CUDA rasterizer has no GL texture layer limit."""
        return 2048

    # ------------------------------------------------------------------
    def upload_cameras(self, cameras: List[FThetaCamera]) -> None:
        device = torch.device(f"cuda:{self.cuda_device_idx}")
        self._cameras = cameras
        self._camera_intrinsics = _pack_cameras(cameras, device)

    # ------------------------------------------------------------------
    def upload_scene(self, scene: TimestampedScene) -> int:
        device = torch.device(f"cuda:{self.cuda_device_idx}")

        timestamps_list: List[torch.Tensor] = []
        int32_list: List[torch.Tensor] = []
        vertices_list: List[torch.Tensor] = []
        triangles_list: List[torch.Tensor] = []
        float_list: List[torch.Tensor] = []

        polyline_pool_headers: List[torch.Tensor] = []
        polygon_pool_headers: List[torch.Tensor] = []
        cube_pool_headers: List[torch.Tensor] = []

        ts_offset = 0
        int32_offset = 0
        vert_offset = 0
        tri_offset = 0
        float_offset = 0

        max_varrays_per_ts_polyline = 0
        max_varrays_per_ts_polygon = 0

        for pool in scene.polyline_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]

            # Compute per-timestamp max varrays from prefix sum
            ps = pool.timestamped_varrays_prefix_sum
            diffs = torch.diff(ps, prepend=torch.tensor([0], dtype=ps.dtype, device=ps.device))
            mvpt = int(diffs.max().item()) if len(diffs) > 0 else 0
            if mvpt > max_varrays_per_ts_polyline:
                max_varrays_per_ts_polyline = mvpt

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = pool.prim_type_id
            header[4] = ts_offset
            header[5] = int32_offset  # ts_varrays_ps
            header[6] = int32_offset + n_ts  # varrays_ps
            header[7] = vert_offset
            header[8] = float_offset  # aabb

            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)
            polyline_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + n_varrays
            vert_offset += n_verts
            float_offset += len(aabbs)

        for pool in scene.polygon_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]
            n_tris = pool.triangles.shape[0]

            ps = pool.timestamped_varrays_prefix_sum
            diffs = torch.diff(ps, prepend=torch.tensor([0], dtype=ps.dtype, device=ps.device))
            mvpt = int(diffs.max().item()) if len(diffs) > 0 else 0
            if mvpt > max_varrays_per_ts_polygon:
                max_varrays_per_ts_polygon = mvpt

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = n_tris
            header[4] = pool.prim_type_id
            header[5] = ts_offset
            header[6] = int32_offset  # ts_varrays_ps
            header[7] = int32_offset + n_ts  # varrays_ps
            header[8] = int32_offset + n_ts + n_varrays  # tri_ps
            header[9] = vert_offset
            header[10] = tri_offset
            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)
            header[11] = float_offset
            polygon_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.triangle_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            tris_padded = torch.zeros(n_tris, 4, dtype=torch.int32, device=device)
            tris_padded[:, :3] = pool.triangles.to(device, dtype=torch.int32)
            triangles_list.append(tris_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + 2 * n_varrays
            vert_offset += n_verts
            tri_offset += n_tris
            float_offset += len(aabbs)

        for pool in scene.cube_pools:
            n_global_ts = pool.timestamps_us.shape[0]
            n_cubes = pool.scales.shape[0]
            n_track_poses = pool.translations.shape[0]

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_cubes
            header[1] = n_global_ts
            header[2] = n_track_poses
            header[3] = pool.prim_type_id
            header[4] = ts_offset  # global timestamps
            header[5] = int32_offset  # cube_ts_ps
            header[6] = ts_offset + n_global_ts  # track timestamps
            header[7] = float_offset  # translations
            header[8] = float_offset + n_track_poses * 3  # quaternions
            header[9] = float_offset + n_track_poses * 7  # scales
            header[10] = float_offset + n_track_poses * 7 + n_cubes * 3  # colors
            header[11] = pool.render_flags
            cube_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            timestamps_list.append(pool.track_timestamps_us.to(device))
            int32_list.append(pool.cube_ts_prefix_sum.to(device, dtype=torch.int32))
            float_list.append(pool.translations.to(device).reshape(-1))
            float_list.append(pool.quaternions.to(device).reshape(-1))
            float_list.append(pool.scales.to(device).reshape(-1))
            float_list.append(pool.colors.to(device).reshape(-1))

            ts_offset += n_global_ts + n_track_poses
            int32_offset += n_cubes
            float_offset += n_track_poses * 7 + n_cubes * 9

        all_timestamps = (
            torch.cat(timestamps_list).contiguous()
            if timestamps_list
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        all_int32 = (
            torch.cat(int32_list).contiguous() if int32_list else torch.empty(0, dtype=torch.int32, device=device)
        )
        all_vertices = (
            torch.cat(vertices_list).contiguous()
            if vertices_list
            else torch.empty((0, 4), dtype=torch.float32, device=device)
        )
        all_triangles = (
            torch.cat(triangles_list).contiguous()
            if triangles_list
            else torch.empty((0, 4), dtype=torch.int32, device=device)
        )
        all_floats = (
            torch.cat(float_list).contiguous() if float_list else torch.empty(0, dtype=torch.float32, device=device)
        )

        all_pl_pools = (
            torch.stack(polyline_pool_headers).contiguous()
            if polyline_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )
        all_pg_pools = (
            torch.stack(polygon_pool_headers).contiguous()
            if polygon_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )
        all_cb_pools = (
            torch.stack(cube_pool_headers).contiguous()
            if cube_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )

        scene_id = len(self._scenes)
        self._scenes.append(
            {
                "timestamps": all_timestamps,
                "int32": all_int32,
                "vertices": all_vertices,
                "triangles": all_triangles,
                "floats": all_floats,
                "polyline_pools": all_pl_pools,
                "polygon_pools": all_pg_pools,
                "cube_pools": all_cb_pools,
                "max_varrays_per_ts_polyline": max_varrays_per_ts_polyline,
                "max_varrays_per_ts_polygon": max_varrays_per_ts_polygon,
            }
        )
        return scene_id

    # ------------------------------------------------------------------
    def set_tessellation_threshold(self, threshold: float) -> None:
        self._tessellation_threshold = threshold

    def set_depth_scaling(self, enabled: bool = True) -> None:
        self.cpp_wrapper.set_depth_scaling(1.0 if enabled else 0.0)

    def set_resolution_scale(
        self, width: int, height: int, reference_width: int = 1280, reference_height: int = 720
    ) -> None:
        scale = min(width / reference_width, height / reference_height)
        self.cpp_wrapper.set_resolution_scale(scale)

    def set_cull_radius(self, scale: float = 1.5) -> None:
        self.cpp_wrapper.set_cull_radius(scale)

    def set_msaa_samples(self, samples: int) -> None:
        self.cpp_wrapper.set_msaa_samples(samples)

    def set_line_widths(
        self,
        polyline_regular: float = 0.0,
        polyline_bev: float = 0.0,
        ego_traj_regular: float = 0.0,
        ego_traj_bev: float = 0.0,
        wireframe: float = 0.0,
    ) -> None:
        self.cpp_wrapper.set_line_widths(
            polyline_regular,
            polyline_bev,
            ego_traj_regular,
            ego_traj_bev,
            wireframe,
        )

    def set_max_tessellation_levels(
        self,
        polyline: int = 4,
        polygon: int = 3,
        cube: int = 3,
    ) -> None:
        self.cpp_wrapper.set_max_tessellation_levels(polyline, polygon, cube)

    def upload_color_palette(self, colors: dict) -> None:
        """Upload a custom color palette.

        ``colors`` maps ``prim_type_id`` (int) to an RGBA tuple
        ``(r, g, b, a)`` where each component is in [0, 255].
        """
        max_prim = max(colors.keys()) + 1 if colors else 0
        palette = torch.zeros(max_prim, dtype=torch.int32)
        for prim_id, rgba in colors.items():
            r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
            a = int(rgba[3]) if len(rgba) > 3 else 255
            packed = r | (g << 8) | (b << 16) | (a << 24)
            palette[prim_id] = packed
        self.cpp_wrapper.upload_color_palette(palette)

    def clear_scenes(self) -> None:
        self._scenes.clear()

    # ------------------------------------------------------------------
    def render_batch(
        self,
        queries: List[Tuple[int, int, int, int]],
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries (tuple-based API).

        Each query is ``(scene_id, camera_id, timestamp_us[, camera_type_id])``.
        Delegates to :meth:`render`.
        """
        device = camera_poses.device
        n = len(queries)
        scene_ids = torch.empty(n, dtype=torch.int32, device=device)
        camera_ids = torch.empty(n, dtype=torch.int32, device=device)
        timestamps_us = torch.empty(n, dtype=torch.int64, device=device)
        camera_type_ids = torch.empty(n, dtype=torch.int32, device=device)
        for i, q in enumerate(queries):
            scene_ids[i] = int(q[0])
            camera_ids[i] = int(q[1])
            ts = q[2]
            timestamps_us[i] = int(ts.item() if isinstance(ts, torch.Tensor) else ts)
            camera_type_ids[i] = int(q[3]) if len(q) > 3 else 0
        return self.render(scene_ids, camera_ids, timestamps_us, camera_type_ids, camera_poses, resolution)

    # ------------------------------------------------------------------
    def render(
        self,
        scene_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamps_us: torch.Tensor,
        camera_type_ids: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries.

        Currently only single-query rendering is supported (one scene, one
        camera at a time).  The API matches ``LudusTimestampedContext.render``
        so it can be used as a drop-in replacement.
        """
        assert self._camera_intrinsics is not None, "call upload_cameras first"

        n = scene_ids.shape[0]
        all_images = []

        plugin = _get_plugin(gl=False)

        for i in range(n):
            sid = int(scene_ids[i].item())
            cid = int(camera_ids[i].item())
            ts_val = int(timestamps_us[i].item())
            cam_type = int(camera_type_ids[i].item())

            sd = self._scenes[sid]
            cam_intrinsics = self._camera_intrinsics[cid : cid + 1].contiguous()
            pose = camera_poses[i : i + 1].contiguous()

            img = plugin.ludus_render_fwd_cuda_timestamped(
                self.cpp_wrapper,
                sd["timestamps"],
                sd["int32"],
                sd["vertices"],
                sd["triangles"],
                sd["floats"],
                sd["polyline_pools"],
                sd["polygon_pools"],
                sd["cube_pools"],
                ts_val,
                self._max_extrapolation_us,
                sd["max_varrays_per_ts_polyline"],
                sd["max_varrays_per_ts_polygon"],
                cam_type,
                cam_intrinsics,
                pose,
                resolution,
                self._tessellation_threshold,
            )
            all_images.append(img)

        return torch.cat(all_images, dim=0)


class LudusTimestampedContext:
    """GPU context for timestamped scene rendering.

    Supports loading multiple scenes once, then rendering hundreds of
    (scene_id, camera_id, timestamp) queries in a single batched call.
    """

    def __init__(self, device=None):
        """Create a new timestamped rendering context."""
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cpp_wrapper = _get_plugin(gl=True).LudusTimestampedStateWrapper(cuda_device_idx)
        self.cuda_device_idx = cuda_device_idx
        self._max_batch_size = self.cpp_wrapper.get_max_batch_size()
        self.needs_vflip = True  # OpenGL renders bottom-up
        self._scene_count = 0

        # Buffer offsets for multi-scene support
        self._global_ts_offset = 0
        self._global_int32_offset = 0
        self._global_vertex_offset = 0
        self._global_triangle_offset = 0
        self._global_pose_offset = 0
        self._global_float_offset = 0
        self._global_polyline_pool_offset = 0
        self._global_polygon_pool_offset = 0
        self._global_cube_pool_offset = 0

        # Streaming configuration
        self._jpeg_streaming_enabled = False
        self._jpeg_quality = 85
        self._stream_frame_count = 0

        # Video streaming
        self._video_streaming_enabled = False
        self._video_encoders = []
        self._video_files = []
        self._video_output_dir = None
        self._video_codec = "h264"
        self._video_bitrate = 10_000_000
        self._video_fps = 30
        self._video_preset = "P4"
        self._video_frame_count = 0
        self._video_width = 0
        self._video_height = 0
        self._video_num_cameras = 0
        self._video_streams = []
        self._video_encode_pool = None
        self._video_encode_futures = []
        self._video_encode_buffers = [None, None]

        # PNG worker pool
        self._png_pool = None
        self._png_futures = []
        self._png_compression = 6

    def upload_cameras(self, cameras: List[FThetaCamera]) -> None:
        """Upload camera intrinsics."""
        device = torch.device(f"cuda:{self.cuda_device_idx}")
        camera_intrinsics = _pack_cameras(cameras, device)
        self.cpp_wrapper.upload_cameras(camera_intrinsics.contiguous())
        self._num_cameras = len(cameras)

    def upload_color_palette(self, colors: dict) -> None:
        """Upload custom color palette for primitive types."""
        device = torch.device(f"cuda:{self.cuda_device_idx}")
        palette = torch.zeros((PRIM_TYPE_COUNT, 4), dtype=torch.float32, device=device)
        palette[:, 3] = -1.0  # Mark as "use default"

        for prim_type_id, rgb in colors.items():
            if 0 <= prim_type_id < PRIM_TYPE_COUNT:
                palette[prim_type_id, :3] = torch.tensor(rgb[:3], dtype=torch.float32)
                palette[prim_type_id, 3] = 1.0

        self.cpp_wrapper.upload_color_palette(palette.contiguous())

    def upload_scene(self, scene: TimestampedScene) -> int:
        """Upload a scene's timestamped data. Returns scene_id."""
        device = torch.device(f"cuda:{self.cuda_device_idx}")

        # Pack scene data into flat buffers
        timestamps_list = []
        int32_list = []
        vertices_list = []
        triangles_list = []
        poses_list = []
        float_list = []

        polyline_pool_headers = []
        polygon_pool_headers = []
        cube_pool_headers = []

        ts_offset = 0
        int32_offset = 0
        vert_offset = 0
        tri_offset = 0
        float_offset = 0

        global_ts_offset = self._global_ts_offset
        global_int32_offset = self._global_int32_offset
        global_vert_offset = self._global_vertex_offset
        global_tri_offset = self._global_triangle_offset
        global_float_offset = self._global_float_offset

        # Process polyline pools
        for pool in scene.polyline_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]

            header = torch.zeros(16, dtype=torch.uint32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = pool.prim_type_id
            header[4] = global_ts_offset + ts_offset
            header[5] = global_int32_offset + int32_offset
            header[6] = global_int32_offset + int32_offset + n_ts
            header[7] = global_vert_offset + vert_offset

            # Per-element AABBs for spatial culling
            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)
            header[8] = global_float_offset + float_offset  # aabb_offset

            polyline_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + n_varrays
            vert_offset += n_verts
            float_offset += len(aabbs)

        # Process polygon pools
        for pool in scene.polygon_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]
            n_tris = pool.triangles.shape[0]

            header = torch.zeros(16, dtype=torch.uint32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = n_tris
            header[4] = pool.prim_type_id
            header[5] = global_ts_offset + ts_offset
            header[6] = global_int32_offset + int32_offset
            header[7] = global_int32_offset + int32_offset + n_ts
            header[8] = global_int32_offset + int32_offset + n_ts + n_varrays
            header[9] = global_vert_offset + vert_offset
            header[10] = global_tri_offset + tri_offset

            # Per-element AABBs for spatial culling
            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)
            header[11] = global_float_offset + float_offset  # aabb_offset

            polygon_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.triangle_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            tris_padded = torch.zeros(n_tris, 4, dtype=torch.int32, device=device)
            tris_padded[:, :3] = pool.triangles.to(device, dtype=torch.int32)
            triangles_list.append(tris_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + 2 * n_varrays
            vert_offset += n_verts
            tri_offset += n_tris
            float_offset += len(aabbs)

        # Process cube pools
        max_cubes_in_pool = 0
        for pool in scene.cube_pools:
            n_global_ts = pool.timestamps_us.shape[0]
            n_cubes = pool.scales.shape[0]
            n_track_poses = pool.translations.shape[0]
            max_cubes_in_pool = max(max_cubes_in_pool, n_cubes)

            header = torch.zeros(16, dtype=torch.uint32, device=device)
            header[0] = n_cubes
            header[1] = n_global_ts
            header[2] = n_track_poses
            header[3] = pool.prim_type_id
            header[4] = global_ts_offset + ts_offset
            header[5] = global_int32_offset + int32_offset
            header[6] = global_ts_offset + ts_offset + n_global_ts
            header[7] = global_float_offset + float_offset
            header[8] = global_float_offset + float_offset + n_track_poses * 3
            header[9] = global_float_offset + float_offset + n_track_poses * 7
            header[10] = global_float_offset + float_offset + n_track_poses * 7 + n_cubes * 3
            header[11] = pool.render_flags

            cube_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            timestamps_list.append(pool.track_timestamps_us.to(device))
            int32_list.append(pool.cube_ts_prefix_sum.to(device, dtype=torch.int32))
            float_list.append(pool.translations.to(device).reshape(-1))
            float_list.append(pool.quaternions.to(device).reshape(-1))
            float_list.append(pool.scales.to(device).reshape(-1))
            float_list.append(pool.colors.to(device).reshape(-1))

            ts_offset += n_global_ts + n_track_poses
            int32_offset += n_cubes
            float_offset += n_track_poses * 7 + n_cubes * 9

        # Build scene descriptor
        scene_desc = torch.zeros(32, dtype=torch.uint32, device=device)
        scene_desc[0] = len(scene.polyline_pools)
        scene_desc[1] = self._global_polyline_pool_offset
        scene_desc[2] = len(scene.polygon_pools)
        scene_desc[3] = self._global_polygon_pool_offset
        scene_desc[4] = len(scene.cube_pools)
        scene_desc[5] = self._global_cube_pool_offset
        scene_desc[12] = 1  # valid flag

        # Concatenate data
        all_timestamps = (
            torch.cat(timestamps_list) if timestamps_list else torch.empty(0, dtype=torch.int64, device=device)
        )
        all_int32 = torch.cat(int32_list) if int32_list else torch.empty(0, dtype=torch.int32, device=device)
        all_vertices = (
            torch.cat(vertices_list) if vertices_list else torch.empty((0, 4), dtype=torch.float32, device=device)
        )
        all_triangles = (
            torch.cat(triangles_list) if triangles_list else torch.empty((0, 4), dtype=torch.int32, device=device)
        )
        all_poses = torch.cat(poses_list) if poses_list else torch.empty((0, 16), dtype=torch.float32, device=device)
        all_floats = torch.cat(float_list) if float_list else torch.empty(0, dtype=torch.float32, device=device)

        all_polyline_pools = (
            torch.stack(polyline_pool_headers)
            if polyline_pool_headers
            else torch.empty((0, 16), dtype=torch.uint32, device=device)
        )
        all_polygon_pools = (
            torch.stack(polygon_pool_headers)
            if polygon_pool_headers
            else torch.empty((0, 16), dtype=torch.uint32, device=device)
        )
        all_cube_pools = (
            torch.stack(cube_pool_headers)
            if cube_pool_headers
            else torch.empty((0, 16), dtype=torch.uint32, device=device)
        )

        scene_id = self.cpp_wrapper.upload_scene(
            scene_desc.view(torch.uint8).contiguous(),
            all_polyline_pools.view(torch.uint8).contiguous(),
            all_polygon_pools.view(torch.uint8).contiguous(),
            all_cube_pools.view(torch.uint8).contiguous(),
            max_cubes_in_pool,
            all_timestamps.contiguous(),
            all_int32.contiguous(),
            all_vertices.contiguous(),
            all_triangles.contiguous(),
            all_poses.contiguous(),
            all_floats.contiguous(),
        )

        # Update global offsets
        self._global_ts_offset += len(all_timestamps)
        self._global_int32_offset += len(all_int32)
        self._global_vertex_offset += all_vertices.shape[0]
        self._global_triangle_offset += all_triangles.shape[0]
        self._global_pose_offset += all_poses.shape[0]
        self._global_float_offset += len(all_floats)
        self._global_polyline_pool_offset += len(scene.polyline_pools)
        self._global_polygon_pool_offset += len(scene.polygon_pools)
        self._global_cube_pool_offset += len(scene.cube_pools)

        self._scene_count += 1
        return scene_id

    def preallocate_buffers(self, max_scenes: int, bytes_per_scene: int = 2 * 1024 * 1024) -> None:
        """Pre-allocate GL data buffers to avoid dynamic resizing.

        Args:
            max_scenes: Maximum number of scenes to budget for.
            bytes_per_scene: Estimated average bytes per scene (default 2 MB).
        """
        self.cpp_wrapper.preallocate_buffers(max_scenes, bytes_per_scene)

    def upload_scenes_batch(self, packed_list) -> list[int]:
        """Upload N scenes from pre-packed buffers with single map/unmap.

        Args:
            packed_list: list of (PackedSceneBuffers, TimestampedScene) pairs.
                Each TimestampedScene is used to count pools/cubes.

        Returns:
            list of scene_ids assigned to the uploaded scenes.
        """

        device = torch.device(f"cuda:{self.cuda_device_idx}")
        n = len(packed_list)
        if n == 0:
            return []

        all_ts: list[torch.Tensor] = []
        all_i32: list[torch.Tensor] = []
        all_verts: list[torch.Tensor] = []
        all_tris: list[torch.Tensor] = []
        all_floats: list[torch.Tensor] = []

        all_polyline_headers: list[torch.Tensor] = []
        all_polygon_headers: list[torch.Tensor] = []
        all_cube_headers: list[torch.Tensor] = []
        all_scene_descs: list[torch.Tensor] = []

        bounds_rows: list[list[int]] = []

        for packed, scene in packed_list:
            ts_data = packed.timestamps.to(device)
            i32_data = packed.int32_data.to(device)
            v_data = packed.vertices.to(device)
            t_data = packed.triangles.to(device)
            f_data = packed.float_data.to(device)

            num_poly_pools = len(packed.polyline_meta)
            num_gon_pools = len(packed.polygon_meta)
            num_cube_pools = len(packed.cube_meta)

            g_ts = self._global_ts_offset
            g_i32 = self._global_int32_offset
            g_vert = self._global_vertex_offset
            g_tri = self._global_triangle_offset
            g_float = self._global_float_offset

            # Build polyline pool headers from metadata
            polyline_headers = []
            for m in packed.polyline_meta:
                h = torch.zeros(16, dtype=torch.uint32, device=device)
                h[0] = m["n_ts"]
                h[1] = m["n_varrays"]
                h[2] = m["n_verts"]
                h[3] = m["prim_type_id"]
                h[4] = g_ts + m["ts_offset"]
                h[5] = g_i32 + m["int32_offset"]
                h[6] = g_i32 + m["int32_offset"] + m["n_ts"]
                h[7] = g_vert + m["vert_offset"]
                h[8] = g_float + m["float_offset"]
                polyline_headers.append(h)

            # Build polygon pool headers
            polygon_headers = []
            for m in packed.polygon_meta:
                h = torch.zeros(16, dtype=torch.uint32, device=device)
                h[0] = m["n_ts"]
                h[1] = m["n_varrays"]
                h[2] = m["n_verts"]
                h[3] = m["n_tris"]
                h[4] = m["prim_type_id"]
                h[5] = g_ts + m["ts_offset"]
                h[6] = g_i32 + m["int32_offset"]
                h[7] = g_i32 + m["int32_offset"] + m["n_ts"]
                h[8] = g_i32 + m["int32_offset"] + m["n_ts"] + m["n_varrays"]
                h[9] = g_vert + m["vert_offset"]
                h[10] = g_tri + m["tri_offset"]
                h[11] = g_float + m["float_offset"]
                polygon_headers.append(h)

            # Build cube pool headers
            max_cubes = 0
            cube_headers = []
            for m in packed.cube_meta:
                ntp = m["n_track_poses"]
                nc = m["n_cubes"]
                max_cubes = max(max_cubes, nc)
                h = torch.zeros(16, dtype=torch.uint32, device=device)
                h[0] = nc
                h[1] = m["n_global_ts"]
                h[2] = ntp
                h[3] = m["prim_type_id"]
                h[4] = g_ts + m["ts_offset"]
                h[5] = g_i32 + m["int32_offset"]
                h[6] = g_ts + m["ts_offset"] + m["n_global_ts"]
                fo = g_float + m["float_offset"]
                h[7] = fo
                h[8] = fo + ntp * 3
                h[9] = fo + ntp * 7
                h[10] = fo + ntp * 7 + nc * 3
                h[11] = m["render_flags"]
                cube_headers.append(h)

            # Scene descriptor
            sd = torch.zeros(32, dtype=torch.uint32, device=device)
            sd[0] = num_poly_pools
            sd[1] = self._global_polyline_pool_offset
            sd[2] = num_gon_pools
            sd[3] = self._global_polygon_pool_offset
            sd[4] = num_cube_pools
            sd[5] = self._global_cube_pool_offset
            sd[12] = 1  # valid
            all_scene_descs.append(sd)

            if polyline_headers:
                all_polyline_headers.append(torch.stack(polyline_headers))
            if polygon_headers:
                all_polygon_headers.append(torch.stack(polygon_headers))
            if cube_headers:
                all_cube_headers.append(torch.stack(cube_headers))

            all_ts.append(ts_data)
            all_i32.append(i32_data)
            all_verts.append(v_data)
            all_tris.append(t_data)
            all_floats.append(f_data)

            bounds_rows.append(
                [
                    num_poly_pools,
                    num_gon_pools,
                    num_cube_pools,
                    max_cubes,
                    len(ts_data),
                    len(i32_data),
                    v_data.shape[0],
                    t_data.shape[0],
                    0,  # poses (unused for now)
                    len(f_data),
                ]
            )

            # Advance global offsets
            self._global_ts_offset += len(ts_data)
            self._global_int32_offset += len(i32_data)
            self._global_vertex_offset += v_data.shape[0]
            self._global_triangle_offset += t_data.shape[0]
            self._global_float_offset += len(f_data)
            self._global_polyline_pool_offset += num_poly_pools
            self._global_polygon_pool_offset += num_gon_pools
            self._global_cube_pool_offset += num_cube_pools

        def _cat_or_empty(lst, shape, dtype):
            if lst:
                return torch.cat(lst)
            return torch.empty(shape, dtype=dtype, device=device)

        cat_sd = torch.stack(all_scene_descs).view(torch.uint8).contiguous()
        cat_poly = _cat_or_empty(all_polyline_headers, (0, 16), torch.uint32).view(torch.uint8).contiguous()
        cat_gon = _cat_or_empty(all_polygon_headers, (0, 16), torch.uint32).view(torch.uint8).contiguous()
        cat_cube = _cat_or_empty(all_cube_headers, (0, 16), torch.uint32).view(torch.uint8).contiguous()
        cat_ts = _cat_or_empty(all_ts, (0,), torch.int64).contiguous()
        cat_i32 = _cat_or_empty(all_i32, (0,), torch.int32).contiguous()
        cat_verts = _cat_or_empty(all_verts, (0, 4), torch.float32).contiguous()
        cat_tris = _cat_or_empty(all_tris, (0, 4), torch.int32).contiguous()
        cat_poses = torch.empty((0, 16), dtype=torch.float32, device=device).contiguous()
        cat_floats = _cat_or_empty(all_floats, (0,), torch.float32).contiguous()

        bounds_t = torch.tensor(bounds_rows, dtype=torch.int32).contiguous()

        first_id = self.cpp_wrapper.upload_scenes_batch(
            cat_sd,
            cat_poly,
            cat_gon,
            cat_cube,
            bounds_t,
            cat_ts,
            cat_i32,
            cat_verts,
            cat_tris,
            cat_poses,
            cat_floats,
        )

        ids = list(range(first_id, first_id + n))
        self._scene_count += n
        return ids

    def set_tessellation_threshold(self, threshold: float) -> None:
        """Set pixel error threshold for adaptive tessellation."""
        self.cpp_wrapper.set_tessellation_threshold(threshold)

    def set_max_tessellation_levels(
        self,
        polyline: Optional[int] = None,
        polygon: Optional[int] = None,
        cube: Optional[int] = None,
    ) -> None:
        """Set max tessellation levels (cap on adaptive subdivision).

        Args:
            polyline: Max level for polylines (0..4). None = leave unchanged.
            polygon: Max level for polygons (0..3). None = leave unchanged.
            cube: Max level for cube edges (0..3). None = leave unchanged.
        """
        # Get current values from backend by passing -1 to mean "no change" - but our C++ API
        # doesn't support that. So we require all three to be passed when calling from Python
        # if user wants to change only one, they could call with the defaults (4, 3, 3).
        # Simpler: accept optional and only call C++ when at least one is not None.
        # We need to pass three ints. So: if None, use backend default (4, 3, 3).
        p = 4 if polyline is None else polyline
        g = 3 if polygon is None else polygon
        c = 3 if cube is None else cube
        self.cpp_wrapper.set_max_tessellation_levels(p, g, c)

    def set_line_widths(
        self,
        polyline_regular: float = 0.0,
        polyline_bev: float = 0.0,
        ego_traj_regular: float = 0.0,
        ego_traj_bev: float = 0.0,
        wireframe: float = 0.0,
    ) -> None:
        """Set line widths in pixels."""
        self.cpp_wrapper.set_line_widths(polyline_regular, polyline_bev, ego_traj_regular, ego_traj_bev, wireframe)

    def set_resolution_scale(
        self, width: int, height: int, reference_width: int = 1280, reference_height: int = 720
    ) -> None:
        """Set resolution scale factor."""
        scale = min(width / reference_width, height / reference_height)
        self.cpp_wrapper.set_resolution_scale(scale)

    def set_depth_scaling(self, enabled: bool = True) -> None:
        """Enable or disable distance-based scaling effects."""
        self.cpp_wrapper.set_depth_scaling(1.0 if enabled else 0.0)

    def set_cull_radius(self, scale: float = 1.5) -> None:
        """Set spatial culling radius as a multiplier on ``depth_max``.

        The actual cull radius per query is ``cam.depth_max * scale``.
        Elements beyond this radius from the camera are discarded in the
        task shader before any mesh work is generated.

        Args:
            scale: Multiplier on the camera's ``depth_max``.
                ``0`` disables culling.  Default ``1.5`` gives generous
                headroom so nothing at the visible boundary pops.
        """
        self.cpp_wrapper.set_cull_radius(scale)

    def set_msaa_samples(self, samples: int) -> None:
        """Set MSAA sample count for antialiasing.

        Args:
            samples: Number of samples (0=disabled, 2, 4, or 8)
        """
        self.cpp_wrapper.set_msaa_samples(samples)

    @property
    def max_batch_size(self) -> int:
        """Max queries per render call, from GL_MAX_ARRAY_TEXTURE_LAYERS."""
        return self._max_batch_size

    def render_batch(
        self,
        queries: List[Tuple[int, int, int, int]],
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries.

        Camera poses must be in FLU convention (Forward-Left-Up).
        The C++ layer handles internal FLU-to-RDF conversion.
        """
        device = camera_poses.device
        n_queries = len(queries)

        queries_bytes = bytearray(n_queries * 32)
        for i, query in enumerate(queries):
            scene_id, camera_id, timestamp_us = query[:3]
            camera_type_id = query[3] if len(query) > 3 else CAMERA_TYPE_REGULAR

            if isinstance(timestamp_us, torch.Tensor):
                timestamp_us = timestamp_us.item()
            if isinstance(scene_id, torch.Tensor):
                scene_id = scene_id.item()
            if isinstance(camera_id, torch.Tensor):
                camera_id = camera_id.item()
            if isinstance(camera_type_id, torch.Tensor):
                camera_type_id = camera_type_id.item()

            struct.pack_into(
                "<IIqIIII",
                queries_bytes,
                i * 32,
                int(scene_id),
                int(camera_id),
                int(timestamp_us),
                int(camera_type_id),
                0,
                0,
                0,
            )

        queries_tensor = (
            torch.frombuffer(queries_bytes, dtype=torch.uint8).reshape(n_queries, 32).to(device).contiguous()
        )

        return self.render_batch_tensor(queries_tensor, camera_poses, resolution)

    def pack_queries_fast(
        self,
        scene_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamps_us: torch.Tensor,
        camera_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pack queries into GPU tensor using vectorized operations."""
        device = timestamps_us.device
        n = timestamps_us.shape[0]

        queries = torch.zeros((n, 8), dtype=torch.int32, device=device)
        queries[:, 0] = scene_ids.to(torch.int32)
        queries[:, 1] = camera_ids.to(torch.int32)

        ts = timestamps_us.to(torch.int64)
        queries[:, 2] = (ts & 0xFFFFFFFF).to(torch.int32)
        queries[:, 3] = (ts >> 32).to(torch.int32)
        queries[:, 4] = camera_type_ids.to(torch.int32)

        return queries.view(torch.uint8).reshape(n, 32).contiguous()

    def render_batch_tensor(
        self,
        queries_tensor: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch using pre-packed query tensor.

        Camera poses must be in FLU convention (Forward-Left-Up).
        Automatically splits into sub-batches if the number of queries
        exceeds GL_MAX_ARRAY_TEXTURE_LAYERS.
        """
        n = queries_tensor.shape[0]
        limit = self._max_batch_size
        plugin = _get_plugin(gl=True)

        if n <= limit:
            return plugin.ludus_timestamped_render_batch(
                self.cpp_wrapper,
                queries_tensor.contiguous(),
                camera_poses.contiguous(),
                resolution,
            )

        parts = []
        for start in range(0, n, limit):
            end = min(start + limit, n)
            part = plugin.ludus_timestamped_render_batch(
                self.cpp_wrapper,
                queries_tensor[start:end].contiguous(),
                camera_poses[start:end].contiguous(),
                resolution,
            )
            parts.append(part)
        return torch.cat(parts, dim=0)

    def render(
        self,
        scene_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamps_us: torch.Tensor,
        camera_type_ids: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Fast render batch using tensor inputs."""
        queries_tensor = self.pack_queries_fast(scene_ids, camera_ids, timestamps_us, camera_type_ids)
        return self.render_batch_tensor(queries_tensor, camera_poses, resolution)

    # ========== Streaming Methods ==========

    def set_jpeg_streaming(self, enabled: bool = True, quality: int = None) -> None:
        """Enable or disable JPEG streaming mode."""
        self._jpeg_streaming_enabled = enabled
        self._stream_frame_count = 0
        if quality is not None:
            self._jpeg_quality = max(1, min(100, quality))

    def set_jpeg_quality(self, quality: int) -> None:
        """Set the JPEG quality for streaming."""
        self._jpeg_quality = max(1, min(100, quality))

    def get_jpeg_quality(self) -> int:
        """Get the current JPEG quality setting."""
        return self._jpeg_quality

    def is_jpeg_streaming_enabled(self) -> bool:
        """Check if JPEG streaming mode is enabled."""
        return self._jpeg_streaming_enabled

    def reset_streaming(self) -> None:
        """Reset the streaming session."""
        self._stream_frame_count = 0

    def render_batch_to_staging(
        self,
        queries_tensor: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[int, bool]:
        """Render batch to staging buffer.

        Camera poses must be in FLU convention (Forward-Left-Up).
        """
        return _get_plugin(gl=True).ludus_timestamped_render_to_staging(
            self.cpp_wrapper, queries_tensor.contiguous(), camera_poses.contiguous(), resolution
        )

    def get_staging_data(self, staging_idx: int) -> torch.Tensor:
        """Get rendered data from staging buffer."""
        return _get_plugin(gl=True).ludus_timestamped_get_staging_data(self.cpp_wrapper, staging_idx)

    def get_staging_data_async(self, staging_idx: int, stream: torch.cuda.Stream) -> torch.Tensor:
        """Get staging buffer as zero-copy view with GPU-side sync."""
        return _get_plugin(gl=True).ludus_timestamped_get_staging_data_async(
            self.cpp_wrapper, staging_idx, stream.cuda_stream
        )

    def stream_batch(
        self,
        queries_tensor: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
        quality: int = None,
    ) -> Tuple[int, Optional[Union[torch.Tensor, list]]]:
        """Unified streaming API."""
        staging_idx, has_prev = self.render_batch_to_staging(queries_tensor, camera_poses, resolution)
        self._stream_frame_count += 1

        if self._video_streaming_enabled:
            if has_prev and self._stream_frame_count > 1:
                prev_staging_idx = 1 - staging_idx
                if self._video_streams:
                    encode_stream = self._video_streams[0]
                    prev_tensor = self.get_staging_data_async(prev_staging_idx, encode_stream)
                    self.encode_video_frame(prev_tensor)
                else:
                    prev_tensor = self.get_staging_data(prev_staging_idx)
                    self.encode_video_frame(prev_tensor)
            return staging_idx, None

        prev_data = None
        if has_prev and self._stream_frame_count > 1:
            prev_staging_idx = 1 - staging_idx
            prev_data = self.get_stream_data(prev_staging_idx, quality)

        return staging_idx, prev_data

    def get_stream_data(
        self,
        staging_idx: int,
        quality: int = None,
    ) -> Optional[Union[torch.Tensor, list]]:
        """Get data from staging buffer in current streaming mode."""
        if self._video_streaming_enabled:
            final_tensor = self.get_staging_data(staging_idx)
            self.encode_video_frame(final_tensor)
            return None
        elif self._jpeg_streaming_enabled:
            q = quality if quality is not None else self._jpeg_quality
            return self.encode_jpeg_batch_staging(staging_idx, q)
        else:
            return self.get_staging_data(staging_idx)

    # ========== NVJPEG Encoding ==========

    def is_nvjpeg_available(self) -> bool:
        """Check if NVJPEG hardware encoder is available."""
        return _get_plugin(gl=True).ludus_timestamped_is_nvjpeg_available(self.cpp_wrapper)

    def encode_jpeg_staging(self, staging_idx: int, image_idx: int, quality: int = None) -> bytes:
        """Encode a single image from staging buffer to JPEG."""
        q = quality if quality is not None else self._jpeg_quality
        return _get_plugin(gl=True).ludus_timestamped_encode_jpeg_staging(self.cpp_wrapper, staging_idx, image_idx, q)

    def encode_jpeg_batch_staging(self, staging_idx: int, quality: int = None) -> list:
        """Encode all images from staging buffer to JPEG."""
        q = quality if quality is not None else self._jpeg_quality
        return _get_plugin(gl=True).ludus_timestamped_encode_jpeg_batch_staging(self.cpp_wrapper, staging_idx, q)

    # ========== Async Host Transfer ==========

    def start_async_host_transfer(self, staging_idx: int) -> int:
        """Start async D2H transfer from staging buffer."""
        return _get_plugin(gl=True).ludus_timestamped_start_async_host_transfer(self.cpp_wrapper, staging_idx)

    def is_pinned_buffer_ready(self, pinned_idx: int) -> bool:
        """Check if a specific pinned buffer is ready."""
        return _get_plugin(gl=True).ludus_timestamped_is_pinned_buffer_ready(self.cpp_wrapper, pinned_idx)

    def wait_pinned_buffer_view(self, pinned_idx: int) -> Optional[torch.Tensor]:
        """Wait for a specific pinned buffer and get zero-copy view."""
        result = _get_plugin(gl=True).ludus_timestamped_wait_pinned_buffer_view(self.cpp_wrapper, pinned_idx)
        if result is None or result.numel() == 0:
            return None
        return result

    def wait_host_transfer(self) -> Optional[torch.Tensor]:
        """Wait for async host transfer to complete."""
        result = _get_plugin(gl=True).ludus_timestamped_wait_host_transfer(self.cpp_wrapper)
        if result is None or result.numel() == 0:
            return None
        return result

    # ========== Video Streaming ==========

    def set_video_streaming(
        self,
        output_dir: str,
        camera_names: List[str] = None,
        codec: str = "h264",
        bitrate: int = 10_000_000,
        fps: int = 30,
        preset: str = "P4",
        resolution: Tuple[int, int] = None,
    ) -> bool:
        """Enable video streaming mode."""
        try:
            import PyNvVideoCodec as nvc  # noqa: F401
        except ImportError:
            print("PyNvVideoCodec not installed.")
            return False

        import os

        os.makedirs(output_dir, exist_ok=True)

        self._video_output_dir = output_dir
        self._video_camera_names = camera_names or []
        self._video_codec = codec
        self._video_bitrate = bitrate
        self._video_fps = fps
        self._video_preset = preset
        self._video_streaming_enabled = True
        self._video_encoders = []
        self._video_files = []
        self._video_frame_count = 0
        self._jpeg_streaming_enabled = False
        self._stream_frame_count = 0

        return True

    def encode_video_frame(self, gpu_tensor: torch.Tensor, async_write: bool = True) -> bool:
        """Encode GPU tensor batch to video files."""
        if not self._video_streaming_enabled:
            return False

        if gpu_tensor.dim() == 3:
            gpu_tensor = gpu_tensor.unsqueeze(0)

        n_cameras = gpu_tensor.shape[0]
        h, w = gpu_tensor.shape[1], gpu_tensor.shape[2]

        if not self._video_encoders:
            if not self._init_video_encoders(w, h, n_cameras):
                return False

        flipped = gpu_tensor.flip(1).contiguous()
        for i in range(n_cameras):
            bs = self._video_encoders[i].Encode(flipped[i])
            if bs:
                self._video_files[i].write(bs)

        self._video_frame_count += 1
        return True

    def _init_video_encoders(self, width: int, height: int, num_cameras: int) -> bool:
        """Initialize video encoders."""
        try:
            import PyNvVideoCodec as nvc
        except ImportError:
            return False

        import os

        for i in range(num_cameras):
            encoder = nvc.CreateEncoder(
                width=width,
                height=height,
                fmt="ABGR",
                usecpuinputbuffer=False,
                codec=self._video_codec,
                bitrate=self._video_bitrate,
                fps=self._video_fps,
                preset=self._video_preset,
            )
            self._video_encoders.append(encoder)

            if self._video_camera_names and i < len(self._video_camera_names):
                filename = f"{self._video_camera_names[i]}.mp4"
            else:
                filename = f"cam_{i:02d}.mp4"
            filepath = os.path.join(self._video_output_dir, filename)
            self._video_files.append(open(filepath, "wb"))

        return True

    def finalize_video(self) -> List[str]:
        """Finalize and close all video files."""
        if not self._video_streaming_enabled or not self._video_encoders:
            return []

        import os

        output_paths = []

        for i, (encoder, f) in enumerate(zip(self._video_encoders, self._video_files)):
            bitstream = encoder.EndEncode()
            if bitstream:
                f.write(bitstream)
            f.close()

            if self._video_camera_names and i < len(self._video_camera_names):
                filename = f"{self._video_camera_names[i]}.mp4"
            else:
                filename = f"cam_{i:02d}.mp4"
            output_paths.append(os.path.join(self._video_output_dir, filename))

        self._video_encoders = []
        self._video_files = []
        self._video_streaming_enabled = False
        self._video_frame_count = 0

        return output_paths

    def is_video_streaming_enabled(self) -> bool:
        """Check if video streaming mode is enabled."""
        return self._video_streaming_enabled

    def remove_scene(self, scene_id: int) -> None:
        """Tombstone a single scene (set valid=0). Data stays in GPU buffers
        but the scene is skipped during rendering. Use clear_scenes() to
        fully reclaim all buffer space."""
        self.cpp_wrapper.remove_scene(scene_id)

    def clear_scenes(self) -> None:
        """Clear all loaded scenes (active buffer set)."""
        self.cpp_wrapper.clear_scenes()
        self._scene_count = 0
        self._global_ts_offset = 0
        self._global_int32_offset = 0
        self._global_vertex_offset = 0
        self._global_triangle_offset = 0
        self._global_pose_offset = 0
        self._global_float_offset = 0
        self._global_polyline_pool_offset = 0
        self._global_polygon_pool_offset = 0
        self._global_cube_pool_offset = 0

    def swap_buffer_sets(self) -> None:
        """Swap front/back GL buffer sets (waits for back set fence)."""
        self.cpp_wrapper.swap_buffer_sets()
