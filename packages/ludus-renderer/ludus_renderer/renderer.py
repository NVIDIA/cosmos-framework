# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
LudusRenderer: High-level GPU-native renderer for Av2 scenes.

This is the main entry point for rendering Av2 scenes using the GPU-native
timestamped rendering pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ._ops import FThetaCamera, LudusTimestampedContext
from .clipgt import ClipgtGpuScene


class LudusRenderer:
    """High-performance GPU-native renderer for Av2 scenes.

    This renderer uses mesh shaders for procedural geometry generation,
    native F-theta fisheye projection, and adaptive tessellation.

    Features:
    - Multi-scene support: load multiple scenes, render any combination
    - Batch rendering: render many camera/timestamp combinations in one call
    - Adaptive tessellation: automatic subdivision for curved fisheye lines
    - Zero mesh building: geometry is generated procedurally in shaders

    Example:
        renderer = LudusRenderer(width=1280, height=720)

        # Load and upload scenes
        scene1 = load_av2_scene("/path/to/scene1")
        renderer.upload_scene(scene1)

        # Render batch
        queries = [
            (scene1.scene_id, camera_id, timestamp_us),
            ...
        ]
        poses = scene1.get_ego_poses_at_timestamps(timestamps, camera_names)
        images = renderer.render_batch(queries, poses)
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        device: Union[str, torch.device] = "cuda",
        tessellation_threshold: float = 1.0,
        msaa_samples: int = 0,
        max_tessellation_level_polyline: Optional[int] = None,
        max_tessellation_level_polygon: Optional[int] = None,
        max_tessellation_level_cube: Optional[int] = None,
    ):
        """Initialize the renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            device: CUDA device to use
            tessellation_threshold: Pixel error threshold for adaptive tessellation.
                                   Lower = more subdivision. 0 = disabled.
            msaa_samples: MSAA sample count (0=disabled, 2, 4, or 8)
            max_tessellation_level_polyline: Cap for polyline subdivision (0..4). None = default 4.
            max_tessellation_level_polygon: Cap for polygon subdivision (0..3). None = default 3.
            max_tessellation_level_cube: Cap for cube edge subdivision (0..3). None = default 3.
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.width = width
        self.height = height
        self.device = device

        # Create the underlying GPU context
        self._ctx = LudusTimestampedContext(device=device)
        self._ctx.set_tessellation_threshold(tessellation_threshold)

        # Set resolution scale for proper line widths at different resolutions
        # Reference resolution is 1280x720 - line widths are scaled proportionally
        self._ctx.set_resolution_scale(width, height)

        # Enable depth-based effects (fog, distance-based line width scaling)
        self._ctx.set_depth_scaling(enabled=True)

        if msaa_samples > 0:
            self._ctx.set_msaa_samples(msaa_samples)
        if (
            max_tessellation_level_polyline is not None
            or max_tessellation_level_polygon is not None
            or max_tessellation_level_cube is not None
        ):
            self._ctx.set_max_tessellation_levels(
                polyline=max_tessellation_level_polyline,
                polygon=max_tessellation_level_polygon,
                cube=max_tessellation_level_cube,
            )

        # Track uploaded scenes and cameras
        self._scenes: Dict[int, ClipgtGpuScene] = {}
        self._cameras_uploaded = False
        self._all_cameras: List[FThetaCamera] = []
        self._camera_id_offset: Dict[int, int] = {}  # scene_id -> camera_id offset

        self._next_scene_id = 0

    def set_tessellation_threshold(self, threshold: float):
        """Set the tessellation threshold.

        Args:
            threshold: Pixel error threshold. Lower = more subdivision. 0 = disabled.
        """
        self._ctx.set_tessellation_threshold(threshold)

    def set_msaa_samples(self, samples: int):
        """Set the MSAA sample count for antialiasing.

        Args:
            samples: Number of samples (0=disabled, 2, 4, or 8)
        """
        self._ctx.set_msaa_samples(samples)

    def set_max_tessellation_levels(
        self,
        polyline: Optional[int] = None,
        polygon: Optional[int] = None,
        cube: Optional[int] = None,
    ):
        """Set max tessellation levels (cap on adaptive subdivision).

        Args:
            polyline: Max level for polylines (0..4). None = leave unchanged.
            polygon: Max level for polygons (0..3). None = leave unchanged.
            cube: Max level for cube edges (0..3). None = leave unchanged.
        """
        self._ctx.set_max_tessellation_levels(polyline=polyline, polygon=polygon, cube=cube)

    def upload_scene(self, scene: ClipgtGpuScene) -> int:
        """Upload a scene to the GPU for rendering.

        Args:
            scene: ClipgtGpuScene to upload

        Returns:
            scene_id: ID to use when creating render queries
        """
        # Upload cameras if this is the first scene or scene has new cameras
        camera_offset = len(self._all_cameras)
        self._all_cameras.extend(scene.cameras)
        self._ctx.upload_cameras(self._all_cameras)

        # Store camera offset for this scene
        scene_id = self._next_scene_id
        self._camera_id_offset[scene_id] = camera_offset
        self._next_scene_id += 1

        # Upload the scene
        uploaded_id = self._ctx.upload_scene(scene.timestamped_scene)

        # Verify IDs match
        assert uploaded_id == scene_id, f"Scene ID mismatch: {uploaded_id} != {scene_id}"

        # Store reference and update scene
        scene.scene_id = scene_id
        self._scenes[scene_id] = scene

        return scene_id

    def get_scene(self, scene_id: int) -> ClipgtGpuScene:
        """Get a previously uploaded scene by ID."""
        return self._scenes[scene_id]

    def render_batch(
        self,
        queries: List[Tuple[int, int, int]],
        camera_poses: Tensor,
    ) -> Tensor:
        """Render a batch of scene/camera/timestamp combinations.

        Args:
            queries: List of (scene_id, camera_id, timestamp_us) tuples.
                    camera_id is the index into the scene's camera list.
            camera_poses: World-to-camera transform matrices [n_queries, 4, 4]

        Returns:
            Rendered images [n_queries, height, width, 4] as RGBA float32
        """
        n_queries = len(queries)
        assert camera_poses.shape == (n_queries, 4, 4), (
            f"camera_poses shape mismatch: {camera_poses.shape} vs ({n_queries}, 4, 4)"
        )

        # Adjust camera_id to global camera index
        adjusted_queries = []
        for scene_id, local_camera_id, timestamp_us in queries:
            global_camera_id = self._camera_id_offset[scene_id] + local_camera_id
            adjusted_queries.append((scene_id, global_camera_id, timestamp_us))

        # Render
        return self._ctx.render_batch(
            adjusted_queries,
            camera_poses.to(self.device),
            resolution=(self.height, self.width),
        )

    def render_scene(
        self,
        scene: ClipgtGpuScene,
        camera_names: List[str],
        timestamps_us: Tensor,
    ) -> Tensor:
        """Convenience method to render a scene at multiple cameras and timestamps.

        This computes camera poses automatically from ego tracks.

        Args:
            scene: Uploaded ClipgtGpuScene
            camera_names: List of camera names to render
            timestamps_us: Timestamps to render at [n_timestamps]

        Returns:
            Rendered images [n_timestamps * n_cameras, height, width, 4]
        """
        if scene.scene_id is None:
            raise ValueError("Scene must be uploaded first")

        timestamps_us = timestamps_us.to(self.device)
        n_timestamps = timestamps_us.shape[0]
        n_cameras = len(camera_names)

        # Get camera IDs
        camera_ids = scene.get_camera_ids(camera_names)

        # Compute camera poses
        poses = scene.get_ego_poses_at_timestamps(timestamps_us, camera_names)

        # Build queries
        queries = []
        for t in range(n_timestamps):
            for c, cam_id in enumerate(camera_ids):
                queries.append((scene.scene_id, cam_id, timestamps_us[t].item()))

        return self.render_batch(queries, poses)

    def render_single(
        self,
        scene: ClipgtGpuScene,
        camera_name: str,
        timestamp_us: int,
        camera_pose: Optional[Tensor] = None,
    ) -> Tensor:
        """Convenience method to render a single image.

        Args:
            scene: Uploaded ClipgtGpuScene
            camera_name: Camera to render from
            timestamp_us: Timestamp to render at
            camera_pose: Optional world-to-camera pose. If None, computed from ego.

        Returns:
            Rendered image [height, width, 4]
        """
        if scene.scene_id is None:
            raise ValueError("Scene must be uploaded first")

        if camera_pose is None:
            ts = torch.tensor([timestamp_us], dtype=torch.int64, device=self.device)
            camera_pose = scene.get_ego_poses_at_timestamps(ts, [camera_name])
        else:
            camera_pose = camera_pose.unsqueeze(0)

        camera_id = scene.get_camera_id(camera_name)
        queries = [(scene.scene_id, camera_id, timestamp_us)]

        result = self.render_batch(queries, camera_pose)
        return result[0]  # Remove batch dimension

    def __del__(self):
        """Cleanup GPU resources."""
        # LudusTimestampedContext handles cleanup in its destructor
        pass
