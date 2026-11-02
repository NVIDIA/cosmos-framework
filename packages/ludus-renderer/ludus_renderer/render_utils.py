# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Rendering utilities for ludus_renderer.

This module provides reusable utilities for:
- Color palettes and schemes
- Polyline pattern generation (dashes, dots)
- Camera creation and pose computation
- Scene adapters for legacy interfaces
- Batch rendering helpers
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# =============================================================================
# Color schemes
# =============================================================================

COLORS_V3 = {
    "lane_line": [98, 183, 249],
    "lane_boundary": [98, 183, 249],
    "poles": [183, 69, 177],
    "road_boundary": [253, 1, 232],
    "wait_line": [108, 179, 59],
    "crosswalk": [139, 93, 255],
    "road_marking": [20, 254, 185],
    "traffic_sign": [8, 2, 255],
    "traffic_light": [100, 100, 100],
    "intersection_area": [80, 80, 120],
    "road_island": [60, 120, 60],
    "ego_trajectory": [0, 255, 0],
}


def build_color_palette_v3():
    """Build a color palette dict for upload_color_palette() using v3 colors."""
    from ludus_renderer.torch.ops import (
        PRIM_CROSSWALK,
        PRIM_DOT_WHITE,
        PRIM_DOT_YELLOW,
        PRIM_EGO_TRAJECTORY,
        PRIM_INTERSECTION,
        PRIM_LANE_BOUNDARY,
        PRIM_LANE_LINE,
        PRIM_LANE_LINE_WHITE_DASHED,
        PRIM_LANE_LINE_WHITE_SOLID,
        PRIM_LANE_LINE_YELLOW_DASHED,
        PRIM_LANE_LINE_YELLOW_SOLID,
        PRIM_POLE,
        PRIM_ROAD_BOUNDARY,
        PRIM_ROAD_ISLAND,
        PRIM_ROAD_MARKING,
        PRIM_TRAFFIC_LIGHT,
        PRIM_TRAFFIC_SIGN,
        PRIM_WAIT_LINE,
    )

    palette = {}
    name_to_prim = {
        "road_boundary": PRIM_ROAD_BOUNDARY,
        "lane_line": PRIM_LANE_LINE,
        "lane_boundary": PRIM_LANE_BOUNDARY,
        "crosswalk": PRIM_CROSSWALK,
        "wait_line": PRIM_WAIT_LINE,
        "poles": PRIM_POLE,
        "road_marking": PRIM_ROAD_MARKING,
        "traffic_sign": PRIM_TRAFFIC_SIGN,
        "traffic_light": PRIM_TRAFFIC_LIGHT,
        "intersection_area": PRIM_INTERSECTION,
        "road_island": PRIM_ROAD_ISLAND,
        "ego_trajectory": PRIM_EGO_TRAJECTORY,
    }

    for name, rgb in COLORS_V3.items():
        if name in name_to_prim:
            prim_id = name_to_prim[name]
            palette[prim_id] = [c / 255.0 for c in rgb]

    # Lane line variants
    white = [1.0, 1.0, 1.0]
    yellow = [1.0, 0.85, 0.0]

    palette[PRIM_LANE_LINE_WHITE_SOLID] = white
    palette[PRIM_LANE_LINE_WHITE_DASHED] = white
    palette[PRIM_LANE_LINE_YELLOW_SOLID] = yellow
    palette[PRIM_LANE_LINE_YELLOW_DASHED] = yellow
    palette[PRIM_DOT_YELLOW] = yellow
    palette[PRIM_DOT_WHITE] = white

    return palette


# =============================================================================
# Scene adapters
# =============================================================================


class EgoTracksAdapter:
    """Adapter to make ClipgtGpuScene.ego_track look like legacy ego_tracks interface."""

    def __init__(self, ego_track):
        self._ego_track = ego_track

    @property
    def timestamps(self):
        return self._ego_track.timestamps

    @property
    def poses(self):
        return self._ego_track.translations

    def get_transforms_at_timestamp(self, timestamps_us):
        """Compute ego-to-world transforms at given timestamps."""
        from scipy.spatial.transform import Rotation as R

        timestamps_us = timestamps_us.cpu()
        n_ts = len(timestamps_us)
        device = self._ego_track.translations.device

        transforms = torch.zeros(n_ts, 1, 4, 4, dtype=torch.float32, device=device)

        for i, ts in enumerate(timestamps_us):
            ts_val = ts.item()
            ego_ts = self._ego_track.timestamps.cpu()

            idx = torch.searchsorted(ego_ts, ts_val)
            if idx == 0:
                idx = 1
            elif idx >= len(ego_ts):
                idx = len(ego_ts) - 1

            t0, t1 = ego_ts[idx - 1].item(), ego_ts[idx].item()
            alpha = 0.0 if t1 == t0 else (ts_val - t0) / (t1 - t0)

            trans0 = self._ego_track.translations[idx - 1].cpu().numpy()
            trans1 = self._ego_track.translations[idx].cpu().numpy()
            trans = trans0 * (1 - alpha) + trans1 * alpha

            quat0 = self._ego_track.quaternions[idx - 1].cpu().numpy()
            quat1 = self._ego_track.quaternions[idx].cpu().numpy()
            quat = quat0 * (1 - alpha) + quat1 * alpha
            quat = quat / np.linalg.norm(quat)

            rot = R.from_quat(quat).as_matrix()
            transform = torch.eye(4, dtype=torch.float32)
            transform[:3, :3] = torch.from_numpy(rot.astype(np.float32))
            transform[:3, 3] = torch.from_numpy(trans.astype(np.float32))
            transforms[i, 0] = transform.to(device)

        return transforms


class SceneAdapter:
    """Adapter to make ClipgtGpuScene compatible with legacy test interface."""

    def __init__(self, clipgt_scene):
        self._scene = clipgt_scene
        self.timestamped_scene = clipgt_scene.timestamped_scene
        self.ego_tracks = EgoTracksAdapter(clipgt_scene.ego_track)
        self.obstacles = None
        self.cameras = {name: self._adapt_camera(name, i) for name, i in clipgt_scene.camera_name_to_id.items()}

    def _adapt_camera(self, name, idx):
        from dataclasses import dataclass

        @dataclass
        class LegacyCamera:
            intrinsics: object
            sensor_to_rig: torch.Tensor

        @dataclass
        class LegacyIntrinsics:
            cx: float
            cy: float
            width: int
            height: int
            poly: np.ndarray
            is_bw_poly: bool
            linear_cde: np.ndarray

        ftheta = self._scene.cameras[idx]
        intrinsics = LegacyIntrinsics(
            cx=ftheta.principal_point[0].item(),
            cy=ftheta.principal_point[1].item(),
            width=int(ftheta.image_size[0].item()),
            height=int(ftheta.image_size[1].item()),
            poly=ftheta.fw_poly.cpu().numpy(),
            is_bw_poly=False,
            linear_cde=np.array([1.0, 0.0, 0.0]),
        )

        sensor_to_rig = self._scene.sensor_to_rig.get(name, torch.eye(4))
        return LegacyCamera(intrinsics=intrinsics, sensor_to_rig=sensor_to_rig)


# =============================================================================
# Scene loading
# =============================================================================


def is_clipgt_directory(scene_path: str) -> bool:
    """Check if scene_path is a clipgt directory (has parquet files)."""
    if not os.path.isdir(scene_path):
        return False
    if os.path.isfile(os.path.join(scene_path, "road_boundary.parquet")):
        return True
    from pathlib import Path

    scene_dir = Path(scene_path)
    ego_files = list(scene_dir.glob("*.egomotion_estimate.parquet"))
    return len(ego_files) > 0


def is_av2_scene(scene_path: str) -> bool:
    """Check if a scene path is an AV2 tar (vs a clipgt directory)."""
    return scene_path.endswith(".tar")


def load_scene_adapted(
    scene_path: str,
    device: torch.device,
    include_ego_trajectory: bool = True,
    include_ego_obstacle: bool = False,
    use_gpu_decoder: Optional[bool] = None,
):
    """Load a scene (clipgt or AV2) and wrap with SceneAdapter for legacy interface."""
    if is_av2_scene(scene_path):
        from ludus_renderer import load_av2_scene

        raw_scene = load_av2_scene(
            scene_path,
            device=device,
            include_ego_trajectory=include_ego_trajectory,
            include_ego_obstacle=include_ego_obstacle,
            use_gpu_decoder=use_gpu_decoder,
        )
    else:
        from ludus_renderer import load_clipgt_scene

        raw_scene = load_clipgt_scene(
            scene_path,
            device=device,
            include_ego_trajectory=include_ego_trajectory,
            include_ego_obstacle=include_ego_obstacle,
        )
    return SceneAdapter(raw_scene)


# =============================================================================
# Camera utilities
# =============================================================================


def create_bev_camera(
    width: int,
    height: int,
    device: torch.device,
    bev_height: float = 80.0,
    fov_deg: float = 60.0,
    near: float = 1.0,
    far: float = 150.0,
):
    """Create a bird's eye view camera with perspective projection."""
    from ludus_renderer.torch import FThetaCamera

    cx, cy = width / 2.0, height / 2.0
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    focal = (height / 2.0) / math.tan(half_fov)

    diagonal_r = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)
    max_ray_angle = math.atan(diagonal_r / focal)

    # Taylor series for tan(α) -> pinhole projection
    poly_coeffs = torch.tensor(
        [
            0.0,
            focal,
            0.0,
            focal / 3.0,
            0.0,
            2.0 * focal / 15.0,
        ],
        device=device,
    )

    return FThetaCamera(
        principal_point=torch.tensor([cx, cy], device=device),
        image_size=torch.tensor([float(width), float(height)], device=device),
        fw_poly=poly_coeffs,
        max_ray_angle=max_ray_angle,
        depth_max=far,
    )


def get_bev_sensor_to_rig(bev_height: float, device: torch.device):
    """Get the sensor-to-rig transform for a BEV camera.

    Sensor (FLU): X=forward (optical axis), Y=left, Z=up
    Rig (FLU):    X=forward, Y=left, Z=up

    For BEV looking straight down with ego forward at top of image:
      Sensor X (depth)    -> Rig -Z (points down)
      Sensor Y (left)     -> Rig +Y (unchanged)
      Sensor Z (up image) -> Rig +X (forward)
    """
    return torch.tensor(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, bev_height],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def get_bev_camera_pose(scene, timestamp, bev_height: float, device: torch.device):
    """Compute world-to-camera matrix (FLU) for BEV at a single timestamp."""
    ego_to_world = scene.ego_tracks.get_transforms_at_timestamp(timestamp.unsqueeze(0))[0, 0]

    sensor_to_rig = get_bev_sensor_to_rig(bev_height, device)
    camera_to_world = ego_to_world @ sensor_to_rig
    return torch.linalg.inv(camera_to_world)


def get_all_bev_camera_poses(scene, timestamps, bev_height: float, device: torch.device):
    """Compute world-to-camera matrices (FLU) for BEV at all timestamps (batched)."""
    ego_to_world = scene.ego_tracks.get_transforms_at_timestamp(timestamps)
    sensor_to_rig = get_bev_sensor_to_rig(bev_height, device)
    camera_to_world = ego_to_world @ sensor_to_rig
    return torch.linalg.inv(camera_to_world)


def get_camera_pose(scene, timestamp, camera_name: str, device: torch.device):
    """Compute world-to-camera matrix (FLU) for a named camera at a single timestamp."""
    cam = scene.cameras[camera_name]
    ego_to_world = scene.ego_tracks.get_transforms_at_timestamp(timestamp.unsqueeze(0))[0, 0]

    sensor_to_rig = cam.sensor_to_rig.to(device)
    camera_to_world = ego_to_world @ sensor_to_rig
    return torch.linalg.inv(camera_to_world)


def get_all_camera_poses(scene, timestamps, camera_name: str, device: torch.device):
    """Compute world-to-camera matrices (FLU) for all timestamps (batched)."""
    cam = scene.cameras[camera_name]
    ego_to_world = scene.ego_tracks.get_transforms_at_timestamp(timestamps)

    sensor_to_rig = cam.sensor_to_rig.to(device)
    camera_to_world = ego_to_world @ sensor_to_rig
    return torch.linalg.inv(camera_to_world)


def get_scene_camera(scene, camera_name: str, target_width: int, target_height: int):
    """Get a camera from the scene by name, with intrinsics scaled to the target resolution.

    Args:
        scene: SceneAdapter wrapping a ClipgtGpuScene
        camera_name: Name of the camera to retrieve
        target_width: Target image width
        target_height: Target image height

    Returns:
        FThetaCamera with intrinsics scaled to target resolution
    """
    from ludus_renderer.torch import FThetaCamera

    raw_scene = scene._scene
    cam_id = raw_scene.camera_name_to_id.get(camera_name)
    if cam_id is None:
        available = list(raw_scene.camera_name_to_id.keys())
        raise ValueError(f"Camera '{camera_name}' not found. Available: {available}")

    cam = raw_scene.cameras[cam_id]

    native_w = cam.image_size[0].item()
    native_h = cam.image_size[1].item()

    if target_width == native_w and target_height == native_h:
        return cam

    scale_x = target_width / native_w
    scale_y = target_height / native_h
    scale = (scale_x + scale_y) / 2.0

    device = cam.principal_point.device
    return FThetaCamera(
        principal_point=torch.tensor(
            [cam.principal_point[0].item() * scale_x, cam.principal_point[1].item() * scale_y], device=device
        ),
        image_size=torch.tensor([float(target_width), float(target_height)], device=device),
        fw_poly=cam.fw_poly * scale,
        max_ray_angle=cam.max_ray_angle,
        linear_distortion=cam.linear_distortion,
        depth_max=cam.depth_max,
    )


def get_available_cameras(scene) -> List[str]:
    """Get list of available camera names from a scene.

    Args:
        scene: SceneAdapter wrapping a ClipgtGpuScene

    Returns:
        List of camera name strings
    """
    return list(scene._scene.camera_name_to_id.keys())


def create_camera(
    width: int,
    height: int,
    device: torch.device,
    bev: bool = False,
    bev_height: float = 80.0,
    bev_fov: float = 60.0,
    scene=None,
    camera_name: str = None,
):
    """Create a camera for rendering.

    If *scene* and *camera_name* are provided, returns the scene camera
    with intrinsics scaled to *width* x *height*.  If *bev* is ``True``,
    returns a synthetic BEV camera instead.

    Args:
        width, height: Target render resolution
        device: Torch device
        bev: If True, create a BEV camera
        bev_height: BEV camera height in metres (default 80)
        bev_fov: BEV camera vertical FOV in degrees (default 60)
        scene: SceneAdapter (required when bev=False)
        camera_name: Scene camera name (required when bev=False)
    """
    if bev:
        return create_bev_camera(width, height, device, bev_height, bev_fov)

    if scene is not None and camera_name is not None:
        return get_scene_camera(scene, camera_name, width, height)

    from ludus_renderer.torch import FThetaCamera

    cx, cy = width / 2.0, height / 2.0
    focal = 400.0
    max_angle = math.radians(90)

    return FThetaCamera(
        principal_point=torch.tensor([cx, cy], device=device),
        image_size=torch.tensor([float(width), float(height)], device=device),
        fw_poly=torch.tensor([0.0, focal, 0.0, 0.0, 0.0, 0.0], device=device),
        max_ray_angle=max_angle,
        depth_max=200.0,
    )


# =============================================================================
# Rendering utilities
# =============================================================================


def render_frame(
    ctx,
    scene,
    scene_id: int,
    timestamps,
    frame_idx: int,
    width: int,
    height: int,
    device: torch.device,
    bev_height: float = None,
    camera_name: str = "camera:front:wide:120fov",
    camera_id: int = 0,
) -> Image.Image:
    """Render a single frame and return as PIL Image.

    Args:
        ctx: LudusTimestampedContext
        scene: Scene with ego_tracks and cameras
        scene_id: Uploaded scene ID
        timestamps: Tensor of timestamps
        frame_idx: Frame index to render
        width, height: Output resolution
        device: Torch device
        bev_height: If provided, use BEV camera at this height
        camera_name: Camera name for extrinsics (sensor_to_rig transform)
        camera_id: Camera ID (index in uploaded cameras list)
    """
    from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR

    ts = timestamps[frame_idx]

    if bev_height is not None:
        world_to_camera = get_bev_camera_pose(scene, ts, bev_height, device)
        camera_type_id = CAMERA_TYPE_BEV
    else:
        world_to_camera = get_camera_pose(scene, ts, camera_name, device)
        camera_type_id = CAMERA_TYPE_REGULAR

    queries = [(scene_id, camera_id, ts.item(), camera_type_id)]
    camera_poses = world_to_camera.unsqueeze(0)

    images = ctx.render_batch(queries, camera_poses, resolution=(height, width))
    img = images[0, :, :, :3]
    if getattr(ctx, "needs_vflip", True):
        img = img.flip(0)
    return Image.fromarray(img.cpu().numpy())


def compute_camera_poses(
    scene, timestamps, device: torch.device, bev_height: float = None, camera_name: str = "camera:front:wide:120fov"
) -> Tuple[torch.Tensor, int]:
    """Compute camera poses for all timestamps.

    Args:
        scene: Scene with ego_tracks and cameras
        timestamps: Tensor of timestamps
        device: Torch device
        bev_height: If provided, compute BEV poses at this height
        camera_name: Camera name for extrinsics (sensor_to_rig transform)

    Returns:
        Tuple of (poses tensor [N, 4, 4], camera_type_id)
    """
    from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR

    if bev_height is not None:
        poses = get_all_bev_camera_poses(scene, timestamps, bev_height, device)
        camera_type_id = CAMERA_TYPE_BEV
    else:
        poses = get_all_camera_poses(scene, timestamps, camera_name, device)
        camera_type_id = CAMERA_TYPE_REGULAR

    return poses, camera_type_id


def render_sequence_gpu(
    ctx,
    scene_id: int,
    timestamps,
    camera_poses: torch.Tensor,
    camera_type_id: int,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    """Render all frames and return as GPU tensor.

    This is the core rendering function that keeps everything on GPU.
    Use this when you need direct access to GPU tensors for further processing.

    Args:
        ctx: LudusTimestampedContext
        scene_id: Uploaded scene ID
        timestamps: Tensor of timestamps to render
        camera_poses: Pre-computed camera poses [N, 4, 4]
        camera_type_id: Camera type (CAMERA_TYPE_REGULAR or CAMERA_TYPE_BEV)
        width, height: Output resolution
        device: Torch device

    Returns:
        torch.Tensor: GPU tensor of shape [N, H, W, 3] with uint8 RGB values
    """
    n_frames = len(timestamps)

    # Build tensor queries
    scene_ids = torch.full((n_frames,), scene_id, dtype=torch.int32, device=device)
    camera_ids = torch.zeros(n_frames, dtype=torch.int32, device=device)
    timestamps_tensor = timestamps.to(torch.int64)
    camera_type_ids = torch.full((n_frames,), camera_type_id, dtype=torch.int32, device=device)

    # Render all frames
    images = ctx.render(
        scene_ids, camera_ids, timestamps_tensor, camera_type_ids, camera_poses, resolution=(height, width)
    )

    images_rgb = images[:, :, :, :3]
    if getattr(ctx, "needs_vflip", True):
        images_rgb = images_rgb.flip(1)
    return images_rgb.contiguous()


def gpu_to_numpy(gpu_tensor: torch.Tensor) -> np.ndarray:
    """Transfer GPU tensor to CPU numpy array.

    Args:
        gpu_tensor: GPU tensor of shape [N, H, W, C]

    Returns:
        np.ndarray: CPU numpy array of same shape
    """
    torch.cuda.synchronize()
    return gpu_tensor.cpu().numpy()


def save_frames(images: np.ndarray, output_dir: str, prefix: str = "frame") -> List[str]:
    """Save numpy images to PNG files.

    Args:
        images: numpy array of shape [N, H, W, C] with uint8 values
        output_dir: Directory to save files
        prefix: Filename prefix (default: "frame")

    Returns:
        List of saved file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i in range(len(images)):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(images[i]).save(path)
        paths.append(path)

    return paths


def render_all_frames(
    ctx,
    scene,
    scene_id: int,
    timestamps,
    width: int,
    height: int,
    device: torch.device,
    bev_height: float = None,
    camera_name: str = "camera:front:wide:120fov",
    verbose: bool = False,
) -> Tuple[List[Image.Image], Dict]:
    """Render all frames and return as PIL Images with timing info.

    This is a convenience function that combines compute_camera_poses,
    render_sequence_gpu, gpu_to_numpy, and PIL image conversion.
    For more control, use the individual functions directly.

    Args:
        ctx: LudusTimestampedContext
        scene: Scene with ego_tracks and cameras
        scene_id: Uploaded scene ID
        timestamps: Tensor of timestamps
        width, height: Output resolution
        device: Torch device
        bev_height: If provided, use BEV camera at this height
        camera_name: Camera name for extrinsics (sensor_to_rig transform)
        verbose: If True, print timing info
    """
    timings = {}
    n_frames = len(timestamps)

    # Compute camera poses
    t0 = time.time()
    camera_poses, camera_type_id = compute_camera_poses(scene, timestamps, device, bev_height, camera_name)
    timings["pose_compute"] = time.time() - t0

    # GPU rendering
    t0 = time.time()
    gpu_images = render_sequence_gpu(ctx, scene_id, timestamps, camera_poses, camera_type_id, width, height, device)
    torch.cuda.synchronize()
    timings["gpu_render"] = time.time() - t0

    # Transfer to CPU
    t0 = time.time()
    cpu_images = gpu_to_numpy(gpu_images)
    timings["cpu_transfer"] = time.time() - t0

    # Convert to PIL
    pil_images = [Image.fromarray(cpu_images[i]) for i in range(n_frames)]

    if verbose:
        print(f"    Pose compute: {timings['pose_compute'] * 1000:.1f}ms")
        print(f"    GPU render:   {timings['gpu_render'] * 1000:.1f}ms ({n_frames / timings['gpu_render']:.1f} FPS)")
        print(f"    CPU transfer: {timings['cpu_transfer'] * 1000:.1f}ms")

    return pil_images, timings


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_gpu_memory_stats() -> Dict:
    """Get GPU memory statistics."""
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }
