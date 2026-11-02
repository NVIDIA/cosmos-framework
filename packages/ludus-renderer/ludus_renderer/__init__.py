# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Ludus Renderer - GPU-native F-theta fisheye mesh shader rendering.

High-level API:
    from ludus_renderer import LudusRenderer, load_clipgt_scene

    scene = load_clipgt_scene("/path/to/clipgt/scene", device="cuda")
    renderer = LudusRenderer(width=1280, height=720)
    renderer.upload_scene(scene.timestamped_scene)
    images = renderer.render_batch(queries, poses)

Low-level API:
    from ludus_renderer import LudusTimestampedContext, TimestampedScene

NVJPEG Encoding (GPU-accelerated JPEG):
    from ludus_renderer import nvjpeg

    images = torch.randint(0, 255, (4, 3, 480, 640), dtype=torch.uint8, device='cuda')
    jpeg_bytes_list = nvjpeg.encode(images, quality=85)
"""

__version__ = "0.1.0"

# High-level API
# NVJPEG GPU encoding (lazy import)
from . import nvjpeg

# Low-level API (from _ops)
from ._ops import (
    CAMERA_TYPE_BEV,
    CAMERA_TYPE_REGULAR,
    CUBE_FLAG_WIREFRAME,
    PRIM_CROSSWALK,
    PRIM_EGO_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_LANE_LINE,
    PRIM_OBSTACLE,
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_STATIC_OBSTACLE,
    PRIM_TYPE_COUNT,
    CapStyle,
    Cube,
    CubePool,
    FThetaCamera,
    # Contexts
    LudusGLContext,
    LudusTimestampedContext,
    ObstaclePool,
    Polygon,
    # Primitives
    Polyline,
    TimestampedPolygonPool,
    # Timestamped pools
    TimestampedPolylinePool,
    TimestampedScene,
    ludus_render,
)
from .augmentation import mirror_augment_scene
from .clipgt import ClipgtGpuScene, EgoTrackData, load_av2_scene, load_clipgt_scene
from .renderer import LudusRenderer

# Scene cache
from .scene_cache import (
    LMDBSceneStore,
    PackedSceneBuffers,
    SceneDatabase,
    pack_scene,
    scene_bytes,
    scene_key,
    scene_to_device,
)

# Utilities
from .util import (
    FLU_TO_OPENCV_MATRIX,
    FLU_TO_OPENGL_MATRIX,
    OPENCV_TO_OPENGL_MATRIX,
    projection_matrix,
    resample_timestamps,
    rgb,
)

__all__ = [
    # Version
    "__version__",
    # High-level API
    "LudusRenderer",
    "ClipgtGpuScene",
    "load_clipgt_scene",
    "load_av2_scene",
    "EgoTrackData",
    "mirror_augment_scene",
    # Low-level contexts
    "LudusGLContext",
    "LudusTimestampedContext",
    "ludus_render",
    # Primitives
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "CapStyle",
    # Timestamped pools
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    # Constants
    "PRIM_ROAD_BOUNDARY",
    "PRIM_LANE_LINE",
    "PRIM_CROSSWALK",
    "PRIM_STATIC_OBSTACLE",
    "PRIM_EGO_TRAJECTORY",
    "PRIM_OBSTACLE",
    "PRIM_EGO_OBSTACLE",
    "PRIM_TYPE_COUNT",
    "CAMERA_TYPE_REGULAR",
    "CAMERA_TYPE_BEV",
    "CUBE_FLAG_WIREFRAME",
    # Utilities
    "rgb",
    "resample_timestamps",
    "projection_matrix",
    "FLU_TO_OPENCV_MATRIX",
    "OPENCV_TO_OPENGL_MATRIX",
    "FLU_TO_OPENGL_MATRIX",
    # Scene cache
    "SceneDatabase",
    "LMDBSceneStore",
    "PackedSceneBuffers",
    "pack_scene",
    "scene_key",
    "scene_bytes",
    "scene_to_device",
    # NVJPEG GPU encoding
    "nvjpeg",
]
