# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Backward compatibility layer for ludus_renderer.torch imports.

New code should import directly from ludus_renderer:
    from ludus_renderer import LudusRenderer, LudusTimestampedContext

This module re-exports the same symbols for backward compatibility:
    from ludus_renderer.torch import LudusTimestampedContext  # Still works
"""

# Re-export from _ops for backward compatibility
from ludus_renderer._ops import (
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
    LudusCudaTimestampedContext,
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

__all__ = [
    # Core
    "LudusGLContext",
    "LudusTimestampedContext",
    "LudusCudaTimestampedContext",
    "ludus_render",
    # Primitives
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "CapStyle",
    # Timestamped
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    "CUBE_FLAG_WIREFRAME",
    # Primitive Type IDs
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
]
