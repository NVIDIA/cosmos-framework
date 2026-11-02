# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Low-level rendering operations for Ludus renderer.

This module re-exports all symbols from the split submodules:
- _plugin: JIT compilation
- primitives: Data classes and packing functions
- context: Ludus rendering contexts (LudusGLContext, LudusTimestampedContext)
"""

# JIT compilation
from ._plugin import _get_plugin, get_log_level, set_log_level

# Primitive data types and packing
from .primitives import (
    CAMERA_TYPE_BEV,
    CAMERA_TYPE_REGULAR,
    CUBE_FLAG_WIREFRAME,
    PRIM_BUFFER_ZONE,
    PRIM_CROSSWALK,
    PRIM_DOT_WHITE,
    PRIM_DOT_YELLOW,
    PRIM_EGO_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_INTERSECTION,
    PRIM_LANE_BOUNDARY,
    PRIM_LANE_LINE,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_OBSTACLE,
    PRIM_POLE,
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_ROAD_ISLAND,
    PRIM_ROAD_MARKING,
    PRIM_STATIC_OBSTACLE,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_TYPE_COUNT,
    PRIM_WAIT_LINE,
    # Data classes
    CapStyle,
    Cube,
    CubePool,
    FThetaCamera,
    ObstaclePool,
    Polygon,
    Polyline,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
    _pack_cameras,
    # Packing functions (internal)
    _pack_cubes,
    _pack_polygons,
    _pack_polylines,
    _triangulate_polygon_ear_clipping,
)

# Ludus context and rendering will be imported from context.py once created
# For now, we'll import from the old ops.py path until migration is complete

__all__ = [
    # Plugin
    "_get_plugin",
    "get_log_level",
    "set_log_level",
    # Constants
    "PRIM_ROAD_BOUNDARY",
    "PRIM_LANE_LINE",
    "PRIM_CROSSWALK",
    "PRIM_STATIC_OBSTACLE",
    "PRIM_EGO_TRAJECTORY",
    "PRIM_OBSTACLE",
    "PRIM_EGO_OBSTACLE",
    "PRIM_WAIT_LINE",
    "PRIM_POLE",
    "PRIM_ROAD_MARKING",
    "PRIM_LANE_BOUNDARY",
    "PRIM_TRAFFIC_LIGHT",
    "PRIM_TRAFFIC_SIGN",
    "PRIM_INTERSECTION",
    "PRIM_ROAD_ISLAND",
    "PRIM_BUFFER_ZONE",
    "PRIM_LANE_LINE_WHITE_SOLID",
    "PRIM_LANE_LINE_WHITE_DASHED",
    "PRIM_LANE_LINE_YELLOW_SOLID",
    "PRIM_LANE_LINE_YELLOW_DASHED",
    "PRIM_DOT_YELLOW",
    "PRIM_DOT_WHITE",
    "PRIM_TYPE_COUNT",
    "CAMERA_TYPE_REGULAR",
    "CAMERA_TYPE_BEV",
    "CUBE_FLAG_WIREFRAME",
    # Data classes
    "CapStyle",
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    # Internal packing helpers retained for backward compatibility
    "_pack_cameras",
    "_pack_cubes",
    "_pack_polygons",
    "_pack_polylines",
    "_triangulate_polygon_ear_clipping",
    # Ludus rendering (will be added)
    "LudusGLContext",
    "ludus_render",
    "LudusTimestampedContext",
    "LudusCudaTimestampedContext",
]

# Import Ludus context classes (these remain in context.py for now,
# we'll create this file next)
try:
    from .context import LudusCudaTimestampedContext, LudusGLContext, LudusTimestampedContext, ludus_render
except ImportError:
    # Fallback during migration - import from old location
    pass
