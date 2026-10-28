# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Direct clipgt and AV2 scene loading for ludus_renderer API.

This module reads clipgt and AV2 parquet files directly and converts them to the
GPU-native TimestampedScene format.

Clipgt elements are static (no timestamps), so we import them as timestamped
elements with a single observation at MIN_TIMESTAMP, making them visible for
all query timestamps.

AV2 elements are timestamped, with many observations per element type keyed by
timestamp_micros. These are loaded into TimestampedPolylinePool/PolygonPool
with real per-timestamp data.

Supports both file naming conventions:
- {element_type}.parquet (e.g., road_boundary.parquet)
- {clip_id}.{element_type}.parquet (e.g., av_xxx.road_boundary.parquet)
"""

import json
import os
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from ._ops import (
    CUBE_FLAG_WIREFRAME,
    PRIM_CROSSWALK,
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
    PRIM_ROAD_BOUNDARY,
    PRIM_ROAD_ISLAND,
    PRIM_ROAD_MARKING,
    PRIM_STATIC_OBSTACLE,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_WAIT_LINE,
    CubePool,
    FThetaCamera,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
)

# Persistent background thread pool (avoids per-call ThreadPoolExecutor creation)
_bg_pool = None


def _get_bg_pool():
    global _bg_pool
    if _bg_pool is None:
        _bg_pool = ThreadPoolExecutor(max_workers=3)
    return _bg_pool


# Ego vehicle dimensions (length, width, height) in meters
# Scaled up 20% for better visibility in BEV
EGO_VEHICLE_SIZE = [5.4, 2.4, 1.8]

# Ego colors
EGO_FRONT_COLOR = [0.9, 0.3, 0.2]  # Light red (front)
EGO_BACK_COLOR = [0.6, 0.1, 0.1]  # Dark red (back)


# ============================================================================
# Obstacle color scheme (matching v3 from imaginaire4)
# ============================================================================

# v3 color scheme from imaginaire4 config_color_bbox.json for obstacles by category
# Each entry: [front_rgb, back_rgb] (normalized 0-1)
OBSTACLE_COLORS_V3 = {
    "Car": [[0 / 255, 46 / 255, 136 / 255], [126 / 255, 206 / 255, 255 / 255]],  # Blue gradient
    "Truck": [[204 / 255, 55 / 255, 0 / 255], [255 / 255, 192 / 255, 64 / 255]],  # Orange gradient
    "Pedestrian": [[148 / 255, 0 / 255, 62 / 255], [255 / 255, 124 / 255, 171 / 255]],  # Magenta/pink gradient
    "Cyclist": [[0 / 255, 80 / 255, 66 / 255], [102 / 255, 208 / 255, 198 / 255]],  # Teal gradient
    "Other": [[53 / 255, 26 / 255, 20 / 255], [166 / 255, 136 / 255, 125 / 255]],  # Brown gradient
}

# Category string to ObjectType mapping (from imaginaire4 clipgt_loader.py)
CATEGORY_TO_OBJECT_TYPE = {
    # Car types
    "automobile": "Car",
    "other_vehicle": "Car",
    "vehicle": "Car",
    "car": "Car",
    # Pedestrian types
    "pedestrian": "Pedestrian",
    "person": "Pedestrian",
    # Cyclist types
    "bicycle": "Cyclist",
    "cyclist": "Cyclist",
    "motorcyclist": "Cyclist",
    "motorcycle": "Cyclist",
    "rider": "Cyclist",
    # Truck types
    "truck": "Truck",
    "bus": "Truck",
    "trailer": "Truck",
    "large_vehicle": "Truck",
    "heavy_truck": "Truck",
    "train_or_tram_car": "Truck",
    "trolley_bus": "Truck",
}

# AV2 obstacle_class integer IDs to ObjectType mapping.
#       Need to find the authoritative AV2 obstacle_class enum definition to
#       complete this mapping. Until then, unknown IDs fall back to 'Car' so
#       obstacles render in the familiar blue color.
AV2_OBSTACLE_CLASS_TO_TYPE = {
    1281: "Car",  # automobile (confirmed in test data)
    1282: "Truck",  # truck (estimated)
    1283: "Truck",  # bus (estimated)
    1284: "Truck",  # trailer (estimated)
    1285: "Truck",  # heavy_truck (estimated)
    1286: "Car",  # other_vehicle (estimated)
    2305: "Cyclist",  # bicycle (estimated)
    2306: "Cyclist",  # motorcycle (estimated)
    2307: "Cyclist",  # rider (estimated)
    2308: "Pedestrian",  # pedestrian (estimated)
    2309: "Pedestrian",  # person (estimated)
    3329: "Truck",  # train_or_tram_car (estimated)
    3330: "Truck",  # trolley_bus (estimated)
}

# Fallback for unknown AV2 class IDs - default to Car (blue) to match
# the reference renderer which uses a uniform blue for all obstacles
AV2_OBSTACLE_CLASS_DEFAULT = "Car"


def _get_obstacle_color(category):
    """Get front/back colors for an obstacle category.

    Accepts string categories (clipgt) or integer class IDs (AV2).
    """
    if not category:
        return OBSTACLE_COLORS_V3["Other"]

    # Try integer class ID first (AV2 format)
    if isinstance(category, (int, np.integer)):
        object_type = AV2_OBSTACLE_CLASS_TO_TYPE.get(int(category), AV2_OBSTACLE_CLASS_DEFAULT)
        return OBSTACLE_COLORS_V3.get(object_type, OBSTACLE_COLORS_V3["Other"])

    # Try parsing string as integer (AV2 stores class as str sometimes)
    try:
        class_id = int(category)
        object_type = AV2_OBSTACLE_CLASS_TO_TYPE.get(class_id, AV2_OBSTACLE_CLASS_DEFAULT)
        return OBSTACLE_COLORS_V3.get(object_type, OBSTACLE_COLORS_V3["Other"])
    except (ValueError, TypeError):
        pass

    # String category lookup (clipgt format)
    cat_lower = str(category).lower()
    object_type = CATEGORY_TO_OBJECT_TYPE.get(cat_lower, "Other")
    return OBSTACLE_COLORS_V3.get(object_type, OBSTACLE_COLORS_V3["Other"])


# Minimum timestamp for static elements (visible at all times)
MIN_TIMESTAMP_US = 0


# ============================================================================
# File loader utilities (tar + directory)
# ============================================================================


class DirectoryFileLoader:
    """Load files from a directory on the filesystem."""

    def __init__(self, base_path: str):
        self.base_path = base_path

    def open(self, relative_path: str) -> BinaryIO:
        full_path = os.path.join(self.base_path, relative_path)
        return open(full_path, "rb")


class TarFileLoader:
    """Load files from a tar archive."""

    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        self._tar_file = tarfile.open(tar_path, "r")
        self._member_by_name: Dict[str, tarfile.TarInfo] = {}
        for m in self._tar_file.getmembers():
            basename = m.name.rsplit("/", 1)[-1] if "/" in m.name else m.name
            self._member_by_name[basename] = m

    def open(self, relative_path: str) -> BinaryIO:
        basename = relative_path.rsplit("/", 1)[-1] if "/" in relative_path else relative_path
        member = self._member_by_name.get(basename)
        if member is None:
            raise FileNotFoundError(f"File '{relative_path}' not found in tar archive '{self.tar_path}'")

        file_obj = self._tar_file.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Cannot extract file '{relative_path}' from tar archive")

        return BytesIO(file_obj.read())

    def close(self):
        if self._tar_file:
            self._tar_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def get_file_loader(path: str):
    """Get an appropriate file loader for the given path (directory or tar)."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    if os.path.isdir(path):
        return DirectoryFileLoader(path)
    elif tarfile.is_tarfile(path):
        return TarFileLoader(path)
    else:
        raise ValueError(f"Path is neither a directory nor a tar file: {path}")


# ============================================================================
# Utility functions
# ============================================================================


def _points_to_tensor(points) -> Optional[Tensor]:
    """Convert list of {x, y, z} dicts or numpy array to tensor of shape (N, 3).

    Returns None if points is empty or contains NaN values.

    Supports:
    - List of dicts with x, y, z keys
    - numpy array of object dtype containing dicts
    - numpy array of float dtype with shape (N, 3)
    """
    if points is None:
        return None

    # Handle array-like
    if hasattr(points, "__len__"):
        if len(points) == 0:
            return None
    else:
        return None

    # Check if it's a numeric numpy array
    if isinstance(points, np.ndarray):
        if points.dtype != np.object_:
            # Direct numeric array
            tensor = torch.tensor(points, dtype=torch.float32)
            if torch.isnan(tensor).any():
                return None
            return tensor

    # Handle list/array of dicts or structured objects
    result = []
    for pt in points:
        if isinstance(pt, dict):
            result.append([pt["x"], pt["y"], pt["z"]])
        elif isinstance(pt, np.ndarray):
            result.append(pt.tolist())
        elif hasattr(pt, "x") and hasattr(pt, "y") and hasattr(pt, "z"):
            result.append([pt.x, pt.y, pt.z])
        else:
            result.append(list(pt))

    if not result:
        return None

    tensor = torch.tensor(result, dtype=torch.float32)
    if torch.isnan(tensor).any():
        return None
    return tensor


def _read_tquat(tquat: Dict[str, Any], translation_key: str = "center") -> Tensor:
    """Read translation + quaternion from dict."""
    return torch.tensor(
        [tquat[translation_key][c] for c in "xyz"] + [tquat["orientation"][c] for c in "xyzw"],
        dtype=torch.float32,
    )


# ============================================================================
# Lane line pattern functions
# ============================================================================


def _interpolate_polyline(polyline: np.ndarray, interval: float = 0.1) -> np.ndarray:
    """Interpolate a polyline to have evenly spaced points."""
    if len(polyline) < 2:
        return polyline

    diffs = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < interval:
        return polyline

    num_points = int(total_length / interval) + 1
    target_distances = np.linspace(0, total_length, num_points)

    result = []
    for d in target_distances:
        idx = np.searchsorted(cumulative, d)
        if idx == 0:
            result.append(polyline[0])
        elif idx >= len(cumulative):
            result.append(polyline[-1])
        else:
            t = (d - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1] + 1e-8)
            pt = polyline[idx - 1] * (1 - t) + polyline[idx] * t
            result.append(pt)

    return np.array(result, dtype=np.float32)


def _apply_dash_pattern(polyline: np.ndarray, dash_length: float, gap_length: float) -> List[Tensor]:
    """Apply a dash pattern to a polyline, returning list of visible segments."""
    if len(polyline) < 2:
        return []

    # Interpolate to get evenly spaced points
    polyline = _interpolate_polyline(polyline, interval=0.2)

    # Compute cumulative distances
    segments = polyline[1:] - polyline[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < dash_length * 0.5:
        return [torch.from_numpy(polyline.astype(np.float32))]

    pattern_length = dash_length + gap_length
    result_segments = []
    pattern_pos = 0.0

    while pattern_pos < total_length:
        dash_start = pattern_pos
        dash_end = min(pattern_pos + dash_length, total_length)

        mask = (cumulative >= dash_start) & (cumulative <= dash_end)
        indices = np.where(mask)[0]

        if len(indices) >= 2:
            dash_points = polyline[indices]
            result_segments.append(torch.from_numpy(dash_points.astype(np.float32)))

        pattern_pos += pattern_length

    return result_segments


def _apply_long_dash_pattern(polyline: np.ndarray) -> List[Tensor]:
    """Apply long dashed pattern: 3m dash, 9m gap."""
    return _apply_dash_pattern(polyline, dash_length=3.0, gap_length=9.0)


def _apply_short_dash_pattern(polyline: np.ndarray) -> List[Tensor]:
    """Apply short dashed pattern: 1.0m dash, 1.0m gap."""
    return _apply_dash_pattern(polyline, dash_length=1.0, gap_length=1.0)


def _apply_dotted_pattern(polyline: np.ndarray) -> List[Tensor]:
    """Apply dotted pattern: 0.5m dot, 1.0m gap."""
    return _apply_dash_pattern(polyline, dash_length=0.5, gap_length=1.0)


def _offset_polyline(polyline: np.ndarray, offset_distance: float) -> np.ndarray:
    """Offset a polyline perpendicular to its direction.

    Positive offset = left side, negative offset = right side.
    Uses the horizontal (XY) plane for offset calculation.
    """
    if len(polyline) < 2:
        return polyline

    result = np.zeros_like(polyline)

    for i in range(len(polyline)):
        # Compute tangent direction
        if i == 0:
            tangent = polyline[1] - polyline[0]
        elif i == len(polyline) - 1:
            tangent = polyline[-1] - polyline[-2]
        else:
            tangent = polyline[i + 1] - polyline[i - 1]

        # Normalize in XY plane
        tangent_xy = tangent[:2]
        length = np.linalg.norm(tangent_xy)
        if length < 1e-8:
            result[i] = polyline[i]
            continue

        tangent_xy = tangent_xy / length

        # Perpendicular in XY plane (rotate 90 degrees left)
        perp = np.array([-tangent_xy[1], tangent_xy[0]])

        # Apply offset
        result[i, 0] = polyline[i, 0] + perp[0] * offset_distance
        result[i, 1] = polyline[i, 1] + perp[1] * offset_distance
        result[i, 2] = polyline[i, 2]  # Keep Z unchanged

    return result


def _create_dual_line(polyline: np.ndarray, left_dashed: bool, separation: float = 0.15):
    """Create dual line pattern (solid + dashed side by side).

    Args:
        polyline: Original center line
        left_dashed: If True, left line is dashed, right is solid (SOLID_DASHED)
                     If False, left is solid, right is dashed (DASHED_SOLID)
        separation: Distance between the two lines in meters

    Returns:
        Tuple of (solid_line, dashed_segments)
    """
    # Offset to create left and right lines
    left_line = _offset_polyline(polyline, separation / 2)
    right_line = _offset_polyline(polyline, -separation / 2)

    if left_dashed:
        # SOLID_DASHED: left is dashed, right is solid
        solid_line = torch.from_numpy(right_line.astype(np.float32))
        dashed_segments = _apply_long_dash_pattern(left_line)
    else:
        # DASHED_SOLID: left is solid, right is dashed
        solid_line = torch.from_numpy(left_line.astype(np.float32))
        dashed_segments = _apply_long_dash_pattern(right_line)

    return solid_line, dashed_segments


def _create_dual_solid_line(polyline: np.ndarray, separation: float = 0.15):
    """Create dual solid line pattern (two parallel solid lines).

    Args:
        polyline: Original center line
        separation: Distance between the two lines in meters

    Returns:
        Tuple of (left_line, right_line) as tensors
    """
    left_line = _offset_polyline(polyline, separation / 2)
    right_line = _offset_polyline(polyline, -separation / 2)

    return (torch.from_numpy(left_line.astype(np.float32)), torch.from_numpy(right_line.astype(np.float32)))


def _create_dual_dotted_line(polyline: np.ndarray, separation: float = 0.15, spacing: float = 1.5):
    """Create dual dotted line pattern (two parallel dotted lines).

    Args:
        polyline: Original center line
        separation: Distance between the two lines in meters
        spacing: Dot spacing in meters

    Returns:
        Tuple of (left_dots, right_dots) as tensors
    """
    left_line = _offset_polyline(polyline, separation / 2)
    right_line = _offset_polyline(polyline, -separation / 2)

    return (
        _sample_dots_along_polyline(left_line, spacing=spacing),
        _sample_dots_along_polyline(right_line, spacing=spacing),
    )


def _sample_dots_along_polyline(polyline: np.ndarray, spacing: float = 1.5) -> Tensor:
    """Sample points along a polyline at regular intervals for dot rendering."""
    if len(polyline) < 2:
        return torch.from_numpy(polyline.astype(np.float32))

    polyline = _interpolate_polyline(polyline, interval=spacing * 0.5)

    diffs = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < spacing * 0.5:
        mid = polyline[len(polyline) // 2]
        return torch.from_numpy(np.array([mid], dtype=np.float32))

    dot_positions = []
    dist = 0.0
    while dist <= total_length:
        idx = np.searchsorted(cumulative, dist)
        if idx == 0:
            dot_positions.append(polyline[0])
        elif idx >= len(cumulative):
            dot_positions.append(polyline[-1])
        else:
            t = (dist - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1] + 1e-8)
            pt = polyline[idx - 1] * (1 - t) + polyline[idx] * t
            dot_positions.append(pt)
        dist += spacing

    return torch.from_numpy(np.array(dot_positions, dtype=np.float32))


# ============================================================================
# Parquet reading functions (with robust error handling)
# ============================================================================


def read_road_boundary(parquet_path: Path) -> List[Tensor]:
    """Read road boundary polylines from parquet."""
    if not parquet_path.exists():
        return []

    lines = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        boundary = row.road_boundary
        if "location" in boundary and boundary["location"] is not None:
            pts = _points_to_tensor(boundary["location"])
            if pts is not None and len(pts) > 1:
                lines.append(pts)
    return lines


def read_lane(parquet_path: Path) -> List[Tensor]:
    """Read lane boundary polylines from parquet (left_rail, right_rail)."""
    if not parquet_path.exists():
        return []

    lines = []
    df = pd.read_parquet(parquet_path)
    for idx, row in df.iterrows():
        lane = row["lane"]
        # Process left rail
        if "left_rail" in lane and lane["left_rail"] is not None:
            pts = _points_to_tensor(lane["left_rail"])
            if pts is not None and len(pts) > 1:
                lines.append(pts)
        # Process right rail
        if "right_rail" in lane and lane["right_rail"] is not None:
            pts = _points_to_tensor(lane["right_rail"])
            if pts is not None and len(pts) > 1:
                lines.append(pts)
    return lines


@dataclass
class LaneLineData:
    """Lane line data separated by color and style."""

    white_solid: List[Tensor]
    white_dashed: List[Tensor]
    yellow_solid: List[Tensor]
    yellow_dashed: List[Tensor]
    yellow_dots: List[Tensor]  # For DOT_SOLID style - rendered as circles


def read_lane_line(parquet_path: Path, simplify_dual_lines: bool = False) -> LaneLineData:
    """Read lane line polylines from parquet, separated by color/style.

    Applies appropriate dash/dot patterns based on style:
    - SOLID: kept as-is
    - LONG_DASHED: 3m dash, 9m gap
    - SHORT_DASHED: 1m dash, 1m gap
    - DOT: sampled points for dot primitive rendering

    Args:
        parquet_path: Path to lane_line parquet file
        simplify_dual_lines: If True, treat SOLID_DASHED, DASHED_SOLID, and SOLID_GROUP
                            as single solid lines (no dual line generation)
    """
    result = LaneLineData(
        white_solid=[],
        white_dashed=[],
        yellow_solid=[],
        yellow_dashed=[],
        yellow_dots=[],
    )

    if not parquet_path.exists():
        return result

    df = pd.read_parquet(parquet_path)
    for idx, row in df.iterrows():
        ll = row["lane_line"]

        # Check both new (line_rail) and old (path) format
        pts = None
        if "line_rail" in ll and ll["line_rail"] is not None:
            pts = _points_to_tensor(ll["line_rail"])
        elif "path" in ll and ll["path"] is not None:
            pts = _points_to_tensor(ll["path"])

        if pts is None or len(pts) <= 1:
            continue

        # Determine color and style
        colors = ll.get("colors", [])
        styles = ll.get("styles", [])

        if hasattr(colors, "__len__") and len(colors) > 0:
            color = str(colors[0]).upper() if colors[0] else "WHITE"
        else:
            color = "WHITE"

        if hasattr(styles, "__len__") and len(styles) > 0:
            style = str(styles[0]).upper() if styles[0] else "SOLID"
        else:
            style = "SOLID"

        # Convert to numpy for pattern processing
        pts_np = pts.numpy()

        # Categorize by color and apply appropriate pattern
        if color == "YELLOW":
            if style == "SOLID_DASHED":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.yellow_solid.append(pts)
                else:
                    # Dual line: per reference config: ["long_dashed", "solid"] = left dashed, right solid
                    solid_line, dashed_segments = _create_dual_line(pts_np, left_dashed=True)
                    result.yellow_solid.append(solid_line)
                    result.yellow_dashed.extend(dashed_segments)
            elif style == "DASHED_SOLID":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.yellow_solid.append(pts)
                else:
                    # Dual line: per reference config: ["solid", "long_dashed"] = left solid, right dashed
                    solid_line, dashed_segments = _create_dual_line(pts_np, left_dashed=False)
                    result.yellow_solid.append(solid_line)
                    result.yellow_dashed.extend(dashed_segments)
            elif style == "SOLID_GROUP":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.yellow_solid.append(pts)
                else:
                    # Dual solid lines (double yellow)
                    left_line, right_line = _create_dual_solid_line(pts_np)
                    result.yellow_solid.append(left_line)
                    result.yellow_solid.append(right_line)
            elif style == "DOT_SOLID_GROUP":
                # Dual dotted lines
                left_dots, right_dots = _create_dual_dotted_line(pts_np, spacing=1.5)
                result.yellow_dots.append(left_dots)
                result.yellow_dots.append(right_dots)
            elif style == "DOT_SOLID_SINGLE":
                # Single dotted line - sample points for dot primitive rendering
                dots = _sample_dots_along_polyline(pts_np, spacing=1.5)
                result.yellow_dots.append(dots)
            elif "SOLID" in style and "DASHED" not in style and "DOT" not in style:
                # SOLID_SINGLE or other solid variants
                result.yellow_solid.append(pts)
            elif "DOT" in style:
                # Other DOT patterns (DOT_DASHED_SINGLE, etc.)
                dots = _sample_dots_along_polyline(pts_np, spacing=0.5)
                result.yellow_dots.append(dots)
            elif "SHORT" in style:
                result.yellow_dashed.extend(_apply_short_dash_pattern(pts_np))
            else:
                # Default to long dashed (LONG_DASHED_SINGLE, OTHER, etc.)
                result.yellow_dashed.extend(_apply_long_dash_pattern(pts_np))
        else:  # WHITE or UNKNOWN
            if style == "SOLID_DASHED":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.white_solid.append(pts)
                else:
                    solid_line, dashed_segments = _create_dual_line(pts_np, left_dashed=True)
                    result.white_solid.append(solid_line)
                    result.white_dashed.extend(dashed_segments)
            elif style == "DASHED_SOLID":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.white_solid.append(pts)
                else:
                    solid_line, dashed_segments = _create_dual_line(pts_np, left_dashed=False)
                    result.white_solid.append(solid_line)
                    result.white_dashed.extend(dashed_segments)
            elif style == "SOLID_GROUP":
                if simplify_dual_lines:
                    # Simplify to single solid line
                    result.white_solid.append(pts)
                else:
                    # Dual solid lines (double white)
                    left_line, right_line = _create_dual_solid_line(pts_np)
                    result.white_solid.append(left_line)
                    result.white_solid.append(right_line)
            elif "SOLID" in style and "DASHED" not in style:
                result.white_solid.append(pts)
            else:
                # Apply appropriate dash pattern
                if "DOT" in style:
                    result.white_dashed.extend(_apply_dotted_pattern(pts_np))
                elif "SHORT" in style:
                    result.white_dashed.extend(_apply_short_dash_pattern(pts_np))
                else:
                    # Default to long dashed (LONG_DASHED_SINGLE, OTHER, etc.)
                    result.white_dashed.extend(_apply_long_dash_pattern(pts_np))

    return result


def read_crosswalk(parquet_path: Path) -> List[Tensor]:
    """Read crosswalk polygons from parquet."""
    if not parquet_path.exists():
        return []

    polygons = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        cw = row.crosswalk
        if "location" in cw and cw["location"] is not None:
            pts = _points_to_tensor(cw["location"])
            if pts is not None and len(pts) > 2:
                polygons.append(pts)
    return polygons


def read_road_marking(parquet_path: Path) -> List[Tensor]:
    """Read road marking polygons from parquet."""
    if not parquet_path.exists():
        return []

    polygons = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        rm = row.road_marking
        if "location" in rm and rm["location"] is not None:
            pts = _points_to_tensor(rm["location"])
            if pts is not None and len(pts) > 2:
                polygons.append(pts)
    return polygons


def read_wait_line(parquet_path: Path) -> List[Tensor]:
    """Read wait line polylines from parquet."""
    if not parquet_path.exists():
        return []

    lines = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        wl = row.wait_line
        if "location" in wl and wl["location"] is not None:
            pts = _points_to_tensor(wl["location"])
            if pts is not None and len(pts) >= 2:
                lines.append(pts)
    return lines


def read_pole(parquet_path: Path) -> List[Tensor]:
    """Read pole polylines from parquet."""
    if not parquet_path.exists():
        return []

    lines = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        pole = row.pole
        if "location" in pole and pole["location"] is not None:
            loc = pole["location"]
            if len(loc) >= 2:
                pts = _points_to_tensor(loc)
                if pts is not None:
                    lines.append(pts)
            elif len(loc) == 1:
                # Create vertical pole from single point (3m default height)
                pt = loc[0]
                base = torch.tensor([[pt["x"], pt["y"], pt["z"]]], dtype=torch.float32)
                top = base.clone()
                top[0, 2] += 3.0  # 3m height
                lines.append(torch.cat([base, top], dim=0))
    return lines


def read_intersection_area(parquet_path: Path) -> List[Tensor]:
    """Read intersection area polygons from parquet."""
    if not parquet_path.exists():
        return []

    polygons = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        area = row.intersection_area
        if "location" in area and area["location"] is not None:
            pts = _points_to_tensor(area["location"])
            if pts is not None and len(pts) > 2:
                polygons.append(pts)
    return polygons


def read_road_island(parquet_path: Path) -> List[Tensor]:
    """Read road island polygons from parquet."""
    if not parquet_path.exists():
        return []

    polygons = []
    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        island = row.road_island
        if "location" in island and island["location"] is not None:
            pts = _points_to_tensor(island["location"])
            if pts is not None and len(pts) > 2:
                polygons.append(pts)
    return polygons


@dataclass
class CubeData:
    """Cube data for static elements like traffic lights/signs."""

    translations: List[List[float]]  # [n, 3]
    quaternions: List[List[float]]  # [n, 4] xyzw
    scales: List[List[float]]  # [n, 3]


def read_traffic_light(parquet_path: Path) -> CubeData:
    """Read traffic lights as cubes from parquet."""
    result = CubeData(translations=[], quaternions=[], scales=[])

    if not parquet_path.exists():
        return result

    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        tl = row.traffic_light
        if "center" in tl and tl["center"] is not None:
            center = [tl["center"]["x"], tl["center"]["y"], tl["center"]["z"]]
            if any(x is None or np.isnan(x) for x in center):
                continue
            result.translations.append(center)

            # Get orientation
            if "orientation" in tl and tl["orientation"] is not None:
                ori = tl["orientation"]
                quat = [ori.get("x", 0), ori.get("y", 0), ori.get("z", 0), ori.get("w", 1)]
            else:
                quat = [0.0, 0.0, 0.0, 1.0]
            result.quaternions.append(quat)

            # Get dimensions
            if "dimensions" in tl and tl["dimensions"] is not None:
                dim = tl["dimensions"]
                scale = [dim.get("x", 0.3), dim.get("y", 0.3), dim.get("z", 0.5)]
            else:
                scale = [0.3, 0.3, 0.5]
            result.scales.append(scale)

    return result


def read_traffic_sign(parquet_path: Path) -> CubeData:
    """Read traffic signs as cubes from parquet."""
    result = CubeData(translations=[], quaternions=[], scales=[])

    if not parquet_path.exists():
        return result

    df = pd.read_parquet(parquet_path)
    for row in df.itertuples():
        ts = row.traffic_sign
        if "center" in ts and ts["center"] is not None:
            center = [ts["center"]["x"], ts["center"]["y"], ts["center"]["z"]]
            if any(x is None or np.isnan(x) for x in center):
                continue
            result.translations.append(center)

            # Get orientation
            if "orientation" in ts and ts["orientation"] is not None:
                ori = ts["orientation"]
                quat = [ori.get("x", 0), ori.get("y", 0), ori.get("z", 0), ori.get("w", 1)]
            else:
                quat = [0.0, 0.0, 0.0, 1.0]
            result.quaternions.append(quat)

            # Get dimensions
            if "dimensions" in ts and ts["dimensions"] is not None:
                dim = ts["dimensions"]
                scale = [dim.get("x", 0.01), dim.get("y", 0.5), dim.get("z", 0.8)]
            else:
                scale = [0.01, 0.5, 0.8]
            result.scales.append(scale)

    return result


# ============================================================================
# Obstacle and ego data
# ============================================================================


@dataclass
class ObstacleData:
    """Obstacle data from clipgt."""

    trackline_id: str
    category: str
    size: Tensor  # [3] xyz dimensions
    timestamps: Tensor  # [n_poses] int64
    poses_tquat: Tensor  # [n_poses, 7] xyz + quaternion xyzw


def read_obstacles(parquet_path: Path) -> List[ObstacleData]:
    """Read obstacles from parquet."""
    if not parquet_path.exists():
        return []

    df = pd.read_parquet(parquet_path)
    raw_data: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)

    for row in df.itertuples():
        trackline_id = row.obstacle["trackline_id"]
        timestamp = row.key["timestamp_micros"]
        raw_data[trackline_id].append((timestamp, row.obstacle))

    obstacles = []
    for trackline_id, rows in raw_data.items():
        timestamps = []
        tquats = []
        for ts, obs_data in rows:
            timestamps.append(ts)
            tquats.append(_read_tquat(obs_data))

        size = obs_data["size"]
        obstacles.append(
            ObstacleData(
                trackline_id=trackline_id,
                category=obs_data["category"],
                size=torch.tensor([size["x"], size["y"], size["z"]]),
                timestamps=torch.tensor(timestamps, dtype=torch.int64),
                poses_tquat=torch.stack(tquats),
            )
        )
    return obstacles


@dataclass
class EgoTrackData:
    """Ego track from clipgt."""

    timestamps: Tensor  # [n_poses] int64
    poses_tquat: Tensor  # [n_poses, 7]

    @property
    def translations(self) -> Tensor:
        """Get translation vectors [n_poses, 3]."""
        return self.poses_tquat[:, :3]

    @property
    def quaternions(self) -> Tensor:
        """Get quaternions [n_poses, 4] as xyzw."""
        return self.poses_tquat[:, 3:]


def read_egomotion_estimate(parquet_path: Path) -> EgoTrackData:
    """Read egomotion from parquet."""
    if not parquet_path.exists():
        return EgoTrackData(
            timestamps=torch.zeros(0, dtype=torch.int64),
            poses_tquat=torch.zeros(0, 7, dtype=torch.float32),
        )

    df = pd.read_parquet(parquet_path)
    timestamps = []
    tquats = []

    for _, row in df.iterrows():
        ego_data = row["egomotion_estimate"]
        key = row["key"]

        if "location" not in ego_data or "orientation" not in ego_data:
            continue
        if not isinstance(key, dict) or "timestamp_micros" not in key:
            continue

        tquats.append(_read_tquat(ego_data, translation_key="location"))
        timestamps.append(key["timestamp_micros"])

    if not tquats:
        return EgoTrackData(
            timestamps=torch.zeros(0, dtype=torch.int64),
            poses_tquat=torch.zeros(0, 7, dtype=torch.float32),
        )

    return EgoTrackData(
        timestamps=torch.tensor(timestamps, dtype=torch.int64),
        poses_tquat=torch.stack(tquats),
    )


# ============================================================================
# Camera calibration
# ============================================================================


@dataclass
class CameraData:
    """Camera intrinsics and extrinsics."""

    name: str
    cx: float
    cy: float
    width: int
    height: int
    poly: np.ndarray  # Polynomial coefficients
    is_bw_poly: bool  # True if polynomial is backward (r→θ), False if forward (θ→r)
    sensor_to_rig: Tensor  # [4, 4]
    linear_cde: np.ndarray  # Linear affine term [C, D, E] for [[C,D],[D,E]] matrix


def read_calibration_estimate(parquet_path: Path) -> List[CameraData]:
    """Read camera calibration from parquet."""
    if not parquet_path.exists():
        return []

    df = pd.read_parquet(parquet_path)
    cal_data = df.iloc[0]["calibration_estimate"]
    rig_data = json.loads(str(cal_data["rig_json"]))["rig"]

    cameras = []
    for sensor in rig_data["sensors"]:
        name = sensor["name"]
        if not name.startswith("camera:"):
            continue
        if sensor.get("properties") is None:
            continue

        props = sensor["properties"]

        # Get polynomial
        poly_key = "polynomial" if "polynomial" in props else "bw-poly"
        if poly_key not in props:
            continue
        poly_coeffs = [float(x) for x in props[poly_key].split()]
        if len(poly_coeffs) < 6:
            poly_coeffs.extend([0.0] * (6 - len(poly_coeffs)))

        # Determine polynomial direction from polynomial-type field
        # pixeldistance-to-angle = backward (r → θ) = needs inversion
        # angle-to-pixeldistance = forward (θ → r) = use directly
        poly_type = props.get("polynomial-type", "")
        if poly_type == "angle-to-pixeldistance":
            is_bw_poly = False  # Forward polynomial
        elif poly_type == "pixeldistance-to-angle":
            is_bw_poly = True  # Backward polynomial
        elif poly_key == "bw-poly":
            is_bw_poly = True  # Explicitly named backward poly
        else:
            # Heuristic fallback: backward poly has small c1 (radians/pixel)
            # Forward poly has large c1 (pixels/radian, ~focal length)
            is_bw_poly = len(poly_coeffs) > 1 and abs(poly_coeffs[1]) < 1.0

        # Get linear affine term [[C,D],[E,1]] - defaults [1,0,0] = identity
        linear_c = float(props.get("linear-c", 1.0))
        linear_d = float(props.get("linear-d", 0.0))
        linear_e = float(props.get("linear-e", 0.0))
        linear_cde = np.array([linear_c, linear_d, linear_e], dtype=np.float32)

        # Get extrinsics
        rpy = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]
        translation = sensor["nominalSensor2Rig_FLU"]["t"]
        # Use intrinsic xyz convention (matching reference renderer)
        # scipy "xyz" = extrinsic XYZ = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        rotation = R.from_euler("xyz", np.radians(rpy)).as_matrix()

        # Apply correction if available (sensor = nominal @ correction)
        if "correction_sensor_R_FLU" in sensor:
            corr_rpy = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]
            corr_rotation = R.from_euler("xyz", np.radians(corr_rpy)).as_matrix()
            rotation = rotation @ corr_rotation

        sensor_to_rig = torch.eye(4, dtype=torch.float32)
        sensor_to_rig[:3, :3] = torch.from_numpy(rotation.astype(np.float32))
        sensor_to_rig[:3, 3] = torch.tensor(translation, dtype=torch.float32)

        cameras.append(
            CameraData(
                name=name,
                cx=float(props["cx"]),
                cy=float(props["cy"]),
                width=int(props["width"]),
                height=int(props["height"]),
                poly=np.array(poly_coeffs, dtype=np.float32),
                is_bw_poly=is_bw_poly,
                sensor_to_rig=sensor_to_rig,
                linear_cde=linear_cde,
            )
        )
    return cameras


# ============================================================================
# Conversion to GPU pools
# ============================================================================


def _polylines_to_pool(
    polylines: List[Tensor],
    prim_type_id: int,
    device: torch.device,
) -> Optional[TimestampedPolylinePool]:
    """Convert polylines to TimestampedPolylinePool."""
    if not polylines:
        return None

    vertices = torch.cat(polylines, dim=0).to(device)
    varrays_prefix_sum = torch.cumsum(torch.tensor([p.shape[0] for p in polylines], dtype=torch.int32), dim=0).to(
        device
    )

    return TimestampedPolylinePool(
        timestamps_us=torch.tensor([MIN_TIMESTAMP_US], dtype=torch.int64, device=device),
        timestamped_varrays_prefix_sum=torch.tensor([len(polylines)], dtype=torch.int32, device=device),
        varrays_prefix_sum=varrays_prefix_sum,
        vertices=vertices,
        prim_type_id=prim_type_id,
    )


def _polygons_to_pool(
    polygons: List[Tensor],
    prim_type_id: int,
    device: torch.device,
) -> Optional[TimestampedPolygonPool]:
    """Convert polygons to TimestampedPolygonPool with triangulation.

    Triangle indices are LOCAL per polygon (each polygon's indices start from 0).
    """
    if not polygons:
        return None

    # Triangulate polygons (fan triangulation with LOCAL indices)
    all_vertices = []
    all_triangles = []
    vert_counts = []
    tri_counts = []

    for poly in polygons:
        n_verts = len(poly)
        if n_verts < 3:
            continue

        all_vertices.append(poly)
        vert_counts.append(n_verts)

        # Fan triangulation with LOCAL indices (0 to n_verts-1)
        n_tris = n_verts - 2
        local_tris = []
        for j in range(1, n_verts - 1):
            local_tris.append([0, j, j + 1])

        all_triangles.append(torch.tensor(local_tris, dtype=torch.int32))
        tri_counts.append(n_tris)

    if not all_vertices:
        return None

    vertices = torch.cat(all_vertices, dim=0).to(device)
    triangles = torch.cat(all_triangles, dim=0).to(device)

    varrays_prefix_sum = torch.cumsum(torch.tensor(vert_counts, dtype=torch.int32), dim=0).to(device)
    triangle_prefix_sum = torch.cumsum(torch.tensor(tri_counts, dtype=torch.int32), dim=0).to(device)

    return TimestampedPolygonPool(
        timestamps_us=torch.tensor([MIN_TIMESTAMP_US], dtype=torch.int64, device=device),
        timestamped_varrays_prefix_sum=torch.tensor([len(all_vertices)], dtype=torch.int32, device=device),
        varrays_prefix_sum=varrays_prefix_sum,
        triangle_prefix_sum=triangle_prefix_sum,
        vertices=vertices,
        triangles=triangles,
        prim_type_id=prim_type_id,
    )


def _obstacles_to_pool(
    obstacles: List[ObstacleData],
    device: torch.device,
) -> Optional[CubePool]:
    """Convert obstacles to CubePool with category-based colors."""
    if not obstacles:
        return None

    n_obstacles = len(obstacles)

    # Compute cumulative track lengths
    track_lengths = [len(obs.timestamps) for obs in obstacles]
    obstacle_ts_prefix_sum = torch.cumsum(torch.tensor(track_lengths, dtype=torch.int32), dim=0).to(device)

    # Concatenate all track poses
    all_timestamps = torch.cat([obs.timestamps for obs in obstacles]).to(device)
    all_translations = torch.cat([obs.poses_tquat[:, :3] for obs in obstacles]).to(device)
    all_quaternions = torch.cat([obs.poses_tquat[:, 3:] for obs in obstacles]).to(device)

    # Global timeline from unique obstacle timestamps
    unique_timestamps = torch.unique(all_timestamps)
    timestamps_us = unique_timestamps.sort()[0]

    # Scales
    scales = torch.stack([obs.size for obs in obstacles]).to(device)

    # Colors: [n_obstacles, 6] - use category-specific colors
    color_np = np.empty((n_obstacles, 6), dtype=np.float32)
    for i, obs in enumerate(obstacles):
        front, back = _get_obstacle_color(obs.category)
        color_np[i, :3] = front
        color_np[i, 3:] = back
    colors = torch.from_numpy(color_np).to(device)

    return CubePool(
        timestamps_us=timestamps_us,
        cube_ts_prefix_sum=obstacle_ts_prefix_sum,
        track_timestamps_us=all_timestamps,
        translations=all_translations,
        quaternions=all_quaternions,
        scales=scales,
        colors=colors,
        prim_type_id=PRIM_OBSTACLE,
        render_flags=CUBE_FLAG_WIREFRAME,  # Obstacles rendered as wireframes
    )


def _ego_trajectory_to_pool(
    ego_track: "EgoTrackData",
    device: torch.device,
) -> Optional[TimestampedPolylinePool]:
    """Create a polyline pool for the ego trajectory.

    The ego trajectory is a green polyline showing the full path taken.
    Uses first timestamp so it's always valid for any query timestamp >= first.
    """
    if ego_track is None or len(ego_track.timestamps) == 0:
        return None

    n_poses = len(ego_track.timestamps)
    translations = ego_track.translations
    if translations.device != device:
        translations = translations.to(device)

    if n_poses > 200:
        stride = max(1, (n_poses - 1) // 199)
        traj_points = translations[::stride]
    else:
        traj_points = translations

    first_ts = ego_track.timestamps[0:1]
    if first_ts.device != device:
        first_ts = first_ts.to(device)

    n_pts = traj_points.shape[0]
    return TimestampedPolylinePool(
        timestamps_us=first_ts,
        timestamped_varrays_prefix_sum=torch.tensor([1], dtype=torch.int32, device=device),
        varrays_prefix_sum=torch.tensor([n_pts], dtype=torch.int32, device=device),
        vertices=traj_points,
        prim_type_id=PRIM_EGO_TRAJECTORY,
    )


def _ego_obstacle_to_pool(
    ego_track: "EgoTrackData",
    device: torch.device,
) -> Optional[CubePool]:
    """Create a CubePool for the ego vehicle obstacle.

    The ego vehicle is rendered as a colored cube at each timestamp.
    Only visible in BEV mode where you want to see the ego position.
    """
    if ego_track is None or len(ego_track.timestamps) == 0:
        return None

    n_poses = len(ego_track.timestamps)

    # Ego is a single cube with all track poses
    cube_ts_prefix_sum = torch.tensor([n_poses], dtype=torch.int32, device=device)

    timestamps = ego_track.timestamps.to(device)
    translations = ego_track.translations.to(device)
    quaternions = ego_track.quaternions.to(device)

    # Fixed ego vehicle size [n_cubes=1, 3]
    scales = torch.tensor([EGO_VEHICLE_SIZE], dtype=torch.float32, device=device)

    # Ego color: front (light red) and back (dark red) [n_cubes=1, 6]
    colors = torch.tensor([EGO_FRONT_COLOR + EGO_BACK_COLOR], dtype=torch.float32, device=device)

    return CubePool(
        timestamps_us=timestamps,
        cube_ts_prefix_sum=cube_ts_prefix_sum,
        track_timestamps_us=timestamps,
        translations=translations,
        quaternions=quaternions,
        scales=scales,
        colors=colors,
        prim_type_id=PRIM_EGO_OBSTACLE,
        render_flags=CUBE_FLAG_WIREFRAME,  # Ego with wireframe
    )


def _static_cubes_to_pool(
    cube_data: CubeData,
    prim_type_id: int,
    device: torch.device,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Optional[CubePool]:
    """Convert static cubes (traffic lights/signs) to CubePool.

    Static cubes need to be visible at any query timestamp. The shader checks
    if query is within [first_ts, last_ts] of each track. We use two timestamps
    (MIN and MAX) with duplicated poses to create a valid range covering all time.
    """
    if not cube_data.translations:
        return None

    n_cubes = len(cube_data.translations)

    # Use MIN and MAX timestamps so cubes are visible at any query time
    MAX_TIMESTAMP_US = 2**62  # Large but safe for int64

    # Pool covers the full time range
    timestamps_us = torch.tensor([MIN_TIMESTAMP_US, MAX_TIMESTAMP_US], dtype=torch.int64, device=device)

    # Each cube has 2 poses (duplicated at min and max timestamps)
    # prefix sum: [2, 4, 6, ...] - each cube has 2 poses
    cube_ts_prefix_sum = torch.arange(2, 2 * n_cubes + 1, step=2, dtype=torch.int32, device=device)

    # Track timestamps: [min, max, min, max, ...] for each cube's two poses
    track_timestamps_us = torch.tensor([MIN_TIMESTAMP_US, MAX_TIMESTAMP_US] * n_cubes, dtype=torch.int64, device=device)

    # Base data
    translations_base = torch.tensor(cube_data.translations, dtype=torch.float32, device=device)
    quaternions_base = torch.tensor(cube_data.quaternions, dtype=torch.float32, device=device)
    scales = torch.tensor(cube_data.scales, dtype=torch.float32, device=device)

    # Duplicate poses for each timestamp: [pose0_t0, pose0_t1, pose1_t0, pose1_t1, ...]
    translations = translations_base.repeat_interleave(2, dim=0)  # [2*n_cubes, 3]
    quaternions = quaternions_base.repeat_interleave(2, dim=0)  # [2*n_cubes, 4]

    # Colors are per-cube (NOT per-pose)
    colors = torch.tensor(
        [list(color) + list(color)] * n_cubes,
        dtype=torch.float32,
        device=device,
    )

    return CubePool(
        timestamps_us=timestamps_us,
        cube_ts_prefix_sum=cube_ts_prefix_sum,
        track_timestamps_us=track_timestamps_us,
        translations=translations,
        quaternions=quaternions,
        scales=scales,
        colors=colors,
        prim_type_id=prim_type_id,
    )


_camera_cpp_ext = None


def _get_camera_cpp_ext():
    global _camera_cpp_ext
    if _camera_cpp_ext is not None:
        return _camera_cpp_ext
    from torch.utils.cpp_extension import load

    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")
    _camera_cpp_ext = load(
        name="camera_convert_cpp",
        sources=[os.path.join(csrc, "camera_convert.cpp")],
        verbose=False,
    )
    return _camera_cpp_ext


def _eval_poly_np(poly: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial poly[0] + poly[1]*x + poly[2]*x^2 + ... vectorized."""
    result = np.zeros_like(x)
    xi = np.ones_like(x)
    for c in poly:
        result += c * xi
        xi *= x
    return result


def _eval_poly_scalar(poly, x: float) -> float:
    """Evaluate polynomial at a single scalar value (no numpy allocation)."""
    result = 0.0
    xi = 1.0
    for c in poly:
        result += float(c) * xi
        xi *= x
    return result


def _cameras_to_ftheta_ref(
    cameras: List[CameraData],
    device: torch.device,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[List[FThetaCamera], Dict[str, int], Dict[str, Tensor]]:
    """Reference (pure-Python/numpy) camera conversion. Used by the PyArrow path."""
    n = len(cameras)
    pp_np = np.empty((n, 2), dtype=np.float32)
    sz_np = np.empty((n, 2), dtype=np.float32)
    fw_np = np.zeros((n, 6), dtype=np.float32)
    ld_np = np.zeros((n, 2, 2), dtype=np.float32)
    mra_np = np.empty(n, dtype=np.float32)
    s2r_np = np.empty((n, 4, 4), dtype=np.float32)

    camera_name_to_id = {}
    cam_names = []
    r_samples = np.linspace(0, 1, 200)

    for i, cam in enumerate(cameras):
        if target_resolution:
            target_w, target_h = target_resolution
            scale_x = target_w / cam.width
            scale_y = target_h / cam.height
            poly_scale = (scale_x + scale_y) / 2
            cx = cam.cx * scale_x
            cy = cam.cy * scale_y
            img_w = float(target_w)
            img_h = float(target_h)
        else:
            poly_scale = 1.0
            cx = cam.cx
            cy = cam.cy
            img_w = float(cam.width)
            img_h = float(cam.height)

        pp_np[i] = [cx, cy]
        sz_np[i] = [img_w, img_h]
        poly = cam.poly

        if cam.is_bw_poly and len(poly) > 1 and poly[1] != 0:
            r_max = max(
                np.sqrt((0 - cam.cx) ** 2 + (0 - cam.cy) ** 2),
                np.sqrt((cam.width - cam.cx) ** 2 + (0 - cam.cy) ** 2),
                np.sqrt((0 - cam.cx) ** 2 + (cam.height - cam.cy) ** 2),
                np.sqrt((cam.width - cam.cx) ** 2 + (cam.height - cam.cy) ** 2),
            )
            rs = r_samples * r_max
            theta_samples = _eval_poly_np(poly, rs)
            valid = theta_samples > 1e-6
            if np.any(valid):
                coeffs = np.polyfit(theta_samples[valid], rs[valid], deg=5)
                fw_poly = coeffs[::-1] * poly_scale
                fw_poly[0] = 0.0
            else:
                fw_k1 = (1.0 / poly[1]) * poly_scale
                fw_poly = np.array([0.0, fw_k1, 0.0, 0.0, 0.0, 0.0])
        else:
            fw_poly = np.array([p * poly_scale for p in poly[:6]])

        fw_np[i, : len(fw_poly)] = fw_poly[:6]

        corners_dx = np.array([0 - cx, img_w - cx, 0 - cx, img_w - cx])
        corners_dy = np.array([0 - cy, 0 - cy, img_h - cy, img_h - cy])
        r_max_corner = np.sqrt(corners_dx**2 + corners_dy**2).max()

        if not cam.is_bw_poly:
            theta_low, theta_high = 0.0, np.pi
            for _ in range(50):
                theta_mid = (theta_low + theta_high) / 2
                r_mid = _eval_poly_scalar(poly, theta_mid)
                if r_mid < r_max_corner:
                    theta_low = theta_mid
                else:
                    theta_high = theta_mid
            theta_corner = theta_mid
        else:
            theta_corner = _eval_poly_scalar(cam.poly, float(r_max_corner))

        mra_np[i] = abs(theta_corner) * 1.05

        ld = cam.linear_cde
        ld_np[i] = [[ld[0], ld[1]], [ld[2], 1.0]]
        s2r_np[i] = cam.sensor_to_rig.numpy()
        camera_name_to_id[cam.name] = i
        cam_names.append(cam.name)

    pp_t = torch.from_numpy(pp_np).to(device)
    sz_t = torch.from_numpy(sz_np).to(device)
    fw_t = torch.from_numpy(fw_np).to(device)
    ld_t = torch.from_numpy(ld_np).to(device)
    s2r_t = torch.from_numpy(s2r_np).to(device)

    camera_list = []
    sensor_to_rig_map = {}
    for i in range(n):
        ftheta = FThetaCamera(
            principal_point=pp_t[i],
            image_size=sz_t[i],
            fw_poly=fw_t[i],
            max_ray_angle=float(mra_np[i]),
            linear_distortion=ld_t[i],
            depth_max=200.0,
        )
        camera_list.append(ftheta)
        sensor_to_rig_map[cam_names[i]] = s2r_t[i]

    return camera_list, camera_name_to_id, sensor_to_rig_map


def _cameras_to_ftheta(
    cameras: List[CameraData],
    device: torch.device,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[List[FThetaCamera], Dict[str, int], Dict[str, Tensor]]:
    """Convert cameras to FThetaCamera list.

    Uses a C++ extension (GIL-free) for the heavy polyfit + bisection work.

    Returns:
        Tuple of:
        - List of FThetaCamera
        - camera_name -> id mapping
        - camera_name -> sensor_to_rig mapping
    """
    n = len(cameras)

    max_poly_len = max(len(cam.poly) for cam in cameras)
    poly_coeffs = np.zeros((n, max_poly_len), dtype=np.float64)
    poly_lengths = np.empty(n, dtype=np.int32)
    is_bw = np.empty(n, dtype=np.bool_)
    cx_raw = np.empty(n, dtype=np.float64)
    cy_raw = np.empty(n, dtype=np.float64)
    w_raw = np.empty(n, dtype=np.float64)
    h_raw = np.empty(n, dtype=np.float64)
    cx_sc = np.empty(n, dtype=np.float64)
    cy_sc = np.empty(n, dtype=np.float64)
    w_sc = np.empty(n, dtype=np.float64)
    h_sc = np.empty(n, dtype=np.float64)
    pscale = np.empty(n, dtype=np.float64)

    pp_np = np.empty((n, 2), dtype=np.float32)
    sz_np = np.empty((n, 2), dtype=np.float32)
    ld_np = np.zeros((n, 2, 2), dtype=np.float32)
    s2r_np = np.empty((n, 4, 4), dtype=np.float32)

    camera_name_to_id = {}
    cam_names = []

    for i, cam in enumerate(cameras):
        plen = len(cam.poly)
        poly_coeffs[i, :plen] = cam.poly[:plen]
        poly_lengths[i] = plen
        is_bw[i] = cam.is_bw_poly
        cx_raw[i] = cam.cx
        cy_raw[i] = cam.cy
        w_raw[i] = float(cam.width)
        h_raw[i] = float(cam.height)

        if target_resolution:
            tw, th = target_resolution
            sx, sy = tw / cam.width, th / cam.height
            cx_sc[i] = cam.cx * sx
            cy_sc[i] = cam.cy * sy
            w_sc[i] = float(tw)
            h_sc[i] = float(th)
            pscale[i] = (sx + sy) / 2
        else:
            cx_sc[i] = cam.cx
            cy_sc[i] = cam.cy
            w_sc[i] = float(cam.width)
            h_sc[i] = float(cam.height)
            pscale[i] = 1.0

        pp_np[i] = [cx_sc[i], cy_sc[i]]
        sz_np[i] = [w_sc[i], h_sc[i]]
        ld = cam.linear_cde
        ld_np[i] = [[ld[0], ld[1]], [ld[2], 1.0]]
        s2r_np[i] = cam.sensor_to_rig.numpy()
        camera_name_to_id[cam.name] = i
        cam_names.append(cam.name)

    ext = _get_camera_cpp_ext()
    fw_t_cpu, mra_t_cpu = ext.compute_camera_params(
        torch.from_numpy(poly_coeffs),
        torch.from_numpy(poly_lengths),
        torch.from_numpy(is_bw),
        torch.from_numpy(cx_raw),
        torch.from_numpy(cy_raw),
        torch.from_numpy(w_raw),
        torch.from_numpy(h_raw),
        torch.from_numpy(cx_sc),
        torch.from_numpy(cy_sc),
        torch.from_numpy(w_sc),
        torch.from_numpy(h_sc),
        torch.from_numpy(pscale),
    )

    pp_t = torch.from_numpy(pp_np).to(device)
    sz_t = torch.from_numpy(sz_np).to(device)
    fw_t = fw_t_cpu.to(device)
    ld_t = torch.from_numpy(ld_np).to(device)
    s2r_t = torch.from_numpy(s2r_np).to(device)
    mra_np = mra_t_cpu.numpy()

    camera_list = []
    sensor_to_rig_map = {}
    for i in range(n):
        ftheta = FThetaCamera(
            principal_point=pp_t[i],
            image_size=sz_t[i],
            fw_poly=fw_t[i],
            max_ray_angle=float(mra_np[i]),
            linear_distortion=ld_t[i],
            depth_max=200.0,
        )
        camera_list.append(ftheta)
        sensor_to_rig_map[cam_names[i]] = s2r_t[i]

    return camera_list, camera_name_to_id, sensor_to_rig_map


# ============================================================================
# Main scene class and loader
# ============================================================================


@dataclass
class ClipgtGpuScene:
    """GPU-native scene from clipgt format.

    Attributes:
        timestamped_scene: Scene data for GPU rendering
        cameras: List of FThetaCamera intrinsics
        camera_name_to_id: Mapping from camera name to index
        sensor_to_rig: Mapping from camera name to sensor-to-rig transform
        ego_track: Ego track data for pose computation
        device: Device tensors are on
        packed: Pre-packed flat buffers for fast GL upload (optional)
    """

    timestamped_scene: TimestampedScene
    cameras: List[FThetaCamera]
    camera_name_to_id: Dict[str, int]
    sensor_to_rig: Dict[str, Tensor]
    ego_track: EgoTrackData
    device: torch.device
    packed: object = None  # Optional[PackedSceneBuffers], avoid circular import

    def get_camera_id(self, camera_name: str) -> int:
        """Get camera ID by name."""
        return self.camera_name_to_id[camera_name]

    def get_camera_ids(self, camera_names: List[str]) -> List[int]:
        """Get camera IDs by names."""
        return [self.camera_name_to_id[name] for name in camera_names]

    def get_timestamps(self) -> Tensor:
        """Get all ego track timestamps."""
        return self.ego_track.timestamps

    def get_ego_poses_at_timestamps(
        self,
        timestamps_us: Tensor,
        camera_names: List[str],
    ) -> Tensor:
        """Compute camera poses (world-to-camera) for given timestamps and cameras.

        Uses linear interpolation for ego poses.

        Args:
            timestamps_us: Timestamps to query [n_timestamps]
            camera_names: Camera names to get poses for

        Returns:
            Camera poses [n_timestamps * n_cameras, 4, 4]
        """
        timestamps_us = timestamps_us.to(self.device)
        n_timestamps = len(timestamps_us)
        n_cameras = len(camera_names)

        poses = []
        for ts in timestamps_us:
            # Interpolate ego pose
            ego_to_world = self._interpolate_ego_pose(ts.item())

            for cam_name in camera_names:
                sensor_to_rig = self.sensor_to_rig[cam_name]
                camera_to_world = ego_to_world @ sensor_to_rig
                world_to_camera = torch.linalg.inv(camera_to_world)
                poses.append(world_to_camera)

        return torch.stack(poses)

    def _interpolate_ego_pose(self, timestamp_us: int) -> Tensor:
        """Interpolate ego pose at timestamp."""
        ego_ts = self.ego_track.timestamps
        if len(ego_ts) == 0:
            return torch.eye(4, dtype=torch.float32, device=self.device)

        idx = torch.searchsorted(ego_ts, timestamp_us)
        if idx == 0:
            idx = 1
        elif idx >= len(ego_ts):
            idx = len(ego_ts) - 1

        t0, t1 = ego_ts[idx - 1].item(), ego_ts[idx].item()
        if t1 == t0:
            alpha = 0.0
        else:
            alpha = (timestamp_us - t0) / (t1 - t0)

        trans0 = self.ego_track.translations[idx - 1]
        trans1 = self.ego_track.translations[idx]
        trans = trans0 * (1 - alpha) + trans1 * alpha

        quat0 = self.ego_track.quaternions[idx - 1]
        quat1 = self.ego_track.quaternions[idx]
        quat = quat0 * (1 - alpha) + quat1 * alpha
        quat = quat / quat.norm()

        # Build transform
        rot = R.from_quat(quat.cpu().numpy()).as_matrix()
        transform = torch.eye(4, dtype=torch.float32, device=self.device)
        transform[:3, :3] = torch.from_numpy(rot.astype(np.float32)).to(self.device)
        transform[:3, 3] = trans

        return transform


# Elements excluded by default (structural data, not visual road markings)
# - lane_boundary: structural data from lane.parquet rails, not visual
# - intersection_area: grey ground polygons not rendered in reference
# - road_island: green ground polygons not rendered in reference
EXCLUDED_BY_DEFAULT = {"lane_boundary", "intersection_area", "road_island"}


def load_clipgt_scene(
    scene_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    target_resolution: Optional[Tuple[int, int]] = None,
    exclude_elements: Optional[set] = None,
    include_ego_trajectory: bool = True,
    include_ego_obstacle: bool = False,
    simplify_dual_lane_lines: bool = True,
    verbose: bool = False,
) -> ClipgtGpuScene:
    """Load a clipgt scene directory directly to GPU-native format.

    Args:
        scene_dir: Path to clipgt scene directory
        device: Device to place tensors on
        target_resolution: Optional (width, height) to scale cameras to
        exclude_elements: Set of element names to exclude. Defaults to EXCLUDED_BY_DEFAULT
                         which excludes lane_boundary, intersection_area, road_island.
                         Pass empty set() to include all elements.
        include_ego_trajectory: If True (default), include ego trajectory polyline (green path)
        include_ego_obstacle: If True, include ego vehicle as a cube (for BEV rendering)
        simplify_dual_lane_lines: If True (default), simplify SOLID_DASHED/DASHED_SOLID
                         lane patterns to a single solid line. If False, render both lines
                         (dual lane rendering).
        verbose: If True (default), print loading progress. Set False to suppress output.

    Returns:
        ClipgtGpuScene ready for GPU rendering
    """
    if isinstance(device, str):
        device = torch.device(device)

    scene_dir = Path(scene_dir)

    # Use default exclusions if not specified
    if exclude_elements is None:
        exclude_elements = EXCLUDED_BY_DEFAULT

    # Detect clip_id prefix for parquet files
    def get_parquet_path(element_type: str) -> Path:
        simple_path = scene_dir / f"{element_type}.parquet"
        if simple_path.exists():
            return simple_path
        matches = list(scene_dir.glob(f"*.{element_type}.parquet"))
        if matches:
            return matches[0]
        return simple_path

    # Read all parquet files
    if verbose:
        print(f"Loading clipgt scene from: {scene_dir}")
        if exclude_elements:
            print(f"  Excluding: {exclude_elements}")

    road_boundary = read_road_boundary(get_parquet_path("road_boundary"))
    lane_boundary = read_lane(get_parquet_path("lane")) if "lane_boundary" not in exclude_elements else []
    lane_line_data = read_lane_line(
        get_parquet_path("lane_line"),
        simplify_dual_lines=simplify_dual_lane_lines,
    )
    crosswalk = read_crosswalk(get_parquet_path("crosswalk"))
    road_marking = read_road_marking(get_parquet_path("road_marking"))
    wait_line = read_wait_line(get_parquet_path("wait_line"))
    pole = read_pole(get_parquet_path("pole"))
    intersection_area = (
        read_intersection_area(get_parquet_path("intersection_area"))
        if "intersection_area" not in exclude_elements
        else []
    )
    road_island = read_road_island(get_parquet_path("road_island")) if "road_island" not in exclude_elements else []
    traffic_light = read_traffic_light(get_parquet_path("traffic_light"))
    traffic_sign = read_traffic_sign(get_parquet_path("traffic_sign"))
    obstacles = read_obstacles(get_parquet_path("obstacle"))
    ego_track = read_egomotion_estimate(get_parquet_path("egomotion_estimate"))
    cameras = read_calibration_estimate(get_parquet_path("calibration_estimate"))

    # Convert to GPU pools
    polyline_pools = []

    # Road boundary
    pool = _polylines_to_pool(road_boundary, PRIM_ROAD_BOUNDARY, device)
    if pool:
        polyline_pools.append(pool)

    # Lane boundary (excluded by default)
    if "lane_boundary" not in exclude_elements:
        pool = _polylines_to_pool(lane_boundary, PRIM_LANE_BOUNDARY, device)
        if pool:
            polyline_pools.append(pool)

    # Lane lines by color/style
    pool = _polylines_to_pool(lane_line_data.white_solid, PRIM_LANE_LINE_WHITE_SOLID, device)
    if pool:
        polyline_pools.append(pool)
    pool = _polylines_to_pool(lane_line_data.white_dashed, PRIM_LANE_LINE_WHITE_DASHED, device)
    if pool:
        polyline_pools.append(pool)
    pool = _polylines_to_pool(lane_line_data.yellow_solid, PRIM_LANE_LINE_YELLOW_SOLID, device)
    if pool:
        polyline_pools.append(pool)
    pool = _polylines_to_pool(lane_line_data.yellow_dashed, PRIM_LANE_LINE_YELLOW_DASHED, device)
    if pool:
        polyline_pools.append(pool)

    # Yellow dots (DOT_SOLID style - rendered as circles)
    pool = _polylines_to_pool(lane_line_data.yellow_dots, PRIM_DOT_YELLOW, device)
    if pool:
        polyline_pools.append(pool)

    # Wait line
    pool = _polylines_to_pool(wait_line, PRIM_WAIT_LINE, device)
    if pool:
        polyline_pools.append(pool)

    # Pole
    pool = _polylines_to_pool(pole, PRIM_POLE, device)
    if pool:
        polyline_pools.append(pool)

    # Ego trajectory (green path showing full ego motion)
    if include_ego_trajectory:
        pool = _ego_trajectory_to_pool(ego_track, device)
        if pool:
            polyline_pools.append(pool)

    # Polygons
    polygon_pools = []

    pool = _polygons_to_pool(crosswalk, PRIM_CROSSWALK, device)
    if pool:
        polygon_pools.append(pool)

    pool = _polygons_to_pool(road_marking, PRIM_ROAD_MARKING, device)
    if pool:
        polygon_pools.append(pool)

    # Intersection area (excluded by default)
    if "intersection_area" not in exclude_elements:
        pool = _polygons_to_pool(intersection_area, PRIM_INTERSECTION, device)
        if pool:
            polygon_pools.append(pool)

    # Road island (excluded by default)
    if "road_island" not in exclude_elements:
        pool = _polygons_to_pool(road_island, PRIM_ROAD_ISLAND, device)
        if pool:
            polygon_pools.append(pool)

    # Cubes
    cube_pools = []

    # Ego obstacle (only for BEV mode - shows ego vehicle as a cube)
    if include_ego_obstacle:
        ego_cube_pool = _ego_obstacle_to_pool(ego_track, device)
        if ego_cube_pool:
            cube_pools.append(ego_cube_pool)

    # Dynamic obstacles
    obstacle_pool = _obstacles_to_pool(obstacles, device)
    if obstacle_pool:
        cube_pools.append(obstacle_pool)

    # Static cubes (traffic lights, signs) - colors match reference v3 scheme
    # traffic_light: [100, 100, 100] = Gray
    pool = _static_cubes_to_pool(traffic_light, PRIM_TRAFFIC_LIGHT, device, color=(100 / 255, 100 / 255, 100 / 255))
    if pool:
        cube_pools.append(pool)

    # traffic_sign: [8, 2, 255] = Blue
    pool = _static_cubes_to_pool(traffic_sign, PRIM_TRAFFIC_SIGN, device, color=(8 / 255, 2 / 255, 255 / 255))
    if pool:
        cube_pools.append(pool)

    # Convert cameras
    camera_list, camera_name_to_id, sensor_to_rig = _cameras_to_ftheta(cameras, device, target_resolution)

    # Move ego track to device
    ego_track = EgoTrackData(
        timestamps=ego_track.timestamps.to(device),
        poses_tquat=ego_track.poses_tquat.to(device),
    )

    # Build scene
    timestamped_scene = TimestampedScene(
        polyline_pools=polyline_pools,
        polygon_pools=polygon_pools,
        cube_pools=cube_pools if cube_pools else None,
    )

    if verbose:
        print(
            f"  Loaded: {len(polyline_pools)} polyline pools, {len(polygon_pools)} polygon pools, {len(cube_pools)} cube pools"
        )
        print(f"  Cameras: {list(camera_name_to_id.keys())}")
        print(f"  Ego timestamps: {len(ego_track.timestamps)} frames")

    return ClipgtGpuScene(
        timestamped_scene=timestamped_scene,
        cameras=camera_list,
        camera_name_to_id=camera_name_to_id,
        sensor_to_rig=sensor_to_rig,
        ego_track=ego_track,
        device=device,
    )


# ============================================================================
# AV2 scene loading
# ============================================================================
# AV2 scenes use timestamped parquet files with different column names than
# clipgt (e.g. cf_road_boundary, dw_lane_line, cf_crosswalks).
# Each observation row has a key["timestamp_micros"] field.
# ============================================================================


# --- Flat polyline representation for GPU-assisted loading ---


@dataclass
class FlatPolylineData:
    """Flat representation of timestamped polylines as contiguous arrays.

    Replaces Dict[int, List[Tensor]] for the GPU path — avoids a
    round-trip through Python dicts and enables batched GPU transforms.
    """

    timestamps_us: Tensor  # [n_rows] int64, per-row timestamp (may have dupes)
    vertices: Tensor  # [n_total_verts, 3] float32
    row_offsets: Tensor  # [n_rows + 1] int32, vertex start/end per row
    unique_timestamps: Optional[Tensor] = None  # [n_unique] int64, pre-computed
    ts_counts_prefix_sum: Optional[Tensor] = None  # [n_unique] int32, pre-computed

    def to(self, device: torch.device) -> "FlatPolylineData":
        return FlatPolylineData(
            timestamps_us=self.timestamps_us.to(device),
            vertices=self.vertices.to(device),
            row_offsets=self.row_offsets.to(device),
            unique_timestamps=self.unique_timestamps.to(device) if self.unique_timestamps is not None else None,
            ts_counts_prefix_sum=self.ts_counts_prefix_sum.to(device)
            if self.ts_counts_prefix_sum is not None
            else None,
        )


def _read_polyline_parquet_flat(
    table,
    data_col_name: str,
    points_field_name: str,
    min_points: int = 2,
) -> Optional[FlatPolylineData]:
    """Read a list<struct<x,y,z>> column into FlatPolylineData.

    Returns None if no valid rows remain after filtering.
    """
    key_col = table.column("key").combine_chunks()
    ts_arr = key_col.field("timestamp_micros").to_numpy()

    data_col = table.column(data_col_name).combine_chunks()
    points_col = data_col.field(points_field_name)
    offsets = points_col.offsets.to_numpy()
    vals = points_col.values
    all_x = vals.field("x").to_numpy(zero_copy_only=False)
    all_y = vals.field("y").to_numpy(zero_copy_only=False)
    all_z = vals.field("z").to_numpy(zero_copy_only=False)
    flat_verts = np.column_stack([all_x, all_y, all_z]).astype(np.float32)

    lengths = np.diff(offsets).astype(np.intp)
    valid = lengths >= min_points

    # Check for NaN only in rows that pass the length filter
    nan_check_idx = np.where(valid)[0]
    for i in nan_check_idx:
        s, e = int(offsets[i]), int(offsets[i + 1])
        if np.isnan(flat_verts[s:e]).any():
            valid[i] = False

    if not valid.any():
        return None

    valid_idx = np.where(valid)[0]
    valid_starts = offsets[valid_idx].astype(np.intp)
    valid_lengths = lengths[valid_idx]

    if len(valid_idx) == len(ts_arr) and valid_lengths.sum() == len(flat_verts):
        # All rows valid and contiguous — skip compaction
        compact_verts = flat_verts
    else:
        # Vectorized gather: build flat index array without Python loop
        total = valid_lengths.sum()
        cumlen = np.cumsum(valid_lengths)
        new_starts = np.empty(len(valid_lengths), dtype=np.intp)
        new_starts[0] = 0
        if len(cumlen) > 1:
            new_starts[1:] = cumlen[:-1]
        within = np.arange(total, dtype=np.intp) - np.repeat(new_starts, valid_lengths)
        gather = np.repeat(valid_starts, valid_lengths) + within
        compact_verts = flat_verts[gather]

    new_offsets = np.zeros(len(valid_idx) + 1, dtype=np.int32)
    np.cumsum(valid_lengths, out=new_offsets[1:])

    return FlatPolylineData(
        timestamps_us=torch.from_numpy(ts_arr[valid_idx].astype(np.int64).copy()),
        vertices=torch.from_numpy(compact_verts.copy()),
        row_offsets=torch.from_numpy(new_offsets),
    )


# --- AV2 parquet readers ---
# All readers use pyarrow.parquet.read_table for direct struct-field access.


# DEPRECATED: JSON road_boundary field is in ego frame and requires ego-to-world
# transform, which is fragile (fails on ~3% of scenes with bad ego data).
# The struct field road_boundary_polyline is already in world frame and is
# bit-identical to correctly-transformed JSON output. Use struct path instead.
#
# def _read_av2_road_boundary_json(fh: BinaryIO) -> Optional[FlatPolylineData]:
#     """Read AV2 road boundaries from the JSON string field in cf_road_boundary.parquet.
#
#     The parquet file has two representations of road boundary vertices:
#     - ``road_boundary``: a JSON string with a ``polyline`` array (original source)
#     - ``road_boundary_polyline``: a struct<x,y,z> list column
#
#     These contain different coordinates. The JSON field matches the original
#     AV2 loader behavior and is the correct source for ego-frame vertices
#     that need ego-to-world transformation.
#     """
#     table = pq.read_table(fh, columns=["key", "cf_road_boundary"])
#     if len(table) == 0 or "key" not in table.schema.names:
#         return None
#
#     key_col = table.column("key").combine_chunks()
#     ts_arr = key_col.field("timestamp_micros").to_numpy()
#     data_col = table.column("cf_road_boundary").combine_chunks()
#     rb_json_col = data_col.field("road_boundary")
#
#     all_verts = []
#     all_offsets = [0]
#     valid_ts = []
#
#     for i in range(len(table)):
#         json_str = rb_json_col[i].as_py()
#         if json_str is None:
#             continue
#         parsed = json.loads(json_str)
#         polyline = parsed.get("polyline", [])
#         if len(polyline) < 2:
#             continue
#         pts = np.array([[pt["x"], pt["y"], pt["z"]] for pt in polyline],
#                        dtype=np.float32)
#         all_verts.append(pts)
#         all_offsets.append(all_offsets[-1] + len(pts))
#         valid_ts.append(ts_arr[i])
#
#     if not all_verts:
#         return None
#
#     return FlatPolylineData(
#         timestamps_us=torch.tensor(valid_ts, dtype=torch.int64),
#         vertices=torch.from_numpy(np.concatenate(all_verts)),
#         row_offsets=torch.tensor(all_offsets, dtype=torch.int32),
#     )


def _read_av2_egomotion(fh: BinaryIO) -> EgoTrackData:
    """Read AV2 egomotion from egomotion_estimate.parquet.

    Fully vectorized via pyarrow struct-field access — no row iteration.
    """
    table = pq.read_table(fh, columns=["key", "egomotion_estimate"])
    if len(table) == 0:
        return EgoTrackData(
            timestamps=torch.zeros(0, dtype=torch.int64),
            poses_tquat=torch.zeros(0, 7, dtype=torch.float32),
        )

    key_col = table.column("key").combine_chunks()
    ego_col = table.column("egomotion_estimate").combine_chunks()
    ts = key_col.field("timestamp_micros").to_numpy()
    loc = ego_col.field("location")
    ori = ego_col.field("orientation")

    tquats = np.column_stack(
        [
            loc.field("x").to_numpy(zero_copy_only=False),
            loc.field("y").to_numpy(zero_copy_only=False),
            loc.field("z").to_numpy(zero_copy_only=False),
            ori.field("x").to_numpy(zero_copy_only=False),
            ori.field("y").to_numpy(zero_copy_only=False),
            ori.field("z").to_numpy(zero_copy_only=False),
            ori.field("w").to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32)

    return EgoTrackData(
        timestamps=torch.from_numpy(ts.astype(np.int64).copy()),
        poses_tquat=torch.from_numpy(tquats),
    )


def _read_av2_obstacles(fh: BinaryIO) -> List[ObstacleData]:
    """Read AV2 dynamic obstacles from object_fused.parquet.

    Vectorized: extracts all columns via pyarrow in one pass, computes
    yaw-to-quaternion without scipy, and groups by obstacle_id with
    numpy argsort + split.
    """
    table = pq.read_table(fh, columns=["key", "object_fused"])
    if len(table) == 0:
        return []

    obj_col = table.column("object_fused").combine_chunks()
    key_col = table.column("key").combine_chunks()

    obstacle_ids = obj_col.field("obstacle_id").to_numpy(zero_copy_only=False)
    ts_all = key_col.field("timestamp_micros").to_numpy()

    center = obj_col.field("cuboid_3D_center")
    cx = center.field("x").to_numpy(zero_copy_only=False)
    cy = center.field("y").to_numpy(zero_copy_only=False)
    cz = center.field("z").to_numpy(zero_copy_only=False)

    direction = obj_col.field("obstacle_direction")
    dx = direction.field("x").to_numpy(zero_copy_only=False)
    dy = direction.field("y").to_numpy(zero_copy_only=False)

    half_axis = obj_col.field("cuboid_3D_halfAxisXYZ")
    hx = half_axis.field("x").to_numpy(zero_copy_only=False)
    hy = half_axis.field("y").to_numpy(zero_copy_only=False)
    hz = half_axis.field("z").to_numpy(zero_copy_only=False)

    obs_class = obj_col.field("obstacle_class").to_numpy(zero_copy_only=False)

    # Vectorized yaw -> quaternion (z-axis rotation only)
    yaw = np.arctan2(dy, dx)
    half_yaw = (yaw * 0.5).astype(np.float32)
    qx = np.zeros(len(yaw), dtype=np.float32)
    qy = np.zeros(len(yaw), dtype=np.float32)
    qz = np.sin(half_yaw)
    qw = np.cos(half_yaw)

    tquats_all = np.column_stack(
        [
            cx,
            cy,
            cz,
            qx,
            qy,
            qz,
            qw,
        ]
    ).astype(np.float32)

    # Filter rows with valid obstacle_id
    valid = ~np.isnan(obstacle_ids.astype(np.float64))
    if not valid.all():
        idx = np.where(valid)[0]
        obstacle_ids = obstacle_ids[idx]
        ts_all = ts_all[idx]
        tquats_all = tquats_all[idx]
        hx, hy, hz = hx[idx], hy[idx], hz[idx]
        obs_class = obs_class[idx]

    # Group by obstacle_id using argsort
    order = np.argsort(obstacle_ids, kind="stable")
    sorted_ids = obstacle_ids[order]
    unique_ids, group_starts = np.unique(sorted_ids, return_index=True)
    group_ends = np.append(group_starts[1:], len(sorted_ids))

    obstacles = []
    for i in range(len(unique_ids)):
        s, e = group_starts[i], group_ends[i]
        if e - s < 2:
            continue
        grp = order[s:e]

        last = grp[-1]
        size = torch.tensor(
            [hx[last] * 2.0, hy[last] * 2.0, hz[last] * 2.0],
            dtype=torch.float32,
        )
        category = str(obs_class[last])

        obstacles.append(
            ObstacleData(
                trackline_id=str(int(unique_ids[i])),
                category=category,
                size=size,
                timestamps=torch.from_numpy(ts_all[grp].astype(np.int64).copy()),
                poses_tquat=torch.from_numpy(tquats_all[grp].copy()),
            )
        )

    return obstacles


# --- Flat pool builders (GPU-compatible) ---


def _build_sorted_vertex_gather(
    row_offsets: Tensor,
    sorted_idx: Tensor,
) -> Tensor:
    """Build a flat gather index to reorder vertices by sorted row order.

    Uses numpy on CPU (where torch.repeat_interleave is slow) and
    torch on CUDA.
    """
    vert_counts = row_offsets[1:] - row_offsets[:-1]
    sorted_counts = vert_counts[sorted_idx]
    sorted_starts = row_offsets[:-1][sorted_idx]

    if row_offsets.device.type == "cpu":
        sc = sorted_counts.numpy().astype(np.intp)
        ss = sorted_starts.numpy().astype(np.intp)
        total = sc.sum()
        cumlen = np.cumsum(sc)
        ns = np.empty(len(sc), dtype=np.intp)
        ns[0] = 0
        if len(cumlen) > 1:
            ns[1:] = cumlen[:-1]
        within = np.arange(total, dtype=np.intp) - np.repeat(ns, sc)
        bases = np.repeat(ss, sc)
        return torch.from_numpy((bases + within).astype(np.int64))

    device = row_offsets.device
    total = sorted_counts.sum().item()
    cum = torch.cumsum(sorted_counts, dim=0)
    new_starts = torch.zeros_like(cum)
    new_starts[1:] = cum[:-1]
    within = torch.arange(total, device=device) - new_starts.repeat_interleave(sorted_counts)
    bases = sorted_starts.repeat_interleave(sorted_counts)
    return bases + within


def _flat_polylines_to_pool(
    data: FlatPolylineData,
    prim_type_id: int,
    device: torch.device,
    pre_sorted: bool = False,
) -> Optional[TimestampedPolylinePool]:
    """Build a TimestampedPolylinePool from FlatPolylineData.

    Uses numpy on CPU for speed. Skips sort when timestamps are already ordered.
    When pre_sorted=True (GPU decoder output), skips the sort check to avoid
    a GPU sync and uses row_offsets directly instead of diff+cumsum.
    """
    if data is None or data.vertices.shape[0] == 0:
        return None

    if device.type == "cpu":
        ts_np = data.timestamps_us.numpy()
        offsets_np = data.row_offsets.numpy()
        vc_np = np.diff(offsets_np).astype(np.int32)
        already_sorted = pre_sorted or len(ts_np) <= 1 or np.all(ts_np[:-1] <= ts_np[1:])

        if already_sorted:
            unique_ts_np, counts_np = np.unique(ts_np, return_counts=True)
            varrays_ps = np.cumsum(vc_np)
            vertices = data.vertices
        else:
            order = np.argsort(ts_np, kind="stable")
            sorted_ts = ts_np[order]
            unique_ts_np, counts_np = np.unique(sorted_ts, return_counts=True)
            sorted_vc = vc_np[order]
            varrays_ps = np.cumsum(sorted_vc)
            gather = _build_sorted_vertex_gather(data.row_offsets, torch.from_numpy(order.astype(np.int64)))
            vertices = data.vertices[gather]

        ts_ps = np.cumsum(counts_np.astype(np.int32))

        return TimestampedPolylinePool(
            timestamps_us=torch.from_numpy(unique_ts_np.copy()),
            timestamped_varrays_prefix_sum=torch.from_numpy(ts_ps),
            varrays_prefix_sum=torch.from_numpy(varrays_ps),
            vertices=vertices,
            prim_type_id=prim_type_id,
        )

    d = data.to(device)

    if pre_sorted:
        if d.unique_timestamps is not None and d.ts_counts_prefix_sum is not None:
            unique_ts = d.unique_timestamps
            ts_ps = d.ts_counts_prefix_sum
        else:
            unique_ts, counts = torch.unique_consecutive(d.timestamps_us, return_counts=True)
            ts_ps = torch.cumsum(counts.int(), dim=0)
        return TimestampedPolylinePool(
            timestamps_us=unique_ts,
            timestamped_varrays_prefix_sum=ts_ps,
            varrays_prefix_sum=d.row_offsets[1:],
            vertices=d.vertices,
            prim_type_id=prim_type_id,
        )

    vert_counts = d.row_offsets[1:] - d.row_offsets[:-1]

    is_sorted = len(d.timestamps_us) <= 1 or bool(torch.all(d.timestamps_us[:-1] <= d.timestamps_us[1:]).item())

    if is_sorted:
        unique_ts, counts = torch.unique_consecutive(d.timestamps_us, return_counts=True)
        varrays_prefix_sum = torch.cumsum(vert_counts, dim=0)
        vertices = d.vertices
    else:
        sorted_idx = torch.argsort(d.timestamps_us, stable=True)
        sorted_ts = d.timestamps_us[sorted_idx]
        unique_ts, counts = torch.unique_consecutive(sorted_ts, return_counts=True)
        sorted_vert_counts = vert_counts[sorted_idx]
        varrays_prefix_sum = torch.cumsum(sorted_vert_counts, dim=0)
        gather_idx = _build_sorted_vertex_gather(d.row_offsets, sorted_idx)
        vertices = d.vertices[gather_idx]

    return TimestampedPolylinePool(
        timestamps_us=unique_ts,
        timestamped_varrays_prefix_sum=torch.cumsum(counts.int(), dim=0),
        varrays_prefix_sum=varrays_prefix_sum,
        vertices=vertices,
        prim_type_id=prim_type_id,
    )


def _flat_polygons_to_pool(
    data: FlatPolylineData,
    prim_type_id: int,
    device: torch.device,
) -> Optional[TimestampedPolygonPool]:
    """Build a TimestampedPolygonPool from FlatPolylineData.

    Fan-triangulates each polygon. Uses numpy on CPU for speed.
    """
    if data is None or data.vertices.shape[0] == 0:
        return None

    if device.type == "cpu":
        ts_np_raw = data.timestamps_us.numpy()
        offsets_np = data.row_offsets.numpy()
        vc_np = np.diff(offsets_np).astype(np.int32)
        already_sorted = len(ts_np_raw) <= 1 or np.all(ts_np_raw[:-1] <= ts_np_raw[1:])

        if already_sorted:
            unique_ts_np, counts_np = np.unique(ts_np_raw, return_counts=True)
            final_vc = vc_np
            vertices = data.vertices
        else:
            order = np.argsort(ts_np_raw, kind="stable")
            sorted_ts = ts_np_raw[order]
            unique_ts_np, counts_np = np.unique(sorted_ts, return_counts=True)
            final_vc = vc_np[order]
            gather = _build_sorted_vertex_gather(data.row_offsets, torch.from_numpy(order.astype(np.int64)))
            vertices = data.vertices[gather]

        tri_c = np.maximum(final_vc - 2, 0)
        total_tris = int(tri_c.sum())

        if total_tris > 0:
            tri_ps = np.cumsum(tri_c)
            tri_starts = np.empty(len(tri_c), dtype=np.intp)
            tri_starts[0] = 0
            if len(tri_ps) > 1:
                tri_starts[1:] = tri_ps[:-1]
            within = np.arange(total_tris, dtype=np.int32) - np.repeat(tri_starts, tri_c).astype(np.int32)
            triangles_np = np.stack(
                [
                    np.zeros(total_tris, dtype=np.int32),
                    within + 1,
                    within + 2,
                ],
                axis=1,
            )
        else:
            triangles_np = np.zeros((0, 3), dtype=np.int32)
            tri_ps = np.cumsum(tri_c)

        return TimestampedPolygonPool(
            timestamps_us=torch.from_numpy(unique_ts_np.copy()),
            timestamped_varrays_prefix_sum=torch.from_numpy(np.cumsum(counts_np.astype(np.int32))),
            varrays_prefix_sum=torch.from_numpy(np.cumsum(final_vc)),
            triangle_prefix_sum=torch.from_numpy(tri_ps.astype(np.int32)),
            vertices=vertices,
            triangles=torch.from_numpy(triangles_np),
            prim_type_id=prim_type_id,
        )

    d = data.to(device)
    vert_counts = d.row_offsets[1:] - d.row_offsets[:-1]

    is_sorted = len(d.timestamps_us) <= 1 or bool(torch.all(d.timestamps_us[:-1] <= d.timestamps_us[1:]).item())

    if is_sorted:
        unique_ts, counts = torch.unique_consecutive(d.timestamps_us, return_counts=True)
        final_vert_counts = vert_counts
        vertices = d.vertices
    else:
        sorted_idx = torch.argsort(d.timestamps_us, stable=True)
        sorted_ts = d.timestamps_us[sorted_idx]
        unique_ts, counts = torch.unique_consecutive(sorted_ts, return_counts=True)
        final_vert_counts = vert_counts[sorted_idx]
        gather_idx = _build_sorted_vertex_gather(d.row_offsets, sorted_idx)
        vertices = d.vertices[gather_idx]

    varrays_prefix_sum = torch.cumsum(final_vert_counts, dim=0)
    tri_counts = (final_vert_counts - 2).clamp(min=0)
    triangle_prefix_sum = torch.cumsum(tri_counts, dim=0)
    total_tris = triangle_prefix_sum[-1].item() if len(tri_counts) > 0 else 0

    if total_tris > 0:
        tri_starts = torch.zeros_like(triangle_prefix_sum)
        tri_starts[1:] = triangle_prefix_sum[:-1]
        within_tri = (torch.arange(total_tris, device=device) - tri_starts.repeat_interleave(tri_counts)).int()
        triangles = torch.stack(
            [
                torch.zeros(total_tris, device=device, dtype=torch.int32),
                within_tri + 1,
                within_tri + 2,
            ],
            dim=1,
        )
    else:
        triangles = torch.zeros(0, 3, dtype=torch.int32, device=device)

    return TimestampedPolygonPool(
        timestamps_us=unique_ts,
        timestamped_varrays_prefix_sum=torch.cumsum(counts.int(), dim=0),
        varrays_prefix_sum=varrays_prefix_sum,
        triangle_prefix_sum=triangle_prefix_sum.int(),
        vertices=vertices,
        triangles=triangles,
        prim_type_id=prim_type_id,
    )


# --- Road boundary ego-to-world transform ---


def _transform_av2_road_boundary_to_world(
    observations: Dict[int, List[Tensor]],
    ego_track: EgoTrackData,
) -> None:
    """Transform road boundary vertices from ego frame to world frame in-place.

    AV2 road boundary points are in ego coordinates (unlike lane lines which
    are already in world coordinates). This uses the ego track to compute
    ego-to-world transforms at each observation timestamp and applies them.
    """
    if not observations or len(ego_track.timestamps) == 0:
        return

    ego_ts = ego_track.timestamps
    ego_trans = ego_track.translations
    ego_quats = ego_track.quaternions

    for ts_us, varrays in observations.items():
        # Interpolate ego pose at this timestamp
        idx = torch.searchsorted(ego_ts, ts_us)
        if idx == 0:
            idx = 1
        elif idx >= len(ego_ts):
            idx = len(ego_ts) - 1

        t0, t1 = ego_ts[idx - 1].item(), ego_ts[idx].item()
        alpha = 0.0 if t1 == t0 else (ts_us - t0) / (t1 - t0)

        trans = ego_trans[idx - 1] * (1 - alpha) + ego_trans[idx] * alpha
        quat = ego_quats[idx - 1] * (1 - alpha) + ego_quats[idx] * alpha
        quat = quat / quat.norm()

        rot = R.from_quat(quat.numpy()).as_matrix()
        ego_to_world = torch.eye(4, dtype=torch.float32)
        ego_to_world[:3, :3] = torch.from_numpy(rot.astype(np.float32))
        ego_to_world[:3, 3] = trans

        for i, verts in enumerate(varrays):
            ones = torch.ones(verts.shape[0], 1, dtype=torch.float32)
            verts_homo = torch.cat([verts, ones], dim=1)
            transformed = (ego_to_world @ verts_homo.T).T[:, :3]
            varrays[i] = transformed


def _quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert quaternions (xyzw) to 3x3 rotation matrices. Works on any device."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Convert quaternions (xyzw) to 3x3 rotation matrices using numpy."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return (
        np.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
            axis=-1,
        )
        .reshape(-1, 3, 3)
        .astype(np.float32)
    )


# DEPRECATED: No longer needed — road_boundary_polyline (struct) is already in
# world frame. Kept commented for reference / future ego-frame data sources.
#
# def _transform_road_boundary_flat(
#     data: FlatPolylineData,
#     ego_track: EgoTrackData,
# ) -> None:
#     """Transform road boundary vertices from ego to world frame in-place.
#
#     On CPU: uses numpy (much faster than torch CPU for these ops).
#     On GPU: fully batched with repeat_interleave + bmm.
#     """
#     if data.vertices.shape[0] == 0 or len(ego_track.timestamps) == 0:
#         return
#
#     device = data.vertices.device
#
#     if device.type == "cpu":
#         ego_ts = ego_track.timestamps.numpy()
#         ego_trans = ego_track.translations.numpy().astype(np.float32)
#         ego_quats = ego_track.quaternions.numpy().astype(np.float32)
#         row_ts = data.timestamps_us.numpy()
#
#         idx = np.searchsorted(ego_ts, row_ts)
#         idx = np.clip(idx, 1, len(ego_ts) - 1)
#
#         t0 = ego_ts[idx - 1].astype(np.float64)
#         t1 = ego_ts[idx].astype(np.float64)
#         denom = np.maximum(t1 - t0, 1.0)
#         alpha = ((row_ts.astype(np.float64) - t0) / denom).astype(np.float32)[:, None]
#
#         trans = ego_trans[idx - 1] * (1 - alpha) + ego_trans[idx] * alpha
#         quat = ego_quats[idx - 1] * (1 - alpha) + ego_quats[idx] * alpha
#         quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
#
#         rot = _quat_to_rotmat_np(quat)  # (n_rows, 3, 3)
#
#         offsets = data.row_offsets.numpy()
#         verts = data.vertices.numpy()
#         for i in range(len(row_ts)):
#             s, e = int(offsets[i]), int(offsets[i + 1])
#             verts[s:e] = (rot[i] @ verts[s:e].T).T + trans[i]
#     else:
#         ego_ts = ego_track.timestamps.to(device)
#         ego_trans = ego_track.translations.to(device)
#         ego_quats = ego_track.quaternions.to(device)
#         row_ts = data.timestamps_us
#
#         idx = torch.searchsorted(ego_ts, row_ts)
#         idx = idx.clamp(1, len(ego_ts) - 1)
#
#         ego_ts_lo = ego_ts[idx - 1]
#         dt = (ego_ts[idx] - ego_ts_lo).clamp(min=1)
#         alpha = ((row_ts - ego_ts_lo).float() / dt.float()).unsqueeze(-1)
#
#         trans = ego_trans[idx - 1] * (1 - alpha) + ego_trans[idx] * alpha
#         quat = ego_quats[idx - 1] * (1 - alpha) + ego_quats[idx] * alpha
#         quat = quat / quat.norm(dim=-1, keepdim=True)
#         rot = _quat_to_rotmat(quat)
#
#         vert_counts = data.row_offsets[1:] - data.row_offsets[:-1]
#         vert_row_idx = torch.arange(len(row_ts), device=device).repeat_interleave(vert_counts)
#         r = rot[vert_row_idx]
#         t = trans[vert_row_idx]
#         data.vertices = (r @ data.vertices.unsqueeze(-1)).squeeze(-1) + t


# --- Main AV2 loader ---


def _load_polylines_pyarrow(loader) -> Dict[str, Optional[FlatPolylineData]]:
    """Load polyline data from parquet files using PyArrow (CPU fallback).

    Road boundary uses the struct field (road_boundary_polyline) which is
    already in world coordinates — no ego-to-world transform needed.
    """
    buf_ll = loader.open("dw_lane_line.parquet")
    buf_rb = loader.open("cf_road_boundary.parquet")
    buf_cw = loader.open("cf_crosswalks.parquet")
    buf_so = loader.open("cf_static_obstacle.parquet")

    def _parse_polyline(buf, data_col, pts_field, min_pts):
        table = pq.read_table(buf, columns=["key", data_col])
        if "key" not in table.schema.names:
            return None
        return _read_polyline_parquet_flat(table, data_col, pts_field, min_points=min_pts)

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_ll = pool.submit(_parse_polyline, buf_ll, "dw_lane_line", "points", 2)
        fut_rb = pool.submit(_parse_polyline, buf_rb, "cf_road_boundary", "road_boundary_polyline", 2)
        fut_cw = pool.submit(_parse_polyline, buf_cw, "cf_crosswalks", "crosswalk_area", 3)
        fut_so = pool.submit(_parse_polyline, buf_so, "cf_static_obstacle", "boundary_points", 2)

        return {
            "dw_lane_line.parquet": fut_ll.result(),
            "cf_road_boundary.parquet": fut_rb.result(),
            "cf_crosswalks.parquet": fut_cw.result(),
            "cf_static_obstacle.parquet": fut_so.result(),
        }


def prefetch_scene(scene_path: Union[str, Path]) -> None:
    """Start reading a scene's tar file into pinned memory on a background thread.

    Call this before load_av2_scene to overlap I/O with GPU work from the
    previous scene. Uses double-buffered pinned memory so prefetch doesn't
    conflict with any in-flight scene processing.

    Preferred: use load_av2_scene(path, prefetch_next=next_path) instead,
    which triggers prefetch at the optimal time (right after the C++ pipeline
    finishes consuming the pinned buffer).

    Example (standalone prefetch)::

        prefetch_scene(paths[0])
        for i, path in enumerate(paths):
            scene = load_av2_scene(path, device="cuda")
            if i + 1 < len(paths):
                prefetch_scene(paths[i + 1])
    """
    from .gpu_parquet import prefetch_tar

    prefetch_tar(str(scene_path))


def load_av2_scene(
    scene_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    target_resolution: Optional[Tuple[int, int]] = None,
    include_ego_trajectory: bool = True,
    include_ego_obstacle: bool = False,
    verbose: bool = False,
    use_gpu_decoder: Optional[bool] = None,
    prefetch_next: Optional[Union[str, Path]] = None,
) -> ClipgtGpuScene:
    """Load an AV2 scene (tar file or directory) to GPU-native format.

    AV2 scenes contain timestamped map elements (lane lines, road boundaries,
    crosswalks, static obstacles) plus dynamic obstacles and ego track.

    Args:
        scene_path: Path to AV2 scene tar file or directory
        device: Device to place tensors on
        target_resolution: Optional (width, height) to scale cameras to
        include_ego_trajectory: If True, include ego trajectory polyline
        include_ego_obstacle: If True, include ego vehicle cube (for BEV)
        verbose: If True (default), print loading progress. Set False to suppress output.
        use_gpu_decoder: If True, force GPU-native parquet decoding (requires
            nvcomp + CUDA device). If False, force PyArrow. If None (default),
            auto-detect: use GPU decoder when available and device is CUDA.

    Returns:
        ClipgtGpuScene ready for GPU rendering
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    scene_path = str(scene_path)

    if verbose:
        print(f"Loading AV2 scene from: {scene_path}")

    # Decide whether to use GPU-native decoder for polyline parquets
    _use_gpu = use_gpu_decoder
    if _use_gpu is None:
        from .gpu_parquet import is_gpu_parquet_available

        _use_gpu = device.type == "cuda" and is_gpu_parquet_available() and scene_path.endswith(".tar")

    _used_scene_loader = False

    if _use_gpu:
        from .gpu_parquet import TarFileEntry, read_tar_to_pinned_buffer

        if verbose:
            print("  Using GPU-native parquet decoder (unified tar read)")

        _t0 = time.perf_counter()
        pinned, entries = read_tar_to_pinned_buffer(scene_path)
        pin_np = pinned.numpy()

        entry_map: Dict[str, TarFileEntry] = {}
        for e in entries:
            bname = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
            entry_map[bname] = e

        def _open_buf(name: str) -> BytesIO:
            ent = entry_map[name]
            return BytesIO(bytes(pin_np[ent.offset : ent.offset + ent.size]))

        _tA = time.perf_counter()

        # Try the unified C++ scene loader (Phase 3: ego+cal+obs parsed, pools ready)
        try:
            from .gpu_parquet import _get_scene_loader_ext

            scene_ext = _get_scene_loader_ext()

            _tA1 = time.perf_counter()

            entry_names = [e.name for e in entries]
            entry_offsets = [e.offset for e in entries]
            entry_sizes = [e.size for e in entries]

            _tw = target_resolution[0] if target_resolution else -1
            _th = target_resolution[1] if target_resolution else -1

            gpu_tensors = scene_ext.load_scene_gpu(
                pinned,
                entry_names,
                entry_offsets,
                entry_sizes,
                device.index if device.index is not None else 0,
                _tw,
                _th,
            )
            _tB = time.perf_counter()

            # Pinned buffer is fully consumed — kick off prefetch for next scene
            if prefetch_next is not None:
                from .gpu_parquet import prefetch_tar

                prefetch_tar(str(prefetch_next))

            # Unpack ego (indices 20, 21)
            ego_track = EgoTrackData(
                timestamps=gpu_tensors[20],
                poses_tquat=gpu_tensors[21],
            )

            # Obstacles are now parsed in C++ — unpack pool-ready tensors
            obstacles = []
            _obs_timestamps_us = gpu_tensors[23]
            _obs_track_ps = gpu_tensors[24]
            _obs_track_ts = gpu_tensors[25]
            _obs_translations = gpu_tensors[26]
            _obs_quaternions = gpu_tensors[27]
            _obs_scales = gpu_tensors[28]
            _obs_colors = gpu_tensors[29]
            _tC = time.perf_counter()

            # Camera data from C++ (indices 32-38, overlapped with GPU pipeline)
            if len(gpu_tensors) > 32 and gpu_tensors[32].shape[0] > 0:
                n_cams = gpu_tensors[32].shape[0]
                cam_pp = gpu_tensors[32]  # [N, 2] float32 GPU
                cam_sz = gpu_tensors[33]  # [N, 2] float32 GPU
                cam_fw = gpu_tensors[34]  # [N, 6] float32 GPU
                cam_ld = gpu_tensors[35]  # [N, 2, 2] float32 GPU
                cam_s2r = gpu_tensors[36]  # [N, 4, 4] float32 GPU
                cam_mra = gpu_tensors[37]  # [N] float32 CPU
                cam_name_bytes = gpu_tensors[38]  # [L] uint8 CPU
                cam_names = cam_name_bytes.numpy().tobytes().decode().split("\x00")

                camera_name_to_id = {name: i for i, name in enumerate(cam_names)}
                sensor_to_rig = {name: cam_s2r[i] for i, name in enumerate(cam_names)}
                mra_np = cam_mra.numpy()
                camera_list = [
                    FThetaCamera(
                        principal_point=cam_pp[i],
                        image_size=cam_sz[i],
                        fw_poly=cam_fw[i],
                        max_ray_angle=float(mra_np[i]),
                        linear_distortion=cam_ld[i],
                        depth_max=200.0,
                    )
                    for i in range(n_cams)
                ]
            else:
                cams = _read_av2_calibration_from_fh(_open_buf("calibration_estimate.parquet"))
                camera_list, camera_name_to_id, sensor_to_rig = _cameras_to_ftheta(cams, device, target_resolution)
            _tD = time.perf_counter()

            _used_scene_loader = True
            # road_boundary_polyline is already in world frame — use gpu_tensors[0..4]
            # directly from C++ (no transform, no Python re-read needed).
            _tE = time.perf_counter()

            if verbose:
                n_obs_tracks = _obs_track_ps.shape[0]
                _cam_in_b = len(gpu_tensors) > 32 and gpu_tensors[32].shape[0] > 0
                _cam_tag = "cameras(in B)" if _cam_in_b else "cameras"
                print(
                    f"    [scene_loader] C++ pipeline: {(_tB - _tA1) * 1000:.2f}ms, "
                    f"obs({n_obs_tracks} tracks): {(_tC - _tB) * 1000:.2f}ms, "
                    f"{_cam_tag}: {(_tD - _tC) * 1000:.2f}ms, "
                    f"total: {(_tD - _tA) * 1000:.2f}ms"
                )

        except Exception as _sl_err:
            import traceback

            traceback.print_exc()
            if verbose:
                print(f"    [scene_loader] failed ({_sl_err}), falling back to legacy path")

        if not _used_scene_loader:
            # Legacy GPU path: separate Python orchestration
            from .gpu_parquet import (
                load_polylines_gpu_native,
                prepare_pipeline_plan,
                scan_polyline_metadata,
            )

            _tA1 = time.perf_counter()
            pq_metas = scan_polyline_metadata(pinned, entries)
            _tA2 = time.perf_counter()
            pipeline_plan = prepare_pipeline_plan(pq_metas[0])
            _tA3 = time.perf_counter()

            buf_ego = _open_buf("egomotion_estimate.parquet")
            buf_obs = _open_buf("object_fused.parquet")
            cal_fh = _open_buf("calibration_estimate.parquet")
            _tA4 = time.perf_counter()

            with ThreadPoolExecutor(max_workers=2) as tp:
                fut_ego = tp.submit(_read_av2_egomotion, buf_ego)
                fut_obs = tp.submit(_read_av2_obstacles, buf_obs)
                fut_cal = tp.submit(_read_av2_calibration_from_fh, cal_fh)

                polyline_data = load_polylines_gpu_native(
                    scene_path,
                    device,
                    preloaded=(pinned, entries),
                    pre_scanned=pq_metas,
                    pre_plan=pipeline_plan,
                )
                _tB = time.perf_counter()
                if verbose:
                    print(
                        f"    [B overhead] scan: {(_tA2 - _tA1) * 1000:.2f}ms, "
                        f"plan: {(_tA3 - _tA2) * 1000:.2f}ms, "
                        f"bufs: {(_tA4 - _tA3) * 1000:.2f}ms, "
                        f"gpu_native: {(_tB - _tA4) * 1000:.2f}ms, "
                        f"total_B_phase: {(_tB - _tA) * 1000:.2f}ms"
                    )

                ego_track = fut_ego.result()
                obstacles = fut_obs.result()
                _tC = time.perf_counter()

                cameras = fut_cal.result()
                _tD = time.perf_counter()

            _cam_pool = ThreadPoolExecutor(max_workers=1)
            fut_cam = _cam_pool.submit(_cameras_to_ftheta, cameras, device, target_resolution)
    else:
        loader = get_file_loader(scene_path)
        polyline_data = _load_polylines_pyarrow(loader)

        buf_ego = loader.open("egomotion_estimate.parquet")
        buf_obs = loader.open("object_fused.parquet")

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ego = pool.submit(_read_av2_egomotion, buf_ego)
            fut_obs = pool.submit(_read_av2_obstacles, buf_obs)
            ego_track = fut_ego.result()
            obstacles = fut_obs.result()

        cameras = _read_av2_calibration(loader)

    if not _used_scene_loader:
        lane_line_flat = polyline_data.get("dw_lane_line.parquet")
        road_boundary_flat = polyline_data.get("cf_road_boundary.parquet")
        crosswalk_flat = polyline_data.get("cf_crosswalks.parquet")
        static_obstacle_flat = polyline_data.get("cf_static_obstacle.parquet")

    # Road boundary (road_boundary_polyline) is already in world frame — no transform.

    # Clip ego track to the time range covered by scene elements.
    # C++ scene_loader path already clips ego in the pipeline.
    if _used_scene_loader:
        if verbose:
            n_ego = ego_track.timestamps.shape[0]
            if n_ego > 0:
                t0 = ego_track.timestamps[0].item()
                t1 = ego_track.timestamps[-1].item()
                print(f"  Clipped ego to scene range: {(t1 - t0) / 1e6:.1f}s ({n_ego} ego poses)")
    else:
        gpu_ts = [
            f.timestamps_us
            for f in [lane_line_flat, road_boundary_flat, crosswalk_flat, static_obstacle_flat]
            if f is not None and f.timestamps_us.is_cuda
        ]
        cpu_ts = [
            f.timestamps_us
            for f in [lane_line_flat, road_boundary_flat, crosswalk_flat, static_obstacle_flat]
            if f is not None and not f.timestamps_us.is_cuda
        ]
        for obs in obstacles:
            (gpu_ts if obs.timestamps.is_cuda else cpu_ts).append(obs.timestamps)

        _all_mins = []
        _all_maxs = []
        if gpu_ts:
            gpu_cat = torch.cat(gpu_ts)
            _all_mins.append(gpu_cat.min())
            _all_maxs.append(gpu_cat.max())
        for t in cpu_ts:
            _all_mins.append(t.min())
            _all_maxs.append(t.max())

        if _all_mins:
            scene_start = min(v.item() for v in _all_mins)
            scene_end = max(v.item() for v in _all_maxs)

            ego_ts = ego_track.timestamps.to(device)
            ego_poses = ego_track.poses_tquat.to(device)
            mask = (ego_ts >= scene_start) & (ego_ts <= scene_end)
            if mask.any():
                ego_track = EgoTrackData(
                    timestamps=ego_ts[mask],
                    poses_tquat=ego_poses[mask],
                )
                if verbose:
                    print(
                        f"  Clipped ego to scene range: {(scene_end - scene_start) / 1e6:.1f}s "
                        f"({len(ego_track.timestamps)} ego poses)"
                    )
            else:
                ego_track = EgoTrackData(timestamps=ego_ts, poses_tquat=ego_poses)

    # Build polyline pools
    polyline_pools = []
    if _used_scene_loader:
        # Road boundary: struct (road_boundary_polyline) is already in world frame
        verts = gpu_tensors[0]
        if verts.shape[0] > 0:
            polyline_pools.append(
                TimestampedPolylinePool(
                    timestamps_us=gpu_tensors[3],
                    timestamped_varrays_prefix_sum=gpu_tensors[4],
                    varrays_prefix_sum=gpu_tensors[2][1:],
                    vertices=verts,
                    prim_type_id=PRIM_ROAD_BOUNDARY,
                )
            )
        # Lane lines and static obstacles: use C++ output
        for idx, prim_id in [
            (1, PRIM_LANE_LINE),
            (3, PRIM_STATIC_OBSTACLE),
        ]:
            verts = gpu_tensors[idx * 5]
            if verts.shape[0] > 0:
                polyline_pools.append(
                    TimestampedPolylinePool(
                        timestamps_us=gpu_tensors[idx * 5 + 3],
                        timestamped_varrays_prefix_sum=gpu_tensors[idx * 5 + 4],
                        varrays_prefix_sum=gpu_tensors[idx * 5 + 2][1:],
                        vertices=verts,
                        prim_type_id=prim_id,
                    )
                )
    else:
        _sorted = _use_gpu
        pool = _flat_polylines_to_pool(road_boundary_flat, PRIM_ROAD_BOUNDARY, device, pre_sorted=_sorted)
        if pool:
            polyline_pools.append(pool)
        pool = _flat_polylines_to_pool(lane_line_flat, PRIM_LANE_LINE, device, pre_sorted=_sorted)
        if pool:
            polyline_pools.append(pool)
        pool = _flat_polylines_to_pool(static_obstacle_flat, PRIM_STATIC_OBSTACLE, device, pre_sorted=_sorted)
        if pool:
            polyline_pools.append(pool)

    # Ego trajectory (always built in Python — trivial)
    if include_ego_trajectory:
        pool = _ego_trajectory_to_pool(ego_track, device)
        if pool:
            polyline_pools.append(pool)

    if _use_gpu and not _used_scene_loader:
        torch.cuda.synchronize(device)
    if _use_gpu:
        _tF = time.perf_counter()

    # Build polygon pools
    polygon_pools = []
    if _used_scene_loader:
        xw_verts = gpu_tensors[2 * 5]
        if xw_verts.shape[0] > 0:
            xw_varrays_ps = gpu_tensors[39] if len(gpu_tensors) > 39 and gpu_tensors[39].numel() > 0 else None
            if xw_varrays_ps is None:
                xw_roff = gpu_tensors[2 * 5 + 2]
                xw_varrays_ps = torch.cumsum(xw_roff[1:] - xw_roff[:-1], dim=0)
            polygon_pools.append(
                TimestampedPolygonPool(
                    timestamps_us=gpu_tensors[2 * 5 + 3],
                    timestamped_varrays_prefix_sum=gpu_tensors[2 * 5 + 4],
                    varrays_prefix_sum=xw_varrays_ps,
                    triangle_prefix_sum=gpu_tensors[30],
                    vertices=xw_verts,
                    triangles=gpu_tensors[31],
                    prim_type_id=PRIM_CROSSWALK,
                )
            )
    else:
        pool = _flat_polygons_to_pool(crosswalk_flat, PRIM_CROSSWALK, device)
        if pool:
            polygon_pools.append(pool)

    if _use_gpu and not _used_scene_loader:
        torch.cuda.synchronize(device)
    if _use_gpu:
        _tG = time.perf_counter()

    # Cube pools
    cube_pools = []

    if include_ego_obstacle:
        ego_cube_pool = _ego_obstacle_to_pool(ego_track, device)
        if ego_cube_pool:
            cube_pools.append(ego_cube_pool)

    if _used_scene_loader and _obs_track_ps.shape[0] > 0:
        cube_pools.append(
            CubePool(
                timestamps_us=_obs_timestamps_us.to(device),
                cube_ts_prefix_sum=_obs_track_ps.to(device),
                track_timestamps_us=_obs_track_ts.to(device),
                translations=_obs_translations.to(device),
                quaternions=_obs_quaternions.to(device),
                scales=_obs_scales.to(device),
                colors=_obs_colors.to(device),
                prim_type_id=PRIM_OBSTACLE,
                render_flags=CUBE_FLAG_WIREFRAME,
            )
        )
    elif not _used_scene_loader:
        obstacle_pool = _obstacles_to_pool(obstacles, device)
        if obstacle_pool:
            cube_pools.append(obstacle_pool)

    if _use_gpu and not _used_scene_loader:
        torch.cuda.synchronize(device)
    if _use_gpu:
        _tH = time.perf_counter()

    # Camera conversion: scene_loader path already has cameras;
    # legacy GPU path collects from background thread;
    # PyArrow path runs reference implementation.
    if _use_gpu and not _used_scene_loader:
        camera_list, camera_name_to_id, sensor_to_rig = fut_cam.result()
        _cam_pool.shutdown(wait=False)
    elif not _use_gpu:
        camera_list, camera_name_to_id, sensor_to_rig = _cameras_to_ftheta_ref(cameras, device, target_resolution)

    # Ensure ego track is on device (already moved during ego clipping for GPU path)
    if ego_track.timestamps.device != device:
        ego_track = EgoTrackData(
            timestamps=ego_track.timestamps.to(device),
            poses_tquat=ego_track.poses_tquat.to(device),
        )

    # Build scene
    timestamped_scene = TimestampedScene(
        polyline_pools=polyline_pools,
        polygon_pools=polygon_pools,
        cube_pools=cube_pools if cube_pools else None,
    )

    if _use_gpu:
        if not _used_scene_loader:
            torch.cuda.synchronize(device)
        _tI = time.perf_counter()

    if verbose:
        if _use_gpu:
            print(f"  Phase breakdown (ms):")
            print(f"    A  tar read + pinned:    {(_tA - _t0) * 1000:6.2f}")
            if _used_scene_loader:
                print(f"    A1 entry map + args:     {(_tA1 - _tA) * 1000:6.2f}")
            print(
                f"    B  C++ pipeline+sync:    {(_tB - _tA1) * 1000:6.2f}"
                if _used_scene_loader
                else f"    B  polyline GPU decode:  {(_tB - _tA) * 1000:6.2f}"
            )
            print(f"    C  ego+obs unpack:       {(_tC - _tB) * 1000:6.2f}")
            print(f"    D  cal+cameras unpack:   {(_tD - _tC) * 1000:6.2f}")
            print(f"    E  road boundary xform:  {(_tE - _tD) * 1000:6.2f}" + (" (in B)" if _used_scene_loader else ""))
            print(f"    F  polyline pools (x3):  {(_tF - _tE) * 1000:6.2f}")
            print(f"    G  polygon pool:         {(_tG - _tF) * 1000:6.2f}")
            print(f"    H  obstacle pool:        {(_tH - _tG) * 1000:6.2f}")
            print(f"    I  cameras+ego+finalize: {(_tI - _tH) * 1000:6.2f}")
            print(f"    TOTAL:                   {(_tI - _t0) * 1000:6.2f}")
        print(
            f"  Loaded: {len(polyline_pools)} polyline pools, {len(polygon_pools)} polygon pools, {len(cube_pools)} cube pools"
        )
        print(f"  Cameras: {list(camera_name_to_id.keys())}")
        print(f"  Ego timestamps: {len(ego_track.timestamps)} frames")

    return ClipgtGpuScene(
        timestamped_scene=timestamped_scene,
        cameras=camera_list,
        camera_name_to_id=camera_name_to_id,
        sensor_to_rig=sensor_to_rig,
        ego_track=ego_track,
        device=device,
    )


def _parse_rig_to_cameras(rig_data: dict) -> List[CameraData]:
    """Build CameraData list from parsed rig JSON dict."""
    cameras = []
    for sensor in rig_data["sensors"]:
        name = sensor["name"]
        if not name.startswith("camera:"):
            continue
        if sensor.get("properties") is None:
            continue

        props = sensor["properties"]

        poly_key = "polynomial" if "polynomial" in props else "bw-poly"
        if poly_key not in props:
            continue
        poly_coeffs = [float(x) for x in props[poly_key].split()]
        if len(poly_coeffs) < 6:
            poly_coeffs.extend([0.0] * (6 - len(poly_coeffs)))

        poly_type = props.get("polynomial-type", "")
        if poly_type == "angle-to-pixeldistance":
            is_bw_poly = False
        elif poly_type == "pixeldistance-to-angle":
            is_bw_poly = True
        elif poly_key == "bw-poly":
            is_bw_poly = True
        else:
            is_bw_poly = len(poly_coeffs) > 1 and abs(poly_coeffs[1]) < 1.0

        linear_c = float(props.get("linear-c", 1.0))
        linear_d = float(props.get("linear-d", 0.0))
        linear_e = float(props.get("linear-e", 0.0))
        linear_cde = np.array([linear_c, linear_d, linear_e], dtype=np.float32)

        rpy = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]
        translation = sensor["nominalSensor2Rig_FLU"]["t"]
        rotation = R.from_euler("xyz", np.radians(rpy)).as_matrix()

        if "correction_sensor_R_FLU" in sensor:
            corr_rpy = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]
            corr_rotation = R.from_euler("xyz", np.radians(corr_rpy)).as_matrix()
            rotation = rotation @ corr_rotation

        sensor_to_rig = torch.eye(4, dtype=torch.float32)
        sensor_to_rig[:3, :3] = torch.from_numpy(rotation.astype(np.float32))
        sensor_to_rig[:3, 3] = torch.tensor(translation, dtype=torch.float32)

        cameras.append(
            CameraData(
                name=name,
                cx=float(props["cx"]),
                cy=float(props["cy"]),
                width=int(props["width"]),
                height=int(props["height"]),
                poly=np.array(poly_coeffs, dtype=np.float32),
                is_bw_poly=is_bw_poly,
                sensor_to_rig=sensor_to_rig,
                linear_cde=linear_cde,
            )
        )
    return cameras


def _read_av2_calibration_from_fh(fh) -> List[CameraData]:
    """Read calibration from a file handle (BytesIO or similar)."""
    table = pq.read_table(fh, columns=["calibration_estimate"])
    cal_struct = table.column("calibration_estimate")[0].as_py()
    rig_data = json.loads(str(cal_struct["rig_json"]))["rig"]
    return _parse_rig_to_cameras(rig_data)


def _read_av2_calibration(loader) -> List[CameraData]:
    """Read calibration from AV2 format via file loader."""
    fh = loader.open("calibration_estimate.parquet")
    return _read_av2_calibration_from_fh(fh)
