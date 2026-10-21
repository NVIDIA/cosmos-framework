# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np

from cosmos_auto.wm.graphics_utils import LineSegment2D, render_geometries
from cosmos_auto.wm.pcd_utils import filter_by_height_relative_to_ego, interpolate_polyline_to_points
from cosmos_auto.wm.wds_utils import get_sample

# Load lane line configuration
LANELINE_CONFIG = json.load(
    open(Path(__file__).parent / "color_scheme" / "config_color_geometry_laneline.json", encoding="utf-8")
)

# For backward compatibility - extract color mapping from config
LANELINE_TYPE_TO_RGB = {k: v["color"] for k, v in LANELINE_CONFIG.items()}


def create_long_dashed_segments(line_segments, segment_interval=0.05):
    """
    Create long dashed line pattern following US highway standards:
    10 feet dash, 30 feet gap (1:3 ratio, 25% visible).

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dashing
        segment_interval: float, distance between adjacent segments in meters

    Returns:
        dashed_segments: np.ndarray, shape [M, 2, 3], dashed line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # US highway standard: 10ft dash, 30ft gap
    dash_length_meters = 10 * 0.3048  # 10 feet = ~3.05 meters
    gap_length_meters = 30 * 0.3048  # 30 feet = ~9.14 meters

    # Convert to segment counts
    dash_segments_count = max(1, int(dash_length_meters / segment_interval))
    gap_segments_count = max(1, int(gap_length_meters / segment_interval))

    if len(line_segments) <= dash_segments_count:
        return line_segments  # Line too short for proper dashing

    dash_segments = []
    total_segments = len(line_segments)

    i = 0
    while i < total_segments:
        # Add dash segments
        end_dash = min(i + dash_segments_count, total_segments)
        dash_segments.extend(line_segments[i:end_dash])

        # Skip gap
        i = end_dash + gap_segments_count

    return np.array(dash_segments) if dash_segments else line_segments[:1]


def create_short_dashed_segments(line_segments, segment_interval=0.05):
    """
    Create short dashed line pattern following US highway standards:
    3 feet dash, 9 feet gap (1:3 ratio, 25% visible).

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dashing
        segment_interval: float, distance between adjacent segments in meters

    Returns:
        dashed_segments: np.ndarray, shape [M, 2, 3], dashed line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # US highway standard: 3ft dash, 9ft gap
    dash_length_meters = 3 * 0.3048  # 3 feet = ~0.91 meters
    gap_length_meters = 9 * 0.3048  # 9 feet = ~2.74 meters

    # Convert to segment counts
    dash_segments_count = max(1, int(dash_length_meters / segment_interval))
    gap_segments_count = max(1, int(gap_length_meters / segment_interval))

    if len(line_segments) <= dash_segments_count:
        return line_segments  # Line too short for proper dashing

    dash_segments = []
    total_segments = len(line_segments)

    i = 0
    while i < total_segments:
        # Add dash segments
        end_dash = min(i + dash_segments_count, total_segments)
        dash_segments.extend(line_segments[i:end_dash])

        # Skip gap
        i = end_dash + gap_segments_count

    return np.array(dash_segments) if dash_segments else line_segments[:1]


def create_dot_dashed_segments(line_segments, segment_interval=0.05):
    """
    Create dot-dashed pattern following US highway standards:
    3 feet dash period, 9 feet gap, with dots within dash period.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dot-dashing
        segment_interval: float, distance between adjacent segments in meters

    Returns:
        dot_dashed_segments: np.ndarray, shape [M, 2, 3], dot-dashed line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # US highway standard: 3ft dash period, 9ft gap
    dash_length_meters = 3 * 0.3048  # 3 feet = ~0.91 meters
    gap_length_meters = 9 * 0.3048  # 9 feet = ~2.74 meters

    # Convert to segment counts
    dash_segments_count = max(1, int(dash_length_meters / segment_interval))
    gap_segments_count = max(1, int(gap_length_meters / segment_interval))

    if len(line_segments) <= dash_segments_count:
        return line_segments[::3]  # Very sparse for short lines

    dot_segments = []
    total_segments = len(line_segments)

    i = 0
    while i < total_segments:
        # Within the dash period, create dots with 1:2 spacing (every 3rd segment)
        dash_end = min(i + dash_segments_count, total_segments)
        for j in range(i, dash_end, 3):  # Every 3rd segment within dash period
            if j < total_segments:
                dot_segments.append(line_segments[j])

        # Skip the gap period
        i = dash_end + gap_segments_count

    return np.array(dot_segments) if dot_segments else line_segments[:1]


def create_dotted_segments_1_9_ratio(line_segments):
    """
    Create dotted line with 1:9 dot to gap ratio (10% visible).

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dotting

    Returns:
        dotted_segments: np.ndarray, shape [M, 2, 3], dotted line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # Simple 1:9 ratio - take every 10th segment
    return line_segments[::10]


def offset_line_segments(line_segments, offset_distance=0.1):
    """
    Create left and right offset versions of line segments for dual-line patterns.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to offset
        offset_distance: float, distance to offset (in meters)

    Returns:
        tuple: (left_segments, right_segments), both np.ndarray shape [N, 2, 3]
    """
    if len(line_segments) == 0:
        return line_segments, line_segments

    left_segments = []
    right_segments = []

    for segment in line_segments:
        p1, p2 = segment[0], segment[1]

        # Calculate direction vector
        direction = p2 - p1
        if np.linalg.norm(direction) == 0:
            left_segments.append(segment)
            right_segments.append(segment)
            continue

        direction = direction / np.linalg.norm(direction)

        # Calculate perpendicular vector (left is 90° counter-clockwise)
        perpendicular = np.array([-direction[1], direction[0], 0])

        # Create offset segments
        left_offset = perpendicular * offset_distance
        right_offset = -perpendicular * offset_distance

        left_segment = np.array([p1 + left_offset, p2 + left_offset])
        right_segment = np.array([p1 + right_offset, p2 + right_offset])

        left_segments.append(left_segment)
        right_segments.append(right_segment)

    return np.array(left_segments), np.array(right_segments)


def apply_laneline_pattern(line_segments, specs):
    """
    Apply the specified pattern to line segments.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply pattern to
        specs: dict, specifications from configuration containing pattern type

    Returns:
        list: list of np.ndarray segments arrays for rendering
    """
    pattern = specs.get("pattern", "solid")

    if pattern == "solid":
        return [line_segments]

    elif pattern == "long_dashed":
        dashed_segments = create_long_dashed_segments(line_segments)
        return [dashed_segments]

    elif pattern == "short_dashed":
        dashed_segments = create_short_dashed_segments(line_segments)
        return [dashed_segments]

    elif pattern == "dot_dashed":
        dot_dashed_segments = create_dot_dashed_segments(line_segments)
        return [dot_dashed_segments]

    elif pattern == "dotted_1_9":
        dotted_segments = create_dotted_segments_1_9_ratio(line_segments)
        return [dotted_segments]

    elif pattern == "dual":
        dual_pattern = specs.get("dual_pattern")
        if not dual_pattern:
            return [line_segments]

        left_segments, right_segments = offset_line_segments(line_segments)
        left_pattern, right_pattern = dual_pattern

        result = []

        # Apply left pattern using recursive call
        left_specs = {"pattern": left_pattern}
        left_results = apply_laneline_pattern(left_segments, left_specs)
        result.extend(left_results)

        # Apply right pattern using recursive call
        right_specs = {"pattern": right_pattern}
        right_results = apply_laneline_pattern(right_segments, right_specs)
        result.extend(right_results)

        return result

    else:
        return [line_segments]  # Fallback


def create_laneline_type_geometry_projection_gl(
    laneline_wds_file,
    camera_poses,
    camera_model,
    depth_max,
):
    """
    Create lane line projections with different geometry patterns and colors.

    Args:
        laneline_wds_file: str, path to the webdataset file containing laneline data
        camera_poses: np.ndarray, shape (T, 4, 4), dtype=np.float32, camera poses of T frames
        camera_model: CameraModel, ftheta or pinhole camera model
        depth_max: float, maximum depth value for rendering

    Returns:
        laneline_type_projection: np.ndarray, shape (T, H, W, 3), dtype=np.uint8,
            rendered lane line projections across T frames
    """
    processed_lanelines = prepare_laneline_type_geometry_data(laneline_wds_file)
    if len(processed_lanelines) == 0:
        return np.zeros((len(camera_poses), camera_model.height, camera_model.width, 3), dtype=np.uint8)

    # Render for each camera pose
    laneline_type_projection = []
    for camera_pose in camera_poses:
        laneline_geometry_object_list = create_laneline_geometry_objects_from_data(
            processed_lanelines,
            camera_pose,
            camera_model,
        )

        laneline_type_projection_one_frame = render_geometries(
            laneline_geometry_object_list,
            camera_model.height,
            camera_model.width,
            depth_max,
            depth_gradient=True,
        )
        laneline_type_projection.append(laneline_type_projection_one_frame)

    laneline_type_projection = np.stack(laneline_type_projection, axis=0).astype(np.uint8)

    return laneline_type_projection


def prepare_laneline_type_geometry_data(laneline_wds_file):
    """
    Preprocess laneline data into segments, colors, and widths for later per-frame projection.

    Args:
        laneline_wds_file: str, path to the webdataset file containing laneline data

    Returns:
        processed_lanelines: list[dict], each with keys 'pattern_segments_list', 'rgb_float', 'line_width'
    """
    laneline_sample = get_sample(laneline_wds_file)
    laneline_data_list = laneline_sample["lanelines.json"]["labels"]
    if len(laneline_data_list) == 1 and laneline_data_list[0].get("labelFamily") == "EMPTY":
        return []

    base_width = 12
    processed_lanelines = []

    for laneline_data in laneline_data_list:
        attribute_list = laneline_data["labelData"]["shape3d"]["attributes"]
        color = None
        style = None

        for attribute in attribute_list:
            if attribute["name"] == "colors":
                color = attribute["enumsList"]["enumsList"][0]
            if attribute["name"] == "styles":
                style = attribute["enumsList"]["enumsList"][0]

        type_name = f"{color} {style}"
        specs = LANELINE_CONFIG.get(type_name, LANELINE_CONFIG.get("OTHER"))
        rgb_float = np.array(specs["color"]) / 255.0

        # Get polyline and subdivide
        polyline = laneline_data["labelData"]["shape3d"]["polyline3d"]["vertices"]
        polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.05)
        line_segments = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1)  # [N, 2, 3]

        pattern_segments_list = apply_laneline_pattern(line_segments, specs)

        processed_lanelines.append(
            {
                "pattern_segments_list": pattern_segments_list,
                "rgb_float": rgb_float,
                "line_width": base_width * specs["thickness_multiplier"],
            }
        )

    return processed_lanelines


def build_worldspace_laneline_geometry(
    processed_lanelines,
    camera_model,
    camera_pose_init=None,
    below_level_filter=False,
):
    """
    Build WorldSpaceLines objects for all lanelines, grouped by line width.

    Geometry is uploaded to the GPU once and projected per-frame by the
    vertex shader, eliminating per-frame CPU projection entirely.

    Args:
        processed_lanelines: list[dict], from prepare_laneline_type_geometry_data.
        camera_model: CameraModel (FThetaCamera or PinholeCamera).
        camera_pose_init: np.ndarray (4,4) or None, for one-time height filtering.

    Returns:
        list[WorldSpaceLines]: one object per unique line_width.
    """
    from cosmos_auto.wm.graphics_utils import WorldSpaceLines

    if not processed_lanelines:
        return []

    width_to_segments = {}
    width_to_colors = {}

    for laneline_info in processed_lanelines:
        rgb = laneline_info["rgb_float"]
        width = laneline_info["line_width"]

        if not below_level_filter and camera_pose_init is not None:
            combined = np.concatenate(laneline_info["pattern_segments_list"], axis=0)
            if filter_by_height_relative_to_ego(
                combined.reshape(-1, 3), camera_model, camera_pose_init, camera_pose_init
            ):
                continue

        for segments in laneline_info["pattern_segments_list"]:
            if len(segments) == 0:
                continue
            if width not in width_to_segments:
                width_to_segments[width] = []
                width_to_colors[width] = []
            width_to_segments[width].append(segments)
            n_verts = len(segments) * 2
            width_to_colors[width].append(np.tile(rgb, (n_verts, 1)))

    result = []
    for width, seg_list in width_to_segments.items():
        all_segs = np.concatenate(seg_list, axis=0)
        all_cols = np.concatenate(width_to_colors[width], axis=0)
        result.append(
            WorldSpaceLines(
                segments=all_segs,
                colors=all_cols,
                line_width=width,
                camera_model=camera_model,
                below_level_filter=below_level_filter,
            )
        )
    return result


def create_laneline_geometry_objects_from_data(
    processed_lanelines,
    camera_pose,
    camera_model,
    camera_pose_init=None,
):
    """
    Build LineSegment2D geometry objects for a single frame from preprocessed laneline data.

    Args:
        processed_lanelines: list[dict], each with keys 'pattern_segments_list', 'rgb_float', 'line_width'
        camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, ftheta or pinhole camera model
        camera_pose_init: np.ndarray, shape (4, 4), dtype=np.float32, camera pose at the start of the clip

    Returns:
        geometry_objects: list[LineSegment2D], geometry objects for the current frame
    """
    geometry_objects = []
    for laneline_info in processed_lanelines:
        # Combine all segments into a single polyline
        combined_segments = np.concatenate(laneline_info["pattern_segments_list"], axis=0)
        # Reshape to [N, 3]
        combined_segments = combined_segments.reshape(-1, 3)
        if camera_pose_init is not None and filter_by_height_relative_to_ego(
            combined_segments, camera_model, camera_pose, camera_pose_init
        ):
            continue

        for segments in laneline_info["pattern_segments_list"]:
            if len(segments) == 0:
                continue
            # Project to image space and filter out the line segments that are out of the image
            xy_and_depth = camera_model.get_xy_and_depth(segments.reshape(-1, 3), camera_pose).reshape(-1, 2, 3)
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]
            if len(valid_xy_and_depth) == 0:
                continue
            geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth, base_color=laneline_info["rgb_float"], line_width=laneline_info["line_width"]
                )
            )

    return geometry_objects
