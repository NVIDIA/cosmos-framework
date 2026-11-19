# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from cosmos_auto.wm.graphics_utils import WorldSpaceLines
from cosmos_auto.wm.pcd_utils import filter_by_height_relative_to_ego
from cosmos_auto.wm.wds_utils import get_sample

_COLOR_CONFIG_PATH = Path(__file__).parent / "color_scheme" / "config_color_road_boundary.json"
_SAMPLE_INTERVAL_M = 0.2
_SOLID_LINE_WIDTH_PX = 12.0


def route_road_boundary_layer(minimap_types, use_road_boundary_colors):
    """Route road boundaries to either subtype colors or the legacy minimap path."""

    minimap_types = list(minimap_types)
    render_colored = bool(use_road_boundary_colors and "road_boundaries" in minimap_types)
    if render_colored:
        minimap_types = [minimap_type for minimap_type in minimap_types if minimap_type != "road_boundaries"]
    return minimap_types, render_colored


@lru_cache(maxsize=1)
def _load_color_config():
    with open(_COLOR_CONFIG_PATH, encoding="utf-8") as config_file:
        config = json.load(config_file)
    groups_by_subtype = {subtype: group for group, settings in config.items() for subtype in settings["subtypes"]}
    return config, groups_by_subtype


def _shape_attributes(label):
    attributes = label.get("labelData", {}).get("shape3d", {}).get("attributes", [])
    return attributes if isinstance(attributes, list) else []


def _extract_boundary_subtype(label):
    for attribute in _shape_attributes(label):
        if isinstance(attribute, dict) and attribute.get("name") == "boundary_type":
            value = attribute.get("text")
            if isinstance(value, str):
                return value.strip().lower() or None
    return None


def _clean_polyline(vertices):
    if isinstance(vertices, (list, tuple)) and vertices and isinstance(vertices[0], dict):
        try:
            vertices = [[point["x"], point["y"], point["z"]] for point in vertices]
        except (KeyError, TypeError):
            return None
    polyline = np.asarray(vertices, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
        return None
    if not np.isfinite(polyline).all():
        return None
    keep = np.concatenate(([True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-8))
    polyline = polyline[keep]
    return polyline if len(polyline) >= 2 else None


def _points_at_distances(polyline, distances):
    vectors = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.clip(np.asarray(distances, dtype=np.float64), 0.0, cumulative[-1])
    indices = np.searchsorted(cumulative[1:], distances, side="right")
    indices = np.clip(indices, 0, len(vectors) - 1)
    fractions = (distances - cumulative[indices]) / lengths[indices]
    return polyline[indices] + fractions[:, None] * vectors[indices]


def _sampled_stroke(polyline):
    total_length = np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum()
    distances = np.arange(0.0, total_length, _SAMPLE_INTERVAL_M)
    if len(distances) == 0 or distances[-1] < total_length:
        distances = np.append(distances, total_length)
    points = _points_at_distances(polyline, distances)
    return np.stack([points[:-1], points[1:]], axis=1)


def _iter_clean_boundaries(road_boundary_wds_file):
    labels = get_sample(road_boundary_wds_file)["road_boundaries.json"].get("labels", [])
    for label in labels:
        if label.get("labelFamily") == "EMPTY":
            continue
        shape3d = label.get("labelData", {}).get("shape3d", {})
        polyline = _clean_polyline(shape3d.get("polyline3d", {}).get("vertices", []))
        if polyline is not None:
            yield label, polyline


def load_colored_road_boundaries(road_boundary_wds_file):
    """Load road boundaries as uniform solid lines colored by subtype group."""

    config, groups_by_subtype = _load_color_config()
    boundaries = []
    for label, polyline in _iter_clean_boundaries(road_boundary_wds_file):
        group = groups_by_subtype.get(_extract_boundary_subtype(label), "other")
        stroke = _sampled_stroke(polyline)
        boundaries.append(
            {
                "polyline": polyline,
                "segments": stroke,
                "color": np.asarray(config[group]["rgb"], dtype=np.float64) / 255.0,
            }
        )
    return boundaries


def build_worldspace_road_boundaries(
    prepared_boundaries,
    camera_model,
    camera_pose_init=None,
    below_level_filter=False,
):
    """Batch subtype-colored road boundaries into one fixed-width draw call."""

    visible_boundaries = []
    for boundary in prepared_boundaries:
        if (
            not below_level_filter
            and camera_pose_init is not None
            and filter_by_height_relative_to_ego(boundary["polyline"], camera_model, camera_pose_init, camera_pose_init)
        ):
            continue
        if len(boundary["segments"]):
            visible_boundaries.append(boundary)

    if not visible_boundaries:
        return []

    return [
        WorldSpaceLines(
            segments=np.concatenate([boundary["segments"] for boundary in visible_boundaries], axis=0),
            colors=np.concatenate(
                [
                    np.tile(
                        boundary["color"],
                        (len(boundary["segments"]) * 2, 1),
                    )
                    for boundary in visible_boundaries
                ],
                axis=0,
            ),
            line_width=_SOLID_LINE_WIDTH_PX,
            camera_model=camera_model,
            below_level_filter=below_level_filter,
        )
    ]
