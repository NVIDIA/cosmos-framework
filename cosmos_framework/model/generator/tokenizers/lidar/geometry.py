# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Calibrated geometry helpers for Pandar128 LiDAR range maps."""

from __future__ import annotations

import numpy as np

from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig

# Native beam order used by the MADS / LiDARGen 128x3600 range-map format.
PANDAR128_ELEVATIONS_DEG = (
    14.436,
    13.535,
    13.082,
    12.624,
    12.165,
    11.702,
    11.239,
    10.771,
    10.305,
    9.83,
    9.356,
    8.88,
    8.401,
    7.921,
    7.438,
    6.953,
    6.467,
    5.978,
    5.487,
    4.996,
    4.501,
    4.007,
    3.509,
    3.013,
    2.512,
    2.013,
    1.885,
    1.761,
    1.637,
    1.511,
    1.386,
    1.258,
    1.13,
    1.008,
    0.88,
    0.756,
    0.63,
    0.505,
    0.379,
    0.251,
    0.124,
    0.0,
    -0.129,
    -0.254,
    -0.38,
    -0.506,
    -0.632,
    -0.76,
    -0.887,
    -1.012,
    -1.141,
    -1.266,
    -1.393,
    -1.519,
    -1.646,
    -1.773,
    -1.901,
    -2.027,
    -2.155,
    -2.282,
    -2.409,
    -2.535,
    -2.663,
    -2.789,
    -2.916,
    -3.044,
    -3.172,
    -3.299,
    -3.425,
    -3.552,
    -3.68,
    -3.806,
    -3.933,
    -4.062,
    -4.19,
    -4.318,
    -4.444,
    -4.571,
    -4.699,
    -4.824,
    -4.951,
    -5.081,
    -5.209,
    -5.336,
    -5.463,
    -5.589,
    -5.718,
    -5.843,
    -5.968,
    -6.1,
    -6.607,
    -7.117,
    -7.624,
    -8.134,
    -8.64,
    -9.149,
    -9.652,
    -10.16,
    -10.665,
    -11.17,
    -11.672,
    -12.174,
    -12.673,
    -13.173,
    -13.67,
    -14.166,
    -14.66,
    -15.154,
    -15.645,
    -16.135,
    -16.622,
    -17.106,
    -17.592,
    -18.072,
    -18.548,
    -19.03,
    -19.501,
    -19.978,
    -20.445,
    -20.918,
    -21.379,
    -21.848,
    -22.304,
    -22.768,
    -23.219,
    -23.678,
    -24.123,
    -25.016,
)


def _elevation_radians() -> np.ndarray:  # [H]
    """Beam elevations of the calibrated Pandar128 rows, in radians."""
    return np.deg2rad(np.asarray(PANDAR128_ELEVATIONS_DEG, dtype=np.float64))


def _azimuth_radians(projection: LidarRangeProjectionConfig) -> np.ndarray:  # [W]
    """Column azimuths of a semantic-width range map, in radians."""
    return np.deg2rad(
        np.linspace(
            projection.azimuth_start_degrees,
            projection.azimuth_end_degrees,
            projection.semantic_width,
            endpoint=projection.azimuth_endpoint,
            dtype=np.float64,
        )
    )


def pandar128_ray_directions(
    *,
    range_projection: LidarRangeProjectionConfig,
) -> np.ndarray:  # [H,W,3]
    """Unit ray direction for every cell of a semantic-width range map.

    Shares the elevation calibration and ``π → -π`` azimuth convention with
    :func:`range_map_to_xyz`, so scaling these directions by a metric range
    reproduces the same sensor-frame XYZ that unprojection produces -- but
    densely, and as a plain multiplication a loss can backpropagate through.
    """
    if range_projection.semantic_height != len(PANDAR128_ELEVATIONS_DEG):
        raise ValueError(
            "Range-map height must match the calibrated Pandar128 elevations, got "
            f"{range_projection.semantic_height} != {len(PANDAR128_ELEVATIONS_DEG)}"
        )
    elevation = _elevation_radians()[:, None]  # [H,1]
    azimuth = _azimuth_radians(range_projection)[None, :]  # [1,W]
    cos_elevation = np.cos(elevation)  # [H,1]
    x = cos_elevation * np.cos(azimuth)  # [H,W]
    y = cos_elevation * np.sin(azimuth)  # [H,W]
    z = np.broadcast_to(np.sin(elevation), x.shape)  # [H,W]
    return np.stack((x, y, z), axis=-1).astype(np.float32)  # [H,W,3]


def range_map_to_xyz(
    metric_range: np.ndarray,
    valid: np.ndarray,
    *,
    range_projection: LidarRangeProjectionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a Pandar128 range map to sensor-frame XYZ.

    Returns ``xyz`` with X forward, Y left, Z up and the flattened indices of
    valid rays. Azimuth follows the tested LiDARGen convention, ``π → -π``.
    When ``range_projection`` is omitted the map width is treated as the
    azimuth canvas, so BEV helpers can unproject already-resized clips.
    """
    if metric_range.ndim != 2 or valid.shape != metric_range.shape:
        raise ValueError(
            f"Expected metric/valid [H,W] with matching shapes, got {metric_range.shape} and {valid.shape}"
        )
    # A map that already sits at display width is its own azimuth canvas.
    # Matching native_width to W lets BEV helpers unproject toy/downsampled
    # clips (16, 64, ...) that are not divisors of the 3600-column native grid.
    width = int(metric_range.shape[1])
    projection = range_projection or LidarRangeProjectionConfig(
        native_width=width,
        semantic_width=width,
        model_width=width,
    )
    if metric_range.shape != (projection.semantic_height, projection.semantic_width):
        raise ValueError(
            "Range-map shape must match the configured Pandar128 elevations and semantic projection, "
            f"got {metric_range.shape} != {(projection.semantic_height, projection.semantic_width)}"
        )

    mask = valid.astype(bool) & np.isfinite(metric_range) & (metric_range > 0)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    elevation = _elevation_radians()[:, None]
    azimuth = _azimuth_radians(projection)[None, :]
    phi = np.broadcast_to(elevation, metric_range.shape)[mask]
    theta = np.broadcast_to(azimuth, metric_range.shape)[mask]
    distance = metric_range[mask].astype(np.float64)
    cos_phi = np.cos(phi)
    xyz = np.stack(
        (
            distance * cos_phi * np.cos(theta),
            distance * cos_phi * np.sin(theta),
            distance * np.sin(phi),
        ),
        axis=1,
    ).astype(np.float32)
    return xyz, np.flatnonzero(mask)
