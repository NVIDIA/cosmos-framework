# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training-local reader contract for canonical camera sidecars v1.0.0 and v1.1.0.

The schema definitions and pose helpers are copied from
``packages/cosmos-data/cosmos_data/multiview/{camera_sidecars,pose}.py`` at
``44167c4f26a``. Keep this bounded copy independent of ingestion dependencies.
Changes to the persisted format need an explicit reader compatibility update;
the canonical Parquet fixtures in ``test_data`` pin the supported v1 format.
"""

import math
from typing import cast

import pyarrow as pa

CAMERA_SIDECAR_SCHEMA_VERSION = b"1.0.0"
CAMERA_CALIBRATION_SCHEMA_NAME = b"cosmos.multiview.camera_calibration"
CAMERA_POSE_SCHEMA_NAME = b"cosmos.multiview.camera_pose"
TIMESTAMPS_SCHEMA_NAME = b"cosmos.multiview.timestamps"

CAMERA_CALIBRATION_SCHEMA = pa.schema(
    [
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("camera_model", pa.string(), nullable=False),
        pa.field("width_px", pa.int64(), nullable=False),
        pa.field("height_px", pa.int64(), nullable=False),
        pa.field("fx_px", pa.float64(), nullable=False),
        pa.field("fy_px", pa.float64(), nullable=False),
        pa.field("cx_px", pa.float64(), nullable=False),
        pa.field("cy_px", pa.float64(), nullable=False),
        pa.field("distortion_model", pa.string(), nullable=False),
        pa.field("distortion_coefficients", pa.list_(pa.float64()), nullable=False),
    ],
    metadata={
        b"schema_name": CAMERA_CALIBRATION_SCHEMA_NAME,
        b"schema_version": CAMERA_SIDECAR_SCHEMA_VERSION,
        b"image_axes": b"x_right_y_down",
        b"length_unit": b"pixel",
    },
)

POSE_SCHEMA = pa.schema(
    [
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("episode_time_ns", pa.int64(), nullable=False),
        pa.field("position_x_m", pa.float64(), nullable=False),
        pa.field("position_y_m", pa.float64(), nullable=False),
        pa.field("position_z_m", pa.float64(), nullable=False),
        pa.field("orientation_qw", pa.float64(), nullable=False),
        pa.field("orientation_qx", pa.float64(), nullable=False),
        pa.field("orientation_qy", pa.float64(), nullable=False),
        pa.field("orientation_qz", pa.float64(), nullable=False),
    ],
    metadata={
        b"schema_name": CAMERA_POSE_SCHEMA_NAME,
        b"schema_version": CAMERA_SIDECAR_SCHEMA_VERSION,
        b"camera_axes": b"x_right_y_down_z_forward",
        b"coordinate_system": b"right_handed",
        b"quaternion_order": b"qw_qx_qy_qz",
        b"transform_direction": b"camera_to_world",
        b"translation_unit": b"meter",
    },
)

TIMESTAMPS_SCHEMA = pa.schema(
    [
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("native_timestamp_ns", pa.int64(), nullable=False),
        pa.field("episode_time_ns", pa.int64(), nullable=False),
    ],
    metadata={
        b"schema_name": TIMESTAMPS_SCHEMA_NAME,
        b"schema_version": CAMERA_SIDECAR_SCHEMA_VERSION,
        b"clock": b"episode_clock",
        b"time_unit": b"nanosecond",
    },
)

# Released v1.1.0 sidecars (e.g. Alpamayo AV). Calibration adds the media-to-native
# pixel scale of ``nvidia_ftheta_polynomial`` rows, whose six backward terms are in
# native sensor pixels while cx/cy/width/height are in media pixels. Timestamps keep
# the v1.0.0 fields and clock semantics.
CAMERA_SIDECAR_SCHEMA_VERSION_V1_1 = b"1.1.0"
CAMERA_CALIBRATION_SCHEMA_V1_1 = pa.schema(
    [
        *(CAMERA_CALIBRATION_SCHEMA.field(name) for name in CAMERA_CALIBRATION_SCHEMA.names[:8]),
        pa.field("pixel_to_native_scale_x", pa.float64(), nullable=False),
        pa.field("pixel_to_native_scale_y", pa.float64(), nullable=False),
        *(CAMERA_CALIBRATION_SCHEMA.field(name) for name in CAMERA_CALIBRATION_SCHEMA.names[8:]),
    ],
    metadata={**(CAMERA_CALIBRATION_SCHEMA.metadata or {}), b"schema_version": CAMERA_SIDECAR_SCHEMA_VERSION_V1_1},
)
TIMESTAMPS_SCHEMA_V1_1 = TIMESTAMPS_SCHEMA.with_metadata(
    {**(TIMESTAMPS_SCHEMA.metadata or {}), b"schema_version": CAMERA_SIDECAR_SCHEMA_VERSION_V1_1}
)

QuaternionXyzw = tuple[float, float, float, float]
Rotation3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def normalize_quaternion_xyzw(
    quaternion: QuaternionXyzw,
    *,
    label: str,
    tolerance: float = 1e-4,
) -> QuaternionXyzw:
    """Return a normalized ``xyzw`` quaternion after validating it is already unit length."""
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label} has an invalid quaternion")
    if not math.isclose(norm, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} quaternion must be normalized")
    return cast(QuaternionXyzw, tuple(value / norm for value in quaternion))


def quaternion_xyzw_to_rotation_matrix(quaternion: QuaternionXyzw) -> Rotation3:
    """Convert a unit ``xyzw`` quaternion to a 3x3 rotation matrix."""
    x, y, z, w = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def validate_camera_to_world_matrix(
    matrix: Matrix4,
    *,
    label: str,
    tolerance: float = 1e-4,
) -> None:
    """Validate that a homogeneous 4x4 matrix contains a proper camera-to-world rotation."""
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{label} values must be finite")
    if any(
        not math.isclose(value, expected, abs_tol=1e-6) for value, expected in zip(matrix[3], (0, 0, 0, 1), strict=True)
    ):
        raise ValueError(f"{label} must be homogeneous")
    rotation = tuple(row[:3] for row in matrix[:3])
    for first_column in range(3):
        for second_column in range(3):
            product = sum(rotation[row][first_column] * rotation[row][second_column] for row in range(3))
            expected = 1.0 if first_column == second_column else 0.0
            if not math.isclose(product, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"{label} rotation must be orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} rotation must have determinant +1")
