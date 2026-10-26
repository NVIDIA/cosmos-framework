# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Camera-pose sidecar conversion for multiview generation training."""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.utils.action_processing import (
    GLOBAL_ASINH_METRIC_GRIPPER_STATS_PATH,
    ActionNormalizer,
    ActionProcessingRecord,
    load_action_normalization_stats,
    make_camera_global_asinh_normalizer,
    make_pose_action_scale_normalizer,
    pad_action_to_max_dim,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.pose_utils import (
    PoseConvention,
    RotationConvention,
    pose_abs_to_rel,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    EGO_POSE,
    LEFT_OPENNESS,
    LEFT_WRIST_POSE,
    RIGHT_OPENNESS,
    RIGHT_WRIST_POSE,
    UNIFIED_ACTION_DIM,
)

# Per-sample camera-to-source-stream mapping, carried from the source through the
# augmentation chain. A sidecar's rows are stamped with the source stream ID, which a
# discovering source cannot recover from the camera name: it binds a row's own streams
# to generic view slots, so the slot a stream lands in varies from sample to sample.
SOURCE_STREAM_ID_KEY = "_camera_source_stream_id"
# Per-sample ``CameraActionSlots`` of a ``unified_wrist_slots`` source, carried through the
# augmentation chain like ``SOURCE_STREAM_ID_KEY``: which unified_v1 block each camera's pose
# rows take and the gripper openness that goes with a wrist camera.
CAMERA_ACTION_SLOTS_KEY = "_camera_action_slots"

_EXPECTED_POSE_METADATA: dict[bytes, bytes] = {
    b"schema_name": b"cosmos.multiview.camera_pose",
    b"camera_axes": b"x_right_y_down_z_forward",
    b"transform_direction": b"camera_to_world",
    b"translation_unit": b"meter",
    b"quaternion_order": b"qw_qx_qy_qz",
}
_POSE_FIELDS: tuple[str, ...] = (
    "stream_id",
    "sample_index",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "orientation_qw",
    "orientation_qx",
    "orientation_qy",
    "orientation_qz",
)
_MADS_POSE_MEMBER = re.compile(r"\.(?P<frame>\d+)\.pose\.(?P<stream_id>camera_.+)\.npy$")
CameraActionNormalization = Literal["scale", "global_asinh"]
# ``per_view_ego``: every view's raw ``xyz + rot6d`` at slots ``0:9``. ``unified_wrist_slots``: a
# wrist camera's rows at its arm's unified_v1 wrist block plus that gripper's openness, every
# other camera's at the ego block ``0:9``.
CameraActionLayout = Literal["per_view_ego", "unified_wrist_slots"]
CameraActionSlot = Literal["ego", "right_wrist", "left_wrist"]
_POSE_SLICE_BY_SLOT: dict[CameraActionSlot, slice] = {
    "ego": EGO_POSE,
    "right_wrist": RIGHT_WRIST_POSE,
    "left_wrist": LEFT_WRIST_POSE,
}
_OPENNESS_INDEX_BY_SLOT: dict[CameraActionSlot, int] = {"right_wrist": RIGHT_OPENNESS, "left_wrist": LEFT_OPENNESS}


@dataclass(frozen=True)
class CameraActionSlotBinding:
    """The unified_v1 block one camera's pose rows take, with the gripper openness of a wrist camera.

    ``openness`` is canonical openness (``0`` closed, ``1`` fully open) per source frame, indexed like
    the pose sidecar's ``sample_index``; action row ``t`` (the move from ``frame_indices[t]`` to
    ``frame_indices[t + 1]``) reads ``openness[frame_indices[t + openness_frame_offset]]``, so ``1``
    pairs a row with its target frame and ``0`` with its source frame, as each single-view Action
    adapter pairs its gripper column. A wrist slot requires ``openness``; the ego slot has none.
    """

    slot: CameraActionSlot
    openness: np.ndarray | None = None  # [F]
    openness_frame_offset: int = 1

    def __post_init__(self) -> None:
        if self.slot not in _POSE_SLICE_BY_SLOT:
            raise ValueError(f"Unknown camera action slot {self.slot!r}; expected {sorted(_POSE_SLICE_BY_SLOT)}")
        if (self.openness is None) != (self.slot == "ego"):
            raise ValueError(f"Camera action slot {self.slot!r}: openness is set exactly for a wrist slot")
        if self.openness_frame_offset not in (0, 1):
            raise ValueError(f"openness_frame_offset must be 0 or 1, got {self.openness_frame_offset}")
        if self.openness is not None:
            openness = np.asarray(self.openness)
            if openness.ndim != 1 or openness.size == 0 or not np.isfinite(openness).all():
                raise ValueError(f"openness must be a nonempty finite [F] array, got shape {openness.shape}")


@dataclass(frozen=True)
class CameraActionSlots:
    """One sample's ``CameraActionSlotBinding`` per camera key and its gripper's fully-open width.

    ``fully_open_gripper_width_m`` is the dataset's verified physical width ``W``: the canonical
    metric-gripper profile's meter bounds are divided by it, exactly as the single-view Lance
    adapters do, so one openness value means the same physical opening across robots.
    """

    by_camera: Mapping[str, CameraActionSlotBinding]
    fully_open_gripper_width_m: float | None

    def __post_init__(self) -> None:
        width = self.fully_open_gripper_width_m
        if width is not None and (not np.isfinite(width) or width <= 0.0):
            raise ValueError(f"fully_open_gripper_width_m must be finite and positive, got {width}")
        if width is None and any(binding.slot != "ego" for binding in self.by_camera.values()):
            raise ValueError("Wrist camera action slots require fully_open_gripper_width_m")


@dataclass(frozen=True)
class CameraPoseActionTensors:
    """Camera-major action tensors derived from sampled multiview poses."""

    action: torch.Tensor  # [V*(T-1),D_model]
    action_raw: torch.Tensor  # [V*(T-1),D_raw]
    raw_action_dim: torch.Tensor  # []
    action_valid_mask: torch.Tensor  # [D_model]
    action_processing_record: ActionProcessingRecord


def read_camera_pose_table_from_bytes(payload: bytes | bytearray | memoryview) -> pa.Table:
    """Read one canonical camera-pose parquet sidecar from bytes."""
    payload_bytes = bytes(payload)
    if not payload_bytes:
        raise ValueError("camera pose sidecar payload is empty")
    return pq.read_table(pa.BufferReader(payload_bytes))


def _quaternion_wxyz_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        quaternion = (
            0.25 * scale,
            (float(rotation[2, 1]) - float(rotation[1, 2])) / scale,
            (float(rotation[0, 2]) - float(rotation[2, 0])) / scale,
            (float(rotation[1, 0]) - float(rotation[0, 1])) / scale,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(rotation)))
        if diagonal_index == 0:
            scale = float(np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0)
            quaternion = (
                (float(rotation[2, 1]) - float(rotation[1, 2])) / scale,
                0.25 * scale,
                (float(rotation[0, 1]) + float(rotation[1, 0])) / scale,
                (float(rotation[0, 2]) + float(rotation[2, 0])) / scale,
            )
        elif diagonal_index == 1:
            scale = float(np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0)
            quaternion = (
                (float(rotation[0, 2]) - float(rotation[2, 0])) / scale,
                (float(rotation[0, 1]) + float(rotation[1, 0])) / scale,
                0.25 * scale,
                (float(rotation[1, 2]) + float(rotation[2, 1])) / scale,
            )
        else:
            scale = float(np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0)
            quaternion = (
                (float(rotation[1, 0]) - float(rotation[0, 1])) / scale,
                (float(rotation[0, 2]) + float(rotation[2, 0])) / scale,
                (float(rotation[1, 2]) + float(rotation[2, 1])) / scale,
                0.25 * scale,
            )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("camera pose rotation produced an invalid quaternion")
    normalized = tuple(float(value / norm) for value in quaternion)
    if normalized[0] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized


def _camera_pose_table_from_mads_frames(
    stream_id: str,
    frames: dict[int, np.ndarray],
    *,
    source_fps: float,
) -> pa.Table:
    """Convert one MADS camera's contiguous OpenCV camera-to-world matrices."""
    if not frames:
        raise ValueError(f"MADS pose archive has no frames for {stream_id!r}")
    expected_indices = list(range(max(frames) + 1))
    if sorted(frames) != expected_indices:
        raise ValueError(f"MADS pose frames for {stream_id!r} must be contiguous from 0")

    rows: list[dict[str, int | float | str]] = []
    for sample_index in expected_indices:
        matrix = frames[sample_index]  # [4,4]
        label = f"MADS pose {stream_id} frame {sample_index}"
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{label} must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
            raise ValueError(f"{label} must be homogeneous")
        rotation = matrix[:3, :3]  # [3,3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"{label} rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"{label} rotation must have determinant +1")
        quaternion = _quaternion_wxyz_from_rotation_matrix(rotation)
        rows.append(
            {
                "stream_id": stream_id,
                "sample_index": sample_index,
                "episode_time_ns": round(sample_index * 1_000_000_000 / source_fps),
                "position_x_m": float(matrix[0, 3]),
                "position_y_m": float(matrix[1, 3]),
                "position_z_m": float(matrix[2, 3]),
                "orientation_qw": quaternion[0],
                "orientation_qx": quaternion[1],
                "orientation_qy": quaternion[2],
                "orientation_qz": quaternion[3],
            }
        )
    schema = pa.schema(
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
            **_EXPECTED_POSE_METADATA,
            b"schema_version": b"1.0.0",
            b"coordinate_system": b"right_handed",
        },
    )
    return pa.Table.from_pylist(rows, schema=schema)


def camera_pose_tables_from_mads_archive(
    payload: bytes,
    *,
    camera_keys: Sequence[str],
    stream_id_mapping: Mapping[str, str],
    source_fps: float,
) -> dict[str, pa.Table]:
    """Convert a MADS per-frame NPY pose tar into canonical per-camera tables."""
    if not payload:
        raise ValueError("MADS pose archive payload is empty")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")

    wanted = set(camera_keys)
    frames_by_stream: dict[str, dict[int, np.ndarray]] = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            match = _MADS_POSE_MEMBER.search(member.name)
            if match is None:
                continue
            stream_id = match.group("stream_id")
            if stream_id not in wanted:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"MADS pose archive member {member.name!r} is not a regular file")
            frame_index = int(match.group("frame"))
            frames = frames_by_stream.setdefault(stream_id, {})
            if frame_index in frames:
                raise ValueError(f"MADS pose archive has duplicate frame {frame_index} for {stream_id!r}")
            frames[frame_index] = np.load(io.BytesIO(stream.read()), allow_pickle=False)  # [4,4]

    missing = [camera_key for camera_key in camera_keys if camera_key not in frames_by_stream]
    if missing:
        raise ValueError(f"MADS pose archive is missing camera streams {missing}")
    missing_stream_ids = [camera_key for camera_key in camera_keys if camera_key not in stream_id_mapping]
    if missing_stream_ids:
        raise ValueError(f"MADS stream ID mapping is missing camera keys {missing_stream_ids}")
    return {
        camera_key: _camera_pose_table_from_mads_frames(
            stream_id_mapping[camera_key],
            frames_by_stream[camera_key],
            source_fps=source_fps,
        )
        for camera_key in camera_keys
    }


def write_camera_pose_table(table: pa.Table, path: Path) -> None:
    """Persist one canonical camera-pose table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _validate_pose_table_schema(table: pa.Table) -> None:
    """Validate the canonical pose sidecar fields and metadata."""
    missing_fields = [field for field in _POSE_FIELDS if field not in table.column_names]
    if missing_fields:
        raise ValueError(f"camera pose sidecar is missing fields {missing_fields}")

    metadata = table.schema.metadata or {}
    mismatched_metadata = {
        key: metadata.get(key) for key, expected in _EXPECTED_POSE_METADATA.items() if metadata.get(key) != expected
    }
    if mismatched_metadata:
        expected = {key.decode("utf-8"): value.decode("utf-8") for key, value in _EXPECTED_POSE_METADATA.items()}
        got = {
            key.decode("utf-8"): (value.decode("utf-8") if value is not None else None)
            for key, value in mismatched_metadata.items()
        }
        raise ValueError(f"camera pose sidecar metadata mismatch: got {got}, expected {expected}")


def _as_float(row: dict[str, Any], key: str, *, row_label: str) -> float:
    """Return one finite float field from a pose row."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row_label} has invalid {key!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"{row_label} has non-finite {key!r}: {value}")
    return value


def _c2w_from_pose_row(row: dict[str, Any], *, row_label: str) -> np.ndarray:
    """Reconstruct one homogeneous camera-to-world matrix from a pose row."""
    qw = _as_float(row, "orientation_qw", row_label=row_label)
    qx = _as_float(row, "orientation_qx", row_label=row_label)
    qy = _as_float(row, "orientation_qy", row_label=row_label)
    qz = _as_float(row, "orientation_qz", row_label=row_label)
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)  # [4]
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{row_label} has non-unit quaternion norm {norm}")
    quaternion = quaternion / norm  # [4]

    w, x, y, z = quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrix = np.eye(4, dtype=np.float32)  # [4,4]
    matrix[:3, :3] = np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )  # [3,3]
    matrix[:3, 3] = np.asarray(
        [
            _as_float(row, "position_x_m", row_label=row_label),
            _as_float(row, "position_y_m", row_label=row_label),
            _as_float(row, "position_z_m", row_label=row_label),
        ],
        dtype=np.float32,
    )  # [3]
    return matrix  # [4,4]


def camera_to_world_from_pose_table(
    table: pa.Table,
    frame_indices: Sequence[int],
    expected_stream_id: str | None = None,
) -> np.ndarray:
    """Select sampled rows and reconstruct canonical camera-to-world matrices."""
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    _validate_pose_table_schema(table)
    if table.num_rows == 0:
        raise ValueError("camera pose sidecar contains no rows")

    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in table.select(_POSE_FIELDS).to_pylist():
        try:
            sample_index = int(row["sample_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("camera pose sidecar has invalid sample_index") from error
        if sample_index in rows_by_index:
            raise ValueError(f"camera pose sidecar has duplicate sample_index {sample_index}")
        rows_by_index[sample_index] = row

    selected_rows: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    for frame_index in frame_indices:
        row = rows_by_index.get(int(frame_index))
        if row is None:
            missing_indices.append(int(frame_index))
            continue
        if expected_stream_id is not None and str(row["stream_id"]) != expected_stream_id:
            raise ValueError(
                f"camera pose sidecar stream_id {row['stream_id']!r} does not match expected "
                f"{expected_stream_id!r} at sample_index {frame_index}"
            )
        selected_rows.append(row)
    if missing_indices:
        raise ValueError(f"camera pose sidecar is missing pose rows for frame indices {missing_indices}")

    poses = [
        _c2w_from_pose_row(row, row_label=f"stream {row['stream_id']} sample_index {row['sample_index']}")
        for row in selected_rows
    ]
    return np.stack(poses, axis=0).astype(np.float32, copy=False)  # [T,4,4]


@cache
def _make_camera_pose_action_normalizer(
    *,
    raw_action_dim: int,
    pose_convention: PoseConvention,
    camera_action_normalization: CameraActionNormalization,
    camera_translation_scale: float,
    camera_rotation_scale: float,
) -> ActionNormalizer:
    """Build the model-space camera action normalizer used by training and inference."""
    if camera_action_normalization == "global_asinh":
        if camera_translation_scale != 1.0:
            raise ValueError("global_asinh camera actions require camera_translation_scale=1.0")
        if camera_rotation_scale != 1.0:
            raise ValueError("global_asinh camera actions require camera_rotation_scale=1.0")
        return make_camera_global_asinh_normalizer(pose_convention=pose_convention)
    return make_pose_action_scale_normalizer(
        raw_action_dim,
        translation_scale=camera_translation_scale,
        rotation_scale=camera_rotation_scale,
    )


@cache
def make_camera_unified_global_asinh_normalizer(
    *,
    pose_convention: PoseConvention,
    fully_open_gripper_width_m: float | None,
) -> ActionNormalizer:
    """The canonical 59-D metric-gripper ``global_asinh`` normalizer in one gripper's openness units.

    The openness slots' meter bounds ``[0, R]`` become ``[0, R / W]``, as in
    ``LanceActionSampler._metric_gripper_stats``; every pose slot keeps the profile's bounds.
    """
    stats = load_action_normalization_stats(
        "global_asinh",
        stats_path=GLOBAL_ASINH_METRIC_GRIPPER_STATS_PATH,
        expected_dim=UNIFIED_ACTION_DIM,
    )
    if fully_open_gripper_width_m is not None:
        gripper_slots = [RIGHT_OPENNESS, LEFT_OPENNESS]
        q01 = stats["q01"].clone()  # [59]
        q99 = stats["q99"].clone()  # [59]
        q01[gripper_slots] /= fully_open_gripper_width_m
        q99[gripper_slots] /= fully_open_gripper_width_m
        stats = {**stats, "q01": q01, "q99": q99}
    return resolve_action_normalization("global_asinh", stats, pose_convention=pose_convention)


# eq=False: generated equality would compare the mask tensor elementwise.
@dataclass(frozen=True, eq=False)
class CameraSlotActionCodec:
    """Decode camera-major unified-slot actions, restoring each row's unfilled slots to exactly zero.

    The record's ``action_valid_mask`` is the union over views, so it cannot tell a head row from a
    wrist row; denormalizing an unfilled slot's zero would otherwise yield a nonzero raw value.
    """

    action_normalizer: ActionNormalizer
    row_valid_mask: torch.Tensor  # [V*(T-1),59] bool

    def decode_to_unified(
        self, normalized_action: torch.Tensor
    ) -> torch.Tensor:  # normalized_action: [...,V*(T-1),59], returns [...,V*(T-1),59]
        if tuple(normalized_action.shape[-2:]) != tuple(self.row_valid_mask.shape):
            raise ValueError(
                f"camera action rows {tuple(normalized_action.shape[-2:])} do not match the recorded slot mask "
                f"{tuple(self.row_valid_mask.shape)}"
            )
        raw = self.action_normalizer.denormalize_action(normalized_action)  # [...,V*(T-1),59]
        mask = self.row_valid_mask.to(device=raw.device)  # [V*(T-1),59]
        return torch.where(mask, raw, torch.zeros_like(raw))  # [...,V*(T-1),59]


def build_unified_camera_slot_action(
    per_view_actions: Sequence[np.ndarray | None],  # V x [T-1,9]; None = no measured pose
    *,
    frame_indices: Sequence[int],
    action_slots: CameraActionSlots,
    camera_keys: Sequence[str],
    pose_convention: PoseConvention,
    max_action_dim: int,
) -> CameraPoseActionTensors:
    """Scatter each view's ``xyz + rot6d`` rows into its unified_v1 block, with a wrist's openness."""
    num_rows = len(frame_indices) - 1
    num_views = len(per_view_actions)
    action_raw = torch.zeros(num_views * num_rows, UNIFIED_ACTION_DIM, dtype=torch.float32)  # [V*(T-1),59]
    row_valid = torch.zeros(num_views * num_rows, UNIFIED_ACTION_DIM, dtype=torch.bool)  # [V*(T-1),59]
    frame_index_array = np.asarray(frame_indices, dtype=np.int64)  # [T]
    for view, (rel_pose, camera_key) in enumerate(zip(per_view_actions, camera_keys, strict=True)):
        if rel_pose is None:
            # Keep the camera's row range, without inventing motion for an
            # unposed view. The caller excludes these rows from Action loss.
            continue
        binding = action_slots.by_camera.get(camera_key)
        if binding is None:
            raise ValueError(f"camera {camera_key!r} has no camera action slot")
        if rel_pose.shape[1] != EGO_POSE.stop - EGO_POSE.start:
            raise ValueError(f"unified camera action slots take xyz + rot6d rows, got width {rel_pose.shape[1]}")
        rows = slice(view * num_rows, (view + 1) * num_rows)
        pose_slice = _POSE_SLICE_BY_SLOT[binding.slot]
        action_raw[rows, pose_slice] = torch.from_numpy(np.ascontiguousarray(rel_pose))  # [T-1,9]
        row_valid[rows, pose_slice] = True
        if binding.openness is not None:
            openness = np.asarray(binding.openness, dtype=np.float32)  # [F]
            offset = binding.openness_frame_offset
            openness_frames = frame_index_array[offset : offset + num_rows]  # [T-1]
            if openness_frames.min() < 0 or openness_frames.max() >= openness.size:
                raise ValueError(
                    f"camera {camera_key!r}: frame indices reach {int(openness_frames.max())}, "
                    f"openness has {openness.size} frames"
                )
            openness_index = _OPENNESS_INDEX_BY_SLOT[binding.slot]
            action_raw[rows, openness_index] = torch.from_numpy(openness[openness_frames])  # [T-1]
            row_valid[rows, openness_index] = True
    if not torch.isfinite(action_raw).all():
        raise ValueError("camera pose action contains non-finite values")
    action_normalizer = make_camera_unified_global_asinh_normalizer(
        pose_convention=pose_convention,
        fully_open_gripper_width_m=action_slots.fully_open_gripper_width_m,
    )
    action_model = torch.where(  # [V*(T-1),59]
        row_valid, action_normalizer.normalize_action(action_raw), torch.zeros_like(action_raw)
    )
    valid_mask_raw = row_valid.any(dim=0)  # [59]
    return CameraPoseActionTensors(
        action=pad_action_to_max_dim(action_model, max_action_dim),  # [V*(T-1),D_model]
        action_raw=action_raw,
        raw_action_dim=torch.tensor(UNIFIED_ACTION_DIM, dtype=torch.long),  # []
        action_valid_mask=pad_action_to_max_dim(valid_mask_raw, max_action_dim).to(dtype=torch.bool),  # [D_model]
        action_processing_record=ActionProcessingRecord(
            raw_action_dim=UNIFIED_ACTION_DIM,
            action_normalizer=None,
            action_valid_mask=valid_mask_raw,
            action_codec=CameraSlotActionCodec(action_normalizer, row_valid.clone()),
        ),
    )


def build_camera_pose_action(
    pose_tables: Sequence[pa.Table],
    *,
    frame_indices: Sequence[int],
    stream_ids: Sequence[str],
    rotation_format: RotationConvention = "rot6d",
    pose_convention: PoseConvention = "backward_chunk_anchored_16f",
    max_action_dim: int = 64,
    camera_action_normalization: CameraActionNormalization = "global_asinh",
    camera_translation_scale: float = 1.0,
    camera_rotation_scale: float = 1.0,
    action_slots: CameraActionSlots | None = None,
    camera_keys: Sequence[str] | None = None,
) -> CameraPoseActionTensors:
    """Build camera-major action rows from sampled absolute camera poses.

    Without ``action_slots`` every view's raw ``xyz + rot6d`` sits at slots ``0:9`` (``per_view_ego``).
    With them (``unified_wrist_slots``, keyed by ``camera_keys`` in view order) each view's rows take
    its camera's unified_v1 block: a wrist camera's at its arm's wrist block plus that gripper's
    openness, any other camera's at ``0:9``. Rows are then normalized by the canonical 59-D
    ``global_asinh`` profile and every slot a view does not fill stays zero, so ``action_raw`` is
    canonical 59-D and ``action_valid_mask`` is the union of the views' slots.
    """
    if len(frame_indices) < 2:
        raise ValueError("at least two frame indices are required to build camera pose actions")
    if len(pose_tables) != len(stream_ids):
        raise ValueError(f"pose table count {len(pose_tables)} does not match stream id count {len(stream_ids)}")
    if max_action_dim <= 0:
        raise ValueError(f"max_action_dim must be positive, got {max_action_dim}")
    if action_slots is not None:
        if camera_keys is None or len(camera_keys) != len(pose_tables):
            raise ValueError("unified camera action slots need one camera key per pose table")
        if camera_action_normalization != "global_asinh":
            raise ValueError("unified camera action slots require camera_action_normalization='global_asinh'")
        if rotation_format != "rot6d":
            raise ValueError("unified camera action slots require rotation_format='rot6d'")
        if camera_translation_scale != 1.0 or camera_rotation_scale != 1.0:
            raise ValueError("global_asinh camera actions require unit camera translation and rotation scales")

    per_view_actions: list[np.ndarray] = []
    raw_action_dim: int | None = None
    for pose_table, stream_id in zip(pose_tables, stream_ids, strict=True):
        c2w = camera_to_world_from_pose_table(  # [T,4,4]
            pose_table,
            frame_indices,
            expected_stream_id=stream_id,
        )
        rel_pose = pose_abs_to_rel(  # [T-1,D_raw]
            c2w,
            rotation_format=rotation_format,
            pose_convention=pose_convention,
        )
        if rel_pose.ndim != 2 or rel_pose.shape[0] != len(frame_indices) - 1:
            raise ValueError(
                f"relative camera poses for stream {stream_id!r} have shape {rel_pose.shape}, "
                f"expected ({len(frame_indices) - 1}, D)"
            )
        if raw_action_dim is None:
            raw_action_dim = int(rel_pose.shape[1])
        elif rel_pose.shape[1] != raw_action_dim:
            raise ValueError(
                f"relative camera pose width for stream {stream_id!r} is {rel_pose.shape[1]}, expected {raw_action_dim}"
            )
        per_view_actions.append(rel_pose.astype(np.float32, copy=False))

    assert raw_action_dim is not None
    if action_slots is not None:
        assert camera_keys is not None
        return build_unified_camera_slot_action(
            per_view_actions,
            frame_indices=frame_indices,
            action_slots=action_slots,
            camera_keys=camera_keys,
            pose_convention=pose_convention,
            max_action_dim=max_action_dim,
        )
    action_raw_np = np.concatenate(per_view_actions, axis=0).astype(np.float32, copy=False)  # [V*(T-1),D_raw]
    action_raw = torch.from_numpy(np.ascontiguousarray(action_raw_np)).float()  # [V*(T-1),D_raw]
    if not torch.isfinite(action_raw).all():
        raise ValueError("camera pose action contains non-finite values")
    valid_mask_raw = torch.ones(raw_action_dim, dtype=torch.bool)  # [D_raw]
    action_normalizer = _make_camera_pose_action_normalizer(
        raw_action_dim=raw_action_dim,
        pose_convention=pose_convention,
        camera_action_normalization=camera_action_normalization,
        camera_translation_scale=camera_translation_scale,
        camera_rotation_scale=camera_rotation_scale,
    )
    action_model = action_normalizer.normalize_action(action_raw)  # [V*(T-1),D_raw]
    action = pad_action_to_max_dim(action_model, max_action_dim)  # [V*(T-1),D_model]
    action_valid_mask = pad_action_to_max_dim(valid_mask_raw, max_action_dim).to(dtype=torch.bool)  # [D_model]
    action_processing_record = ActionProcessingRecord(
        raw_action_dim=raw_action_dim,
        action_normalizer=action_normalizer,
        action_valid_mask=valid_mask_raw,
    )
    return CameraPoseActionTensors(
        action=action,
        action_raw=action_raw,
        raw_action_dim=torch.tensor(raw_action_dim, dtype=torch.long),  # []
        action_valid_mask=action_valid_mask,
        action_processing_record=action_processing_record,
    )
