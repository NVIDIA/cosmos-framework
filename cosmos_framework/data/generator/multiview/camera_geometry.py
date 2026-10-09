# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional camera metadata, selected only after the final RGB training window.

Pixel coordinates denote pixel centers at integer locations. Calibration stays
in its native lens model; ``image_from_calibration`` records the aspect-preserving
resize and center crop into the returned video. Poses stay float64 in the supplied
OpenCV camera-to-world frame (meters), without per-camera recentering.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.multiview.camera_geometry_contract import (
    CAMERA_CALIBRATION_SCHEMA,
    CAMERA_CALIBRATION_SCHEMA_V1_1,
    POSE_SCHEMA,
    TIMESTAMPS_SCHEMA,
    TIMESTAMPS_SCHEMA_V1_1,
    Matrix4,
    QuaternionXyzw,
    normalize_quaternion_xyzw,
    quaternion_xyzw_to_rotation_matrix,
    validate_camera_to_world_matrix,
)
from cosmos_framework.data.generator.multiview.camera_geometry_native import (
    MADS_FULL_FRAME_EXPORT,
    mads_media_from_calibration,
)
from cosmos_framework.model.generator.utils.camera_rays import SUPPORTED_RAY_CAMERA_MODELS, validate_token_geometry
from cosmos_framework.utils.generator.video_preprocess import resize_and_crop_affine

SOURCE_GEOMETRY_KEY = "_camera_geometry_source"
SOURCE_FPS_KEY = "_camera_source_fps"
SOURCE_CALIBRATION_ERROR_KEY = "_camera_calibration_error"
# Per-view ``[V,3]`` float64 rows (zoom, top fraction, left fraction) of the crop frame
# extraction applied; absent means every view was center-cropped at zoom 1.
VIEW_CROP_KEY = "_view_crop"
# Vehicle cameras sit within a few meters of the rear-axle rig origin.
EPISODE_START_RIG_MAX_OFFSET_M = 10.0
# Rig key marking a pose-world rig, whose camera-to-rig transform is each selected frame's pose.
PER_FRAME_CAMERA_TO_RIG_KEY = "per_frame_camera_to_rig"

SOURCE_RIG_JSON_KEY = "_rig_json"


class CameraCalibrationError(ValueError):
    """Malformed sample calibration, distinct from IO and training-layout errors."""


@dataclass
class CameraGeometrySource:
    """One physical camera, before view/frame sampling; frame IDs index its video."""

    calibration: dict[str, Any] | None = None
    poses: dict[int, Matrix4] = field(default_factory=dict)
    times_ns: dict[int, int] | None = None
    native_times_ns: dict[int, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


class CameraGeometry(TypedDict):
    camera_keys: list[str]
    frame_indices: torch.Tensor  # [V,T]
    camera_to_world: torch.Tensor  # [V,T,4,4]
    pose_valid: torch.Tensor  # [V,T]
    calibration: list[dict[str, Any] | None]
    calibration_valid: torch.Tensor  # [V]
    image_from_calibration: torch.Tensor  # [V,3,3]
    image_hw: torch.Tensor  # [V,2]
    frame_times_ns: torch.Tensor  # [V,T]
    time_valid: torch.Tensor  # [V,T]
    time_source: list[list[str]]
    native_timestamps_ns: torch.Tensor  # [V,T]
    native_time_valid: torch.Tensor  # [V,T]
    provenance: dict[str, Any]


def _format_prompt_number(value: float) -> str:
    """Format geometry compactly while preserving useful calibration precision."""
    if value == 0.0:
        value = 0.0
    return f"{value:.6g}"


def _format_prompt_matrix(matrix: torch.Tensor) -> str:
    """Render a small matrix as an unambiguous row-major numeric block."""
    return "[" + "; ".join(", ".join(_format_prompt_number(float(value)) for value in row) for row in matrix) + "]"


class AddCameraGeometryToCaption(Augmentor):
    """Prefix each selected view caption with its intrinsics and rig-relative pose."""

    reference_cameras: tuple[str, ...] = (
        "camera_front_wide_120fov",
        "camera_front_fisheye_200fov",
    )

    def __init__(self, caption_key: str = "ai_caption", geometry_key: str = "camera_geometry") -> None:
        super().__init__([], None, None)
        self.caption_key = caption_key
        self.geometry_key = geometry_key

    def _reference_view(self, geometry: CameraGeometry) -> int | None:
        camera_keys = geometry["camera_keys"]
        pose_valid = geometry["pose_valid"]  # [V,T]
        for camera in self.reference_cameras:
            if camera in camera_keys:
                preferred = camera_keys.index(camera)
                if bool(pose_valid[preferred].any()):
                    return preferred
        return next((view for view in range(len(camera_keys)) if bool(pose_valid[view].any())), None)

    def _intrinsics_text(self, geometry: CameraGeometry, view: int) -> str:
        calibration = geometry["calibration"][view]
        if calibration is None:
            return "intrinsics unavailable"
        native_k = torch.tensor(  # [3,3]
            [
                [calibration["fx_px"], 0.0, calibration["cx_px"]],
                [0.0, calibration["fy_px"], calibration["cy_px"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        distortion = ", ".join(_format_prompt_number(float(value)) for value in calibration["distortion_coefficients"])
        output_height, output_width = (int(value) for value in geometry["image_hw"][view].tolist())
        return (
            f"native {calibration['camera_model']} image {calibration['width_px']}x{calibration['height_px']} pixels, "
            f"K={_format_prompt_matrix(native_k)} pixels, distortion={calibration['distortion_model']}[{distortion}], "
            f"native-to-model-image affine={_format_prompt_matrix(geometry['image_from_calibration'][view])}, "
            f"model image {output_width}x{output_height} pixels"
        )

    def _extrinsics_text(self, geometry: CameraGeometry, view: int, reference_view: int | None) -> str:
        if reference_view is None:
            return "relative extrinsics unavailable"
        jointly_valid = geometry["pose_valid"][reference_view] & geometry["pose_valid"][view]  # [T]
        valid_frames = torch.nonzero(jointly_valid, as_tuple=False).flatten()  # [N_valid]
        if valid_frames.numel() == 0:
            return "relative extrinsics unavailable"
        frame = int(valid_frames[0])
        camera_to_reference = (  # [4,4]
            torch.linalg.inv(geometry["camera_to_world"][reference_view, frame])
            @ geometry["camera_to_world"][view, frame]
        )
        rotation = camera_to_reference[:3, :3]  # [3,3]
        position = camera_to_reference[:3, 3]  # [3]
        viewing_direction = rotation[:, 2]  # [3]
        horizontal_offset_degrees = math.degrees(  # []
            math.atan2(float(viewing_direction[0]), float(viewing_direction[2]))
        )
        vertical_offset_degrees = math.degrees(  # []
            math.atan2(
                -float(viewing_direction[1]),
                math.hypot(float(viewing_direction[0]), float(viewing_direction[2])),
            )
        )
        source_frame = int(geometry["frame_indices"][view, frame])
        reference_camera = geometry["camera_keys"][reference_view]
        return (
            f"relative to reference camera {reference_camera} at source frame {source_frame}, this camera center is "
            f"at [{_format_prompt_number(float(position[0]))} right, "
            f"{_format_prompt_number(float(position[1]))} down, "
            f"{_format_prompt_number(float(position[2]))} forward] meters and its optical axis points along "
            f"[{_format_prompt_number(float(viewing_direction[0]))}, "
            f"{_format_prompt_number(float(viewing_direction[1]))}, "
            f"{_format_prompt_number(float(viewing_direction[2]))}] in the reference-camera coordinates. "
            f"Its horizontal viewing offset is {_format_prompt_number(horizontal_offset_degrees)}° "
            f"(positive right) and vertical viewing offset is {_format_prompt_number(vertical_offset_degrees)}° "
            f"(positive up). The exact current-camera-to-reference transform is "
            f"{_format_prompt_matrix(camera_to_reference[:3])}"
        )

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        captions = data_dict[self.caption_key]
        geometry: CameraGeometry = data_dict[self.geometry_key]
        camera_keys = geometry["camera_keys"]
        if not isinstance(captions, list) or len(captions) != len(camera_keys):
            raise ValueError(
                "Camera-geometry caption prefixes require one separate caption per selected camera: "
                f"captions={len(captions) if isinstance(captions, list) else None}, cameras={len(camera_keys)}"
            )
        reference_view = self._reference_view(geometry)
        prefixed: list[str] = []
        for view, caption in enumerate(captions):
            caption_text = caption if isinstance(caption, str) else json.dumps(caption)
            prefix = (
                "Camera calibration for this view uses OpenCV camera axes (x right, y down, z forward). "
                f"Intrinsics: {self._intrinsics_text(geometry, view)}. "
                f"Extrinsics: {self._extrinsics_text(geometry, view, reference_view)}."
            )
            prefixed.append(f"{prefix}\n\n{caption_text}")
        data_dict[self.caption_key] = prefixed
        return data_dict


def format_camera_geometry_captions(captions: object, geometry: dict[str, Any]) -> list[str]:
    """Use the training decorator verbatim in inference and fixed validation."""
    if not isinstance(captions, list) or not all(isinstance(caption, str) for caption in captions):
        raise ValueError("Camera calibration captions require one separately formatted caption per view")
    return AddCameraGeometryToCaption()({"ai_caption": captions, "camera_geometry": geometry})["ai_caption"]


_SIDECAR_SCHEMAS: dict[str, tuple[pa.Schema, ...]] = {
    "calibration": (CAMERA_CALIBRATION_SCHEMA, CAMERA_CALIBRATION_SCHEMA_V1_1),
    "pose": (POSE_SCHEMA,),
    "timestamps": (TIMESTAMPS_SCHEMA, TIMESTAMPS_SCHEMA_V1_1),
}


def _without_list_item_names(schema: pa.Schema) -> pa.Schema:
    """Parquet writers name list children ``item`` or ``element``; the name has no meaning."""
    return pa.schema(
        [
            pa.field(item.name, pa.list_(item.type.value_type), nullable=item.nullable)
            if pa.types.is_list(item.type)
            else item
            for item in schema
        ]
    )


def _match_sidecar_schema(role: str, table_schema: pa.Schema) -> pa.Schema | None:
    metadata = table_schema.metadata or {}
    fields = _without_list_item_names(table_schema)
    for schema in _SIDECAR_SCHEMAS[role]:
        if not fields.equals(_without_list_item_names(schema), check_metadata=False):
            continue
        expected = (schema.metadata or {}).items()
        if schema is CAMERA_CALIBRATION_SCHEMA:
            # Static camera calibration has no version-specific interpretation beyond
            # its exact Arrow fields.  Older Lance exports may omit only the canonical
            # schema metadata, so accept that bounded legacy form while still rejecting
            # different fields/types or contradictory metadata values.
            matches = all(metadata.get(key, value) == value for key, value in expected)
        else:
            # Versioned layouts and pose/timestamp frame and clock semantics must remain exact.
            matches = all(metadata.get(key) == value for key, value in expected)
        if matches:
            return schema
    return None


def _calibration_from_v1_1(row: dict[str, Any]) -> dict[str, Any]:
    """Pack a v1.1.0 ``nvidia_ftheta_polynomial`` row as media-pixel F-theta coefficients.

    Media and native pixels are related by a per-axis affine resize, so offsets from
    the principal point scale exactly by ``(s_x, s_y)``. Measuring radius in
    ``s_y``-scaled media pixels turns the native backward polynomial into
    ``p_i * s_y**i``; residual x anisotropy is the lens matrix ``[[s_y / s_x, 0], [0, 1]]``.
    """
    calibration = dict(row)
    scale_x = calibration.pop("pixel_to_native_scale_x")
    scale_y = calibration.pop("pixel_to_native_scale_y")
    if not all(math.isfinite(scale) and scale > 0 for scale in (scale_x, scale_y)):
        raise ValueError(f"Invalid pixel-to-native scale: x={scale_x}, y={scale_y}")
    if calibration["distortion_model"] != "nvidia_ftheta_polynomial":
        if (scale_x, scale_y) != (1.0, 1.0):
            raise ValueError(f"Unsupported pixel-to-native scale for {calibration['distortion_model']} calibration")
        return calibration
    polynomial = list(calibration["distortion_coefficients"])
    if calibration["camera_model"] != "ftheta" or not 2 <= len(polynomial) <= 6:
        raise ValueError(f"Unsupported nvidia_ftheta_polynomial calibration with {len(polynomial)} terms")
    polynomial += [0.0] * (6 - len(polynomial))
    lens = [] if scale_x == scale_y else [scale_y / scale_x, 0.0, 0.0]
    calibration.update(
        distortion_model="ftheta",
        distortion_coefficients=[
            calibration["cx_px"],
            calibration["cy_px"],
            float(calibration["width_px"]),
            float(calibration["height_px"]),
            *(value * scale_y**power for power, value in enumerate(polynomial)),
            1.0,
            *lens,
        ],
    )
    return calibration


def read_camera_sidecars(
    stream_id: str, payloads: dict[str, bytes | None], provenance: dict[str, Any]
) -> CameraGeometrySource:
    """Decode canonical v1.0/v1.1 Parquet sidecars; absence is distinct from invalid data."""
    # The training-local contract pins the supported canonical sidecar versions.
    source = CameraGeometrySource(provenance=dict(provenance))
    for role in ("calibration", "pose", "timestamps"):
        payload = payloads.get(role)
        if payload is None:
            continue
        table = pq.read_table(pa.BufferReader(payload))
        metadata = table.schema.metadata or {}
        schema = _match_sidecar_schema(role, table.schema)
        if schema is None:
            raise ValueError(f"Unsupported {role} schema for {stream_id}")
        rows = table.to_pylist()
        if any(
            row["stream_id"] != stream_id
            or any(value is None for value in row.values())
            or None in row.get("distortion_coefficients", ())
            for row in rows
        ):
            raise ValueError(f"Invalid {role} stream identity or null values for {stream_id}")
        if role == "calibration":
            if len(rows) != 1:
                raise ValueError(f"Expected one static calibration for {stream_id}")
            source.calibration = (
                _calibration_from_v1_1(rows[0]) if schema is CAMERA_CALIBRATION_SCHEMA_V1_1 else rows[0]
            )
            _validate_calibration(source.calibration)
            continue
        indexed = {row["sample_index"]: row for row in rows}
        if len(indexed) != len(rows) or any(index < 0 for index in indexed):
            raise ValueError(f"Duplicate or negative {role} frame IDs for {stream_id}")
        if role == "timestamps":
            origin = metadata.get(b"episode_time_origin", b"episode_start").decode()
            native_origin = metadata.get(b"native_timestamp_origin", b"provider_clock").decode()
            if origin not in {"episode_start", "sample_start"} or native_origin not in {
                "sample_start",
                "provider_clock",
            }:
                raise ValueError(f"Unsupported timestamp origin for {stream_id}")
            source.provenance.update(time_origin=origin, native_timestamp_origin=native_origin)
            source.times_ns = {index: row["episode_time_ns"] for index, row in indexed.items()}
            source.native_times_ns = {index: row["native_timestamp_ns"] for index, row in indexed.items()}
            ordered_times = [source.times_ns[index] for index in sorted(indexed)]
            # Canonical timestamps are nondecreasing; distinct frames may share a timestamp.
            if any(b < a for a, b in zip(ordered_times, ordered_times[1:])):
                raise ValueError(f"Decreasing timestamps for {stream_id}")
        else:
            for index, row in indexed.items():
                quaternion = cast(QuaternionXyzw, tuple(row[f"orientation_q{axis}"] for axis in "xyzw"))
                rotation = quaternion_xyzw_to_rotation_matrix(normalize_quaternion_xyzw(quaternion, label=stream_id))
                matrix = cast(
                    Matrix4,
                    tuple((*rotation[i], row[f"position_{axis}_m"]) for i, axis in enumerate("xyz"))
                    + ((0.0, 0.0, 0.0, 1.0),),
                )
                validate_camera_to_world_matrix(matrix, label=f"{stream_id} frame {index}")
                source.poses[index] = matrix
    return source


def episode_start_rig_geometry(sources: dict[str, CameraGeometrySource]) -> dict[str, Any]:
    """Derive camera-to-rig extrinsics from poses anchored at the episode-start rig.

    Canonical AV pose sidecars (e.g. Alpamayo) express camera-to-world in the FLU ego
    rig frame at episode time zero, so each camera's pose at that instant is its static
    rig extrinsic. A pose far from the rig origin means a map-anchored world frame.
    """
    camera_to_rig: dict[str, Matrix4] = {}
    for camera, source in sources.items():
        if source.times_ns is None or source.provenance.get("time_origin") != "episode_start":
            raise ValueError(f"{camera}: rig extrinsics require episode-start timestamps")
        start = min((index for index, time in source.times_ns.items() if time == 0), default=None)
        if start is None or start not in source.poses:
            raise ValueError(f"{camera}: missing camera pose at episode start")
        pose = source.poses[start]
        offset_m = math.hypot(*(row[3] for row in pose[:3]))
        if offset_m > EPISODE_START_RIG_MAX_OFFSET_M:
            raise ValueError(f"{camera}: episode-start pose is {offset_m:.1f} m from the rig origin")
        camera_to_rig[camera] = pose
    if not camera_to_rig:
        raise ValueError("Sensor rig requires at least one camera")
    return {"camera_to_rig": camera_to_rig}


def pose_world_rig_geometry(sources: dict[str, CameraGeometrySource]) -> dict[str, Any]:
    """Use the shared pose world frame as the rig, for cameras that move within a scene.

    Every camera must carry world poses in one scene-fixed frame. ``camera_to_rig``
    holds each camera's first pose only to identify rig membership; RigRoPE reads the
    selected frames' ``camera_to_world`` because these cameras have no static extrinsic.
    """
    camera_to_rig: dict[str, Matrix4] = {}
    for camera, source in sources.items():
        if not source.poses:
            raise ValueError(f"{camera}: a pose-world rig requires camera poses")
        camera_to_rig[camera] = source.poses[min(source.poses)]
    if not camera_to_rig:
        raise ValueError("Sensor rig requires at least one camera")
    return {"camera_to_rig": camera_to_rig, PER_FRAME_CAMERA_TO_RIG_KEY: True}


def _validate_calibration(calibration: dict[str, Any]) -> None:
    coefficients = calibration["distortion_coefficients"]
    width, height = calibration["width_px"], calibration["height_px"]
    scalars = [calibration[key] for key in ("fx_px", "fy_px", "cx_px", "cy_px")]
    if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in (*scalars, *coefficients)):
        raise ValueError("Invalid camera calibration dimensions or nonfinite coefficients")
    if min(scalars[:2]) <= 0:
        raise ValueError(f"Invalid focal length: fx={scalars[0]}, fy={scalars[1]}")
    # Validate the principal point in the frozen calibration's own pixel domain,
    # before applying the separate calibration-to-media and training resize maps.
    if not (0 <= scalars[2] <= width and 0 <= scalars[3] <= height):
        raise ValueError(
            f"Invalid principal point: cx={scalars[2]}, cy={scalars[3]}, "
            f"calibration width={width}, height={height}, camera_model={calibration['camera_model']}"
        )
    if calibration["camera_model"] == "ftheta":
        if (
            calibration["distortion_model"] != "ftheta"
            or len(coefficients) not in (11, 14)
            or coefficients[5] == 0
            or min(coefficients[2:4]) <= 0
        ):
            raise ValueError("Unsupported ftheta calibration")
        if list(coefficients[2:4]) != [width, height]:
            warnings.warn(
                f"{calibration.get('stream_id', 'ftheta')}: Ftheta coefficient dimensions {coefficients[2:4]} disagree with "
                f"calibration dimensions {[width, height]}; retaining supplied calibration and assuming shared pixel coordinates.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif calibration["camera_model"] == "pinhole":
        model = calibration["distortion_model"]
        if not (
            (model == "none" and not coefficients) or (model == "opencv" and len(coefficients) in {4, 5, 8, 12, 14})
        ):
            raise ValueError("Unsupported pinhole distortion model")
    elif calibration["camera_model"] == "fisheye":
        if calibration["distortion_model"] != "equidistant" or len(coefficients) != 4:
            raise ValueError("Unsupported fisheye calibration; expected equidistant k1..k4")
    else:
        raise ValueError(f"Unsupported camera model: {calibration['camera_model']}")


class FinalizeCameraGeometry(Augmentor):
    """Align metadata to retained physical views and source frames, without RNG draws.

    Invalid calibration on a selected camera skips the clip (returns ``None``)
    so WebDataset workers drop it instead of crashing the job. Other per-clip
    geometry ValueErrors (layout, FPS, clock origin) skip the same way.
    Canonical sidecar readers still raise on malformed parquet.
    """

    require_camera_poses: bool
    require_camera_calibration: bool
    filter_invalid_camera_calibration: bool
    fallback_invalid_camera_calibration: bool
    calibration_token_stride: int
    _pose_filter_counts: dict[tuple[int, int], tuple[int, int]]
    _calibration_filter_counts: dict[tuple[int, int], tuple[int, int]]
    _calibration_fallback_counts: tuple[int, int]

    def __init__(
        self,
        require_camera_poses: bool = False,
        require_camera_calibration: bool = False,
        filter_invalid_camera_calibration: bool = False,
        calibration_token_stride: int = 32,
        fallback_invalid_camera_calibration: bool = False,
    ) -> None:
        super().__init__([], None, None)
        if (
            filter_invalid_camera_calibration or fallback_invalid_camera_calibration
        ) and not require_camera_calibration:
            raise ValueError("Calibration filtering or fallback requires require_camera_calibration=True")
        if filter_invalid_camera_calibration and fallback_invalid_camera_calibration:
            raise ValueError("Invalid calibration is either filtered or falls back, not both")
        if calibration_token_stride <= 0:
            raise ValueError("Calibration token stride must be positive")
        self.require_camera_poses = require_camera_poses
        self.require_camera_calibration = require_camera_calibration
        self.filter_invalid_camera_calibration = filter_invalid_camera_calibration
        self.fallback_invalid_camera_calibration = fallback_invalid_camera_calibration
        # The caller selects the grid used by its geometry adapter.
        self.calibration_token_stride = calibration_token_stride
        self._pose_filter_counts = {}
        self._calibration_filter_counts = {}
        self._calibration_fallback_counts = (0, 0)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        if self.fallback_invalid_camera_calibration:
            return self._finalize_or_fall_back(data_dict)
        if not self.filter_invalid_camera_calibration:
            try:
                return self._finalize(data_dict)
            except ValueError as error:
                log.warning(
                    f"Skipping camera-geometry clip {data_dict.get('__key__')!r}: {error}",
                    rank0_only=False,
                )
                return None
        video = data_dict["video"]
        resolution = tuple((video[-1] if isinstance(video, list) else video).shape[-2:])
        checked, dropped = self._calibration_filter_counts.get(resolution, (0, 0))
        checked += 1
        worker = torch.utils.data.get_worker_info()
        context = (
            f"CameraCalibrationFilter worker={worker.id if worker is not None else 'main'} "
            f"pid={os.getpid()} resolution={resolution[0]}x{resolution[1]}"
        )
        try:
            result = self._finalize(data_dict)
        except CameraCalibrationError as error:
            dropped += 1
            result = None
            # Emit every rejection so the persisted rank logs form a complete rejection ledger.
            log.warning(
                f"{context} checked={checked} calibration_drops={dropped}: "
                f"Dropping sample {data_dict.get('__key__')!r}: {error}",
                rank0_only=False,
            )
        self._calibration_filter_counts[resolution] = (checked, dropped)
        if result is not None:
            # Pass cumulative worker counters through accepted rows. The training monitor
            # de-duplicates repeated CP-window rows and reports rejections separately from fallback.
            result["camera_geometry"]["provenance"]["calibration_filter_stats"] = {
                "worker": f"{os.getpid()}:{resolution[0]}x{resolution[1]}",
                "checked": checked,
                "rejected": dropped,
            }
        if checked == 1 or checked % 100 == 0:
            log.info(
                f"{context} checked={checked} calibration_drops={dropped} drop_pct={100.0 * dropped / checked:.2f}",
                rank0_only=False,
            )
        return result

    def _finalize_or_fall_back(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Keep a sample whose geometry is absent or invalid, with ``camera_geometry=None``.

        A row the source loaded no geometry for (no source, or a ``None`` one) passes through
        without geometry. Any other sample goes through the same checks as the filter, and one
        failing them keeps its media and captions but carries no geometry, so a geometry-
        conditioned model takes its fallback for it instead of losing the sample. The pose
        requirement, when set, still drops samples and logs its own drops.
        """
        if SOURCE_CALIBRATION_ERROR_KEY not in data_dict and data_dict.get(SOURCE_GEOMETRY_KEY) is None:
            data_dict.pop(SOURCE_GEOMETRY_KEY, None)
            data_dict.pop(SOURCE_FPS_KEY, None)
            data_dict.pop(SOURCE_RIG_JSON_KEY, None)
            data_dict["camera_geometry"] = None
            return data_dict
        checked, fell_back = self._calibration_fallback_counts
        checked += 1
        try:
            result = self._finalize(data_dict)
        except CameraCalibrationError as error:
            fell_back += 1
            for key in (SOURCE_CALIBRATION_ERROR_KEY, SOURCE_GEOMETRY_KEY, SOURCE_FPS_KEY, SOURCE_RIG_JSON_KEY):
                data_dict.pop(key, None)
            data_dict["camera_geometry"] = None
            result = data_dict
            if fell_back <= 3 or fell_back % 100 == 0:
                log.warning(
                    f"CameraCalibrationFallback pid={os.getpid()} checked={checked} fallbacks={fell_back}: "
                    f"sample {data_dict.get('__key__')!r} keeps no camera geometry: {error}",
                    rank0_only=False,
                )
        self._calibration_fallback_counts = (checked, fell_back)
        if checked == 1 or checked % 100 == 0:
            log.info(
                f"CameraCalibrationFallback pid={os.getpid()} checked={checked} fallbacks={fell_back} "
                f"fallback_pct={100.0 * fell_back / checked:.2f}",
                rank0_only=False,
            )
        return result

    def _finalize(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        parse_error = data_dict.pop(SOURCE_CALIBRATION_ERROR_KEY, None)
        if parse_error is not None:
            raise CameraCalibrationError(parse_error)
        sources: dict[str, CameraGeometrySource] = data_dict.pop(SOURCE_GEOMETRY_KEY)
        source_fps: list[float] = data_dict.pop(SOURCE_FPS_KEY)
        rig_json = data_dict.pop(SOURCE_RIG_JSON_KEY, None)
        cameras = list(data_dict["camera_keys_selection"])
        indices = data_dict["frame_indices"].tolist()
        views, frames = len(cameras), len(indices)
        video = data_dict["video"]
        clips = video if isinstance(video, list) else [video]
        if (
            not views
            or frames != int(data_dict["num_video_frames_per_view"])
            or any(
                clip.ndim != 4 or clip.shape[1] != views * frames or clip.shape[-2:] != clips[-1].shape[-2:]
                for clip in clips
            )
        ):
            raise ValueError("Camera geometry does not match the final RGB layout")
        height, width = clips[-1].shape[-2:]
        original_hw = data_dict["original_hw"].tolist()
        if (
            len(original_hw) != views
            or any(len(hw) != 2 or min(hw) <= 0 for hw in original_hw)
            or len(source_fps) != views
            or any(fps <= 0 or not math.isfinite(fps) for fps in source_fps)
        ):
            raise ValueError("Invalid camera source dimensions or FPS")
        view_crops = data_dict.get(VIEW_CROP_KEY)
        crops = [(1.0, 0.5, 0.5)] * views if view_crops is None else [tuple(crop) for crop in view_crops.tolist()]
        if len(crops) != views or any(len(crop) != 3 for crop in crops):
            raise ValueError("View crops do not match the camera views")
        poses = torch.eye(4, dtype=torch.float64).repeat(views, frames, 1, 1)  # [V,T,4,4]
        pose_valid = torch.zeros((views, frames), dtype=torch.bool)  # [V,T]
        affine = torch.eye(3, dtype=torch.float64).repeat(views, 1, 1)  # [V,3,3]
        times = torch.zeros((views, frames), dtype=torch.int64)  # [V,T]
        time_valid = torch.zeros((views, frames), dtype=torch.bool)  # [V,T]
        native_times = torch.zeros_like(times)  # [V,T]
        native_valid = torch.zeros_like(time_valid)  # [V,T]
        calibration: list[dict[str, Any] | None] = []
        time_source: list[list[str]] = []
        provenance: list[dict[str, Any]] = []
        origins = {
            sources[camera].provenance.get("time_origin", "sample_start")
            for camera in cameras
            if camera in sources and sources[camera].times_ns is not None
        }
        if len(origins) > 1:
            raise ValueError("Camera timestamp sidecars declare incompatible clock origins")
        clock_origin = next(iter(origins), "sample_start")
        for view, camera in enumerate(cameras):
            source = sources.get(camera, CameraGeometrySource())
            cal = deepcopy(source.calibration)
            if cal is None and rig_json is not None:
                raise CameraCalibrationError(f"missing calibration for {camera} with rig_json present")
            calibration.append(cal)
            provenance.append(dict(source.provenance))
            try:
                rig = source.provenance.get("rig_geometry")
                if "rig_geometry" in source.provenance and (rig is None or camera not in rig["camera_to_rig"]):
                    raise ValueError(f"{camera}: missing calibrated rig extrinsic")
                if self.require_camera_calibration and cal is None:
                    raise ValueError(f"{camera}: ray attention requires native calibration")
                if cal is not None:
                    _validate_calibration(cal)
                    media_from_calibration = source.provenance.get("media_from_calibration")
                    if (
                        self.require_camera_calibration
                        and media_from_calibration is None
                        and source.provenance.get("image_export") == MADS_FULL_FRAME_EXPORT
                    ):
                        media_from_calibration = mads_media_from_calibration(
                            cal["distortion_coefficients"], media_hw=tuple(original_hw[view])
                        )
                        provenance[-1]["media_from_calibration"] = media_from_calibration
                    if self.require_camera_calibration:
                        if cal["camera_model"] not in SUPPORTED_RAY_CAMERA_MODELS:
                            raise ValueError(
                                f"{camera}: ray attention cannot unproject camera model {cal['camera_model']!r}"
                            )
                        if media_from_calibration is None and [cal["height_px"], cal["width_px"]] != original_hw[view]:
                            raise ValueError(f"{camera}: unresolved calibration-to-media mapping")
                    if [cal["height_px"], cal["width_px"]] != original_hw[view] and media_from_calibration is None:
                        warnings.warn(
                            f"{camera}: calibration dimensions {[cal['height_px'], cal['width_px']]} differ from "
                            f"media dimensions {original_hw[view]} (H, W); retaining supplied calibration and assuming "
                            "shared pixel coordinates for the media resize.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    zoom, crop_top, crop_left = crops[view]
                    affine[view] = resize_and_crop_affine(
                        tuple(original_hw[view]), (height, width), zoom=zoom, crop_center=(crop_top, crop_left)
                    )  # [3,3]
                    if self.require_camera_calibration and media_from_calibration is not None:
                        media_affine = torch.as_tensor(media_from_calibration, dtype=torch.float64)  # [3,3]
                        if media_affine.shape != (3, 3) or not torch.isfinite(media_affine).all():
                            raise ValueError(f"{camera}: invalid calibration-to-media mapping")
                        affine[view] = affine[view] @ media_affine  # [3,3]
                    if self.filter_invalid_camera_calibration or self.fallback_invalid_camera_calibration:
                        validate_token_geometry(
                            cal,
                            affine[view],
                            image_hw=(height, width),
                            stride=self.calibration_token_stride,
                        )
            except ValueError as error:
                raise CameraCalibrationError(
                    f"{camera}: {error}; sample={source.provenance.get('sample_key', data_dict.get('__key__'))}, "
                    f"calibration_source={source.provenance.get('ftheta_intrinsic')}, "
                    f"media_hw={original_hw[view]}"
                ) from error
            labels: list[str] = []
            for frame, index in enumerate(indices):
                if index in source.poses:
                    poses[view, frame] = torch.tensor(source.poses[index], dtype=torch.float64)  # [4,4]
                    pose_valid[view, frame] = True  # []
                if source.times_ns is None and clock_origin == "sample_start":
                    times[view, frame] = round(index * 1e9 / source_fps[view])  # []
                    time_valid[view, frame] = True  # []
                    labels.append("nominal_fps")
                elif source.times_ns is not None and index in source.times_ns:
                    times[view, frame] = source.times_ns[index]  # []
                    time_valid[view, frame] = True  # []
                    labels.append("sidecar")
                else:
                    labels.append("missing")
                if index in source.native_times_ns:
                    native_times[view, frame] = source.native_times_ns[index]  # []
                    native_valid[view, frame] = True  # []
            time_source.append(labels)
        if self.require_camera_poses:
            missing_poses = views > 1 and not bool(pose_valid.all())
            # Worker-local cumulative counts, separated by output resolution. Checked
            # includes single-camera samples, which bypass the pose requirement.
            checked, dropped = self._pose_filter_counts.get((height, width), (0, 0))
            checked += 1
            dropped += int(missing_poses)
            self._pose_filter_counts[(height, width)] = (checked, dropped)
            log_summary = checked == 1 or checked % 100 == 0
            log_drop = missing_poses and (dropped <= 3 or dropped % 100 == 0)
            if log_summary or log_drop:
                worker = torch.utils.data.get_worker_info()
                context = (
                    f"CameraPoseFilter worker={worker.id if worker is not None else 'main'} "
                    f"pid={os.getpid()} resolution={height}x{width}"
                )
                if log_summary:
                    log.info(
                        f"{context} checked={checked} missing_pose_drops={dropped} "
                        f"drop_pct={100.0 * dropped / checked:.2f}",
                        rank0_only=False,
                    )
                if log_drop:
                    missing = {
                        camera: [index for frame, index in enumerate(indices) if not pose_valid[view, frame]][:8]
                        for view, camera in enumerate(cameras)
                        if not pose_valid[view].all()
                    }
                    log.warning(
                        f"{context}: Dropping sample {data_dict.get('__key__')!r} from {data_dict.get('__url__')!r}: "
                        f"missing required camera poses (first 8 source frames per camera): {missing}",
                        rank0_only=False,
                    )
            if missing_poses:
                return None
        offset = int(times[time_valid].min()) if time_valid.any() else 0  # []
        times[time_valid] -= offset  # [N_valid]
        url = data_dict.get("__url__", "")
        source_url = f"{url.root}/{url.path}" if hasattr(url, "root") else str(url)
        data_dict["camera_geometry"] = CameraGeometry(
            camera_keys=cameras,
            frame_indices=torch.tensor(indices, dtype=torch.int64).repeat(views, 1),  # [V,T]
            camera_to_world=poses,
            pose_valid=pose_valid,
            calibration=calibration,
            calibration_valid=torch.tensor([cal is not None for cal in calibration], dtype=torch.bool),  # [V]
            image_from_calibration=affine,
            image_hw=torch.tensor([height, width], dtype=torch.int64).repeat(views, 1),  # [V,2]
            frame_times_ns=times,
            time_valid=time_valid,
            time_source=time_source,
            native_timestamps_ns=native_times,
            native_time_valid=native_valid,
            provenance={
                "sample_key": str(data_dict["__key__"]),
                "source_url": source_url,
                "cameras": provenance,
                "time_origin": clock_origin,
                "time_offset_ns": offset,
                "source_fps": source_fps,
                "rig_json": rig_json,
            },
        )
        return data_dict
