# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training-local readers for sparse MADS poses and native F-theta intrinsics.

Copied from ``pipelines/sila/multiview/ingestion/av/camera_geometry.py`` at
``44167c4f26a``, retaining only the pose and F-theta readers used by WebDataset.
Poses are OpenCV camera-to-world matrices in meters. F-theta coefficients
calibrate the native distorted RGB, without substituting a pinhole projection.
"""

from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import cast

import numpy as np

from cosmos_framework.data.generator.multiview.camera_geometry_contract import (
    Matrix4,
    validate_camera_to_world_matrix,
)

_POSE_MEMBER = re.compile(r"\.(?P<frame>\d+)\.pose\.camera_(?P<camera>.+)\.npy$")
_FTHETA_MEMBER = re.compile(r"\.ftheta_intrinsic\.camera_(?P<camera>.+)\.npy$")
_FTHETA_COEFFICIENT_COUNTS = (11, 14)

# Native sidecars scale calibration uniformly, truncating the reported width.
# Raw rig dimensions were checked against all 1,034 calibrations in the 94-clip
# benchmark. Matched raw/exported frames verify full-frame, half-pixel resizing.
# Keep these profiles explicit: image dimensions alone cannot establish a crop.
MADS_FULL_FRAME_EXPORT = "mads_rdshq_full_frame_v1"
_MADS_RAW_WH_BY_CALIBRATION_WH = {
    (1924, 1084): (3848, 2168),
    (1920, 1080): (3840, 2160),
    (1719, 1080): (1936, 1216),
    (1350, 1080): (1920, 1536),
}


def mads_media_from_calibration(
    coefficients: tuple[float, ...] | list[float], *, media_hw: tuple[int, int]
) -> list[list[float]]:
    """Map native MADS calibration pixels into its full-frame exported video.

    This applies to frozen RDS-HQ calibration and full-frame exported media.
    Calibration uses origin-zero uniform scaling from the raw rig; video resize
    uses pixel centers. In particular, width 1719 represents 1719.473684... raw
    pixels after uniform scaling, while the encoder emits 1720 columns.
    """
    if len(coefficients) not in _FTHETA_COEFFICIENT_COUNTS:
        raise ValueError("MADS image mapping requires native F-theta calibration")
    calibration_wh = tuple(coefficients[2:4])
    if min(*calibration_wh, *media_hw) <= 0 or not all(np.isfinite(value) for value in calibration_wh):
        raise ValueError("MADS image mapping requires positive finite dimensions")
    raw_wh = _MADS_RAW_WH_BY_CALIBRATION_WH.get(calibration_wh)
    if raw_wh is None:
        # The ablation treats provider calibration as frozen. For other sensor
        # windows, honor the requested full-frame resize without fitting a rig.
        sx, sy = media_hw[1] / calibration_wh[0], media_hw[0] / calibration_wh[1]
        return [[sx, 0.0, (sx - 1.0) / 2], [0.0, sy, (sy - 1.0) / 2], [0.0, 0.0, 1.0]]
    raw_w, raw_h = raw_wh
    calibration_h = coefficients[3]
    scale = calibration_h / raw_h
    height, width = media_hw
    sx, sy = width / raw_w, height / raw_h
    return [[sx / scale, 0.0, (sx - 1.0) / 2], [0.0, sy / scale, (sy - 1.0) / 2], [0.0, 0.0, 1.0]]


@dataclass(frozen=True, slots=True)
class CanonicalCameraCalibration:
    """One static camera calibration in the Pose QA sidecar contract."""

    camera_model: str
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    intrinsics: tuple[float, float, float, float]


def parse_pose_entries(payload: bytes) -> dict[str, dict[int, Matrix4]]:
    """Read camera-to-world matrices by frame ID, preserving gaps for consumers to mask."""
    arrays = _load_npy_tar(payload, pattern=_POSE_MEMBER, modality="pose")
    poses_by_camera: dict[str, dict[int, Matrix4]] = {}
    for member_path, match, array in arrays:
        camera = match.group("camera")
        frame_index = int(match.group("frame"))
        matrix = _as_camera_to_world_matrix(array, label=f"pose {member_path}")
        camera_frames = poses_by_camera.setdefault(camera, {})
        if frame_index in camera_frames:
            raise ValueError(f"duplicate pose frame {frame_index} for camera {camera}")
        camera_frames[frame_index] = matrix
    return poses_by_camera


def parse_ftheta_intrinsic_tar(payload: bytes) -> dict[str, CanonicalCameraCalibration]:
    """Return per-camera f-theta calibrations from one nested npy tar."""
    calibrations: dict[str, CanonicalCameraCalibration] = {}
    for member_path, match, array in _load_npy_tar(payload, pattern=_FTHETA_MEMBER, modality="ftheta_intrinsic"):
        camera = match.group("camera")
        if camera in calibrations:
            raise ValueError(f"duplicate ftheta intrinsic for camera {camera}")
        calibrations[camera] = parse_ftheta_intrinsic(array, label=member_path)
    return calibrations


def parse_ftheta_intrinsic(array: np.ndarray, *, label: str) -> CanonicalCameraCalibration:
    """Interpret a native MADS f-theta vector, retaining optional affine lens terms.

    Provider layout is ``[cx, cy, width, height, a0, a1, a2, a3, a4, a5, flag]``.
    The flag distinguishes radius-to-angle from angle-to-radius coefficients.
    The legacy equivalent focal length is retained for metadata compatibility; ray attention
    uses the complete native polynomial and optional final ``[c,d,e]`` terms.
    Pose QA cannot score ``camera_model=ftheta``; those cameras remain attached
    because that is the native MADS image-formation model. MADS does not ingest
    pinhole as a substitute on the same still-distorted frames.
    """
    if array.ndim != 1 or array.size not in _FTHETA_COEFFICIENT_COUNTS:
        raise ValueError(f"{label} ftheta intrinsic must have 11 or 14 coefficients")
    values = tuple(float(value) for value in array.reshape(-1))
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{label} ftheta intrinsic must contain finite values")
    cx_px, cy_px, _width, _height, _a0, linear_rad_per_px, *_rest = values
    if linear_rad_per_px == 0.0:
        raise ValueError(f"{label} ftheta linear coefficient must be non-zero")
    focal_px = abs(1.0 / linear_rad_per_px)
    return CanonicalCameraCalibration(
        camera_model="ftheta",
        distortion_model="ftheta",
        distortion_coefficients=values,
        intrinsics=(focal_px, focal_px, cx_px, cy_px),
    )


def _load_npy_tar(
    payload: bytes,
    *,
    pattern: re.Pattern[str],
    modality: str,
) -> tuple[tuple[str, re.Match[str], np.ndarray], ...]:
    members = _npy_tar_members(payload)
    if not members:
        raise ValueError(f"{modality} sidecar tar contains no .npy members")
    matched: list[tuple[str, re.Match[str], np.ndarray]] = []
    for member_path, array in members:
        match = pattern.search(PurePosixPath(member_path).name)
        if match is None:
            continue
        matched.append((member_path, match, array))
    if not matched:
        raise ValueError(f"{modality} sidecar tar has no camera {modality} npy members")
    return tuple(matched)


def _npy_tar_members(payload: bytes) -> tuple[tuple[str, np.ndarray], ...]:
    members: list[tuple[str, np.ndarray]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not PurePosixPath(member.name).name.endswith(".npy"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"tar member is not a regular file: {member.name}")
                try:
                    array = np.load(io.BytesIO(extracted.read()), allow_pickle=False)
                except (ValueError, OSError) as exc:
                    raise ValueError(f"could not load npy member {member.name}") from exc
                members.append((member.name, np.asarray(array)))
    except tarfile.TarError as exc:
        raise ValueError("camera geometry sidecar must be a nested tar of .npy members") from exc
    return tuple(members)


def _as_camera_to_world_matrix(array: np.ndarray, *, label: str) -> Matrix4:
    if array.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {array.shape}")
    matrix = cast(Matrix4, tuple(tuple(float(value) for value in row) for row in array))
    validate_camera_to_world_matrix(matrix, label=label)
    return matrix
