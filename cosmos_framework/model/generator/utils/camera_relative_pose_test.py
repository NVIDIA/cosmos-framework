# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.model.generator.utils.camera_relative_pose import load_camera_pose_geometry

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _poses(num_frames: int) -> list[list[list[float]]]:
    """Identity rotations translating along x by the frame index."""
    poses = torch.eye(4, dtype=torch.float64).repeat(num_frames, 1, 1)  # [T,4,4]
    poses[:, 0, 3] = torch.arange(num_frames, dtype=torch.float64)
    return poses.tolist()


def _write_sidecar(tmp_path: Path, key: str, **fields: Any) -> str:
    path = tmp_path / f"{key}.json"
    path.write_text(json.dumps({"camera_key": key, "camera_to_world": _poses(10), **fields}))
    return str(path)


def test_source_frame_indices_select_the_mapped_pose_rows(tmp_path: Path) -> None:
    mapping = [0, 2, 4, 6, 8]
    paths = [_write_sidecar(tmp_path, key, fps=10.0, source_frame_indices=mapping) for key in ("front", "rear")]

    geometry = load_camera_pose_geometry(["front", "rear"], paths, num_frames=3, start_frame=1)

    assert geometry["camera_to_world"][:, :, 0, 3].tolist() == [[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]


def test_source_frame_indices_must_be_all_or_none(tmp_path: Path) -> None:
    paths = [
        _write_sidecar(tmp_path, "front", fps=10.0, source_frame_indices=[0, 2, 4, 6, 8]),
        _write_sidecar(tmp_path, "rear", fps=10.0),
    ]

    with pytest.raises(ValueError, match="all provide source_frame_indices, or none"):
        load_camera_pose_geometry(["front", "rear"], paths, num_frames=3)


def _static_pose() -> torch.Tensor:  # returns [4,4]
    static = torch.eye(4, dtype=torch.float64)  # [4,4]
    static[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    return static


def _calibrated_fields(**calibration: float) -> dict[str, Any]:
    """The image metadata RigRoPE inference requires beside the poses; ``calibration`` overrides fields."""
    pinhole = {
        "camera_model": "pinhole",
        "distortion_model": "none",
        "distortion_coefficients": [],
        "width_px": 1280,
        "height_px": 720,
        "fx_px": 900.0,
        "fy_px": 900.0,
        "cx_px": 639.5,
        "cy_px": 359.5,
        **calibration,
    }
    return {"calibration": pinhole, "image_hw": [720, 1280], "image_from_calibration": torch.eye(3).tolist()}


def test_static_pose_covers_absolute_source_frame_indices(tmp_path: Path) -> None:
    static = _static_pose()  # [4,4]
    path = _write_sidecar(
        tmp_path, "front", camera_to_world=static.tolist(), fps=10.0, source_frame_indices=[100, 102, 104]
    )

    geometry = load_camera_pose_geometry(["front"], [path], num_frames=3)

    torch.testing.assert_close(geometry["camera_to_world"], static.expand(1, 3, 4, 4))


def test_static_pose_takes_a_single_pose_valid_flag(tmp_path: Path) -> None:
    invalid = _write_sidecar(tmp_path, "front", camera_to_world=_static_pose().tolist(), pose_valid=False)
    with pytest.raises(ValueError, match="contains missing poses"):
        load_camera_pose_geometry(["front"], [invalid], num_frames=3)

    per_frame = _write_sidecar(tmp_path, "rear", camera_to_world=_static_pose().tolist(), pose_valid=[True] * 3)
    with pytest.raises(ValueError, match="single pose_valid flag"):
        load_camera_pose_geometry(["rear"], [per_frame], num_frames=3)


def test_static_pose_without_rig_geometry_makes_a_4x4_camera_to_rig(tmp_path: Path) -> None:
    static = _static_pose()  # [4,4]
    path = _write_sidecar(tmp_path, "front", camera_to_world=static.tolist(), **_calibrated_fields())

    geometry = load_camera_pose_geometry(
        ["front"], [path], num_frames=3, fps=10.0, image_hw=(720, 1280), require_sensor_rig=True
    )

    assert geometry["rig_geometry"] == {"camera_to_rig": {"front": static.tolist()}, "per_frame_camera_to_rig": True}


@pytest.mark.parametrize(
    ("mapping", "expected_source_fps"),
    [({"fps": 30.0, "source_frame_indices": [2, 4, 6, 8]}, 30.0), ({}, 24.0)],
    ids=["mapped", "unmapped"],
)
def test_rig_provenance_carries_the_clock_frame_indices_count(
    tmp_path: Path, mapping: dict[str, Any], expected_source_fps: float
) -> None:
    """RigRoPE times LiDAR source-frame IDs against ``frame_indices`` at ``provenance["source_fps"]``.

    A mapped sidecar counts its own 30 FPS frames, so 24 FPS output must not relabel that clock.
    """
    paths = [_write_sidecar(tmp_path, key, **mapping, **_calibrated_fields()) for key in ("front", "rear")]

    geometry = load_camera_pose_geometry(
        ["front", "rear"], paths, num_frames=3, fps=24.0, image_hw=(720, 1280), require_sensor_rig=True
    )

    assert geometry["provenance"]["source_fps"] == [expected_source_fps] * 2
    frame_indices = geometry["frame_indices"].double()  # [V,F]
    frame_times_s = geometry["frame_times_ns"].double() / 1e9  # [V,F]
    torch.testing.assert_close(
        frame_times_s - frame_times_s[:, :1], (frame_indices - frame_indices[:, :1]) / expected_source_fps
    )


def test_mapped_cameras_must_share_a_source_fps(tmp_path: Path) -> None:
    paths = [
        _write_sidecar(tmp_path, "front", fps=30.0, source_frame_indices=[0, 1, 2]),
        _write_sidecar(tmp_path, "rear", fps=15.0, source_frame_indices=[0, 1, 2]),
    ]

    with pytest.raises(ValueError, match="different source FPS"):
        load_camera_pose_geometry(["front", "rear"], paths, num_frames=3)


@pytest.mark.parametrize(
    "invalid", [{"fx_px": 0.0}, {"fy_px": -900.0}, {"cx_px": math.inf}], ids=["zero_fx", "negative_fy", "inf_cx"]
)
def test_calibrated_inference_geometry_rejects_invalid_calibration(tmp_path: Path, invalid: dict[str, float]) -> None:
    path = _write_sidecar(tmp_path, "front", camera_to_world=_static_pose().tolist(), **_calibrated_fields(**invalid))

    with pytest.raises(ValueError, match="'front': invalid calibration in pose_path"):
        load_camera_pose_geometry(["front"], [path], num_frames=3, fps=10.0, image_hw=(720, 1280))


def test_calibrated_inference_geometry_rejects_malformed_calibration(tmp_path: Path) -> None:
    fields = _calibrated_fields()
    del fields["calibration"]["fx_px"]
    path = _write_sidecar(tmp_path, "front", camera_to_world=_static_pose().tolist(), **fields)

    with pytest.raises(ValueError, match="'front': invalid calibration in pose_path"):
        load_camera_pose_geometry(["front"], [path], num_frames=3, fps=10.0, image_hw=(720, 1280))


@pytest.mark.parametrize("fields", [{}, {"fps": 0.0}], ids=["missing", "zero"])
def test_source_frame_indices_require_a_positive_source_fps(tmp_path: Path, fields: dict[str, Any]) -> None:
    paths = [_write_sidecar(tmp_path, "front", source_frame_indices=[0, 2, 4, 6, 8], **fields)]

    with pytest.raises(ValueError, match="require a positive source fps"):
        load_camera_pose_geometry(["front"], paths, num_frames=3)
