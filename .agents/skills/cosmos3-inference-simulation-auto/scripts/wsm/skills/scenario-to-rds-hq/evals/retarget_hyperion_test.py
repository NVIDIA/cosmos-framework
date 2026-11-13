# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check pose composition, scene preservation, and prepared Hyperion calibration."""

import importlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from cosmos_auto.wm.camera.ftheta import FThetaCamera
from cosmos_auto.wm.wds_utils import get_sample

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def helper() -> Iterator[ModuleType]:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        yield importlib.import_module("retarget_hyperion")


@pytest.fixture
def native_clip(helper: ModuleType, tmp_path: Path) -> dict[str, Any]:
    io_helper = importlib.import_module("rds_hq_io")
    vehicle = np.tile(np.eye(4), (3, 1, 1))  # [3,4,4]
    vehicle[1, :3, 3] = [2, 0, 0]  # [3]
    vehicle[2, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # [3,3], 90-degree left turn
    vehicle[2, :3, 3] = [4, 1, 0.5]  # [3]
    source_mount = np.array([[0, 0, 1, 3], [-1, 0, 0, 2], [0, -1, 0, 1.6], [0, 0, 0, 1]])  # [4,4]
    camera_poses = vehicle @ source_mount  # [3,4,4]
    object_pose = np.eye(4)  # [4,4]
    object_pose[:3, 3] = [12, 0, 0.8]  # [3]
    objects = [
        {
            "parked": {
                "object_type": "Car",
                "object_is_moving": False,
                "object_lwh": [4.5, 1.8, 1.6],
                "object_to_world": object_pose.tolist(),  # [4,4]
            }
        }
        for _ in range(3)
    ]
    layers = {layer: {"labels": []} for layer in helper.MAP_GEOMETRY}
    layers["3d_road_boundaries"]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "polyline3d": {"vertices": [[0, 4, 0], [30, 4, 0]]},
                    "attributes": [{"name": "boundary_type", "text": "curb"}],
                }
            }
        }
    ]
    root = tmp_path / "native"
    io_helper.write_clip(
        root,
        "turning",
        {"camera_source": camera_poses},
        {"camera_source": np.array([500, 500, 320, 240, 640, 480])},  # each [6]
        objects,
        layers,
        {"state_to_legacy_render_state": {}, "traffic_light_states": {}},
        camera_type="pinhole",
    )
    # Sparse source object observations must survive retargeting byte for byte.
    object_tar = root / "all_object_info" / "turning.tar"
    records = io_helper.read_archive(object_tar, "turning")
    object_tar.unlink()
    io_helper.write_archive(
        object_tar, "turning", {key: value for key, value in records.items() if "000001" not in key}
    )
    rgb = root / "video_camera_source" / "turning.tar"
    rgb.parent.mkdir()
    rgb.write_bytes(b"source RGB placeholder; never a Hyperion view")
    np.save(tmp_path / "vehicle_to_world.npy", vehicle)  # [3,4,4]
    return {"root": root, "vehicle": vehicle}


def test_retargeted_poses_and_scene_through_real_reader(
    helper: ModuleType, native_clip: dict[str, Any], tmp_path: Path
) -> None:
    source, output = native_clip["root"], tmp_path / "hyperion"
    source_bytes = {path.relative_to(source): path.read_bytes() for path in source.glob("*/*.tar")}
    report = helper.retarget_clip(source, output, "turning", native_clip["vehicle"], source_camera_type="pinhole")
    assert report["pose_frames"] == 3 and report["object_frames"] == 2
    assert len(report["cameras"]) == 11
    assert report["source_only_components"] == ["video_camera_source"]
    assert not (output / "video_camera_source").exists()
    assert not (output / "pinhole_intrinsic").exists()
    for layer in helper.ANNOTATION_LAYERS:
        relative = Path(layer) / "turning.tar"
        assert (output / relative).read_bytes() == source_bytes[relative]
        assert report["annotation_sha256"][layer] == helper.sha256(source / relative)
    assert source_bytes == {path.relative_to(source): path.read_bytes() for path in source.glob("*/*.tar")}
    poses = get_sample(output / "pose" / "turning.tar")
    intrinsics = get_sample(output / "ftheta_intrinsic" / "turning.tar")
    rig = helper.load_rig()
    for camera, calibration in rig.items():
        mount = np.asarray(calibration["camera_to_vehicle"])  # [4,4]
        np.testing.assert_allclose(poses[f"000000.pose.{camera}.npy"], mount)
        final = poses[f"000002.pose.{camera}.npy"]  # [4,4]
        x, y, z = mount[:3, 3]  # [3]
        np.testing.assert_allclose(final[:3, 3], [4 - y, 1 + x, 0.5 + z])
        np.testing.assert_allclose(final[0, :3], -mount[1, :3])
        np.testing.assert_allclose(final[1, :3], mount[0, :3])
        values = intrinsics[f"ftheta_intrinsic.{camera}.npy"]  # [14]
        np.testing.assert_array_equal(values, calibration["ftheta_intrinsics"])
        model = FThetaCamera.from_numpy(values, device="cpu")
        expected_size = (1720, 1080) if "fisheye" in camera else (1924, 1084)
        assert (model.width, model.height) == expected_size
    assert json.loads((output / "retargeting" / "turning.json").read_text()) == report
    with pytest.raises(ValueError, match="already exists"):
        helper.retarget_clip(source, output, "turning", native_clip["vehicle"], source_camera_type="pinhole")


def test_mount_directions_and_rigid_rotations(helper: ModuleType) -> None:
    rig = helper.load_rig()
    expected_forward = {
        "camera_front_wide_120fov": [1, 0, 0],
        "camera_front_tele_30fov": [1, 0, 0],
        "camera_cross_left_120fov": [0.4, 0.9, 0],
        "camera_cross_right_120fov": [0.4, -0.9, 0],
        "camera_rear_left_70fov": [-0.87, 0.5, 0],
        "camera_rear_right_70fov": [-0.87, -0.5, 0],
        "camera_rear_tele_30fov": [-1, 0, 0],
        "camera_front_fisheye_200fov": [1, 0, 0],
        "camera_left_fisheye_200fov": [0, 1, 0],
        "camera_rear_fisheye_200fov": [-1, 0, 0],
        "camera_right_fisheye_200fov": [0, -1, 0],
    }
    assert set(rig) == set(expected_forward)
    for name, camera in rig.items():
        transform = np.asarray(camera["camera_to_vehicle"])  # [4,4]
        helper.check_transform(transform, name)
        assert np.dot(transform[:3, 2], expected_forward[name]) > 0.8
        assert transform[2, 1] < -0.8  # Camera-down points towards the ground.


@pytest.mark.parametrize(
    ("camera", "source_size", "target_size", "source_center", "source_polynomial"),
    [
        (
            "camera_front_wide_120fov",
            [3848, 2168],
            [1924, 1084],
            [1909.240723, 1101.091431],
            [0, 5.39158645551652e-4, -1.86381798883417e-9, 5.39678458172377e-12, -7.74941020201421e-16],
        ),
        (
            "camera_front_fisheye_200fov",
            [1936, 1216],
            [1720, 1080],
            [965.977661, 606.655396],
            [0, 2.49706231988966e-3, -1.66305525794996e-7, -9.38988997667423e-10, 5.61799521833772e-13],
        ),
    ],
)
def test_resized_rays_match_pinned_source_calibration(
    helper: ModuleType,
    camera: str,
    source_size: list[int],
    target_size: list[int],
    source_center: list[float],
    source_polynomial: list[float],
) -> None:
    # Independent source values from the pinned rig; no download during the test.
    u, v = np.meshgrid(np.linspace(0.02, 0.98, 13), np.linspace(0.02, 0.98, 9))  # each [9,13]
    fractions = np.stack((u.ravel(), v.ravel()), axis=1)  # [117,2]
    offset = fractions * source_size - source_center  # [117,2], original sensor pixels
    radius = np.linalg.norm(offset, axis=1)  # [117]
    angle = np.polynomial.polynomial.polyval(radius, source_polynomial)  # [117]
    xy = offset / radius[:, None] * np.sin(angle[:, None])  # [117,2]
    expected = np.column_stack((xy, np.cos(angle)))  # [117,3]
    intrinsic = np.asarray(helper.load_rig()[camera]["ftheta_intrinsics"])  # [14]
    model = FThetaCamera.from_numpy(intrinsic, device="cpu")
    actual, valid = model.pixel2ray_np(fractions * target_size)  # [117,3], [117]
    assert valid.all()
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize("invalid", ["frame_count", "rotation", "nan"])
def test_bad_vehicle_poses_do_not_publish(
    helper: ModuleType, native_clip: dict[str, Any], tmp_path: Path, invalid: str
) -> None:
    poses = native_clip["vehicle"].copy()  # [3,4,4]
    if invalid == "frame_count":
        poses = poses[:2]  # [2,4,4]
    elif invalid == "rotation":
        poses[0, 0, 0] = 2
    else:
        poses[0, 0, 3] = np.nan
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        helper.retarget_clip(native_clip["root"], output, "turning", poses, source_camera_type="pinhole")
    assert not output.exists()


def test_cli_and_ftheta_source(helper: ModuleType, native_clip: dict[str, Any], tmp_path: Path) -> None:
    assert helper.__file__ is not None
    script = Path(helper.__file__)
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(native_clip["root"]),
            str(output),
            "turning",
            "--vehicle-poses",
            str(tmp_path / "vehicle_to_world.npy"),
            "--source-camera-type",
            "pinhole",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(json.loads(completed.stdout)["cameras"]) == 11
    # A second pass from F-theta data must use the supplied vehicle poses again,
    # without accumulating camera offsets from the already-retargeted cameras.
    second = tmp_path / "second"
    helper.retarget_clip(output, second, "turning", native_clip["vehicle"])
    for layer in ("pose", "ftheta_intrinsic", *helper.ANNOTATION_LAYERS):
        assert (output / layer / "turning.tar").read_bytes() == (second / layer / "turning.tar").read_bytes()
