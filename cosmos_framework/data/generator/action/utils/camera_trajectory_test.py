# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import numpy as np
import pytest

from cosmos_framework.data.generator.action.utils.camera_trajectory import (
    CameraTrainParams,
    load_custom_pose_jsons,
    parse_camera_string,
    parse_custom_trajectories,
    trajectory_to_action,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _params(num_frames: int) -> CameraTrainParams:
    return CameraTrainParams(
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.0,
        rotation_scale=1.0,
        num_frames=num_frames,
    )


def test_parse_camera_string_emits_one_more_pose_than_frames() -> None:
    poses = parse_camera_string("w-4")

    assert poses.shape == (5, 4, 4)
    np.testing.assert_array_equal(poses[0], np.eye(4))


def test_parse_camera_string_translates_forward_along_plus_z() -> None:
    poses = parse_camera_string("w-3", forward_speed=0.1)

    # OpenCV convention: forward is +Z, and nothing rotates.
    np.testing.assert_allclose(poses[-1][:3, 3], np.array([0.0, 0.0, 0.3]), atol=1e-12)
    np.testing.assert_allclose(poses[-1][:3, :3], np.eye(3), atol=1e-12)


def test_parse_camera_string_stay_holds_pose() -> None:
    poses = parse_camera_string("stay-5")

    assert poses.shape == (6, 4, 4)
    for pose in poses:
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-12)


def test_parse_camera_string_pano_completes_full_yaw() -> None:
    poses = parse_camera_string("pano-8")

    assert poses.shape == (9, 4, 4)
    # A full 360 degree yaw returns to the identity rotation.
    np.testing.assert_allclose(poses[-1][:3, :3], np.eye(3), atol=1e-9)


@pytest.mark.parametrize(
    "camera_string",
    [
        "W-30",  # commands are lower case
        "forward-30",  # not a command
        "w30",  # no frame count
        "w-3o",  # frame count is not an integer
        "w-30;right-15",  # wrong separator
        "w-30-5",  # extra field on a command that takes none
        "right-15|w-15",  # '|' belongs to parse_custom_trajectories
        "orbit-150-far",  # orbit radius is not a number
    ],
)
def test_parse_camera_string_rejects_malformed_commands(camera_string: str) -> None:
    # Each of these used to parse to a camera that never moves.
    with pytest.raises(ValueError, match="Invalid camera command"):
        parse_camera_string(camera_string)


@pytest.mark.parametrize("camera_string", ["pano-0", "orbit-0"])
def test_parse_camera_string_rejects_a_zero_frame_pano_or_orbit(camera_string: str) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        parse_camera_string(camera_string)


def test_parse_camera_string_still_accepts_blank_entries_and_spaces() -> None:
    # A trailing comma, an empty entry and spaces around '-' are all accepted.
    np.testing.assert_array_equal(parse_camera_string("w-2, ,right-1,"), parse_camera_string("w-2,right-1"))
    np.testing.assert_array_equal(parse_camera_string(" w - 2 "), parse_camera_string("w-2"))


def test_parse_camera_string_orbit_radius_override() -> None:
    # Half an orbit puts the camera opposite its start, two radii ahead.
    poses = parse_camera_string("orbit-4-2.0")

    np.testing.assert_allclose(poses[2][:3, 3], np.array([0.0, 0.0, 4.0]), atol=1e-12)


def test_parse_custom_trajectories_splits_on_pipe() -> None:
    trajectories = parse_custom_trajectories("w-2|s-3")

    assert len(trajectories) == 2
    assert trajectories[0].shape == (3, 4, 4)
    assert trajectories[1].shape == (4, 4, 4)


def test_trajectory_to_action_pads_short_trajectories() -> None:
    poses = parse_camera_string("w-2")  # 3 poses
    assert poses.shape[0] == 3

    action = trajectory_to_action(poses, 6, _params(6))

    # One action per transition.
    assert action.shape[0] == 5


def test_trajectory_to_action_truncates_long_trajectories() -> None:
    poses = parse_camera_string("w-10")  # 11 poses

    action = trajectory_to_action(poses, 4, _params(4))

    assert action.shape[0] == 3


def test_load_custom_pose_jsons_reads_each_path(tmp_path) -> None:
    poses = np.tile(np.eye(4), (3, 1, 1))
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(poses.tolist()))
    second.write_text(json.dumps(poses.tolist()))

    trajectories = load_custom_pose_jsons(f"{first},{second}")

    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert trajectory.shape == (3, 4, 4)


def test_load_custom_pose_jsons_rejects_wrong_shape(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError, match=r"expected shape \(T, 4, 4\)"):
        load_custom_pose_jsons(str(bad))
