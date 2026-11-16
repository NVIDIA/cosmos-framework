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


def test_parse_camera_string_rejects_a_command_without_a_frame_count() -> None:
    """A typo must fail loudly. Skipping it silently shortens the trajectory instead."""
    with pytest.raises(ValueError, match="expected '<cmd>-<frames>'"):
        parse_camera_string("w-30,stay")


def test_parse_camera_string_rejects_a_non_integer_frame_count() -> None:
    """`w-6l` (letter l for 1) used to vanish from the trajectory and report success."""
    with pytest.raises(ValueError, match="expected a whole number of frames"):
        parse_camera_string("w-6l")


@pytest.mark.parametrize("command", ["rigth-46", "turnleft-30", "foo-10", "W-30", "-30", "|w-30"])
def test_parse_camera_string_rejects_an_unknown_command(command) -> None:
    """A misspelled command used to run as `stay`: the camera held still and nothing said why."""
    with pytest.raises(ValueError, match="Unknown camera command"):
        parse_camera_string(f"w-30,{command}")


@pytest.mark.parametrize("count", ["1.5", "+5", "1e2", "\u0663", "", "-5"])
def test_parse_camera_string_rejects_a_frame_count_that_is_not_ascii_digits(count) -> None:
    """int() alone would accept '+5' or other scripts' digits; a float or a negative is a typo."""
    with pytest.raises(ValueError, match="Malformed"):
        parse_camera_string(f"w-{count}")


@pytest.mark.parametrize("cmd", ["w-30-40", "w-30|a-60", "orbit-60-1-2"])
def test_parse_camera_string_rejects_extra_fields(cmd) -> None:
    """Only orbit takes a third field (its radius); elsewhere the extra part used to be ignored."""
    with pytest.raises(ValueError, match="Malformed camera command"):
        parse_camera_string(cmd)


def test_parse_camera_string_rejects_a_malformed_orbit_radius() -> None:
    with pytest.raises(ValueError, match="Malformed orbit radius"):
        parse_camera_string("orbit-60-abc")


@pytest.mark.parametrize("cmd", ["pano-0", "orbit-0"])
def test_parse_camera_string_rejects_a_full_turn_in_zero_frames(cmd) -> None:
    """These divide 360 degrees by the frame count, which used to raise ZeroDivisionError."""
    with pytest.raises(ValueError, match="at least 1 frame"):
        parse_camera_string(cmd)


def test_parse_camera_string_accepts_every_command() -> None:
    """All 15 commands chain, with an orbit radius and spaces around the separators."""
    commands = ["w", "s", "a", "d", "u", "n", "up", "down", "left", "right", "cw", "ccw", "stay", "pano", "orbit"]
    poses = parse_camera_string(" , ".join(f"{c}-4" for c in commands) + ",orbit-4-1.5, w - 2")

    assert poses.shape == (1 + 4 * len(commands) + 4 + 2, 4, 4)


def test_parse_camera_string_still_tolerates_empty_segments() -> None:
    """Trailing and doubled commas are harmless separators, not typos."""
    assert parse_camera_string("w-30,,").shape == parse_camera_string("w-30").shape
    assert parse_camera_string("w-30,").shape[0] == 31


def _captured_warnings(monkeypatch) -> list[str]:
    """Collect log.warning calls. The logger is loguru-backed, which caplog does not capture."""
    from cosmos_framework.data.generator.action.utils import camera_trajectory

    seen: list[str] = []
    monkeypatch.setattr(camera_trajectory.log, "warning", lambda message, **_: seen.append(message))
    return seen


def test_trajectory_to_action_says_one_frame_not_one_frames(monkeypatch) -> None:
    seen = _captured_warnings(monkeypatch)

    trajectory_to_action(parse_camera_string("w-61"), 61, _params(61))

    assert len(seen) == 1
    assert "last 1 frame of requested motion" in seen[0]


def test_trajectory_to_action_reports_padding_with_the_counts(monkeypatch) -> None:
    """Padding holds the camera still for the remainder, and that is invisible in the output."""
    seen = _captured_warnings(monkeypatch)

    trajectory_to_action(parse_camera_string("w-10"), 61, _params(61))

    assert len(seen) == 1
    assert "11 poses for 61 frames" in seen[0]
    assert "final 50 frames" in seen[0]


def test_trajectory_to_action_reports_truncation_with_the_counts(monkeypatch) -> None:
    """Truncation discards requested motion, which looks like the model ignoring the request."""
    seen = _captured_warnings(monkeypatch)

    trajectory_to_action(parse_camera_string("w-200"), 61, _params(61))

    assert len(seen) == 1
    assert "201 poses for 61 frames" in seen[0]
    assert "last 140 frames" in seen[0]


def test_trajectory_to_action_stays_silent_when_the_counts_match(monkeypatch) -> None:
    """A trajectory that fits must not warn, or the warning stops meaning anything."""
    seen = _captured_warnings(monkeypatch)

    trajectory_to_action(parse_camera_string("w-60"), 61, _params(61))

    assert seen == []
