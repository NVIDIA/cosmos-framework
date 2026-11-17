# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the camera-trajectory diagnostics in inference.

``_resolve_camera_trajectory_poses`` reshapes what the user asked for: it discards everything
after a '|', and pads or truncates to the frame budget. Each is reported, once per distinct
message, because the first sample is resolved twice per run without keep_going.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.inference import camera

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def seen(monkeypatch) -> list[str]:
    """Collect log.warning calls, with a clean seen-set for each test."""
    camera._WARNED_CAMERA_TRAJECTORY.clear()
    captured: list[str] = []
    monkeypatch.setattr(camera.log, "warning", lambda message, **_: captured.append(message))
    yield captured
    camera._WARNED_CAMERA_TRAJECTORY.clear()


def test_padding_to_the_frame_budget_is_reported(seen) -> None:
    poses = camera._resolve_camera_trajectory_poses("w-10", 61)

    assert poses.shape == (61, 4, 4)
    assert len(seen) == 1
    assert "11 poses for 61 frames" in seen[0]
    assert "final 50 frames" in seen[0]


def test_truncation_to_the_frame_budget_is_reported(seen) -> None:
    poses = camera._resolve_camera_trajectory_poses("w-200", 61)

    assert poses.shape == (61, 4, 4)
    assert len(seen) == 1
    assert "201 poses for 61 frames" in seen[0]
    assert "last 140 frames" in seen[0]


def test_a_discarded_pipe_tail_is_reported(seen) -> None:
    """Only the first segment is generated; nothing in the output shows the rest was dropped."""
    camera._resolve_camera_trajectory_poses("w-60|s-60", 61)

    assert len(seen) == 1
    assert "contains '|'" in seen[0]
    assert "Only 'w-60' is generated" in seen[0]


def test_a_pipe_with_nothing_after_it_is_not_reported(seen) -> None:
    """'w-60|' discards nothing, so saying the rest is discarded would be wrong."""
    camera._resolve_camera_trajectory_poses("w-60|", 61)

    assert seen == []


def test_warnings_are_logged_on_every_rank(monkeypatch) -> None:
    """Samples are sharded across replicas, so a rank-0-only warning hides the other ranks' samples."""
    camera._WARNED_CAMERA_TRAJECTORY.clear()
    calls: list[dict] = []
    monkeypatch.setattr(camera.log, "warning", lambda message, **kwargs: calls.append(kwargs))

    camera._resolve_camera_trajectory_poses("w-10", 61)
    camera._WARNED_CAMERA_TRAJECTORY.clear()

    assert calls == [{"rank0_only": False}]


@pytest.mark.parametrize("spec", ["w-60", "w-30,right-15,w-15", "w-60|s-60", "poses.json"])
def test_check_camera_trajectory_accepts_valid_specs(spec) -> None:
    """A JSON pose path is checked when it is loaded, not here."""
    camera.check_camera_trajectory(spec)


@pytest.mark.parametrize("spec", ["w-30,rigth-20", "w-6l", "  ", "|w-60"])
def test_check_camera_trajectory_rejects_malformed_specs(spec) -> None:
    with pytest.raises(ValueError):
        camera.check_camera_trajectory(spec)


def test_a_trajectory_that_fits_stays_silent(seen) -> None:
    poses = camera._resolve_camera_trajectory_poses("w-60", 61)

    assert poses.shape == (61, 4, 4)
    assert seen == []


def test_the_same_mismatch_is_reported_once_however_often_it_resolves(seen) -> None:
    for _ in range(3):
        camera._resolve_camera_trajectory_poses("w-10", 61)

    assert len(seen) == 1


def test_distinct_mismatches_are_reported_separately(seen) -> None:
    camera._resolve_camera_trajectory_poses("w-10", 61)
    camera._resolve_camera_trajectory_poses("w-61|s-61", 61)

    assert len(seen) == 3  # pad, pipe discard, truncation by 1 frame
    assert len(set(seen)) == 3


def test_an_empty_trajectory_still_raises(seen) -> None:
    with pytest.raises(ValueError, match="camera_trajectory is empty"):
        camera._resolve_camera_trajectory_poses("   ", 61)


def test_a_malformed_command_is_an_error_not_a_shorter_trajectory(seen) -> None:
    """A typo used to be skipped, so the camera silently did less than asked."""
    for spec in ("w-30,turnleft,a-20", "w-30,turnleft-20"):
        with pytest.raises(ValueError, match="Unknown camera command 'turnleft'"):
            camera._resolve_camera_trajectory_poses(spec, 61)
    with pytest.raises(ValueError, match="Malformed frame count 'xx'"):
        camera._resolve_camera_trajectory_poses("w-30,a-xx", 61)


# The command checks live here, not in parse_camera_string, because that parser is synced from
# an upstream copy. These tests pin what inference rejects before the parser sees the string.


def test_a_command_without_a_frame_count_is_rejected() -> None:
    """A typo must fail loudly. The parser skips it silently, which shortens the trajectory."""
    with pytest.raises(ValueError, match="expected '<cmd>-<frames>'"):
        camera.check_camera_trajectory("w-30,stay")


def test_a_non_integer_frame_count_is_rejected() -> None:
    """`w-6l` (letter l for 1) used to vanish from the trajectory and report success."""
    with pytest.raises(ValueError, match="expected a whole number of frames"):
        camera.check_camera_trajectory("w-6l")


@pytest.mark.parametrize("command", ["rigth-46", "turnleft-30", "foo-10", "W-30", "-30"])
def test_an_unknown_command_is_rejected(command) -> None:
    """The parser runs a misspelled command as `stay`: the camera holds still and nothing says why."""
    with pytest.raises(ValueError, match="Unknown camera command"):
        camera.check_camera_trajectory(f"w-30,{command}")


@pytest.mark.parametrize("count", ["1.5", "+5", "1_0", "1e2", "٣", "", "-5"])
def test_a_frame_count_that_is_not_ascii_digits_is_rejected(count) -> None:
    """int() alone would accept '+5', '1_0' or other scripts' digits; a float or a negative is a typo."""
    with pytest.raises(ValueError, match="Malformed"):
        camera.check_camera_trajectory(f"w-{count}")


@pytest.mark.parametrize("cmd", ["w-30-40", "orbit-60-1-2"])
def test_extra_fields_are_rejected(cmd) -> None:
    """Only orbit takes a third field (its radius); the parser ignores any other extra part."""
    with pytest.raises(ValueError, match="Malformed camera command"):
        camera.check_camera_trajectory(cmd)


@pytest.mark.parametrize("radius", ["abc", "nan", "inf"])
def test_a_malformed_orbit_radius_is_rejected(radius) -> None:
    """The parser falls back to radius 1.0 for 'abc', and would build NaN poses for 'nan' or 'inf'."""
    with pytest.raises(ValueError, match="Malformed orbit radius"):
        camera.check_camera_trajectory(f"orbit-60-{radius}")


@pytest.mark.parametrize("cmd", ["pano-0", "orbit-0"])
def test_a_full_turn_in_zero_frames_is_rejected(cmd) -> None:
    """The parser divides 360 degrees by the frame count, which raises ZeroDivisionError."""
    with pytest.raises(ValueError, match="at least 1 frame"):
        camera.check_camera_trajectory(cmd)


def test_every_command_with_spaces_and_an_orbit_radius_is_accepted() -> None:
    spec = " , ".join(f"{c}-4" for c in sorted(camera._CAMERA_COMMANDS)) + ",orbit-4-1.5, w - 2"
    camera.check_camera_trajectory(spec)


def test_empty_segments_are_still_accepted() -> None:
    """Trailing and doubled commas are harmless separators, which the parser skips."""
    camera.check_camera_trajectory("w-30,,")
    camera.check_camera_trajectory("w-30,")


def test_the_accepted_commands_match_the_parser() -> None:
    """Every accepted name must move the camera in the parser, except `stay`.

    The parser runs an unknown name as `stay`, so a name the parser dropped or renamed would show
    up here as a pose sequence identical to `stay`.
    """
    still = camera.parse_camera_string("stay-4")
    for name in sorted(camera._CAMERA_COMMANDS - {"stay"}):
        poses = camera.parse_camera_string(f"{name}-4")
        assert poses.shape == still.shape, name
        assert not np.allclose(poses, still), f"{name!r} does not move the camera in parse_camera_string"
