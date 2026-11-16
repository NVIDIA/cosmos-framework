# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the camera-trajectory diagnostics in inference.

``_resolve_camera_trajectory_poses`` reshapes what the user asked for: it discards everything
after a '|', and pads or truncates to the frame budget. Each is reported, once per distinct
message, because the first sample is resolved twice per run without keep_going.
"""

from __future__ import annotations

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
