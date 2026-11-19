# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scoring: the right pane, and never a bare number without its control."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from failuregen import scoring


def _write_clip(path: Path, frames: list[np.ndarray], fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tb = Fraction(1, fps)
    with av.open(str(path), "w") as c:
        s = c.add_stream("h264", rate=fps)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        s.codec_context.time_base = tb
        for i, f in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            frame.pts, frame.time_base = i, tb
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)


def _canvas(left: int, right: int) -> np.ndarray:
    """A frame laid out like the model's: wide view on top, two wrist panes underneath."""
    h, w = 96, 96
    img = np.zeros((h, w, 3), dtype=np.uint8)
    top = int(h * 2 / 3)
    img[:top] = 90
    img[top:, : w // 2] = left
    img[top:, w // 2 :] = right
    return img


def _steady(value: int = 40, n: int = 64) -> list[int]:
    return [value] * n


def _changes_after_injection(before: int = 40, after: int = 220, n: int = 64) -> list[int]:
    """Constant up to the injection frame (32), then different.

    The metric measures change against the injection frame itself, so a fixture that changes at
    that frame puts the whole difference in the pre window instead.
    """
    return [before] * 40 + [after] * (n - 40)


def _run(
    dir_: Path,
    arm: str,
    left: list[int],
    right: list[int],
    *,
    mode="transplant",
    recipe="r",
    t_inject=6.0,
    failure_time_s=6.4,
    episode=1,
) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    _write_clip(dir_ / "synth.mp4", [_canvas(l, r) for l, r in zip(left, right)])
    (dir_ / "record.json").write_text(
        json.dumps(
            {
                "name": dir_.name,
                "recipe": recipe,
                "mode": mode,
                "arm": arm,
                "target_phase": "handover",
                "fps": 5,
                "failure_time_s": failure_time_s,
                "host": {"source": "X", "episode_index": episode, "t_start": 0.0, "t_inject": t_inject},
            }
        )
    )
    return dir_


def test_the_pane_scored_is_the_holding_hand(profile, tmp_path):
    """The record names the arm; defaulting to the first one measures a hand that never changes.

    Here only the right pane changes. Scored as the right hand it is a large change; scored as the
    left it is nothing, which is what a hardcoded pane reported after every handover.
    """
    run = _run(tmp_path / "right_hand", "right", _steady(), _changes_after_injection())
    right = scoring.score_run(profile, run)
    assert right is not None and right.post > 0.2

    record = json.loads((run / "record.json").read_text())
    (run / "record.json").write_text(json.dumps(record | {"arm": "left"}))
    left = scoring.score_run(profile, run)
    assert left.post < 0.02


def test_a_run_without_a_control_is_unscorable_not_scored(profile, tmp_path):
    """An absolute number cannot be interpreted: untouched clips range from 0.08 to 0.26 across
    phases, so a run whose control is missing must be reported as such."""
    _run(tmp_path / "inj", "left", _changes_after_injection(), _steady())
    scores = scoring.score_directory(profile, tmp_path)
    assert len(scores) == 1
    assert scores[0].control is None and scores[0].ratio is None
    assert scores[0].hit is False
    summary = scoring.summarise(scores)
    assert summary["r"]["unscorable"] == 1 and summary["r"]["hits"] == 0


def test_ratio_is_measured_against_the_matched_control(profile, tmp_path):
    injected = _changes_after_injection(40, 200)
    control = _changes_after_injection(40, 60)  # the same clip untouched still drifts
    _run(tmp_path / "inj", "left", injected, _steady())
    _run(tmp_path / "ctl", "left", control, _steady(), mode="control", recipe="control")
    scores = scoring.score_directory(profile, tmp_path)
    assert len(scores) == 1  # the control is the yardstick, not a result
    s = scores[0]
    assert s.control is not None and s.ratio == pytest.approx(s.post / s.control, rel=1e-6)
    assert s.ratio > scoring.RATIO_THRESHOLD and s.hit


def test_controls_are_matched_on_host_phase_instant_and_arm(profile, tmp_path):
    """A control from a different host or instant is not a yardstick for this run."""
    _run(tmp_path / "inj", "left", _changes_after_injection(), _steady(), episode=1, t_inject=6.0)
    _run(
        tmp_path / "ctl_elsewhere",
        "left",
        _steady(),
        _steady(),
        mode="control",
        recipe="control",
        episode=2,
        t_inject=6.0,
    )
    _run(
        tmp_path / "ctl_later", "left", _steady(), _steady(), mode="control", recipe="control", episode=1, t_inject=9.0
    )
    scores = scoring.score_directory(profile, tmp_path)
    assert scores[0].control is None


def test_an_injection_at_the_very_end_is_not_scored(profile, tmp_path):
    """With no frames after the injection there is nothing to measure, and reporting zero would
    read as a failure that did not happen."""
    _run(tmp_path / "late", "left", _steady(n=32), _steady(n=32), failure_time_s=6.4)
    assert scoring.score_run(profile, tmp_path / "late") is None
