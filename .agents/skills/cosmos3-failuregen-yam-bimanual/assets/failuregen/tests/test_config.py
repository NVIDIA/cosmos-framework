# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Profile validation. These messages are the whole value of the module."""

from __future__ import annotations

import pytest
from failuregen.config import Profile

BASE = {
    "robot": {
        "state_dim": 14,
        "arms": {
            "left": {"joints": [0, 6], "jaw": 6, "free": 0.098},
            "right": {"joints": [7, 13], "jaw": 13, "free": 0.988},
        },
        "hold_band": [0.15, 0.5],
        "phase_arm": {"handover": "left"},
    },
    "cameras": {"front": "f", "wrist": {"left": "wl", "right": "wr"}},
    "dataset": {
        "roots": {"X": "/tmp/x"},
        "annotations": "a.json",
        "source_fps": 30,
        "target_fps": 5,
        "chunk_length": 16,
        "clean_outcome": "success_clean",
        "failure_labels": ["drop"],
    },
}


def _profile(**patch):
    import copy

    raw = copy.deepcopy(BASE)
    for path, value in patch.items():
        node = raw
        *head, tail = path.split(".")
        for key in head:
            node = node[key]
        node[tail] = value
    return Profile.from_dict(raw)


def test_the_reference_layout_loads():
    p = _profile()
    assert p.robot.jaw_columns == [6, 13]
    assert p.robot.joint_columns == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    assert p.dataset.stride == 6 and p.dataset.chunk_seconds == pytest.approx(3.2)
    assert p.robot.holding_arm("handover") == "left"
    assert p.robot.holding_arm("unlisted") == "left"  # first declared arm


def test_a_free_value_inside_the_hold_band_is_rejected():
    """Otherwise a released jaw still reads as holding and no slip is detectable anywhere."""
    with pytest.raises(ValueError, match="inside hold_band"):
        _profile(**{"robot.hold_band": [0.05, 0.5]})


def test_columns_outside_the_state_vector_are_rejected():
    with pytest.raises(ValueError, match="outside state_dim"):
        _profile(**{"robot.state_dim": 7})


def test_a_camera_or_phase_naming_an_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        _profile(**{"cameras.wrist": {"middle": "wm"}})
    with pytest.raises(ValueError, match="unknown arm"):
        _profile(**{"robot.phase_arm": {"handover": "third"}})


def test_a_target_rate_above_the_source_rate_is_rejected():
    p = _profile(**{"dataset.target_fps": 60})
    with pytest.raises(ValueError, match="exceeds source_fps"):
        _ = p.dataset.stride


def test_notebook_eef_profile_is_valid(tmp_path):
    from failuregen.yam_training.failure_types import profile_dict

    raw = profile_dict(
        tmp_path / "prepared",
        tmp_path / "dataset",
        tmp_path / "model",
        {"annotations": "failuregen_annotations.json"},
        [{"phase": "lift_extract", "score_arm": "left", "label": "pickup_slip"}],
    )
    p = Profile.from_dict(raw)
    assert p.model["action_encoding"] == "precomputed"
    assert p.model["action_dim"] == 20
    assert p.model["domain_name"] == "xdof_yam"
    assert p.cameras.front_fraction == 0.5
    assert set(p.cameras.wrist) == set(p.robot.arms)
