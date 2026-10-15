# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""A two-arm profile matching the reference robot's layout, with no dataset behind it.

The numeric invariants are properties of the splice and the scorer, so they are tested without a
dataset or a model. That is the point of keeping those modules free of both.
"""

from __future__ import annotations

import numpy as np
import pytest
from failuregen.config import ArmSpec, CameraProfile, DatasetProfile, Profile, RobotProfile


@pytest.fixture
def profile() -> Profile:
    robot = RobotProfile(
        state_dim=14,
        arms={"left": ArmSpec(joints=(0, 6), jaw=6, free=0.098), "right": ArmSpec(joints=(7, 13), jaw=13, free=0.988)},
        hold_band=(0.15, 0.5),
        phase_arm={"lift_extract": "left", "handover": "left", "transport": "right"},
        jaw_range=(0.098, 0.988),
    )
    cameras = CameraProfile(front="front", wrist={"left": "wrist_left", "right": "wrist_right"}, front_fraction=2 / 3)
    dataset = DatasetProfile(
        roots={"X": __import__("pathlib").Path("/nonexistent")},
        annotations="a.json",
        source_fps=30,
        target_fps=5,
        chunk_length=16,
        clean_outcome="success_clean",
        failure_labels=("drop",),
    )
    return Profile(robot=robot, cameras=cameras, dataset=dataset)


@pytest.fixture
def host_states() -> np.ndarray:
    """A host holding the object in its left hand and moving slowly."""
    n = 33
    s = np.zeros((n, 14), dtype=np.float32)
    s[:, 0:6] = np.linspace(0.0, 0.3, n)[:, None] + np.arange(6) * 0.01
    s[:, 7:13] = 1.5 + np.linspace(0.0, 0.1, n)[:, None]
    s[:, 6] = 0.30  # left jaw closed on the object
    s[:, 13] = 0.988  # right jaw free
    return s


@pytest.fixture
def donor_states() -> np.ndarray:
    """A donor standing a metre away that opens its left jaw partway through: a slip."""
    n = 33
    s = np.zeros((n, 14), dtype=np.float32)
    s[:, 0:6] = 1.0 + np.linspace(0.0, 0.9, n)[:, None] - np.arange(6) * 0.02
    s[:, 7:13] = -0.5
    s[:, 6] = 0.32
    s[16:, 6] = 0.85  # aperture springs out of the holding band
    s[:, 13] = 0.988
    return s
