# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from cosmos_framework.data.vfm.action.agibot_gear_fk import AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST
from cosmos_framework.data.vfm.action.agibot_gear_spec import (
    AGIBOT_GEAR_EXT_STATE_ROBOT_ORIENTATION_SLICE,
    AGIBOT_GEAR_EXT_STATE_ROBOT_POSITION_SLICE,
    AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD,
    get_agibot_gear_urdf_path,
)

_REQUIRES_MUJOCO = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="requires mujoco until CI docker images include it",
)


def test_get_agibot_gear_urdf_path_exists() -> None:
    assert get_agibot_gear_urdf_path().name == "G1_omnipicker_calibrated.urdf"
    assert get_agibot_gear_urdf_path().parent.name == "urdf_visualizer"
    assert get_agibot_gear_urdf_path().is_file()


@pytest.mark.L0
def test_agibot_gripper_to_opencv_composes_extra_180deg_z_rotation() -> None:
    expected_left = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    expected_right = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST["left_wrist"], expected_left, atol=1e-6)
    np.testing.assert_allclose(AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST["right_wrist"], expected_right, atol=1e-6)


def test_gripper_open_fraction_conversions() -> None:
    from cosmos_framework.data.vfm.action.agibot_gear_fk import convert_gripper_state_to_open_fraction

    open_jitter_degrees = np.array([0.0, 0.21733333], dtype=np.float32)
    np.testing.assert_allclose(
        convert_gripper_state_to_open_fraction(open_jitter_degrees),
        [1.0, 0.9981889],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        convert_gripper_state_to_open_fraction(np.array([0.0], dtype=np.float32)),
        [1.0],
        atol=1e-6,
    )

    radians = np.array([-AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD, -AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD / 2.0, 0.0])
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(radians), [1.0, 0.5, 0.0], atol=1e-6)

    actuator_degrees = np.array([0.0, 60.0, 120.0], dtype=np.float32)
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(actuator_degrees), [1.0, 0.5, 0.0], atol=1e-6)


@pytest.mark.L0
def test_gripper_open_fraction_clips_small_actuator_overshoot() -> None:
    from cosmos_framework.data.vfm.action.agibot_gear_fk import convert_gripper_state_to_open_fraction

    actuator_degrees = np.array([122.1857], dtype=np.float32)  # [1]
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(actuator_degrees), [0.0], atol=1e-6)


@_REQUIRES_MUJOCO
def test_standard_body_head_layout_uses_state19_as_waist_lift() -> None:
    from cosmos_framework.data.vfm.action.agibot_gear_fk import compute_fk_transforms_batch

    states = np.zeros((2, 32), dtype=np.float32)  # [T,S]
    states[:, 17] = 0.25  # head pitch
    states[:, 18] = 0.5  # waist pitch
    states[:, 19] = 0.1  # waist lift
    states[1, 19] = 0.2

    fk = compute_fk_transforms_batch(states, "agibot_gear_gripper")

    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk[key][1, 2, 3] - fk[key][0, 2, 3], 0.1, atol=1e-6)


@_REQUIRES_MUJOCO
def test_ext_fk_uses_correct_state_indices() -> None:
    """Verify ext FK reads arm joints from state[54:68], not state[0:14]."""
    from cosmos_framework.data.vfm.action.agibot_gear_fk import compute_fk_transforms

    state_ext = np.zeros(94, dtype=np.float32)
    state_ext[54:61] = [0.1, -0.2, 0.3, -0.1, 0.2, 0.5, -0.3]  # left arm
    state_ext[61:68] = [-0.1, 0.2, -0.3, 0.1, -0.2, -0.5, 0.3]  # right arm
    state_ext[82] = 0.0  # head yaw
    state_ext[83] = 0.3  # head pitch
    state_ext[84] = 0.5  # waist pitch
    state_ext[85] = 0.35  # waist lift

    state_std = np.zeros(32, dtype=np.float32)
    state_std[0:7] = state_ext[54:61]  # left arm
    state_std[7:14] = state_ext[61:68]  # right arm
    state_std[16] = 0.0  # head yaw
    state_std[17] = 0.3  # head pitch
    state_std[18] = 0.5  # waist pitch
    state_std[19] = 0.35  # waist lift

    fk_ext = compute_fk_transforms(state_ext, "agibot_gear_gripper_ext")
    fk_std = compute_fk_transforms(state_std, "agibot_gear_gripper")

    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk_ext[key], fk_std[key], atol=1e-6, err_msg=f"FK mismatch for {key}")


@_REQUIRES_MUJOCO
def test_ext_fk_applies_robot_base_motion_to_batch_poses() -> None:
    """Ext FK folds state/robot pose into all head and wrist trajectories."""
    from cosmos_framework.data.vfm.action.agibot_gear_fk import compute_fk_transforms_batch

    states = np.zeros((2, 94), dtype=np.float32)  # [T,S]
    states[:, AGIBOT_GEAR_EXT_STATE_ROBOT_ORIENTATION_SLICE] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    states[1, AGIBOT_GEAR_EXT_STATE_ROBOT_POSITION_SLICE] = np.array([0.4, -0.2, 0.0], dtype=np.float32)

    fk = compute_fk_transforms_batch(states, "agibot_gear_gripper_ext")

    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk[key][1, :3, 3] - fk[key][0, :3, 3], [0.4, -0.2, 0.0], atol=1e-6)
