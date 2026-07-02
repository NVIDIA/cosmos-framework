# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from cosmos_framework.data.vfm.action.datasets.human_hand_pose_lerobot_dataset import (
    HumanHandPoseLeRobotDataset,
)


def _row(frame_index: int) -> dict:
    positions = np.zeros((21, 3), dtype=np.float32)
    positions[[4, 8, 12, 16, 20], 2] = np.arange(1, 6, dtype=np.float32) * 0.01
    rotations = np.zeros((21, 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    return {
        "observation.state.camera_position": np.zeros(3, dtype=np.float32),
        "observation.state.camera_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "observation.state.hand_right_cam": positions.reshape(-1),
        "observation.state.hand_right_cam_rotation": rotations.reshape(-1),
        "observation.state.hand_left_cam": positions.reshape(-1),
        "observation.state.hand_left_cam_rotation": rotations.reshape(-1),
        "frame_index": frame_index,
    }


def test_static_hand_pose_builds_57d_action() -> None:
    dataset = object.__new__(HumanHandPoseLeRobotDataset)
    dataset._chunk_length = 2
    dataset._pose_convention = "backward_framewise"

    action = dataset._build_raw_action([_row(0), _row(1), _row(2)]).numpy()

    assert action.shape == (2, 57)
    assert np.isfinite(action).all()
    np.testing.assert_allclose(action[:, :3], 0.0)
    np.testing.assert_allclose(action[:, 9:12], 0.0)
    np.testing.assert_allclose(action[:, 33:36], 0.0)


def test_action_spec_matches_released_width() -> None:
    dataset = object.__new__(HumanHandPoseLeRobotDataset)

    assert dataset.action_dim == 57
    assert dataset._action_spec().dim == dataset.action_dim
