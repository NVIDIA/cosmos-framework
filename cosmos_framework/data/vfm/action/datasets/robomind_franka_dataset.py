# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND Franka LeRobot dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]

_IMAGE_FEATURES = {
    "front": "observation.images.camera_front",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
}
_STATE_FEATURE = "observation.states.end_effector"
_ACTION_FEATURE = "actions.joint_position"

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production RoboMIND Franka wrapper conversion to OpenCV coordinates.
_ROBOMIND_FRANKA_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent / "stats/robomind_franka_stats.json"


def _dual_arm_action_spec():
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


class RoboMINDFrankaDataset(ActionBaseDataset):
    """RoboMIND Franka dual-arm dataset with 20D cartesian actions::

        [left_pos_delta(3), left_rot6d_delta(6), left_gripper(1),
         right_pos_delta(3), right_rot6d_delta(6), right_gripper(1)]

    Single-arm shards, split/filter logic, image augmentation, fast
    initialization, and alternate viewpoints are omitted.
    """


    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        embodiment_type: str = "robomind-franka-dual",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        if embodiment_type != "robomind-franka-dual":
            raise NotImplementedError("This minimal RoboMIND dataset only supports robomind-franka-dual.")
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal RoboMIND dataset only supports concat_view.")
        self._embodiment_type = embodiment_type
        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )

    @property
    def action_dim(self) -> int:
        return 20

    def _action_spec(self) -> ActionSpec:
        return _dual_arm_action_spec()

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        first_row = self._rows[idx]
        episode = self._episodes[int(first_row["episode_index"])]

        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        video = self._load_concat_video(episode, observation_rows)
        raw_action, initial_pose_left, initial_pose_right = self._build_raw_action(observation_rows, action_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            initial_pose=initial_pose_left,
            initial_pose_right=initial_pose_right,
            additional_view_description=(
                "The top row shows a third-person perspective looking towards the dual-arm Franka robot from the front. "
                "The bottom-left view looks at the scene from the left side, and the bottom-right view looks at the scene from the right side."
            ),
        )

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames_by_view = {
            name: decode_video_frames(
                self._video_path(episode, video_key),
                [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
                self._tolerance_s,
            )
            for name, video_key in _IMAGE_FEATURES.items()
        }

        front = frames_by_view["front"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]
        _, _, h_front, w_front = front.shape
        half_h, half_w = h_front // 2, w_front // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([front, bottom], dim=-2)

    def _build_relative_poses(
        self,
        positions: np.ndarray,
        euler_xyz: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        poses_abs = build_abs_pose_from_components(positions, euler_xyz, "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _ROBOMIND_FRANKA_TO_OPENCV
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        return poses_rel, initial_pose

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        gripper = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)

        poses_rel_left, initial_pose_left = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        poses_rel_right, initial_pose_right = self._build_relative_poses(state[:, 6:9], state[:, 9:12])
        action = np.concatenate(
            [
                poses_rel_left[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [7]],
                poses_rel_right[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [15]],
            ],
            axis=-1,
        )
        return torch.from_numpy(action).float(), initial_pose_left, initial_pose_right
