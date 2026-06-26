# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UMI LeRobot dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["wrist_view"]

# Default image key for wrist camera in UMI LeRobot datasets.
_IMAGE_FEATURE = "observation.images.camera0"
_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"

_NORMALIZER_PATH = Path(__file__).parent / "stats/umi_lerobot_stats.json"


class UMILeRobotDataset(ActionBaseDataset):
    """UMI dataset converted to LeRobot format with 10D cartesian actions:

        [pos_delta(3), rot6d_delta(6), gripper_width(1)]

    Expects a LeRobot v2 dataset with:
      * ``observation.images.image``: wrist-mounted RGB video (configurable via
        ``image_key``).
      * ``observation.state``: 7D EEF state ``[pos(3), rot_axisangle(3),
        gripper_width(1)]``.
      * ``action``: 7D commanded state in the same format.

    Absolute axis-angle EEF poses are converted to backward-framewise rot6d
    relative poses, and the gripper width is taken from the commanded action.
    """

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "wrist_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        image_key: str = _IMAGE_FEATURE,
    ) -> None:
        if viewpoint != "wrist_view":
            raise NotImplementedError("This UMI dataset only supports wrist_view.")
        super().__init__(
            root=root,
            domain_name="umi",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._image_key = image_key

    @property
    def action_dim(self) -> int:
        return 10

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        video = self._load_video(episode, observation_rows)
        raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
        )

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        return decode_video_frames(
            self._video_path(episode, self._image_key),
            [float(episode.get(f"videos/{self._image_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # State is 7D: [pos(3), rot_axisangle(3), gripper_width(1)]
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "axisangle")

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)

        # Gripper width from commanded action (7th column)
        gripper = np.asarray([row[_ACTION_FEATURE][6] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose
