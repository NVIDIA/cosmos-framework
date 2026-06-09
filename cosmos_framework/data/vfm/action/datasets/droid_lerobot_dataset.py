# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DROID LeRobot dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats
from cosmos_framework.data.vfm.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    compute_idle_frames,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]

_IMAGE_FEATURES = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
_STATE_FEATURE = "observation.state.cartesian_position"

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production DROID wrapper conversion from Franka panda_link8 to OpenCV.
_DROID_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent / "stats/droid_lerobot_stats.json"


class DROIDLeRobotDataset(ActionBaseDataset):
    """DROID dataset with 10D cartesian actions ``[pos_delta(3), rot6d_delta(6), gripper(1)]``.

    Joint-space actions, filter dictionaries, temporal-segment validation, state
    prefixing, and image augmentation from the production wrapper are omitted.
    """

    _normalization_method = "quantile_rot"

    def __init__(
        self,
        root: str = "/path/to/cosmos3_action_datasets/droid_plus_lerobot_640x360_20260412",
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal DROID dataset only supports concat_view.")
        super().__init__(
            root=root,
            domain_name="droid_lerobot",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
        )

    @property
    def action_dim(self) -> int:
        return 10

    @property
    def action_names(self) -> list[str]:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper()).names

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        """Return action normalization stats for this dataset as torch tensors."""
        return {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(_NORMALIZER_PATH)).items()
        }

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        return compute_idle_frames(
            action,
            build_action_spec(Pos(), Rot("rot6d"), Gripper()),
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        first_row = self._rows[idx]
        episode = self._episodes[int(first_row["episode_index"])]

        observation_rows = self._rows[idx : idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        video = self._load_concat_video(episode, observation_rows)
        raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice(task.split(" | "))

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
            additional_view_description=(
                "The top row is from the wrist-mounted camera. "
                "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
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

        wrist = frames_by_view["wrist"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]
        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist, bottom], dim=-2)

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DROID_TO_OPENCV

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        gripper = np.asarray([row["action.gripper_position"] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        gripper = 1.0 - gripper
        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose
