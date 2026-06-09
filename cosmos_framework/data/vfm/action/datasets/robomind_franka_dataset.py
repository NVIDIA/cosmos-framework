# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal RoboMIND Franka LeRobot dataset for Cosmos Action examples.

This is a stripped-down counterpart of the internal RoboMIND Franka dataset
wrapper. It intentionally keeps only the pieces needed by the cookbook asset:
RoboMIND Franka dual-arm, concat-view video, backward-framewise rot6d actions,
    and quantile-rotation action normalization.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.vfm.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import (
    build_abs_pose_from_components,
    compute_idle_frames,
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

_NORMALIZER_PATH = Path(__file__).parent / "robomind_franka_normalization.json"
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


def _dual_arm_action_spec():
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


class RoboMINDFrankaDataset(Dataset):
    """RoboMIND Franka dual-arm dataset matching Cosmos Action v1.2 defaults.

    The supported action layout is 20D::

        [left_pos_delta(3), left_rot6d_delta(6), left_gripper(1),
         right_pos_delta(3), right_rot6d_delta(6), right_gripper(1)]

    Unsupported production features such as single-arm shards, split/filter
    logic, image augmentation, fast initialization, and alternate viewpoints are
    intentionally omitted to keep the cookbook dependency small and readable.
    """

    def __init__(
        self,
        root: str = (
            "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/"
            "cosmos3_action_datasets/RoboMIND_20251228"
        ),
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        embodiment_type: str = "robomind-franka-dual",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        sample_stride: int = 1,
    ) -> None:
        super().__init__()
        if embodiment_type != "robomind-franka-dual":
            raise NotImplementedError("This minimal RoboMIND dataset only supports robomind-franka-dual.")
        if pose_convention != "backward_framewise":
            raise NotImplementedError("This minimal RoboMIND dataset only supports backward_framewise pose deltas.")
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal RoboMIND dataset only supports concat_view.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._embodiment_type = embodiment_type
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id(embodiment_type)
        self._norm_stats: dict[str, torch.Tensor] | None = None

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        self._tasks = {
            int(row["task_index"]): str(row["task"])
            for row in pq.read_table(self._root / "meta" / "tasks.parquet").to_pylist()
        }
        self._rows = sorted(
            (
                row
                for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet"))
                for row in pq.read_table(path).to_pylist()
            ),
            key=lambda row: int(row["index"]),
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return 20

    @property
    def action_names(self) -> list[str]:
        return _dual_arm_action_spec().names

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

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

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return self._root / rel

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

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        spec = _dual_arm_action_spec()
        idle_frames = compute_idle_frames(
            action,
            spec,
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )
        normalized_action = normalize_action(action, "quantile_rot", self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        """Return action normalization stats for this dataset as torch tensors."""
        return {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(_NORMALIZER_PATH), stats_key="global_raw").items()
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = self.load_action_stats()
        return self._norm_stats

    def __len__(self) -> int:
        return max(0, (len(self._rows) - self._chunk_length + self._sample_stride - 1) // self._sample_stride)
