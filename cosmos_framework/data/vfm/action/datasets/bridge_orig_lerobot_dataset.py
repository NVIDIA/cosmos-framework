# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal Bridge Orig LeRobot dataset for Cosmos Action examples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
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
Viewpoint = Literal["ego_view"]

_IMAGE_FEATURE = "observation.images.image_0"
_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"

# Raw Bridge state -> kinematics frame. The WidowX controller records
# R_state = R_fk @ DEFAULT_ROTATION.T, so R_fk = R_state @ DEFAULT_ROTATION.
_DEFAULT_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float32,
)

# Kinematics frame -> OpenCV frame used by Cosmos action training.
_BRIDGE_TO_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

# Re-reference from ee_gripper_link to gripper_link in the kinematics frame.
_TCP_TO_FLANGE = np.array(
    [
        [1.0, 0.0, 0.0, -0.093575],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent / "bridge_orig_lerobot_normalization.json"
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


class BridgeOrigLeRobotDataset(Dataset):
    """Bridge Orig LeRobot dataset matching Cosmos Action v1.2 defaults.

    Supported action layout is 10D:

        [pos_delta(3), rot6d_delta(6), gripper(1)]

    This stripped wrapper keeps only the local LeRobot asset path used by the
    action cookbook: single `image_0` ego-view video, backward-framewise rot6d
    actions, and quantile-rotation normalization.
    """

    def __init__(
        self,
        root: str = "/path/to/cosmos3_action_datasets/bridge_raw",
        fps: float = 5.0,
        chunk_length: int = 16,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "ego_view",
        sample_stride: int = 1,
    ) -> None:
        super().__init__()
        if pose_convention != "backward_framewise":
            raise NotImplementedError("This minimal Bridge dataset only supports backward_framewise pose deltas.")
        if viewpoint != "ego_view":
            raise NotImplementedError("This minimal Bridge dataset only supports ego_view.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id("bridge_orig_lerobot")
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
        return 10

    @property
    def action_names(self) -> list[str]:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper()).names

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
            self._video_path(episode, _IMAGE_FEATURE),
            [float(episode.get(f"videos/{_IMAGE_FEATURE}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

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

    def _build_raw_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")

        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DEFAULT_ROTATION.astype(poses_abs.dtype)
        poses_abs = poses_abs @ _TCP_TO_FLANGE.astype(poses_abs.dtype)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _BRIDGE_TO_OPENCV.astype(poses_abs.dtype)

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        gripper = np.asarray([row[_ACTION_FEATURE][6] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        action = np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
        return torch.from_numpy(action).float(), initial_pose

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        spec = build_action_spec(Pos(), Rot("rot6d"), Gripper())
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
