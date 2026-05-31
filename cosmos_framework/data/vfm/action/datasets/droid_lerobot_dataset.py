# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal DROID LeRobot dataset for Cosmos Action v1.2 defaults."""

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

_NORMALIZER_PATH = Path(__file__).parent / "droid_lerobot_normalization.json"
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


class DROIDLeRobotDataset(Dataset):
    """DROID Action dataset matching the v1.2 midtrain config default.

    The supported action layout is 10D ``[pos_delta(3), rot6d_delta(6), gripper(1)]``.
    Unsupported branches from the production wrapper, such as joint-space
    actions, filter dictionaries, temporal-segment validation, state prefixing,
    and image augmentation, are intentionally omitted.
    """

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
        super().__init__()
        if pose_convention != "backward_framewise":
            raise NotImplementedError("This minimal DROID dataset only supports backward_framewise pose deltas.")
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal DROID dataset only supports concat_view.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id("droid_lerobot")
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
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DROID_TO_OPENCV

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        gripper = np.asarray([row["action.gripper_position"] for row in action_rows], dtype=np.float32).reshape(-1, 1)
        gripper = 1.0 - gripper
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
        normalized_action = normalize_action(action, "quantile", self._load_norm_stats())
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

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is not None:
            return self._norm_stats
        self._norm_stats = {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(_NORMALIZER_PATH)).items()
        }
        return self._norm_stats

    def __len__(self) -> int:
        return max(0, len(self._rows) - self._chunk_length)
