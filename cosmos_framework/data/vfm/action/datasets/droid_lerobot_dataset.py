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
from cosmos_framework.data.vfm.action.action_spec import Gripper, Joint, Pos, Rot, build_action_spec
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
# joint_pos (8D = 7 arm joints + gripper) features, matching the internal
# DROIDLeRobotDataset(action_space="joint_pos", use_state=...). These are
# absolute joint commands/states (no normalization is applied for joint_pos,
# matching the internal canonical run which leaves action_normalization=None).
_JOINT_ACTION_FEATURE = "action.joint_position"          # [7] commanded joints
_ACTION_GRIPPER_FEATURE = "action.gripper_position"      # [1] commanded gripper
_JOINT_STATE_FEATURE = "observation.state.joint_positions"   # [7] observed joints
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"  # [1] observed gripper
# Columns whose parquet dtype is a list<float> (need to_pylist -> stacked array).
_LIST_COLUMNS = {_STATE_FEATURE, _JOINT_ACTION_FEATURE, _JOINT_STATE_FEATURE}
_ACTION_SPACES = ("ee_pose", "joint_pos")

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production DROID wrapper conversion from Franka panda_link8 to OpenCV.
_DROID_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH = Path(__file__).parent / "droid_lerobot_normalization.json"
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


class DROIDLeRobotDataset(Dataset):
    """DROID Action dataset.

    Two action layouts:
      * ``action_space="ee_pose"`` (default): 10D ``[pos_delta(3), rot6d_delta(6),
        gripper(1)]``, quantile-normalized (the v1.2 midtrain default).
      * ``action_space="joint_pos"``: 8D ``[joint(7), gripper(1)]`` absolute joint
        commands, NOT normalized, with ``use_state=True`` prepending the initial
        observed joint+gripper state → ``(chunk+1, 8)`` — matching the internal
        ``Cosmos3-Nano-Policy-DROID`` post-training run.
    Filter dictionaries, temporal-segment validation, and image augmentation from
    the production wrapper are intentionally omitted.
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
        action_space: str = "ee_pose",
        use_state: bool = False,
        action_normalization: str | None = "quantile",
    ) -> None:
        super().__init__()
        if pose_convention != "backward_framewise":
            raise NotImplementedError("This minimal DROID dataset only supports backward_framewise pose deltas.")
        if viewpoint != "concat_view":
            raise NotImplementedError("This minimal DROID dataset only supports concat_view.")
        if action_space not in _ACTION_SPACES:
            raise NotImplementedError(f"action_space must be one of {_ACTION_SPACES}, got {action_space!r}.")
        if use_state and action_space != "joint_pos":
            raise NotImplementedError("use_state is only supported with action_space='joint_pos'.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._action_space = action_space
        self._use_state = bool(use_state)
        # joint_pos trains on raw 8D joint values (the internal canonical run
        # leaves action_normalization=None); ee_pose keeps quantile normalization.
        self._action_normalization = None if action_space == "joint_pos" else action_normalization
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
        # Compact, lazy frame index. Materializing every frame as a Python dict
        # (``sorted(... pq.read_table(path).to_pylist() ...)``) does not scale:
        # the full DROID success shard is ~18M frames, which is tens of GB of
        # dicts plus an 18M-element Python sort at construction, and each
        # DataLoader worker faults in its own copy. Instead we read only the
        # columns the sample builder needs into contiguous numpy arrays
        # (~1 GB total) -- read-only after init, so worker forks share them
        # copy-on-write.
        if action_space == "joint_pos":
            feature_cols = [_JOINT_ACTION_FEATURE, _ACTION_GRIPPER_FEATURE, _JOINT_STATE_FEATURE, _GRIPPER_STATE_FEATURE]
        else:
            feature_cols = [_STATE_FEATURE, _ACTION_GRIPPER_FEATURE]
        columns = ["index", "episode_index", "task_index", "timestamp", *feature_cols]
        index_parts, episode_parts, task_parts, ts_parts = [], [], [], []
        feature_parts: dict[str, list] = {c: [] for c in feature_cols}
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=columns)
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            for c in feature_cols:
                if c in _LIST_COLUMNS:
                    feature_parts[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
                else:
                    feature_parts[c].append(np.asarray(table[c].to_numpy(), dtype=np.float32))
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        # Per-feature arrays keyed by parquet column name (read-only after init).
        self._feat = {
            c: np.concatenate(feature_parts[c], axis=0).astype(np.float32)[order] for c in feature_cols
        }

        # Group frames into episodes and keep only within-episode chunk windows.
        # The global frame index is ordered by episode in LeRobot v3, so episodes
        # are contiguous blocks once sorted by ``index``. The previous code sliced
        # the flat row list (``rows[idx : idx + chunk + 1]``) with no boundary
        # guard, so ~one chunk of samples per episode silently mixed two episodes;
        # restricting to in-episode windows yields ``total - n_episodes * chunk``
        # valid samples (matching the production dataset).
        assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

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
        return 8 if self._action_space == "joint_pos" else 10

    def _action_spec(self):
        if self._action_space == "joint_pos":
            return build_action_spec(Joint(n=7, label="joint"), Gripper())
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @property
    def action_names(self) -> list[str]:
        return self._action_spec().names

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _window_rows(self, start: int, stop: int, episode_index: int) -> list[dict[str, Any]]:
        """Reconstruct the per-frame dicts the sample builder consumes for the
        half-open frame window ``[start, stop)`` from the compact column arrays.
        ``start``/``stop`` are guaranteed to lie within a single episode."""
        return [
            {
                "episode_index": episode_index,
                "task_index": int(self._row_task[j]),
                "timestamp": float(self._row_timestamp[j]),
                **{c: self._feat[c][j] for c in self._feat},
            }
            for j in range(start, stop)
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        # Map the flat sample index to a within-episode frame window.
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        observation_rows = self._window_rows(start, start + self._chunk_length + 1, episode_index)

        video = self._load_concat_video(episode, observation_rows)
        if self._action_space == "joint_pos":
            raw_action = self._build_joint_action(observation_rows)
            extras: dict[str, Any] = {}
        else:
            action_rows = observation_rows[: self._chunk_length]
            raw_action, initial_pose = self._build_raw_action(observation_rows, action_rows)
            extras = {"initial_pose": initial_pose}
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice(task.split(" | "))

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            additional_view_description=(
                "The top row is from the wrist-mounted camera. "
                "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
            ),
            **extras,
        )

    def _build_joint_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """8D joint-position action ``[joint(7), gripper(1)]`` over the chunk, matching
        the internal ``action_space='joint_pos'``. The window is ``chunk+1`` frames:
        ``row[0]`` is the initial observed state (prepended when ``use_state``), and
        ``rows[1:]`` are the ``chunk`` commanded actions. Gripper is flipped (1 - g).
        No normalization is applied (internal canonical run uses raw joint values)."""
        action_rows = observation_rows[1:]
        joints = np.asarray([r[_JOINT_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 7]
        gripper = np.asarray([r[_ACTION_GRIPPER_FEATURE] for r in action_rows], dtype=np.float32).reshape(-1, 1)
        gripper = 1.0 - gripper
        action = np.concatenate([joints, gripper], axis=-1)  # [chunk, 8]
        if self._use_state:
            init = observation_rows[0]
            init_joint = np.asarray(init[_JOINT_STATE_FEATURE], dtype=np.float32)  # [7]
            init_gripper = np.asarray([1.0 - float(init[_GRIPPER_STATE_FEATURE])], dtype=np.float32)  # [1]
            initial_state = np.concatenate([init_joint, init_gripper])[None, :]  # [1, 8]
            action = np.concatenate([initial_state, action], axis=0)  # [chunk + 1, 8]
        return torch.from_numpy(action).float()

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
        spec = self._action_spec()
        idle_frames = compute_idle_frames(
            action,
            spec,
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )
        if self._action_normalization is None:
            out_action = action
        else:
            out_action = normalize_action(action, self._action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": out_action,
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
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0
