# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DROID-Merged LeRobot dataset used by the action FD DROID recipe."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    _ACTION_GRIPPER_FEATURE,
    _GRIPPER_STATE_FEATURE,
    _JOINT_ACTION_FEATURE,
    _JOINT_STATE_FEATURE,
    _LIST_COLUMNS,
    _STATE_FEATURE,
    DROIDLeRobotDataset,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id

_MERGED_ROOT_SPLITS = ("success", "failure")
_ROW_ROOT_KEY = "_cosmos_root_idx"


class DROIDMergedLeRobotDataset(DROIDLeRobotDataset):
    """DROID-Merged dataset with merged ``success`` + ``failure`` root support.

    This is intentionally separate from ``DROIDLeRobotDataset`` so the released
    DROID policy recipe keeps its single-root behavior unchanged.
    """

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "policy",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: str = "concat_view",
        action_space: str = "ee_pose",
        use_state: bool = False,
        action_normalization: str | None = None,
        use_image_augmentation: bool = False,
        use_filter_dict: bool = False,
        filter_dict_path: str | None = None,
        split: str = "train",
        use_success_only: bool = False,
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("DROIDMergedLeRobotDataset only supports concat_view.")
        if action_space not in ("ee_pose", "joint_pos"):
            raise NotImplementedError("action_space must be 'ee_pose' or 'joint_pos'.")
        if use_state and action_space != "joint_pos":
            raise NotImplementedError("use_state is only supported with action_space='joint_pos'.")
        if use_filter_dict and not filter_dict_path:
            raise ValueError("use_filter_dict=True requires filter_dict_path")
        if split != "train":
            raise NotImplementedError("DROIDMergedLeRobotDataset currently supports split='train' only.")
        if pose_convention != "backward_framewise":
            raise NotImplementedError("DROIDMergedLeRobotDataset only supports backward_framewise pose deltas.")

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = 1
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = "droid_lerobot"
        self._domain_id = get_domain_id(self._domain_name)
        self._action_normalization = None if action_space == "joint_pos" else action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._root = Path(root)
        self._roots = self._resolve_lerobot_roots(self._root, use_success_only=use_success_only)
        self._infos = [json.loads((root_path / "meta" / "info.json").read_text()) for root_path in self._roots]
        self._action_space = action_space
        self._use_state = bool(use_state)
        self._use_image_augmentation = bool(use_image_augmentation)
        self._image_augmentor = None
        self._use_filter_dict = bool(use_filter_dict)
        self._filter_dict_path = filter_dict_path
        self._episodes, self._tasks, episode_maps = self._load_metadata(self._roots)

        if action_space == "joint_pos":
            feature_cols = [_JOINT_ACTION_FEATURE, _ACTION_GRIPPER_FEATURE, _JOINT_STATE_FEATURE, _GRIPPER_STATE_FEATURE]
        else:
            feature_cols = [_STATE_FEATURE, _ACTION_GRIPPER_FEATURE]
        columns = ["index", "episode_index", "task_index", "timestamp", *feature_cols]
        index_parts, episode_parts, root_parts, task_parts, ts_parts = [], [], [], [], []
        feature_parts: dict[str, list] = {c: [] for c in feature_cols}
        for root_idx, root_path in enumerate(self._roots):
            for path in sorted((root_path / "data").glob("chunk-*/file-*.parquet")):
                table = pq.read_table(path, columns=columns)
                local_episode = table["episode_index"].to_numpy().astype(np.int64)
                global_episode = np.asarray(
                    [episode_maps[root_idx][int(episode_idx)] for episode_idx in local_episode],
                    dtype=np.int64,
                )
                index_parts.append(table["index"].to_numpy())
                episode_parts.append(global_episode)
                root_parts.append(np.full(global_episode.shape, root_idx, dtype=np.int64))
                task_parts.append(table["task_index"].to_numpy())
                ts_parts.append(table["timestamp"].to_numpy())
                for c in feature_cols:
                    if c in _LIST_COLUMNS:
                        feature_parts[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
                    else:
                        feature_parts[c].append(np.asarray(table[c].to_numpy(), dtype=np.float32))
        if not index_parts:
            roots = ", ".join(str(root_path) for root_path in self._roots)
            raise FileNotFoundError(f"No DROID-Merged parquet shards found under: {roots}")

        episode_all = np.concatenate(episode_parts).astype(np.int64)
        index_all = np.concatenate(index_parts).astype(np.int64)
        order = np.lexsort((index_all, episode_all))
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_root = np.concatenate(root_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._feat = {
            c: np.concatenate(feature_parts[c], axis=0).astype(np.float32)[order] for c in feature_cols
        }

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

        if self._use_filter_dict:
            with open(self._filter_dict_path) as f:
                filter_dict = json.load(f)
            seg_ep_pos, seg_win_start, seg_len = [], [], []
            for pos in range(len(self._ep_vals)):
                valid = int(max(0, ep_counts[pos] - self._chunk_length))
                if valid <= 0:
                    continue
                ep_id = str(self._episodes[int(self._ep_vals[pos])]["episode_id"])
                key = (
                    f"gs://xembodiment_data/r2d2/r2d2-data-full/{ep_id}/recordings/"
                    f"MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{ep_id}/trajectory.h5"
                )
                ranges = filter_dict.get(key)
                if ranges is None:
                    continue
                for s, e in ranges:
                    ws = max(int(s), 0)
                    we = min(int(e) - self._chunk_length, valid)
                    if we - ws > 0:
                        seg_ep_pos.append(pos)
                        seg_win_start.append(ws)
                        seg_len.append(we - ws)
            self._seg_ep_pos = np.asarray(seg_ep_pos, dtype=np.int64)
            self._seg_win_start = np.asarray(seg_win_start, dtype=np.int64)
            self._seg_cum = np.cumsum(seg_len).astype(np.int64) if seg_len else np.zeros(0, dtype=np.int64)

    @staticmethod
    def _is_lerobot_root(path: Path) -> bool:
        return (path / "meta" / "info.json").is_file()

    @classmethod
    def _resolve_lerobot_roots(cls, root: Path, *, use_success_only: bool) -> list[Path]:
        if cls._is_lerobot_root(root):
            return [root]

        roots: list[Path] = []
        split_names = ("success",) if use_success_only else _MERGED_ROOT_SPLITS
        for split_name in split_names:
            split_root = root / split_name
            if cls._is_lerobot_root(split_root):
                roots.append(split_root)
                continue
            if split_root.is_dir():
                roots.extend(sorted(path for path in split_root.iterdir() if cls._is_lerobot_root(path)))

        if roots:
            return roots
        raise FileNotFoundError(
            f"{root} is not a LeRobot root and no merged DROID split roots were found under "
            f"{', '.join(split_names)}."
        )

    @staticmethod
    def _load_metadata(roots: list[Path]) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], str], list[dict[int, int]]]:
        episodes: dict[int, dict[str, Any]] = {}
        tasks: dict[tuple[int, int], str] = {}
        episode_maps: list[dict[int, int]] = []
        next_episode_idx = 0

        for root_idx, root_path in enumerate(roots):
            episode_map: dict[int, int] = {}
            for path in sorted((root_path / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
                for row in pq.read_table(path).to_pylist():
                    local_episode_idx = int(row["episode_index"])
                    if local_episode_idx in episode_map:
                        continue
                    global_episode_idx = next_episode_idx
                    next_episode_idx += 1
                    episode_map[local_episode_idx] = global_episode_idx
                    row = dict(row)
                    row[_ROW_ROOT_KEY] = root_idx
                    episodes[global_episode_idx] = row
            if not episode_map:
                raise FileNotFoundError(f"No episode metadata found under {root_path / 'meta' / 'episodes'}")
            episode_maps.append(episode_map)

            tasks_df = pd.read_parquet(root_path / "meta" / "tasks.parquet")
            task_texts = tasks_df["task"] if "task" in tasks_df.columns else tasks_df.index
            for task, task_index in zip(task_texts, tasks_df["task_index"]):
                tasks[(root_idx, int(task_index))] = str(task)

        return episodes, tasks, episode_maps

    def _window_rows(self, start: int, stop: int, episode_index: int) -> list[dict[str, Any]]:
        return [
            {
                "episode_index": episode_index,
                "root_idx": int(self._row_root[j]),
                "task_index": int(self._row_task[j]),
                "timestamp": float(self._row_timestamp[j]),
                **{c: self._feat[c][j] for c in self._feat},
            }
            for j in range(start, stop)
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        if self._use_filter_dict:
            seg = int(np.searchsorted(self._seg_cum, idx, side="right"))
            base = int(self._seg_cum[seg - 1]) if seg > 0 else 0
            ep = int(self._seg_ep_pos[seg])
            start = int(self._ep_starts[ep]) + int(self._seg_win_start[seg]) + (idx - base)
        else:
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
        task = self._tasks[(int(observation_rows[0]["root_idx"]), int(observation_rows[0]["task_index"]))]
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

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        root_idx = int(episode.get(_ROW_ROOT_KEY, 0))
        info = self._infos[root_idx]
        root = self._roots[root_idx]
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
        rel = info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return root / rel
