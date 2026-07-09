# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed DROID action dataset.

Drop-in for DROIDLeRobotDataset that reads everything from LanceDB — per-frame
labels from ``{table}_frames`` / ``{table}_tasks`` / ``{table}_episodes`` and the
pre-composed video from ``{table}`` (see tools/lance_datagen/build_composed_droid.py).
Inherits the base loader's indexing, pose math, and action assembly, so labels
stay bit-exact; only the H.264 re-encode of the video is lossy.
"""

from __future__ import annotations

import json
import random
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.permutation import Permutation
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import (
    _ACTION_GRIPPER_FEATURE,
    _GRIPPER_STATE_FEATURE,
    _JOINT_ACTION_FEATURE,
    _JOINT_STATE_FEATURE,
    _STATE_FEATURE,
    DROIDLeRobotDataset,
)
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

_ADDITIONAL_VIEW_DESC = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective "
    "views of the scene from opposite sides, with the robot visible."
)


def _resolve_device(device: str | None) -> torch.device | None:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if device in (None, "cpu"):
        return None
    return torch.device(device)


def _read_all(db, name: str, columns: list[str]) -> pa.RecordBatch:
    tbl = db.open_table(name)
    perm = Permutation.identity(tbl).select_columns(columns).with_format("arrow")
    return perm.__getitems__(list(range(tbl.count_rows())))


class LanceDROIDComposedDataset(DROIDLeRobotDataset):
    """Action loader reading labels + pre-composed episodes from LanceDB (no LeRobot tree).

    Decodes a single composed video stream per episode instead of 3 views.
    """

    def __init__(
        self,
        lance_uri: str,
        *,
        table: str = "droid_composed",
        fps: float = 15.0,
        chunk_length: int = 16,
        mode: str = "joint",
        viewpoint: str = "concat_view",
        action_space: str = "ee_pose",
        use_state: bool = False,
        action_normalization: str | None = "quantile",
        use_filter_dict: bool = False,
        filter_dict_path: str | None = None,
        decode_device: str | None = "cpu",
        decoder_cache_size: int = 32,
        storage_options: dict | None = None,
    ) -> None:
        # Same validations as the base loader (whose parquet-reading __init__ we bypass).
        if viewpoint != "concat_view":
            raise NotImplementedError("LanceDROIDComposedDataset only supports concat_view.")
        if action_space not in ("ee_pose", "joint_pos"):
            raise NotImplementedError(f"action_space must be 'ee_pose' or 'joint_pos', got {action_space!r}.")
        if use_state and action_space != "joint_pos":
            raise NotImplementedError("use_state is only supported with action_space='joint_pos'.")
        if use_filter_dict and not filter_dict_path:
            raise ValueError("use_filter_dict=True requires filter_dict_path")

        # Config attributes the inherited label/indexing code reads.
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = 1
        self._mode = mode
        self._pose_convention = "backward_framewise"
        self._viewpoint = viewpoint
        self._domain_name = "droid_lerobot"
        self._domain_id = get_domain_id(self._domain_name)
        self._action_normalization = None if action_space == "joint_pos" else action_normalization
        self._norm_stats = None
        self._rows = None  # base per-frame dict list is never built here
        self._action_space = action_space
        self._use_state = bool(use_state)
        self._use_image_augmentation = False
        self._image_augmentor = None
        self._use_filter_dict = bool(use_filter_dict)
        self._filter_dict_path = filter_dict_path

        # Labels: build the same compact arrays the base builds from parquet.
        db = lancedb.connect(lance_uri, storage_options=storage_options)
        feature_cols = (
            [_JOINT_ACTION_FEATURE, _ACTION_GRIPPER_FEATURE, _JOINT_STATE_FEATURE, _GRIPPER_STATE_FEATURE]
            if action_space == "joint_pos"
            else [_STATE_FEATURE, _ACTION_GRIPPER_FEATURE]
        )
        frames = _read_all(
            db,
            f"{table}_frames",
            ["episode_index", "task_index", "timestamp", *[c.replace(".", "__") for c in feature_cols]],
        )
        self._row_episode = frames.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)
        self._row_task = frames.column("task_index").to_numpy(zero_copy_only=False).astype(np.int64)
        self._row_timestamp = frames.column("timestamp").to_numpy(zero_copy_only=False).astype(np.float64)
        self._feat = {}
        for c in feature_cols:
            arr = frames.column(c.replace(".", "__"))
            if pa.types.is_fixed_size_list(arr.type):
                self._feat[c] = np.asarray(arr.values).reshape(len(arr), arr.type.list_size)
            else:
                self._feat[c] = arr.to_numpy(zero_copy_only=False)

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous in the frames table"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

        tasks = _read_all(db, f"{table}_tasks", ["task_index", "task"])
        self._tasks = dict(zip(tasks.column("task_index").to_pylist(), tasks.column("task").to_pylist()))
        eps = _read_all(db, f"{table}_episodes", ["episode_index", "episode_id"])
        self._episodes = {
            int(i): {"episode_index": int(i), "episode_id": s}
            for i, s in zip(eps.column("episode_index").to_pylist(), eps.column("episode_id").to_pylist())
        }

        # Keep-ranges window filter — same construction as the base loader.
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

        # Video: lazy per-worker handles into the composed table.
        self._lance_uri = lance_uri
        self._table = table
        self._decode_device = _resolve_device(decode_device)
        self._cache_size = decoder_cache_size
        self._storage_options = storage_options
        self._perm = None
        self._ep_row: dict[int, int] | None = None
        self._decoders: dict[int, VideoDecoder] | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_perm", "_ep_row", "_decoders"):
            state[k] = None
        return state

    def _ensure_open(self) -> None:
        if self._decoders is not None:
            return
        tbl = lancedb.connect(self._lance_uri, storage_options=self._storage_options).open_table(self._table)
        ep = Permutation.identity(tbl).select_columns(["episode_index"]).with_format("arrow")
        rows = ep.__getitems__(list(range(tbl.count_rows())))
        self._ep_row = {int(rows.column("episode_index")[i].as_py()): i for i in range(rows.num_rows)}
        self._perm = Permutation.identity(tbl).select_columns(["video_bytes"]).with_format("arrow")
        self._decoders = {}

    def _read_clip_bytes(self, rows: list[int]) -> dict[int, bytes]:
        # Plain large_binary via the Permutation API. TODO: move to blob-v2 after optimizations.
        # take returns rows sorted by offset, so key by row instead of relying on order.
        rows = sorted({int(r) for r in rows})
        col = self._perm.__getitems__(rows).column("video_bytes")
        return {r: col[i].as_py() for i, r in enumerate(rows)}

    def _build_decoder(self, data: bytes) -> VideoDecoder:
        device = str(self._decode_device) if self._decode_device else None
        return VideoDecoder(data, seek_mode="approximate", device=device)

    def _ensure_decoders(self, ep_indices: list[int]) -> None:
        needed = list(dict.fromkeys(ep_indices))
        needed_set = set(needed)
        missing = [e for e in needed if e not in self._decoders]
        if not missing:
            return
        clips = self._read_clip_bytes([self._ep_row[e] for e in missing])
        for e in missing:
            while len(self._decoders) >= self._cache_size:
                victim = next((k for k in self._decoders if k not in needed_set), None)
                if victim is None:
                    break
                self._decoders.pop(victim)
            self._decoders[e] = self._build_decoder(clips[self._ep_row[e]])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self._ensure_open()
        n = len(indices)
        specs, plan = [], {}
        for sp, idx in enumerate(indices):
            idx = int(idx)
            mode = self._choose_mode()
            ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
            prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
            offset = idx - prev
            start = int(self._ep_starts[ep]) + offset
            ep_index = int(self._ep_vals[ep])
            obs = self._window_rows(start, start + self._chunk_length + 1, ep_index)
            if self._action_space == "joint_pos":
                action = self._build_joint_action(obs)
                extras = {}
            else:
                action, initial_pose = self._build_raw_action(obs, obs[: self._chunk_length])
                extras = {"initial_pose": initial_pose}
            task = self._tasks[int(obs[0]["task_index"])]
            specs.append(
                {"mode": mode, "action": action, "extras": extras, "ai_caption": random.choice(task.split(" | "))}
            )
            clip_idx = [offset + k for k in range(self._chunk_length + 1)]
            e = plan.setdefault(ep_index, {"frames": [], "owners": []})
            lo = len(e["frames"])
            e["frames"].extend(clip_idx)
            e["owners"].append((sp, lo, lo + len(clip_idx)))

        self._ensure_decoders(list(plan.keys()))
        decoded: list[torch.Tensor | None] = [None] * n
        for ep_index, e in plan.items():
            dec = self._decoders[ep_index]
            frames = dec.get_frames_at(indices=e["frames"]).data
            for sp, lo, hi in e["owners"]:
                decoded[sp] = frames[lo:hi].to(torch.float32) / 255.0

        results = []
        for sp in range(n):
            s = specs[sp]
            results.append(
                self._build_result(
                    mode=s["mode"],
                    video=decoded[sp],
                    action=s["action"],
                    ai_caption=s["ai_caption"],
                    additional_view_description=_ADDITIONAL_VIEW_DESC,
                    **s["extras"],
                )
            )
        return results


class LanceDROIDComposedIterable(torch.utils.data.IterableDataset):
    """Streams windows from LanceDROIDComposedDataset with episode-level shuffling."""

    def __init__(self, composed: LanceDROIDComposedDataset, seed: int = 42):
        super().__init__()
        self._ds = composed
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self):
        blocks = self._ds.get_shuffle_blocks()
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        shard = int(self.shard_rank) * nw + wid
        total = max(1, int(self.shard_world_size) * nw)
        epoch = 0
        while True:
            g = torch.Generator().manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[shard::total]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._ds[idx]
            epoch += 1


def get_lance_action_droid_sft_dataset(
    *,
    lance_uri: str,
    table: str = "droid_composed",
    decode_device: str | None = "cpu",
    storage_options: dict | None = None,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "policy",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: Any = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
):
    """Lance drop-in for ``get_action_droid_sft_dataset``: same DROID action SFT
    stack (``ActionTransformPipeline`` + ``ActionSFTDataset``), reading labels and
    pre-composed episodes from LanceDB instead of the raw LeRobot tree."""
    dataset = LanceDROIDComposedDataset(
        lance_uri,
        table=table,
        decode_device=decode_device,
        storage_options=storage_options,
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed) if iterable_shuffle else sft


__all__ = [
    "LanceDROIDComposedDataset",
    "LanceDROIDComposedIterable",
    "get_lance_action_droid_sft_dataset",
]
