# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed DROID action dataset.

Drop-in for DROIDLeRobotDataset that reads everything from LanceDB — per-frame
labels from ``{table}_frames`` / ``{table}_tasks`` / ``{table}_episodes`` and the
pre-composed video from ``{table}`` (see tools/lance_datagen/build_composed_droid.py).
The split/span index is built with the base's own helpers and the sample dict is
assembled to match what the lazy LeRobot readers return, so the inherited
``__getitem__`` (pose math, gripper handling, action assembly) runs unchanged —
labels stay bit-exact; only the H.264 re-encode of the video is lossy.
"""

from __future__ import annotations

import json
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.permutation import Permutation
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.generator.action.action_processing import resolve_action_normalization
from cosmos_framework.data.generator.action.datasets import droid_lerobot_dataset_config as _cfg
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    _normalize_split,
    build_episode_spans,
    split_episode_ids,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import (
    _DROID_TO_OPENCV,
    DROIDLeRobotDataset,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

_DEFAULT_VERSION = "droid_plus_lerobot_320x180_20260406"


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
        version: str = _DEFAULT_VERSION,
        fps: float = 15.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "policy",
        viewpoint: str = "concat_view",
        action_space: str = "midtrain",
        use_state: bool = False,
        action_normalization: str | None = None,
        use_filter_dict: bool = False,
        filter_dict_path: str | None = None,
        sample_stride: int = 1,
        decode_device: str | None = "cpu",
        decoder_cache_size: int = 32,
        storage_options: dict | None = None,
    ) -> None:
        # Same argument surface as the base loader, minus what only applies to
        # raw-LeRobot reading (root/video_mode/history/augmentation/temp-seg).
        if viewpoint != "concat_view":
            raise NotImplementedError("LanceDROIDComposedDataset only supports concat_view.")
        if use_state and action_space != "joint_pos":
            raise NotImplementedError("use_state is only supported with action_space='joint_pos'.")
        if use_filter_dict and not filter_dict_path:
            raise ValueError("use_filter_dict=True requires filter_dict_path")
        if split.lower() == "val_temp_seg":
            raise NotImplementedError("val_temp_seg is not supported by the Lance loader.")

        # -- config attributes the inherited code reads (base + DROID init, sans LeRobot IO) --
        self._memprofile = False
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._split_seed = split_seed
        self._split_val_ratio = split_val_ratio
        self._split = _normalize_split(split)
        self._mode = mode
        self._embodiment_type = "droid_lerobot"
        self._viewpoint = viewpoint
        self._pose_convention = "backward_framewise"
        self._rotation_format = "rot6d"
        self._action_normalizer = None
        if action_normalization is not None:
            self._action_normalizer = resolve_action_normalization(
                action_normalization, self._load_norm_stats(action_normalization)
            )
        self._tolerance_s = 2e-4
        self._skip_video_loading = False
        self._sample_stride = int(sample_stride)
        self._min_episode_length_frames = None
        self._domain_id = get_domain_id(self._embodiment_type)
        self._to_opencv = _DROID_TO_OPENCV

        self._use_success_only = True  # subset selection happened at convert time
        self._video_mode = None
        self._action_space = action_space
        self._use_state = use_state
        self._use_filter_dict = use_filter_dict
        self._filter_dict_path = filter_dict_path
        self._max_num_history_actions = 0
        self._use_image_augmentation = False
        self._image_augmentor = None
        self._is_val_temp_seg = False

        self._image_features = _cfg.IMAGE_FEATURES[version]
        self._state_features = _cfg.STATE_FEATURES[version]
        self._action_features = _cfg.ACTION_FEATURES[version]
        self._is_flat_action = _cfg.IS_FLAT_ACTION[version]
        self._has_multi_language_annotations = _cfg.HAS_MULTI_LANGUAGE_ANNOTATIONS[version]
        self._is_gripper_action_flipped = _cfg.IS_GRIPPER_ACTION_FLIPPED[version]

        # Label-window plan: feature -> window length, mirroring the base's
        # delta_timestamps ([0..k]*dt lists; our frames are contiguous per episode,
        # so a window is a plain row slice of that length).
        obs_len, act_len = self._chunk_length + 1, self._chunk_length
        self._label_windows: dict[str, int] = {
            self._state_features: obs_len,
            self._action_features: act_len,
        }
        if action_space == "joint_pos":
            self._label_windows[_cfg._JOINT_ACTION_FEATURE] = act_len
            if use_state:
                self._label_windows[_cfg._JOINT_STATE_FEATURE] = obs_len
                self._label_windows[_cfg._GRIPPER_STATE_FEATURE] = obs_len

        # -- labels from Lance: same per-frame arrays the LeRobot parquets hold --
        db = lancedb.connect(lance_uri, storage_options=storage_options)
        frames = _read_all(
            db,
            f"{table}_frames",
            ["episode_index", "task_index", *[c.replace(".", "__") for c in self._label_windows]],
        )
        self._row_task = frames.column("task_index").to_numpy(zero_copy_only=False).astype(np.int64)
        row_episode = frames.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)
        self._feat: dict[str, np.ndarray] = {}
        for c in self._label_windows:
            arr = frames.column(c.replace(".", "__"))
            if pa.types.is_fixed_size_list(arr.type):
                self._feat[c] = np.asarray(arr.values).reshape(len(arr), arr.type.list_size)
            else:
                self._feat[c] = arr.to_numpy(zero_copy_only=False)

        assert np.all(np.diff(row_episode) >= 0), "episode_index is not contiguous in the frames table"
        ep_vals, ep_starts, ep_counts = np.unique(row_episode, return_index=True, return_counts=True)
        self._ep_row_start = {int(v): int(s) for v, s in zip(ep_vals, ep_starts)}
        # episode_id in span records is positional (0..N-1) — map to the table's episode_index.
        self._ep_index_of = {i: int(v) for i, v in enumerate(ep_vals)}

        tasks = _read_all(db, f"{table}_tasks", ["task_index", "task"])
        self._tasks = dict(zip(tasks.column("task_index").to_pylist(), tasks.column("task").to_pylist()))
        eps = _read_all(db, f"{table}_episodes", ["episode_index", "episode_id"])
        ep_id_str = dict(zip(eps.column("episode_index").to_pylist(), eps.column("episode_id").to_pylist()))

        # -- split + span index via the base's own helpers (identical semantics) --
        episodes_meta = {
            "dataset_from_index": [int(s) for s in ep_starts],
            "dataset_to_index": [int(s + c) for s, c in zip(ep_starts, ep_counts)],
            "length": [int(c) for c in ep_counts],
        }
        episode_ids = split_episode_ids(
            total_episodes=len(ep_vals), seed=self._split_seed, val_ratio=self._split_val_ratio, split=self._split
        )
        episode_spans, _, _ = build_episode_spans(
            episodes=episodes_meta,
            episode_ids=episode_ids,
            chunk_length=self._chunk_length,
            sample_stride=self._sample_stride,
        )
        self._episode_records: list[tuple[int, int, int, int]] = []
        self._episode_cum_ends: list[int] = []
        self._num_valid_indices = 0
        if not use_filter_dict:
            for episode_id, sample_start, valid_len in episode_spans:
                self._episode_records.append((0, sample_start, valid_len, episode_id))
                self._num_valid_indices += valid_len
                self._episode_cum_ends.append(self._num_valid_indices)
        else:
            # Keep-ranges filter — same construction as the base loader.
            with open(filter_dict_path) as f:
                filter_dict = json.load(f)
            for episode_id, sample_start, valid_len in episode_spans:
                eid = str(ep_id_str.get(self._ep_index_of[episode_id], ""))
                key = (
                    f"gs://xembodiment_data/r2d2/r2d2-data-full/{eid}/recordings/"
                    f"MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{eid}/trajectory.h5"
                )
                ranges = filter_dict.get(key)
                if ranges is None:
                    continue
                for s, e in ranges:
                    sub_start = max(s, 0)
                    sub_end = min(e - self._chunk_length, valid_len)
                    sub_valid_len = max(0, sub_end - sub_start)
                    if sub_valid_len > 0:
                        self._episode_records.append((0, sample_start + sub_start, sub_valid_len, episode_id))
                        self._num_valid_indices += sub_valid_len
                        self._episode_cum_ends.append(self._num_valid_indices)

        # -- video: lazy per-worker handles into the composed table --
        self._lance_uri = lance_uri
        self._table = table
        self._decode_device = _resolve_device(decode_device)
        self._cache_size = decoder_cache_size
        self._storage_options = storage_options
        self._perm = None
        self._ep_row: dict[int, int] | None = None
        self._decoders: dict[int, VideoDecoder] | None = None
        self._pending_window: tuple[int, int] | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_perm", "_ep_row", "_decoders"):
            state[k] = None
        return state

    # -- labels: assemble the LeRobot-shaped sample dict from the Lance arrays --

    def _fetch_sample(self, idx: int) -> tuple[str, int, int, dict[str, Any]]:
        mode = self._choose_mode()
        dataset_idx, row_idx, episode_id, _ = self._resolve_index(idx)
        sample: dict[str, Any] = {
            c: torch.from_numpy(self._feat[c][row_idx : row_idx + n].copy()).float()
            for c, n in self._label_windows.items()
        }
        sample["task"] = self._tasks[int(self._row_task[row_idx])]
        ep_index = self._ep_index_of[episode_id]
        self._pending_window = (ep_index, row_idx - self._ep_row_start[ep_index])
        return mode, dataset_idx, row_idx, sample

    # -- video: decode the composed clip window instead of composing 3 views --

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        self._ensure_open()
        ep_index, offset = self._pending_window
        self._ensure_decoders([ep_index])
        idxs = [offset + k for k in range(self._chunk_length + 1)]
        frames = self._decoders[ep_index].get_frames_at(indices=idxs).data  # (T,C,H,W) uint8
        return frames.to(torch.float32) / 255.0  # [0,1] float, as the base expects

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

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        # Warm the decoder cache for the whole batch (one batched byte read), then
        # let the inherited per-sample path assemble each result.
        self._ensure_open()
        eps = {self._ep_index_of[self._resolve_index(int(i))[2]] for i in indices}
        self._ensure_decoders(list(eps))
        return [self[int(i)] for i in indices]


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
    version: str = _DEFAULT_VERSION,
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
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
):
    """Lance drop-in for ``get_action_droid_sft_dataset``: same DROID action SFT
    stack (``ActionTransformPipeline`` + ``ActionSFTDataset``), reading labels and
    pre-composed episodes from LanceDB instead of the raw LeRobot tree."""
    dataset = LanceDROIDComposedDataset(
        lance_uri,
        table=table,
        version=version,
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
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed) if iterable_shuffle else sft


__all__ = [
    "LanceDROIDComposedDataset",
    "LanceDROIDComposedIterable",
    "get_lance_action_droid_sft_dataset",
]
