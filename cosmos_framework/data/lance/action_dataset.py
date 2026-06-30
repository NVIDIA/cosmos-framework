# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed DROID action dataset.

Replaces DROIDLeRobotDataset with a version that reads from LanceDB for improved I/O.
Inherits indexing, pose math, and action assembly from the base loader.
"""
from __future__ import annotations

import random
from typing import Any

import lancedb
import numpy as np
import torch
from lancedb.permutation import Permutation
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

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


class _FreeBaseRowsMixin:
    """Frees ActionBaseDataset._rows to reduce memory footprint when using many workers."""

    def _free_base_rows(self) -> None:
        self._rows = None


class LanceDROIDComposedDataset(_FreeBaseRowsMixin, DROIDLeRobotDataset):
    """Action loader using pre-composed, pre-resized episodes stored in LanceDB.

    Decodes a single video stream per episode instead of 3 views.
    """
    def __init__(
        self,
        root: str,
        lance_uri: str,
        *,
        table: str = "droid_composed",
        decode_device: str | None = "cpu",
        decoder_cache_size: int = 32,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, **kwargs)
        self._free_base_rows()
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
        db = (lancedb.connect(self._lance_uri, storage_options=self._storage_options)
              if self._storage_options else lancedb.connect(self._lance_uri))
        tbl = db.open_table(self._table)
        ep = Permutation.identity(tbl).select_columns(["episode_index"]).with_format("arrow")
        rows = ep.__getitems__(list(range(tbl.count_rows())))
        self._ep_row = {int(rows.column("episode_index")[i].as_py()): i for i in range(rows.num_rows)}
        self._perm = Permutation.identity(tbl).select_columns(["video_bytes"]).with_format("arrow")
        self._decoders = {}

    def _read_clip_bytes(self, rows: list[int]) -> list[bytes]:
        # Plain large_binary via the Permutation API. Composed clips are small (<~2MB) where this
        # is fastest on S3. TODO: move to blob-v2 — for larger per-row payloads (>=~8-16MB) a
        # parallel blob-v2 read beats plain (read concurrently, not a serial take_blobs loop).
        batch = self._perm.__getitems__([int(r) for r in rows])
        return [batch.column("video_bytes")[i].as_py() for i in range(batch.num_rows)]

    def _build_decoder(self, data: bytes) -> VideoDecoder:
        device = str(self._decode_device) if self._decode_device else None
        return VideoDecoder(data, seek_mode="approximate", device=device)

    def _ensure_decoders(self, ep_indices: list[int]) -> None:
        needed = list(dict.fromkeys(ep_indices))
        needed_set = set(needed)
        missing = [e for e in needed if e not in self._decoders]
        if not missing:
            return
        datas = self._read_clip_bytes([self._ep_row[e] for e in missing])
        for e, data in zip(missing, datas):
            while len(self._decoders) >= self._cache_size:
                victim = next((k for k in self._decoders if k not in needed_set), None)
                if victim is None:
                    break
                self._decoders.pop(victim)
            self._decoders[e] = self._build_decoder(data)

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
            specs.append({
                "mode": mode,
                "action": action,
                "extras": extras,
                "ai_caption": random.choice(task.split(" | "))
            })
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
            results.append(self._build_result(
                mode=s["mode"], video=decoded[sp], action=s["action"], ai_caption=s["ai_caption"],
                additional_view_description=_ADDITIONAL_VIEW_DESC, **s["extras"],
            ))
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


__all__ = ["LanceDROIDComposedDataset", "LanceDROIDComposedIterable"]
