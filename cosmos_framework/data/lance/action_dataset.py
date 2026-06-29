# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed DROID action dataset.

Replaces DROIDLeRobotDataset with a version that reads from LanceDB for improved I/O.
Inherits indexing, pose math, and action assembly from the base loader.
"""
from __future__ import annotations

import random
from typing import Any

import lance
import lancedb
import numpy as np
import torch
import torch.nn.functional as F
from lancedb.permutation import Permutation
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import (
    _IMAGE_FEATURES,
    DROIDLeRobotDataset,
)

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


class LanceDROIDDataset(_FreeBaseRowsMixin, DROIDLeRobotDataset):
    """LanceDB-backed version of DROIDLeRobotDataset.

    Stores original mp4 bytes in a Lance table.
    """
    def __init__(
        self,
        root: str,
        lance_uri: str,
        *,
        frames_table: str = "droid",
        decode_device: str | None = "cpu",
        decoder_cache_size: int = 8,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, **kwargs)
        self._free_base_rows()
        self._lance_uri = lance_uri
        self._frames_name = frames_table
        self._videos_name = f"{frames_table}_videos"
        self._decode_device = _resolve_device(decode_device)
        self._decoder_cache_size = decoder_cache_size
        self._storage_options = storage_options
        self._db = None
        self._frames_perm = None
        self._videos_dataset = None
        self._file_row_index: dict[tuple[str, int, int], int] | None = None
        self._decoders: dict[tuple[str, int, int], VideoDecoder] | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_db", "_frames_perm", "_videos_dataset", "_file_row_index", "_decoders"):
            state[k] = None
        return state

    def _ensure_lance_open(self) -> None:
        if self._decoders is not None:
            return
        so = self._storage_options
        if so:
            self._db = lancedb.connect(self._lance_uri, storage_options=so)
        else:
            self._db = lancedb.connect(self._lance_uri)
        frames_table = self._db.open_table(self._frames_name)
        self._frames_perm = Permutation.identity(frames_table).with_format("arrow")
        self._videos_dataset = lance.dataset(
            f"{self._lance_uri}/{self._videos_name}.lance", storage_options=so
        )
        rows = self._videos_dataset.to_table(
            columns=["video_key", "chunk_index", "file_index"]
        ).to_pylist()
        self._file_row_index = {
            (str(r["video_key"]), int(r["chunk_index"]), int(r["file_index"])): i
            for i, r in enumerate(rows)
        }
        self._decoders = {}

    def _decoder_for(self, video_key: str, chunk: int, file: int) -> VideoDecoder:
        key = (video_key, chunk, file)
        dec = self._decoders.get(key)
        if dec is None:
            row = self._file_row_index[key]
            blob = self._videos_dataset.take_blobs(blob_column="video_bytes", indices=[row])[0]
            data = blob.readall()
            blob.close()
            if self._decode_device:
                dec = VideoDecoder(data, device=str(self._decode_device))
            else:
                dec = VideoDecoder(data)
            if len(self._decoders) >= self._decoder_cache_size:
                self._decoders.pop(next(iter(self._decoders)))
            self._decoders[key] = dec
        return dec

    def _video_chunk_file(self, episode: dict[str, Any], video_key: str) -> tuple[int, int]:
        ci = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        fi = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        return ci, fi

    def _concat_views(self, wrist: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self._use_image_augmentation:
            if self._image_augmentor is None:
                import torchvision.transforms as T
                _, _, h, w = wrist.shape
                self._image_augmentor = T.Compose([
                    T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                    T.Resize((h, w), antialias=True),
                    T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                ])
            n, m = wrist.shape[0], wrist.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([wrist, left, right], dim=0))
            wrist, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist, bottom], dim=-2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self._ensure_lance_open()
        n = len(indices)
        specs: list[dict[str, Any]] = []
        plan: dict[tuple[str, int, int], dict[str, Any]] = {}
        for sp, idx in enumerate(indices):
            idx = int(idx)
            mode = self._choose_mode()
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
            obs = self._window_rows(start, start + self._chunk_length + 1, episode_index)
            timestamps = [float(r["timestamp"]) for r in obs]

            if self._action_space == "joint_pos":
                action = self._build_joint_action(obs)
                extras: dict[str, Any] = {}
            else:
                action, initial_pose = self._build_raw_action(obs, obs[: self._chunk_length])
                extras = {"initial_pose": initial_pose}
            task = self._tasks[int(obs[0]["task_index"])]
            specs.append({
                "mode": mode, "action": action, "extras": extras,
                "ai_caption": random.choice(task.split(" | ")),
            })

            for name, video_key in _IMAGE_FEATURES.items():
                ci, fi = self._video_chunk_file(episode, video_key)
                dec = self._decoder_for(video_key, ci, fi)
                avg = dec.metadata.average_fps
                from_ts = float(episode.get(f"videos/{video_key}/from_timestamp", 0.0))
                qts = [from_ts + t for t in timestamps]
                fidx = [round(t * avg) for t in qts]
                entry = plan.setdefault((video_key, ci, fi), {"fidx": [], "owners": []})
                lo = len(entry["fidx"])
                entry["fidx"].extend(fidx)
                entry["owners"].append((sp, name, lo, lo + len(fidx), qts))

        decoded: list[dict[str, torch.Tensor]] = [{} for _ in range(n)]
        for key, entry in plan.items():
            dec = self._decoder_for(*key)
            batch = dec.get_frames_at(indices=entry["fidx"])
            frames = batch.data
            pts = batch.pts_seconds.to("cpu").to(torch.float32)
            for sp, name, lo, hi, qts in entry["owners"]:
                q = torch.tensor(qts, dtype=torch.float32)
                amin = torch.cdist(q[:, None], pts[lo:hi, None], p=1).min(1).indices
                sel = frames[lo:hi].index_select(0, amin.to(frames.device))
                decoded[sp][name] = sel.to(torch.float32) / 255.0

        results = []
        for sp in range(n):
            fbv = decoded[sp]
            video = self._concat_views(fbv["wrist"], fbv["left"], fbv["right"])
            s = specs[sp]
            results.append(
                self._build_result(
                    mode=s["mode"], video=video, action=s["action"], ai_caption=s["ai_caption"],
                    additional_view_description=_ADDITIONAL_VIEW_DESC, **s["extras"],
                )
            )
        return results


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
        self._comp = None
        self._ep_row: dict[int, int] | None = None
        self._decoders: dict[int, VideoDecoder] | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_comp", "_ep_row", "_decoders"):
            state[k] = None
        return state

    def _ensure_open(self) -> None:
        if self._decoders is not None:
            return
        self._comp = lance.dataset(f"{self._lance_uri}/{self._table}.lance", storage_options=self._storage_options)
        rows = self._comp.to_table(columns=["episode_index"]).to_pylist()
        self._ep_row = {int(r["episode_index"]): i for i, r in enumerate(rows)}
        meta = self._comp.schema.field("video_bytes").metadata or {}
        self._is_blob = meta.get(b"lance-encoding:blob") == b"true"
        self._decoders = {}

    def _read_clip_bytes(self, rows: list[int]) -> list[bytes]:
        if self._is_blob:
            out = []
            for blob in self._comp.take_blobs(blob_column="video_bytes", indices=rows):
                out.append(blob.readall())
                blob.close()
            return out
        col = self._comp.take(rows, columns=["video_bytes"]).column("video_bytes")
        return [v.as_py() for v in col]

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


__all__ = ["LanceDROIDDataset", "LanceDROIDComposedDataset", "LanceDROIDComposedIterable"]
