# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed DROID action dataset.

Drop-in for :class:`DROIDLeRobotDataset` that serves the multi-view video from
a LanceDB ``*_videos`` blob-v2 table instead of seeking mp4 files on disk. The
per-frame tabular/index logic, pose math, action assembly, and the concat-view
layout are inherited unchanged, so the output is identical to the base loader;
only the video I/O path differs.

Video read path (mirrors ``lerobot-lancedb``):
  * ``lance.LanceDataset.take_blobs`` streams the original mp4 bytes from the
    blob-v2 column (range reads, no full-file copy on disk),
  * a per-worker ``torchcodec.VideoDecoder`` cache decodes windows on the fly,
  * ``decode_device="cuda"`` routes decode to NVDEC on the GPU.

``__getitems__`` is the hot path: the PyTorch ``DataLoader`` hands the whole
batch's indices at once, so every frame needed by the batch is decoded with one
``get_frames_at`` call per video file — large, contiguous NVDEC work instead of
3 tiny per-sample calls. With ``decode_device="cpu"`` the decoder is byte-
identical to the base loader's torchcodec path, so frames match bit-for-bit
(used by the equivalence test). The frames table is opened through the LanceDB
Permutation API, following ``training/object-detection`` and ``lerobot-lancedb``.
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


class LanceDROIDDataset(DROIDLeRobotDataset):
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
        self._lance_uri = lance_uri
        self._frames_name = frames_table
        self._videos_name = f"{frames_table}_videos"
        self._decode_device = _resolve_device(decode_device)
        self._decoder_cache_size = decoder_cache_size
        self._storage_options = storage_options
        # lazily (re)built per worker — see __getstate__/_ensure_lance_open.
        self._db = None
        self._frames_perm = None
        self._videos_dataset = None
        self._file_row_index: dict[tuple[str, int, int], int] | None = None
        self._decoders: dict[tuple[str, int, int], VideoDecoder] | None = None

    # ── worker-safe lazy handles ──────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_db", "_frames_perm", "_videos_dataset", "_file_row_index", "_decoders"):
            state[k] = None
        return state

    def _ensure_lance_open(self) -> None:
        if self._decoders is not None:
            return
        so = self._storage_options
        self._db = lancedb.connect(self._lance_uri, storage_options=so) if so else lancedb.connect(self._lance_uri)
        frames_table = self._db.open_table(self._frames_name)
        # Permutation handle over the frames table (columnar identity read).
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
            if self._decode_device is not None:
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
        """Wrist on top; the two exteriors resized to half and concatenated on the
        bottom — identical to :meth:`DROIDLeRobotDataset._load_concat_video`."""
        if self._use_image_augmentation:
            if self._image_augmentor is None:
                import torchvision.transforms as T

                _, _, h, w = wrist.shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                        T.Resize((h, w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            n, m = wrist.shape[0], wrist.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([wrist, left, right], dim=0))
            wrist, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist, bottom], dim=-2)

    # ── batched fetch (the DataLoader hot path) ───────────────────────
    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self._ensure_lance_open()
        n = len(indices)

        # Phase 1 — per sample: map index → window, build action (reuses base
        # logic), and register the per-view frame indices into a per-decoder plan.
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
            specs.append(
                {
                    "mode": mode,
                    "action": action,
                    "extras": extras,
                    "ai_caption": random.choice(task.split(" | ")),
                }
            )

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

        # Phase 2 — one batched decode per video file; slice frames back to owners.
        decoded: list[dict[str, torch.Tensor]] = [{} for _ in range(n)]
        for key, entry in plan.items():
            dec = self._decoder_for(*key)
            batch = dec.get_frames_at(indices=entry["fidx"])
            frames = batch.data  # (M, C, H, W) uint8 on decode device
            pts = batch.pts_seconds.to("cpu").to(torch.float32)
            for sp, name, lo, hi, qts in entry["owners"]:
                q = torch.tensor(qts, dtype=torch.float32)
                amin = torch.cdist(q[:, None], pts[lo:hi, None], p=1).min(1).indices
                sel = frames[lo:hi].index_select(0, amin.to(frames.device))
                decoded[sp][name] = sel.to(torch.float32) / 255.0

        # Phase 3 — concat views + assemble the result dict (base logic).
        results = []
        for sp in range(n):
            fbv = decoded[sp]
            video = self._concat_views(fbv["wrist"], fbv["left"], fbv["right"])
            s = specs[sp]
            results.append(
                self._build_result(
                    mode=s["mode"],
                    video=video,
                    action=s["action"],
                    ai_caption=s["ai_caption"],
                    additional_view_description=_ADDITIONAL_VIEW_DESC,
                    **s["extras"],
                )
            )
        return results


class LanceDROIDComposedDataset(DROIDLeRobotDataset):
    """Fastest action loader: decodes a pre-composed, pre-resized, short-GOP
    per-episode clip (one stream) instead of 3 full views + resize + concat.

    Built by ``tools/lance_datagen/build_composed_droid.py``. Uses ``seek_mode=
    "approximate"`` (skips the full-file scan — cheap decoder init for the
    shuffled, many-file pattern) and a per-worker LRU decoder cache. Output
    matches the base loader within H.264 re-encode tolerance (the resize/concat
    is the base's exact op, done once offline). Index/action logic inherited.
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
        self._comp = lance.dataset(
            f"{self._lance_uri}/{self._table}.lance", storage_options=self._storage_options
        )
        rows = self._comp.to_table(columns=["episode_index"]).to_pylist()
        self._ep_row = {int(r["episode_index"]): i for i, r in enumerate(rows)}
        # A plain large_binary column is read far faster on object storage with a
        # columnar `take` (uses the IO thread pool) than `take_blobs` (which streams
        # BlobFile handles read one-at-a-time -> serialized GETs, ~6x slower on S3).
        # Blob encoding only pays off for multi-GB payloads; training clips are <2MB.
        meta = self._comp.schema.field("video_bytes").metadata or {}
        self._is_blob = meta.get(b"lance-encoding:blob") == b"true"
        self._decoders = {}

    def _read_clip_bytes(self, rows: list[int]) -> list[bytes]:
        """Fetch the mp4 bytes for the given table rows, batched. Uses a columnar
        take for plain binary (parallel IO) and take_blobs for a blob column."""
        if self._is_blob:
            out = []
            for blob in self._comp.take_blobs(blob_column="video_bytes", indices=rows):
                out.append(blob.readall())
                blob.close()
            return out
        col = self._comp.take(rows, columns=["video_bytes"]).column("video_bytes")
        return [v.as_py() for v in col]

    def _build_decoder(self, data: bytes) -> VideoDecoder:
        if self._decode_device is not None:
            return VideoDecoder(data, seek_mode="approximate", device=str(self._decode_device))
        return VideoDecoder(data, seek_mode="approximate")

    def _ensure_decoders(self, ep_indices: list[int]) -> None:
        """Batch-fetch all cache-missing episode clips in ONE ``take_blobs`` call.

        On S3 this issues the GETs concurrently (~2.3× faster than fetching per episode
        in a loop, measured); on a single-episode batch it degrades to one read."""
        needed = list(dict.fromkeys(ep_indices))
        needed_set = set(needed)
        missing = [e for e in needed if e not in self._decoders]
        if not missing:
            return
        datas = self._read_clip_bytes([self._ep_row[e] for e in missing])
        for e, data in zip(missing, datas):
            # evict an LRU entry NOT needed by the current batch (never drop a hit we're
            # about to decode); if all cached entries are needed, exceed the cap this batch.
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
        specs: list[dict[str, Any]] = []
        plan: dict[int, dict[str, Any]] = {}
        for sp, idx in enumerate(indices):
            idx = int(idx)
            mode = self._choose_mode()
            ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
            prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
            offset = idx - prev  # frame offset within the episode (== within the clip)
            start = int(self._ep_starts[ep]) + offset
            ep_index = int(self._ep_vals[ep])
            obs = self._window_rows(start, start + self._chunk_length + 1, ep_index)
            if self._action_space == "joint_pos":
                action = self._build_joint_action(obs)
                extras: dict[str, Any] = {}
            else:
                action, initial_pose = self._build_raw_action(obs, obs[: self._chunk_length])
                extras = {"initial_pose": initial_pose}
            task = self._tasks[int(obs[0]["task_index"])]
            specs.append({"mode": mode, "action": action, "extras": extras,
                          "ai_caption": random.choice(task.split(" | "))})
            clip_idx = [offset + k for k in range(self._chunk_length + 1)]
            e = plan.setdefault(ep_index, {"frames": [], "owners": []})
            lo = len(e["frames"])
            e["frames"].extend(clip_idx)
            e["owners"].append((sp, lo, lo + len(clip_idx)))

        self._ensure_decoders(list(plan.keys()))  # one batched take_blobs for all missing clips
        decoded: list[torch.Tensor | None] = [None] * n
        for ep_index, e in plan.items():
            dec = self._decoders[ep_index]
            frames = dec.get_frames_at(indices=e["frames"]).data  # (M, C, 270, 320) uint8
            for sp, lo, hi in e["owners"]:
                decoded[sp] = frames[lo:hi].to(torch.float32) / 255.0

        results = []
        for sp in range(n):
            s = specs[sp]
            results.append(
                self._build_result(
                    mode=s["mode"], video=decoded[sp], action=s["action"],
                    ai_caption=s["ai_caption"], additional_view_description=_ADDITIONAL_VIEW_DESC,
                    **s["extras"],
                )
            )
        return results


class LanceDROIDComposedIterable(torch.utils.data.IterableDataset):
    """Episode-shuffle stream over a :class:`LanceDROIDComposedDataset` (borrowed from
    the base ``ActionIterableShuffleDataset``).

    Shuffles per-episode block ORDER and streams windows WITHIN each episode
    sequentially, sharded disjointly across (rank, worker). Because consecutive windows
    share an episode, the per-worker decoder for that episode's clip is built ONCE and
    reused for all its windows — instead of ``RandomSampler`` rebuilding it (a fresh
    ``take_blobs`` + ``VideoDecoder``) on nearly every window. This keeps batch diversity
    (N workers stream N different episodes) while making blob reads sequential — a large
    win in the data-bound / object-store regime. Re-shuffles each epoch, streams forever.
    """

    def __init__(self, composed: LanceDROIDComposedDataset, seed: int = 42):
        super().__init__()
        self._ds = composed
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self):
        blocks = self._ds.get_shuffle_blocks()  # per-episode (start, length), inherited
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        shard = int(self.shard_rank) * nw + wid
        total = max(1, int(self.shard_world_size) * nw)
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[shard::total]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._ds[idx]
            epoch += 1


__all__ = ["LanceDROIDDataset", "LanceDROIDComposedDataset", "LanceDROIDComposedIterable"]
