# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tar-backed DPO preference dataset (HPDv3-style) — port of cosmos-rl's
`tools/dataset/wfm/tar_dpo_dataset.py` onto posttrain's `DPOPreferenceDataset`.

Reads (winner, loser) JPG bytes directly from ONE concatenated plain `images.tar` on
lustre via byte-offset seek+read — avoids the lustre small-file metadata bottleneck.
JSONL row schema (produced by cosmos-rl `scripts/t2i_dpo/{build_tar_index,resolve_pairs}.py`):

    {"prompt": ..., "win_offset": int, "win_length": int, "lose_offset": int,
     "lose_length": int, "score_gap"/"wm_gap"/"align_gap": float, "fps": int, ...}

Alignment note (wallace parity): cosmos-rl feeds HPDv3 images as T=1 VIDEO samples
(`winner_video` key, VIDEO_RES_SIZE_INFO buckets) — so `media` defaults to "video" with
`num_frames=1` here to reproduce the exact preprocessing of the wallace runs.
`__getitem__` returns the parent's `(prompt, winner_dict, loser_dict)` contract, so the
Env/trainer/objective are untouched.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from typing import Any

import torch

from cosmos_framework.posttrain.data.datasets.dpo_preference import (
    DPOPreferenceDataset,
    _canonical_sha256_digest,
    _snap_to_bucket,
)


class _SharedTarHandle:
    """Per-process lazy file handle for the plain tar (opened on first read so it never
    crosses a DataLoader fork boundary)."""

    def __init__(self, tar_path: str) -> None:
        self.tar_path = tar_path
        self._fh: Any = None
        self._lock = threading.Lock()

    def read(self, offset: int, length: int) -> bytes:
        if self._fh is None:
            with self._lock:
                if self._fh is None:
                    self._fh = open(self.tar_path, "rb", buffering=0)
        with self._lock:
            self._fh.seek(offset)
            buf = self._fh.read(length)
        if len(buf) != length:
            raise RuntimeError(f"TarDPOPreferenceDataset: short read at offset={offset} length={length} got={len(buf)}")
        return buf


_TAR_HANDLES: dict[str, _SharedTarHandle] = {}
_TAR_HANDLES_LOCK = threading.Lock()


def _get_handle(tar_path: str) -> _SharedTarHandle:
    # Double-checked so two threads can't each construct a handle for the same path
    # (benign under the GIL — last write won — but explicit and no-GIL-proof).
    if tar_path not in _TAR_HANDLES:
        with _TAR_HANDLES_LOCK:
            if tar_path not in _TAR_HANDLES:
                _TAR_HANDLES[tar_path] = _SharedTarHandle(tar_path)
    return _TAR_HANDLES[tar_path]


class TarDPOPreferenceDataset(DPOPreferenceDataset):
    """`DPOPreferenceDataset` whose media come from byte offsets into one tar instead of
    per-pair file paths. Reuses the parent's bucket-snap (bit-parity with cosmos-rl),
    symmetric cfg-dropout, and `_side` batch format."""

    def __init__(
        self,
        jsonl_path: str,
        tar_path: str,
        resolution: str | None = "720",
        num_frames: int = 1,
        caption_key: str = "prompt",
        min_score_gap: float = 0.0,
        min_wm_gap: float = 0.0,
        min_align_gap: float = 0.0,
        cfg_dropout_rate: float = 0.1,
        default_fps: int = 1,
        seed: int = 0,
        media: str = "video",  # wallace parity: T=1 video samples, VIDEO res table
    ) -> None:
        super().__init__(
            jsonl_path,
            resolution=resolution,
            num_frames=num_frames,
            caption_key=caption_key,
            min_score_gap=min_score_gap,
            min_wm_gap=min_wm_gap,
            min_align_gap=min_align_gap,
            cfg_dropout_rate=cfg_dropout_rate,
            default_fps=default_fps,
            seed=seed,
            media=media,
        )
        if not tar_path:
            raise ValueError("TarDPOPreferenceDataset requires tar_path")
        self._default_tar_spec = os.fspath(tar_path)
        self._default_tar_path = self._resolve_media_path(tar_path)
        # Fail loud up front on a wrong-schema jsonl (path-based rows belong to the parent).
        for i, e in enumerate(self.entries[:5]):
            for k in ("win_offset", "win_length", "lose_offset", "lose_length"):
                if k not in e:
                    raise ValueError(
                        f"TarDPOPreferenceDataset: entry {i} missing '{k}' — this jsonl looks "
                        f"path-based; use DPOPreferenceDataset (rl.data without tar_path) instead."
                    )

    def cursor_fingerprint(self) -> str:
        """Extend the row fingerprint with stable, cheap tar manifest identity."""
        cached = getattr(self, "_tar_cursor_fingerprint", None)
        if cached is not None:
            return str(cached)
        digest = hashlib.sha256(super().cursor_fingerprint().encode())
        tar_paths = {
            self._resolve_media_path(entry.get("tar_path") or self._default_tar_spec) for entry in self.entries
        }
        for resolved in sorted(tar_paths):
            logical_path = os.path.relpath(resolved, self._jsonl_dir)
            try:
                stat = os.stat(resolved)
                identity = (os.path.normpath(logical_path), stat.st_size)
            except OSError:
                identity = (os.path.normpath(logical_path), None)
            digest.update(repr(identity).encode())
        self._tar_cursor_fingerprint = digest.hexdigest()
        return str(self._tar_cursor_fingerprint)

    def _cursor_manifest_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Normalize the effective archive path while retaining all other row fields."""
        manifest_entry = dict(entry)
        tar_spec = entry.get("tar_path") or self._default_tar_spec
        resolved = self._resolve_media_path(tar_spec)
        manifest_entry["tar_path"] = os.path.normpath(os.path.relpath(resolved, self._jsonl_dir))
        return manifest_entry

    def membership_ids(self) -> tuple[str, ...]:
        """Pair identities over prompt and immutable archive-member metadata.

        Validation source setup must stay O(rows), not O(media bytes). The canonical archive
        locator plus offset/length is always an alias; each row-provided digest adds another alias.
        This recognizes the same range when only one manifest declares a digest and copied ranges
        when both manifests declare matching digests, without opening the tar.
        """
        identities: list[str] = []
        for entry in self.entries:
            tar_path = os.path.realpath(self._resolve_media_path(entry.get("tar_path") or self._default_tar_spec))
            winner_identities = self._tar_member_identities(
                entry,
                "winner",
                tar_path,
                int(entry["win_offset"]),
                int(entry["win_length"]),
            )
            loser_identities = self._tar_member_identities(
                entry,
                "loser",
                tar_path,
                int(entry["lose_offset"]),
                int(entry["lose_length"]),
            )
            for winner_identity in winner_identities:
                for loser_identity in loser_identities:
                    payload = {
                        "prompt": entry.get(self.caption_key, ""),
                        "winner": winner_identity,
                        "loser": loser_identity,
                    }
                    encoded = json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                    identities.append(hashlib.sha256(encoded).hexdigest())
        return tuple(sorted(identities))

    @staticmethod
    def _tar_member_identities(
        entry: dict[str, Any],
        side: str,
        path: str,
        offset: int,
        length: int,
    ) -> tuple[str, ...]:
        """Return the canonical archive range plus every declared member digest."""
        identities = [f"tar-range:{path}:{offset}:{length}"]
        for key in (f"{side}_sha256", f"{side}_digest"):
            value = entry.get(key)
            if value not in (None, ""):
                identity = f"digest:{_canonical_sha256_digest(value, key)}"
                if identity not in identities:
                    identities.append(identity)
        return tuple(identities)

    def __getitem__(self, idx: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        entry = self.entries[idx]
        # Symmetric cfg-dropout: ONE decision per pair (identical winner/loser text tokens).
        prompt = entry.get(self.caption_key, "")
        if self._rng.random() < self.cfg_dropout_rate:
            prompt = ""

        tar_spec = entry.get("tar_path") or self._default_tar_spec
        tar_path = self._resolve_media_path(tar_spec)
        handle = _get_handle(tar_path)
        winner = self._read_jpg(handle, int(entry["win_offset"]), int(entry["win_length"]))
        loser = self._read_jpg(handle, int(entry["lose_offset"]), int(entry["lose_length"]))

        if self.resolution is None:
            image_size = None
        else:
            winner, image_size = _snap_to_bucket(winner, self.resolution, table=self._table)
            loser, _ = _snap_to_bucket(loser, self.resolution, table=self._table)
        if winner.shape != loser.shape:
            raise ValueError(
                f"DPO winner/loser shape mismatch at idx={idx}: {tuple(winner.shape)} vs "
                f"{tuple(loser.shape)} — the AR-bin agreement filter should have removed this row."
            )
        if self.media == "image":
            winner, loser = winner.squeeze(1), loser.squeeze(1)  # [C,H,W] for the image routing key
        if image_size is None:
            h, w = winner.shape[-2], winner.shape[-1]
            image_size = torch.tensor([h, w, h, w], dtype=torch.float)
        fps = float(entry.get("fps", self.default_fps))
        return (
            prompt,
            self._side(winner, prompt, image_size, fps),
            self._side(loser, prompt, image_size, fps),
        )

    @staticmethod
    def _read_jpg(handle: _SharedTarHandle, offset: int, length: int) -> torch.Tensor:
        """JPG bytes at (offset, length) → [C, T=1, H, W] uint8 (parity with cosmos-rl)."""
        import numpy as np
        from PIL import Image

        pil = Image.open(io.BytesIO(handle.read(offset, length)))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H, W, C] (copy: PIL buffers are read-only)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(1)
