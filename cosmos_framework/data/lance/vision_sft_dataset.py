# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed local vision-SFT (video+caption) dataset.

A drop-in alternative to the local ``LocalSFTDataset`` (the faithful map-style
representative of cosmos ``SFTDataset``). Instead of seeking the source mp4 on
disk and resizing it per sample, it decodes a **pre-resized, short-GOP** per-clip
mp4 from a Lance blob-v2 column and tokenizes the same caption.

Built by ``tools/lance_datagen/build_vision_sft.py``: each clip is decoded once,
resized to the training resolution (the base loader's exact resize op), and
re-encoded all-intra (``gop=1``) into one per-clip blob. The Lance loader then
applies the *same* ``entire_chunk`` window math + temporal subsample + spatial
center-crop + temporal truncation as the base, so its output matches within
H.264 re-encode tolerance; the caption is stored verbatim so token ids are exact.

Structure mirrors ``action_dataset.LanceDROIDComposedDataset`` exactly:
  * worker-safe lazy lance handle (``__getstate__`` nulls it, ``_ensure_open``
    rebuilds it per worker),
  * a per-worker ``torchcodec.VideoDecoder`` LRU cache, each built from
    ``lance.dataset(...).take_blobs(...)[0].readall()`` with
    ``seek_mode="approximate"`` (cheap init for shuffled many-file reads — every
    frame is a keyframe so approximate seek is exact),
  * batched ``__getitems__`` that groups the frame decodes per clip (one
    ``get_frames_at`` per clip instead of one per sample).
"""
from __future__ import annotations

import json
from typing import Any, Optional

import lance
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import select_caption

_MAX_CAPTION_TOKENS = 1024
_META_COLS = [
    "clip_id", "width", "height", "start_frame", "end_frame",
    "temporal_interval", "enc_h", "enc_w", "fps", "caption_json", "caption",
]


def _resolve_device(device: str | None) -> torch.device | None:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if device in (None, "cpu"):
        return None
    return torch.device(device)


class LanceVisionSFTDataset(torch.utils.data.Dataset):
    """Map-style local vision-SFT loader backed by a Lance blob-v2 video table.

    Output dict matches ``LocalSFTDataset.__getitem__`` (``video`` uint8 C,T,H,W;
    ``text_token_ids``; SFT metadata). Worker-safe: only connection params are
    pickled; each worker reopens its own lance handle + decoder cache."""

    def __init__(
        self,
        lance_uri: str,
        *,
        table: str = "vision_sft",
        resolution: str = "256",
        num_video_frames: int = 16,
        temporal_interval_mode: str = "entire_chunk",
        frame_selection_mode: str = "first",
        temporal_compression_factor: int = 4,
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "Qwen/Qwen2.5-7B",
        use_system_prompt: bool = False,
        max_caption_tokens: int = _MAX_CAPTION_TOKENS,
        decode_device: str | None = "cpu",
        decoder_cache_size: int = 32,
        storage_options: dict | None = None,
    ) -> None:
        assert temporal_interval_mode in ("force_one", "max_30fps", "entire_chunk")
        assert frame_selection_mode in ("center", "first", "random")
        self._lance_uri = lance_uri
        self._table = table
        self._resolution_str = resolution
        self.num_video_frames = num_video_frames
        self.temporal_interval_mode = temporal_interval_mode
        self.frame_selection_mode = frame_selection_mode
        self.temporal_compression_factor = temporal_compression_factor
        self.use_system_prompt = use_system_prompt
        self.max_caption_tokens = max_caption_tokens
        self.tokenizer_name = tokenizer_name
        self._decode_device = _resolve_device(decode_device)
        self._cache_size = decoder_cache_size
        self._storage_options = storage_options
        self._tokenizer = tokenizer

        # lazily (re)built per worker — see __getstate__/_ensure_open.
        self._ds = None
        self._rows: list[dict] | None = None
        self._decoders: dict[int, VideoDecoder] | None = None

        # length is needed eagerly (for samplers) — read it once, then close.
        ds = lance.dataset(f"{lance_uri}/{table}.lance", storage_options=storage_options)
        self._length = ds.count_rows()

    # ── worker-safe lazy handles ──────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_ds", "_rows", "_decoders", "_tokenizer"):
            state[k] = None
        return state

    def _ensure_open(self) -> None:
        if self._decoders is not None:
            return
        self._ds = lance.dataset(
            f"{self._lance_uri}/{self._table}.lance", storage_options=self._storage_options
        )
        self._rows = self._ds.to_table(columns=_META_COLS).to_pylist()
        self._decoders = {}

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            from cosmos_framework.data.vfm.sequence_packing import add_special_tokens

            tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tok, _ = add_special_tokens(tok)
            self._tokenizer = tok
        return self._tokenizer

    def _decoder(self, row: int) -> VideoDecoder:
        d = self._decoders.get(row)
        if d is None:
            blob = self._ds.take_blobs(blob_column="video_bytes", indices=[row])[0]
            data = blob.readall()
            blob.close()
            if self._decode_device is not None:
                d = VideoDecoder(data, seek_mode="approximate", device=str(self._decode_device))
            else:
                d = VideoDecoder(data, seek_mode="approximate")
            if len(self._decoders) >= self._cache_size:
                self._decoders.pop(next(iter(self._decoders)))
            self._decoders[row] = d
        return d

    def __len__(self) -> int:
        return self._length

    skip_tokenize: bool = False  # benchmark raw-video mode toggle (picklable)

    def _tokenize(self, caption: str) -> list[int]:
        if self.skip_tokenize:
            return []
        from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption

        ids = tokenize_caption(
            caption, self._ensure_tokenizer(), is_video=True, use_system_prompt=self.use_system_prompt
        )
        return ids[: self.max_caption_tokens]

    # ── window math (identical to LocalSFTDataset) ────────────────────
    def _window_plan(self, meta: dict) -> tuple[int, int, int]:
        """Return (start_frame, end_frame, temporal_interval) within the stored clip.

        The stored clip already spans only [start_frame, end_frame] of the source
        (it was decoded from the full source but covers all of it; for these SFT
        windows start_frame=0 and end_frame=last). We replicate the base's math on
        the source frame indices, which the stored clip indexes 1:1 (it holds every
        source frame at the resized resolution)."""
        window_start = meta["start_frame"]
        window_end = meta["end_frame"]
        # stored clip covers the whole source, so its frame count == source total.
        clip_total = meta["_clip_total"]
        actual_end = min(window_end, clip_total - 1)
        frames_in_window = actual_end - window_start + 1
        if self.num_video_frames == -1:
            return window_start, actual_end, meta["temporal_interval"]
        if frames_in_window < self.num_video_frames:
            raise ValueError(f"Not enough frames in window for {meta['clip_id']}")
        if self.temporal_interval_mode == "force_one":
            temporal_interval = 1
        elif self.temporal_interval_mode == "max_30fps":
            temporal_interval = max(1, int(meta["fps"] / 30.0))
        else:
            temporal_interval = max(1, frames_in_window // self.num_video_frames)
        num_before = (self.num_video_frames - 1) * temporal_interval + 1
        if self.frame_selection_mode == "first":
            start_frame = window_start
        elif self.frame_selection_mode == "center":
            start_frame = window_start + (frames_in_window - num_before) // 2
        else:
            import random

            start_frame = window_start + random.randint(0, max(0, frames_in_window - num_before))
        end_frame = start_frame + num_before - 1
        return start_frame, end_frame, temporal_interval

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self._ensure_open()
        n = len(indices)

        # Phase 1 — per sample: resolve clip metadata, compute the window frame
        # indices, register them into a per-clip decode plan.
        specs: list[dict[str, Any]] = []
        plan: dict[int, dict[str, Any]] = {}
        for sp, idx in enumerate(indices):
            row = int(idx)
            r = self._rows[row]
            dec = self._decoder(row)
            clip_total = dec.metadata.num_frames
            r = {**r, "_clip_total": clip_total}
            start_frame, end_frame, ti = self._window_plan(r)
            frame_idx = list(range(start_frame, end_frame + 1, ti))

            # spatial center-crop params (the base's exact op, deferred to decode)
            target_w, target_h = self._target_size(r)
            crop_y = round((r["enc_h"] - target_h) / 2)
            crop_x = round((r["enc_w"] - target_w) / 2)

            caption_key, caption, _ = select_caption(self._window_dict(r))
            specs.append(
                {
                    "row": row, "clip_id": r["clip_id"], "fps": r["fps"],
                    "clip_total": clip_total, "win_idx": 0, "temporal_interval": ti,
                    "start_frame": start_frame, "end_frame": end_frame,
                    "crop": (crop_y, crop_x, target_h, target_w),
                    "caption": caption, "caption_key": caption_key,
                }
            )
            e = plan.setdefault(row, {"frames": [], "owners": []})
            lo = len(e["frames"])
            e["frames"].extend(frame_idx)
            e["owners"].append((sp, lo, lo + len(frame_idx)))

        # Phase 2 — one batched decode per clip; slice frames back to owners.
        decoded: list[torch.Tensor | None] = [None] * n
        for row, e in plan.items():
            dec = self._decoder(row)
            frames = dec.get_frames_at(indices=e["frames"]).data  # (M, C, enc_h, enc_w) uint8
            for sp, lo, hi in e["owners"]:
                decoded[sp] = frames[lo:hi]

        # Phase 3 — crop + temporal truncate + tokenize + assemble (base logic).
        results = []
        for sp in range(n):
            s = specs[sp]
            vid = decoded[sp]  # (T, C, enc_h, enc_w) uint8
            cy, cx, th, tw = s["crop"]
            # temporal truncation to compression_factor*N + 1 (base order: trunc then crop)
            t = vid.shape[0]
            target_t = (t - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
            vid = vid[:target_t, :, cy : cy + th, cx : cx + tw]  # (T,C,th,tw)
            video = vid.permute(1, 0, 2, 3).contiguous().to(torch.uint8)  # (C,T,H,W)

            text_ids = self._tokenize(s["caption"])
            image_size = torch.tensor([th, tw, th, tw], dtype=torch.float32)
            padding_mask = torch.zeros((1, th, tw), dtype=torch.float32)
            results.append(
                dict(
                    __key__=s["clip_id"],
                    __url__=s["clip_id"],
                    fps=s["fps"],
                    n_orig_video_frames=s["clip_total"],
                    chunk_index=s["win_idx"],
                    frame_start=s["start_frame"],
                    frame_end=s["end_frame"],
                    num_frames=video.shape[1],
                    video=video,
                    num_multiplier=s["temporal_interval"],
                    padding_mask=padding_mask,
                    image_size=image_size,
                    ai_caption=s["caption"],
                    sampled_caption_style=s["caption_key"],
                    text_token_ids=torch.tensor(text_ids, dtype=torch.long),
                )
            )
        return results

    # ── helpers ───────────────────────────────────────────────────────
    def _target_size(self, r: dict) -> tuple[int, int]:
        """Recover (target_w, target_h) from the stored resized size + orig aspect.

        ``enc_h/enc_w`` is the resize-ratio size; the crop target is the
        ``VIDEO_RES_SIZE_INFO`` bucket for the original aspect ratio."""
        from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import _get_aspect_ratio
        from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO

        ar = _get_aspect_ratio(r["width"], r["height"])
        # resolution bucket inferred from enc size: the stored clip was resized so
        # that max(target_w/in_w, target_h/in_h); recover target from the bucket that
        # the converter used. We carry resolution implicitly via the bucket lookup at
        # the build resolution — default "256".
        target_w, target_h = VIDEO_RES_SIZE_INFO[self._resolution()][ar]
        return target_w, target_h

    def _resolution(self) -> str:
        return getattr(self, "_resolution_str", "256")

    def _window_dict(self, r: dict) -> dict:
        """Reconstruct a t2w_window-shaped dict for select_caption."""
        w: dict[str, Any] = {}
        if r.get("caption_json"):
            w["caption_json"] = json.loads(r["caption_json"])
        if r.get("caption"):
            w["caption"] = r["caption"]
        return w


__all__ = ["LanceVisionSFTDataset"]
