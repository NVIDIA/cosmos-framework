# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed local vision-SFT (video+caption) dataset.

Alternative to SFTDataset that decodes pre-resized, short-GOP per-clip mp4s from LanceDB.
Reuses the base's caption selection and tokenization logic.
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional

import lancedb
import numpy as np
import torch
from lancedb.permutation import Permutation
from torchcodec.decoders import VideoDecoder
from transformers import AutoTokenizer

from cosmos_framework.data.generator.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    _DURATION_TEMPLATE,
    _RESOLUTION_TEMPLATE,
    _select_caption,
)
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log

_MAX_CAPTION_TOKENS = 1024
_META_COLS = [
    "clip_id",
    "width",
    "height",
    "start_frame",
    "end_frame",
    "temporal_interval",
    "enc_h",
    "enc_w",
    "fps",
    "caption_json",
    "caption",
]


def _resolve_device(device: str | None) -> torch.device | None:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if device in (None, "cpu"):
        return None
    return torch.device(device)


class LanceVisionSFTDataset(torch.utils.data.Dataset):
    """Map-style local vision-SFT loader backed by LanceDB.

    Decodes pre-resized clips in-process, avoiding ffmpeg subprocess overhead.
    """

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
        cfg_dropout_rate: float = 0.0,
        cfg_dropout_keep_metadata: bool = False,
        caption_suffix: str = "",
        conditioning_fps: float = 24,
        conditioning_fps_noise_std: float = 0.0,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
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
        self.cfg_dropout_rate = cfg_dropout_rate
        self.cfg_dropout_keep_metadata = cfg_dropout_keep_metadata
        self.caption_suffix = caption_suffix.strip()
        self.conditioning_fps = conditioning_fps
        self.conditioning_fps_noise_std = conditioning_fps_noise_std
        self.append_duration_fps_timestamps = append_duration_fps_timestamps
        self.append_resolution_info = append_resolution_info
        self.tokenizer_name = tokenizer_name
        self._decode_device = _resolve_device(decode_device)
        self._cache_size = decoder_cache_size
        self._storage_options = storage_options
        self._tokenizer = tokenizer

        self._perm = None
        self._rows: list[dict] | None = None
        self._decoders: dict[int, VideoDecoder] | None = None

        self._length = lancedb.connect(lance_uri, storage_options=storage_options).open_table(table).count_rows()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("_perm", "_rows", "_decoders", "_tokenizer"):
            state[k] = None
        return state

    def _ensure_open(self) -> None:
        if self._decoders is not None:
            return
        tbl = lancedb.connect(self._lance_uri, storage_options=self._storage_options).open_table(self._table)
        meta_perm = Permutation.identity(tbl).select_columns(_META_COLS).with_format("arrow")
        self._rows = meta_perm.__getitems__(list(range(tbl.count_rows()))).to_pylist()
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

    def _ensure_decoders(self, rows: list[int]) -> None:
        needed = list(dict.fromkeys(rows))
        needed_set = set(needed)
        missing = [r for r in needed if r not in self._decoders]
        if not missing:
            return
        clips = self._read_clip_bytes(missing)
        for r in missing:
            while len(self._decoders) >= self._cache_size:
                victim = next((k for k in self._decoders if k not in needed_set), None)
                if victim is None:
                    break
                self._decoders.pop(victim)
            self._decoders[r] = self._build_decoder(clips[r])

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tok, _ = add_special_tokens(tok)
            self._tokenizer = tok
        return self._tokenizer

    def _decoder(self, row: int) -> VideoDecoder:
        d = self._decoders.get(row)
        if d is None:
            d = self._build_decoder(self._read_clip_bytes([row])[row])
            if len(self._decoders) >= self._cache_size:
                self._decoders.pop(next(iter(self._decoders)))
            self._decoders[row] = d
        return d

    def __len__(self) -> int:
        return self._length

    skip_tokenize: bool = False

    def _tokenize(self, caption: str) -> list[int]:
        if self.skip_tokenize:
            return []
        ids = tokenize_caption(
            caption, self._ensure_tokenizer(), is_video=True, use_system_prompt=self.use_system_prompt
        )
        return ids[: self.max_caption_tokens]

    def _window_plan(self, meta: dict) -> tuple[int, int, int] | None:
        window_start, window_end = meta["start_frame"], meta["end_frame"]
        clip_total = meta["_clip_total"]
        actual_end = min(window_end, clip_total - 1)
        frames_in_window = actual_end - window_start + 1
        if self.num_video_frames == -1:
            return window_start, actual_end, meta["temporal_interval"]
        if frames_in_window < self.num_video_frames:
            # base behavior: warn and skip the sample, never crash the worker
            log.warning(f"Not enough frames in window for {meta['clip_id']}. Skipping sample.")
            return None

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
            start_frame = window_start + random.randint(0, max(0, frames_in_window - num_before))
        return start_frame, start_frame + num_before - 1, temporal_interval

    def _finalize_caption(
        self,
        caption: str,
        used_structured_json: bool,
        *,
        num_decoded_frames: int,
        fps: float,
        target_h: int,
        target_w: int,
    ) -> str:
        """The base's caption post-processing (suffix, CFG dropout, duration/resolution
        conditioning text), applied in the same order as SFTDataset.process_one_sample."""
        cond_fps = fps if self.conditioning_fps < 0 else self.conditioning_fps
        if self.conditioning_fps_noise_std > 0:
            cond_fps = cond_fps * float(np.exp(np.random.randn() * self.conditioning_fps_noise_std))
        if self.caption_suffix and not used_structured_json:
            caption = (caption + " " + self.caption_suffix).strip()
        if self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            caption = ""
        if self.append_duration_fps_timestamps and not used_structured_json:
            caption = caption + " " + _DURATION_TEMPLATE.format(duration=num_decoded_frames / cond_fps, fps=cond_fps)
        if self.append_resolution_info and not used_structured_json:
            caption = caption + " " + _RESOLUTION_TEMPLATE.format(height=target_h, width=target_w)
        caption = caption.strip()
        if not self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            caption = ""
        return caption

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        return self.__getitems__([int(idx)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any] | None]:
        """Batched fetch. A slot is ``None`` when the base loader would skip the sample
        (short window / no usable caption) — the same contract as ``process_one_sample``."""
        self._ensure_open()
        n = len(indices)
        self._ensure_decoders([int(i) for i in indices])

        specs: list[dict | None] = []
        plan: dict[int, dict] = {}
        for sp, idx in enumerate(indices):
            row = int(idx)
            r = self._rows[row]
            dec = self._decoder(row)
            clip_total = dec.metadata.num_frames
            r = {**r, "_clip_total": clip_total}
            wp = self._window_plan(r)
            sel = _select_caption(self._window_dict(r))
            if sel is None:
                log.warning(f"No known caption key found for sample {r['clip_id']}. Skipping sample.")
            if wp is None or sel is None:
                specs.append(None)
                continue
            start_frame, end_frame, ti = wp
            # Clamp to the stored clip exactly as the base's sequential decode loop does
            # (out-of-range indices simply don't yield frames there).
            frame_idx = [i for i in range(start_frame, end_frame + 1, ti) if 0 <= i < clip_total]

            target_w, target_h = self._target_size(r)  # (w, h) per VIDEO_RES_SIZE_INFO
            if r["enc_h"] < target_h or r["enc_w"] < target_w:
                raise ValueError(
                    f"stored clip {r['clip_id']} is {r['enc_h']}x{r['enc_w']} but resolution="
                    f"{self._resolution_str!r} needs {target_h}x{target_w}: the table was built "
                    f"at a smaller --resolution than requested"
                )
            crop_y = round((r["enc_h"] - target_h) / 2)
            crop_x = round((r["enc_w"] - target_w) / 2)
            caption_key, caption, used_structured_json = sel
            cid = r["clip_id"]
            win_idx = int(cid.rsplit("_w", 1)[1]) if "_w" in cid and cid.rsplit("_w", 1)[1].isdigit() else 0
            specs.append(
                {
                    "row": row,
                    "clip_id": cid,
                    "fps": r["fps"],
                    "clip_total": clip_total,
                    "win_idx": win_idx,
                    "temporal_interval": ti,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "crop": (crop_y, crop_x, target_h, target_w),
                    "caption": caption,
                    "caption_key": caption_key,
                    "used_structured_json": used_structured_json,
                }
            )
            e = plan.setdefault(row, {"frames": [], "owners": []})
            lo = len(e["frames"])
            e["frames"].extend(frame_idx)
            e["owners"].append((sp, lo, lo + len(frame_idx)))

        decoded: list[torch.Tensor | None] = [None] * n
        for row, e in plan.items():
            dec = self._decoder(row)
            frames = dec.get_frames_at(indices=e["frames"]).data
            for sp, lo, hi in e["owners"]:
                decoded[sp] = frames[lo:hi]

        results: list[dict[str, Any] | None] = []
        for sp in range(n):
            s = specs[sp]
            if s is None:
                results.append(None)
                continue
            vid = decoded[sp]
            cy, cx, th, tw = s["crop"]
            t = vid.shape[0]
            target_t = (t - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
            vid = vid[:target_t, :, cy : cy + th, cx : cx + tw]
            video = vid.permute(1, 0, 2, 3).contiguous().to(torch.uint8)

            caption = self._finalize_caption(
                s["caption"],
                s["used_structured_json"],
                num_decoded_frames=video.shape[1],
                fps=s["fps"],
                target_h=th,
                target_w=tw,
            )
            text_ids = self._tokenize(caption)
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
                    ai_caption=caption,
                    sampled_caption_style=s["caption_key"],
                    text_token_ids=torch.tensor(text_ids, dtype=torch.long),
                )
            )
        return results

    def _target_size(self, r: dict) -> tuple[int, int]:
        ar = get_aspect_ratio(r["width"], r["height"])
        return VIDEO_RES_SIZE_INFO[self._resolution_str][ar]

    def _window_dict(self, r: dict) -> dict:
        w: dict[str, Any] = {}
        if r.get("caption_json"):
            w["caption_json"] = json.loads(r["caption_json"])
        if r.get("caption"):
            w["caption"] = r["caption"]
        return w


class LanceVisionSFTIterable(torch.utils.data.IterableDataset):
    """Streams clip-windows from LanceVisionSFTDataset with per-(rank, worker) shuffle.

    Mirrors SFTDataset's iterable/self-sharding contract so it drops into the
    training packing stack; adds conditioning_fps to match the SFTDataset sample.
    """

    def __init__(self, dataset: LanceVisionSFTDataset, conditioning_fps: float = 24.0, seed: int = 42):
        super().__init__()
        self._ds = dataset
        self._cond_fps = float(conditioning_fps)
        self._seed = int(seed)
        # Set by RankPartitionedDataLoader; None falls back to torch.distributed
        # (the same contract as the base SFTDataset.__iter__).
        self.shard_world_size = None
        self.shard_rank = None

    def __len__(self) -> int:
        return len(self._ds)

    def _shard(self) -> tuple[int, int]:
        ws, rk = self.shard_world_size, self.shard_rank
        if ws is None or rk is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws, rk = torch.distributed.get_world_size(), torch.distributed.get_rank()
            else:
                ws, rk = 1, 0
        return int(ws), int(rk)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        ws, rk = self._shard()
        shard = rk * nw + wid
        total = max(1, ws * nw)
        n = len(self._ds)
        epoch = 0
        while True:
            g = torch.Generator().manual_seed(self._seed + epoch)
            for i in torch.randperm(n, generator=g).tolist()[shard::total]:
                s = self._ds[i]
                if s is None:  # base contract: skipped sample
                    continue
                s["conditioning_fps"] = self._cond_fps
                yield s
            epoch += 1


def get_lance_vision_sft_dataset(
    *,
    lance_uri: str,
    table: str = "vision_sft",
    resolution: str = "256",
    num_video_frames: int = 16,
    frame_selection_mode: str = "first",
    temporal_interval_mode: str = "entire_chunk",
    tokenizer_config: Any = None,
    cfg_dropout_rate: float = 0.1,
    cfg_dropout_keep_metadata: bool = False,
    caption_suffix: str = "",
    conditioning_fps: float = 24.0,
    conditioning_fps_noise_std: float = 0.0,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    decode_device: str | None = "cpu",
    seed: int = 42,
) -> LanceVisionSFTIterable:
    """Build the iterable Lance vision-SFT dataset for the training packing stack.

    Caption knobs default to the base ``get_sft_dataset`` factory's values."""
    tok = getattr(tokenizer_config, "tokenizer", None) if tokenizer_config is not None else None
    ds = LanceVisionSFTDataset(
        lance_uri,
        table=table,
        resolution=resolution,
        num_video_frames=num_video_frames,
        frame_selection_mode=frame_selection_mode,
        temporal_interval_mode=temporal_interval_mode,
        tokenizer=tok,
        cfg_dropout_rate=cfg_dropout_rate,
        cfg_dropout_keep_metadata=cfg_dropout_keep_metadata,
        caption_suffix=caption_suffix,
        conditioning_fps=conditioning_fps,
        conditioning_fps_noise_std=conditioning_fps_noise_std,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        decode_device=decode_device,
    )
    return LanceVisionSFTIterable(ds, conditioning_fps=conditioning_fps, seed=seed)


__all__ = ["LanceVisionSFTDataset", "LanceVisionSFTIterable", "get_lance_vision_sft_dataset"]
