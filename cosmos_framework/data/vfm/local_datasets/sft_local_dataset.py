# SPDX-License-Identifier: OpenMDW-1.1
"""Local, map-style vision-SFT dataset — a faithful representative of ``SFTDataset``.

The shipped :class:`~cosmos_framework.data.vfm.local_datasets.sft_dataset.SFTDataset`
is an ``IterableDataset`` that streams video bytes + caption JSONL from S3, packs
sequences, and shards across ranks. For a dataloader benchmark we want a *map-style*
loader over a fixed local subset so the base path and the Lance path read the exact
same samples by index, with no S3/packing/sharding in the way.

This class reproduces the **per-sample work** of ``SFTDataset.process_one_sample``
verbatim — the part that actually costs CPU and that the Lance loader must match:

  * resolution sizing from :data:`VIDEO_RES_SIZE_INFO` (resize-ratio + center-crop),
  * the ``entire_chunk`` temporal-interval / frame-selection window math,
  * ``ffmpeg_decode_video`` full-clip decode + temporal subsample,
  * temporal truncation to ``compression_factor * N + 1``,
  * caption selection (``caption_json`` preferred -> ``caption_json_to_prompt``),
  * tokenization via the cosmos ``tokenize_caption`` + ``add_special_tokens``.

It reads the same ``video_dataset_file.jsonl`` the official
``captions_to_sft_jsonl`` converter produces, with each ``vision_path`` resolved
relative to the JSONL's directory. ``frame_selection_mode="first"`` and
``cfg_dropout_rate=0`` are used so per-sample output is deterministic, which is
what the equivalence check against the Lance loader needs.

Output dict per sample:
  ``video`` uint8 (C, T, H, W), ``text_token_ids`` LongTensor, plus the SFT
  metadata fields (``ai_caption``, ``num_frames``, ``image_size``, ...).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import torch

from cosmos_framework.data.vfm.local_datasets.helper import (
    ffmpeg_decode_video,
    get_video_metadata,
)
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY, caption_json_to_prompt

_MAX_CAPTION_TOKENS = 1024


def _get_aspect_ratio(width: int, height: int) -> str:
    """Same bucket boundaries as ``helper.get_aspect_ratio`` (kept local so the
    converter's stored ``width``/``height`` map to the same output size)."""
    ratio = width / height
    if ratio < 0.65:
        return "9,16"
    elif ratio < 0.88:
        return "3,4"
    elif ratio < 1.16:
        return "1,1"
    elif ratio < 1.55:
        return "4,3"
    return "16,9"


def select_caption(t2w_window: dict) -> tuple[str, str, bool]:
    """Mirror of ``sft_dataset._select_caption`` for the deterministic-default
    keys present in this dataset.

    Priority: ``caption_json`` (structured, serialised verbatim) -> ``caption``
    (dense). Returns ``(caption_key, caption_text, used_structured_json)``."""
    if CAPTION_JSON_KEY in t2w_window:
        raw = t2w_window[CAPTION_JSON_KEY]
        if isinstance(raw, dict):
            return CAPTION_JSON_KEY, caption_json_to_prompt(raw), True
        return CAPTION_JSON_KEY, str(raw).strip(), True
    raw = t2w_window["caption"]
    return "caption", raw.strip().rstrip(".") + ".", False


class LocalSFTDataset(torch.utils.data.Dataset):
    """Map-style local stand-in for ``SFTDataset`` (one window per sample)."""

    def __init__(
        self,
        jsonl_path: str,
        *,
        num_video_frames: int = 16,
        resolution: str = "256",
        temporal_interval_mode: str = "entire_chunk",
        frame_selection_mode: str = "first",
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "Qwen/Qwen2.5-7B",
        use_system_prompt: bool = False,
        max_caption_tokens: int = _MAX_CAPTION_TOKENS,
        temporal_compression_factor: int = 4,
        ffmpeg_threads: int = 2,
    ) -> None:
        assert temporal_interval_mode in ("force_one", "max_30fps", "entire_chunk")
        assert frame_selection_mode in ("center", "first", "random")
        assert resolution in VIDEO_RES_SIZE_INFO
        self.jsonl_path = jsonl_path
        self.num_video_frames = num_video_frames
        self.resolution = resolution
        self.temporal_interval_mode = temporal_interval_mode
        self.frame_selection_mode = frame_selection_mode
        self.use_system_prompt = use_system_prompt
        self.max_caption_tokens = max_caption_tokens
        self.temporal_compression_factor = temporal_compression_factor
        self.ffmpeg_threads = ffmpeg_threads
        self.output_sizes = VIDEO_RES_SIZE_INFO[resolution]
        self.tokenizer_name = tokenizer_name
        self._base_dir = os.path.dirname(os.path.abspath(jsonl_path))

        # one sample == one (video, window) pair (sample_by_window semantics)
        self.metadata: list[dict] = []
        with open(jsonl_path) as fh:
            for line in fh:
                rec = json.loads(line)
                for win_idx, window in enumerate(rec["t2w_windows"]):
                    self.metadata.append({**rec, "win_idx": win_idx, "window": window})

        self._tokenizer = tokenizer  # may be None -> built lazily (worker-safe)

    # ── worker-safe lazy tokenizer ───────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            from cosmos_framework.data.vfm.sequence_packing import add_special_tokens

            tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tok, _ = add_special_tokens(tok)
            self._tokenizer = tok
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_path(self, vision_path: str) -> str:
        if "://" in vision_path or vision_path.startswith("/"):
            return vision_path
        return os.path.join(self._base_dir, vision_path)

    skip_tokenize: bool = False  # benchmark raw-video mode toggle (picklable)

    def _tokenize(self, caption: str) -> list[int]:
        if self.skip_tokenize:
            return []
        from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption

        ids = tokenize_caption(
            caption, self._ensure_tokenizer(), is_video=True, use_system_prompt=self.use_system_prompt
        )
        return ids[: self.max_caption_tokens]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.metadata[idx]
        window = meta["window"]
        window_start = window["start_frame"]
        window_end = window["end_frame"]

        # output resolution (resize-ratio + center crop) — identical to SFTDataset
        input_w, input_h = meta["width"], meta["height"]
        aspect_ratio = _get_aspect_ratio(input_w, input_h)
        target_w, target_h = self.output_sizes[aspect_ratio]
        resize_ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))
        crop_y, crop_x = (round((resize_h - target_h) / 2), round((resize_w - target_w) / 2))

        video_path = self._resolve_path(meta["vision_path"])
        video_info = get_video_metadata(video_path)
        original_fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        actual_end = min(window_end, total_frames - 1)
        frames_in_window = actual_end - window_start + 1

        if self.num_video_frames == -1:
            temporal_interval = window["temporal_interval"]
            start_frame = window_start
            end_frame = actual_end
        else:
            if frames_in_window < self.num_video_frames:
                raise ValueError(f"Not enough frames in window for {meta['uuid']}")
            if self.temporal_interval_mode == "force_one":
                temporal_interval = 1
            elif self.temporal_interval_mode == "max_30fps":
                temporal_interval = max(1, int(original_fps / 30.0))
            else:  # entire_chunk
                temporal_interval = max(1, frames_in_window // self.num_video_frames)
            num_frames_before_downsample = (self.num_video_frames - 1) * temporal_interval + 1
            if self.frame_selection_mode == "first":
                start_frame = window_start
            elif self.frame_selection_mode == "center":
                start_frame = window_start + (frames_in_window - num_frames_before_downsample) // 2
            else:  # random
                import random

                max_offset = frames_in_window - num_frames_before_downsample
                start_frame = window_start + random.randint(0, max(0, max_offset))
            end_frame = start_frame + num_frames_before_downsample - 1

        video_chunk = []
        for fidx, frame in enumerate(
            ffmpeg_decode_video(video_path, scale_hw=(resize_h, resize_w), num_threads=self.ffmpeg_threads)
        ):
            if fidx < start_frame:
                continue
            elif fidx <= end_frame:
                if (fidx - start_frame) % temporal_interval == 0:
                    video_chunk.append(frame)
            else:
                break

        if not video_chunk:
            raise ValueError(f"No frames decoded for {meta['uuid']}")

        video_chunk = np.stack(video_chunk, axis=0)  # [T,H,W,3]
        target_t = (video_chunk.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
        video_chunk = video_chunk[:target_t, crop_y : crop_y + target_h, crop_x : crop_x + target_w]
        video_chunk = np.transpose(video_chunk, (3, 0, 1, 2))  # [3,T,H,W]
        video = torch.from_numpy(np.ascontiguousarray(video_chunk)).to(torch.uint8)

        image_size = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32)
        padding_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)

        caption_key, caption, _used_json = select_caption(window)
        text_ids = self._tokenize(caption)

        return dict(
            __key__=f"{meta['uuid']}_w{meta['win_idx']}",
            __url__=video_path,
            fps=original_fps,
            n_orig_video_frames=total_frames,
            chunk_index=meta["win_idx"],
            frame_start=start_frame,
            frame_end=end_frame,
            num_frames=video.shape[1],
            video=video,
            num_multiplier=temporal_interval,
            padding_mask=padding_mask,
            image_size=image_size,
            ai_caption=caption,
            sampled_caption_style=caption_key,
            text_token_ids=torch.tensor(text_ids, dtype=torch.long),
        )


__all__ = ["LocalSFTDataset", "select_caption"]
