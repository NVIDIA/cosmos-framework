# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Finite, deterministic Reasoner documents derived from the generator SFT dataset."""

import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import boto3

from cosmos_framework.data.generator.local_datasets.helper import (
    client_config,
    download_from_s3,
    ffmpeg_decode_video,
    get_video_metadata,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    SFTWindowFraming,
    enumerate_sft_captions,
    enumerate_sft_cfg_variants,
    render_sft_caption,
    resolve_sft_window_framing,
    sft_metadata_sort_key,
)
from cosmos_framework.utils import log

SFTVideoInfoResolver = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class SFTReasonerDocument:
    """One tokenized SFT prompt variant before BOS/EOS/mRoPE sequence packing."""

    sample_key: str
    vision_path: str
    window_index: int
    caption_key: str
    used_structured_json: bool
    cfg_dropped: bool
    caption: str
    text_token_ids: tuple[int, ...]
    conditioning_fps: float
    framing: SFTWindowFraming


class SFTVideoInfoProbe:
    """Resolve ffprobe metadata and the actual decoded frame count used by SFT."""

    def __init__(
        self,
        s3_credentials: Mapping[str, Any],
        output_sizes: Mapping[str, tuple[int, int]] | None = None,
    ):
        self._s3_client = boto3.client("s3", **dict(s3_credentials), config=client_config)
        self._output_sizes = output_sizes

    def __call__(self, metadata: dict[str, Any]) -> Mapping[str, Any]:
        vision_path = metadata["vision_path"]
        video_bytes = download_from_s3(self._s3_client, vision_path)
        if video_bytes is None:
            raise RuntimeError(f"Failed to download video while resolving SFT framing: {vision_path}")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_input:
            tmp_input.write(video_bytes)
            tmp_input.flush()
            video_info = get_video_metadata(tmp_input.name)
            scale_hw = None
            if self._output_sizes is not None:
                target_width, target_height = self._output_sizes[metadata["aspect_ratio"]]
                input_width, input_height = metadata["width"], metadata["height"]
                resize_ratio = max(target_width / input_width, target_height / input_height)
                scale_hw = round(input_height * resize_ratio), round(input_width * resize_ratio)
            decoded_total_frames = sum(1 for _ in ffmpeg_decode_video(tmp_input.name, scale_hw=scale_hw, num_threads=2))
            return {**video_info, "decoded_total_frames": decoded_total_frames}


def iter_sft_reasoner_documents(
    dataset: SFTDataset,
    *,
    video_info_resolver: SFTVideoInfoResolver | None = None,
) -> Iterator[SFTReasonerDocument]:
    """Enumerate every reachable Nano SFT single-caption document exactly once.

    Unlike ``SFTDataset.__iter__``, this function is finite and does not repeat,
    shuffle, randomly choose a window, or sample CFG. Metadata uses the same
    stable UUID-hash ordering as ``get_sft_dataset``; windows, caption choices,
    and reachable CFG states are expanded in deterministic order.

    A custom ``video_info_resolver`` must return ``fps``, ffprobe
    ``total_frames``, and the independently measured ``decoded_total_frames``.
    The default resolver downloads each unique video once and follows the same
    scaled ffmpeg decode path as training so duration text remains byte-identical
    when ffprobe overestimates a damaged or variable-frame-rate source.

    Positive conditioning-FPS noise is deliberately unsupported because it
    creates an unbounded set of text prompts. Fixed-length random frame
    selection is also rejected: the current cache identity has no frame-offset
    variant, while Nano's standard native-window recipe does not need one.
    """

    if dataset.conditioning_fps_noise_std > 0:
        raise ValueError(
            "Offline SFT Reasoner document enumeration requires conditioning_fps_noise_std=0; "
            f"got {dataset.conditioning_fps_noise_std}"
        )
    if dataset.num_video_frames != -1 and dataset.frame_selection_mode == "random":
        raise ValueError(
            "Offline SFT Reasoner document enumeration does not support random fixed-length frame selection"
        )

    if getattr(dataset, "is_initialized", False):
        raise ValueError(
            "Reasoner documents must be enumerated before SFTDataset.__iter__ mutates, pads, and shards metadata"
        )

    resolve_video_info = (
        video_info_resolver
        if video_info_resolver is not None
        else SFTVideoInfoProbe(dataset.s3_credentials, dataset.output_sizes)
    )
    cfg_variants = enumerate_sft_cfg_variants(dataset.cfg_dropout_rate)
    video_info_cache: dict[str, Mapping[str, Any]] = {}

    for metadata in sorted(dataset.metadata, key=sft_metadata_sort_key):
        vision_path = metadata["vision_path"]
        if vision_path not in video_info_cache:
            video_info_cache[vision_path] = resolve_video_info(metadata)
        video_info = video_info_cache[vision_path]
        original_fps = float(video_info["fps"])
        total_frames = int(video_info["total_frames"])
        if "decoded_total_frames" not in video_info:
            raise ValueError(
                "SFT video_info_resolver must report decoded_total_frames so offline caption framing "
                f"matches the training decoder: {vision_path}"
            )
        decoded_total_frames = int(video_info["decoded_total_frames"])
        target_width, target_height = dataset.output_sizes[metadata["aspect_ratio"]]

        for window_index, t2w_window in enumerate(metadata["t2w_windows"]):
            framing = resolve_sft_window_framing(
                metadata,
                window_index,
                original_fps=original_fps,
                total_frames=total_frames,
                decoded_total_frames=decoded_total_frames,
                num_video_frames=dataset.num_video_frames,
                temporal_interval_mode=dataset.temporal_interval_mode,
                frame_selection_mode=dataset.frame_selection_mode,
                temporal_compression_factor=dataset.temporal_compression_factor,
                target_height=target_height,
                target_width=target_width,
            )
            if framing is None:
                log.warning(f"Skipping empty or too-short SFT window during Reasoner enumeration: {metadata['uuid']}")
                continue

            caption_variants = enumerate_sft_captions(t2w_window)
            if not caption_variants:
                log.warning(
                    f"Skipping SFT window with no selectable caption during Reasoner enumeration: {framing.sample_key}"
                )
                continue

            effective_fps = original_fps / framing.temporal_interval
            conditioning_fps = float(effective_fps if dataset.conditioning_fps < 0 else dataset.conditioning_fps)
            for caption_key, base_caption, used_structured_json in caption_variants:
                for cfg_dropped in cfg_variants:
                    caption = render_sft_caption(
                        base_caption,
                        used_structured_json=used_structured_json,
                        cfg_dropped=cfg_dropped,
                        cfg_dropout_keep_metadata=dataset.cfg_dropout_keep_metadata,
                        caption_suffix=dataset.caption_suffix,
                        append_duration_fps_timestamps=dataset.append_duration_fps_timestamps,
                        append_resolution_info=dataset.append_resolution_info,
                        num_frames=framing.num_frames,
                        conditioning_fps=conditioning_fps,
                        target_height=target_height,
                        target_width=target_width,
                    )
                    text_ids, caption = dataset._tokenize_caption(caption)
                    yield SFTReasonerDocument(
                        sample_key=framing.sample_key,
                        vision_path=vision_path,
                        window_index=window_index,
                        caption_key=caption_key,
                        used_structured_json=used_structured_json,
                        cfg_dropped=cfg_dropped,
                        caption=caption,
                        text_token_ids=tuple(int(token_id) for token_id in text_ids),
                        conditioning_fps=conditioning_fps,
                        framing=framing,
                    )
