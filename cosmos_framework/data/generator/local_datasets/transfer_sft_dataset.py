# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transfer SFT dataset: (control, target) pairs for Cosmos Transfer post-training.

Subclasses SFTDataset — all frame loading, rank partitioning, and iteration
logic is inherited.  Only ``process_one_sample()`` is overridden to add the
control signal and repackage the output as ``video=[control, target]``.

Supported control types
-----------------------
- ``edge``:  Canny edge map, computed on-the-fly.  Recommended starting point.
- ``blur``:  Gaussian blur, computed on-the-fly.
- ``depth``: Precomputed depth video at ``metadata["control_path"]``.
- ``seg``:   Precomputed segmentation video at ``metadata["control_path"]``.

Workflow
--------
1. Run cosmos-curator with ``--upload-clip-info-in-chunks``.
2. Convert to transfer JSONL::

       python -m cosmos_framework.scripts.curator_to_sft_jsonl \\
           --curator-output outputs/curator_split/ \\
           -o outputs/transfer_sft.jsonl \\
           --control-type edge

3. Wire into an experiment config inheriting from exp302 (video) or exp301 (image)::

       from cosmos_framework.data.generator.local_datasets.transfer_sft_dataset import (
           get_transfer_sft_dataset,
       )
       dataset=L(get_transfer_sft_dataset)(
           jsonl_paths="s3://your-bucket/transfer_sft.jsonl",
           resolution="480",
           num_video_frames=61,
           control_type="edge",
           tokenizer_config=...,
       )
"""

import gzip
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import boto3
import cv2
import numpy as np
import torch

from cosmos_framework.data.generator.local_datasets.helper import (
    client_config,
    download_from_s3,
    ffmpeg_decode_video,
    get_aspect_ratio,
    parse_s3_url,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import SFTDataset
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils import log
from cosmos_framework.utils.flags import INTERNAL

_ON_THE_FLY_CONTROLS = frozenset({"edge", "blur"})
_PRECOMPUTED_CONTROLS = frozenset({"depth", "seg"})
_MAX_VIDEO_DURATION_S = 61.0
_MIN_WINDOW_FRAMES = 61

# Canny threshold presets matching AddControlInputEdge.
_EDGE_THRESHOLDS = [
    (20, 50),  # very_low
    (50, 100),  # low
    (100, 200),  # medium
    (200, 300),  # high
    (300, 400),  # very_high
]


def _compute_edge(frames: np.ndarray) -> np.ndarray:
    """Canny edge map from uint8 [T, H, W, 3] RGB → uint8 [T, H, W, 3]."""
    low, high = random.choice(_EDGE_THRESHOLDS)
    edges = np.zeros_like(frames)
    for t in range(frames.shape[0]):
        gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, low, high)
        edges[t] = np.stack([edge, edge, edge], axis=-1)
    return edges


def _compute_blur(frames: np.ndarray) -> np.ndarray:
    """Gaussian blur from uint8 [T, H, W, 3] → uint8 [T, H, W, 3]."""
    ksize = random.choice([11, 21, 31, 41])
    blurred = np.zeros_like(frames)
    for t in range(frames.shape[0]):
        blurred[t] = cv2.GaussianBlur(frames[t], (ksize, ksize), 0)
    return blurred


def _normalize(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] → float32 [3, T, H, W] in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).float()  # [T, H, W, 3]
    return (x.permute(3, 0, 1, 2) / 255.0 - 0.5) / 0.5  # [3, T, H, W]


class TransferSFTDataset(SFTDataset):
    """SFT dataset for Cosmos Transfer post-training.

    Inherits all frame loading, rank partitioning, and iteration from
    ``SFTDataset``.  Overrides ``process_one_sample()`` to compute or load a
    control signal and return ``video=[control, target]`` in the format
    expected by the Transfer training loop.
    """

    def __init__(
        self,
        *args,
        control_type: str = "edge",
        **kwargs,
    ):
        assert control_type in _ON_THE_FLY_CONTROLS | _PRECOMPUTED_CONTROLS, f"Unknown control_type={control_type!r}"
        super().__init__(*args, **kwargs)
        self.control_type = control_type

    def _load_precomputed_control(
        self,
        control_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        resize_hw: tuple[int, int],
        crop_y: int,
        crop_x: int,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Decode a precomputed control video and extract the matching frame window."""
        ctrl_bytes = download_from_s3(self.s3_client, control_path)
        if ctrl_bytes is None:
            raise RuntimeError(f"Failed to read control video: {control_path}")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(ctrl_bytes)
            tmp.flush()
            frames = list(ffmpeg_decode_video(tmp.name, scale_hw=resize_hw, num_threads=2))
        ctrl = np.stack(frames, axis=0)  # [T_full, H, W, 3]
        chunk = []
        for idx in range(ctrl.shape[0]):
            if idx < start_frame:
                continue
            if idx > end_frame:
                break
            if (idx - start_frame) % temporal_interval == 0:
                chunk.append(ctrl[idx])
        if not chunk:
            return np.empty((0, target_h, target_w, 3), dtype=np.uint8)
        arr = np.stack(chunk, axis=0)
        return arr[:, crop_y : crop_y + target_h, crop_x : crop_x + target_w]

    def process_one_sample(self, metadata: dict) -> dict | None:
        sample = super().process_one_sample(metadata)
        if sample is None:
            return None

        # Parent returns video as uint8 [3, T, H, W].  Convert to [T, H, W, 3]
        # numpy for on-the-fly control computation, then normalize both streams.
        video_uint8 = sample["video"]  # [3, T, H, W] uint8
        target_np = video_uint8.permute(1, 2, 3, 0).numpy()  # [T, H, W, 3] uint8
        T, H, W = target_np.shape[:3]

        if self.control_type == "edge":
            control_np = _compute_edge(target_np)
        elif self.control_type == "blur":
            control_np = _compute_blur(target_np)
        else:
            ctrl_path = metadata.get("control_path")
            if not ctrl_path:
                log.warning(f"No control_path for {metadata['uuid']}, skipping")
                return None
            # Re-derive the resize / crop parameters to align the control video.
            input_w, input_h = metadata["width"], metadata["height"]
            target_w_out, target_h_out = self.output_sizes[metadata["aspect_ratio"]]
            resize_ratio = max(target_w_out / input_w, target_h_out / input_h)
            resize_h = round(input_h * resize_ratio)
            resize_w = round(input_w * resize_ratio)
            crop_y = round((resize_h - target_h_out) / 2)
            crop_x = round((resize_w - target_w_out) / 2)
            # Derive end_frame from T (post-truncation) rather than sample["frame_end"]
            # (pre-truncation). The parent truncates to target_t = (N-1)//tcf*tcf+1
            # frames, which may be less than N when (num_video_frames-1) % tcf != 0.
            # Using frame_end directly would request more frames than T from the
            # control video, causing the shape-mismatch check below to drop every
            # depth/seg sample silently for non-standard frame counts.
            effective_end_frame = sample["frame_start"] + (T - 1) * sample["num_multiplier"]
            try:
                control_np = self._load_precomputed_control(
                    ctrl_path,
                    start_frame=sample["frame_start"],
                    end_frame=effective_end_frame,
                    temporal_interval=sample["num_multiplier"],
                    resize_hw=(resize_h, resize_w),
                    crop_y=crop_y,
                    crop_x=crop_x,
                    target_h=target_h_out,
                    target_w=target_w_out,
                )
            except Exception as exc:
                log.warning(f"Failed to load control for {metadata['uuid']}: {exc}")
                return None
            if control_np.shape[0] == 0 or control_np.shape[0] != T:
                log.warning(f"Control/target frame count mismatch for {metadata['uuid']}: {control_np.shape[0]} vs {T}")
                return None

        control_tensor = _normalize(control_np)  # [3, T, H, W] float32 [-1, 1]
        target_tensor = _normalize(target_np)  # [3, T, H, W] float32 [-1, 1]

        sample["video"] = [control_tensor, target_tensor]
        sample["dataset_name"] = "video_transfer"
        sample["selected_caption_type"] = "transfer_caption"

        # Always provide a SequencePlan with shared temporal positions.
        cond_indexes = sample["sequence_plan"].condition_frame_indexes_vision if "sequence_plan" in sample else []
        sample["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=cond_indexes,
            share_vision_temporal_positions=True,
        )

        # Two vision streams → wrap image_size as a list.
        sample["image_size"] = [sample["image_size"], sample["image_size"]]

        return sample


def _load_transfer_metadata(
    s3_client: Any,
    jsonl_url: str,
    min_frames: int | None,
    max_duration_s: float | None = _MAX_VIDEO_DURATION_S,
    uuid_prefix: str = "",
    min_short_edge: int = 0,
    control_type: str = "edge",
) -> list[dict]:
    """Load transfer SFT metadata, passing control_path through for depth/seg."""
    log.info(f"Loading transfer SFT metadata from {jsonl_url}", rank0_only=False)
    metadata_list: list[dict] = []
    num_raw = 0

    with io.BytesIO() as buffer:
        if jsonl_url.startswith("s3://"):
            bucket, key = parse_s3_url(jsonl_url)
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        else:
            path = Path(jsonl_url).absolute()
            jsonl_url = str(path)
            buffer.write(path.read_bytes())
        buffer.seek(0)

        line_iter = gzip.open(buffer, "rb") if jsonl_url.endswith(".gz") else buffer
        for line in line_iter:
            num_raw += 1
            record = json.loads(line.decode("utf-8"))
            uuid = f"{uuid_prefix}{record['uuid']}" if uuid_prefix else record["uuid"]

            if max_duration_s is not None and record["duration"] > max_duration_s:
                continue
            if min_short_edge > 0 and min(record["width"], record["height"]) < min_short_edge:
                continue
            if control_type in _PRECOMPUTED_CONTROLS and not record.get("control_path"):
                continue

            kept_windows = [
                w
                for w in (record.get("t2w_windows") or [])
                if min_frames is None or w["end_frame"] - w["start_frame"] + 1 >= min_frames
            ]
            if not kept_windows:
                continue

            vision_path = record["vision_path"]
            if "://" not in vision_path and not vision_path.startswith("/"):
                vision_path = f"{os.path.dirname(jsonl_url)}/{vision_path}"

            entry: dict[str, Any] = {
                "uuid": uuid,
                "vision_path": vision_path,
                "width": record["width"],
                "height": record["height"],
                "nb_frames": record.get("nb_frames"),
                "framerate": record.get("framerate"),
                "aspect_ratio": get_aspect_ratio(record["width"], record["height"]),
                "t2w_windows": kept_windows,
            }
            if record.get("control_path"):
                ctrl_path = record["control_path"]
                if "://" not in ctrl_path and not ctrl_path.startswith("/"):
                    ctrl_path = f"{os.path.dirname(jsonl_url)}/{ctrl_path}"
                entry["control_path"] = ctrl_path
            metadata_list.append(entry)

    log.info(
        f"Transfer SFT: kept {len(metadata_list)} / {num_raw} records from {jsonl_url}",
        rank0_only=False,
    )
    return metadata_list


def get_transfer_sft_dataset(
    jsonl_paths: str | list[str],
    resolution: str = "480",
    num_video_frames: int = 61,
    control_type: str = "edge",
    temporal_interval_mode: str = "entire_chunk",
    frame_selection_mode: str = "center",
    tokenizer_config: Optional[Any] = None,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    min_short_edge: int = 0,
    min_window_frames: int | None = _MIN_WINDOW_FRAMES,
    max_duration_s: float | None = _MAX_VIDEO_DURATION_S,
    conditioning_config: dict[int, float] | None = None,
    temporal_compression_factor: int = 4,
    **kwargs,
) -> TransferSFTDataset:
    """Create a TransferSFTDataset from one or more JSONL files.

    Args:
        jsonl_paths: Local path(s) or ``s3://`` URI(s) to transfer SFT JSONL(s)
            produced by ``curator_to_sft_jsonl.py --control-type <type>``.
            Multiple files are concatenated; uuids are prefixed to avoid collisions.
        resolution: Output resolution bucket, e.g. ``"480"`` or ``"720"``.
        num_video_frames: Frames per clip (≥ 61 to pass the converter hard filter).
        control_type: ``"edge"`` | ``"blur"`` | ``"depth"`` | ``"seg"``.
            Edge and blur are computed on-the-fly; depth and seg require
            ``control_path`` entries in the JSONL.
        temporal_interval_mode: ``"force_one"`` | ``"max_30fps"`` | ``"entire_chunk"``.
        frame_selection_mode: ``"center"`` | ``"first"`` | ``"random"``.
        tokenizer_config: Tokenizer config (same as ``get_sft_dataset``).
        cfg_dropout_rate: Caption dropout rate for CFG training.
        use_system_prompt: Include system prompt in tokenization.
        append_duration_fps_timestamps: Append duration/FPS text to captions.
        append_resolution_info: Append resolution text to captions.
        min_short_edge: Drop videos whose shortest edge is below this value.
        min_window_frames: Metadata pre-filter for t2w_window length.  Default
            61 matches historical behavior.  Set to None for short-task datasets.
        max_duration_s: Metadata pre-filter for video duration.  Default 61.0
            matches historical behavior.  Set to None for short-task datasets.
        conditioning_config: I2V conditioning frame distribution, e.g.
            ``{0: 0.7, 1: 0.2, 2: 0.1}`` (70% unconditioned).
        temporal_compression_factor: VAE temporal compression factor (default 4).
    """
    log.info(f"get_transfer_sft_dataset: ignoring unknown kwargs: {list(kwargs)}")
    assert resolution in VIDEO_RES_SIZE_INFO, f"Unknown resolution {resolution!r}; known: {sorted(VIDEO_RES_SIZE_INFO)}"

    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]

    if INTERNAL:
        with open("credentials/gcs.secret") as f:
            credentials = json.load(f)
    else:
        credentials = {}

    s3_client = boto3.client("s3", **credentials, config=client_config)

    metadata_list: list[dict] = []
    for idx, jsonl_url in enumerate(jsonl_paths):
        prefix = f"{idx}/" if len(jsonl_paths) > 1 else ""
        metadata_list.extend(
            _load_transfer_metadata(
                s3_client,
                jsonl_url,
                min_frames=min_window_frames,
                max_duration_s=max_duration_s,
                uuid_prefix=prefix,
                min_short_edge=min_short_edge,
                control_type=control_type,
            )
        )

    total_windows = sum(len(m["t2w_windows"]) for m in metadata_list)
    log.info(
        f"Transfer SFT: {len(metadata_list)} videos, {total_windows} windows, "
        f"control_type={control_type}, resolution={resolution}"
    )

    return TransferSFTDataset(
        metadata=metadata_list,
        num_video_frames=num_video_frames,
        resolution=resolution,
        s3_credentials=credentials,
        control_type=control_type,
        temporal_interval_mode=temporal_interval_mode,
        frame_selection_mode=frame_selection_mode,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        use_system_prompt=use_system_prompt,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        conditioning_config=conditioning_config,
        temporal_compression_factor=temporal_compression_factor,
    )
