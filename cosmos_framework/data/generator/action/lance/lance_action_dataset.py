# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_caption_attribute_adapter import (
    ACTION_CAPTION_ATTRIBUTE_ADAPTER,
    ActionCaptionAttributeValue,
    ViewComposition,
)
from cosmos_framework.data.generator.lance.action_window_index import ActionWindowLocation
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec
from cosmos_framework.data.generator.local_simulation.core import LocalByteRangeBackend as ByteRangeBackend

DEFAULT_SOURCE_METADATA_BLOCK_SIZE: int = 64


DEFAULT_SOURCE_METADATA_CACHE_BLOCKS: int = 4096


DEFAULT_VIDEO_DECODE_THREADS: int = 2


DEFAULT_SLOW_SAMPLE_LOG_THRESHOLD_S: float = 10.0


DEFAULT_DENSE_CAPTION_PROBABILITY: float = 0.8


ACTION_NORMALIZERS_DIR: Path = Path(__file__).parents[1] / "normalizers"


DEFAULT_LANCE_ACTION_MODES: tuple[str, ...] = ("forward_dynamics", "inverse_dynamics", "wam")


FD_ID_LANCE_ACTION_MODES: tuple[str, ...] = ("forward_dynamics", "inverse_dynamics")


TargetFps = int | float | Sequence[int | float] | None


NormalizedTargetFps = float | tuple[float, ...] | None


_FPS_SEED_OFFSET: int = 0x4650535F5354524D  # "FPS_STRM"


@dataclass(frozen=True)
class LanceActionSampleRequest:
    """One resolved Lance Action sample request from the main-process sampler."""

    draw_count: int
    global_index: int
    aux_seed: int
    location: ActionWindowLocation
    source_frame_indices: tuple[int, ...]
    row_ids_by_view: dict[str, int]
    source_version: int
    source_index_version: int
    dataset_length: int
    index_space_fingerprint: str
    frame_count: int
    source_fps: float
    shard_id: str
    episode_index: int
    resolution: str | None = None
    target_fps: float | None = None
    caption: str | None = None
    caption_mode: str = "short"
    dense_caption_action_sample_row_id: int | None = None
    dense_caption_chunk_start_source_frame: int | None = None
    dense_caption_chunk_end_source_frame: int | None = None


class InvalidActionSampleError(RuntimeError):
    """Signal that one deterministic Action anchor is semantically invalid."""


def require_action_caption(value: Any, *, dataset_name: str) -> str:
    """Return a non-blank Action caption or reject the deterministic sample."""
    if value is None:
        raise InvalidActionSampleError(f"{dataset_name} caption is missing.")
    caption = value if isinstance(value, str) else str(value)
    if not caption.strip():
        raise InvalidActionSampleError(f"{dataset_name} caption must contain non-whitespace text.")
    return caption


def resolve_lance_action_mode(mode: str, aux_seed: int) -> str:
    """Resolve deterministic mixed-mode configs to a concrete training objective.

    ``joint`` draws uniformly from forward / inverse / wam. ``fd_id`` draws
    uniformly from forward / inverse only.
    """
    if mode == "fd_id":
        return random.Random(aux_seed).choice(FD_ID_LANCE_ACTION_MODES)
    if mode != "joint":
        return mode
    return random.Random(aux_seed).choice(DEFAULT_LANCE_ACTION_MODES)


@dataclass(frozen=True)
class MaterializedActionWindowTiming:
    """Worker-local time split for materializing one Lance Action window."""

    submit_video_ms: float
    act2_ms: float
    video_wait_ms: float
    total_ms: float


@dataclass(frozen=True)
class MaterializedActionWindow:
    """Storage-neutral contents of one selected source window."""

    videos: dict[str, np.ndarray]
    per_frame_fields: dict[str, np.ndarray]
    per_clip_fields: dict[str, np.ndarray]
    caption: str | None = None
    timing: MaterializedActionWindowTiming | None = None
    target_fps: float | None = None


class BaseLanceActionDataset(Dataset):
    """Materialize Action windows from Lance metadata and ranged object reads."""

    # Dataset adapters override this only after physical gripper calibration.
    fully_open_gripper_width_m: float | None = None

    def get_fully_open_gripper_width_m(self, sample: dict[str, Any] | None = None) -> float | None:
        """Return the physical gripper width for one sample or this dataset."""
        if sample is not None and "fully_open_gripper_width_m" in sample:
            return float(sample["fully_open_gripper_width_m"])
        return self.fully_open_gripper_width_m

    @staticmethod
    def _video_view_tchw(
        window: MaterializedActionWindow,
        view_name: str,
        expected_frames: int,
    ) -> torch.Tensor:
        """Validate and convert one decoded HWC video view to TCHW."""
        source_video = window.videos.get(view_name)
        if source_video is None:
            raise ValueError(f"Materialized window is missing view {view_name!r}.")
        if source_video.ndim != 4 or source_video.shape[0] != expected_frames or source_video.shape[-1] != 3:
            raise ValueError(
                f"Expected video {view_name!r} [T+1,H,W,3] aligned with action fields, got {tuple(source_video.shape)}."
            )
        return torch.from_numpy(np.ascontiguousarray(source_video)).permute(0, 3, 1, 2).contiguous()

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        *,
        view_names: tuple[str, ...],
        chunk_length: int = 16,
        fps: TargetFps = 10,
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        source_dataset_name: str | None = None,
        skip_video_loading: bool = False,
        enable_video_decode_resize: bool = False,
        view_read_workers: int = 2,
        video_decode_threads: int = DEFAULT_VIDEO_DECODE_THREADS,
        metadata_block_size: int = DEFAULT_SOURCE_METADATA_BLOCK_SIZE,
        metadata_cache_blocks: int = DEFAULT_SOURCE_METADATA_CACHE_BLOCKS,
        one_anchor_per_row: bool = False,
        pad_end: bool = False,
        max_subtasks_per_episode: int | None = None,
        retention_quality_score_threshold: float | None = 0.0,
        max_retries: int = 10,
        slow_sample_log_threshold_s: float = DEFAULT_SLOW_SAMPLE_LOG_THRESHOLD_S,
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
        row_index_cache_tag: str | None = None,
        action_caption_dataset_name: str | None = None,
    ) -> None:
        super().__init__()
        if not view_names:
            raise ValueError("view_names must not be empty.")
        if len(set(view_names)) != len(view_names):
            raise ValueError(f"view_names must not contain duplicates, got {view_names}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}.")
        if fps is None:
            fps_options = None
            normalized_fps: NormalizedTargetFps = None
        elif isinstance(fps, Sequence) and not isinstance(fps, (str, bytes)):
            if not fps:
                raise ValueError("fps sequence must not be empty.")
            fps_options = tuple(float(value) for value in fps)
            normalized_fps = fps_options
        else:
            fps_options = (float(fps),)
            normalized_fps = fps_options[0]
        if fps_options is not None:
            invalid_fps = [value for value in fps_options if not math.isfinite(value) or value <= 0]
            if invalid_fps:
                raise ValueError(f"fps values must be finite and positive, got {invalid_fps}.")
        if metadata_block_size < 1:
            raise ValueError(f"metadata_block_size must be positive, got {metadata_block_size}.")
        if metadata_cache_blocks < 1:
            raise ValueError(f"metadata_cache_blocks must be positive, got {metadata_cache_blocks}.")
        if view_read_workers < 1:
            raise ValueError(f"view_read_workers must be positive, got {view_read_workers}.")
        if video_decode_threads < 0:
            raise ValueError(f"video_decode_threads must be non-negative, got {video_decode_threads}.")
        slow_sample_log_threshold_s = float(slow_sample_log_threshold_s)
        if slow_sample_log_threshold_s < 0:
            raise ValueError(f"slow_sample_log_threshold_s must be non-negative, got {slow_sample_log_threshold_s}.")
        dense_caption_probability = float(dense_caption_probability)
        if not 0.0 <= dense_caption_probability <= 1.0:
            raise ValueError(f"dense_caption_probability must be in [0.0, 1.0], got {dense_caption_probability}.")

        self.lance_dataset_spec: LanceActionDatasetSpec = lance_dataset_spec
        self.creds_path: Path = Path(creds_path)
        self.view_names: tuple[str, ...] = view_names
        self.chunk_length: int = chunk_length
        self.fps: NormalizedTargetFps = normalized_fps
        self.split: str = split.lower().strip()
        self.index_fps: float | None = (
            None if fps_options is None else min(fps_options) if self.split in {"train", "full"} else fps_options[0]
        )
        self.val_ratio: float = val_ratio
        self.seed: int = seed
        self.skip_video_loading: bool = skip_video_loading
        self.enable_video_decode_resize: bool = enable_video_decode_resize
        self.view_read_workers: int = view_read_workers
        self.video_decode_threads: int = video_decode_threads
        self.metadata_block_size: int = metadata_block_size
        self.metadata_cache_blocks: int = metadata_cache_blocks
        self.max_retries: int = max_retries
        self.pad_end: bool = pad_end
        self.one_anchor_per_row: bool = one_anchor_per_row
        self.retention_quality_score_threshold: float | None = (
            None if retention_quality_score_threshold is None else float(retention_quality_score_threshold)
        )
        self.slow_sample_log_threshold_s: float = slow_sample_log_threshold_s
        self.dense_caption_probability: float = dense_caption_probability
        self.max_subtasks_per_episode: int | None = max_subtasks_per_episode
        self.action_caption_dataset_name: str = action_caption_dataset_name or lance_dataset_spec.name
        self.source_dataset_name: str = self._resolve_source_dataset_name(lance_dataset_spec, source_dataset_name)
        self.row_index_cache_tag = row_index_cache_tag

    @staticmethod
    def _resolve_source_dataset_name(spec: LanceActionDatasetSpec, source_dataset_name: str | None) -> str:
        """Resolve the dataset-specific Action cache selected by this loader."""
        if spec.combine_source_datasets:
            if source_dataset_name not in (None, spec.name):
                raise ValueError(
                    f"Combined Action spec {spec.name!r} selects cache {spec.name!r}, "
                    f"got source_dataset_name={source_dataset_name!r}."
                )
            return spec.name
        if source_dataset_name is not None:
            source_datasets = tuple(spec.source_datasets)
            if source_datasets and source_dataset_name not in source_datasets:
                raise ValueError(
                    f"source_dataset_name={source_dataset_name!r} is not listed in "
                    f"lance_dataset_spec.source_datasets={source_datasets!r}."
                )
            return source_dataset_name
        if len(spec.source_datasets) != 1:
            raise ValueError("source_dataset_name is required unless the Action spec selects exactly one dataset.")
        return spec.source_datasets[0]

    def _resolve_target_fps(self, source_fps: float, aux_seed: int) -> float:
        """Resolve one deterministic effective FPS without upsampling.

        For multiple configured rates, training samples one rate per draw while
        validation and inference consistently use the first configured rate.
        """
        fps = self.fps
        if fps is None:
            return source_fps
        if isinstance(fps, tuple):
            configured_fps = (
                random.Random(aux_seed ^ _FPS_SEED_OFFSET).choice(fps) if self.split in {"train", "full"} else fps[0]
            )
        else:
            configured_fps = fps
        return min(configured_fps, source_fps)

    def _window_fps(self, window: MaterializedActionWindow) -> float:
        """Return the effective FPS carried by a materialized window."""
        if window.target_fps is not None:
            return window.target_fps
        fps = self.fps
        if not isinstance(fps, (int, float)):
            raise RuntimeError("Materialized Action window is missing its resolved target FPS.")
        return float(fps)

    def _validate_source_row(
        self,
        request: LanceActionSampleRequest,
        source_row: dict[str, Any],
        view_name: str,
    ) -> None:
        """Detect cache/source drift before reading media objects."""
        action_spans = self.lance_dataset_spec.action_spans
        source_view_name = str(source_row[action_spans.view_name])
        if source_view_name != view_name:
            raise ValueError(f"Source row view is {source_view_name!r}, expected {view_name!r}.")
        if int(source_row[action_spans.frame_count]) != int(request.frame_count):
            raise ValueError(f"Source row frame count for view {view_name!r} does not match the row-index cache.")
        if float(source_row[action_spans.source_fps]) != float(request.source_fps):
            raise ValueError(f"Source row FPS for view {view_name!r} does not match the row-index cache.")
        if int(source_row[action_spans.episode_index]) != int(request.episode_index):
            raise ValueError("Source row episode index does not match the row-index cache.")

    def _video_decode_resize_size(self, view_row: dict[str, Any], resolution: str | None) -> tuple[int, int] | None:
        """Return optional TorchCodec resize size as ``(height, width)`` for one source video."""
        if not self.enable_video_decode_resize or resolution is None:
            return None
        action_spans = self.lance_dataset_spec.action_spans
        source_width = int(view_row[action_spans.width])
        source_height = int(view_row[action_spans.height])
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"Source video dimensions must be positive, got width={source_width}, height={source_height}."
            )
        from cosmos_framework.data.generator.action.utils.transforms import find_closest_target_size

        target_width, target_height = find_closest_target_size(source_height, source_width, resolution)
        scaling_ratio = min(target_width / source_width, target_height / source_height, 1.0)
        return (
            int(scaling_ratio * source_height + 0.5),
            int(scaling_ratio * source_width + 0.5),
        )

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Build one dataset-specific trainer-facing item."""
        raise NotImplementedError

    def _build_training_item_from_request(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Build one dataset-specific item from an already resolved sample request."""
        return self._build_training_item(request.global_index, request.aux_seed, source_rows, window)

    def _sample_key(self, global_index: int, source_version: int) -> str:
        """Return a cheap stable fallback key for sample-level audit logs."""
        return f"{self.source_dataset_name}:v{source_version}:idx{global_index}"

    def _load_resolved_item(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]] | None = None,
        object_backend: ByteRangeBackend | None = None,
    ) -> dict[str, Any]:
        """Decode captured local RGB/ACT2; never open the metadata's source URI.

        Export has already resolved the anchor, caption, timing and source rows.
        The backend is a closed URI-to-local-file map. Decode the full local MP4
        with the same TorchCodec reader and source-frame selection as the online
        range reader; neither Lance nor object-store clients are instantiated.
        """
        from cosmos_framework.data.generator.action.utils.act2_range_reader import read_act2_window
        from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader

        if source_rows is None or not isinstance(object_backend, ByteRangeBackend):
            raise ValueError("Bimanual materialization requires captured rows and a local byte-range backend.")
        if request.caption_mode != "short":
            raise ValueError("The bundle must contain the final, already-resolved caption.")
        if set(source_rows) != set(self.view_names):
            raise ValueError("Captured RGB views must match the selected dataset views exactly.")
        frame_indices = np.asarray(request.source_frame_indices, dtype=np.int64)
        if frame_indices.ndim != 1 or frame_indices.size < 2:
            raise ValueError("An Action window requires at least two source observations.")
        for view_name, row in source_rows.items():
            self._validate_source_row(request, row, view_name)
        spans = self.lance_dataset_spec.action_spans
        primary_row = source_rows[self.view_names[0]]
        action_window = read_act2_window(
            object_backend,
            str(primary_row[spans.action_url]),
            frame_indices,
            source_dataset=str(primary_row[spans.source_dataset]),
            num_frames=request.frame_count,
        )
        videos: dict[str, np.ndarray] = {}
        if not self.skip_video_loading:
            for view_name, row in source_rows.items():
                payload = object_backend.get(str(row[spans.clip_url]))
                if len(payload) != int(row[spans.nb_bytes]):
                    raise ValueError(f"Local RGB byte count does not match captured metadata for {view_name!r}.")
                reader = TorchCodecVideoReader(
                    payload,
                    num_threads=self.video_decode_threads,
                    resize_size=self._video_decode_resize_size(row, request.resolution),
                )
                if reader.metadata.num_frames != request.frame_count:
                    raise ValueError(f"Local RGB frame count does not match captured metadata for {view_name!r}.")
                videos[view_name] = reader.get_frames_nhwc_uint8(frame_indices.tolist())
        window = MaterializedActionWindow(
            videos=videos,
            per_frame_fields=action_window.per_frame,
            per_clip_fields=action_window.per_clip,
            caption=request.caption,
            target_fps=request.target_fps,
        )
        item = self._build_training_item_from_request(request, source_rows, window)
        if ACTION_CAPTION_ATTRIBUTE_ADAPTER.supports(self.action_caption_dataset_name):
            item["action_caption_attributes"] = self._resolve_action_caption_attributes(item, window)
        return item

    def _action_caption_view_composition(self, item: dict[str, Any]) -> ViewComposition | None:
        """Optionally specialize dataset-level caption semantics for one sample."""
        del item
        return None

    def _resolve_action_caption_attributes(
        self,
        item: dict[str, Any],
        window: MaterializedActionWindow,
    ) -> dict[str, ActionCaptionAttributeValue]:
        """Resolve caption metadata from the centralized dataset protocol."""
        logical_video_frames = item.get("video_num_frames")
        video = item.get("video")
        action = item.get("action")
        if logical_video_frames is not None:
            observation_count = int(logical_video_frames)
        elif video is not None and getattr(video, "ndim", 0) == 4:
            observation_count = int(video.shape[1])
        elif action is not None and getattr(action, "ndim", 0) >= 1:
            observation_count = int(action.shape[0]) + 1
        else:
            observation_count = self.chunk_length + 1
        return ACTION_CAPTION_ATTRIBUTE_ADAPTER.resolve(
            self.action_caption_dataset_name,
            fps=self._window_fps(window),
            observation_count=observation_count,
            view_count=len(self.view_names),
            view_composition=self._action_caption_view_composition(item),
        )

    def materialize_local_sample(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]],
        backend: ByteRangeBackend,
    ) -> dict[str, Any]:
        """Materialize captured metadata through a filesystem-only byte-range backend."""
        if request.caption_mode != "short":
            raise ValueError("Offline Action records must include a resolved short caption.")
        item = self._load_resolved_item(request, source_rows, backend)
        item["__key__"] = self._sample_key(request.global_index, request.source_version)
        item["source_dataset_name"] = self.source_dataset_name
        item["action_sample_caption_mode"] = request.caption_mode
        return item
