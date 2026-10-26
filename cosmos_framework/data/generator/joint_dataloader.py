# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
import multiprocessing
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, ClassVar, Dict, Union

import numpy as np
import torch
import webdataset
from torch.utils.data.dataloader import default_collate

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.cost_model.budget import IterationTimeBudget, IterationTimeBudgetConfig
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY
from cosmos_framework.data.generator.sample_token_counter import SampleTokenCounter
from cosmos_framework.data.generator.token_mix_control import TokenMixControlConfig, TokenMixController
from cosmos_framework.data.generator.worker_timing import (
    _TIMING_KEYS as _TIMING_KEYS,
)
from cosmos_framework.data.generator.worker_timing import (
    _aggregate_worker_timing as _aggregate_worker_timing,
)

_BATCH_TIMING_KEYS = {
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
}
_ACTION_SAMPLER_METADATA_KEYS = {
    "action_sampler_family",
    "action_sampler_rank",
    "action_sampler_world_size",
    "action_sampler_worker_id",
    "action_sampler_num_workers",
    "action_sampler_seed",
    "action_sampler_worker_seed",
    "action_sampler_use_deterministic_seed",
    "action_sampler_rank_group",
    "action_sampler_rank_group_rank_start",
    "action_sampler_rank_group_rank_end",
    "action_sampler_rank_group_members",
    "action_sampler_rank_group_fingerprint",
    "action_sampler_draw_count",
    "action_sampler_index",
    "action_sampler_aux_seed",
}
ACTION_SAMPLER_DROPPED_DRAW_COUNT_KEY = "action_sampler_dropped_draw_count"
_DROP_SAMPLE_LOG_FIELDS = (
    "__key__",
    "source_repo_id",
    "action_sampler_family",
    "action_sampler_rank",
    "action_sampler_worker_id",
    "action_sampler_draw_count",
    "action_sampler_index",
    "action_sampler_aux_seed",
    "action_sample_row_id",
    "action_sample_local_start_frame",
    "action_sample_caption_mode",
    "action_sample_fingerprint",
)


def _extend_action_sampler_draw_counts(draw_counts: list[int], sample: Mapping[str, Any]) -> None:
    """Append Lance sampler draw counts from one sample-like dict."""
    value = sample.get("action_sampler_draw_count")
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        raw_values = value.detach().cpu().reshape(-1).tolist()  # [N]
        draw_counts.extend(int(item) for item in raw_values)
    elif isinstance(value, (list, tuple)):
        draw_counts.extend(int(item) for item in value)
    else:
        draw_counts.append(int(value))


def _drop_marker_to_bool(value: Any) -> bool:
    """Normalize one optional drop marker value to bool."""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1).any().item())  # []
    if isinstance(value, (list, tuple)):
        return any(_drop_marker_to_bool(item) for item in value)
    return bool(value)


def _sample_should_drop(sample: Mapping[str, Any]) -> bool:
    """Return whether a sample-like dict is marked for deterministic packer-side drop."""
    return _drop_marker_to_bool(sample.get(DROP_SAMPLE_KEY))


def _format_sample_log_value(value: Any) -> str:
    """Return a compact string for one sample metadata value."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)  # [N]
        if flat.numel() == 0:
            return "[]"
        if flat.numel() == 1:
            return repr(flat.item())
        return repr(flat.tolist())
    if isinstance(value, (list, tuple)):
        return repr([_format_sample_log_value(item) for item in value])
    return repr(value)


def _format_drop_sample_log_fields(sample: Mapping[str, Any], dataset_name: str, drop_count: int) -> str:
    """Return a log payload identifying one dropped sample."""
    reason = sample.get(DROP_SAMPLE_REASON_KEY, "unknown")
    fields = [
        f"count={drop_count}",
        f"stream={dataset_name!r}",
        f"reason={_format_sample_log_value(reason)}",
    ]
    for key in _DROP_SAMPLE_LOG_FIELDS:
        value = sample.get(key)
        if value is not None:
            fields.append(f"{key}={_format_sample_log_value(value)}")
    return " ".join(fields)


def _batch_size_from_collated_batch(batch: dict) -> int:
    """Count the samples in a collated batch.

    Camera batches are sized by their vision stream. The LiDAR-only AV recipe decodes no
    camera at all, so it carries neither ``images`` nor ``video`` and the sweeps are the
    only per-sample list left to count.
    """
    for key in ("images", "video", "lidar"):
        value = batch.get(key)
        if value is not None:
            return len(value)
    raise KeyError("A collated batch needs one of 'images', 'video' or 'lidar' to be split into samples.")


def custom_collate_fn(batch: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """
    Collate like default_collate while preserving ragged media, text tokens, and text metadata as per-sample lists.

    Legacy scalar/dict ``ai_caption`` values retain default_collate behavior;
    only separate-view caption lists are preserved as ragged per-sample values.
    """
    list_collate_keys = {
        "text_token_ids",
        "text_token_lengths",
        "images",
        "video",
        "action",
        "action_raw",
        "domain_id",
        "sequence_plan",
        "sound",
        "raw_action_dim",
        "action_valid_mask",
        "image_size",
        "action_processing_record",
        "camera_geometry",
        # Like "video": a per-sample list of range clips, which default_collate would try to
        # stack even though the two sensors' clips differ in length and resolution.
        "lidar",
        # The BEV pair is [control, target]. Outside this set it goes to default_collate,
        # which transposes it into a list of stacked tensors rather than failing; the
        # per-sample split then yields one 5-D tensor where a list of clips is expected.
        "radar",
        DROP_SAMPLE_KEY,
        DROP_SAMPLE_REASON_KEY,
        *_ACTION_SAMPLER_METADATA_KEYS,
    }

    # Data keys where a per-sample value of ``None`` is a meaningful signal
    # (e.g. audio extraction failed for that sample → ``sound=None`` paired
    # with ``plan.has_sound=False``).  These keys must be kept as a list with
    # ``None`` placeholders so the model can align per-sample data 1:1 with
    # per-sample plans.  Dropping the entire key on any None would leave the
    # remaining sound tensors mis-aligned with the plans whose ``has_sound``
    # flag was set BEFORE collation, causing ``sequence_packing`` to index
    # past the end of ``x0_tokens_sound``.
    sparse_data_keys = {"sound", "camera_geometry"}

    # Handle the case where the batch is already a dictionary (e.g. column-wise batching)
    if isinstance(batch, dict):
        return {
            key: (
                value
                if key in list_collate_keys
                or (key == "ai_caption" and isinstance(value, list) and any(isinstance(item, list) for item in value))
                else default_collate(value)
            )
            for key, value in batch.items()
        }

    # Handle standard list of samples
    elem = batch[0]
    if isinstance(elem, dict):
        # Some Action datasets add optional metadata keys (for example
        # ``additional_view_description`` for concat-view captions) only for a
        # subset of samples.  PyTorch can batch such samples together when
        # DataLoader batch_size > 1; collating only elem's keys and indexing
        # every sample by that key turns the optional field into a fatal
        # KeyError.  Use the union of keys and skip optional keys that are not
        # present in every sample.  Required training keys still fail loudly via
        # downstream assertions if actually missing.
        result = {}
        keys = set().union(*(d.keys() for d in batch))
        for key in keys:
            if key in _TIMING_KEYS:
                continue
            values = [d.get(key) for d in batch]
            if key == DROP_SAMPLE_KEY:
                # Drop markers may be present only on rejected samples. Keep a
                # per-sample bool list so DataLoader(batch_size > 1) can still
                # drop exactly those consumed draws and report them for resume.
                result[key] = [_drop_marker_to_bool(value) for value in values]
                continue
            if key == DROP_SAMPLE_REASON_KEY:
                result[key] = ["" if value is None else str(value) for value in values]
                continue
            if key == "action_processing_record":
                result[key] = values
                continue
            if key == "ai_caption" and any(isinstance(value, list) for value in values):
                result[key] = values
                continue
            if any(value is None for value in values):
                # Sparse data keys keep their None placeholders to preserve
                # 1:1 alignment with sequence_plan.  Other (optional metadata)
                # keys not present in every sample are dropped.
                if key in sparse_data_keys:
                    result[key] = values
                continue
            if key in list_collate_keys:
                result[key] = values
            else:
                result[key] = default_collate(values)
        result.update(_aggregate_worker_timing(batch))
        return result
    else:
        return default_collate(batch)


@dataclass
class _PackingMetrics:
    """Per-batch packing statistics collected during the packing loop.

    Also serves as the single source of truth for packing-related metric names
    via ``STATS_SPEC``, which the packing stats monitor callback consumes to
    drive accumulation and logging.
    """

    current_sequence_length: int = 0
    num_samples: int = 0
    dropped_count: int = 0
    dropped_action_sampler_draw_counts: list[int] = field(default_factory=list)
    from_buffer: int = 0
    from_workers: int = 0
    # Sum of per-sample cost-model seconds over the batch. Stays 0 when no
    # iteration-time budget is configured, since the cost model is not consulted.
    sample_seconds: float = 0.0

    STATS_SPEC: ClassVar[list[tuple[str, str, str]]] = [
        # (batch_key, wandb_suffix, aggregation_type)
        ("_num_tokens", "token_fraction", "scalar"),
        ("_num_samples", "samples_per_batch", "list"),
        ("_from_buffer", "from_buffer", "list"),
        ("_from_workers", "from_workers", "list"),
        ("_buffer_size", "buffer_size", "list"),
        ("_dropped_count", "dropped", "scalar"),
        # Milliseconds rather than seconds because the monitor aggregates these as ints.
        ("_projected_iteration_ms", "projected_iteration_ms", "list"),
    ]

    def attach_to(self, output_batch: dict, buffer_size: int, projected_iteration_sec: float | None = None) -> None:
        """Write packing statistics into the output batch dict.

        Args:
            output_batch: Batch to annotate.
            buffer_size: Samples left over in the look-ahead buffer.
            projected_iteration_sec: Cost-model projection of this batch's
                iteration time, or ``None`` when no budget is configured (in which
                case the metric is left off the batch entirely rather than logged
                as a misleading zero).
        """
        output_batch["_num_tokens"] = self.current_sequence_length
        output_batch["_num_samples"] = self.num_samples
        output_batch["_from_buffer"] = self.from_buffer
        output_batch["_from_workers"] = self.from_workers
        output_batch["_buffer_size"] = buffer_size
        output_batch["_dropped_count"] = self.dropped_count
        if projected_iteration_sec is not None:
            output_batch["_projected_iteration_ms"] = int(round(projected_iteration_sec * 1000))
        if self.dropped_action_sampler_draw_counts:
            output_batch[ACTION_SAMPLER_DROPPED_DRAW_COUNT_KEY] = list(self.dropped_action_sampler_draw_counts)


@dataclass
class _AsyncBatchBuildResult:
    """The outcome of building one packed batch."""

    sequence_id: int
    batch: dict[str, Any] | None = None
    error: BaseException | None = None
    end_of_stream: bool = False


class _AsyncBatchBuilder:
    """Build one requested batch at a time on a persistent daemon thread.

    The caller submits one sequence ID, the daemon advances the existing
    packing iterator once, and the caller receives that exact batch object
    from the result queue. A second request is rejected until the first result
    is consumed, so at most one future batch is built at a time.
    """

    def __init__(self, batch_iterator: Iterator[dict[str, Any]], timeout_s: float) -> None:
        self._batch_iterator: Iterator[dict[str, Any]] = batch_iterator
        self._timeout_s: float = timeout_s
        self._requests: queue.Queue[int | None] = queue.Queue()
        self._results: queue.Queue[_AsyncBatchBuildResult | None] = queue.Queue()
        self._outstanding_sequence_id: int | None = None
        self._closed: bool = False
        self._stream_ended: bool = False
        self._thread: threading.Thread = threading.Thread(target=self._run, name="async-batch-builder", daemon=True)
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def request_batch(self, sequence_id: int) -> None:
        """Ask the daemon thread to build exactly one batch."""
        if self._closed:
            raise RuntimeError("Cannot request a batch after the async batch builder is closed.")
        if self._outstanding_sequence_id is not None:
            raise RuntimeError(
                "Cannot request another batch while async batch building is already in progress "
                f"for sequence_id={self._outstanding_sequence_id}."
            )
        self._outstanding_sequence_id = sequence_id
        self._requests.put(sequence_id)

    def get_batch(self) -> dict[str, Any]:
        """Wait for and return the exact batch object produced for the outstanding request."""
        sequence_id = self._outstanding_sequence_id
        if sequence_id is None:
            if self._stream_ended:
                # Re-entering __iter__ after the packing loop ran dry. The builder cannot
                # restart, and the interesting failure is upstream: a child dataloader
                # yielded nothing, so name it rather than the empty request queue.
                raise RuntimeError(
                    "The packed-batch stream has already ended, so async batch building cannot "
                    "resume. A child dataloader produced no samples on this rank: check its "
                    "earlier warnings for an exhausted shard (a Lance replay slot with no rows "
                    "assigned) or for every sample being dropped or discarded as oversized."
                )
            raise RuntimeError("No async batch building request is outstanding.")
        if self._closed:
            raise RuntimeError("Async batch builder was closed while waiting for a batch.")

        try:
            result = self._results.get(timeout=self._timeout_s)
        except queue.Empty as error:
            producer_alive = self.is_alive
            # A timed-out request may still be running in the daemon thread, so it cannot be
            # safely abandoned or retried. Close the builder and make the timeout terminal.
            self.close()
            raise TimeoutError(
                f"Timed out waiting for async batch building sequence_id={sequence_id}; "
                f"producer_alive={producer_alive}."
            ) from error

        if self._closed or result is None:
            raise RuntimeError("Async batch builder was closed while waiting for a batch.")
        if result.sequence_id != sequence_id:
            raise RuntimeError(
                "Async batch builder returned an unexpected result: "
                f"expected sequence_id={sequence_id}, got sequence_id={result.sequence_id}."
            )
        self._outstanding_sequence_id = None

        if result.error is not None:
            raise RuntimeError(f"Async batch building failed for sequence_id={sequence_id}.") from result.error
        if result.end_of_stream:
            self._stream_ended = True
            raise StopIteration
        if result.batch is None:
            raise RuntimeError(f"Async batch builder returned no batch for sequence_id={sequence_id}.")
        return result.batch

    def _run(self) -> None:
        """Build one batch for each request and publish its result by reference."""
        while True:
            sequence_id = self._requests.get()
            if sequence_id is None or self._closed:
                return

            result = _AsyncBatchBuildResult(sequence_id=sequence_id)
            try:
                result.batch = next(self._batch_iterator)
            except StopIteration:
                result.end_of_stream = True
            except BaseException as error:
                result.error = error

            if self._closed:
                return
            # queue.Queue stores object references within this process. The batch is neither
            # copied nor serialized here; the consumer receives the exact dictionary above.
            self._results.put(result)
            if result.end_of_stream or result.error is not None:
                return

    def close(self) -> None:
        """Permanently close the builder and perform a bounded join of its daemon thread."""
        if not self._closed:
            self._closed = True
            self._requests.put(None)
            self._results.put(None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=min(1.0, self._timeout_s))


class JointDataLoader(SampleTokenCounter, webdataset.WebLoader):
    r"""
    A joint dataloader that supports loading both images and videos.
    """

    _DEFAULT_LOOKAHEAD_LIMIT: ClassVar[int] = 10
    _DROP_SAMPLE_LOG_FIRST_N: ClassVar[int] = 20
    _DROP_SAMPLE_LOG_EVERY_N: ClassVar[int] = 1000

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None,
        max_samples_per_batch: int | None,
        lidar_spatial_compression: Sequence[int] | None = None,
        lidar_temporal_compression_factor: int | None = None,
        radar_spatial_compression: Sequence[int] | None = None,
        radar_temporal_compression_factor: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        prewarm: bool = True,
        prewarm_concurrency: int = 1,
        prewarm_rank_stagger_s: float = 0.0,
        default_lookahead_limit: int = _DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        lazy_initialize_child_iterators: bool = False,
        iteration_time_budget: IterationTimeBudgetConfig | None = None,
        forkserver_preload_modules: list[str] | None = None,
        lidar_patch_spatial_hw: int | tuple[int, int] | None = None,
        radar_patch_spatial_hw: int | tuple[int, int] | None = None,
        enable_async_batch_building: bool = False,
        async_batch_building_timeout_s: float = 1200.0,
        distributed_row_index_cache: bool = True,
    ) -> None:
        """
        Initialize the JointDataLoader with multiple datasets.

        The effective mini-batch size is bounded by max_sequence_length and/or
        max_samples_per_batch. When both are set, both limits apply; at least one
        must be set. This permits single-sample replay with a token ceiling.

        ``iteration_time_budget`` is an independent, additional ceiling: when set, a batch
        grows only while it satisfies both the token limit above and the projected
        iteration time.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": dataloader, "ratio": data_ratio}
            tokenizer_spatial_compression_factor: The spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: The temporal compression factor of the tokenizer.
            patch_spatial: Spatial pathification factor.
            lidar_patch_spatial_hw: LiDAR patch side or (height, width); None inherits patch_spatial.
            radar_patch_spatial_hw: Radar patch side or (height, width); None inherits patch_spatial.
            max_samples_per_batch: Max number of samples per packed batch, independent of the token ceiling.
            lidar_spatial_compression: ``(height, width)`` compression of the LiDAR VAE. Required only
                for streams whose samples carry a ``lidar`` key, whose clips are costed with the
                LiDAR VAE rather than the camera's — the two compress time differently (4x versus
                1x), and an item costed with the wrong factor silently over-packs the batch.
            lidar_temporal_compression_factor: Temporal compression of the LiDAR VAE.
            radar_spatial_compression: ``(height, width)`` compression of the radar VAE. Required only
                for streams whose samples carry a ``radar`` key. Radar cannot borrow the LiDAR pair
                even on a run that carries both sensors: the two VAEs read different grids (a square
                BEV versus a range image) and the recipes pick their tokenizer versions
                independently, so one pair of factors cannot price both streams.
            radar_temporal_compression_factor: Temporal compression of the radar VAE.
            sound_latent_fps: Sound tokenizer latent rate in Hz (e.g. 25). If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz (e.g. 48000). Used with sound_latent_fps to estimate
                sound token count.
            prewarm_concurrency: Number of independent dataloaders to prewarm at
                once. Keep this bounded to avoid an object-store request burst.
            prewarm_rank_stagger_s: Maximum deterministic delay across distributed
                ranks before child iterator startup. This smooths the job-wide
                object-store request ramp without changing stream order.
            distributed_row_index_cache: Prepare image/video/multiview caches with disjoint
                node scans and row-ID redistribution before starting child workers.
            default_lookahead_limit: Packing-loop look-ahead fallback for dataloaders not in
                ``lookahead_limits``.
            lookahead_limits: Optional ``{dataset_name: int}`` per-dataloader override.
            uniae_chunk_frames: Optional UniAE full chunk size, or resolution-keyed chunk sizes.
            uniae_pad_frames: Optional UniAE boundary padding frames per chunk.
            lazy_initialize_child_iterators: If True, instantiate child dataloader objects
                now but defer ``iter(child_dataloader)`` and optional prewarm until
                this joint loader is iterated.  This lets resume logic restore
                dataloader-owned sampler state before PyTorch workers prefetch.
            iteration_time_budget: Optional cost-model ceiling on the projected
                wall-clock time of one iteration. ``None``, or a config whose
                ``iteration_time_target`` is ``None``, packs on tokens alone.
            forkserver_preload_modules: Optional modules imported once by the
                multiprocessing forkserver before it creates child workers.
            enable_async_batch_building: Build one future packed batch on a background
                thread for loaders using the shared synchronous packing iterator.
            async_batch_building_timeout_s: Maximum time to wait for an async batch.

        Example:
            joint_loader = IterativeJointDataLoader(
                dataloaders{
                    "image_data": {
                        "dataloader": webdataset.WebLoader(...),
                        "ratio": 4,
                    },
                    "video_data": {
                        "dataloader": torch.utils.data.DataLoader(...),
                        "ratio": 1,
                    },
                }
            )
        """
        if async_batch_building_timeout_s <= 0:
            raise ValueError(f"async_batch_building_timeout_s must be positive, got {async_batch_building_timeout_s}.")
        # Set this before initializing child iterators.
        self.enable_async_batch_building: bool = enable_async_batch_building
        # Async batch builder state
        self.async_batch_building_timeout_s: float = float(async_batch_building_timeout_s)
        self._async_batch_builder: _AsyncBatchBuilder | None = None
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []
        self.lookahead_limits: list[int] = []
        SampleTokenCounter.__init__(
            self,
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            lidar_spatial_compression=lidar_spatial_compression,
            lidar_temporal_compression_factor=lidar_temporal_compression_factor,
            radar_spatial_compression=radar_spatial_compression,
            radar_temporal_compression_factor=radar_temporal_compression_factor,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            lidar_patch_spatial_hw=lidar_patch_spatial_hw,
            radar_patch_spatial_hw=radar_patch_spatial_hw,
        )
        self.max_sequence_length = max_sequence_length
        self.max_samples_per_batch = max_samples_per_batch
        if prewarm_concurrency < 1:
            raise ValueError(f"prewarm_concurrency must be at least 1, got {prewarm_concurrency}.")
        if not math.isfinite(prewarm_rank_stagger_s) or prewarm_rank_stagger_s < 0:
            raise ValueError(f"prewarm_rank_stagger_s must be finite and nonnegative, got {prewarm_rank_stagger_s}.")
        self.prewarm_concurrency = prewarm_concurrency
        self.prewarm_rank_stagger_s: float = float(prewarm_rank_stagger_s)
        self.forkserver_preload_modules: list[str] = list(forkserver_preload_modules or [])
        self.default_lookahead_limit = int(default_lookahead_limit)
        self.prewarm: bool = bool(prewarm)
        self.distributed_row_index_cache = distributed_row_index_cache
        self.lazy_initialize_child_iterators: bool = bool(lazy_initialize_child_iterators)
        self._child_iterators_initialized: bool = False
        self._drop_sample_log_count: int = 0

        assert self.max_sequence_length is not None or self.max_samples_per_batch is not None, (
            "At least one of max_sequence_length or max_samples_per_batch must be set."
        )

        self.iteration_time_budget: IterationTimeBudget | None = (
            iteration_time_budget.build() if iteration_time_budget is not None else None
        )
        if self.iteration_time_budget is not None:
            log.info(f"JointDataLoader: packing to {self.iteration_time_budget.describe()}")
            self.iteration_time_budget.warn_if_unreachable()

        _lookahead_overrides: Dict[str, int] = dict(lookahead_limits) if lookahead_limits else {}
        unknown = set(_lookahead_overrides) - set(dataloaders)
        assert not unknown, f"lookahead_limits references unknown dataloaders {unknown}; valid: {sorted(dataloaders)}"

        for dataset_name, dataloader_data in dataloaders.items():
            if dataloader_data is None:
                continue
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            if dataloader_data["ratio"] <= 0:
                continue
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"], collate_fn=custom_collate_fn))
            self.data_ratios.append(dataloader_data["ratio"])
            self.lookahead_limits.append(int(_lookahead_overrides.get(dataset_name, self.default_lookahead_limit)))

        self.global_id = 0
        self.ratio_sum = sum(self.data_ratios)

        total = self.ratio_sum if self.ratio_sum > 0 else 1.0
        lines = [f"JointDataLoader: {len(self.dataset_name_list)} streams"]
        for name, ratio in zip(self.dataset_name_list, self.data_ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

        self.data_len = 0
        self.dataloaders: list[Iterator[Any]] = []
        self.buffers: list[deque] = []
        for data in self.dataloader_list:
            self.data_len += len(data)

        if self.lazy_initialize_child_iterators:
            log.info(
                "JointDataLoader: child iterator initialization deferred until first iter(dataloader).",
                rank0_only=False,
            )
        else:
            self._initialize_child_iterators_once()

    def _validate_async_child_startup(self) -> None:
        """Check worker startup even when a synchronous child is wrapped by a hybrid loader."""
        # Keep PyTorch/WebDataset/Lance worker creation outside the packer thread.
        # Workerless loaders can initialize lazily without fetching every stream first.
        if not self.prewarm:
            if any(getattr(loader, "num_workers", None) != 0 for loader in self.dataloader_list):
                raise ValueError("Async batch building without prewarm requires num_workers=0 on every child loader.")

    def _initialize_child_iterators_once(self) -> None:
        """Create child iterators and optionally prewarm them once."""
        if self._child_iterators_initialized:
            return
        if self.enable_async_batch_building:
            self._validate_async_child_startup()
        preload_modules = list(self.forkserver_preload_modules)
        for dataloader in self.dataloader_list:
            preload_modules.extend(getattr(dataloader, "forkserver_preload_modules", []))
        if preload_modules:
            preload_modules = list(dict.fromkeys(["__main__", *preload_modules]))
            multiprocessing.set_forkserver_preload(preload_modules)
            log.info(
                f"JointDataLoader: configured forkserver preload modules: {preload_modules}",
                rank0_only=False,
            )
        if self.prewarm and self.prewarm_rank_stagger_s > 0 and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            delay_s = self.prewarm_rank_stagger_s * rank / world_size
            log.info(
                f"JointDataLoader: delaying child iterator startup by {delay_s:.1f}s "
                f"(rank={rank}/{world_size}, maximum={self.prewarm_rank_stagger_s:.1f}s).",
                rank0_only=False,
            )
            time.sleep(delay_s)
        log.info(
            f"JointDataLoader: initializing {len(self.dataloader_list)} child iterator(s).",
            rank0_only=False,
        )
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        self.buffers = [deque() for _ in range(len(self.dataloader_list))]
        self._child_iterators_initialized = True

        # Pre-warm all dataloaders: force worker process spawning and first
        # batch loading so that slow dataset initialisation (e.g. action
        # datasets with spawn workers) happens here rather than mid-training
        # where it would cause NCCL collective timeouts.
        if self.prewarm:
            self._prewarm_dataloaders()
        else:
            log.info(
                "JointDataLoader: prewarm DISABLED (debug mode); first iteration may incur per-stream cold-load cost"
            )

    def _log_drop_sample(self, sample: Mapping[str, Any], dataset_name: str) -> None:
        """Log a rate-limited audit line for one packer-filtered sample."""
        self._drop_sample_log_count += 1
        drop_count = self._drop_sample_log_count
        should_log = drop_count <= self._DROP_SAMPLE_LOG_FIRST_N or drop_count % self._DROP_SAMPLE_LOG_EVERY_N == 0
        if not should_log:
            return
        log.info(
            "JointDataLoader: filtered drop-sample " + _format_drop_sample_log_fields(sample, dataset_name, drop_count),
            rank0_only=False,
        )

    def _prewarm_dataloader(self, index: int, name: str, dl_iter: Any) -> None:
        """Produce and buffer one batch from a single dataloader."""
        import time

        started_at = time.monotonic()
        try:
            batch = next(dl_iter)
        except StopIteration:
            log.warning(f"Pre-warm: dataloader {name!r} is empty, skipping")
            return
        elapsed = time.monotonic() - started_at

        batch_size = _batch_size_from_collated_batch(batch)

        # Split the collated batch into individual samples and push them
        # into the buffer — identical to the splitting logic in
        # _get_next_sample — so the samples are not wasted.
        for sample_index in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if key in _BATCH_TIMING_KEYS:
                    sample[key] = value
                elif isinstance(value, list) and key in self._MULTI_ITEM_KEYS:
                    elem = value[sample_index]
                    if isinstance(elem, list):
                        sample[key] = elem
                    else:
                        sample[key] = value[sample_index : sample_index + 1]
                elif isinstance(value, list):
                    sample[key] = value[sample_index]
                elif isinstance(value, torch.Tensor) and value.dim() > 0:
                    sample[key] = value[sample_index : sample_index + 1]
                else:
                    sample[key] = value[sample_index : sample_index + 1]
            self.buffers[index].append(sample)

        log.info(
            f"Pre-warm: dataloader {name!r} ready — {batch_size} samples buffered in {elapsed:.1f}s",
            rank0_only=False,
        )

    def _prewarm_dataloaders(self) -> None:
        """Force all dataloader iterators to spawn workers and produce one batch.

        The first ``next()`` call on an ``InfiniteDataLoader`` iterator triggers
        ``DataLoader.__iter__()`` which spawns worker processes.  For action
        dataloaders using ``multiprocessing_context='spawn'``, each worker must
        fully initialise heavy datasets (BridgeOrigLeRobotDataset, embodiment_a, etc.)
        from scratch.  If this happens lazily during training, the resulting
        delay (potentially minutes) causes NCCL collective timeouts when faster
        ranks enter the forward pass while slower ranks are still loading data.

        By pulling one batch from every dataloader here — before any training
        iteration — we ensure all workers are alive and warmed up.  The fetched
        samples are pushed into the per-dataloader buffer so they are consumed
        normally by the first iteration that selects that dataloader.

        A ``dist.barrier()`` at the end synchronises all ranks so that training
        only begins once every rank has finished pre-warming.
        """
        prewarm_items = list(enumerate(zip(self.dataset_name_list, self.dataloaders)))
        if self.prewarm_concurrency == 1 or len(prewarm_items) < 2:
            for index, (name, dl_iter) in prewarm_items:
                self._prewarm_dataloader(index, name, dl_iter)
        else:
            worker_count = min(self.prewarm_concurrency, len(prewarm_items))
            log.info(
                f"Pre-warm: starting {len(prewarm_items)} dataloaders with concurrency={worker_count}",
                rank0_only=False,
            )
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dataloader-prewarm") as executor:
                futures = [
                    executor.submit(self._prewarm_dataloader, index, name, dl_iter)
                    for index, (name, dl_iter) in prewarm_items
                ]
                for future in futures:
                    future.result()

        # Synchronise so training only starts once every rank is warmed up.
        if torch.distributed.is_initialized():
            log.info("Pre-warm: waiting at barrier for all ranks …")
            torch.distributed.barrier()
            log.info("Pre-warm: all ranks ready")

    def _compute_sample_cost(self, data_batch: dict) -> tuple[int, float]:
        """
        Packed length and projected iteration-time cost of one sample.

        Args:
            data_batch (dict): One unpacked sample.

        Returns:
            tuple[int, float]: Token count, and the seconds this sample would add to
            this rank's iteration time. The seconds are 0 when no iteration-time
            budget is configured, in which case the cost model is never consulted.
        """
        und_tokens, gen_tokens = self._compute_token_split_per_sample(data_batch)
        num_tokens = und_tokens + gen_tokens
        if self.iteration_time_budget is None:
            return num_tokens, 0.0
        return num_tokens, self.iteration_time_budget.sample_seconds(und_tokens, gen_tokens)

    def _sample_fits(
        self,
        *,
        num_tokens: int,
        sample_seconds: float,
        packed_tokens: int,
        packed_sample_seconds: float,
        batch_started: bool,
        max_sequence_length: int | None = None,
    ) -> bool:
        """
        Whether one more sample may join the batch under both packing ceilings.

        The token ceiling can reject the very first sample, in which case the caller
        drops it as unpackable. The time ceiling never does: it only stops a batch
        from growing. A target set below the cost of a single sample therefore yields
        one-sample batches rather than an empty stream, which keeps a mis-set target
        from silently starving training.

        Args:
            num_tokens: Candidate's packed length.
            sample_seconds: Candidate's projected cost in seconds.
            packed_tokens: Tokens already in the batch.
            packed_sample_seconds: Projected seconds already in the batch.
            batch_started: Whether the batch already holds at least one sample.

        Returns:
            True when the candidate fits.
        """
        token_limit = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        if token_limit is not None and packed_tokens + num_tokens >= token_limit:
            return False
        if self.iteration_time_budget is not None and batch_started:
            if not self.iteration_time_budget.has_room_for(packed_sample_seconds, sample_seconds):
                return False
        return True

    def _projected_iteration_sec(self, packed_sample_seconds: float) -> float | None:
        """Cost-model projection of a packed batch's iteration time, or None if unbudgeted."""
        if self.iteration_time_budget is None:
            return None
        return self.iteration_time_budget.projected_seconds(packed_sample_seconds)

    # Keys whose value per sample is a list of tensors to be flattened into one list in the batch
    _FLATTEN_LIST_KEYS = {"image_size"}

    def _update_output_batch(self, output_batch: dict, output: dict):
        for key, value in output.items():
            if key in _BATCH_TIMING_KEYS:
                if key not in output_batch:
                    output_batch[key] = value
            elif key in self._FLATTEN_LIST_KEYS and isinstance(value, list):
                if key not in output_batch:
                    output_batch[key] = value
                else:
                    output_batch[key].extend(value)
            elif key not in output_batch:
                output_batch[key] = [value]
            else:
                output_batch[key].append(value)

    def __len__(self) -> int:
        return self.data_len

    # Keys where each sample may hold multiple tensors (e.g. per-view captions
    # or multiple video clips). Nested per-sample lists are preserved; a bare
    # tensor is wrapped so _update_output_batch always yields list[list[Tensor]].
    _MULTI_ITEM_KEYS = {"text_token_ids", "images", "video", "action", "action_raw", "sound"}

    def _get_next_sample(self, index_id: int) -> dict:
        """Pop the next single-sample dict from the buffer for the given dataloader.

        If the buffer is empty, fetches the next collated batch from the inner
        dataloader and splits it into individual samples.

        Splitting rules:
            - Multi-item list values (keys in ``_MULTI_ITEM_KEYS``): a nested
              per-sample list is preserved; a bare element is sliced via
              ``v[i:i+1]`` to yield a single-element list ``[tensor]``.
            - Per-sequence metadata list values (all other list keys, e.g.
              ``sequence_plan``, ``domain_id``): direct-indexed via ``v[i]``
              to yield the bare element.
            - Tensor values ``(B, ...)``: sliced to ``(1, ...)`` via
              ``v[i : i + 1]`` to preserve the batch dimension.

        After ``_update_output_batch`` accumulates samples, the packed output
        batch has the following shapes:
            - Multi-item keys (``text_token_ids``, ``video``, ``images``,
              ``action_raw``, ``action``): ``list[list[Tensor]]`` — each inner
              list contains every item from one sub-sample.
            - Per-sequence metadata keys (``sequence_plan``, ``domain_id``,
              ``dataset_name``, etc.): ``list[element]`` — flat list.
            - Tensor-origin keys: ``list[Tensor(1, ...)]``.

        Args:
            index_id: Index of the dataloader to fetch from.

        Returns:
            A single-sample dictionary.
        """
        buffer = self.buffers[index_id]
        if not buffer:
            try:
                batch = next(self.dataloaders[index_id])
            except StopIteration:
                raise

            batch_size = _batch_size_from_collated_batch(batch)

            for i in range(batch_size):
                sample = {}
                for k, v in batch.items():
                    if k in _BATCH_TIMING_KEYS:
                        sample[k] = v
                    elif isinstance(v, list) and k in self._MULTI_ITEM_KEYS:
                        # For multi-item keys (images, video, etc.), the collated
                        # value is a list with one element per sample.  If the element
                        # is itself a list (e.g. image editing: [src, tgt]), use v[i]
                        # directly to avoid wrapping it in a redundant single-element
                        # list.  Otherwise keep the v[i:i+1] slice so that
                        # _update_output_batch produces list[list[Tensor]].
                        elem = v[i]
                        if isinstance(elem, list):
                            sample[k] = elem
                        else:
                            sample[k] = v[i : i + 1]
                    elif isinstance(v, list):
                        sample[k] = v[i]
                    else:
                        sample[k] = v[i : i + 1]
                buffer.append(sample)

        return buffer.popleft()

    def set_start_iteration(self, iteration: int) -> None:
        """Set the batch sequence and request async work if child iterators already exist."""
        if self._async_batch_builder is not None:
            raise RuntimeError("set_start_iteration must be called before async batch building starts.")
        self.global_id = iteration
        if self.enable_async_batch_building and self._child_iterators_initialized:
            self._start_async_batch_builder()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._initialize_child_iterators_once()
        if not self.enable_async_batch_building:
            return self._iter_synchronous()
        return self._iter_asynchronous()

    def _iter_synchronous(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError("The subclass must implement its packing iterator.")

    def _start_async_batch_builder(self) -> _AsyncBatchBuilder:
        """Create the async builder and request its first batch."""
        if self._async_batch_builder is not None:
            raise RuntimeError("Async batch building has already started.")
        self._initialize_child_iterators_once()
        async_batch_builder = _AsyncBatchBuilder(
            self._iter_synchronous(),
            timeout_s=self.async_batch_building_timeout_s,
        )
        self._async_batch_builder = async_batch_builder
        async_batch_builder.request_batch(self.global_id)
        return async_batch_builder

    def _iter_asynchronous(self) -> Iterator[dict[str, Any]]:
        """Yield each completed batch while the async builder builds its replacement."""
        async_batch_builder = self._async_batch_builder or self._start_async_batch_builder()

        while True:
            try:
                batch = async_batch_builder.get_batch()
            except StopIteration:
                break

            # Request one replacement before yielding. The request queue contains only a
            # sequence ID, and the result queue returns this exact batch object by reference.
            async_batch_builder.request_batch(self.global_id)
            yield batch

    def close(self) -> None:
        """Permanently stop the optional async batch builder."""
        if self._async_batch_builder is not None:
            # Keep the closed builder reference: its bounded join may return while the daemon
            # is still blocked in data loading, so starting a replacement could race with it.
            self._async_batch_builder.close()


class IterativeJointDataLoader(JointDataLoader):
    r"""
    An iterative joint dataloader that supports loading multiple modalities.

    The behavior depends on the ``seed`` parameter:

    - **seed is not None** (Default):
      The modality is randomly selected at each iteration based on the probability distribution
      derived from the ratios. The random state is seeded with ``seed + global_id``, ensuring
      that all ranks select the same modality at the same iteration (assuming synchronized global_id).
      This prevents load imbalance due to mixed resolutions across ranks.

    - **seed is None**:
      The modality selection follows a deterministic round-robin pattern based on the ratios.
      For example, with 2 modalities (image and video) and ratio 2:1:
        - Iterations 0, 1: all ranks process images
        - Iteration 2: all ranks process videos
        - ... and so on.
      This also ensures all ranks process the same modality at the same iteration.

    ``enable_async_batch_building=True`` advances this same batch iterator on
    one daemon thread and builds exactly one future batch. The default keeps
    the original calling-thread behavior.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        lidar_spatial_compression: Sequence[int] | None = None,
        lidar_temporal_compression_factor: int | None = None,
        radar_spatial_compression: Sequence[int] | None = None,
        radar_temporal_compression_factor: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        seed: int | None = 42,
        prewarm: bool = True,
        prewarm_concurrency: int = 1,
        prewarm_rank_stagger_s: float = 0.0,
        default_lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        enable_async_batch_building: bool = False,
        async_batch_building_timeout_s: float = 1200.0,
        packing_limits: dict[str, dict[str, int]] | None = None,
        batch_collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        lazy_initialize_child_iterators: bool = False,
        iteration_time_budget: IterationTimeBudgetConfig | None = None,
        token_mix_control: TokenMixControlConfig | None = None,
        forkserver_preload_modules: list[str] | None = None,
        lidar_patch_spatial_hw: int | tuple[int, int] | None = None,
        radar_patch_spatial_hw: int | tuple[int, int] | None = None,
        distributed_row_index_cache: bool = True,
    ) -> None:
        self.batch_collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = batch_collate_fn
        self.packing_limits: dict[str, dict[str, int]] = {
            name: dict(limits) for name, limits in (packing_limits or {}).items()
        }
        unknown_streams = set(self.packing_limits) - set(dataloaders)
        if unknown_streams:
            raise ValueError(f"packing_limits references unknown streams: {sorted(unknown_streams)}")
        for name, limits in self.packing_limits.items():
            if not limits or set(limits) - {"max_sequence_length", "max_samples_per_batch"}:
                raise ValueError(f"Invalid packing limits for {name}: {limits}")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits.values()):
                raise ValueError(f"Packing limits for {name} must be positive integers: {limits}")
        # A time ceiling skews the token mix here exactly as it does on the random
        # loader, but the correction cannot be applied here: this loader seeds the draw
        # so every rank picks the same modality, and weights estimated from a rank's own
        # batches would differ per rank and desynchronize it. Correcting it would take
        # an all-reduce of the tallies, which the packer's daemon thread makes unsafe.
        # Refused rather than silently skipped, since the skew is otherwise invisible.
        # A single source has no mix to skew, so it is left alone.
        # Checked against the raw dataloaders dict before super().__init__() so an invalid
        # config fails fast instead of first instantiating and prewarming every child loader.
        positive_ratio_count = sum(1 for d in dataloaders.values() if d is not None and d["ratio"] > 0)
        if (
            positive_ratio_count > 1
            and iteration_time_budget is not None
            and (token_mix_control is None or token_mix_control.enabled)
        ):
            raise ValueError(
                "iteration_time_budget pulls each source's share of the tokens away from its configured ratio, by "
                "as much as its tokens per batch differ from the other sources', and IterativeJointDataLoader "
                "cannot correct that without desynchronizing the modality its ranks agree on. Use "
                "RandomJointDataLoader, whose ranks already draw independently, or accept the skew explicitly "
                "with token_mix_control=TokenMixControlConfig(enabled=False)."
            )

        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            lidar_spatial_compression=lidar_spatial_compression,
            lidar_temporal_compression_factor=lidar_temporal_compression_factor,
            radar_spatial_compression=radar_spatial_compression,
            radar_temporal_compression_factor=radar_temporal_compression_factor,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            prewarm=prewarm,
            prewarm_concurrency=prewarm_concurrency,
            prewarm_rank_stagger_s=prewarm_rank_stagger_s,
            distributed_row_index_cache=distributed_row_index_cache,
            default_lookahead_limit=default_lookahead_limit,
            lookahead_limits=lookahead_limits,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            lazy_initialize_child_iterators=lazy_initialize_child_iterators,
            iteration_time_budget=iteration_time_budget,
            forkserver_preload_modules=forkserver_preload_modules,
            lidar_patch_spatial_hw=lidar_patch_spatial_hw,
            radar_patch_spatial_hw=radar_patch_spatial_hw,
            enable_async_batch_building=enable_async_batch_building,
            async_batch_building_timeout_s=async_batch_building_timeout_s,
        )

        self.seed = seed
        # Calculate probabilities for random sampling
        total_ratio = sum(self.data_ratios)
        self.data_probs = np.array([ratio / total_ratio for ratio in self.data_ratios])

    def _iter_synchronous(self) -> Iterator[dict[str, Any]]:
        """Pack batches on the caller or the sole async producer thread."""
        while True:
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + self.global_id)
                index_id = rng.choice(len(self.dataloader_list), p=self.data_probs)
            else:
                data_id = self.global_id % self.ratio_sum
                index_id = self._get_dataloader_index(data_id)

            metrics = _PackingMetrics()
            output_batch = dict()
            samples: list[dict[str, Any]] = []
            skipped_samples = deque()
            lookahead_limit = self.lookahead_limits[index_id]
            lookahead_count = 0

            stream_name = self.dataset_name_list[index_id]
            limits = self.packing_limits.get(stream_name, {})
            token_limit = limits.get("max_sequence_length", self.max_sequence_length)
            sample_limit = limits.get("max_samples_per_batch", self.max_samples_per_batch)

            while True:
                # Check max samples limit first
                if sample_limit is not None and metrics.num_samples >= sample_limit:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if metrics.num_samples > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                if _sample_should_drop(output):
                    # Marked samples are already consumed Lance sampler draws.
                    # Record their draw counts so checkpoint resume advances past them.
                    self._log_drop_sample(output, self.dataset_name_list[index_id])
                    metrics.dropped_count += 1
                    _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, output)
                    continue

                num_tokens_in_current_sample, sample_seconds = self._compute_sample_cost(output)

                if not self._sample_fits(
                    num_tokens=num_tokens_in_current_sample,
                    sample_seconds=sample_seconds,
                    packed_tokens=metrics.current_sequence_length,
                    packed_sample_seconds=metrics.sample_seconds,
                    batch_started=metrics.num_samples > 0,
                    max_sequence_length=token_limit,
                ):
                    if metrics.num_samples == 0:
                        # Only the token limit can reject an empty batch, so this sample does not
                        # fit at any batch size: discard it and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {token_limit}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, output)
                        continue

                    # The batch is non-empty and the selected sample overruns the remaining token
                    # or time budget. Instead of stopping immediately (creating large padding), we
                    # buffer this large sample and try to find a smaller one that fits.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.sample_seconds += sample_seconds
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                if self.batch_collate_fn is not None:
                    samples.append(output)
                else:
                    self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if metrics.num_samples == 0:
                return

            # Heterogeneous pools use the same aligned format as random-mixed
            # loading; ordinary iterative recipes retain their existing format.
            if self.batch_collate_fn is not None:
                output_batch = self.batch_collate_fn(samples)
            metrics.attach_to(
                output_batch,
                buffer_size=len(self.buffers[index_id]),
                projected_iteration_sec=self._projected_iteration_sec(metrics.sample_seconds),
            )
            self.global_id += 1
            yield output_batch

    def _get_dataloader_index(self, data_id: float) -> int:
        """Maps global id to the corresponding dataloader index based on ratio."""
        for i, r in enumerate(self.data_ratios):
            if data_id < r:
                return i
            data_id -= r
        raise ValueError("Invalid data_id")


class RankPartitionedDataLoader:
    """Assigns each rank to exactly one dataset based on ratios.

    For N GPUs with datasets having ratios r_1:r_2:...:r_k, the first
    N * r_1 / sum(r) ranks are assigned dataset 1, the next N * r_2 / sum(r)
    ranks are assigned dataset 2, etc.  Each rank instantiates a single
    PyTorch DataLoader for its assigned dataset.

    The sharding information (``shard_world_size`` and ``shard_rank``) is set
    on each dataset so that it shards data only across ranks that share the
    same dataset, rather than across the full world.

    Example:
        With 128 GPUs and datasets ``{"video": {"dataset": ..., "ratio": 3},
        "image": {"dataset": ..., "ratio": 1}}``:

        - Ranks   0-95  -> video  (shard_world_size=96, shard_rank=0..95)
        - Ranks  96-127 -> image  (shard_world_size=32, shard_rank=0..31)
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ):
        """
        Args:
            datasets: Mapping of dataset name to config dict with keys:

                - ``"dataset"`` (required): a lazy config or dataset instance.
                - ``"ratio"`` (required): positive int weight.
                - ``"dataloader_kwargs"`` (optional): dict of keyword arguments
                  that override the top-level ``**dataloader_kwargs`` for this
                  dataset only (e.g. different ``num_workers`` or ``batch_size``).

            **dataloader_kwargs: Default kwargs forwarded to
                ``torch.utils.data.DataLoader``. ``collate_fn`` defaults to
                ``custom_collate_fn`` if not given.
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        log.info(f"RankPartitionedDataLoader: world_size: {world_size} and rank: {rank}", rank0_only=False)

        _VALID_KEYS = {"dataset", "ratio", "dataloader_kwargs"}
        names: list[str] = []
        dataset_configs: list[Any] = []
        ratios: list[int] = []
        per_dataset_kwargs: list[dict[str, Any]] = []
        for name, cfg in datasets.items():
            extra = set(cfg.keys()) - _VALID_KEYS
            assert not extra, f"Dataset {name!r}: unexpected keys {extra}. Allowed: {_VALID_KEYS}"
            if cfg["ratio"] <= 0:
                log.warning(
                    f"RankPartitionedDataLoader: Skipping dataset {name} with ratio {cfg['ratio']}", rank0_only=False
                )
                continue
            names.append(name)
            dataset_configs.append(cfg["dataset"])
            ratios.append(cfg["ratio"])
            per_dataset_kwargs.append(cfg.get("dataloader_kwargs", {}))

        assert len(names) > 0, "No datasets with positive ratios provided."
        assert world_size >= len(names), (
            f"world_size ({world_size}) must be >= number of datasets ({len(names)}) "
            f"so each dataset gets at least one rank."
        )

        total_ratio = sum(ratios)
        ideal = [r / total_ratio * world_size for r in ratios]
        allocations = [max(1, int(q)) for q in ideal]
        remaining = world_size - sum(allocations)
        if remaining > 0:
            remainders = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
            for j in range(remaining):
                allocations[remainders[j]] += 1
        elif remaining < 0:
            deficit = -remaining
            while deficit > 0:
                best = max(
                    (i for i in range(len(allocations)) if allocations[i] > 1),
                    key=lambda i: (allocations[i] - ideal[i], allocations[i]),
                )
                allocations[best] -= 1
                deficit -= 1

        expected_ratios = [r / total_ratio for r in ratios]
        actual_ratios = [a / world_size for a in allocations]
        lines = [f"RankPartitionedDataLoader allocation ({world_size} GPUs):"]
        start = 0
        for i, (name, alloc) in enumerate(zip(names, allocations)):
            end = start + alloc - 1
            lines.append(
                f"  {name} (ratio {ratios[i]}): ranks {start}-{end} ({alloc} GPUs) "
                f"| expected {expected_ratios[i]:.2%}, actual {actual_ratios[i]:.2%}"
            )
            start += alloc
        log.info("\n".join(lines), rank0_only=False)

        cumulative = 0
        my_dataset_idx = -1
        for i, alloc in enumerate(allocations):
            if rank < cumulative + alloc:
                my_dataset_idx = i
                break
            cumulative += alloc
        assert my_dataset_idx >= 0

        shard_rank = rank - cumulative
        shard_world_size = allocations[my_dataset_idx]

        dataset: Any = instantiate(dataset_configs[my_dataset_idx])
        dataset.shard_world_size = shard_world_size
        dataset.shard_rank = shard_rank
        dataset.shard_id = my_dataset_idx

        merged_kwargs = {**dataloader_kwargs, **per_dataset_kwargs[my_dataset_idx]}
        merged_kwargs.setdefault("collate_fn", custom_collate_fn)
        self.dataloader = torch.utils.data.DataLoader(dataset, **merged_kwargs)
        self.dataset_name = names[my_dataset_idx]
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


class PackingDataLoader(JointDataLoader):
    """Packs multiple samples from a single dataloader into token-budget-constrained batches.

    Unlike the other ``JointDataLoader`` subclasses which manage multiple
    dataloaders with configurable ratios, this class wraps a single dataloader
    and greedily packs consecutive samples until the token budget
    (``max_sequence_length``) or sample count limit (``max_samples_per_batch``)
    is reached.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader | webdataset.WebLoader,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        dataset_name: str = "default",
        lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        iteration_time_budget: IterationTimeBudgetConfig | None = None,
    ):
        """
        Args:
            dataloader: A single dataloader (or lazy config) to draw samples from.
            tokenizer_spatial_compression_factor: Spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: Temporal compression factor of the tokenizer.
            patch_spatial: Spatial patchification factor.
            max_sequence_length: Max total tokens per packed batch, also enforced with a sample limit.
            max_samples_per_batch: Max number of samples per packed batch, also enforced with a token limit.
            sound_latent_fps: Sound tokenizer latent rate in Hz. If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz.
            dataset_name: Name tag attached to every sample in the output batch.
            lookahead_limit: Packing-loop look-ahead for the wrapped dataloader.
            uniae_chunk_frames: Optional UniAE full chunk size, or resolution-keyed chunk sizes.
            uniae_pad_frames: Optional UniAE boundary padding frames per chunk.
            iteration_time_budget: Optional cost-model ceiling on the projected iteration time.
        """
        wrapped = {dataset_name: {"dataloader": dataloader, "ratio": 1}}
        super().__init__(
            dataloaders=wrapped,
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            lookahead_limits={dataset_name: int(lookahead_limit)},
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            iteration_time_budget=iteration_time_budget,
        )

    def __iter__(self):
        inner = self.dataloader_list[0]
        ds_name = getattr(inner, "dataset_name", self.dataset_name_list[0])

        while True:
            current_sequence_length = 0
            packed_sample_seconds = 0.0
            num_samples = 0
            dropped_count = 0
            output_batch: dict = {}
            dropped_action_sampler_draw_counts: list[int] = []

            skipped_samples: deque = deque()
            # PackingDataLoader wraps a single dataloader, so lookahead_limits has one entry.
            lookahead_limit = self.lookahead_limits[0]
            lookahead_count = 0

            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break

                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                try:
                    output = self._get_next_sample(0)
                except StopIteration:
                    break

                if _sample_should_drop(output):
                    # Marked samples are already consumed Lance sampler draws.
                    # Record their draw counts so checkpoint resume advances past them.
                    self._log_drop_sample(output, ds_name)
                    dropped_count += 1
                    _extend_action_sampler_draw_counts(dropped_action_sampler_draw_counts, output)
                    continue

                num_tokens_in_current_sample, sample_seconds = self._compute_sample_cost(output)

                if not self._sample_fits(
                    num_tokens=num_tokens_in_current_sample,
                    sample_seconds=sample_seconds,
                    packed_tokens=current_sequence_length,
                    packed_sample_seconds=packed_sample_seconds,
                    batch_started=len(output_batch) > 0,
                ):
                    if len(output_batch) == 0:
                        # Only the token limit can reject an empty batch, so this sample does not
                        # fit at any batch size: discard it and get the next sample.
                        log.error(
                            f"PackingDataLoader: Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        dropped_count += 1
                        _extend_action_sampler_draw_counts(dropped_action_sampler_draw_counts, output)
                        continue

                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                current_sequence_length += num_tokens_in_current_sample
                packed_sample_seconds += sample_seconds
                num_samples += 1
                # Allows the dataset name to be overridden by the sample itself.
                output["dataset_name"] = output.get("dataset_name", ds_name)
                self._update_output_batch(output_batch, output)

            for sample in reversed(skipped_samples):
                self.buffers[0].appendleft(sample)

            if len(output_batch) == 0:
                return

            # Same key _PackingMetrics.attach_to writes on the other loops, so token
            # consumers (the iter_speed callback, the packing monitor) see this loop too.
            output_batch["_num_tokens"] = current_sequence_length
            output_batch["_dropped_count"] = dropped_count
            projected_iteration_sec = self._projected_iteration_sec(packed_sample_seconds)
            if projected_iteration_sec is not None:
                output_batch["_projected_iteration_ms"] = int(round(projected_iteration_sec * 1000))
            if dropped_action_sampler_draw_counts:
                output_batch[ACTION_SAMPLER_DROPPED_DRAW_COUNT_KEY] = list(dropped_action_sampler_draw_counts)
            self.global_id += 1
            yield output_batch


class RandomJointDataLoader(JointDataLoader):
    r"""
    A random joint dataloader that supports loading multiple modalities with stochastic sampling.

    In this dataloader, the modality is randomly selected at each iteration based on the
    probability distribution derived from the ratios. Each rank independently samples a
    modality, so different ranks may process different modalities at the same iteration.

    For example, with 2 modalities (image and video) and ratio 2:1:
        - Each iteration has 66.7% probability of selecting images
        - Each iteration has 33.3% probability of selecting videos
        - The selection is independent across iterations and ranks

    Note: Unlike IterativeJointDataLoader, this does not guarantee synchronized modality
    selection across ranks.

    When ``iteration_time_budget`` is set, or ``token_mix_control`` is given
    explicitly, the ratios are held as target *token* shares rather than as draw
    probabilities: whatever makes tokens per batch differ by source -- typically a
    time ceiling, but not necessarily -- would otherwise hand an expensive source's
    share to a cheap one. See ``token_mix_control`` for what adapts the draw, and
    :mod:`cosmos_framework.data.generator.token_mix_control` for why that is only
    safe on this loader.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        default_lookahead_limit: int = JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT,
        lookahead_limits: Dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        iteration_time_budget: IterationTimeBudgetConfig | None = None,
        token_mix_control: TokenMixControlConfig | None = None,
        distributed_row_index_cache: bool = True,
    ):
        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            distributed_row_index_cache=distributed_row_index_cache,
            default_lookahead_limit=default_lookahead_limit,
            lookahead_limits=lookahead_limits,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            iteration_time_budget=iteration_time_budget,
        )

        # Convert data ratios to probabilities
        self.data_ratios = np.array([ratio / sum(self.data_ratios) for ratio in self.data_ratios])

        # Only worth running when a time ceiling is in force: that is what makes tokens
        # per batch differ by source, which is what pulls the token mix off the ratios.
        # The ratios then read as the target token share rather than as the draw itself.
        self.token_mix: TokenMixController | None = None
        if self.iteration_time_budget is not None or token_mix_control is not None:
            # A time ceiling is the usual reason tokens per batch differ by source, but not
            # the only possible one, so an explicitly configured token_mix_control is honored
            # on its own -- the caller asked for it, so build it with their parameters even
            # with no time ceiling in force. Only iteration_time_budget on its own falls back
            # to the default config, since there is nothing else to size it from.
            config = token_mix_control if token_mix_control is not None else TokenMixControlConfig()
            if config.enabled:
                self.token_mix = config.build(self.dataset_name_list, self.data_ratios)
        if self.token_mix is not None:
            log.info("RandomJointDataLoader: adapting the draw to hold the token mix on the configured ratios.")

    def __iter__(self):
        while True:
            probs = self.data_ratios if self.token_mix is None else self.token_mix.sampling_probs
            index_id = np.random.choice(len(self.dataloader_list), p=probs)

            metrics = _PackingMetrics()
            output_batch = dict()
            skipped_samples = deque()
            lookahead_limit = self.lookahead_limits[index_id]
            lookahead_count = 0

            while True:
                # Check max samples limit first
                if self.max_samples_per_batch is not None and metrics.num_samples >= self.max_samples_per_batch:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                if _sample_should_drop(output):
                    # Marked samples are already consumed Lance sampler draws.
                    # Record their draw counts so checkpoint resume advances past them.
                    self._log_drop_sample(output, self.dataset_name_list[index_id])
                    metrics.dropped_count += 1
                    _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, output)
                    continue

                num_tokens_in_current_sample, sample_seconds = self._compute_sample_cost(output)

                if not self._sample_fits(
                    num_tokens=num_tokens_in_current_sample,
                    sample_seconds=sample_seconds,
                    packed_tokens=metrics.current_sequence_length,
                    packed_sample_seconds=metrics.sample_seconds,
                    batch_started=len(output_batch) > 0,
                ):
                    if len(output_batch) == 0:
                        # Only the token limit can reject an empty batch, so this sample does not
                        # fit at any batch size: discard it and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, output)
                        continue

                    # The batch is non-empty and the selected sample overruns the remaining token
                    # or time budget. Instead of stopping immediately (creating large padding), we
                    # buffer this large sample and try to find a smaller one that fits.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.sample_seconds += sample_seconds
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if len(output_batch) == 0:
                return

            metrics.attach_to(
                output_batch,
                buffer_size=len(self.buffers[index_id]),
                projected_iteration_sec=self._projected_iteration_sec(metrics.sample_seconds),
            )
            if self.token_mix is not None:
                # The packed length, after every ceiling, look-ahead miss and drop, is
                # exactly the quantity whose split across sources is being held.
                self.token_mix.observe(index_id, metrics.current_sequence_length)
            yield output_batch


@dataclass(frozen=True)
class SampleTokenMixConfig:
    """Estimate source token lengths per sample accepted into a packed batch.

    Ratios become target token shares when enabled. The estimate is updated once
    per accepted sample; deferred candidates do not update the estimate, warmup,
    or feedback until accepted. Cumulative accepted token shares also adjust
    subsequent draw weights toward the target. This feedback is best effort and
    does not restrict fallback.
    ``half_life_samples`` controls only the token-length estimate; feedback counts
    tokens since the loader was created. Both loaders use the shared
    ``TokenMixController`` estimator.
    """

    enabled: bool = True
    half_life_samples: float = 50.0
    warmup_samples_per_source: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_life_samples) or self.half_life_samples <= 0:
            raise ValueError("half_life_samples must be finite and positive.")
        if self.warmup_samples_per_source < 1:
            raise ValueError("warmup_samples_per_source must be positive.")


@dataclass(frozen=True)
class _Candidate:
    source: int
    sample: dict[str, Any]
    tokens: int
    seconds: float


def collate_mixed_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every column aligned and expose ragged vision items under one key.

    Pixel normalization belongs to the model input boundary. Here image and video
    tensors retain their native shapes/dtypes, with a per-sample image flag. Empty
    vision lists represent samples with another modality, such as LiDAR. Optional
    metadata stays aligned even when only one source supplies it.
    """
    if not samples:
        raise ValueError("Cannot collate an empty mixed batch.")
    excluded = {"images", "video", *_BATCH_TIMING_KEYS}
    keys = set().union(*(sample.keys() for sample in samples)) - excluded
    batch = {key: [sample.get(key) for sample in samples] for key in keys}
    batch["vision"] = [sample.get("images", sample.get("video", [])) for sample in samples]
    batch["sample_is_image"] = ["images" in sample for sample in samples]
    return batch


class HybridJointDataLoader:
    """Select one child loader for each rank-owned payload.

    ``context_parallel_size`` defines the repeating CP owner slots. Ranks
    with the same slot in different replica groups select the same child for a
    given fetch, while slots within one group may select different children.
    A size of one synchronizes child selection across every rank, so this
    loader is useful with or without context parallelism.

    ``enable_async_batch_building=True`` selects and builds the next payload
    on one background thread. Keep child batch building synchronous to retain
    only one future packed batch across the entire hybrid loader. Child worker
    initialization and prewarming still happen on the calling thread.
    """

    def __init__(
        self,
        dataloaders: Mapping[str, Mapping[str, Any]],
        context_parallel_size: int = 1,
        seed: int = 42,
        enable_async_batch_building: bool = False,
        async_batch_building_timeout_s: float = 1200.0,
    ) -> None:
        if isinstance(context_parallel_size, bool) or not isinstance(context_parallel_size, int):
            raise TypeError("context_parallel_size must be an integer.")
        if context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive.")
        if async_batch_building_timeout_s <= 0:
            raise ValueError(f"async_batch_building_timeout_s must be positive, got {async_batch_building_timeout_s}.")

        self.dataset_name_list: list[str] = []
        self.dataloader_list: list[Any] = []
        ratios: list[float] = []
        for name, entry in dataloaders.items():
            if set(entry) != {"dataloader", "ratio"}:
                raise ValueError(f"Invalid child config for {name!r}: {entry}")
            ratio = float(entry["ratio"])
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError(f"Child ratio for {name!r} must be finite and nonnegative, got {ratio}.")
            if ratio == 0:
                continue
            self.dataset_name_list.append(str(name))
            self.dataloader_list.append(instantiate(entry["dataloader"]))
            ratios.append(ratio)
        if not self.dataloader_list:
            raise ValueError("At least one child loader must have a positive ratio.")

        total_ratio = sum(ratios)
        self.data_probs: np.ndarray = np.asarray([ratio / total_ratio for ratio in ratios], dtype=np.float64)
        self.context_parallel_size: int = context_parallel_size
        self.seed: int = int(seed)
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # ParallelDims puts overlay axes last, so modulo maps a global rank to
        # its rank-local owner slot for CP and to slot zero when the size is one.
        self.context_parallel_rank: int = global_rank % context_parallel_size
        self.global_id: int = 0
        self._child_iterators: list[Iterator[dict[str, Any]]] = []
        self.enable_async_batch_building: bool = enable_async_batch_building
        self.async_batch_building_timeout_s: float = float(async_batch_building_timeout_s)
        self._async_batch_builder: _AsyncBatchBuilder | None = None

        lines = [
            "HybridJointDataLoader: "
            f"CP slot {self.context_parallel_rank}/{self.context_parallel_size}, "
            f"{len(self.dataset_name_list)} children"
        ]
        for name, probability in zip(self.dataset_name_list, self.data_probs, strict=True):
            lines.append(f"  {name}: {probability:.1%}")
        log.info("\n".join(lines), rank0_only=False)

    def _selected_child_index(self, fetch_index: int) -> int:
        """Return this rank slot's child for one raw fetch."""
        owner_slot = fetch_index * self.context_parallel_size + self.context_parallel_rank
        rng = np.random.RandomState(self.seed + owner_slot)
        return int(rng.choice(len(self.dataloader_list), p=self.data_probs))

    def _initialize_child_iterators_once(self) -> None:
        if self._child_iterators:
            return
        # Initialize and prewarm every child before the first training step.
        # Deferring an unused child until its first draw could stall collectives.
        for dataloader in self.dataloader_list:
            if self.enable_async_batch_building and isinstance(dataloader, JointDataLoader):
                dataloader._validate_async_child_startup()
            initialize = getattr(dataloader, "_initialize_child_iterators_once", None)
            if callable(initialize):
                initialize()
        self._child_iterators = [iter(dataloader) for dataloader in self.dataloader_list]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._initialize_child_iterators_once()
        if not self.enable_async_batch_building:
            return self._iter_synchronous()
        return self._iter_asynchronous()

    def _iter_synchronous(self) -> Iterator[dict[str, Any]]:
        """Select and fetch each payload on the caller or the sole producer thread."""
        while True:
            child_index = self._selected_child_index(self.global_id)
            try:
                batch = next(self._child_iterators[child_index])
            except StopIteration:
                return
            if not isinstance(batch, dict):
                child_name = self.dataset_name_list[child_index]
                raise TypeError(f"Child loader {child_name!r} yielded {type(batch).__name__}; expected dict.")
            self.global_id += 1
            yield batch

    def _iter_asynchronous(self) -> Iterator[dict[str, Any]]:
        """Build one future payload from the next selected child."""
        if self._async_batch_builder is None:
            self._async_batch_builder = _AsyncBatchBuilder(
                self._iter_synchronous(), timeout_s=self.async_batch_building_timeout_s
            )
            self._async_batch_builder.request_batch(self.global_id)
        async_batch_builder = self._async_batch_builder

        while True:
            try:
                batch = async_batch_builder.get_batch()
            except StopIteration:
                return
            async_batch_builder.request_batch(self.global_id)
            yield batch

    def set_start_iteration(self, iteration: int) -> None:
        """Resume the selector and each child at its rank-local fetch count."""
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError("iteration must be an integer.")
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        if self._child_iterators:
            raise RuntimeError("set_start_iteration must be called before child iteration starts.")

        child_iterations = [0] * len(self.dataloader_list)
        for fetch_index in range(iteration):
            child_iterations[self._selected_child_index(fetch_index)] += 1
        for dataloader, child_iteration in zip(self.dataloader_list, child_iterations, strict=True):
            set_start_iteration = getattr(dataloader, "set_start_iteration", None)
            if callable(set_start_iteration):
                set_start_iteration(child_iteration)
        self.global_id = iteration

    def close(self) -> None:
        """Stop background packing and forward cleanup to child loaders."""
        if self._async_batch_builder is not None:
            # Retain the closed builder so another iterator cannot restart it.
            self._async_batch_builder.close()
        for dataloader in self.dataloader_list:
            close = getattr(dataloader, "close", None)
            if callable(close):
                close()


class RandomMixedJointDataLoader(JointDataLoader):
    """Mix sources within each packed batch without altering the legacy loaders.

    Each rank owns an independent RNG. Under a time budget, configured ratios
    target token shares; without it, ratios are sample draw probabilities unless
    sample token-mix control is explicitly enabled. Random draws allow, but do not
    require, several sources in a batch.

    Lookahead counts rejections separately for each source. Once a source reaches
    its limit, candidate draws exclude it for the rest of that batch and continue
    among the remaining sources with renormalized probabilities. All sources get
    a new lookahead allowance in the next batch. This prioritizes filling batches:
    source exclusions and deferred candidates can shift the realized token mix
    away from the configured shares, even with token-mix control enabled.

    Every candidate draw samples a source with the current mixing probabilities,
    then retries that source's oldest deferred candidate before fetching fresh
    data. Deferred candidates retain per-source order, not global priority. The
    first candidate in a batch may exceed the time target; a token limit still
    rejects individually oversized samples using the base contract.
    Finite sources are retired on exhaustion; the remaining sources continue
    with renormalized probabilities until all sources and pending samples finish.

    ``enable_async_batch_building=True`` runs this same packing iterator on one
    background thread, preparing one future CPU batch while training consumes
    the current batch. Source selection, deferred samples and token-mix state
    stay on that single producer thread.
    """

    def __init__(
        self,
        dataloaders: dict[str, Any],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        lidar_spatial_compression: Sequence[int] | None = None,
        lidar_temporal_compression_factor: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        default_lookahead_limit: int = 10,
        lookahead_limits: dict[str, int] | None = None,
        uniae_chunk_frames: int | Mapping[str, int] | None = None,
        uniae_pad_frames: int | None = None,
        iteration_time_budget: IterationTimeBudgetConfig | None = None,
        token_mix_control: SampleTokenMixConfig | None = None,
        seed: int = 42,
        prewarm: bool = True,
        prewarm_concurrency: int = 1,
        prewarm_rank_stagger_s: float = 0.0,
        lazy_initialize_child_iterators: bool = False,
        forkserver_preload_modules: list[str] | None = None,
        enable_async_batch_building: bool = False,
        async_batch_building_timeout_s: float = 1200.0,
        distributed_row_index_cache: bool = True,
        lidar_patch_spatial_hw: int | tuple[int, int] | None = None,
    ) -> None:
        if default_lookahead_limit < 1 or any(value < 1 for value in (lookahead_limits or {}).values()):
            raise ValueError("Lookahead limits must be positive.")
        ratios = [entry["ratio"] for entry in dataloaders.values() if entry is not None]
        if not ratios or not all(math.isfinite(ratio) and ratio >= 0 for ratio in ratios) or not any(ratios):
            raise ValueError("Source ratios must be finite and nonnegative, with at least one positive ratio.")
        super().__init__(
            dataloaders=dataloaders,
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=max_samples_per_batch,
            lidar_spatial_compression=lidar_spatial_compression,
            lidar_temporal_compression_factor=lidar_temporal_compression_factor,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
            prewarm=prewarm,
            prewarm_concurrency=prewarm_concurrency,
            prewarm_rank_stagger_s=prewarm_rank_stagger_s,
            distributed_row_index_cache=distributed_row_index_cache,
            default_lookahead_limit=default_lookahead_limit,
            lookahead_limits=lookahead_limits,
            uniae_chunk_frames=uniae_chunk_frames,
            uniae_pad_frames=uniae_pad_frames,
            lazy_initialize_child_iterators=lazy_initialize_child_iterators,
            iteration_time_budget=iteration_time_budget,
            forkserver_preload_modules=forkserver_preload_modules,
            enable_async_batch_building=enable_async_batch_building,
            async_batch_building_timeout_s=async_batch_building_timeout_s,
            lidar_patch_spatial_hw=lidar_patch_spatial_hw,
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rng: np.random.Generator = np.random.default_rng(np.random.SeedSequence([seed, rank]))
        config = token_mix_control or SampleTokenMixConfig(enabled=self.iteration_time_budget is not None)
        # Legacy batch-named fields count observations; here each is one accepted sample.
        # Sample lengths can differ far more than packed batches, so keep this draw unclipped.
        self.token_mix: TokenMixController = TokenMixController(
            self.dataset_name_list,
            self.data_ratios,
            TokenMixControlConfig(
                enabled=config.enabled,
                half_life_batches=config.half_life_samples,
                warmup_batches_per_source=config.warmup_samples_per_source,
                max_weight_ratio=None,
                feedback=True,
            ),
        )
        self.accepted_tokens: np.ndarray = np.zeros(len(self.dataset_name_list), dtype=np.int64)
        self._active: np.ndarray = np.ones(len(self.dataset_name_list), dtype=bool)
        self._pending: list[deque[_Candidate]] = [deque() for _ in self.dataloader_list]

    def _fresh_candidate(
        self, metrics: _PackingMetrics, worker_timings: list[dict[str, Any]], eligible_sources: np.ndarray
    ) -> _Candidate | None:
        """Draw a source and retry its pending candidates before fetching fresh data."""
        while True:
            available_sources = self._active.copy()
            for source, pending in enumerate(self._pending):
                if pending:
                    available_sources[source] = True
            available_sources &= eligible_sources
            if not np.any(available_sources):
                return None

            source = int(self.rng.choice(len(self.dataloader_list), p=self.token_mix.probabilities(available_sources)))

            pending = self._pending[source]
            if pending:
                # Deferred candidates are bounded per source:
                # pending_count[source] <= lookahead_limits[source].
                # Therefore, total pending candidates <= sum(self.lookahead_limits).
                # Retry the selected source's pending candidates before fresh draws
                # so unvisited candidates cannot accumulate beyond its allowance.
                metrics.from_buffer += 1
                return pending.popleft()

            from_buffer = bool(self.buffers[source])
            try:
                sample = self._get_next_sample(source)
            except StopIteration:
                self._active[source] = False
                continue
            metrics.from_buffer += int(from_buffer)
            metrics.from_workers += int(not from_buffer)
            if not from_buffer:
                worker_timings.append(
                    {
                        "dataset_name": sample.get("task_family", self.dataset_name_list[source]),
                        **{key: sample[key] for key in _BATCH_TIMING_KEYS if key in sample},
                    }
                )
            if _sample_should_drop(sample):
                self._log_drop_sample(sample, sample.get("task_family", self.dataset_name_list[source]))
                metrics.dropped_count += 1
                _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, sample)
                continue

            tokens, seconds = self._compute_sample_cost(sample)
            if tokens <= 0:
                raise ValueError("A packable sample must contain at least one token.")

            # Check whether this sample can fit even in an empty batch. Unlike a
            # temporary overflow in the packing loop, exceeding the hard token
            # ceiling makes it permanently unpackable, so drop rather than defer it.
            # batch_started=False ignores the time target, allowing costly singletons.
            # Exclude these drops from token-mix observations, but retain their sampler
            # draw counts so checkpoint resume can account for discarded draws.
            if not self._sample_fits(
                num_tokens=tokens,
                sample_seconds=seconds,
                packed_tokens=0,
                packed_sample_seconds=0.0,
                batch_started=False,
            ):
                metrics.dropped_count += 1
                _extend_action_sampler_draw_counts(metrics.dropped_action_sampler_draw_counts, sample)
                continue

            return _Candidate(source, sample, tokens, seconds)

    def _iter_synchronous(self) -> Iterator[dict[str, Any]]:
        """Pack source-weighted draws on the caller or the sole async producer thread."""
        while any(self._pending) or np.any(self._active):
            metrics = _PackingMetrics()
            samples: list[dict[str, Any]] = []
            deferred: list[deque[_Candidate]] = [deque() for _ in self.dataloader_list]
            source_stats: dict[str, dict[str, int | float]] = {}
            worker_timings: list[dict[str, Any]] = []
            eligible_sources = np.ones(len(self.dataloader_list), dtype=bool)
            stop_reason = "sample_limit"

            while self.max_samples_per_batch is None or len(samples) < self.max_samples_per_batch:
                candidate = self._fresh_candidate(metrics, worker_timings, eligible_sources)

                if candidate is None:
                    stop_reason = "lookahead" if np.any(~eligible_sources) else "source_exhausted"
                    break

                if not self._sample_fits(
                    num_tokens=candidate.tokens,
                    sample_seconds=candidate.seconds,
                    packed_tokens=metrics.current_sequence_length,
                    packed_sample_seconds=metrics.sample_seconds,
                    batch_started=bool(samples),
                ):
                    deferred[candidate.source].append(candidate)
                    if len(deferred[candidate.source]) >= self.lookahead_limits[candidate.source]:
                        eligible_sources[candidate.source] = False
                    continue

                name = candidate.sample.get("task_family", self.dataset_name_list[candidate.source])
                sample = dict(candidate.sample)
                sample["dataset_name"] = name
                samples.append(sample)
                metrics.num_samples += 1
                metrics.current_sequence_length += candidate.tokens
                metrics.sample_seconds += candidate.seconds
                self.accepted_tokens[candidate.source] += candidate.tokens
                self.token_mix.observe(candidate.source, candidate.tokens)
                stats = source_stats.setdefault(name, {"_num_tokens": 0, "_num_samples": 0, "_sample_seconds": 0.0})
                stats["_num_tokens"] += candidate.tokens
                stats["_num_samples"] += 1
                stats["_sample_seconds"] += candidate.seconds

            # Recheck observed rejections against the completed batch: smaller
            # samples admitted later may have consumed the remaining budgets.
            # Mark a budget as limiting only when increasing it could admit an
            # observed candidate that still fits under the other budget.
            time_limited = False
            token_limited = False
            if (
                samples
                and stop_reason != "sample_limit"
                and self.max_sequence_length is not None
                and self.iteration_time_budget is not None
            ):
                for rejected in chain.from_iterable(deferred):
                    tokens_fit = metrics.current_sequence_length + rejected.tokens < self.max_sequence_length
                    time_fits = self.iteration_time_budget.has_room_for(metrics.sample_seconds, rejected.seconds)
                    if tokens_fit and not time_fits:
                        time_limited = True
                    if not tokens_fit and time_fits:
                        token_limited = True
                    if time_limited and token_limited:
                        break

            # Within each source, pending candidates preceded fresh draws. Put
            # candidates skipped this batch ahead of those we have not visited yet.
            for source, deferred_candidates in enumerate(deferred):
                self._pending[source].extendleft(reversed(deferred_candidates))
            if not samples:
                return

            batch = collate_mixed_samples(samples)
            batch["_source_packing_metrics"] = source_stats
            # One timing record per child batch fetch, including dropped draws.
            # Buffered/deferred samples do not charge their worker a second time.
            batch["_source_worker_timings"] = worker_timings
            projected_iteration_sec = self._projected_iteration_sec(metrics.sample_seconds)
            # Keep the projection unrounded for threshold comparisons in the
            # monitor. This metadata describes the whole packed batch, not any
            # one source's share of it, and does not change admission decisions.
            batch["_packing_budget"] = {
                "token_limit": self.max_sequence_length,
                "time_target_sec": (
                    self.iteration_time_budget.iteration_time_target if self.iteration_time_budget is not None else None
                ),
                "projected_sec": projected_iteration_sec,
                "stop_reason": stop_reason,
                "time_limited": time_limited,
                "token_limited": token_limited,
            }
            metrics.attach_to(
                batch,
                buffer_size=sum(len(pending) for pending in self._pending)
                + sum(len(buffer) for buffer in self.buffers),
                projected_iteration_sec=projected_iteration_sec,
            )
            self.global_id += 1
            yield batch
