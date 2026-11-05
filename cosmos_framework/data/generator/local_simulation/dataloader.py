# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in, infinite training loader for encoded local simulation bundles."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict
from itertools import islice
from typing import Any

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_framework.utils import log
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader, custom_collate_fn
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.data.generator.local_simulation.resume import (
    LOCAL_SIMULATION_RESUME_KEY,
    consume_local_resume_state,
)


class _ResumeDistributedSampler(DistributedSampler):
    """Skip consumed indices before fetching data or prefetching worker batches."""

    start_index: int = 0

    def _indices(self) -> Iterator[int]:
        return super().__iter__()

    def __iter__(self) -> Iterator[int]:
        return islice(self._indices(), self.start_index, None)


class _StratumDistributedSampler(_ResumeDistributedSampler):
    """Keep Transfer length buckets aligned across ranks, padding within buckets."""

    _groups: list[list[int]]

    def __init__(self, dataset: LocalSimulationDataset, **kwargs: Any) -> None:
        super().__init__(dataset, **kwargs)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            groups[record["stratum"]].append(index)
        self._groups = list(groups.values())

    def __len__(self) -> int:
        return sum(math.ceil(len(group) / self.num_replicas) for group in self._groups)

    def _indices(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        groups = [group.copy() for group in self._groups]
        if self.shuffle:
            rng.shuffle(groups)
        for indices in groups:
            if self.shuffle:
                rng.shuffle(indices)
            size = math.ceil(len(indices) / self.num_replicas) * self.num_replicas
            indices = (indices * math.ceil(size / len(indices)))[:size]
            yield from indices[self.rank :: self.num_replicas]


class LocalSimulationDataLoader(DataLoader[dict[str, Any]]):
    """Cycle a local bundle, reshuffling and refreshing augmentations each epoch.

    Workers are recreated each epoch so they see ``dataset.set_epoch``. This
    loader is constructed only by explicitly selected local configs; it never
    redirects a production loader or resolves a source catalog.
    """

    _local_dataset: LocalSimulationDataset
    _local_sampler: _ResumeDistributedSampler
    _epoch: int
    _stream_iterator: Iterator[dict[str, Any]] | None

    def __init__(
        self,
        dataset: LocalSimulationDataset,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 1,
        pin_memory: bool = False,
        shuffle: bool = True,
        group_by_stratum: bool = False,
        seed: int = 42,
        rank: int | None = None,
        world_size: int | None = None,
        collate_fn: Callable[..., Any] = custom_collate_fn,
    ) -> None:
        if not len(dataset):
            raise ValueError("Local training requires a nonempty bundle.")
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._local_dataset = dataset
        sampler_type = _StratumDistributedSampler if group_by_stratum else _ResumeDistributedSampler
        self._local_sampler = sampler_type(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False
        )
        self._epoch = 0
        self._stream_iterator = None
        worker_options: dict[str, Any] = (
            {"prefetch_factor": prefetch_factor, "multiprocessing_context": "spawn"} if num_workers else {}
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=self._local_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_fn=collate_fn,
            **worker_options,
        )

    def _iterate_epochs(self) -> Iterator[dict[str, Any]]:
        while True:
            self._local_dataset.set_epoch(self._epoch)
            self._local_sampler.set_epoch(self._epoch)
            yield from super().__iter__()
            self._epoch += 1
            self._local_sampler.start_index = 0

    def set_start_iteration(self, iteration: int) -> None:
        """Seek to a consumed child-batch count before creating its iterator."""
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        if self._stream_iterator is not None:
            raise RuntimeError("Local stream position must be restored before creating its iterator.")
        self._epoch, batch_offset = divmod(iteration, len(self))
        assert self.batch_size is not None
        self._local_sampler.start_index = batch_offset * self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._stream_iterator is None:
            self._stream_iterator = self._iterate_epochs()
        return self._stream_iterator


class LocalSimulationJointDataLoader(IterativeJointDataLoader):
    """Restore local child positions through the trainer's fetch-count hook.

    Fixed sample-count packing seeks each child without decoding its prefix.
    Token/time ceilings and larger child batches require deterministic packing
    replay, including drops and lookahead buffers, before yielding resumed data.
    """

    _resume_replay_batches: int
    _fetch_count: int
    _fetch_origin: int
    restart_data_iterator_on_handoff: bool
    _stream_fetch_counts: list[int]
    _resume_identity: dict[str, Any]

    def __init__(self, *, restart_data_iterator_on_handoff: bool = False, **kwargs: Any) -> None:
        self._resume_replay_batches = 0
        # Resume must position children before prewarming or spawning workers.
        kwargs["lazy_initialize_child_iterators"] = True
        super().__init__(**kwargs)
        if any(not isinstance(loader, LocalSimulationDataLoader) for loader in self.dataloader_list):
            raise TypeError("LocalSimulationJointDataLoader requires local simulation children.")
        self._fetch_count = 0
        self._fetch_origin = 0
        self.restart_data_iterator_on_handoff = restart_data_iterator_on_handoff
        self._stream_fetch_counts = [0] * len(self.dataloader_list)
        self._resume_identity = {
            "streams": self.dataset_name_list,
            "ratios": self.data_ratios,
            "seed": self.seed,
            "packing_limits": self.packing_limits,
            "max_samples_per_batch": self.max_samples_per_batch,
            "max_sequence_length": self.max_sequence_length,
            # Compare effective settings: per-stream lookahead overrides can
            # mask the default, and token costing determines batch boundaries.
            "lookahead_limits": self.lookahead_limits,
            "token_cost": {
                "tokenizer_spatial_compression_factor": self.tokenizer_spatial_compression_factor,
                "tokenizer_temporal_compression_factor": self.tokenizer_temporal_compression_factor,
                "patch_spatial": self.patch_spatial,
                "uniae_chunk_frames": self.uniae_chunk_frames,
                "uniae_pad_frames": self.uniae_pad_frames,
                "lidar_spatial_compression": self.lidar_spatial_compression,
                "lidar_temporal_compression_factor": self.lidar_temporal_compression_factor,
                "lidar_patch_spatial_hw": self.lidar_patch_spatial_hw,
                "radar_spatial_compression": self.radar_spatial_compression,
                "radar_temporal_compression_factor": self.radar_temporal_compression_factor,
                "radar_patch_spatial_hw": self.radar_patch_spatial_hw,
                "sound_latent_fps": self.sound_latent_fps,
                "audio_sample_rate": self.audio_sample_rate,
            },
            "iteration_time_budget": asdict(self.iteration_time_budget) if self.iteration_time_budget else None,
            "children": [self._child_identity(loader) for loader in self.dataloader_list],
        }
        if restart_data_iterator_on_handoff:
            self._resume_identity["restart_data_iterator_on_handoff"] = True

    @staticmethod
    def _child_identity(loader: Any) -> dict[str, Any]:
        assert isinstance(loader, LocalSimulationDataLoader)
        dataset = loader._local_dataset
        sampler = loader._local_sampler
        bundle = json.dumps([dataset.manifest, dataset.records], sort_keys=True).encode()
        identity: dict[str, Any] = {
            "bundle_digest": hashlib.sha256(bundle).hexdigest(),
            "augmentation_seed": dataset.seed,
            "transform_overrides": dataset.transform_overrides,
            "sampler_seed": sampler.seed,
            "sampler_kind": type(sampler).__name__,
            "shuffle": sampler.shuffle,
            "rank": sampler.rank,
            "world_size": sampler.num_replicas,
            "batch_size": loader.batch_size,
        }
        # Stage 0 adds per-stream resolution/frame bounds. Keep existing
        # unbounded loader identities compatible with their saved checkpoints.
        if dataset.max_num_frames is not None:
            identity["bounded_transfer_policy"] = {
                "transform_overrides": dataset.transform_overrides,
                "max_num_frames": dataset.max_num_frames,
            }
        return identity

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in super().__iter__():
            index = self.dataset_name_list.index(batch["dataset_name"][0])
            self._stream_fetch_counts[index] += 1
            self._fetch_count += 1
            # Capture at the consumer boundary: the async builder may already
            # have advanced global_id and child cursors for the next batch.
            batch[LOCAL_SIMULATION_RESUME_KEY] = {
                "format_version": 1,
                "fetch_count": self._fetch_count,
                "fetch_origin": self._fetch_origin,
                "stream_fetch_counts": self._stream_fetch_counts.copy(),
                "identity": self._resume_identity,
            }
            yield batch

    def set_start_iteration(self, iteration: int) -> None:
        """Restore the stream selector and the corresponding local consumption."""
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        if self._child_iterators_initialized:
            raise RuntimeError("Local resume must precede child iterator initialization.")
        sample_limits = [
            self.packing_limits.get(name, {}).get("max_samples_per_batch", self.max_samples_per_batch)
            for name in self.dataset_name_list
        ]
        # With lookahead disabled the packer stops after its first sample,
        # even when the configured sample ceiling is larger.
        sample_limits = [
            1 if lookahead <= 0 else limit for lookahead, limit in zip(self.lookahead_limits, sample_limits)
        ]
        fixed_consumption = self.iteration_time_budget is None and all(
            self.packing_limits.get(name, {}).get("max_sequence_length", self.max_sequence_length) is None
            and limit is not None
            and loader.batch_size == 1
            for name, limit, loader in zip(self.dataset_name_list, sample_limits, self.dataloader_list)
        )
        state = consume_local_resume_state()
        counts = [0] * len(self.dataloader_list)
        self._fetch_origin = 0
        if state is not None:
            if state.get("format_version") != 1 or state.get("identity") != self._resume_identity:
                raise ValueError("Local checkpoint stream identity does not match this run.")
            counts = state["stream_fetch_counts"]
            self._fetch_origin = state.get("fetch_origin", 0)
            if (
                state["fetch_count"] != iteration
                or not isinstance(self._fetch_origin, int)
                or isinstance(self._fetch_origin, bool)
                or not 0 <= self._fetch_origin <= iteration
                or len(counts) != len(self.dataloader_list)
                or any(not isinstance(count, int) or count < 0 for count in counts)
                or sum(counts) != iteration - self._fetch_origin
            ):
                raise ValueError("Local checkpoint counts do not match the trainer's restored fetch count.")
        elif iteration and self.restart_data_iterator_on_handoff:
            # A new stage can retain the trainer/scheduler counter while the
            # explicitly excluded dataloader component starts afresh. Keep a
            # separate raw-fetch origin so its later full resumes still compare
            # against the trainer's absolute fetch counter.
            self._fetch_origin = iteration
        elif iteration and fixed_consumption:
            if len(counts) == 1:
                counts[0] = iteration
            else:
                raise ValueError(
                    "Multi-stream local resume requires stream counts saved by LocalSimulationResumeCallback. "
                    "This checkpoint has no local stream state; restart from iteration zero."
                )
        self._fetch_count = iteration
        self._stream_fetch_counts = counts.copy()
        relative_iteration = iteration - self._fetch_origin
        self._resume_replay_batches = 0 if fixed_consumption else relative_iteration
        for loader in self.dataloader_list:
            assert isinstance(loader, LocalSimulationDataLoader)
            loader.set_start_iteration(0)
        if fixed_consumption:
            for loader, count, limit in zip(self.dataloader_list, counts, sample_limits):
                assert isinstance(loader, LocalSimulationDataLoader) and limit is not None
                loader.set_start_iteration(count * limit)
        super().set_start_iteration(relative_iteration)

    def _initialize_child_iterators_once(self) -> None:
        if self._child_iterators_initialized:
            return
        super()._initialize_child_iterators_once()
        if self._resume_replay_batches:
            log.info(f"Restoring local packing state by replaying {self._resume_replay_batches} batches.")
            self.global_id = 0
            self._stream_fetch_counts = [0] * len(self.dataloader_list)
            replay = self._iter_synchronous()
            for _ in range(self._resume_replay_batches):
                batch = next(replay)
                index = self.dataset_name_list.index(batch["dataset_name"][0])
                self._stream_fetch_counts[index] += 1
            self._resume_replay_batches = 0
