# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CosmosDataLoader — slim orchestrator that wires the four dataflow roles
(DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator) inside
each DataLoader worker. The canonical training dataloader.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np
import torch
import torch.utils.data

from cosmos_framework.data.generator.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)
from cosmos_framework.data.generator.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.generator.dataflow.collators import DefaultBatchCollator
from cosmos_framework.utils import log


class _DataflowIterableDataset(torch.utils.data.IterableDataset):
    """Wires distributor -> processor -> batcher -> collator inside a worker."""

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher,
        collator: BatchCollator,
        dp_rank: int,
        dp_world_size: int,
        processing_threads: int,
    ):
        super().__init__()
        self._distributor = distributor
        self._processor = processor
        self._batcher = batcher
        self._collator = collator
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._processing_threads = processing_threads

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)

        def _process_one(item):
            if isinstance(item, dict):
                meta = {k: item.pop(k) for k in list(item) if k.startswith("_dp_")}
            else:
                meta = {}
            sample = self._processor.process(item)
            if meta and isinstance(sample, dict):
                sample.update(meta)
            return sample

        def _processed():
            if self._processing_threads == 1:
                for item in raw:
                    yield _process_one(item)
                return

            # Keep one small, bounded in-process pool per DataLoader worker.
            # executor.map preserves source order, so sharding, packing, and
            # checkpoint/resume positions remain identical to serial processing.
            with ThreadPoolExecutor(max_workers=self._processing_threads) as executor:
                while True:
                    chunk = list(islice(raw, self._processing_threads))
                    if not chunk:
                        return
                    yield from executor.map(_process_one, chunk)

        for group in self._batcher.batches(_processed()):
            has_meta = bool(group) and isinstance(group[0], dict) and "_dp_epoch" in group[0]
            if has_meta:
                epochs = [s["_dp_epoch"] for s in group]
                positions = [s["_dp_stream_pos"] for s in group]
                if self._batcher.preserves_source_order:
                    # The final emitted sample is the exact stream cursor even
                    # when a large fixed batch spans one or more short epochs.
                    # Independent maxima are invalid across an epoch boundary
                    # because positions restart at zero in each epoch.
                    cursor_epoch = group[-1]["_dp_epoch"]
                    cursor_pos = group[-1]["_dp_stream_pos"]
                else:
                    cursor_epoch = max(epochs)
                    cursor_pos = max(positions)
                # A reordering batcher can leave lower positions buffered when
                # yielding a multi-sample group. Preserve the existing strict
                # rejection for that unsupported resume combination.
                if len(group) > 1 and not self._batcher.preserves_source_order:
                    contiguous = min(epochs) == cursor_epoch and sorted(positions) == list(
                        range(min(positions), cursor_pos + 1)
                    )
                    if not contiguous:
                        raise ValueError(
                            "Map-style resume cannot safely stamp a multi-sample batch whose "
                            "_dp_stream_pos values are non-contiguous or span epochs (reordering "
                            "batcher + batch_size>1). Use max_batch_size=1 with pool packing, a "
                            "sequential (order-preserving) batcher, or an iterable (non-resumable) "
                            "source."
                        )
                clean = [{k: v for k, v in s.items() if not k.startswith("_dp_")} for s in group]
                batch = self._collator.collate(clean)
                batch["sample_worker_id"] = torch.tensor([worker_id] * len(group))
                batch["sample_epoch"] = torch.tensor([cursor_epoch] * len(group))
                batch["sample_index"] = torch.tensor([cursor_pos] * len(group))
            else:
                batch = self._collator.collate(group)
            yield batch


class CosmosDataLoader(torch.utils.data.DataLoader):
    """Public entry point: bring any dataset into training via four roles.

    Either pass an explicit ``batcher`` (and optional ``collator``), or pass a
    bare ``batch_size=N`` for stock fixed-size batching — the loader then builds
    ``SimpleBatcher(N)`` + ``DefaultBatchCollator()``. Passing both is an error.

    DP coordinates: ``parallel_dims.dp_coord`` > ``torch.distributed`` > (0, 1).
    """

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher | None = None,
        collator: BatchCollator | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        processing_threads: int = 1,
        parallel_dims=None,
    ):
        if batch_size is not None and batcher is not None:
            raise ValueError(
                "Pass either batch_size= (sugar) or an explicit batcher=, not both."
            )
        if batch_size is None and batcher is None:
            raise ValueError("Provide either a batcher= or a batch_size=.")
        if batch_size is not None:
            batcher = SimpleBatcher(batch_size=batch_size)
        if collator is None:
            collator = DefaultBatchCollator()
        processing_threads = int(processing_threads)
        if processing_threads < 1:
            raise ValueError(
                "CosmosDataLoader: processing_threads must be >= 1, "
                f"got {processing_threads}"
            )

        if parallel_dims is not None:
            dp_rank, dp_world_size = parallel_dims.dp_coord
        elif torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()
            if dp_world_size > 1:
                log.info(
                    "CosmosDataLoader: using global rank for DP sharding. "
                    "For FSDP+TP/PP pass parallel_dims= for the correct DP rank.",
                    rank0_only=True,
                )
        else:
            dp_rank, dp_world_size = 0, 1

        dataset = _DataflowIterableDataset(
            distributor=distributor,
            processor=processor,
            batcher=batcher,
            collator=collator,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            processing_threads=processing_threads,
        )

        from cosmos_framework.data.generator.dataflow.distributors import MapDistributor

        if isinstance(distributor, MapDistributor) and num_workers > 0 and not persistent_workers:
            log.info(
                "CosmosDataLoader: MapDistributor requires persistent_workers=True for "
                "correct stateful resume; overriding to True.",
                rank0_only=True,
            )
            persistent_workers = True

        if persistent_workers and num_workers == 0:
            log.info(
                "CosmosDataLoader: persistent_workers=True ignored because num_workers=0.",
                rank0_only=True,
            )
            persistent_workers = False

        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(dataset, batch_size=None, **loader_kwargs)


class JointCosmosDataLoader:
    """Wraps multiple ``CosmosDataLoader`` instances with ratio-based seeded selection.

    One output batch = one inner loader, selected deterministically by ratio at each
    step.  Adds a ``"dataset_name"`` key to every yielded batch so downstream callbacks
    can route state updates to the correct inner loader.

    Parameters
    ----------
    dataloaders:
        ``{name: {"dataloader": CosmosDataLoader, "ratio": int}}`` mapping.
        Entries with ``ratio <= 0`` are silently skipped.
    seed:
        Base seed for per-step dataset selection.  Step ``i`` uses
        ``np.random.RandomState(seed + i)`` — fully reproducible on resume via
        ``set_start_iteration``.
    """

    def __init__(
        self,
        dataloaders: dict,
        seed: int = 42,
    ) -> None:
        entries = [
            (name, cfg["dataloader"], cfg["ratio"])
            for name, cfg in dataloaders.items()
            if cfg.get("ratio", 0) > 0
        ]
        if not entries:
            raise ValueError("JointCosmosDataLoader: no dataloaders with ratio > 0")

        self._names: list[str] = [e[0] for e in entries]
        if "global_id" in self._names:
            raise ValueError(
                "JointCosmosDataLoader: dataset name 'global_id' is reserved "
                "by the checkpoint state format; use a different name."
            )
        self._loaders: list[CosmosDataLoader] = [e[1] for e in entries]
        ratios = np.array([e[2] for e in entries], dtype=float)
        self._probs: np.ndarray = ratios / ratios.sum()
        self._seed = seed
        self._global_id = 0
        # Iterators are created lazily on the first __iter__ call so that
        # DataLoaderStateCallback.load_state_dict can install resume env vars
        # before workers are spawned (for num_workers > 0, iter(DataLoader)
        # forks workers immediately; env vars must be set in the parent first).
        self._iterators: list | None = None

        total = ratios.sum()
        lines = [f"JointCosmosDataLoader: {len(self._names)} streams"]
        for name, ratio in zip(self._names, ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

    def set_start_iteration(self, iteration: int) -> None:
        """Restore deterministic selection sequence after checkpoint resume.

        Called by ``JointCosmosDataLoaderStateCallback.load_state_dict`` and by the
        trainer (if present) via ``hasattr`` guard.
        """
        self._global_id = iteration

    def __iter__(self):
        # Lazy init: create iterators here (not in __init__) so that
        # load_state_dict can set resume env vars before workers fork.
        if self._iterators is None:
            self._iterators = [iter(loader) for loader in self._loaders]
        while True:
            rng = np.random.RandomState(self._seed + self._global_id)
            idx = int(rng.choice(len(self._loaders), p=self._probs))
            try:
                batch = next(self._iterators[idx])
            except StopIteration:
                # Inner CosmosDataLoaders are infinite; this guard handles
                # the unlikely case of a finite IterableDataset inner source.
                self._iterators[idx] = iter(self._loaders[idx])
                batch = next(self._iterators[idx])
            batch["dataset_name"] = self._names[idx]
            self._global_id += 1
            yield batch
