# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CosmosDataLoader — slim orchestrator that wires the four dataflow roles
(DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator) inside
each DataLoader worker.

Lives in dataflow/loader.py during the migration so it coexists with the legacy
cosmos_framework/data/vfm/data_packer_dataloader.py; the cleanup PR makes this
the canonical loader.
"""

from __future__ import annotations

import torch
import torch.utils.data

from cosmos_framework.utils import log
from cosmos_framework.data.vfm.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)
from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator


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
    ):
        super().__init__()
        self._distributor = distributor
        self._processor = processor
        self._batcher = batcher
        self._collator = collator
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)
        samples = (self._processor.process(item) for item in raw)
        for group in self._batcher.batches(samples):
            yield self._collator.collate(group)


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
        )

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
