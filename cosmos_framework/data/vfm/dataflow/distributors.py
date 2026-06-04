# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in DataDistributor implementations.

IterableDistributor wraps any Python iterable / IterableDataset with
round-robin DP x worker sharding (no resume). MapDistributor wraps a map-style
Dataset with per-epoch shuffle + slice sharding (resume lands in a later plan).
"""

from __future__ import annotations

from typing import Any, Iterator

from cosmos_framework.data.vfm.dataflow.base import DataDistributor


class IterableDistributor(DataDistributor):
    """Round-robin shard of an iterable: each (rank, worker) sees every
    ``dp_world_size * num_workers``-th item starting at
    ``dp_rank * num_workers + worker_id``. Not resumable."""

    def __init__(self, iterable: Any):
        self._iterable = iterable

    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ) -> Iterator[Any]:
        total_streams = dp_world_size * num_workers
        my_stream = dp_rank * num_workers + worker_id
        for i, item in enumerate(self._iterable):
            if i % total_streams == my_stream:
                yield item


import torch


class MapDistributor(DataDistributor):
    """Per-epoch shuffle + slice sharding of a map-style Dataset. Resume (env-var
    fast-forward) is added in a later plan; for now the ABC no-op defaults apply."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        seed: int = 0,
        shuffle: bool = True,
        name: str = "",
    ):
        self._dataset = dataset
        self._seed = seed
        self._shuffle = shuffle
        self._name = name

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ):
        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        n = len(self._dataset)  # type: ignore[arg-type]
        epoch = 0
        while True:
            if self._shuffle:
                g = torch.Generator().manual_seed(self._seed + epoch)
                perm = torch.randperm(n, generator=g).tolist()
            else:
                perm = list(range(n))
            stream_slice = perm[stream_id::total_streams]
            for pos in range(len(stream_slice)):
                yield self._dataset[stream_slice[pos]]
            epoch += 1
