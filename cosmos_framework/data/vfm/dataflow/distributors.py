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
