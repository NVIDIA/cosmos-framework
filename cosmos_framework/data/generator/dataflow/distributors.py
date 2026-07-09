# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in DataDistributor implementations.

IterableDistributor wraps any Python iterable / IterableDataset with
round-robin DP x worker sharding (no resume). MapDistributor wraps a map-style
Dataset with per-epoch shuffle + slice sharding (resume lands in a later plan).
"""

from __future__ import annotations

from typing import Any, Iterator

from cosmos_framework.data.generator.dataflow.base import DataDistributor


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

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        import os

        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        n = len(self._dataset)  # type: ignore[arg-type]
        if n == 0:
            return
        if stream_id >= n:
            return
        _pfx = f"COSMOS_DL_STATE_{self._name}_" if self._name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_EPOCH", 0))
        resume_pos = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_INDEX", -1))
        epoch = resume_epoch
        while True:
            if self._shuffle:
                g = torch.Generator().manual_seed(self._seed + epoch)
                perm = torch.randperm(n, generator=g).tolist()
            else:
                perm = list(range(n))
            stream_slice = perm[stream_id::total_streams]
            start = (resume_pos + 1) if epoch == resume_epoch else 0
            for pos in range(start, len(stream_slice)):
                item = self._dataset[stream_slice[pos]]
                if isinstance(item, dict):
                    yield {"_dp_epoch": epoch, "_dp_stream_pos": pos, **item}
                else:
                    yield item
            epoch += 1


from cosmos_framework.utils.lazy_config import instantiate


class RankPartitionedDistributor(DataDistributor):
    """Allocate whole DP ranks to datasets by ratio; the chosen dataset self-shards.
    Ports RankPartitionedDataLoader (joint_dataloader.py:660-757) minus the inner
    torch DataLoader (CosmosDataLoader owns workers/collation)."""

    def __init__(self, datasets: dict):
        self._datasets_cfg = datasets
        self._cached = None  # built dataset for this rank, set on first stream()

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        if self._cached is None:
            self._cached = self._allocate_and_build(dp_rank, dp_world_size)
        yield from iter(self._cached)

    def _allocate_and_build(self, rank, world_size):
        names, dataset_configs, ratios = [], [], []
        for name, cfg in self._datasets_cfg.items():
            if cfg["ratio"] <= 0:
                continue
            names.append(name)
            dataset_configs.append(cfg["dataset"])
            ratios.append(cfg["ratio"])
        assert len(names) > 0, "No datasets with positive ratios"
        assert world_size >= len(names), f"world_size {world_size} < num datasets {len(names)}"
        # PORT the allocation verbatim from joint_dataloader.py:707-744:
        total_ratio = sum(ratios)
        ideal = [r / total_ratio * world_size for r in ratios]
        allocations = [max(1, int(q)) for q in ideal]
        remaining = world_size - sum(allocations)
        if remaining > 0:
            order = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
            for j in range(remaining):
                allocations[order[j]] += 1
        elif remaining < 0:
            deficit = -remaining
            while deficit > 0:
                best = max(
                    (i for i in range(len(allocations)) if allocations[i] > 1),
                    key=lambda i: (allocations[i] - ideal[i], allocations[i]),
                )
                allocations[best] -= 1
                deficit -= 1
        cumulative = 0
        idx = -1
        for i, a in enumerate(allocations):
            if rank < cumulative + a:
                idx = i
                break
            cumulative += a
        assert idx >= 0
        shard_rank = rank - cumulative
        shard_world_size = allocations[idx]
        cfg = dataset_configs[idx]
        ds = cfg if isinstance(cfg, torch.utils.data.IterableDataset) else instantiate(cfg)
        ds.shard_world_size = shard_world_size
        ds.shard_rank = shard_rank
        ds.shard_id = idx
        return ds


import random as _random_mod


class MixtureDistributor(DataDistributor):
    """Ratio-weighted merge of multiple distributors into one stream (homogeneous
    join). Generalizes PackingIterableDataset's weighted _get_next_sample."""

    def __init__(self, sources: dict, seed: int = 0):
        # sources: {name: (DataDistributor, ratio_float)}
        self._names = list(sources.keys())
        self._dists = [sources[n][0] for n in self._names]
        self._ratios = [float(sources[n][1]) for n in self._names]
        self._seed = seed

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        rng = _random_mod.Random(self._seed + dp_rank * 100003 + worker_id)
        iters = [d.stream(dp_rank, dp_world_size, worker_id, num_workers) for d in self._dists]
        while True:
            idx = rng.choices(range(len(iters)), weights=self._ratios, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = self._dists[idx].stream(dp_rank, dp_world_size, worker_id, num_workers)
                yield next(iters[idx])
