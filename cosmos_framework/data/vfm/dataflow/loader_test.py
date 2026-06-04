# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""End-to-end tests for the CosmosDataLoader orchestrator (explicit roles +
batch_size sugar, single process)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.data.vfm.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    IterableDistributor,
    MapDistributor,
    SimpleBatcher,
)


class _MapDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"x": torch.tensor([float(idx)]), "i": idx}


def test_explicit_roles_end_to_end():
    loader = CosmosDataLoader(
        distributor=IterableDistributor([{"x": torch.tensor([float(i)])} for i in range(6)]),
        processor=IdentityProcessor(),
        batcher=SimpleBatcher(batch_size=2),
    )
    batches = []
    it = iter(loader)
    for _ in range(3):
        batches.append(next(it))
    assert [b["x"].shape[0] for b in batches] == [2, 2, 2]
    assert torch.equal(batches[0]["x"].flatten(), torch.tensor([0.0, 1.0]))


def test_batch_size_sugar_builds_simple_batcher_and_default_collator():
    loader = CosmosDataLoader(
        distributor=MapDistributor(_MapDS(10), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=4,
    )
    it = iter(loader)
    batch = next(it)
    assert batch["x"].shape == (4, 1)
    assert batch["i"].tolist() == [0, 1, 2, 3]


def test_batch_size_with_explicit_batcher_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        CosmosDataLoader(
            distributor=IterableDistributor([]),
            processor=IdentityProcessor(),
            batch_size=4,
            batcher=SimpleBatcher(batch_size=2),
        )


def test_requires_batcher_or_batch_size():
    with pytest.raises(ValueError, match="batcher.*batch_size|batch_size.*batcher"):
        CosmosDataLoader(
            distributor=IterableDistributor([]),
            processor=IdentityProcessor(),
        )


def test_multiworker_disjoint_and_complete_one_epoch():
    loader = CosmosDataLoader(
        distributor=MapDistributor(_MapDS(12), shuffle=False),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=2,
    )
    it = iter(loader)
    seen = [next(it)["i"].item() for _ in range(12)]
    assert sorted(seen) == list(range(12))
    assert len(set(seen)) == 12
