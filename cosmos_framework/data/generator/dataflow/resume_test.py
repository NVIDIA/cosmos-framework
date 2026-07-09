# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint->restart resume parity for CosmosDataLoader(MapDistributor) using
CosmosDataLoaderStateCallback. Single process, num_workers=0."""

from __future__ import annotations

import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
)


class _IdDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"id": torch.tensor(idx)}


def _build(seed=0):
    return CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False, seed=seed),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=0,
    )


def test_resume_continues_without_dup_or_skip():
    cb = CosmosDataLoaderStateCallback()
    loader = _build()
    it = iter(loader)
    seen_ids = []
    for _ in range(5):
        b = next(it)
        cb._update_state_from_batch(b)
        seen_ids.append(b["id"].item())
    assert seen_ids == [0, 1, 2, 3, 4]

    state = cb.state_dict()
    assert state[0]["index"] == 4
    cb2 = CosmosDataLoaderStateCallback()
    cb2.load_state_dict(state)

    loader2 = _build()
    it2 = iter(loader2)  # one iterator: env-var fast-forward happens once, then continues
    resumed = [next(it2)["id"].item() for _ in range(3)]
    assert resumed == [5, 6, 7]
