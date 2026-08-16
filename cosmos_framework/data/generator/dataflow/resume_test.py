# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint->restart resume parity for CosmosDataLoader(MapDistributor) using
CosmosDataLoaderStateCallback. Single process, num_workers=0."""

from __future__ import annotations

import threading
from itertools import islice

import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.data.generator.dataflow import (
    ContiguousBatcher,
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
    RawItemProcessor,
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


def test_contiguous_multisample_batches_resume_without_dup_or_skip():
    callback = CosmosDataLoaderStateCallback()
    loader = CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False),
        processor=IdentityProcessor(),
        batcher=ContiguousBatcher(max_batch_size=4),
        num_workers=0,
    )
    first = next(iter(loader))
    callback._update_state_from_batch(first)

    assert first["id"].tolist() == [0, 1, 2, 3]
    assert callback.state_dict()[0]["index"] == 3

    resumed_callback = CosmosDataLoaderStateCallback()
    resumed_callback.load_state_dict(callback.state_dict())
    resumed_loader = CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False),
        processor=IdentityProcessor(),
        batcher=ContiguousBatcher(max_batch_size=4),
        num_workers=0,
    )

    assert next(iter(resumed_loader))["id"].tolist() == [4, 5, 6, 7]


def test_contiguous_batcher_sequence_ceiling_preserves_fixed_batch_size():
    batcher = ContiguousBatcher(max_batch_size=3, max_tokens=4)
    samples = iter(
        [
            {"id": 0, "input_ids": [1, 2]},
            {"id": 1, "input_ids": [1, 2, 3, 4]},
            {"id": 2, "input_ids": [1]},
        ]
    )
    assert [[item["id"] for item in batch] for batch in batcher.batches(samples)] == [
        [0, 1, 2]
    ]


def test_contiguous_batcher_rejects_sample_over_sequence_ceiling():
    batcher = ContiguousBatcher(max_batch_size=2, max_tokens=3)
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        list(batcher.batches(iter([{"input_ids": [1, 2, 3, 4]}])))


def test_map_distributor_normalizes_environment_style_seed_string():
    distributor = MapDistributor(_IdDS(4), shuffle=True, seed="42")

    assert distributor._seed == 42
    first = [item["id"].item() for item in islice(distributor.stream(0, 1, 0, 1), 4)]
    second = [
        item["id"].item() for item in islice(MapDistributor(_IdDS(4), shuffle=True, seed=42).stream(0, 1, 0, 1), 4)
    ]
    assert first == second


class _ConcurrentIdentityProcessor(RawItemProcessor):
    def __init__(self, participants: int):
        self._barrier = threading.Barrier(participants, timeout=2)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def process(self, item):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        self._barrier.wait()
        with self._lock:
            self.active -= 1
        return item


def test_processing_threads_preserve_order_and_run_concurrently():
    processor = _ConcurrentIdentityProcessor(participants=4)
    loader = CosmosDataLoader(
        distributor=MapDistributor(_IdDS(8), shuffle=False),
        processor=processor,
        batch_size=1,
        num_workers=0,
        processing_threads="4",
    )

    batches = list(islice(loader, 4))

    assert [batch["id"].item() for batch in batches] == [0, 1, 2, 3]
    assert processor.peak == 4


def test_processing_threads_rejects_zero():
    with pytest.raises(ValueError, match="processing_threads"):
        CosmosDataLoader(
            distributor=MapDistributor(_IdDS(1), shuffle=False),
            processor=IdentityProcessor(),
            batch_size=1,
            processing_threads=0,
        )
