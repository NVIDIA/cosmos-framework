# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for IterableDistributor (and, later, MapDistributor)."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.distributors import IterableDistributor


def test_iterable_single_rank_single_worker_sees_everything():
    d = IterableDistributor(range(6))
    got = list(d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1))
    assert got == [0, 1, 2, 3, 4, 5]


def test_iterable_sharding_is_disjoint_and_covers_all():
    seen = []
    for r in range(2):
        for w in range(2):
            d = IterableDistributor(range(12))
            seen.append(set(d.stream(dp_rank=r, dp_world_size=2, worker_id=w, num_workers=2)))
    for a in range(4):
        for b in range(a + 1, 4):
            assert seen[a].isdisjoint(seen[b]), (a, b, seen[a], seen[b])
    assert set().union(*seen) == set(range(12))


def test_iterable_stream_indices_match_formula():
    d = IterableDistributor(range(12))
    got = list(d.stream(dp_rank=1, dp_world_size=2, worker_id=0, num_workers=2))
    assert got == [2, 6, 10]


import torch

from cosmos_framework.data.vfm.dataflow.distributors import MapDistributor


class _MapDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"i": idx}


def test_map_no_shuffle_is_sequential_and_sharded():
    d = MapDistributor(_MapDS(12), shuffle=False)
    it = d.stream(dp_rank=1, dp_world_size=2, worker_id=0, num_workers=2)
    first = [next(it)["i"] for _ in range(3)]
    assert first == [2, 6, 10]
    assert next(it)["i"] == 2


def test_map_shuffle_is_seeded_and_reproducible():
    a = MapDistributor(_MapDS(20), shuffle=True, seed=123)
    b = MapDistributor(_MapDS(20), shuffle=True, seed=123)
    ia = a.stream(0, 1, 0, 1)
    ib = b.stream(0, 1, 0, 1)
    assert [next(ia)["i"] for _ in range(20)] == [next(ib)["i"] for _ in range(20)]


def test_map_shuffle_first_epoch_is_a_permutation():
    d = MapDistributor(_MapDS(20), shuffle=True, seed=7)
    it = d.stream(0, 1, 0, 1)
    first_epoch = [next(it)["i"] for _ in range(20)]
    assert sorted(first_epoch) == list(range(20))


def test_map_sharding_disjoint_and_covers_one_epoch():
    seen = []
    for r in range(2):
        for w in range(2):
            it = MapDistributor(_MapDS(12), shuffle=False).stream(r, 2, w, 2)
            seen.append({next(it)["i"] for _ in range(3)})
    for a in range(4):
        for b in range(a + 1, 4):
            assert seen[a].isdisjoint(seen[b])
    assert set().union(*seen) == set(range(12))
