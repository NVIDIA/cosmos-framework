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
