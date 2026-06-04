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


def test_map_empty_dataset_terminates():
    d = MapDistributor(_MapDS(0), shuffle=False)
    assert list(d.stream(0, 1, 0, 1)) == []


def test_map_empty_shard_terminates_no_hang():
    # dataset smaller than dp_world_size * num_workers -> stream_id 3 >= n 2 -> empty shard.
    it = MapDistributor(_MapDS(2), shuffle=False).stream(
        dp_rank=0, dp_world_size=1, worker_id=3, num_workers=4
    )
    assert list(it) == []


def test_map_nonempty_shard_still_infinite():
    # guard must NOT terminate a valid non-empty shard (stays infinite).
    it = MapDistributor(_MapDS(4), shuffle=False).stream(0, 1, 0, 1)
    first = [next(it)["i"] for _ in range(4)]
    assert first == [0, 1, 2, 3]
    assert next(it)["i"] == 0  # wraps into epoch 1, still producing


import os


def test_map_resume_fast_forwards_from_env(monkeypatch):
    monkeypatch.setenv("DP_STATE_WORKER_0_EPOCH", "0")
    monkeypatch.setenv("DP_STATE_WORKER_0_INDEX", "3")
    d = MapDistributor(_MapDS(10), shuffle=False)
    it = d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1)
    first = next(it)
    assert first["i"] == 4
    assert first["_dp_epoch"] == 0
    assert first["_dp_stream_pos"] == 4


def test_map_attaches_dp_meta_when_no_resume():
    d = MapDistributor(_MapDS(4), shuffle=False)
    it = d.stream(dp_rank=0, dp_world_size=1, worker_id=0, num_workers=1)
    s0 = next(it)
    assert s0["i"] == 0 and s0["_dp_epoch"] == 0 and s0["_dp_stream_pos"] == 0


def test_map_resume_env_is_consumed_once(monkeypatch):
    monkeypatch.setenv("DP_STATE_WORKER_0_INDEX", "2")
    d = MapDistributor(_MapDS(6), shuffle=False)
    next(d.stream(0, 1, 0, 1))
    assert "DP_STATE_WORKER_0_INDEX" not in os.environ


def test_map_name_namespaces_env(monkeypatch):
    monkeypatch.setenv("DP_STATE_vlm_WORKER_0_INDEX", "1")
    d = MapDistributor(_MapDS(6), shuffle=False, name="vlm")
    first = next(d.stream(0, 1, 0, 1))
    assert first["i"] == 2


from cosmos_framework.data.vfm.dataflow.distributors import RankPartitionedDistributor


class _ShardAwareDS(torch.utils.data.IterableDataset):
    def __init__(self, tag):
        self.tag = tag
        self.shard_world_size = None
        self.shard_rank = None
        self.shard_id = None

    def __iter__(self):
        yield {"tag": self.tag, "sw": self.shard_world_size, "sr": self.shard_rank, "sid": self.shard_id}


def _rp():
    return RankPartitionedDistributor({
        "video": {"dataset": _ShardAwareDS("video"), "ratio": 3},
        "image": {"dataset": _ShardAwareDS("image"), "ratio": 1},
    })


def test_rank_partition_allocates_and_sets_shards():
    # world=4, ratios 3:1 -> ranks 0-2 video (shard_world_size=3), rank 3 image.
    r0 = next(_rp().stream(dp_rank=0, dp_world_size=4, worker_id=0, num_workers=1))
    assert r0["tag"] == "video" and r0["sw"] == 3 and r0["sr"] == 0
    r2 = next(_rp().stream(dp_rank=2, dp_world_size=4, worker_id=0, num_workers=1))
    assert r2["tag"] == "video" and r2["sr"] == 2
    r3 = next(_rp().stream(dp_rank=3, dp_world_size=4, worker_id=0, num_workers=1))
    assert r3["tag"] == "image" and r3["sw"] == 1 and r3["sr"] == 0


from cosmos_framework.data.vfm.dataflow.distributors import MixtureDistributor


def test_mixture_draws_from_both_by_ratio():
    a = IterableDistributor([{"src": "a", "i": i} for i in range(100000)])
    b = IterableDistributor([{"src": "b", "i": i} for i in range(100000)])
    m = MixtureDistributor({"a": (a, 3.0), "b": (b, 1.0)}, seed=0)
    it = m.stream(0, 1, 0, 1)
    draws = [next(it)["src"] for _ in range(400)]
    frac_a = draws.count("a") / len(draws)
    assert 0.6 < frac_a < 0.85


def test_mixture_is_seeded_reproducible():
    def build():
        a = IterableDistributor([{"i": i} for i in range(100000)])
        b = IterableDistributor([{"i": -i} for i in range(100000)])
        return MixtureDistributor({"a": (a, 1.0), "b": (b, 1.0)}, seed=42)
    it1 = build().stream(0, 1, 0, 1)
    it2 = build().stream(0, 1, 0, 1)
    assert [next(it1)["i"] for _ in range(50)] == [next(it2)["i"] for _ in range(50)]
