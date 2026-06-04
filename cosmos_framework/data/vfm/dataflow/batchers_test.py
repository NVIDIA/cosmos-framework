# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for SimpleBatcher."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher


def _src(n):
    return iter([{"i": i} for i in range(n)])


def test_simple_batcher_groups_in_fixed_size():
    groups = list(SimpleBatcher(batch_size=3).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5], [6]]


def test_simple_batcher_drop_last_discards_partial():
    groups = list(SimpleBatcher(batch_size=3, drop_last=True).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5]]


def test_simple_batcher_empty_source_yields_nothing():
    assert list(SimpleBatcher(batch_size=3).batches(_src(0))) == []


import torch

from cosmos_framework.data.vfm.dataflow.batchers import PoolPackingBatcher


def _txt(n_tokens, tag=0):
    return {"input_ids": torch.zeros(n_tokens, dtype=torch.long), "tag": tag}


def test_pool_emits_oversized_sample_as_singleton():
    b = PoolPackingBatcher(max_tokens=1000, pool_size=4, max_batch_size=8, long_threshold=500)
    groups = list(b.batches(iter([_txt(600), _txt(10), _txt(10)])))
    assert [len(g) for g in groups][0] == 1
    assert sum(len(g) for g in groups) == 3


def test_pool_respects_max_batch_size():
    b = PoolPackingBatcher(max_tokens=10_000, pool_size=8, max_batch_size=1, long_threshold=6400)
    groups = list(b.batches(iter([_txt(10) for _ in range(5)])))
    assert all(len(g) == 1 for g in groups)
    assert len(groups) == 5


def test_pool_packs_multiple_within_budget():
    b = PoolPackingBatcher(max_tokens=100, pool_size=8, max_batch_size=8, long_threshold=6400)
    groups = list(b.batches(iter([_txt(10) for _ in range(8)])))
    assert len(groups) == 1
    assert len(groups[0]) == 8


def test_pool_sample_size_default_is_len_input_ids():
    b = PoolPackingBatcher(max_tokens=100, pool_size=4, max_batch_size=4, long_threshold=6400)
    assert b.sample_size(_txt(7)) == 7


def test_pool_sample_size_fn_override():
    b = PoolPackingBatcher(
        max_tokens=100, pool_size=4, max_batch_size=4, long_threshold=6400,
        size_fn=lambda s: 3,
    )
    assert b.sample_size(_txt(7)) == 3


def test_pool_does_not_mix_modalities():
    img = {"input_ids": torch.zeros(10, dtype=torch.long), "pixel_values": torch.zeros(4, 8)}
    txt = _txt(10)
    b = PoolPackingBatcher(max_tokens=10_000, pool_size=8, max_batch_size=8, long_threshold=6400)
    groups = list(b.batches(iter([img, txt])))
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)
