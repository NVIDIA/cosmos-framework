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


from cosmos_framework.data.vfm.dataflow.batchers import SequentialPackingBatcher


def _vid(text_len, t=1, h=64, w=64):
    return {"text_token_ids": torch.arange(text_len), "video": torch.zeros(3, t, h, w)}


def test_sequential_size_uses_vae_formula():
    b = SequentialPackingBatcher(
        max_sequence_length=100000,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    # text(5) + 1 (eos) + vision: latent_h=64//(16*2)=2, latent_w=2, latent_t=1 -> 2*2*1+2=6
    assert b.sample_size(_vid(5)) == 5 + 1 + 6


def test_sequential_packs_in_order_until_budget():
    b = SequentialPackingBatcher(
        max_sequence_length=40,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    groups = list(b.batches(iter([_vid(5) for _ in range(7)])))
    assert len(groups[0]) == 3   # ~12 tokens each; 3 fit under 40


def test_sequential_discards_oversized_when_batch_empty():
    b = SequentialPackingBatcher(
        max_sequence_length=10,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
    )
    groups = list(b.batches(iter([_vid(50), _vid(1)])))
    flat = [s for g in groups for s in g]
    assert all(s["text_token_ids"].shape[0] == 1 for s in flat)  # big one discarded


def test_sequential_count_only_mode():
    # max_sequence_length=None + max_samples_per_batch=2 -> fixed 2-sample groups.
    b = SequentialPackingBatcher(
        max_sequence_length=None,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=2,
    )
    groups = list(b.batches(iter([_vid(5) for _ in range(6)])))
    assert all(len(g) == 2 for g in groups)
    assert len(groups) == 3


def test_sequential_requires_exactly_one_mode():
    import pytest
    with pytest.raises(AssertionError):
        SequentialPackingBatcher(  # both None -> invalid
            max_sequence_length=None,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            patch_spatial=2,
            max_samples_per_batch=None,
        )
