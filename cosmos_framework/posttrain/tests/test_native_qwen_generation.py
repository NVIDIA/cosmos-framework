# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU regression controls for native generation, not real-model qualification."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import generation_cache_position
from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    _qwen3_vl_causal_mask,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
)
from cosmos_framework.posttrain.model.qwen3_vl_moe import override_qwen3_vl_moe_for_padded_batches
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _dense_config, _moe_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _net(family: str, override: bool, attention: str = "sdpa"):
    def prepare(net):
        if override:
            override_qwen3_vl_forward(net.model)
        if family == "qwen3_vl_moe":
            override_qwen3_vl_moe_for_padded_batches(net)

    torch.manual_seed(42)
    config = (_dense_config if family == "qwen3_vl" else _moe_config)()
    # Real native checkpoints declare multimodal RoPE. Supply a head-dim-8
    # declaration explicitly: Transformers 4 leaves the tiny MoE default None.
    config.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    return build_qwen3_vl_policy(
        model_type=family,
        model_config=config,
        dtype=torch.float32,
        attn_implementation=attention,
        initialize_on_meta=False,
        hooks=Qwen3VLBuildHooks(prepare_model=prepare),
    ).net.eval()


def _inputs(modality: str):
    if modality == "text":
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        media = {}
    else:
        ids = torch.tensor([[1, 118, 120 if modality == "image" else 121, 119, 2, 3, 4]])
        media = {
            "pixel_values" if modality == "image" else "pixel_values_videos": torch.randn(4, 24),
            "image_grid_thw" if modality == "image" else "video_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), **media}


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("modality", ["text", "image", "video"])
def test_native_generate_cached_matches_full_recomputation(family, override, modality):
    net = _net(family, override)
    batch = _inputs(modality)
    before = {name: (id(tensor), tensor.detach().clone()) for name, tensor in net.named_parameters()}
    options = dict(
        max_new_tokens=3,
        min_new_tokens=3,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        suppress_tokens=[118, 119, 120, 121],
        return_dict_in_generate=True,
        output_logits=True,
    )
    with torch.inference_mode():
        net.model.rope_deltas = None
        full = net.generate(**batch, use_cache=False, **options)
        net.model.rope_deltas = None
        cached = net.generate(**batch, use_cache=True, **options)
    torch.testing.assert_close(cached.sequences, full.sequences, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(cached.logits), torch.stack(full.logits), rtol=1e-5, atol=5e-7)
    for name, tensor in net.named_parameters():
        owner, value = before[name]
        assert id(tensor) == owner
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("suffix_length", [1, 3])
def test_native_cached_suffix_uses_absolute_causal_positions(family, attention, suffix_length):
    net = _net(family, True, attention)
    batch = _inputs("image")
    ids = batch.pop("input_ids")
    mask = batch.pop("attention_mask")
    prefix_length = ids.shape[1] - suffix_length
    with torch.inference_mode():
        full = net(
            input_ids=ids, attention_mask=mask, cache_position=torch.arange(ids.shape[1]), use_cache=False, **batch
        ).logits
        prefix = net(
            input_ids=ids[:, :prefix_length],
            attention_mask=mask[:, :prefix_length],
            cache_position=torch.arange(prefix_length),
            use_cache=True,
            **batch,
        )
        saved = copy.deepcopy(prefix.past_key_values)
        for _ in range(2):
            fork = copy.deepcopy(saved)
            out = net(
                input_ids=ids[:, prefix_length:],
                attention_mask=mask,
                past_key_values=fork,
                cache_position=torch.arange(prefix_length, ids.shape[1]),
                use_cache=True,
            ).logits
            torch.testing.assert_close(out, full[:, prefix_length:], rtol=1e-5, atol=5e-7)
            assert saved.get_seq_length() == prefix_length
            assert fork.get_seq_length() == ids.shape[1]


@pytest.mark.parametrize("embeddings", [False, True])
def test_generation_positions_use_prepared_query_and_preserve_explicit_positions(embeddings):
    query = torch.zeros(1, 3, 8) if embeddings else torch.ones(1, 3, dtype=torch.long)
    key = "inputs_embeds" if embeddings else "input_ids"
    cache = SimpleNamespace(get_seq_length=lambda: 7)
    prepared = {key: query, "past_key_values": cache}
    position = generation_cache_position(prepared)
    torch.testing.assert_close(position, torch.tensor([7, 8, 9]), rtol=0, atol=0)
    assert prepared["cache_position"] is position
    explicit = torch.tensor([41, 42, 43])
    prepared["cache_position"] = explicit
    assert generation_cache_position(prepared) is explicit
    prepared = {key: query}
    torch.testing.assert_close(generation_cache_position(prepared), torch.arange(3), rtol=0, atol=0)


def test_sdpa_cached_suffix_masks_padding_keys_and_keeps_explicit_mask():
    cache = DynamicCache()
    cache.update(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8), 0)
    embeds = torch.zeros(1, 3, 8)
    positions = torch.arange(4, 7)
    padding = torch.tensor([[0, 1, 1, 1, 1, 1, 0]])
    config = SimpleNamespace(_attn_implementation="sdpa")
    actual = _qwen3_vl_causal_mask(config, embeds, padding, positions, cache)
    allowed = (torch.arange(7)[None, :] <= positions[:, None]) & padding.bool()
    expected = torch.where(allowed, 0.0, torch.finfo(embeds.dtype).min)[None, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert _qwen3_vl_causal_mask(config, embeds, actual, positions, cache) is actual


def test_sdpa_uncached_right_padded_training_mask_is_unchanged():
    embeds = torch.zeros(2, 4, 8, requires_grad=True)
    padding = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert (
        _qwen3_vl_causal_mask(SimpleNamespace(_attn_implementation="sdpa"), embeds, padding, torch.arange(4), None)
        is None
    )


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
def test_native_sdpa_left_padding_matches_unpadded_and_split_prefix(family):
    net = _net(family, True)
    ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]])
    mask = ids.ne(0).long()
    with torch.inference_mode():
        full = net(input_ids=ids, attention_mask=mask, cache_position=torch.arange(7), use_cache=False).logits
        for row, start in [(0, 3), (1, 0)]:
            single = ids[row : row + 1, start:]
            out = net(
                input_ids=single,
                attention_mask=torch.ones_like(single),
                cache_position=torch.arange(single.shape[1]),
                use_cache=False,
            ).logits
            torch.testing.assert_close(out, full[row : row + 1, start:], rtol=1e-5, atol=5e-7)
        prefix = net(input_ids=ids[:, :4], attention_mask=mask[:, :4], cache_position=torch.arange(4), use_cache=True)
        suffix = net(
            input_ids=ids[:, 4:],
            attention_mask=mask,
            cache_position=torch.arange(4, 7),
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits
        torch.testing.assert_close(suffix, full[:, 4:], rtol=1e-5, atol=5e-7)
