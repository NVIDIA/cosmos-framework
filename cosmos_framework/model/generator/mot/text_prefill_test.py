# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Text K/V prefill preserves packed positions, CFG isolation and cached denoising."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
    prefill_inference_text_kv,
)
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLTextModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

pytestmark = [pytest.mark.L0, pytest.mark.GPU, pytest.mark.gpus(1)]


def _model():
    config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
    )
    model = (
        Qwen3VLTextModel(
            config,
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=False,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    model.rotary_emb.float()
    return model, SimpleNamespace(language_model=SimpleNamespace(model=model))


def _pack(model, lengths):
    gen_len = 8
    ids = torch.randint(0, 128, (sum(lengths),), device="cuda")
    text_indices, gen_indices, sample_lens, modes, splits = [], [], [], [], []
    start = 0
    for length in lengths:
        text_indices.extend(range(start, start + length))
        gen_indices.extend(range(start + length, start + length + gen_len))
        sample_lens.append(length + gen_len)
        modes.extend(["causal", "full"])
        splits.extend([length, gen_len])
        start += length + gen_len
    text_indices = torch.tensor(text_indices, device="cuda", dtype=torch.long)
    gen_indices = torch.tensor(gen_indices, device="cuda", dtype=torch.long)
    hidden = torch.randn(start, 128, device="cuda", dtype=torch.bfloat16)
    hidden[text_indices] = model.embed_tokens(ids)
    # Non-contiguous, three-axis positions exercise modality boundary tokens.
    positions = torch.stack([torch.arange(start, device="cuda") * scale for scale in (1, 2, 3)])
    packed, mask, _ = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=modes,
        split_lens=splits,
        sample_lens=sample_lens,
        packed_und_token_indexes=text_indices,
        packed_gen_token_indexes=gen_indices,
        num_heads=2,
        head_dim=64,
        num_layers=len(model.layers),
        token_shapes=[(1, 1, gen_len)] * len(lengths),
    )
    template = SimpleNamespace(
        text_ids=ids,
        text_indexes=text_indices,
        position_ids=positions,
        sample_lens=sample_lens,
        sequence_length=start,
        attn_modes=modes,
        split_lens=splits,
    )
    return packed, mask, positions, template


@pytest.mark.parametrize("lengths", [(6,), (6, 3)])
@pytest.mark.parametrize("compile_layers", [False, True])
@torch.no_grad()
def test_prefill_matches_joint_and_skips_reasoner(lengths, compile_layers):
    model, net = _model()
    if compile_layers:
        for layer in model.layers:
            layer.compile(fullgraph=True, dynamic=False)
    pack, mask, positions, template = _pack(model, lengths)
    install_inference_memory_attention_dispatch(net)
    joint_caches = make_inference_text_kv_cache(2)
    reference, _ = model(
        pack, attention_mask=mask, position_ids=positions, memory=InferenceTextKVMemoryState(joint_caches)
    )
    caches = make_inference_text_kv_cache(2)
    prefill_inference_text_kv(net, template, caches)
    assert all(cache.is_initialized for cache in caches)
    for joint, prefilled in zip(joint_caches, caches, strict=True):
        for expected, actual in zip(joint.get(), prefilled.get(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Moving just the UND blocks to CPU makes accidental reasoner execution fail,
    # including projections on empty tensors. Residents remain on GPU.
    for layer in model.layers:
        for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
            getattr(layer, name).cpu()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            getattr(layer.self_attn, name).cpu()
    actual, _ = model(pack, attention_mask=mask, position_ids=positions, memory=InferenceTextKVMemoryState(caches))
    # Different attention GEMM shapes may change BF16 rounding; this is numerical
    # equivalence, not a claim of bitwise identity.
    torch.testing.assert_close(get_gen_seq(actual), get_gen_seq(reference), atol=0.02, rtol=0.02)


@torch.no_grad()
def test_prefill_rejects_reused_cache():
    model, net = _model()
    _, _, _, template = _pack(model, (6,))
    install_inference_memory_attention_dispatch(net)
    caches = make_inference_text_kv_cache(2)
    prefill_inference_text_kv(net, template, caches)
    with pytest.raises(ValueError, match="empty cache"):
        prefill_inference_text_kv(net, template, caches)


@torch.no_grad()
def test_prefill_empty_prompt():
    model, net = _model()
    _, _, _, template = _pack(model, (0,))
    caches = make_inference_text_kv_cache(2)
    prefill_inference_text_kv(net, template, caches)
    for cache in caches:
        key, value = cache.get()
        assert key.shape == value.shape == (1, 0, 1, 64)
