# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Numerical parity gate for the reasoner/generator split used by CPU offloading.

The offloaded denoise path runs the understanding ("reasoner") tower once as a
prefill (``ReasonerMemoryState`` mode ``"prefill"``) that caches the per-layer
understanding K/V, then runs the generation tower alone (mode ``"gen"``) reusing
that cache. This test asserts the generator output of the split path matches the
standard joint forward on identical inputs, so enabling offloading cannot change
results.

Uses a tiny dense ``Qwen3VLTextModel`` in fp32 (no checkpoint required) and the
``two_way`` attention path (the only layout the split supports).
"""

from typing import cast

import torch

from cosmos_framework.data.vfm.sequence_packing import get_gen_seq
from cosmos_framework.model.vfm.mot.attention import build_packed_sequence
from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLTextModel
from cosmos_framework.model.vfm.utils.memory import ReasonerMemoryState
from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_tiny_text_model(device: torch.device) -> Qwen3VLTextConfig:
    # head_dim * num_attention_heads == hidden_size; small dims for a fast test.
    config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_scaling=None,
        rms_norm_eps=1e-6,
    )
    model = Qwen3VLTextModel(config, qk_norm_for_text=True, qk_norm_for_diffusion=True)
    return model.to(device=device, dtype=torch.float32).eval()


def _build_two_way_pack(model: Qwen3VLTextModel, device: torch.device, und_len: int = 6, gen_len: int = 10):
    """Single-sample two_way pack: a causal understanding block then a full generation block."""
    cfg = model.config
    total = und_len + gen_len
    hidden = torch.randn(total, cfg.hidden_size, device=device, dtype=torch.float32)

    packed_und_token_indexes = cast(torch.LongTensor, torch.arange(0, und_len, device=device, dtype=torch.long))
    packed_gen_token_indexes = cast(torch.LongTensor, torch.arange(und_len, total, device=device, dtype=torch.long))
    position_ids = torch.arange(total, device=device, dtype=torch.int32)

    input_pack, attention_meta, natten = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=["causal", "full"],
        split_lens=[und_len, gen_len],
        sample_lens=[total],
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
        num_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        num_layers=cfg.num_hidden_layers,
        token_shapes=[(1, 1, gen_len)],
    )
    assert natten is None
    return input_pack, attention_meta, position_ids


def test_gen_only_matches_joint():
    device = _device()
    torch.manual_seed(0)
    model = _build_tiny_text_model(device)
    input_pack, attention_meta, position_ids = _build_two_way_pack(model, device)

    with torch.no_grad():
        joint_out, _ = model(input_pack, attention_mask=attention_meta, position_ids=position_ids, memory=None)
        gen_joint = get_gen_seq(joint_out)

        memory = ReasonerMemoryState(model.config.num_hidden_layers)
        memory.set_mode("prefill")
        model(input_pack, attention_mask=attention_meta, position_ids=position_ids, memory=memory)
        assert memory.is_initialized, "reasoner K/V cache not fully populated after prefill"

        memory.set_mode("gen")
        split_out, _ = model(input_pack, attention_mask=attention_meta, position_ids=position_ids, memory=memory)
        gen_split = get_gen_seq(split_out)

    torch.testing.assert_close(gen_split, gen_joint, atol=1e-4, rtol=1e-4)


def test_joint_path_unchanged_when_memory_none():
    """memory=None must be deterministic and identical run-to-run (joint path untouched)."""
    device = _device()
    torch.manual_seed(0)
    model = _build_tiny_text_model(device)
    input_pack, attention_meta, position_ids = _build_two_way_pack(model, device)

    with torch.no_grad():
        out1, _ = model(input_pack, attention_mask=attention_meta, position_ids=position_ids, memory=None)
        out2, _ = model(input_pack, attention_mask=attention_meta, position_ids=position_ids, memory=None)

    torch.testing.assert_close(get_gen_seq(out1), get_gen_seq(out2), atol=0.0, rtol=0.0)
