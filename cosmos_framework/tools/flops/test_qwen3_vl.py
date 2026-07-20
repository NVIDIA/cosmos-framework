# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

from cosmos_framework.tools.flops.qwen3_vl import (
    compute_attention_flops,
    compute_layernorm_flops,
    compute_mlp_flops,
    compute_qwen3vl_flops,
    compute_qwen3vl_flops_from_config,
    compute_text_decoder_flops,
)


@pytest.mark.L0
def test_attention_bias_controls_qkvo_projection_bias_terms() -> None:
    seq_len = 3
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4

    without_bias = compute_attention_flops(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        has_bias=False,
    )
    with_bias = compute_attention_flops(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        has_bias=True,
    )

    q_bias_flops = seq_len * num_heads * head_dim
    k_bias_flops = seq_len * num_kv_heads * head_dim
    v_bias_flops = seq_len * num_kv_heads * head_dim
    o_bias_flops = seq_len * hidden_size
    assert with_bias - without_bias == q_bias_flops + k_bias_flops + v_bias_flops + o_bias_flops


@pytest.mark.L0
def test_text_decoder_qk_norm_counts_query_and_kv_heads_for_gqa() -> None:
    total_tokens = 3
    hidden_size = 8
    intermediate_size = 16
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4

    actual = compute_text_decoder_flops(
        total_tokens=total_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_text_layers=1,
        head_dim=head_dim,
        is_causal=False,
        attention_bias=False,
    )

    attn_flops = compute_attention_flops(
        seq_len=total_tokens,
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        is_causal=False,
        has_bias=False,
    )
    mlp_flops = compute_mlp_flops(total_tokens, hidden_size, intermediate_size, use_swiglu=True)
    layernorm_flops = 2 * compute_layernorm_flops(total_tokens, hidden_size)
    qk_norm_flops = compute_layernorm_flops(
        total_tokens * (num_attention_heads + num_key_value_heads),
        head_dim,
    )

    assert actual == attn_flops + mlp_flops + layernorm_flops + qk_norm_flops


@pytest.mark.L0
def test_qwen3vl_flops_from_config_uses_attention_bias() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        attention_bias=False,
    )
    vision_config = SimpleNamespace(
        depth=0,
        hidden_size=4,
        intermediate_size=8,
        num_heads=1,
        spatial_merge_size=2,
    )
    config = SimpleNamespace(text_config=text_config, vision_config=vision_config)

    from_config = compute_qwen3vl_flops_from_config(
        config=config,
        total_tokens=5,
        visual_tokens=1,
        num_patches=None,
        is_causal=False,
    )
    direct_without_bias = compute_qwen3vl_flops(
        num_text_layers=text_config.num_hidden_layers,
        num_vision_layers=vision_config.depth,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        vision_hidden_size=vision_config.hidden_size,
        vision_intermediate_size=vision_config.intermediate_size,
        vision_num_heads=vision_config.num_heads,
        vocab_size=text_config.vocab_size,
        total_tokens=5,
        visual_tokens=1,
        num_patches=None,
        head_dim=text_config.head_dim,
        spatial_merge_size=vision_config.spatial_merge_size,
        is_causal=False,
        attention_bias=False,
    )
    direct_with_bias = compute_qwen3vl_flops(
        num_text_layers=text_config.num_hidden_layers,
        num_vision_layers=vision_config.depth,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        vision_hidden_size=vision_config.hidden_size,
        vision_intermediate_size=vision_config.intermediate_size,
        vision_num_heads=vision_config.num_heads,
        vocab_size=text_config.vocab_size,
        total_tokens=5,
        visual_tokens=1,
        num_patches=None,
        head_dim=text_config.head_dim,
        spatial_merge_size=vision_config.spatial_merge_size,
        is_causal=False,
        attention_bias=True,
    )

    assert from_config == direct_without_bias
    assert from_config["total_flops"] < direct_with_bias["total_flops"]
