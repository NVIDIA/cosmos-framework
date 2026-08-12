# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Mock heavy external deps before importing any flops module.
for _mod in ["torch", "torch.distributed", "loguru", "loguru._logger"]:
    sys.modules.setdefault(_mod, MagicMock())
_mock_log = MagicMock()
_mock_utils = MagicMock()
_mock_utils.log = _mock_log
sys.modules.setdefault("cosmos_framework", MagicMock())
sys.modules.setdefault("cosmos_framework.utils", _mock_utils)
sys.modules.setdefault("cosmos_framework.utils.log", _mock_log)

from cosmos_framework.tools.flops.omni_mot import (
    OmniMoTModelDescriptor,
    _compute_per_sample_attn_flops,
    _extract_padding_tokens,
    compute_omni_mot_flops_per_batch,
    get_omni_mot_model_descriptor,
)
from cosmos_framework.tools.flops.qwen3_vl import (
    compute_attention_flops,
    compute_layernorm_flops,
    compute_mlp_flops,
    compute_qwen3vl_flops,
    compute_qwen3vl_flops_from_config,
    compute_text_decoder_flops,
)

# ---------------------------------------------------------------------------
# Shared tiny config: D=8, 1 layer, 2 Q-heads, 2 KV-heads, head_dim=4
# ---------------------------------------------------------------------------


def _mini_cfg(**overrides: object) -> object:
    defaults: dict[str, object] = dict(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        vocab_size=32,
        use_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        decoder_sparse_step=1,
        latent_patch_size=2,
        latent_channel_size=4,
        action_dim=8,
        sound_dim=8,
        frequency_embedding_size=8,
        predict_text_tokens=False,
    )
    defaults.update(overrides)
    return get_omni_mot_model_descriptor(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _extract_padding_tokens
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_extract_padding_tokens_complete_pairs_returns_zero() -> None:
    # [causal, full, causal, full] — every causal is paired with a full
    assert _extract_padding_tokens([4, 4, 3, 2], ["causal", "full", "causal", "full"]) == 0


@pytest.mark.L0
def test_extract_padding_tokens_trailing_causal_is_padding() -> None:
    # [causal, full, causal] — the trailing lone causal is padding
    assert _extract_padding_tokens([4, 4, 5], ["causal", "full", "causal"]) == 5


@pytest.mark.L0
def test_extract_padding_tokens_lone_causal_only() -> None:
    # A single lone causal entry — the whole thing is padding
    assert _extract_padding_tokens([7], ["causal"]) == 7


# ---------------------------------------------------------------------------
# _compute_per_sample_attn_flops
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_per_sample_attn_flops() -> None:
    # Batch mode: S_gen=0 → gen_attn must be zero; und uses the causal S^2 formula
    und_attn, gen_attn = _compute_per_sample_attn_flops(n_heads=2, d_head=4, B=1, S_und=4, S_gen=0)
    assert und_attn == Decimal(4 * 1 * 2 * 4 * 4 * 4)
    assert gen_attn == Decimal(0)

    # Packed mode: two samples (und=4, gen=4) and (und=3, gen=2)
    n, d = 2, 4
    und_attn, gen_attn = _compute_per_sample_attn_flops(
        n_heads=n,
        d_head=d,
        B=1,
        S_und=7,  # total across samples (not used in packed mode)
        S_gen=6,
        split_lens=[4, 4, 3, 2],
        attn_modes=["causal", "full", "causal", "full"],
    )
    assert und_attn == Decimal(4 * n * d * (4 * 4 + 3 * 3))
    assert gen_attn == Decimal(4 * n * d * (4 * (4 + 4) + 2 * (3 + 2)))


# ---------------------------------------------------------------------------
# compute_omni_mot_flops_per_batch
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_forward_only_golden_value_no_gen_tokens() -> None:
    """Verify the exact forward FLOPs for a minimal dense config with no gen tokens.

    Manual derivation (D=8, n_heads=2, n_kv_heads=2, d_head=4, intermediate=16,
    B=1, S_und=4, S_gen=0, 1 layer):

      q_dim = kv_dim = 8

      attn_und_proj  = 3 × 2×1×4×8×8               = 1 536   (Q, K, V projections)
      und_attn_dot   = 4×1×2×4×4×4                  =   512
      attn_o_proj    = 2×1×4×8×8                    =   512
      attn_qk_norm   = 2 × (5×1×4×2×4)             =   320   (Q-norm + KV-norm, und only)
      und_softmax    = 512 × (5/16)                 =   160
      layer_attn     = 1536 + 320 + 512 + 512 + 160 = 3 040

      Wait — layer_attn = attn_und_proj + attn_gen_proj + attn_qk_norm + attn_dot + attn_o_proj
                        = 1536 + 0 + 320 + 512 + 512  = 2 880
      layer_softmax  = 160
      dense_mlp(4)   = compute_mlp_flops(4,8,16)    = 3 264  (gate+up+down=3072, act+mul=192)
      layer_norm     = 2 × (5×1×4×8)               =   320
      layer_flops    = 2880 + 3264 + 320 + 160      = 6 624

      final_norm     = 5×1×4×8                      =   160
      fp             = 6624 + 160                   = 6 784
    """
    cfg = _mini_cfg()
    total = compute_omni_mot_flops_per_batch(cfg, B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=0.0)
    assert total == Decimal(6784)


@pytest.mark.L0
def test_total_equals_forward_times_one_plus_backward_ratio() -> None:
    """total = fp × (1 + ratio) when freeze_und=False and AC=False."""
    cfg = _mini_cfg()
    fp = compute_omni_mot_flops_per_batch(cfg, B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=0.0)
    total = compute_omni_mot_flops_per_batch(cfg, B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=2.0)
    assert total == 3 * fp


@pytest.mark.L0
def test_activation_checkpointing_adds_block_flops() -> None:
    """AC raises the total by exactly total_block_flops.

    With no embedding (vision_tokens=0) the forward pass is:
      fp = block_flops + final_norm
    so  block_flops = fp - final_norm = total(ratio=0) - 5×B×S_und×D.
    """
    cfg = _mini_cfg()
    B, text_tokens = 1, 4
    final_norm = Decimal(5 * B * text_tokens * cfg.hidden_size)  # 160

    total_no_ac = compute_omni_mot_flops_per_batch(
        cfg, B=B, text_tokens=text_tokens, vision_tokens=0, backwardpass_ratio=0.0
    )
    total_ac = compute_omni_mot_flops_per_batch(
        cfg,
        B=B,
        text_tokens=text_tokens,
        vision_tokens=0,
        backwardpass_ratio=0.0,
        use_activation_checkpointing=True,
    )
    expected_block_flops = total_no_ac - final_norm
    assert total_ac - total_no_ac == expected_block_flops


@pytest.mark.L0
def test_freeze_und_with_no_gen_tokens_gives_zero_backward() -> None:
    """When S_gen=0 and freeze_und=True, the backward pass contributes zero FLOPs."""
    cfg = _mini_cfg()
    fp = compute_omni_mot_flops_per_batch(cfg, B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=0.0)
    total_freeze = compute_omni_mot_flops_per_batch(
        cfg, B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=2.0, freeze_und=True
    )
    # No gen tokens → gen pathway has zero cost → backward is zero → total = fp
    assert total_freeze == fp


@pytest.mark.L0
def test_freeze_und_with_gen_tokens_reduces_total_vs_full_backward() -> None:
    """freeze_und=True saves backward FLOPs when gen tokens are present."""
    cfg = _mini_cfg()
    kwargs: dict[str, object] = dict(B=1, text_tokens=4, action_tokens=4, action_gen=True, backwardpass_ratio=2.0)
    total_full = compute_omni_mot_flops_per_batch(cfg, freeze_und=False, **kwargs)  # type: ignore[arg-type]
    total_freeze = compute_omni_mot_flops_per_batch(cfg, freeze_und=True, **kwargs)  # type: ignore[arg-type]
    assert total_freeze < total_full


@pytest.mark.L0
def test_moe_layer_uses_different_flops_than_dense() -> None:
    """MoE config produces different MLP FLOPs than the equivalent dense config."""
    dense_cfg = _mini_cfg(use_moe=False)
    moe_cfg = _mini_cfg(
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        decoder_sparse_step=1,
    )
    kwargs: dict[str, object] = dict(B=1, text_tokens=4, vision_tokens=0, backwardpass_ratio=0.0)
    total_dense = compute_omni_mot_flops_per_batch(dense_cfg, **kwargs)  # type: ignore[arg-type]
    total_moe = compute_omni_mot_flops_per_batch(moe_cfg, **kwargs)  # type: ignore[arg-type]
    assert total_dense != total_moe


@pytest.mark.L0
def test_text_prediction_adds_lm_head_flops() -> None:
    """predict_text_tokens=True adds 2×B×text_tokens×D×vocab_size to forward FLOPs."""
    cfg_no_pred = _mini_cfg(predict_text_tokens=False)
    cfg_pred = _mini_cfg(predict_text_tokens=True)
    B, text_tokens = 1, 4
    fp_no_pred = compute_omni_mot_flops_per_batch(
        cfg_no_pred, B=B, text_tokens=text_tokens, vision_tokens=0, backwardpass_ratio=0.0
    )
    fp_pred = compute_omni_mot_flops_per_batch(
        cfg_pred, B=B, text_tokens=text_tokens, vision_tokens=0, backwardpass_ratio=0.0
    )
    expected_lm_head = Decimal(2 * B * text_tokens * cfg_pred.hidden_size * cfg_pred.vocab_size)
    assert fp_pred - fp_no_pred == expected_lm_head


# ---------------------------------------------------------------------------
# qwen3_vl
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_attention_bias_controls_qkvo_projection_bias_terms() -> None:
    seq_len, hidden_size, num_heads, num_kv_heads, head_dim = 3, 8, 2, 1, 4

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

    expected_bias_delta = (
        seq_len * num_heads * head_dim  # Q bias
        + seq_len * num_kv_heads * head_dim  # K bias
        + seq_len * num_kv_heads * head_dim  # V bias
        + seq_len * hidden_size  # O bias
    )
    assert with_bias - without_bias == expected_bias_delta


@pytest.mark.L0
def test_text_decoder_qk_norm_counts_query_and_kv_heads_for_gqa() -> None:
    total_tokens, hidden_size, intermediate_size = 3, 8, 16
    num_attention_heads, num_key_value_heads, head_dim = 2, 1, 4

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

    expected = (
        compute_attention_flops(
            seq_len=total_tokens,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            is_causal=False,
            has_bias=False,
        )
        + compute_mlp_flops(total_tokens, hidden_size, intermediate_size, use_swiglu=True)
        + 2 * compute_layernorm_flops(total_tokens, hidden_size)
        + compute_layernorm_flops(total_tokens * (num_attention_heads + num_key_value_heads), head_dim)
    )
    assert actual == expected


@pytest.mark.L0
def test_qwen3vl_flops_from_config_reads_attention_bias() -> None:
    text_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        attention_bias=False,
    )
    vision_cfg = SimpleNamespace(depth=0, hidden_size=4, intermediate_size=8, num_heads=1, spatial_merge_size=2)
    config = SimpleNamespace(text_config=text_cfg, vision_config=vision_cfg)

    kwargs = dict(total_tokens=5, visual_tokens=1, num_patches=None, is_causal=False)
    from_config = compute_qwen3vl_flops_from_config(config=config, **kwargs)  # type: ignore[arg-type]
    direct_false = compute_qwen3vl_flops(
        num_text_layers=1,
        num_vision_layers=0,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_hidden_size=4,
        vision_intermediate_size=8,
        vision_num_heads=1,
        vocab_size=32,
        head_dim=4,
        spatial_merge_size=2,
        attention_bias=False,
        **kwargs,  # type: ignore[arg-type]
    )
    direct_true = compute_qwen3vl_flops(
        num_text_layers=1,
        num_vision_layers=0,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_hidden_size=4,
        vision_intermediate_size=8,
        vision_num_heads=1,
        vocab_size=32,
        head_dim=4,
        spatial_merge_size=2,
        attention_bias=True,
        **kwargs,  # type: ignore[arg-type]
    )

    assert from_config == direct_false
    assert from_config["total_flops"] < direct_true["total_flops"]


# ---------------------------------------------------------------------------
# gen_moe_shared_expert
# ---------------------------------------------------------------------------


def _make_shared_expert_cfg(
    *,
    shared_expert: bool,
    shared_expert_scale: int = 1,
    use_moe: bool = True,
) -> OmniMoTModelDescriptor:
    return get_omni_mot_model_descriptor(
        hidden_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        use_moe=use_moe,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        decoder_sparse_step=2,
        mlp_only_layers=[],
        gen_moe_shared_expert=shared_expert,
        gen_moe_shared_expert_intermediate_scale=shared_expert_scale,
    )


def _compute_shared_expert(
    cfg: OmniMoTModelDescriptor,
    *,
    backwardpass_ratio: float = 0.0,
    freeze_und: bool = False,
    action_gen: bool = True,
    use_activation_checkpointing: bool = False,
) -> Decimal:
    return compute_omni_mot_flops_per_batch(
        cfg=cfg,
        B=2,
        text_tokens=5,
        vision_tokens=0,
        action_tokens=7,
        vision_gen=False,
        action_gen=action_gen,
        freeze_und=freeze_und,
        backwardpass_ratio=backwardpass_ratio,
        use_activation_checkpointing=use_activation_checkpointing,
    )


def _expected_shared_expert_forward_flops(scale: int = 1) -> Decimal:
    # Shared expert uses SwiGLU matmuls only (gate+up+down), omitting activation/adds.
    batch_size = 2
    gen_tokens = 7
    hidden_size = 8
    shared_intermediate_size = 4 * scale
    num_moe_layers = 2
    return Decimal(num_moe_layers * 6 * batch_size * gen_tokens * hidden_size * shared_intermediate_size)


@pytest.mark.L0
def test_shared_expert_adds_gen_only_swiglu_flops() -> None:
    disabled = _compute_shared_expert(_make_shared_expert_cfg(shared_expert=False))
    enabled = _compute_shared_expert(_make_shared_expert_cfg(shared_expert=True))

    assert enabled - disabled == _expected_shared_expert_forward_flops()


@pytest.mark.L0
def test_shared_expert_intermediate_scale_scales_flops_linearly() -> None:
    baseline = _compute_shared_expert(_make_shared_expert_cfg(shared_expert=False))
    scale_one_delta = _compute_shared_expert(_make_shared_expert_cfg(shared_expert=True)) - baseline
    scale_two_delta = (
        _compute_shared_expert(_make_shared_expert_cfg(shared_expert=True, shared_expert_scale=2)) - baseline
    )

    assert scale_two_delta == 2 * scale_one_delta


@pytest.mark.L0
@pytest.mark.parametrize(
    ("use_moe", "action_gen"),
    [
        (False, True),
        (True, False),
    ],
)
def test_shared_expert_is_ignored_without_moe_layers_or_gen_tokens(use_moe: bool, action_gen: bool) -> None:
    disabled = _compute_shared_expert(
        _make_shared_expert_cfg(shared_expert=False, use_moe=use_moe), action_gen=action_gen
    )
    enabled = _compute_shared_expert(
        _make_shared_expert_cfg(shared_expert=True, use_moe=use_moe), action_gen=action_gen
    )

    assert enabled == disabled


@pytest.mark.L0
def test_shared_expert_is_included_in_frozen_und_backward_and_recomputation() -> None:
    disabled = _make_shared_expert_cfg(shared_expert=False)
    enabled = _make_shared_expert_cfg(shared_expert=True)
    expected_forward_delta = _expected_shared_expert_forward_flops()

    frozen_und_delta = _compute_shared_expert(
        enabled, backwardpass_ratio=2.0, freeze_und=True
    ) - _compute_shared_expert(
        disabled,
        backwardpass_ratio=2.0,
        freeze_und=True,
    )
    checkpointed_delta = _compute_shared_expert(enabled, use_activation_checkpointing=True) - _compute_shared_expert(
        disabled,
        use_activation_checkpointing=True,
    )

    assert frozen_und_delta == 3 * expected_forward_delta
    assert checkpointed_delta == 2 * expected_forward_delta
