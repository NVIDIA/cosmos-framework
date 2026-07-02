# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""
Unit tests for the und_k_norm_for_gen QK-norm fix in PackedAttentionMoT.

Background
----------
Nemotron-3-2B has qk_norm_for_text=False (reasoner: Identity) but
qk_norm_for_diffusion=True (generator: RMSNorm).  In packed joint attention
this means the generator's full attention computes norm(Q_gen) · K_und_raw^T
where K_und_raw has uncontrolled magnitude, dominating the attention logits.

The fix (use_und_k_norm_for_gen=True) adds k_norm_und_for_gen (RMSNorm) that
normalises K_und specifically for the gen→und cross-attention path, while the
reasoner's own causal self-attention continues to use raw K_und.

Tests
-----
1. Init: k_norm_und_for_gen is None when disabled or for Qwen3 (both norms present).
2. Backward compat: passing packed_key_states_normalized=None is a no-op.
3. Gen scale invariance (two_way): gen output unchanged when K_und is scaled but
   normed keys are provided.
4. Reasoner uses raw K (two_way): reasoner output changes when K_und is scaled.
5. Gen scale invariance (three_way): same as (3) but for three_way attention.
"""

import pytest
import torch

from cosmos_framework.model.generator.mot.attention import (
    build_packed_sequence,
    three_way_attention,
    two_way_attention,
)
from cosmos_framework.model.generator.mot.unified_mot import (
    LayerTypes,
    PackedAttentionMoT,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    Nemotron3DenseVLRMSNorm,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.data.generator.sequence_packing.runtime import (
    get_gen_seq,
    get_und_seq,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64
SEQS_PER_BATCH = 2
MAX_SEQ_LEN = 16
UND_SCALE = 100.0  # scale factor used to test K_und invariance


def _make_tiny_nemotron_config() -> Nemotron3DenseVLTextConfig:
    return Nemotron3DenseVLTextConfig(
        hidden_size=NUM_Q_HEADS * HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=1,
        attention_bias=False,
    )


def _make_tiny_qwen3_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        hidden_size=NUM_Q_HEADS * HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=1,
        attention_bias=False,
    )


def _make_packs(device: torch.device, impl: str, seed: int = 42):
    """
    Build minimal QKV packs for two_way / three_way attention tests.

    Returns
    -------
    q_pack, k_pack_raw, v_pack, k_pack_normed, split_info, packed_und_idx
        k_pack_raw   : K pack where causal (und) tokens have scale UND_SCALE
        k_pack_normed: K pack where causal (und) tokens are RMS-normalised
        packed_und_idx: LongTensor of und token positions in the flat sequence
    """
    torch.manual_seed(seed)
    sample_lens = [10, 8]
    full_len = sum(sample_lens)

    # Build und/gen splits: first half of each sample → und (causal), rest → gen (full)
    split_lens = []
    attn_modes = []
    packed_und_idx = []
    packed_gen_idx = []
    token_shapes = []
    pos = 0
    for slen in sample_lens:
        und_len = slen // 2
        gen_len = slen - und_len
        split_lens.extend([und_len, gen_len])
        attn_modes.extend(["causal", "full"])
        packed_und_idx.extend(range(pos, pos + und_len))
        packed_gen_idx.extend(range(pos + und_len, pos + slen))
        token_shapes.append((1, gen_len))  # (H, W) for three_way NATTEN metadata
        pos += slen

    und_idx_t = torch.tensor(packed_und_idx, dtype=torch.long, device=device)
    gen_idx_t = torch.tensor(packed_gen_idx, dtype=torch.long, device=device)

    def make_pack(tensor):
        pack, meta, _ = build_packed_sequence(
            impl,
            packed_sequence=tensor,
            attn_modes=attn_modes,
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=und_idx_t,
            packed_gen_token_indexes=gen_idx_t,
            num_heads=NUM_Q_HEADS,
            head_dim=HEAD_DIM,
            num_layers=1,
            token_shapes=token_shapes,
        )
        return pack, meta

    q_raw = torch.randn(full_len, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_raw = torch.randn(full_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_raw = torch.randn(full_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

    # Scale up und portion of K to simulate large-magnitude und keys (the bug scenario)
    k_scaled = k_raw.clone()
    k_scaled[und_idx_t] = k_scaled[und_idx_t] * UND_SCALE

    # Manually RMS-normalise the und portion for the normed pack
    k_normed_flat = k_scaled.clone()
    k_und = k_normed_flat[und_idx_t].float()  # [N_und, kv_heads, head_dim]
    rms = k_und.pow(2).mean(-1, keepdim=True).sqrt() + 1e-5
    k_normed_flat[und_idx_t] = (k_und / rms).to(k_raw.dtype)

    q_pack, split_info = make_pack(q_raw)
    k_pack_raw, _ = make_pack(k_scaled)
    v_pack, _ = make_pack(v_raw)
    k_pack_normed, _ = make_pack(k_normed_flat)

    # Also build an unscaled k pack (for backward-compat comparison)
    k_pack_unscaled, _ = make_pack(k_raw)

    return q_pack, k_pack_raw, v_pack, k_pack_normed, k_pack_unscaled, split_info, und_idx_t


# ---------------------------------------------------------------------------
# 1. Init tests (CPU, L0)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_k_norm_und_for_gen_disabled_by_default():
    """k_norm_und_for_gen must be None when use_und_k_norm_for_gen=False (default)."""
    config = _make_tiny_nemotron_config()
    layer_types = LayerTypes("nemotron_dense")
    attn = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=layer_types,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=False,
    )
    assert attn.k_norm_und_for_gen is None


@pytest.mark.L0
@pytest.mark.CPU
def test_k_norm_und_for_gen_not_created_when_both_pathways_have_norm():
    """k_norm_und_for_gen must be None when qk_norm_for_text=True (Qwen3 config).

    When the reasoner already has QK norm, K_und is already normalised — no fix needed.
    """
    config = _make_tiny_qwen3_config()
    layer_types = LayerTypes("qwen3_vl_dense")
    attn = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=layer_types,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,  # flag on, but condition not met
    )
    assert attn.k_norm_und_for_gen is None


@pytest.mark.L0
@pytest.mark.CPU
def test_k_norm_und_for_gen_created_for_nemotron_when_enabled():
    """k_norm_und_for_gen must be an RMSNorm with weight=1 for Nemotron + flag enabled."""
    config = _make_tiny_nemotron_config()
    layer_types = LayerTypes("nemotron_dense")
    attn = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=layer_types,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
    )
    assert attn.k_norm_und_for_gen is not None
    assert isinstance(attn.k_norm_und_for_gen, Nemotron3DenseVLRMSNorm)
    assert attn.k_norm_und_for_gen.weight.shape == (attn.head_dim,)
    # weight initialised to 1 → pure RMSNorm at step 0
    torch.testing.assert_close(attn.k_norm_und_for_gen.weight, torch.ones(attn.head_dim))


# ---------------------------------------------------------------------------
# 2. Backward compat: packed_key_states_normalized=None is a no-op (GPU, L0)
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_two_way_attention_normed_none_is_noop():
    """two_way_attention output must be identical when packed_key_states_normalized=None."""
    device = torch.device("cuda")
    q_pack, k_pack_raw, v_pack, _, _, split_info, _ = _make_packs(device, "two_way")

    out_baseline = two_way_attention(q_pack, k_pack_raw, v_pack)
    out_explicit_none = two_way_attention(q_pack, k_pack_raw, v_pack, packed_key_states_normalized=None)

    torch.testing.assert_close(
        get_und_seq(out_baseline),
        get_und_seq(out_explicit_none),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        get_gen_seq(out_baseline),
        get_gen_seq(out_explicit_none),
        atol=0,
        rtol=0,
    )


@pytest.mark.L0
def test_three_way_attention_normed_none_is_noop():
    """three_way_attention output must be identical when packed_key_states_normalized=None."""
    device = torch.device("cuda")
    q_pack, k_pack_raw, v_pack, _, _, split_info, _ = _make_packs(device, "three_way")

    out_baseline = three_way_attention(q_pack, k_pack_raw, v_pack, natten_metadata=None)
    out_explicit_none = three_way_attention(
        q_pack, k_pack_raw, v_pack, natten_metadata=None, packed_key_states_normalized=None
    )

    torch.testing.assert_close(get_und_seq(out_baseline), get_und_seq(out_explicit_none), atol=0, rtol=0)
    torch.testing.assert_close(get_gen_seq(out_baseline), get_gen_seq(out_explicit_none), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 3 & 4. Scale invariance and reasoner raw-K tests (GPU, L1)
# ---------------------------------------------------------------------------


@pytest.mark.L1
def test_two_way_gen_output_independent_of_raw_und_key_scale():
    """Gen output must be identical when raw K_und scale changes but normed keys are fixed.

    When packed_key_states_normalized is provided, two_way_attention routes gen full-attention
    through get_all_seq(packed_key_states_normalized), not packed_key_states.  So regardless
    of whether packed_key_states has 1x or 100x K_und, the gen output is the same.
    Reasoner output (causal path) uses packed_key_states and therefore differs.
    """
    device = torch.device("cuda")
    q_pack, k_pack_raw, v_pack, k_pack_normed, k_pack_unscaled, _, _ = _make_packs(device, "two_way")

    # Both calls use the SAME k_pack_normed → gen sees identical keys → identical gen output.
    out_raw = two_way_attention(q_pack, k_pack_raw, v_pack, packed_key_states_normalized=k_pack_normed)
    out_unscaled = two_way_attention(q_pack, k_pack_unscaled, v_pack, packed_key_states_normalized=k_pack_normed)

    # Gen outputs must be bitwise identical (same normed keys).
    torch.testing.assert_close(
        get_gen_seq(out_raw),
        get_gen_seq(out_unscaled),
        atol=0,
        rtol=0,
        msg="Gen output must be identical when only raw K_und scale differs (normed keys fixed)",
    )

    # Reasoner outputs must differ (different raw K_und: 100x vs 1x).
    und_diff = (get_und_seq(out_raw) - get_und_seq(out_unscaled)).abs().max()
    assert und_diff > 1e-2, f"Reasoner output should differ with scaled K_und (max diff {und_diff:.4f})"


@pytest.mark.L1
def test_three_way_gen_output_independent_of_raw_und_key_scale():
    """Same property as two_way: gen output unchanged when raw K_und differs but normed keys fixed."""
    device = torch.device("cuda")
    q_pack, k_pack_raw, v_pack, k_pack_normed, k_pack_unscaled, _, _ = _make_packs(device, "three_way")

    out_raw = three_way_attention(
        q_pack, k_pack_raw, v_pack, natten_metadata=None, packed_key_states_normalized=k_pack_normed
    )
    out_unscaled = three_way_attention(
        q_pack, k_pack_unscaled, v_pack, natten_metadata=None, packed_key_states_normalized=k_pack_normed
    )

    # Gen outputs must be bitwise identical (same normed keys).
    torch.testing.assert_close(
        get_gen_seq(out_raw),
        get_gen_seq(out_unscaled),
        atol=0,
        rtol=0,
        msg="Gen output must be identical when only raw K_und scale differs (three_way, normed keys fixed)",
    )

    # Reasoner outputs must differ.
    und_diff = (get_und_seq(out_raw) - get_und_seq(out_unscaled)).abs().max()
    assert und_diff > 1e-2, f"Reasoner output should differ with scaled K_und (max diff {und_diff:.4f})"
