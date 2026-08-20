# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for ``include_gen_pathway``.

Reasoner-only inference never runs the MoT generation tower, so it can leave the
``*_moe_gen`` duplicates unbuilt.  ``dcp.load`` is pull-based (it requests only
the keys the live model exposes), so a model built without those modules also
never reads their tensors off disk.

The flag defaults to ``True`` everywhere; these tests pin both the disabled
behaviour and the unchanged default.
"""

import torch.nn as nn

from cosmos_framework.model.generator.mot.unified_mot import (
    LayerTypes,
    MoTDecoderLayer,
    Nemotron3DenseVLMoTConfig,
    PackedAttentionMoT,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)

NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16

# Every generation-pathway module gated by the flag, by owning class.
_ATTN_GEN_MODULES = (
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
    "q_norm_moe_gen",
    "k_norm_moe_gen",
)
_LAYER_GEN_MODULES = (
    "mlp_moe_gen",
    "input_layernorm_moe_gen",
    "post_attention_layernorm_moe_gen",
)


def _tiny_config() -> Nemotron3DenseVLTextConfig:
    return Nemotron3DenseVLTextConfig(
        hidden_size=NUM_Q_HEADS * HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=1,
        attention_bias=False,
    )


def _make_attention(*, include_gen_pathway: bool | None = None) -> PackedAttentionMoT:
    kwargs = {} if include_gen_pathway is None else {"include_gen_pathway": include_gen_pathway}
    return PackedAttentionMoT(
        _tiny_config(),
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        **kwargs,
    )


def _make_layer(*, include_gen_pathway: bool | None = None) -> MoTDecoderLayer:
    kwargs = {} if include_gen_pathway is None else {"include_gen_pathway": include_gen_pathway}
    return MoTDecoderLayer(
        config=_tiny_config(),
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        **kwargs,
    )


def test_attention_omits_gen_modules_when_disabled() -> None:
    attn = _make_attention(include_gen_pathway=False)

    for name in _ATTN_GEN_MODULES:
        assert not hasattr(attn, name), f"{name} should not be built when include_gen_pathway=False"
    # The cross-attention K norm only exists to serve the generation pathway.
    assert attn.k_norm_und_for_gen is None
    # The understanding pathway is untouched.
    assert isinstance(attn.q_proj, nn.Linear)
    assert isinstance(attn.o_proj, nn.Linear)


def test_attention_builds_gen_modules_by_default() -> None:
    attn = _make_attention()

    for name in _ATTN_GEN_MODULES:
        assert hasattr(attn, name), f"{name} must still be built by default"


def test_decoder_layer_omits_gen_modules_when_disabled() -> None:
    layer = _make_layer(include_gen_pathway=False)

    for name in _LAYER_GEN_MODULES:
        assert not hasattr(layer, name), f"{name} should not be built when include_gen_pathway=False"
    # Nothing anywhere in the layer (including the nested attention) carries gen weights.
    assert not [name for name, _ in layer.named_parameters() if "moe_gen" in name]
    # The understanding pathway still has its full parameter set.
    assert [name for name, _ in layer.named_parameters() if name.startswith("mlp.")]


def test_decoder_layer_builds_gen_modules_by_default() -> None:
    layer = _make_layer()

    for name in _LAYER_GEN_MODULES:
        assert hasattr(layer, name), f"{name} must still be built by default"
    assert [name for name, _ in layer.named_parameters() if "moe_gen" in name]


def test_mot_config_includes_gen_pathway_by_default() -> None:
    assert Nemotron3DenseVLMoTConfig({}).include_gen_pathway is True


def test_mot_config_forwards_disabled_flag() -> None:
    assert Nemotron3DenseVLMoTConfig({}, include_gen_pathway=False).include_gen_pathway is False
