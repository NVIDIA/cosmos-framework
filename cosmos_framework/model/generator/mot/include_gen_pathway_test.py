# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for reciprocal MoT pathway inclusion flags.

Reasoner-only inference never runs the MoT generation tower, so it can leave the
``*_moe_gen`` duplicates unbuilt.  ``dcp.load`` is pull-based (it requests only
the keys the live model exposes), so a model built without those modules also
never reads their tensors off disk.

The flag defaults to ``True`` everywhere; these tests pin both the disabled
behaviour and the unchanged default.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.unified_mot import (
    LayerTypes,
    MoTDecoderLayer,
    Nemotron3DenseVLMoTConfig,
    Nemotron3DenseVLTextForCausalLM,
    PackedAttentionMoT,
    prune_und_pathway_,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue

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
_ATTN_UND_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
_LAYER_UND_MODULES = ("mlp", "input_layernorm", "post_attention_layernorm")


def _tiny_config() -> Nemotron3DenseVLTextConfig:
    return Nemotron3DenseVLTextConfig(
        hidden_size=NUM_Q_HEADS * HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=1,
        attention_bias=False,
    )


def _make_attention(
    *, include_gen_pathway: bool | None = None, include_und_pathway: bool | None = None
) -> PackedAttentionMoT:
    kwargs = {} if include_gen_pathway is None else {"include_gen_pathway": include_gen_pathway}
    if include_und_pathway is not None:
        kwargs["include_und_pathway"] = include_und_pathway
    return PackedAttentionMoT(
        _tiny_config(),
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        **kwargs,
    )


def _make_layer(*, include_gen_pathway: bool | None = None, include_und_pathway: bool | None = None) -> MoTDecoderLayer:
    kwargs = {} if include_gen_pathway is None else {"include_gen_pathway": include_gen_pathway}
    if include_und_pathway is not None:
        kwargs["include_und_pathway"] = include_und_pathway
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


def test_attention_omits_und_modules_when_disabled() -> None:
    attn = _make_attention(include_und_pathway=False)

    for name in _ATTN_UND_MODULES:
        assert not hasattr(attn, name), f"{name} should not be built when include_und_pathway=False"
    assert attn.k_norm_und_for_gen is None
    for name in _ATTN_GEN_MODULES:
        assert hasattr(attn, name), f"{name} must remain in a generator-only attention module"


def test_decoder_layer_omits_und_modules_when_disabled() -> None:
    layer = _make_layer(include_und_pathway=False)

    for name in _LAYER_UND_MODULES:
        assert not hasattr(layer, name), f"{name} should not be built when include_und_pathway=False"
    for name in _LAYER_GEN_MODULES:
        assert hasattr(layer, name), f"{name} must remain in a generator-only decoder layer"
    assert not [name for name, _ in layer.named_parameters() if "moe_gen" not in name]


def test_und_pathway_is_included_by_default() -> None:
    attn = _make_attention()
    layer = _make_layer()

    assert Nemotron3DenseVLMoTConfig({}).include_und_pathway is True
    for name in _ATTN_UND_MODULES:
        assert hasattr(attn, name)
    for name in _LAYER_UND_MODULES:
        assert hasattr(layer, name)


def _tiny_mot_config(*, include_und_pathway: bool = True) -> Nemotron3DenseVLMoTConfig:
    return Nemotron3DenseVLMoTConfig(
        {
            "vocab_size": 32,
            "hidden_size": NUM_Q_HEADS * HEAD_DIM,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": NUM_Q_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "enable_mrope": False,
        },
        include_und_pathway=include_und_pathway,
    )


def _assert_generator_only_structure(causal_lm: nn.Module) -> None:
    assert not hasattr(causal_lm, "lm_head")
    assert not hasattr(causal_lm.model, "embed_tokens")
    assert not hasattr(causal_lm.model, "norm")
    assert hasattr(causal_lm.model, "norm_moe_gen")
    assert hasattr(causal_lm.model, "rotary_emb")

    layer = causal_lm.model.layers[0]
    for name in _LAYER_UND_MODULES:
        assert not hasattr(layer, name)
    for name in _ATTN_UND_MODULES:
        assert not hasattr(layer.self_attn, name)

    parameter_names = [name for name, _ in causal_lm.named_parameters()]
    assert parameter_names
    assert all("moe_gen" in name for name in parameter_names)


def test_for_causal_lm_builds_generator_only_structure() -> None:
    causal_lm = Nemotron3DenseVLTextForCausalLM(_tiny_mot_config(include_und_pathway=False))

    _assert_generator_only_structure(causal_lm)
    with pytest.raises(RuntimeError, match="include_und_pathway=False"):
        causal_lm.model.reasoner_forward(torch.ones((1, 1), dtype=torch.long), cache=None)
    with pytest.raises(RuntimeError, match="include_und_pathway=False"):
        causal_lm.generate_reasoner_text(torch.ones((1, 1), dtype=torch.long), max_new_tokens=0)


def test_prune_und_pathway_is_idempotent_and_preserves_gen_fqns() -> None:
    with torch.device("meta"):
        causal_lm = Nemotron3DenseVLTextForCausalLM(_tiny_mot_config())
        causal_lm.visual = nn.Linear(2, 2)
    gen_parameter_names = {name for name, _ in causal_lm.named_parameters() if "moe_gen" in name}

    assert prune_und_pathway_(causal_lm) is causal_lm
    assert prune_und_pathway_(causal_lm) is causal_lm

    _assert_generator_only_structure(causal_lm)
    assert {name for name, _ in causal_lm.named_parameters()} == gen_parameter_names
    assert all(parameter.is_meta for parameter in causal_lm.parameters())
    assert causal_lm.model.include_und_pathway is False
    assert causal_lm.model.layers[0].include_und_pathway is False
    assert causal_lm.model.layers[0].self_attn.include_und_pathway is False
    assert not hasattr(causal_lm, "visual")


class _GenOnlyMemory(MemoryState):
    def init(self, hidden_states: dict, device: torch.device) -> None:
        del hidden_states, device

    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        del layer_idx
        return MemoryValue()

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        del layer_idx, kv_to_store

    def is_gen_only(self) -> bool:
        return True


def _passthrough_gen_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: Any,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, None]:
    del packed_key_states, packed_value_states, attention_mask, natten_metadata, packed_key_states_normalized
    assert memory_value is not None
    gen = get_gen_seq(packed_query_states).flatten(-2, -1)
    empty_und = gen.new_empty((0, gen.shape[-1]))
    return from_und_gen_splits(empty_und, gen, packed_query_states), None


@pytest.mark.parametrize("prune_after_build", [False, True])
def test_generator_only_joint_forward_requires_and_accepts_gen_only_memory(prune_after_build: bool) -> None:
    causal_lm = Nemotron3DenseVLTextForCausalLM(_tiny_mot_config(include_und_pathway=prune_after_build))
    if prune_after_build:
        prune_und_pathway_(causal_lm)
    for layer in causal_lm.model.layers:
        layer.self_attn.dispatch_attention_fn = _passthrough_gen_attention

    hidden = torch.randn(4, NUM_Q_HEADS * HEAD_DIM)
    pack, attention_mask, _ = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=["full"],
        split_lens=[hidden.shape[0]],
        sample_lens=[hidden.shape[0]],
        packed_und_token_indexes=torch.empty(0, dtype=torch.long),
        packed_gen_token_indexes=torch.arange(hidden.shape[0]),
        num_heads=NUM_Q_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
    )

    with pytest.raises(RuntimeError, match="requires a MemoryState"):
        causal_lm(pack, attention_mask, torch.arange(hidden.shape[0]))

    output, metadata = causal_lm(
        pack,
        attention_mask,
        torch.arange(hidden.shape[0]),
        memory=_GenOnlyMemory(),
    )
    assert metadata == {}
    assert get_und_seq(output).shape == (0, hidden.shape[-1])
    assert get_gen_seq(output).shape == hidden.shape
