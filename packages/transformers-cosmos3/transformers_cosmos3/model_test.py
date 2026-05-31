# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import re

from transformers_cosmos3.model import DROP_PATTERNS, KEY_MAPPING

_DROP_RE = re.compile("|".join(DROP_PATTERNS))
_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in KEY_MAPPING.items()
)
_UNDERSTANDING_PREFIXES: tuple[str, ...] = ("lm_head.", "model.language_model.", "model.visual.")


def to_hf_weight_name(name: str) -> str:
    for pattern, replacement in _KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def is_understanding_weight_name(name: str) -> bool:
    if _DROP_RE.search(name) is not None:
        return False
    return to_hf_weight_name(name).startswith(_UNDERSTANDING_PREFIXES)


def test_flat_diffusers_transformer_keys_map_to_hf_qwen3_vl_language_model() -> None:
    assert to_hf_weight_name("embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert to_hf_weight_name("norm.weight") == "model.language_model.norm.weight"
    assert (
        to_hf_weight_name("layers.18.input_layernorm.weight") == "model.language_model.layers.18.input_layernorm.weight"
    )
    assert (
        to_hf_weight_name("layers.18.self_attn.to_q.weight") == "model.language_model.layers.18.self_attn.q_proj.weight"
    )
    assert (
        to_hf_weight_name("layers.18.self_attn.to_out.weight")
        == "model.language_model.layers.18.self_attn.o_proj.weight"
    )
    assert (
        to_hf_weight_name("layers.18.self_attn.norm_q.weight")
        == "model.language_model.layers.18.self_attn.q_norm.weight"
    )
    assert (
        to_hf_weight_name("layers.18.self_attn.norm_k.weight")
        == "model.language_model.layers.18.self_attn.k_norm.weight"
    )


def test_existing_hf_and_vllm_language_model_keys_are_normalized() -> None:
    assert (
        to_hf_weight_name("model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        to_hf_weight_name("language_model.model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert to_hf_weight_name("language_model.lm_head.weight") == "lm_head.weight"


def test_flat_vision_encoder_keys_map_to_hf_qwen3_vl_visual() -> None:
    assert to_hf_weight_name("blocks.0.attn.qkv.weight") == "model.visual.blocks.0.attn.qkv.weight"
    assert (
        to_hf_weight_name("deepstack_merger_list.0.linear_fc1.weight")
        == "model.visual.deepstack_merger_list.0.linear_fc1.weight"
    )
    assert to_hf_weight_name("visual.patch_embed.proj.weight") == "model.visual.patch_embed.proj.weight"


def test_understanding_filter_keeps_only_qwen3_vl_tower_weights() -> None:
    kept = (
        "lm_head.weight",
        "layers.18.self_attn.to_q.weight",
        "layers.18.self_attn.norm_q.weight",
        "blocks.0.attn.qkv.weight",
    )
    dropped = (
        "layers.18.self_attn.add_q_proj.weight",
        "layers.18.self_attn.to_add_out.weight",
        "layers.18.self_attn.norm_added_q.weight",
        "layers.18.input_layernorm_moe_gen.weight",
        "proj_in.weight",
        "proj_out.weight",
        "time_embedder.linear_1.weight",
        "audio_proj_in.weight",
        "action_proj_in.fc.weight",
        "random_extra.weight",
    )

    assert all(is_understanding_weight_name(name) for name in kept)
    assert not any(is_understanding_weight_name(name) for name in dropped)
