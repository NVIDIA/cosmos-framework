# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.model import _diffusers_to_net_key
from cosmos_framework.scripts._convert_model_to_diffusers import (
    _TIME_EMBEDDER_KEY_REMAP,
    _remap_language_model_key,
)

_TRANSFORMER_SHARD = "transformer/diffusion_pytorch_model.safetensors"


def test_remap_language_model_key():
    cases = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "model.norm_moe_gen.weight": "norm_moe_gen.weight",
        "lm_head.weight": "lm_head.weight",
        "model.layers.18.self_attn.q_proj.weight": "layers.18.self_attn.to_q.weight",
        "model.layers.18.self_attn.k_proj.weight": "layers.18.self_attn.to_k.weight",
        "model.layers.18.self_attn.v_proj.weight": "layers.18.self_attn.to_v.weight",
        "model.layers.18.self_attn.o_proj.weight": "layers.18.self_attn.to_out.weight",
        "model.layers.18.self_attn.q_norm.weight": "layers.18.self_attn.norm_q.weight",
        "model.layers.18.self_attn.k_norm.weight": "layers.18.self_attn.norm_k.weight",
        "model.layers.18.self_attn.q_proj_moe_gen.weight": "layers.18.self_attn.add_q_proj.weight",
        "model.layers.18.self_attn.k_proj_moe_gen.weight": "layers.18.self_attn.add_k_proj.weight",
        "model.layers.18.self_attn.v_proj_moe_gen.weight": "layers.18.self_attn.add_v_proj.weight",
        "model.layers.18.self_attn.o_proj_moe_gen.weight": "layers.18.self_attn.to_add_out.weight",
        "model.layers.18.self_attn.q_norm_moe_gen.weight": "layers.18.self_attn.norm_added_q.weight",
        "model.layers.18.self_attn.k_norm_moe_gen.weight": "layers.18.self_attn.norm_added_k.weight",
        "model.layers.18.mlp.gate_proj.weight": "layers.18.mlp.gate_proj.weight",
        "model.layers.18.mlp_moe_gen.down_proj.weight": "layers.18.mlp_moe_gen.down_proj.weight",
        "model.layers.18.input_layernorm.weight": "layers.18.input_layernorm.weight",
    }
    for source_key, diffusers_key in cases.items():
        assert _remap_language_model_key(source_key) == diffusers_key


def test_language_model_remap_round_trips_with_inference_loader():
    """Exported language-model keys must map back to their OmniMoTModel.net
    keys through the inference loader's inverse mapping.
    """
    source_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.norm_moe_gen.weight",
        "lm_head.weight",
        "model.layers.18.self_attn.q_proj.weight",
        "model.layers.18.self_attn.o_proj.weight",
        "model.layers.18.self_attn.q_norm.weight",
        "model.layers.18.self_attn.q_proj_moe_gen.weight",
        "model.layers.18.self_attn.o_proj_moe_gen.weight",
        "model.layers.18.self_attn.k_norm_moe_gen.weight",
        "model.layers.18.mlp.gate_proj.weight",
        "model.layers.18.mlp_moe_gen.up_proj.weight",
        "model.layers.18.input_layernorm_moe_gen.weight",
    ]
    for source_key in source_keys:
        diffusers_key = _remap_language_model_key(source_key)
        assert _diffusers_to_net_key(diffusers_key, _TRANSFORMER_SHARD) == f"language_model.{source_key}"


def test_modality_projection_keys_round_trip_with_inference_loader():
    """Non-language-model keys the exporter assembles (modality projections,
    timestep embedder, modality embeds) must map back to their net keys.
    """
    exported_to_net = {
        "proj_in.weight": "vae2llm.weight",
        "proj_in.bias": "vae2llm.bias",
        "proj_out.weight": "llm2vae.weight",
        "audio_proj_in.weight": "sound2llm.weight",
        "audio_proj_out.bias": "llm2sound.bias",
        "audio_modality_embed": "sound_modality_embed",
        "action_proj_in.fc.weight": "action2llm.fc.weight",
        "action_proj_out.bias.weight": "llm2action.bias.weight",
        "action_modality_embed": "action_modality_embed",
    }
    for exported_key, net_key in exported_to_net.items():
        assert _diffusers_to_net_key(exported_key, _TRANSFORMER_SHARD) == net_key

    for source_key, exported_suffix in _TIME_EMBEDDER_KEY_REMAP.items():
        exported_key = f"time_embedder.{exported_suffix}"
        assert _diffusers_to_net_key(exported_key, _TRANSFORMER_SHARD) == f"time_embedder.{source_key}"
