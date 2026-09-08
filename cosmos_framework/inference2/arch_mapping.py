# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single source of truth for the arch fields the forward converter writes into a
diffusers ``transformer/config.json`` (model.config -> transformer/config.json).

Only the *reversible* arch fields live here — the ones that are flat ``model.config`` knobs.
The transformer config also carries LM-text fields (attention_bias, intermediate_size,
rope_*, rms_norm_eps, vocab_size — they live under ``vlm_config.model_instance``) and
structural/derived fields (hidden_size, heads, layers, patch_latent_dim, timestep_scale).
Those are forward-only (read from the model's ``net`` at export) and are enumerated in
``EXCLUDED_TRANSFORMER_ARGS`` so the lockstep test can prove ``ARCH_MAP`` + the excluded set
exactly partition the real converter's ``Cosmos3OmniTransformer(...)`` init — i.e. that
``read_model_arch`` covers every reversible converter arg.
"""

from typing import Any

# (transformer/config.json key, model.config dotted path). One entry per reversible field.
ARCH_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("action_gen", ("action_gen",)),
    ("action_dim", ("max_action_dim",)),  # transformer action_dim == model.config.max_action_dim
    ("num_embodiment_domains", ("num_embodiment_domains",)),
    ("sound_gen", ("sound_gen",)),
    ("sound_dim", ("sound_dim",)),
    ("sound_latent_fps", ("sound_latent_fps",)),
    ("latent_channel", ("state_ch",)),  # net.latent_channel == model.config.state_ch
    ("base_fps", ("diffusion_expert_config", "base_fps")),
    ("enable_fps_modulation", ("diffusion_expert_config", "enable_fps_modulation")),
    ("latent_patch_size", ("diffusion_expert_config", "patch_spatial")),  # net.latent_patch_size == patch_spatial
    ("unified_3d_mrope_reset_spatial_ids", ("diffusion_expert_config", "unified_3d_mrope_reset_spatial_ids")),
    ("unified_3d_mrope_temporal_modality_margin", ("diffusion_expert_config", "unified_3d_mrope_temporal_modality_margin")),
)

# Forward-only args of the converter's Cosmos3OmniTransformer(...) init that do NOT map back
# to a flat model.config knob (LM-text -> vlm_config.model_instance; structural/derived).
EXCLUDED_TRANSFORMER_ARGS: frozenset[str] = frozenset(
    {
        # from lm_cfg (live under vlm_config.model_instance):
        "attention_bias",
        "attention_dropout",
        "intermediate_size",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "vocab_size",
        # structural / derived:
        "head_dim",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "patch_latent_dim",
        "timestep_scale",
    }
)


def _get_path(cfg: dict, path: tuple[str, ...]) -> Any:
    node = cfg
    for key in path:
        node = node[key]
    return node


def _has_path(cfg: dict, path: tuple[str, ...]) -> bool:
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def read_model_arch(model_config: dict) -> dict[str, Any]:
    """model.config -> {transformer_key: value} for the reversible arch fields present in
    ``model_config``. Consumed by the forward converter to fill ``transformer/config.json``."""
    out: dict[str, Any] = {}
    for tkey, path in ARCH_MAP:
        if _has_path(model_config, path):
            out[tkey] = _get_path(model_config, path)
    return out
