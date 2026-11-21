# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit ordinary attention projections in the pinned N1.7 action head.

Freezing original parameters does not disable the native head's training modes,
dropout or gradient path. The category-specific projectors remain untouched.
"""

from __future__ import annotations

from collections import Counter

from cosmos_framework.posttrain.config.gr00t_bc import HEAD_ADAPTER_PROFILE


def resolve_head_attention_targets(model) -> dict[str, tuple[int, int]]:
    """Validate the actual native graph before streaming weights or creating adapters."""
    import torch
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from gr00t.model.modules.dit import AlternateVLDiT, SelfAttentionTransformer
    from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
    from torch import nn

    head = model.action_head
    if type(head) is not Gr00tN1d7ActionHead or type(head.model) is not AlternateVLDiT:
        raise ValueError("GR00T head adapters require the pinned native AlternateVLDiT head")
    required = {
        "model_type": "Gr00tN1d7",
        "backbone_embedding_dim": 2048,
        "hidden_size": 1024,
        "input_embedding_dim": 1536,
        "max_state_dim": 132,
        "state_history_length": 1,
        "max_action_dim": 132,
        "action_horizon": 40,
        "max_num_embodiments": 32,
        "num_inference_timesteps": 4,
        "use_alternate_vl_dit": True,
        "attend_text_every_n_blocks": 2,
        "use_vlln": True,
        "state_dropout_prob": 0.2,
    }
    if any(getattr(head.config, key, None) != value for key, value in required.items()):
        raise ValueError("GR00T head geometry/dropout differs from its explicit adapter profile")
    if any(getattr(head, key, None) is not True for key in ("tune_projector", "tune_diffusion_model", "tune_vlln")):
        raise ValueError("GR00T head adapters preserve full-head native tune flags and training modes")
    if (
        len(head.model.transformer_blocks) != 32
        or type(head.vl_self_attention) is not SelfAttentionTransformer
        or len(head.vl_self_attention.transformer_blocks) != 4
    ):
        raise ValueError("GR00T head adapters require 32 DiT blocks and four untouched VL blocks")
    categories = {
        "state_encoder.layer1": (32, 132, 1024),
        "state_encoder.layer2": (32, 1024, 1536),
        "action_encoder.W1": (32, 132, 1536),
        "action_encoder.W2": (32, 3072, 1536),
        "action_encoder.W3": (32, 1536, 1536),
        "action_decoder.layer1": (32, 1024, 1024),
        "action_decoder.layer2": (32, 1024, 132),
    }
    actual_categories = {name for name, module in head.named_modules() if isinstance(module, CategorySpecificLinear)}
    if actual_categories != set(categories):
        raise ValueError("GR00T category-specific projector inventory differs")
    for name, shape in categories.items():
        module = head.get_submodule(name)
        if (
            type(module) is not CategorySpecificLinear
            or tuple(module.W.shape) != shape
            or tuple(module.b.shape) != (32, shape[-1])
        ):
            raise ValueError(f"GR00T category-specific projector differs: {name}")
    parameters = tuple(head.parameters())
    if len(parameters) != 537 or any(value.dtype != torch.float32 for value in parameters):
        raise ValueError("GR00T head adapters preserve all 537 original FP32 parameters")
    attentions = [module for module in head.modules() if isinstance(module, Attention)]
    if len(attentions) != 36 or any(
        type(module) is not Attention
        or type(module.processor) is not AttnProcessor2_0
        or getattr(module, "fused_projections", False)
        or any(hasattr(module, name) for name in ("to_qkv", "to_kv"))
        for module in attentions
    ):
        raise ValueError("GR00T head adapters require all 36 native unfused attention processors")
    occurrences = Counter(id(module) for _, module in model.named_modules(remove_duplicate=False))
    targets = {}
    for index, block in enumerate(head.model.transformer_blocks):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            name = f"action_head.model.transformer_blocks.{index}.attn1.{projection}"
            module = model.get_submodule(name)
            width = 2048 if index % 2 == 0 and projection in {"to_k", "to_v"} else 1536
            if (
                type(module) is not nn.Linear
                or module.bias is None
                or tuple(module.weight.shape) != (1536, width)
                or tuple(module.bias.shape) != (1536,)
                or occurrences[id(module)] != 1
            ):
                raise ValueError(f"GR00T head target must be an unaliased ordinary biased attention Linear: {name}")
            if block.attn1.heads != 32 or block.attn1.to_out[1].p != 0.2:
                raise ValueError("GR00T head attention geometry/dropout differs")
            targets[name] = (1536, width)
    return targets


def configured_head_targets(model, adaptation):
    """An unset profile preserves the existing full-head and frozen-head policies."""
    adaptation.__post_init__()
    if adaptation.action_head_adapter_profile is None:
        return None
    if adaptation.action_head_adapter_profile != HEAD_ADAPTER_PROFILE:
        raise ValueError("unsupported GR00T head adapter profile")
    return resolve_head_attention_targets(model)
