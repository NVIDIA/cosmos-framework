# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen3.5 canonical BF16 inference with a split-consistent gated-delta state.

The recurrent reduction is FP64, rounded to FP32 after every token. The same
kernel serves prefill, suffixes and one-token decode. FP32 recurrent cache is
allocated per request; no global Transformers classes/methods are modified.
"""

from importlib.metadata import version
from inspect import signature
from types import MethodType

import torch
from torch import nn

from cosmos_framework.utils.generator.bf16_cache import (
    _adapter_children,
    _attention_forward,
    _inference_only,
    _linear_forward,
    _require_inference,
)
from cosmos_framework.utils.generator.bf16_cache_contract import QWEN3_5_PROFILE as PROFILE


def _recurrent(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs
):
    import triton
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

    from cosmos_framework.utils.generator.qwen3_5_cache_kernel import _recurrence_kernel

    _require_inference(query.device)
    if (
        query.ndim != 4
        or key.shape != query.shape
        or value.shape[:3] != query.shape[:3]
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or query.device != key.device
        or query.device != value.device
        or query.shape[0] != 1
        or query.shape[-1] != 128
        or value.shape[-1] != 128
        or g.shape != query.shape[:3]
        or beta.shape != g.shape
        or g.dtype != torch.float32
        or beta.dtype != torch.bfloat16
        or g.device != query.device
        or beta.device != query.device
    ):
        raise ValueError(f"{PROFILE} requires batch-one BF16 gated-delta inputs with 128-wide key/value heads")
    b, t, h, k = query.shape
    v = value.shape[-1]
    if initial_state is not None and (
        initial_state.shape != (b, h, k, v)
        or initial_state.dtype != torch.float32
        or initial_state.device != query.device
        or not initial_state.is_contiguous()
    ):
        raise ValueError(f"{PROFILE} requires contiguous FP32 recurrent state on the input GPU")
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    output = torch.empty(value.shape, device=query.device, dtype=query.dtype)
    final = torch.empty((b, h, k, v), device=query.device, dtype=torch.float32) if output_final_state else None
    with torch.cuda.device(query.device):
        _recurrence_kernel[(triton.cdiv(v, 32), h, b)](
            query,
            key,
            value,
            g,
            beta,
            initial_state if initial_state is not None else query,
            output,
            final if final is not None else query,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *g.stride(),
            *beta.stride(),
            t,
            h,
            k,
            v,
            initial_state is not None,
            output_final_state,
            triton.next_power_of_2(k),
            32,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output, final


def _prepare_cache(model, args, kwargs):
    from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

    _inference_only(model, args, kwargs)
    call = model._cosmos_forward_signature.bind_partial(*args, **kwargs)
    if (call.arguments.get("input_ids") is None) == (call.arguments.get("inputs_embeds") is None):
        raise ValueError(f"{PROFILE} requires exactly one of input_ids or inputs_embeds before creating cache state")
    cache = call.arguments.get("past_key_values")
    options = call.arguments.get("kwargs", {})
    use_cache = call.arguments.get("use_cache", options.get("use_cache", model.config.text_config.use_cache))
    if use_cache is None:
        use_cache = model.config.text_config.use_cache
    if cache is None and not use_cache:
        return None
    if cache is None:
        cache = DynamicCache(config=model.config)
        call.arguments["past_key_values"] = cache
    owned = getattr(cache, "_cosmos_canonical_owner", None)
    if owned is not None and owned is not model._cosmos_cache_owner:
        raise ValueError(f"{PROFILE} cannot consume another model instance's recurrent cache")
    config = model.config.text_config
    kinds = config.layer_types
    device = next(model.parameters()).device
    if len(cache.layers) != len(kinds):
        raise ValueError(f"{PROFILE} requires the model's complete DynamicCache layer layout")
    pending = []
    shape = (1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)
    # Validate every existing layer before changing any cache object or state.
    for kind, layer in zip(kinds, cache.layers, strict=True):
        if kind == "full_attention":
            if type(layer) is not DynamicLayer:
                raise ValueError(f"{PROFILE} requires ordinary dynamic full-attention layers")
            if getattr(layer, "is_initialized", False):
                if owned is None:
                    raise ValueError(f"{PROFILE} cannot reinterpret a pre-existing untagged attention cache")
                key, value = layer.keys, layer.values
                if (
                    not isinstance(key, torch.Tensor)
                    or not isinstance(value, torch.Tensor)
                    or key.ndim != 4
                    or value.shape != key.shape
                    or key.shape[:2] != (1, config.num_key_value_heads)
                    or key.shape[-1] != config.head_dim
                    or key.dtype != torch.bfloat16
                    or value.dtype != key.dtype
                    or key.device != device
                    or value.device != device
                ):
                    raise ValueError(f"{PROFILE} requires its original BF16 dynamic KV state")
        elif kind == "linear_attention":
            if type(layer) is not LinearAttentionLayer:
                raise ValueError(f"{PROFILE} requires ordinary linear-attention cache layers")
            initialized = layer.is_recurrent_states_initialized or layer.is_conv_states_initialized
            if initialized and owned is None:
                raise ValueError(f"{PROFILE} cannot reinterpret an untagged BF16 recurrent cache; recompute its prefix")
            if layer.has_previous_state and not (
                layer.is_conv_states_initialized and layer.is_recurrent_states_initialized
            ):
                raise ValueError(f"{PROFILE} cannot continue an incomplete recurrent cache")
            if layer.is_conv_states_initialized:
                conv = layer.conv_states
                channels = (
                    2 * config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim
                )
                if (
                    not isinstance(conv, torch.Tensor)
                    or conv.shape != (1, channels, config.linear_conv_kernel_dim)
                    or conv.dtype != torch.bfloat16
                    or conv.device != device
                ):
                    raise ValueError(f"{PROFILE} requires its original BF16 convolution state")
            if layer.is_recurrent_states_initialized:
                state = layer.recurrent_states
                if (
                    not isinstance(state, torch.Tensor)
                    or state.shape != shape
                    or state.dtype != torch.float32
                    or state.device != device
                    or not state.is_contiguous()
                ):
                    raise ValueError(f"{PROFILE} cache requires its original contiguous FP32 recurrent state")
            else:
                pending.append(layer)
        else:
            raise ValueError(f"{PROFILE} does not support cache layer type {kind!r}")
    # Stage all allocations before publication. Stock convolution initialization
    # remains BF16 and never reinitializes this already-created recurrent state.
    states = [torch.zeros(shape, device=device, dtype=torch.float32) for _ in pending]
    for layer, state in zip(pending, states, strict=True):
        layer.recurrent_states = state
        layer.is_recurrent_states_initialized = True
    cache._cosmos_canonical_owner = model._cosmos_cache_owner
    return call.args, call.kwargs


def validate_model(model, *, adapter_type=None):
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if (
        installed not in (None, PROFILE)
        or type(model).__module__ != "transformers.models.qwen3_5.modeling_qwen3_5"
        or type(model).__name__ != "Qwen3_5ForConditionalGeneration"
        or model.training
        or version("torch").split("+")[0] != "2.10.0"
        or version("transformers") != "5.8.1"
        or version("triton") != "3.6.0"
    ):
        raise ValueError(f"{PROFILE} requires stock Qwen3.5 eval with Torch2.10.0/Transformers5.8.1/Triton3.6.0")
    devices = {p.device for p in model.parameters()}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
    ):
        raise ValueError(f"{PROFILE} requires one A100 without offload")
    text = model.config.text_config
    if (
        text.head_dim not in (64, 128, 256)
        or text.num_attention_heads % text.num_key_value_heads
        or getattr(text, "is_causal", True) is False
        or text.linear_key_head_dim != 128
        or text.linear_value_head_dim != 128
        or text.linear_num_value_heads % text.linear_num_key_heads
        or set(text.layer_types) != {"full_attention", "linear_attention"}
    ):
        raise ValueError(f"{PROFILE} requires the qualified full/linear attention layout and 128-wide recurrent heads")
    configs = [model.config, model.config.text_config, model.config.vision_config]
    configs.extend(getattr(m, "config", None) for m in model.modules())
    accepted = ("sdpa", PROFILE) if installed else ("sdpa",)
    if any(getattr(c, "_attn_implementation", "sdpa") not in accepted for c in configs if c is not None):
        raise ValueError(f"{PROFILE} requires the SDPA architecture")
    if adapter_type is None and any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError(f"{PROFILE} requires BF16 merged parameters")
    owned = _adapter_children(model, adapter_type)
    for name, module in model.named_modules():
        if id(module) in owned:
            continue
        if isinstance(module, nn.Linear) or (adapter_type is not None and isinstance(module, adapter_type)):
            if module.weight.dtype != torch.bfloat16 or not module.weight.is_contiguous():
                raise ValueError(f"{PROFILE} requires contiguous BF16 base linear weights: {name}")
    if not isinstance(model.lm_head, nn.Linear):
        raise ValueError(f"{PROFILE} requires the stock linear output head")


def install(model, *, adapter_type=None):
    validate_model(model, adapter_type=adapter_type)
    if getattr(model, "_cosmos_deployment_profile", None) == PROFILE:
        return list(model._cosmos_deployment_modules)
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    AttentionInterface.register(PROFILE, _attention_forward)
    AttentionMaskInterface.register(PROFILE, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    changed = []
    owned = _adapter_children(model, adapter_type)
    for name, module in model.named_modules():
        if id(module) in owned:
            continue
        if isinstance(module, nn.Linear) and (name.startswith("model.language_model.") or name == "lm_head"):
            forward = MethodType(_linear_forward, module)
            if hasattr(module, "_hf_hook"):
                module._old_forward = forward
            else:
                module.forward = forward
            changed.append(name)
        elif type(module).__name__ == "Qwen3_5Attention":
            module.config._attn_implementation = PROFILE
            changed.append(name)
        elif type(module).__name__ == "Qwen3_5GatedDeltaNet":
            module.chunk_gated_delta_rule = _recurrent
            module.recurrent_gated_delta_rule = _recurrent
            changed.append(name)
    model._cosmos_forward_signature = signature(model.forward)
    model._cosmos_cache_owner = object()
    model.register_forward_pre_hook(_prepare_cache, with_kwargs=True)
    model._cosmos_deployment_profile = PROFILE
    model._cosmos_deployment_modules = tuple(changed)
    return changed
