# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit canonical BF16 inference, shared by portable and merged loaders.

Weights, activations and KV remain BF16. Only attention's local reductions use
FP64; fixed-tile linear reductions use FP32. There is no whole-model FP32 copy,
parameter replacement, attention score matrix, weight cache or adapter math.
"""

from importlib.metadata import version
from inspect import signature
from types import MethodType

import torch
from torch import nn

from cosmos_framework.utils.generator.bf16_cache_contract import FAST_PROFILE, PROFILE, QWEN3_VL_PROFILES


def _require_inference(device):
    # Components are public modules too: a direct projection call must fail,
    # not silently return a detached Triton result during gradient computation.
    with torch.cuda.device(device):
        if (
            torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(f"{PROFILE} requires eager no_grad/inference_mode, and no autocast or CUDA graphs")


def _attention_signature(model):
    configs = [
        ("root", model.config),
        ("text", model.config.text_config),
        ("vision", model.config.vision_config),
    ]
    configs.extend((name, module.config) for name, module in model.named_modules() if hasattr(module, "config"))
    return tuple((name, id(config), config._attn_implementation) for name, config in configs)


def _inference_only(model, args, kwargs):
    from transformers.cache_utils import DynamicCache

    if (
        model.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(f"{PROFILE} requires eager eval, no_grad/inference_mode, and no autocast or CUDA graphs")
    # Bind the pinned stock signature, including positional cache/embedding
    # arguments, before any attention layer can mutate a caller-owned cache.
    call = model._cosmos_forward_signature.bind_partial(*args, **kwargs).arguments
    inputs = call.get("input_ids")
    embeds = call.get("inputs_embeds")
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[0] != 1):
        raise ValueError(f"{PROFILE} requires batch-one input_ids")
    if embeds is not None and (embeds.ndim != 3 or embeds.shape[0] != 1):
        raise ValueError(f"{PROFILE} requires batch-one inputs_embeds")
    cache = call.get("past_key_values")
    if cache is not None and (type(cache) is not DynamicCache or getattr(cache, "offloading", False)):
        raise ValueError(f"{PROFILE} supports an ordinary DynamicCache without offload")
    if model._cosmos_deployment_profile != PROFILE:
        # Reject live backend changes before mutating any caller-owned cache.
        if _attention_signature(model) != model._cosmos_deployment_attention:
            raise ValueError("attention execution changed after deployment installation; reload the model")


def _rms_norm_forward(self, inputs):
    import triton

    from cosmos_framework.utils.generator.bf16_cache_kernels import _rms_norm_kernel

    _require_inference(inputs.device)
    if (
        inputs.dtype != torch.bfloat16
        or inputs.device.type != "cuda"
        or self.weight.device != inputs.device
        or self.weight.dtype not in (torch.bfloat16, torch.float32)
        or tuple(self.weight.shape) != (inputs.shape[-1],)
    ):
        raise ValueError("canonical v2 RMSNorm requires BF16 CUDA inputs and a matching BF16/FP32 norm weight")
    flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    output = torch.empty_like(flat)
    if flat.shape[0]:
        with torch.cuda.device(inputs.device):
            _rms_norm_kernel[(flat.shape[0],)](
                flat,
                self.weight.contiguous(),
                output,
                flat.shape[1],
                self.variance_epsilon,
                triton.next_power_of_2(flat.shape[1]),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output.reshape(inputs.shape)


def linear(inputs, weight, bias=None):
    """Fixed M/N/K tiles without padding/copying full prompt-sized matrices."""
    import triton

    from cosmos_framework.utils.generator.bf16_cache_kernels import _linear_kernel

    if (
        inputs.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or inputs.device != weight.device
        or inputs.device.type != "cuda"
        or weight.ndim != 2
        or inputs.shape[-1] != weight.shape[1]
        or not weight.is_contiguous()
        or (bias is not None and (bias.dtype != weight.dtype or bias.device != weight.device))
    ):
        raise ValueError(f"{PROFILE} requires BF16 CUDA linear inputs and contiguous BF16 weights")
    flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    m, k = flat.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=inputs.device, dtype=inputs.dtype)
    _require_inference(inputs.device)
    with torch.cuda.device(inputs.device):
        if not m:
            return output.reshape(*inputs.shape[:-1], n)
        _linear_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            flat,
            weight,
            bias if bias is not None else flat,
            output,
            m,
            n,
            k,
            bias is not None,
            32,
            64,
            64,
            num_warps=4,
            num_stages=3,
        )
    return output.reshape(*inputs.shape[:-1], n)


def _linear_forward(self, inputs):
    return linear(inputs, self.weight, self.bias)


def _attention_warp_count(module, query_length, head_dim):
    # Long Qwen3-VL prefill benefits from eliminating cross-warp exchanges in
    # each query's reductions. This is the *same* kernel, 64-key tiles, FP64
    # arithmetic and online recurrence; decode/short inputs retain the original
    # launch. Keep the four-warp launch as the independent correctness reference.
    family = getattr(getattr(module, "config", None), "model_type", None)
    return 1 if family == "qwen3_vl_text" and query_length >= 1024 and head_dim == 128 else 4


def _attention_inputs(module, query, key, value, attention_mask, dropout, kwargs):
    b, h, m, d = query.shape
    n = key.shape[-2]
    if (
        module.training
        or dropout != 0
        or kwargs.get("is_causal", True) is False
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or query.device != key.device
        or query.device != value.device
        or value.shape != key.shape
        or key.shape[0] != b
        or key.shape[-1] != d
        or b != 1
        or n < m
        or d not in (64, 128, 256)
        or h % key.shape[1]
    ):
        raise ValueError(f"{PROFILE} supports batch-one causal BF16 attention with ordinary grouped KV heads")
    if attention_mask is not None:
        if (
            attention_mask.ndim != 4
            or attention_mask.shape[:2] != (1, 1)
            or attention_mask.shape[-2] not in (1, m)
            or attention_mask.shape[-1] != n
            or attention_mask.device != query.device
            or attention_mask.dtype not in (torch.bool, torch.bfloat16, torch.float32, torch.float64)
        ):
            raise ValueError(f"{PROFILE} requires a broadcastable four-dimensional causal/padding mask")
        mask = attention_mask.expand(1, 1, m, n)
        strides = (mask.stride(0), mask.stride(2), mask.stride(3))
    else:
        mask, strides = query, (0, 0, 0)
    _require_inference(query.device)
    return (b, h, m, d, n), mask, strides


def _attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    import triton

    from cosmos_framework.utils.generator.bf16_cache_kernels import _attention_kernel

    (b, h, m, d, n), mask, strides = _attention_inputs(module, query, key, value, attention_mask, dropout, kwargs)
    output = torch.empty((b, h, m, d), device=query.device, dtype=query.dtype)
    _require_inference(query.device)
    with torch.cuda.device(query.device):
        if not m:
            return output.transpose(1, 2).contiguous(), None
        _attention_kernel[(m, h, b)](
            query,
            key,
            value,
            output,
            mask,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *strides,
            m,
            n,
            h,
            key.shape[1],
            d,
            d**-0.5 if scaling is None else scaling,
            attention_mask is not None,
            attention_mask is not None and attention_mask.dtype == torch.bool,
            64,
            triton.next_power_of_2(d),
            num_warps=_attention_warp_count(module, m, d),
            num_stages=1,
        )
    return output.transpose(1, 2).contiguous(), None


def _tensorcore_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    if query.shape[-2] < 1024 or query.shape[-1] != 128:
        return _attention_forward(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
    import triton

    from cosmos_framework.utils.generator.bf16_prefill_kernels import _attention_tensorcore_kernel

    (b, h, m, d, n), mask, strides = _attention_inputs(module, query, key, value, attention_mask, dropout, kwargs)
    output = torch.empty((b, h, m, d), device=query.device, dtype=query.dtype)
    with torch.cuda.device(query.device):
        _attention_tensorcore_kernel[(triton.cdiv(m, 16), h, b)](
            query,
            key,
            value,
            output,
            mask,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *strides,
            m,
            n,
            h,
            key.shape[1],
            d,
            d**-0.5 if scaling is None else scaling,
            attention_mask is not None,
            attention_mask is not None and attention_mask.dtype == torch.bool,
            16,
            64,
            triton.next_power_of_2(d),
            num_warps=4,
            num_stages=1,
        )
    return output.transpose(1, 2).contiguous(), None


def _adapter_children(model, adapter_type):
    # FP32 factor modules are owned by adapter effective_weight(); they must
    # never be reclassified as frozen BF16 linears or independently patched.
    return {
        id(child)
        for module in model.modules()
        if adapter_type is not None and isinstance(module, adapter_type)
        for child in module.modules()
        if child is not module
    }


def validate_model(model, *, adapter_type=None, profile=PROFILE):
    """Validate the entire profile before mutating modules or their configurations."""
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if profile not in QWEN3_VL_PROFILES:
        raise ValueError(f"unsupported Qwen3-VL deployment profile: {profile!r}")
    if installed not in (None, profile):
        raise ValueError("a different deployment profile is already installed")
    if (
        type(model).__module__ != "transformers.models.qwen3_vl.modeling_qwen3_vl"
        or type(model).__name__ != "Qwen3VLForConditionalGeneration"
        or model.training
        or version("torch").split("+")[0] != "2.10.0"
        or version("transformers") != "5.8.1"
        or version("triton") != "3.6.0"
    ):
        raise ValueError(f"{PROFILE} requires stock Qwen3-VL eval with Torch 2.10.0, Transformers 5.8.1, Triton 3.6.0")
    devices = {p.device for p in model.parameters()}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
    ):
        raise ValueError(f"{PROFILE} is qualified on a single A100 (SM80), without CPU/disk offload")
    sections = [model.config, model.config.text_config, model.config.vision_config]
    sections.extend(getattr(module, "config", None) for module in model.modules())
    accepted_attention = ("sdpa", profile) if installed else ("sdpa",)
    if any(
        getattr(section, "_attn_implementation", "sdpa") not in accepted_attention
        for section in sections
        if section is not None
    ):
        raise ValueError(f"{PROFILE} requires the stock SDPA architecture before profile installation")
    if adapter_type is None and any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError(f"{PROFILE} requires a BF16 merged model")
    owned = _adapter_children(model, adapter_type)
    for name, module in model.named_modules():
        if id(module) in owned:
            continue
        if isinstance(module, nn.Linear) or (adapter_type is not None and isinstance(module, adapter_type)):
            if module.weight.dtype != torch.bfloat16 or not module.weight.is_contiguous():
                raise ValueError(f"{PROFILE} requires BF16 contiguous base linear weights: {name}")
    text = model.config.text_config
    head_dim = getattr(text, "head_dim", text.hidden_size // text.num_attention_heads)
    if (
        head_dim not in (64, 128, 256)
        or text.num_attention_heads % text.num_key_value_heads
        or getattr(text, "is_causal", True) is False
        or any(
            getattr(module, "is_causal", True) is False
            for module in model.modules()
            if type(module).__name__ == "Qwen3VLTextAttention"
        )
    ):
        raise ValueError(f"{PROFILE} requires causal grouped attention with head_dim 64, 128 or 256")
    if not isinstance(model.lm_head, nn.Linear):
        raise ValueError(f"{PROFILE} requires the stock linear output head")


def install(model, *, adapter_type=None, profile=PROFILE):
    """Install only language projections/head and causal attention, preserving ownership."""
    validate_model(model, adapter_type=adapter_type, profile=profile)
    if getattr(model, "_cosmos_deployment_profile", None) == profile:
        return list(model._cosmos_deployment_modules)
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    AttentionInterface.register(
        profile, _tensorcore_attention_forward if profile == FAST_PROFILE else _attention_forward
    )
    # The mask registration is essential for multi-token cached suffixes. It
    # retains the existing SDPA causal/padding mask and its cache offset.
    AttentionMaskInterface.register(profile, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
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
        elif type(module).__name__ == "Qwen3VLTextAttention":
            module.config._attn_implementation = profile
            changed.append(name)
        elif (
            profile != PROFILE
            and name.startswith("model.language_model.")
            and type(module).__name__ == "Qwen3VLTextRMSNorm"
        ):
            forward = MethodType(_rms_norm_forward, module)
            if hasattr(module, "_hf_hook"):
                module._old_forward = forward
            else:
                module.forward = forward
            changed.append(name)
    model._cosmos_forward_signature = signature(model.forward)
    model.register_forward_pre_hook(_inference_only, with_kwargs=True)
    model._cosmos_deployment_profile = profile
    if profile != PROFILE:
        model._cosmos_deployment_attention = _attention_signature(model)
    model._cosmos_deployment_modules = tuple(changed)
    return changed
