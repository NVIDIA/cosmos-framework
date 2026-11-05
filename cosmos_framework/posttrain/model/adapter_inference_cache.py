# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in, version-checked effective weights for portable inference.

Cached tensors are ordinary detached values, never Parameters or buffers. The
portable/checkpoint state remains the original base, factors and initial factors.
Version-tracked mutations are detected by identity/storage/version signatures.
Fused optimizer steps require the explicit optimizer hook below; unsafe ``.data``
writes require explicit clearing. Neither necessarily advances tensor versions.
"""

import weakref

import torch


def _signature(module):
    values = [module.weight, module.bias, module.lora_A.weight, module.lora_B.weight]
    values.extend(getattr(module, name, None) for name in ("magnitude", "initial_A", "initial_B"))
    tensors = []
    for value in values:
        if value is None:
            tensors.append(None)
            continue
        try:
            version = value._version
        except RuntimeError as error:
            raise ValueError(
                "effective-weight caching requires version-tracked tensors; load outside inference_mode "
                "or disable cache_effective_weights"
            ) from error
        tensors.append(
            (id(value), version, value.data_ptr(), value.dtype, value.device, tuple(value.shape), tuple(value.stride()))
        )
    return (module.method, module.execution, module._lora_rank, module._lora_alpha, module._lora_use_rslora, *tensors)


class _EffectiveWeightCache:
    def __init__(self):
        self.signature = None
        self.weight = None
        self.hits = self.misses = self.invalidations = 0

    def clear(self):
        if self.weight is not None:
            self.invalidations += 1
        self.signature = self.weight = None


def validate_cache_sources(module):
    """Validate before installing any inference hooks or cache state."""
    _signature(module)


def _before_load(module, *_args):
    cache = getattr(module, "_cosmos_effective_weight_cache", None)
    if cache is not None:
        cache.clear()


def configure_weight_cache(module, enabled):
    if enabled:
        validate_cache_sources(module)
        if getattr(module, "_cosmos_effective_weight_cache", None) is None:
            module._cosmos_effective_weight_cache = _EffectiveWeightCache()
            module._cosmos_effective_weight_cache_hook = module.register_load_state_dict_pre_hook(_before_load)
    else:
        cache = getattr(module, "_cosmos_effective_weight_cache", None)
        if cache is not None:
            cache.clear()
            module._cosmos_effective_weight_cache = None
            module._cosmos_effective_weight_cache_hook.remove()
            del module._cosmos_effective_weight_cache_hook
        unbind_adapter_inference_cache_optimizers(module)


def effective_weight_for_inference(module):
    """Reuse only a still-current weight; never retain an autograd graph."""
    device_type = module.weight.device.type
    if (
        module.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled(device_type)
        or torch.compiler.is_compiling()
    ):
        raise RuntimeError("adapter weight caching requires eager eval, no_grad/inference_mode and no autocast")
    if device_type == "cuda":
        with torch.cuda.device(module.weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("adapter weight caching does not support CUDA graphs")
    if module.execution != "merged_weight_bf16_v1":
        raise ValueError("canonical adapter inference requires merged_weight_bf16_v1 execution")
    cache = getattr(module, "_cosmos_effective_weight_cache", None)
    if cache is None:
        return module.effective_weight().to(module.weight.dtype)
    signature = _signature(module)
    if cache.signature != signature:
        cache.clear()
        weight = module.effective_weight().to(module.weight.dtype)
        assert not weight.requires_grad and weight.grad_fn is None
        cache.weight, cache.signature = weight, signature
        cache.misses += 1
    else:
        cache.hits += 1
    return cache.weight


def clear_adapter_inference_cache(net):
    """Clear all derived weights, including after unsafe version-bypassing writes."""
    for module in net.modules():
        cache = getattr(module, "_cosmos_effective_weight_cache", None)
        if cache is not None:
            cache.clear()


def bind_adapter_inference_cache_optimizer(net, optimizer):
    """Invalidate after optimizer steps, including fused updates without versions.

    Bind every optimizer that can update this model while its inference cache is
    enabled. The returned removable handle belongs to the caller. This does not
    enable training through the inference-only profile, change optimizer groups,
    or retain the model's lifetime. ``train(True)`` also clears derived weights.
    """
    reference = weakref.ref(net)

    def invalidate(_optimizer, _args, _kwargs):
        model = reference()
        if model is not None:
            clear_adapter_inference_cache(model)

    bindings = getattr(net, "_cosmos_cache_optimizer_hooks", None)
    if bindings is None:
        bindings = net._cosmos_cache_optimizer_hooks = weakref.WeakKeyDictionary()
    if optimizer in bindings:
        bindings[optimizer].remove()
    # Clear before partial updates too: a multi-group optimizer can update an
    # earlier group and then raise, in which case its post-hook never runs.
    handle = _OptimizerCacheHooks(
        optimizer.register_step_pre_hook(invalidate), optimizer.register_step_post_hook(invalidate)
    )
    bindings[optimizer] = handle
    clear_adapter_inference_cache(net)
    return handle


class _OptimizerCacheHooks:
    def __init__(self, *handles):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()


def unbind_adapter_inference_cache_optimizers(net):
    """Remove explicit optimizer bindings when disabling inference caching."""
    bindings = getattr(net, "_cosmos_cache_optimizer_hooks", None)
    if bindings is not None:
        for handle in bindings.values():
            handle.remove()
        bindings.clear()


def adapter_inference_cache_info(net):
    """Report derived memory separately from model parameters and KV state."""
    seen = set()
    result = {name: 0 for name in ("enabled_modules", "cached_bytes", "hits", "misses", "invalidations")}
    for module in net.modules():
        cache = getattr(module, "_cosmos_effective_weight_cache", None)
        if cache is None:
            continue
        result["enabled_modules"] += 1
        for name in ("hits", "misses", "invalidations"):
            result[name] += getattr(cache, name)
        if cache.weight is not None and cache.weight.data_ptr() not in seen:
            seen.add(cache.weight.data_ptr())
            result["cached_bytes"] += cache.weight.numel() * cache.weight.element_size()
    return result
