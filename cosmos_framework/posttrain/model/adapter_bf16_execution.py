# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit new-artifact rounded-weight execution; legacy oracles are unchanged."""

import copy
from importlib.metadata import version
from types import MethodType

PROFILE = "merged_weight_bf16_v1"
CONTROL_PROFILE = "qwen3_vl_hf_stock_v1"
_MODELS = {
    "qwen3_vl": ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLForConditionalGeneration"),
    "qwen3_5": ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForConditionalGeneration"),
}


def validate_model_execution(net, config, family):
    if config.model_execution == CONTROL_PROFILE and family != "qwen3_vl":
        raise ValueError(f"{CONTROL_PROFILE} requires the dense HF Qwen3-VL training path")
    if config.model_execution not in (PROFILE, CONTROL_PROFILE) or family not in _MODELS:
        raise ValueError(f"{PROFILE} requires the dense HF Qwen3-VL or Qwen3.5 training path")
    if (type(net).__module__, type(net).__name__) != _MODELS[family]:
        raise ValueError(f"{PROFILE} requires the exact stock HF model class; select model_backend='hf'")
    if version("transformers") != "5.8.1" or version("torch").split("+")[0] != "2.10.0":
        raise ValueError(f"{PROFILE} requires Transformers 5.8.1 / PyTorch 2.10.0")
    configs = [net.config, net.config.text_config, net.config.vision_config]
    configs.extend(getattr(module, "config", None) for module in net.modules())
    if any(getattr(section, "_attn_implementation", "sdpa") != "sdpa" for section in configs if section is not None):
        raise ValueError(f"{PROFILE} requires SDPA throughout the model")


def _effective_norm(self, inputs):
    # A shallow, call-local view reuses the exact stock normalization code and
    # its precision/order (Qwen3.5's additive weight differs from Qwen3-VL).
    # Never swap a live Parameter, register an effective tensor, or keep a graph
    # on the module. Checkpoint replay and concurrent calls get independent views.
    view = copy.copy(self)
    view._parameters = {
        name: None if value is None else value.to(inputs.dtype) for name, value in self._parameters.items()
    }
    return type(self).forward(view, inputs)


def install_effective_norm(module):
    if any(module.children()) or any(buffer is not None for buffer in module.buffers()):
        raise ValueError("merged-weight extra normalization requires a stateless leaf module")
    module.forward = MethodType(_effective_norm, module)


def install_model_execution(net, config):
    if net.config.model_type == "qwen3_vl":
        from cosmos_framework.posttrain.model.qwen3_vl_hf import patch_ordered_forward

        patch_ordered_forward(net.model)
    elif net.config.model_type == "qwen3_5":
        from cosmos_framework.posttrain.model.qwen3_5 import patch_qwen3_5_forward

        patch_qwen3_5_forward(net.model)
    net._posttrain_model_execution = config.model_execution


def _canonical_adapter_forward(self, inputs):
    from cosmos_framework.posttrain.model.adapter_inference_cache import effective_weight_for_inference
    from cosmos_framework.utils.generator.bf16_cache import linear

    return linear(inputs, effective_weight_for_inference(self), self.bias)


def _cached_visual_adapter_forward(self, inputs):
    from torch.nn.functional import linear

    from cosmos_framework.posttrain.model.adapter_inference_cache import effective_weight_for_inference

    return linear(inputs, effective_weight_for_inference(self), self.bias)


def install_adapter_inference_profile(net, profile, *, cache_effective_weights=False):
    """Explicit portable inference override; never changes training or artifact tensors.

    Load the portable adapter with its ordinary strict loader first, then call
    this on the resulting single-device eval model. It cannot train or resume.
    The merged artifact declares the same override via --execution-profile.
    Optional derived-weight caching preserves arithmetic and original state.
    Version-tracked updates invalidate it. Bind any updating optimizer with
    bind_adapter_inference_cache_optimizer (fused steps may bypass versions),
    and use clear_adapter_inference_cache for unsafe .data writes. Cached bytes
    are reported separately from KV/parameters.
    """
    import torch

    from cosmos_framework.posttrain.checkpoint.adapters import _validate_state, state_spec
    from cosmos_framework.posttrain.model.adapter_inference_cache import (
        configure_weight_cache,
        unbind_adapter_inference_cache_optimizers,
        validate_cache_sources,
    )
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
    from cosmos_framework.utils.generator.bf16_cache import install, validate_model
    from cosmos_framework.utils.generator.bf16_cache_contract import (
        PROFILE_FAMILIES,
        QWEN3_5_PROFILE,
        validate_adapter_identity,
    )

    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"unknown portable inference profile: {profile!r}")
    identity = getattr(net, "_posttrain_adapter_identity", {})
    validate_adapter_identity(identity, profile=profile)
    if profile == QWEN3_5_PROFILE:
        from cosmos_framework.utils.generator.qwen3_5_cache import install, validate_model
    spec = state_spec(identity)
    state = net.state_dict()
    _validate_state({name: state[name] for name in spec if name in state}, identity)
    for name, parameter in net.named_parameters(remove_duplicate=False):
        expected = torch.float32 if name in spec else torch.bfloat16
        if parameter.dtype != expected:
            raise ValueError(f"portable inference dtype differs for {name}: expected {expected}, got {parameter.dtype}")
    validate_model(net, adapter_type=PosttrainAdapterLinear)
    targets = [(name, module) for name, module in net.named_modules() if isinstance(module, PosttrainAdapterLinear)]
    if not targets or any(module.execution != PROFILE for _, module in targets):
        raise ValueError("canonical cache requires rounded-weight adapters throughout the portable model")
    if not isinstance(cache_effective_weights, bool):
        raise ValueError("cache_effective_weights must be a boolean")
    if cache_effective_weights:
        for _, module in targets:
            validate_cache_sources(module)
    # All checks precede mutation. Reuse the adapter's existing effective-weight
    # implementation, including DoRA magnitude and PiSSA initial-factor semantics.
    changed = install(net, adapter_type=PosttrainAdapterLinear)
    if not cache_effective_weights:
        unbind_adapter_inference_cache_optimizers(net)
    for name, module in targets:
        configure_weight_cache(module, cache_effective_weights)
        if name.startswith("model.language_model."):
            forward = MethodType(_canonical_adapter_forward, module)
            if hasattr(module, "_hf_hook"):
                module._old_forward = forward
            else:
                module.forward = forward
            if name not in changed:
                changed.append(name)
        else:
            # Vision/projector keep their stock F.linear boundary; only the
            # identical rounded effective tensor is reused when explicitly enabled.
            forward = MethodType(
                _cached_visual_adapter_forward if cache_effective_weights else PosttrainAdapterLinear.forward, module
            )
            if hasattr(module, "_hf_hook"):
                module._old_forward = forward
            else:
                module.forward = forward
    net._cosmos_deployment_modules = tuple(changed)
    return changed
