# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned, opt-in FP64 low-rank inference; never a training/export reinterpretation."""

from importlib.metadata import version
from inspect import signature
from types import MethodType

import torch
from torch.nn import functional as F

PROFILE = "qwen3_vl_factorized_fp64_branch_v1"


def validate_identity(identity):
    resolved = identity.get("resolved", {})
    dependencies = identity.get("dependencies", {})
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v3"
        or identity.get("architecture") != "qwen3_vl"
        or identity.get("base_storage_dtype") != "bfloat16"
        or resolved.get("model_execution") != "qwen3_vl_hf_stock_v1"
        or not resolved.get("targets")
        or any(
            row["settings"].get("execution", "factorized_v1") != "factorized_v1"
            or row["settings"]["method"] not in ("lora", "dora", "pissa")
            or row["settings"]["dropout"] != 0
            for row in resolved.get("targets", [])
        )
        or dependencies.get("torch") != "2.10.0"
        or dependencies.get("transformers") != "5.8.1"
    ):
        raise ValueError(f"{PROFILE} requires a v3 dense Qwen3-VL stock-HF factorized artifact with zero dropout")


def _guard(module):
    if (
        module.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.is_autocast_enabled("cpu")
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(f"{PROFILE} requires eager eval and no_grad/inference_mode; no autocast, graphs or training")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(f"{PROFILE} requires FP32 matmul without TF32; set float32_matmul_precision('highest')")


def _forward(self, inputs):
    _guard(self)
    tensors = (self.weight, self.lora_A.weight, self.lora_B.weight)
    if inputs.dtype != torch.float32 or any(t.dtype != torch.float32 or t.device != inputs.device for t in tensors):
        raise ValueError(f"{PROFILE} requires FP32 inputs and parameters on the original CUDA device")
    # Keep separate projections. A dense direction exists ONLY for DoRA's row
    # norm, never as an effective-weight forward. The frozen base projection is
    # FP32; the two low-rank projections and final sum use FP64.
    base = F.linear(inputs, self.weight, self.bias).double()
    a, b = self.lora_A.weight.double(), self.lora_B.weight.double()
    delta = F.linear(F.linear(inputs.double(), a), b) * self._lora_scale
    if self.method == "lora":
        return (base + delta).to(inputs.dtype)
    if self.method == "pissa":
        initial = (
            F.linear(F.linear(inputs.double(), self.initial_A.double()), self.initial_B.double()) * self._lora_scale
        )
        return (base + (delta - initial)).to(inputs.dtype)
    if self.method == "dora":
        direction = self.weight.double() + self._lora_scale * (b @ a)
        ratio = self.magnitude.double() / direction.norm(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
        without_bias = F.linear(inputs, self.weight, None).double()
        return (base + (ratio - 1) * without_bias + ratio * delta).to(inputs.dtype)
    raise ValueError(f"{PROFILE}: method changed after installation; reload the portable model")


def _before_forward(model, args, kwargs):
    from transformers.cache_utils import DynamicCache

    _guard(model)
    _validate_attention(model)
    record = model._cosmos_factorized_profile
    if any(p.dtype != torch.float32 or p.device != record["device"] for p in model.parameters()):
        raise ValueError(f"{PROFILE} parameters changed precision/device; restore the profile before conversion")
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    for name, ndim in (("input_ids", 2), ("inputs_embeds", 3)):
        value = call.get(name)
        if value is not None and (value.ndim != ndim or value.shape[0] != 1):
            raise ValueError(f"{PROFILE} requires batch-one {name}")
    cache = call.get("past_key_values")
    if cache is not None and (type(cache) is not DynamicCache or getattr(cache, "offloading", False)):
        raise ValueError(f"{PROFILE} supports ordinary DynamicCache without offload")
    for name, module, _, _ in record["forwards"]:
        if (
            model.get_submodule(name) is not module
            or module.execution != "factorized_v1"
            or getattr(module.lora_dropout, "p", 0)
        ):
            raise ValueError(f"{PROFILE} adapter structure/settings changed; restore and reload the profile")


def _validate_attention(model):
    sections = [model.config, model.config.text_config, model.config.vision_config]
    sections.extend(getattr(m, "config", None) for m in model.modules())
    if any(getattr(c, "_attn_implementation", "sdpa") != "sdpa" for c in sections if c is not None):
        raise ValueError(f"{PROFILE} requires stock SDPA throughout; restore before changing attention")


def install(model, *, adapter_type):
    """Validate before changing forwards. No parameter/buffer/ownership mutation."""
    validate_identity(getattr(model, "_posttrain_adapter_identity", {}))
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if installed not in (None, PROFILE):
        raise ValueError("a different deployment profile is already installed")
    if (
        type(model).__module__ != "transformers.models.qwen3_vl.modeling_qwen3_vl"
        or type(model).__name__ != "Qwen3VLForConditionalGeneration"
        or model.training
        or version("torch").split("+")[0] != "2.10.0"
        or version("transformers") != "5.8.1"
    ):
        raise ValueError(f"{PROFILE} requires stock Qwen3-VL eval, Torch 2.10.0 / Transformers 5.8.1")
    devices = {p.device for p in model.parameters()}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
    ):
        raise ValueError(f"{PROFILE} requires a single A100 (SM80), without offload")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError(f"{PROFILE} requires explicit FP32 materialization before installation")
    _validate_attention(model)
    targets = [(n, m) for n, m in model.named_modules(remove_duplicate=False) if isinstance(m, adapter_type)]
    expected = model._posttrain_adapter_identity["resolved"]["targets"]
    if {n for n, _ in targets} != {r["name"] for r in expected} or any(
        m.execution != "factorized_v1" or m.method not in ("lora", "dora", "pissa") or getattr(m.lora_dropout, "p", 0)
        for _, m in targets
    ):
        raise ValueError(f"{PROFILE} adapter placement/settings differ from the artifact")
    if any(getattr(m, "_cosmos_effective_weight_cache", None) is not None for _, m in targets):
        raise ValueError(f"{PROFILE} does not support derived effective-weight caching")
    if installed:
        record = model._cosmos_factorized_profile
        if any(model.get_submodule(n) is not m for n, m, _, _ in record["forwards"]):
            raise ValueError("installed profile's adapter modules were replaced; restore before reloading")
        return [n for n, _ in targets]
    forwards, seen = [], set()
    for name, module in targets:
        if id(module) in seen:
            continue
        seen.add(id(module))
        attribute = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
        forwards.append((name, module, attribute, getattr(module, attribute)))
    model._cosmos_factorized_profile = {
        "forwards": forwards,
        "signature": signature(model.forward),
        "device": next(iter(devices)),
    }
    for _, module, attribute, _ in forwards:
        setattr(module, attribute, MethodType(_forward, module))
    model._cosmos_factorized_profile["hook"] = model.register_forward_pre_hook(_before_forward, with_kwargs=True)
    model._cosmos_deployment_profile = PROFILE
    return [n for n, _ in targets]


def restore(model):
    """Restore original factorized forwards; retain the original FP32 tensors."""
    if getattr(model, "_cosmos_deployment_profile", None) != PROFILE:
        raise ValueError(f"{PROFILE} is not installed; refusing to restore another profile")
    record = model._cosmos_factorized_profile
    if any(model.get_submodule(n) is not m for n, m, _, _ in record["forwards"]):
        raise ValueError("adapter modules were replaced; cannot safely restore this profile")
    for _, module, attribute, original in record["forwards"]:
        setattr(module, attribute, original)
    record["hook"].remove()
    del model._cosmos_factorized_profile
    del model._cosmos_deployment_profile
