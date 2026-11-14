# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit Qwen3-VL deployment arithmetic, without changing serialized weights.

The optional profile retains BF16 parameters and public logits, but carries
intermediate linear outputs, normalization and residual additions in FP32.
It requires Framework loading (or explicit installation after stock HF loading);
plain ``from_pretrained`` alone does not install this execution contract.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from cosmos_framework.utils.generator import qwen3_vl_projection_contract as qwen_fixed
from cosmos_framework.utils.generator.bf16_cache_contract import (
    CORRECTED_PROFILE,
    FAST_PROFILE,
    PROFILE_FAMILIES,
    QWEN3_5_PROFILE,
    QWEN3_VL_PROFILES,
)
from cosmos_framework.utils.generator.gemma_sdpa_layout import PROFILES as GEMMA_PROFILES
from cosmos_framework.utils.generator.smollm_consistent import PROFILES as SMOLLM_PROFILES

PROFILE = "qwen3_vl_bf16_residual_fp32_v1"


def validate_deployment_environment(profile: str | None, device: str, *, device_map: str = "auto") -> None:
    """Fail before allocating weights on an unsupported v2/fast deployment host."""
    if profile not in (CORRECTED_PROFILE, FAST_PROFILE, *SMOLLM_PROFILES, *GEMMA_PROFILES, qwen_fixed.PROFILE):
        return  # Preserve the legacy loaders' original validation boundary.
    if device_map not in ("auto", device):
        raise ValueError(f"{profile} requires one rank-local GPU; alternate device maps/offload are unsupported")
    from importlib.metadata import version

    import torch

    target = torch.device(device)
    if target.type != "cuda" or torch.cuda.get_device_capability(target) != (8, 0):
        raise ValueError(f"{profile} requires one A100 (SM80); select its CUDA UUID before loading")
    if profile == qwen_fixed.PROFILE and "A100" not in torch.cuda.get_device_name(target):
        raise ValueError(f"{profile} requires an A100; SM80 alone does not identify the admitted GPU")
    required = {"torch": "2.10.0", "transformers": "5.8.1", "triton": "3.6.0"}
    if any(version(name).split("+")[0] != expected for name, expected in required.items()):
        raise ValueError(f"{profile} requires the pinned cosmos-framework-reasoner-runtime environment: {required}")


def deployment_profile(path: str) -> str | None:
    """Require matching config and transaction declarations before model loading."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("model config must be a JSON object")
    declared = config.get("cosmos_inference_profile")
    marker_path = Path(path) / ".materialized"
    try:
        marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
    except (OSError, ValueError) as error:
        if declared is None:
            return None  # Untagged legacy exports do not acquire a new marker requirement.
        raise ValueError("deployment profile requires a readable committed .materialized marker") from error
    identity = marker.get("identity", {}) if isinstance(marker, dict) else {}
    metadata = identity.get("export_metadata", {}) if isinstance(identity, dict) else {}
    precision = metadata.get("deployment_precision", {}) if isinstance(metadata, dict) else {}
    precision = precision if isinstance(precision, dict) else {}
    bound = precision.get("execution_profile")
    if declared is None and bound is None:
        return None
    if (
        declared not in (PROFILE, *PROFILE_FAMILIES, *SMOLLM_PROFILES, *GEMMA_PROFILES, qwen_fixed.PROFILE)
        or bound != declared
    ):
        raise ValueError("missing, conflicting or unknown deployment profile in config/.materialized; re-export")
    if declared == qwen_fixed.PROFILE:
        qwen_fixed.validate_identity(metadata.get("adapter_identity", {}))
        qwen_fixed.validate_configuration(config)
        if (
            config.get("dtype", config.get("torch_dtype")) != "float32"
            or config.get("cosmos_model_execution") != "qwen3_vl_hf_stock_v1"
            or identity.get("output_dtype") != "float32"
            or metadata.get("format") != "cosmos-merged-adapter-v1"
            or precision
            != {
                "dtype": "float32",
                "execution_profile": declared,
                "weight_conversion": "unchanged_float32_merge",
                "equivalence": qwen_fixed.EXPORT_EQUIVALENCE,
            }
        ):
            raise ValueError("fixed Qwen3-VL projections require their committed FP32 stock adapter merge")
        return declared
    if declared in GEMMA_PROFILES:
        from cosmos_framework.utils.generator.gemma_sdpa_layout import (
            export_equivalence,
            validate_configuration,
            validate_identity,
        )

        validate_identity(metadata.get("adapter_identity", {}), profile=declared)
        validate_configuration(config)
        if (
            config.get("model_type") != "gemma"
            or config.get("dtype", config.get("torch_dtype")) != "float32"
            or identity.get("output_dtype") != "float32"
            or precision
            != {
                "dtype": "float32",
                "execution_profile": declared,
                "weight_conversion": "unchanged_float32_merge",
                "equivalence": export_equivalence(declared),
            }
            or metadata.get("format") != "cosmos-merged-adapter-v1"
        ):
            raise ValueError("Gemma deployment requires its committed FP32 adapter merge/profile")
        return declared
    if declared in SMOLLM_PROFILES:
        from cosmos_framework.utils.generator.smollm_consistent import validate_identity

        validate_identity(metadata.get("adapter_identity", {}), profile=declared)
        if (
            config.get("model_type") != "llama"
            or config.get("dtype", config.get("torch_dtype")) != "float32"
            or identity.get("output_dtype") != "float32"
            or precision
            != {
                "dtype": "float32",
                "execution_profile": declared,
                "weight_conversion": "unchanged_float32_merge",
                "equivalence": "versioned_fp64_reduction_inference_requires_qualification",
            }
            or metadata.get("format") != "cosmos-merged-adapter-v1"
        ):
            raise ValueError("SmolLM deployment requires its committed FP32 adapter merge/profile")
        return declared
    if (
        config.get("model_type") != PROFILE_FAMILIES.get(declared, "qwen3_vl")
        or config.get("dtype", config.get("torch_dtype")) != "bfloat16"
        or identity.get("output_dtype") != "bfloat16"
        or precision.get("dtype") != "bfloat16"
        or metadata.get("format") != "cosmos-merged-adapter-v1"
    ):
        raise ValueError("residual FP32 deployment requires a committed dense Qwen3-VL BF16 merged artifact")
    if declared in PROFILE_FAMILIES:
        from cosmos_framework.utils.generator.bf16_cache_contract import validate_adapter_identity

        validate_adapter_identity(metadata.get("adapter_identity", {}), profile=declared)
        if config.get("cosmos_model_execution") != "merged_weight_bf16_v1":
            raise ValueError("canonical cache profile requires its merged-weight model execution identity")
    return declared


def _linear(self, inputs):
    import torch

    result = torch.mm(inputs.reshape(-1, self.in_features).to(torch.bfloat16), self.weight.T, out_dtype=torch.float32)
    if self.bias is not None:
        result = result + self.bias.float()
    return result.view(*inputs.shape[:-1], self.out_features)


def _head(self, inputs):
    import torch
    from torch.nn import functional as F

    return F.linear(inputs.to(torch.bfloat16), self.weight, self.bias)


def _embedding(self, inputs):
    from torch.nn import functional as F

    return F.embedding(
        inputs, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
    ).float()


def _layer_norm(self, inputs):
    from torch.nn import functional as F

    return F.layer_norm(
        inputs.float(),
        self.normalized_shape,
        None if self.weight is None else self.weight.float(),
        None if self.bias is None else self.bias.float(),
        self.eps,
    )


def _conv3d(self, inputs):
    return self._conv_forward(inputs.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def _inference_only(model, args):
    import torch

    if model.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("residual FP32 deployment requires eval(), no_grad()/inference_mode(), and no autocast")


def install_deployment_profile(model, profile: str, *, adapter_identity=None) -> list[str]:
    """Install an inference-only, single-CUDA-device SDPA profile on stock Qwen3-VL.

    Parameter/module ownership and state-dict keys are unchanged. Validate the
    entire placement before mutation; support repeated installation as a no-op.
    CUDA's BF16-input/FP32-output GEMM is required; CPU/offload, model parallelism,
    training, autocast and alternate attention kernels are outside this profile.
    """
    import torch
    from torch import nn

    if profile == qwen_fixed.PROFILE:
        from cosmos_framework.inference.reasoner.qwen3_vl_fixed_mma import install

        return install(model, identity=adapter_identity or {})
    if profile in GEMMA_PROFILES:
        from cosmos_framework.utils.generator.gemma_sdpa_layout import install

        return install(model, identity=adapter_identity or {}, profile=profile)
    if profile in SMOLLM_PROFILES:
        from cosmos_framework.utils.generator.smollm_consistent import install

        return install(model, identity=adapter_identity or {}, profile=profile)
    if profile == QWEN3_5_PROFILE:
        from cosmos_framework.utils.generator.qwen3_5_cache import install

        return install(model)
    if profile in QWEN3_VL_PROFILES:
        from cosmos_framework.utils.generator.bf16_cache import install

        return install(model, profile=profile)
    if profile != PROFILE:
        raise ValueError(f"unknown deployment profile: {profile!r}")
    if (
        type(model).__name__ != "Qwen3VLForConditionalGeneration"
        or type(model).__module__ != "transformers.models.qwen3_vl.modeling_qwen3_vl"
        or model.config.model_type != "qwen3_vl"
        or any(
            section._attn_implementation != "sdpa"
            for section in (model.config, model.config.text_config, model.config.vision_config)
        )
        or model.training
    ):
        raise ValueError("residual FP32 deployment requires stock dense Qwen3-VL in eval mode with SDPA")
    parameters = list(model.parameters())
    devices = {p.device for p in parameters}
    if len(devices) != 1 or next(iter(devices)).type != "cuda" or any(p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError("residual FP32 deployment requires all parameters BF16 on one CUDA device; no offload")
    if any("lora_" in name or "magnitude" in name for name, _ in model.named_parameters()):
        raise ValueError("residual FP32 deployment requires a merged model without adapter branches")
    if getattr(model, "_cosmos_deployment_profile", None) == profile:
        return list(model._cosmos_deployment_modules)
    patch = model.model.visual.patch_embed
    if getattr(patch.forward, "_cosmos_channels_last_3d", False):
        raise ValueError("install residual FP32 deployment before the BF16 patch-embed compatibility override")
    replacements = []
    for name, module in model.named_modules():
        function = {
            nn.Linear: _head if name == "lm_head" else _linear,
            nn.Embedding: _embedding,
            nn.LayerNorm: _layer_norm,
            nn.Conv3d: _conv3d,
        }.get(type(module))
        if function is not None:
            if isinstance(module, nn.Embedding) and module.max_norm is not None:
                raise ValueError("deployment embeddings must not mutate their weights with max_norm")
            replacements.append((name, module, function))
    if not any(name == "lm_head" for name, _, _ in replacements):
        raise ValueError("residual FP32 deployment requires the stock linear lm_head")
    # Fail before mutation on runtimes/devices that lack BF16 -> FP32 GEMM.
    device = next(iter(devices))
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        probe = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
        torch.mm(probe, probe.T, out_dtype=torch.float32)
    for _, module, function in replacements:
        forward = types.MethodType(function, module)
        if hasattr(module, "_hf_hook"):
            module._old_forward = forward
        else:
            module.forward = forward
    model.register_forward_pre_hook(_inference_only)
    model._cosmos_deployment_profile = profile
    model._cosmos_deployment_modules = tuple(name for name, _, _ in replacements)
    return list(model._cosmos_deployment_modules)
