# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit unadapted controls for a qualified inference arithmetic profile.

This is a local, immutable base loader, not an adapter or checkpoint loader.
The only admission is the recorded Gemma1-2B BF16 snapshot promoted to FP32
with the same canonical buffer and operators as its portable/merged profile.
"""

from __future__ import annotations

import copy
from importlib.metadata import version
from pathlib import Path

import torch

from cosmos_framework.utils.generator import gemma_sdpa_layout as gemma
from cosmos_framework.utils.hf_buffers import buffer_contract, prepare_adapter_buffers
from cosmos_framework.utils.hf_snapshot import snapshot_identity


def _validate_device(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("unadapted Gemma deployment requires one rank-local A100 SM80 GPU")


def load_base_model(base_path, *, profile, dtype, device, attn_implementation=None):
    """Validate content/profile compatibility before construction or mutation."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if profile != gemma.SDPA64_PROFILE:
        raise ValueError(f"unsupported base_execution_profile: {profile!r}")
    if dtype not in ("auto", "float32", "fp32") or attn_implementation not in (None, "sdpa"):
        raise ValueError(f"{profile} requires dtype='float32' and SDPA")
    if torch.is_inference_mode_enabled():
        raise ValueError("construct the base runtime outside inference_mode to retain tensor version counters")
    device = torch.device(device)
    _validate_device(device)
    dependencies = {name: version(name).split("+")[0] for name in ("torch", "transformers", "triton")}
    sources = {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(base_path, weights=False),
    }
    identity = {
        "format": gemma.BASE_FORMAT,
        "architecture": "gemma",
        "base_storage_dtype": "bfloat16",
        **sources,
        "model_contract": copy.deepcopy(gemma.BUFFER_CONTRACT),
        "dependencies": dependencies,
        "execution_profile": profile,
        "dtype": "float32",
    }
    gemma.validate_base_identity(identity, profile=profile)
    # Model identity excludes export markers; reject them explicitly so a base
    # route cannot reinterpret an adapter export that happens to share tensors.
    if any((Path(base_path) / name).exists() for name in ("adapter_manifest.json", ".materialized")):
        raise ValueError("base_execution_profile requires an unadapted snapshot, not an adapter/export directory")
    config = AutoConfig.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    if (
        config.model_type != "gemma"
        or getattr(config, "cosmos_inference_profile", None) is not None
        or getattr(config, "quantization_config", None)
        or any(getattr(config, key, None) != value for key, value in gemma.ARCHITECTURE.items())
        or buffer_contract(config) != identity["model_contract"]
    ):
        raise ValueError("base_execution_profile requires the unquantized recorded Gemma1 architecture")
    gemma.validate_configuration(config)
    processor = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    net, info = AutoModelForCausalLM.from_pretrained(
        base_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    problems = {
        key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)
    }
    if problems:
        raise ValueError(f"incomplete unadapted base; no weights may be initialized: {problems}")
    if prepare_adapter_buffers(net) != identity["model_contract"]:
        raise ValueError("unadapted Gemma canonical embedding-scale contract changed")
    net.float().eval().requires_grad_(False)
    if sources != {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(base_path, weights=False),
    }:
        raise ValueError("base/processor changed during construction; use an immutable local snapshot")
    gemma.install(net, identity=identity, profile=profile)
    processor.padding_side = "left"
    provenance = {
        "kind": "unadapted_model_deployment",
        "execution_profile": profile,
        "identity": copy.deepcopy(identity),
        "dtype": "float32",
        "base_path": str(base_path),
        "processor_path": str(base_path),
        "parameter_policy": "all base parameters frozen; no adapters; eager eval with no_grad/inference_mode",
        "precision_contract": "BF16 stored values promoted to FP32; canonical FP32 buffer; FP64 attention temporaries",
        "generation_config": net.generation_config.to_dict(),
        "backend": "Transformers SDPA, single A100 SM80, eager inference, batch one",
    }
    return net, processor, provenance
