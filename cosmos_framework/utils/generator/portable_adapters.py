# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared portable artifact assembly for public inference consumers.

Reuse the adapter identity, selector and tensor machinery, without importing a
trainer, training model builder or qualification factory. Heavy dependencies are
lazy so ordinary merged inference retains its existing dependency boundary.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version

from cosmos_framework.utils.generator.bf16_cache_contract import PROFILE as CANONICAL  # noqa: F401
from cosmos_framework.utils.generator.bf16_cache_contract import QWEN3_VL_PROFILES
from cosmos_framework.utils.generator.factorized_inference import PROFILE as FACTORIZED
from cosmos_framework.utils.generator.gemma_sdpa_layout import PROFILES as GEMMA_PROFILES
from cosmos_framework.utils.generator.portable_qwen_vl import PROFILE as ORDERED_VLM
from cosmos_framework.utils.generator.smollm_consistent import PROFILES as SMOLLM_PROFILES


@dataclass(frozen=True)
class PortableAdapterConfig:
    directory: str
    execution_profile: str
    cache_effective_weights: bool = False
    processor_path: str | None = None

    def __post_init__(self):
        if not isinstance(self.directory, str) or not self.directory:
            raise ValueError("portable_adapter.directory must name a complete local portable adapter")
        profiles = (*QWEN3_VL_PROFILES, FACTORIZED, *SMOLLM_PROFILES, *GEMMA_PROFILES, ORDERED_VLM)
        if self.execution_profile not in profiles:
            raise ValueError(f"portable execution_profile must be one of {profiles!r}")
        if type(self.cache_effective_weights) is not bool:
            raise ValueError("cache_effective_weights must be a boolean")
        if (
            self.execution_profile in (FACTORIZED, *SMOLLM_PROFILES, *GEMMA_PROFILES, ORDERED_VLM)
            and self.cache_effective_weights
        ):
            raise ValueError(
                "this profile preserves its declared adapter execution; effective-weight cache is unsupported"
            )


def _validate_device(device):
    import torch

    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("these portable deployment profiles require one A100 (SM80), without device sharding/offload")


def load_portable_model(base_path, config, *, dtype, device, attn_implementation=None):
    """Load an immutable local base + portable artifact with an explicit profile.

    Validate the captured artifact, dependencies, base/processor content hashes
    and resolved placement BEFORE installing adapters. No PiSSA decomposition,
    missing-state initialization, training checkpoint or remote-code fallback.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from cosmos_framework.posttrain.checkpoint.adapters import (
        adapter_configuration,
        load_adapter_state,
        make_adapter_identity,
        read_adapter_snapshot,
        snapshot_identity,
    )
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, install_adapters, resolve_targets
    from cosmos_framework.utils.generator import (
        factorized_inference,
        gemma_sdpa_layout,
        portable_qwen_vl,
        smollm_consistent,
    )
    from cosmos_framework.utils.generator.bf16_cache_contract import validate_adapter_identity

    if not isinstance(config, PortableAdapterConfig):
        config = PortableAdapterConfig(**config)
    profile = config.execution_profile
    target_dtype = "bfloat16" if profile in QWEN3_VL_PROFILES else "float32"
    aliases = {"bf16": "bfloat16", "fp32": "float32"}
    if aliases.get(dtype, dtype) not in ("auto", target_dtype) or attn_implementation not in (None, "sdpa"):
        raise ValueError(f"{profile} requires dtype={target_dtype!r} and SDPA")
    if torch.is_inference_mode_enabled():
        raise ValueError(
            "construct the portable runtime outside inference_mode so cache sources retain version counters"
        )
    record, state = read_adapter_snapshot(config.directory)
    identity = record["identity"]
    if profile in QWEN3_VL_PROFILES:
        validate_adapter_identity(identity, profile=profile)
    elif profile in SMOLLM_PROFILES:
        smollm_consistent.validate_identity(identity, profile=profile)
    elif profile in GEMMA_PROFILES:
        gemma_sdpa_layout.validate_identity(identity, profile=profile)
    elif profile == ORDERED_VLM:
        portable_qwen_vl.validate_identity(identity)
    else:
        factorized_inference.validate_identity(identity)
    for name, expected in identity["dependencies"].items():
        try:
            actual = version(name).split("+")[0]
        except ModuleNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError(f"portable dependency mismatch for {name}: artifact={expected!r}, installed={actual!r}")
    device = torch.device(device)
    _validate_device(device)
    processor_path = config.processor_path or base_path
    sources = {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(processor_path, weights=False),
    }
    for name, actual in sources.items():
        if actual != identity[name]:
            raise ValueError(f"portable {name} identity mismatch; stage the exact recorded immutable snapshot")
    hf_config = AutoConfig.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    family = (
        identity["architecture"]
        if profile == ORDERED_VLM
        else ("gemma" if profile in GEMMA_PROFILES else ("llama" if profile in SMOLLM_PROFILES else "qwen3_vl"))
    )
    if hf_config.model_type != family or getattr(hf_config, "quantization_config", None):
        raise ValueError(f"{profile} requires an unquantized stock {family} base")
    source_identity = None
    if profile in GEMMA_PROFILES:
        from cosmos_framework.posttrain.model.hf_adapter_buffers import buffer_contract, validate_buffer_contract

        gemma_sdpa_layout.validate_configuration(hf_config)
        validate_buffer_contract(identity, hf_config)
        source_identity = {**sources, "model_contract": buffer_contract(hf_config)}
    auto_class = AutoModelForCausalLM if profile in (*SMOLLM_PROFILES, *GEMMA_PROFILES) else AutoModelForImageTextToText
    processor = None
    if profile == ORDERED_VLM:
        processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True, trust_remote_code=False)
        portable_qwen_vl.validate_processor(processor, family)
        net = portable_qwen_vl.load_base(base_path, hf_config, device, identity)
    else:
        net, info = auto_class.from_pretrained(
            base_path,
            config=hf_config,
            dtype=torch.bfloat16,
            device_map={"": str(device)},
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        problems = {
            k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)
        }
        if problems:
            raise ValueError(f"incomplete portable base model: {problems}; no missing weights may be initialized")
    adapters = adapter_configuration(identity)
    manifest = resolve_targets(net, adapters, family=family)
    expected = make_adapter_identity(
        base_path,
        processor_path,
        manifest,
        identity["media"],
        base_dtype="bfloat16",
        architecture=family,
        source_identity=source_identity,
    )
    if expected != identity:
        differences = sorted(k for k in set(expected) | set(identity) if expected.get(k) != identity.get(k))
        raise ValueError(f"portable compatibility mismatch before injection: {differences}")
    if profile in GEMMA_PROFILES:
        from cosmos_framework.posttrain.model.hf_adapter_buffers import prepare_adapter_buffers

        # Reuse the training/export contract before injection; promoting a
        # BF16-rounded nonpersistent scale cannot recover the recorded value.
        if prepare_adapter_buffers(net) != identity["model_contract"]:
            raise ValueError("portable Gemma embedding-scale contract changed during construction")
    install_adapters(net, adapters, initialize=False, family=family)
    net._posttrain_adapter_identity = identity
    load_adapter_state(net, identity, state)
    net.eval()
    if profile in SMOLLM_PROFILES:
        net.float()
        smollm_consistent.install(net, identity=identity, adapter_type=PosttrainAdapterLinear, profile=profile)
    elif profile in GEMMA_PROFILES:
        net.float()
        gemma_sdpa_layout.install(net, identity=identity, adapter_type=PosttrainAdapterLinear, profile=profile)
    elif profile == FACTORIZED:
        net.float()  # Explicit deployment precision; installer never replaces owned tensors.
        factorized_inference.install(net, adapter_type=PosttrainAdapterLinear)
    elif profile == ORDERED_VLM:
        net.float()  # Explicit FP32 reference deployment, not original BF16-storage inference.
        portable_qwen_vl.install_inference_guard(net, identity)
    else:
        from cosmos_framework.posttrain.model.adapter_bf16_execution import install_adapter_inference_profile

        install_adapter_inference_profile(net, profile, cache_effective_weights=config.cache_effective_weights)
    processor_class = AutoTokenizer if profile in (*SMOLLM_PROFILES, *GEMMA_PROFILES) else AutoProcessor
    if processor is None:
        processor = processor_class.from_pretrained(processor_path, local_files_only=True, trust_remote_code=False)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    # Detect snapshot edits during construction. The portable tensors themselves
    # came from the single verified byte payload retained above.
    if sources != {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(processor_path, weights=False),
    }:
        raise ValueError("base/processor snapshot changed while loading; use an immutable local snapshot")
    provenance = {
        "kind": "portable_adapter_deployment",
        "execution_profile": profile,
        "dtype": target_dtype,
        "cache_effective_weights": config.cache_effective_weights,
        "identity": copy.deepcopy(identity),
        "weights_sha256": record["weights_sha256"],
        "manifest_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        "manifest_digest_encoding": "canonical JSON, sort_keys=True",
        "base_path": str(base_path),
        "processor_path": str(processor_path),
        "adapter_directory": config.directory,
        "generation_config": net.generation_config.to_dict(),
        "backend": "Transformers SDPA, single A100 SM80, eager inference, batch one",
    }
    if profile == ORDERED_VLM:
        provenance["precision_contract"] = {
            "artifact_base_storage": identity["base_storage_dtype"],
            "deployment_parameters": "float32",
            "component_execution": sorted(
                {row["settings"].get("execution", "factorized_v1") for row in identity["resolved"]["targets"]}
            ),
            "original_storage_equivalence": "not claimed; this loader is the explicit FP32 reference",
            "cache_consistent_numerics": "not claimed; recorded ordered forward and native SDPA",
        }
    return net, processor, provenance


def model_cache_signature(model):
    """Invalidate runtime feature/KV caches on tracked model mutations as well."""
    tensors = list(model.named_parameters(remove_duplicate=False))
    # Nonpersistent rotary/cache buffers are expected to change during inference.
    tensors.extend((n, b) for n, b in model.named_buffers() if n.endswith(("initial_A", "initial_B")))
    return (
        getattr(model, "_cosmos_deployment_profile", None),
        tuple(
            (name, id(module.config), getattr(module.config, "_attn_implementation", None))
            for name, module in model.named_modules()
            if hasattr(module, "config")
        ),
        tuple((n, id(t), t._version, t.data_ptr(), t.dtype, t.device, tuple(t.shape)) for n, t in tensors),
        tuple(
            (n, id(m), getattr(m, "method", None), getattr(m, "_lora_scale", None), getattr(m, "execution", None))
            for n, m in model.named_modules()
            if hasattr(m, "lora_A")
        ),
    )


def clear_derived_weights(model):
    from cosmos_framework.posttrain.model.adapter_inference_cache import clear_adapter_inference_cache

    clear_adapter_inference_cache(model)
