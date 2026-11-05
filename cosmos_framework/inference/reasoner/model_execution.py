# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate the stock-loader contract of versioned MoE adapter merges.

No adapter branches or custom arithmetic are installed. The qualified stock
loader requires FP32, SDPA and its grouped expert backend at the recorded pins.
"""

import json
from importlib.metadata import version
from pathlib import Path

PROFILE = "qwen3_vl_moe_stock_v1"
MERGED_WEIGHT_PROFILE = "merged_weight_bf16_v1"
FACTOR_PROFILE = "qwen3_vl_hf_stock_v1"


def model_execution_profile(path):
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("model config must be a JSON object")
    declared = config.get("cosmos_model_execution")
    try:
        marker = json.loads((Path(path) / ".materialized").read_text())
    except (OSError, ValueError) as error:
        if declared is None:
            return None
        raise ValueError("model execution profile requires a committed .materialized marker") from error
    identity = marker.get("identity", {}) if isinstance(marker, dict) else {}
    metadata = identity.get("export_metadata", {}) if isinstance(identity, dict) else {}
    adapter = metadata.get("adapter_identity", {}) if isinstance(metadata, dict) else {}
    resolved = adapter.get("resolved", {}) if isinstance(adapter, dict) else {}
    bound = resolved.get("model_execution") if isinstance(resolved, dict) else None
    if declared is None and bound is None:
        return None
    if declared not in (PROFILE, MERGED_WEIGHT_PROFILE, FACTOR_PROFILE) or bound != declared:
        raise ValueError("missing, conflicting or unknown model execution profile in config/.materialized; re-export")
    if declared == MERGED_WEIGHT_PROFILE:
        dtype = config.get("dtype", config.get("torch_dtype"))
        targets = resolved.get("targets", [])
        if (
            config.get("model_type") not in ("qwen3_vl", "qwen3_5")
            or dtype not in ("float32", "bfloat16")
            or identity.get("output_dtype") != dtype
            or metadata.get("format") != "cosmos-merged-adapter-v1"
            or adapter.get("format") != "cosmos-posttrain-adapter-v4"
            or adapter.get("architecture") != config.get("model_type")
            or adapter.get("base_storage_dtype") != "bfloat16"
            or not targets
            or any(
                row.get("settings", {}).get("execution") != MERGED_WEIGHT_PROFILE
                or row.get("settings", {}).get("dropout") != 0
                for row in targets
            )
            or adapter.get("dependencies", {}).get("transformers") != "5.8.1"
            or adapter.get("dependencies", {}).get("torch") != "2.10.0"
        ):
            raise ValueError("merged-weight execution requires a committed dense Qwen3-VL/Qwen3.5 v4 adapter merge")
        precision = metadata.get("deployment_precision", {})
        if dtype == "bfloat16":
            expected = {
                "dtype": "bfloat16",
                "weight_conversion": "float32_merge_then_bfloat16",
                "equivalence": "precision_specific_dense_execution_v1",
            }
            if precision.get("execution_profile") is not None:
                from cosmos_framework.utils.generator.bf16_cache_contract import PROFILE as CACHE_PROFILE
                from cosmos_framework.utils.generator.bf16_cache_contract import validate_adapter_identity

                validate_adapter_identity(adapter)
                expected["execution_profile"] = CACHE_PROFILE
                if config.get("cosmos_inference_profile") != CACHE_PROFILE:
                    raise ValueError("canonical cache deployment is not bound to the model config")
            if precision != expected:
                raise ValueError("new BF16 execution requires its precision-specific deployment contract")
        elif precision or config.get("cosmos_inference_profile") is not None:
            raise ValueError("FP32 equivalence must not acquire BF16 deployment arithmetic")
    elif declared == FACTOR_PROFILE:
        if (
            config.get("model_type") != "qwen3_vl"
            or config.get("dtype", config.get("torch_dtype")) != "float32"
            or identity.get("output_dtype") != "float32"
            or metadata.get("format") != "cosmos-merged-adapter-v1"
            or adapter.get("format") != "cosmos-posttrain-adapter-v3"
            or adapter.get("architecture") != "qwen3_vl"
            or adapter.get("base_storage_dtype") != "bfloat16"
            or not resolved.get("targets")
            or any(
                row.get("settings", {}).get("execution", "factorized_v1") != "factorized_v1"
                for row in resolved["targets"]
            )
            or adapter.get("dependencies", {}).get("transformers") != "5.8.1"
            or adapter.get("dependencies", {}).get("torch") != "2.10.0"
        ):
            raise ValueError("stock HF factorized execution requires a committed FP32 Qwen3-VL v3 adapter merge")
    else:
        if (
            config.get("model_type") != "qwen3_vl_moe"
            or config.get("dtype", config.get("torch_dtype")) != "float32"
            or identity.get("output_dtype") != "float32"
            or metadata.get("format") != "cosmos-merged-adapter-v1"
            or adapter.get("format") != "cosmos-posttrain-adapter-v3"
            or adapter.get("architecture") != "qwen3_vl_moe"
            or adapter.get("dependencies", {}).get("transformers") != "5.8.1"
            or adapter.get("dependencies", {}).get("torch") != "2.10.0"
        ):
            raise ValueError("stock MoE execution requires a committed FP32 Qwen3-VL-MoE v3 adapter merge")
    if version("transformers") != "5.8.1" or version("torch").split("+")[0] != "2.10.0":
        raise ValueError("versioned stock execution requires Transformers 5.8.1 / PyTorch 2.10.0")
    return declared


def validate_loaded_model(model):
    import torch

    profile = getattr(model.config, "cosmos_model_execution", None)
    if profile in (MERGED_WEIGHT_PROFILE, FACTOR_PROFILE):
        family = model.config.model_type
        expected = {"qwen3_vl": "Qwen3VLForConditionalGeneration", "qwen3_5": "Qwen3_5ForConditionalGeneration"}
        dtype = getattr(model.config, "dtype", None)
        configs = [model.config, model.config.text_config, model.config.vision_config]
        configs.extend(getattr(module, "config", None) for module in model.modules())
        if (
            family not in expected
            or (profile == FACTOR_PROFILE and (family != "qwen3_vl" or dtype != torch.float32))
            or type(model).__name__ != expected[family]
            or type(model).__module__ != f"transformers.models.{family}.modeling_{family}"
            or dtype not in (torch.float32, torch.bfloat16)
            or any(parameter.dtype != dtype for parameter in model.parameters())
            or any(
                getattr(section, "_attn_implementation", "sdpa") != "sdpa" for section in configs if section is not None
            )
            or any("lora_" in name or name.endswith(".magnitude") for name, _ in model.named_parameters())
        ):
            raise ValueError("loaded merged-weight model differs from its stock dtype/SDPA/adapter-free contract")
        return
    configs = [model.config, model.config.text_config, model.config.vision_config]
    configs.extend((model.model.language_model.config, model.model.visual.config))
    configs.extend(
        module.config
        for module in model.modules()
        if type(module).__name__ in ("Qwen3VLMoeTextAttention", "Qwen3VLMoeVisionAttention")
    )
    if (
        type(model).__module__ != "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"
        or any(config._attn_implementation != "sdpa" for config in configs)
        or model.config.text_config._experts_implementation != "grouped_mm"
        or any(
            module.config._experts_implementation != "grouped_mm"
            for module in model.modules()
            if type(module).__name__ == "Qwen3VLMoeTextExperts"
        )
        or any(parameter.dtype != torch.float32 for parameter in model.parameters())
    ):
        raise ValueError("loaded stock MoE execution differs from its FP32/SDPA/grouped_mm contract")
