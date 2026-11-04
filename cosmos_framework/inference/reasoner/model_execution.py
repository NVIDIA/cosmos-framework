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
    if declared != PROFILE or bound != declared:
        raise ValueError("missing, conflicting or unknown model execution profile in config/.materialized; re-export")
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
        raise ValueError("stock MoE execution requires Transformers 5.8.1 / PyTorch 2.10.0")
    return declared


def validate_loaded_model(model):
    import torch

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
