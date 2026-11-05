# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""New-artifact dense-weight BF16 training on two explicit stock HF paths.

Uses the existing SFT trainer/packer/checkpoint flow. SFT_MODEL_PATH is a pinned
Qwen3-VL or Qwen3.5 snapshot; SFT_DATA_ROOT contains mixed-media SFT JSONL rows.
SFT_MERGED_VARIANT=vision_pissa|language_pissa rotates independent methods.
Legacy factorized checkpoints cannot warm-start this execution contract.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.configs.sft_family_adapters import make_config as family_config


def make_config():
    config = family_config()
    if config.job.group not in ("qwen3_vl", "qwen3_5"):
        raise ValueError("merged-weight recipe binds only dense Qwen3-VL and Qwen3.5")
    config.model_backend = "hf"
    config.model_hf = HFModelConfig(
        name_or_path=os.environ["SFT_MODEL_PATH"], dtype="bfloat16", attn_impl="sdpa", grad_ckpt=True
    )
    variant = os.environ.get("SFT_MERGED_VARIANT", "vision_pissa")
    if variant == "vision_pissa":
        settings = [
            dict(method="lora", rank=8, alpha=12, use_rslora=False, lr_multiplier=1.0, weight_decay=0.01),
            dict(method="dora", rank=4, alpha=9, use_rslora=True, lr_multiplier=0.3),
            dict(method="pissa", rank=3, alpha=6, use_rslora=True, lr_multiplier=0.1),
        ]
    elif variant == "language_pissa":
        settings = [
            dict(method="pissa", rank=4, alpha=7, use_rslora=True, lr_multiplier=0.8),
            dict(method="lora", rank=6, alpha=9, use_rslora=False, lr_multiplier=0.4),
            dict(method="dora", rank=2, alpha=3, use_rslora=True, lr_multiplier=0.15),
        ]
    else:
        raise ValueError("SFT_MERGED_VARIANT must be vision_pissa|language_pissa")
    components = [
        replace(
            group,
            **values,
            dropout=0.0,
            execution="merged_weight_bf16_v1",
            initialization="svd_lowrank" if values["method"] == "pissa" else "default",
        )
        for group, values in zip(config.rl.adapters.components, settings, strict=True)
    ]
    config.rl.adapters = AdapterConfig(
        model_execution="merged_weight_bf16_v1",
        components=components,
        extra_modules=[
            AdapterExtraModuleConfig(targets=["model.language_model.norm"], lr_multiplier=0.07, weight_decay=0.01)
        ],
    )
    config.job.name = f"merged_weight_{variant}_seed42"
    return config
