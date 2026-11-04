# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent language/merger/vision/action settings for public Alpamayo-R1."""

from dataclasses import replace

from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
from cosmos_framework.posttrain.configs.sft_alpamayo_adapters import make_config as alpamayo_config


def make_config():
    config = alpamayo_config()
    settings = [
        dict(method="lora", rank=8, alpha=12, dropout=0.1, lr_multiplier=0.9, weight_decay=0.01),
        dict(method="dora", rank=4, alpha=7, use_rslora=True, dropout=0.05, lr_multiplier=0.3),
        dict(method="pissa", rank=3, alpha=6, lr_multiplier=0.1),
        dict(method="dora", rank=4, alpha=9, use_rslora=True, dropout=0.1, lr_multiplier=0.8),
        dict(method="lora", rank=1, alpha=2, lr_multiplier=1.2),
    ]
    config.rl.adapters.components = [
        replace(component, **{"use_rslora": False, "dropout": 0.0, "initialization": "default", **values})
        for component, values in zip(config.rl.adapters.components, settings, strict=True)
    ]
    config.rl.adapters.extra_modules = [
        AdapterExtraModuleConfig(targets=["vlm.model.language_model.layers.35.self_attn.k_norm"], lr_multiplier=0.07),
    ]
    config.job.name = "independent_components_seed42"
    return config
