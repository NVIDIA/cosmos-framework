# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Two independent language projection groups on the existing HF Qwen3 path."""

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.configs.sft_family_adapters import make_config as family_config


def make_config():
    config = family_config()
    if config.job.group != "qwen3":
        raise ValueError("this independent language recipe requires a Qwen3 text snapshot")
    config.rl.adapters.components = [
        AdapterComponentConfig(
            targets=["model.layers.*.self_attn.q_proj"],
            rank=8,
            alpha=12,
            use_rslora=True,
            dropout=0.1,
            lr_multiplier=1.0,
            weight_decay=0.01,
        ),
        AdapterComponentConfig(
            targets=["model.layers.*.self_attn.v_proj"],
            method="pissa",
            rank=4,
            alpha=7,
            lr_multiplier=0.4,
        ),
    ]
    config.rl.adapters.extra_modules = [AdapterExtraModuleConfig(targets=["model.norm"], lr_multiplier=0.1)]
    config.job.name = "independent_language_seed42"
    return config
