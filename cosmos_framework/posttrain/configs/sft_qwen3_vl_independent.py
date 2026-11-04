# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent component settings on the native dense Qwen3-VL SFT path.

SFT_INDEPENDENT_VARIANT=vision_pissa|language_pissa selects two useful method
rotations. Both inherit concrete targets, processor settings and data from the
existing family recipe. Qualification is checkpoint/profile specific.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
from cosmos_framework.posttrain.configs.sft_family_adapters import make_config as family_config


def make_config():
    config = family_config()
    if config.job.group != "qwen3_vl":
        raise ValueError("this independent-component recipe requires a dense Qwen3-VL snapshot")
    variant = os.environ.get("SFT_INDEPENDENT_VARIANT", "vision_pissa")
    if variant == "vision_pissa":
        settings = [
            dict(method="lora", rank=8, alpha=12, dropout=0.1, lr_multiplier=1.0, weight_decay=0.01),
            dict(method="dora", rank=4, alpha=9, use_rslora=True, dropout=0.05, lr_multiplier=0.3),
            dict(method="pissa", rank=3, alpha=6, use_rslora=True, initialization="svd_lowrank", lr_multiplier=0.1),
        ]
    elif variant == "language_pissa":
        settings = [
            dict(method="pissa", rank=4, alpha=7, use_rslora=True, lr_multiplier=0.8),
            dict(method="lora", rank=6, alpha=9, dropout=0.15, lr_multiplier=0.4),
            dict(method="dora", rank=2, alpha=3, dropout=0.05, lr_multiplier=0.15),
        ]
    else:
        raise ValueError("SFT_INDEPENDENT_VARIANT must be vision_pissa|language_pissa")
    config.rl.adapters.components = [
        replace(component, **{"use_rslora": False, "dropout": 0.0, "initialization": "default", **values})
        for component, values in zip(config.rl.adapters.components, settings, strict=True)
    ]
    config.rl.adapters.extra_modules = [
        AdapterExtraModuleConfig(targets=["model.language_model.norm"], lr_multiplier=0.07, weight_decay=0.01)
    ]
    config.job.name = f"independent_{variant}_seed42"
    return config
