# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent component adapters on GR00T N1.7 with a 2x2 data-parallel mesh."""

from dataclasses import replace

from cosmos_framework.posttrain.configs.sft_gr00t_adapters import make_config as gr00t_config


def make_config():
    config = gr00t_config()
    settings = [
        dict(method="lora", rank=8, alpha=12, dropout=0.1, lr_multiplier=0.9, weight_decay=0.01),
        dict(method="dora", rank=4, alpha=7, use_rslora=True, dropout=0.05, lr_multiplier=0.3),
        dict(method="pissa", rank=3, alpha=6, lr_multiplier=0.1),
        dict(method="lora", rank=4, alpha=9, use_rslora=True, dropout=0.1, lr_multiplier=0.8),
    ]
    config.rl.adapters.components = [
        replace(group, **{"use_rslora": False, "dropout": 0.0, "initialization": "default", **values})
        for group, values in zip(config.rl.adapters.components, settings, strict=True)
    ]
    config.model.parallelism.dp_replicate = config.model.parallelism.dp_shard = 2
    config.job.name = "independent_components_hsdp_seed42"
    return config
