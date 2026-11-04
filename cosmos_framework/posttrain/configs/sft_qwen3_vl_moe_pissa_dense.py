# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""New-run dense-effective PiSSA on native Qwen3-VL-MoE; never rewrites legacy artifacts."""

from dataclasses import replace

from cosmos_framework.posttrain.configs.sft_family_adapters import make_config as family_config


def make_config():
    config = family_config()
    if config.job.group != "qwen3_vl_moe":
        raise ValueError("this recipe requires the native Qwen3-VL-MoE architecture")
    config.rl.adapters.components = [
        replace(group, method="pissa", execution="pissa_dense_v1", dropout=0) for group in config.rl.adapters.components
    ]
    config.job.name = "pissa_dense_v1_seed42"
    return config
