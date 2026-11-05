# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Matched dense/factorized adapters with identical HF5 modality/normalization paths.

SFT_COMPARISON_EXECUTION=merged|factorized changes only adapter execution and
its explicit artifact identity. Data, targets, parameters, optimizer and budget
are shared. Set FAMILY=qwen3_vl and a pinned SFT_MODEL_PATH/SFT_DATA_ROOT.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.configs.sft_merged_weight_bf16 import make_config as merged_config


def make_config():
    config = merged_config()
    if config.job.group != "qwen3_vl":
        raise ValueError("the matched WTS recipe requires dense HF Qwen3-VL")
    mode = os.environ.get("SFT_COMPARISON_EXECUTION", "merged")
    if mode == "factorized":
        config.rl.adapters = replace(
            config.rl.adapters,
            model_execution="qwen3_vl_hf_stock_v1",
            components=[replace(group, execution="factorized_v1") for group in config.rl.adapters.components],
        )
    elif mode != "merged":
        raise ValueError("SFT_COMPARISON_EXECUTION must be merged|factorized")
    config.job.name = f"wts_{mode}_seed42"
    return config
