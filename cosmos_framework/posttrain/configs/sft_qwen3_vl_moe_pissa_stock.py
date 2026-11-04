# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""New training contract: dense PiSSA and stock-compatible native MoE execution."""

from cosmos_framework.posttrain.configs.sft_qwen3_vl_moe_pissa_dense import make_config as dense_config


def make_config():
    config = dense_config()
    config.rl.adapters.model_execution = "qwen3_vl_moe_stock_v1"
    config.job.name = "pissa_stock_v1_seed42"
    return config
