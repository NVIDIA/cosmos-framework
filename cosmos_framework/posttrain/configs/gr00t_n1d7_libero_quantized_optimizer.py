# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native four-GPU GR00T Joint PPO with 8-bit AdamW moments and explicitly owned FP32 update masters.

Use rl.optimizer.name=paged_adamw_8bit for managed state, or
rl.optimizer.name=adamw rl.optimizer.impl=fused for the matched FP32-state
control. The public recipe keeps its original optimizer/storage policy.
"""

from dataclasses import replace

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_public_4gpu import make_config as make_base_config


def make_config():
    config = make_base_config()
    config.rl.optimizer = replace(config.rl.optimizer, name="adamw_bnb_8bit", impl="", update_policy="fp32_master")
    config.job.name = "gr00t_n1d7_libero_quantized_optimizer"
    return config
