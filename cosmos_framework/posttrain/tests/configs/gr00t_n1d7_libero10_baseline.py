# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh HF baseline: native simulator protocol, BF16, no optimizer updates or checkpoints."""

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as training_config


def make_config() -> PosttrainJointPPOConfig:
    config = training_config()
    config.job.name = "libero_10_baseline"
    config.rl.resume = "off"
    config.rl.max_steps = 0
    config.rl.save_every = 0
    config.rl.validation.save_checkpoint = False
    return config
