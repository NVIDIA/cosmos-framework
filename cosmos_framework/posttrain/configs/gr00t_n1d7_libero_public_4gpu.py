# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four-GPU public GR00T-N1.7/LIBERO-Spatial Joint PPO profile."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config as make_base_config


def make_config() -> PosttrainJointPPOConfig:
    """Use one environment per GPU and one optimizer update per decision window."""
    config = make_base_config()
    config.rl.num_envs = 4
    config.rl.window_decisions = 16
    config.rl.global_minibatch = 64
    config.rl.microbatch = 2
    config.job.name = "gr00t_n1d7_libero_public_4gpu"
    config.job.wandb_mode = "disabled"
    return config
