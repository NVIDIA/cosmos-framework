# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four-GPU public π0.5/LIBERO-Spatial Joint PPO profile."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.configs.pi05_libero_spatial_ppo import make_config as make_base_config


def make_config() -> PosttrainJointPPOConfig:
    """Use two complete episodes per learner rank and one optimizer update per window."""
    config = make_base_config()
    config.rl.num_envs = 4
    config.rl.window_decisions = 96
    config.rl.global_minibatch = 384
    config.rl.microbatch = 16
    config.job.name = "pi05_libero_spatial_public_4gpu"
    config.job.wandb_mode = "disabled"
    return config
