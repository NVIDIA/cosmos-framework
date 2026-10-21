# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""BF16 Joint PPO on the calibrated LIBERO-10 runtime; baseline comparison is test-owned."""

from pathlib import Path

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as base_config


def make_config() -> PosttrainJointPPOConfig:
    config = base_config()
    config.model.backbone_buffer_dtype = "bfloat16"
    config.rollout.evaluation_mode = "native_ode"
    config.environment.config.eval_profile = "aligned_200"
    config.environment.config.eval_manifest = str(
        Path(config.environment.config.eval_manifest).with_name("libero_10-eval-aligned_200.json")
    )
    config.job.name = "libero_10_calibrated_bf16"
    config.rl.max_steps = 80
    config.rl.resume = "off"
    config.rl.validation.save_checkpoint = False
    return config
