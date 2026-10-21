# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 / LIBERO-10 bindings for Joint PPO."""

from pathlib import Path

from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config as spatial_root
from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_PROTOCOLS


def make_config() -> PosttrainJointPPOConfig:
    config = spatial_root()
    config.job.name = "libero_10"
    environment = config.environment.config
    environment.suite = "libero_10"
    environment.execution_horizon = LIBERO_PROTOCOLS[environment.suite].execution_horizon
    environment.max_episode_steps = LIBERO_PROTOCOLS[environment.suite].max_episode_steps
    environment.eval_profile = "legacy_100"
    environment.train_manifest = str(Path(environment.train_manifest).with_name("libero_10-train.json"))
    environment.eval_manifest = str(Path(environment.eval_manifest).with_name("libero_10-eval.json"))
    config.model.checkpoint_path = str(Path(config.model.checkpoint_path).with_name("libero_10"))
    config.model.execution_horizon = environment.execution_horizon
    return config
