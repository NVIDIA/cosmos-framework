# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""RLinf OpenPI π0.5 / LIBERO-Spatial on the existing Joint PPO entry."""

import os
from pathlib import Path

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.posttrain.config.config import build_root
from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import PI05_SPATIAL_EXECUTION_PROFILE, execution_protocol
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig


def make_config() -> PosttrainJointPPOConfig:
    config = build_root(
        PosttrainJointPPOConfig, "joint_ppo", register_experiment_configs=False, register_base_configs=False
    )
    resources = Path(os.environ.get("POSTTRAIN_RESOURCE_ROOT", "/resources"))
    checkpoint = resources / "checkpoints" / "pi05_libero"
    manifests = Path(os.environ.get("LIBERO_MANIFEST_ROOT", "/opt/posttrain/libero/manifests"))
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    config.model = L(Pi05ModelConfig)(
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(resources / "tokenizers" / "paligemma_tokenizer.model"),
        norm_stats_path=str(checkpoint / "physical-intelligence/libero/norm_stats.json"),
        action_chunk=protocol.execution_horizon,
    )
    config.policy = L("cosmos_framework.posttrain.model.pi05.build.build_joint_policy")(config="${model}")
    config.rollout = L("cosmos_framework.posttrain.model.pi05.build.build_joint_rollout")(config="${model}")
    config.environment = L("cosmos_framework.posttrain.embodied.sims.libero.adapter.LiberoEnvironment")(
        config=LiberoConfig(
            execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
            execution_horizon=protocol.execution_horizon,
            max_episode_steps=protocol.max_episode_steps,
            eval_profile="shared_500",
            train_manifest=str(manifests / "libero_spatial-shared_500-train.json"),
            eval_manifest=str(manifests / "libero_spatial-shared_500-eval.json"),
            sim_python=os.environ.get("POSTTRAIN_SIM_PYTHON", ""),
        )
    )
    config.reward = L("cosmos_framework.posttrain.algorithm.reward.decision.build_decision_reward_provider")(
        transform=L("cosmos_framework.posttrain.embodied.sims.libero.reward.LiberoDecisionReward")(
            suite="${environment.config.suite}", execution_profile="${environment.config.execution_profile}"
        )
    )
    config.recipe_check = L("cosmos_framework.posttrain.model.pi05.build.validate_joint_recipe")(
        model="${model}", environment="${environment.config}", recipe="${rl}"
    )
    config.rl.num_envs = 64
    config.rl.window_decisions = 8 * (protocol.max_episode_steps // protocol.execution_horizon)
    config.rl.global_minibatch = 2048
    config.rl.microbatch = 128
    config.rl.verify_behavior_logprobs = False
    config.rl.seed = 42
    config.rl.dual_clip = None
    config.rl.loss_reduction = "fixed_episode_mean"
    config.rl.time_limit_bootstrap = "zero"
    updates = config.rl.decisions // config.rl.global_minibatch
    config.rl.max_steps = 1000 * updates
    config.rl.save_every = 40 * updates
    config.rl.validation.every_steps = 40 * updates
    config.job.project = "cosmos3_action"
    config.job.name = "pi05_libero_spatial"
    return config
