# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-owned construction and recipe checks for π0.5 Spatial Joint PPO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.rl import JointPPOConfig
    from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
    from cosmos_framework.posttrain.model.pi05.serving.body import Pi05PolicyBody


def validate_joint_recipe(
    model: Pi05ModelConfig, environment: LiberoConfig, recipe: JointPPOConfig, *, ranks: int
) -> None:
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import PI05_SPATIAL_EXECUTION_PROFILE
    from cosmos_framework.posttrain.model.pi05.source import Pi05Source

    model.__post_init__()
    environment.__post_init__()
    if environment.suite != "libero_spatial" or environment.execution_profile != PI05_SPATIAL_EXECUTION_PROFILE:
        raise ValueError("pi05 requires the RLinf Spatial execution profile")
    if (model.action_chunk, model.denoise_steps) != (environment.execution_horizon, 3):
        raise ValueError("pi05 Spatial requires matching five-action chunks and three denoising steps")
    if recipe.loss_reduction != "fixed_episode_mean":
        raise ValueError("pi05 Spatial execution requires rl.loss_reduction=fixed_episode_mean")
    if recipe.window_decisions * environment.execution_horizon % environment.max_episode_steps:
        raise ValueError("pi05 Spatial rl.window_decisions must contain complete physical episodes")
    episodes_per_rank = (recipe.num_envs // ranks) * (
        recipe.window_decisions * environment.execution_horizon // environment.max_episode_steps
    )
    # Success in the first chunk leaves only one valid decision per physical episode.
    if episodes_per_rank < 2:
        raise ValueError(
            "pi05 Spatial requires at least two complete physical episodes per learner rank for actor-local "
            f"normalization; got {episodes_per_rank} from rl.num_envs={recipe.num_envs}, "
            f"rl.window_decisions={recipe.window_decisions}, learner ranks={ranks}"
        )
    Pi05Source.resolve(model)


def build_joint_policy(config: Pi05ModelConfig, *, output: str, parallel_dims: ParallelDims) -> Pi05TrainPolicy:
    from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy

    return Pi05TrainPolicy(config, parallel_dims=parallel_dims)


def build_joint_rollout(config: Pi05ModelConfig, *, output: str, replica: int) -> Pi05PolicyBody:
    from cosmos_framework.posttrain.model.pi05.serving.body import Pi05PolicyBody

    return Pi05PolicyBody(config, output=str(Path(output) / f"rollout-{replica}"), verify_weight_values=True)
