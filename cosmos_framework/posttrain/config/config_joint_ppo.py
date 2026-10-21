# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-independent Joint PPO root and cross-role configuration checks."""

from __future__ import annotations

import attrs
from omegaconf import DictConfig, OmegaConf

from cosmos_framework.utils.lazy_config import LazyDict, instantiate
from cosmos_framework.posttrain.config.config import PosttrainConfig
from cosmos_framework.posttrain.config.rl import JointPPOConfig


@attrs.define(slots=False)
class PosttrainJointPPOConfig(PosttrainConfig):
    algorithm: str = "joint_ppo"
    rl: JointPPOConfig = attrs.field(factory=JointPPOConfig)
    policy: LazyDict | None = None
    rollout: LazyDict | None = None
    environment: LazyDict | None = None
    reward: LazyDict | None = None
    recipe_check: LazyDict | None = None


def validate_joint_ppo_config(config: PosttrainJointPPOConfig, ranks: int) -> None:
    from cosmos_framework.utils.launch_defaults import (
        get_wandb_project,
    )
    from cosmos_framework.posttrain.config.rl import require_full_emergency_plan

    config = resolve_joint_config(config)
    if config.algorithm != "joint_ppo":
        raise ValueError("joint PPO requires algorithm=joint_ppo")
    recipe = instantiate(config.rl)
    recipe.__post_init__()
    recipe.optimizer.__post_init__()
    recipe.validation.__post_init__()
    require_full_emergency_plan(
        recipe,
        algorithm="joint PPO",
        reason="its loop saves the full plan at every boundary, at the cadence and at a termination trigger alike "
        "(entrypoints/joint_ppo.py, save_full)",
    )
    if ranks < 1 or recipe.num_envs % ranks:
        raise ValueError("joint PPO requires a positive learner width and equal environment shards")
    if recipe.decisions // ranks <= 1:
        raise ValueError(
            "rl.num_envs * rl.window_decisions / learner ranks must exceed 1 for actor-local normalization"
        )
    updates = recipe.decisions // recipe.global_minibatch
    for name, value in (
        ("max_steps", recipe.max_steps),
        ("save_every", recipe.save_every),
        ("validation.every_steps", recipe.validation.every_steps),
    ):
        if value % updates:
            raise ValueError(f"rl.{name} must be divisible by {updates} updates per window")
    for name in ("policy", "rollout", "environment", "reward"):
        spec = getattr(config, name)
        if spec is None or "_target_" not in spec:
            raise ValueError(f"config.{name} must declare a LazyCall builder")
    environment = instantiate(config.environment)
    environment.validate_configuration()
    if config.recipe_check is not None:
        instantiate(config.recipe_check, ranks=ranks)


def make_config() -> PosttrainJointPPOConfig:
    from cosmos_framework.posttrain.config.config import build_root

    return build_root(
        PosttrainJointPPOConfig, "joint_ppo", register_experiment_configs=False, register_base_configs=False
    )


def resolve_joint_config(config: PosttrainJointPPOConfig | DictConfig) -> PosttrainJointPPOConfig:
    """Bind lazy component interpolations to the single root before any component is read."""
    structured = (
        config if isinstance(config, DictConfig) else OmegaConf.structured(config, flags={"allow_objects": True})
    )
    OmegaConf.resolve(structured)
    return OmegaConf.to_object(structured)
