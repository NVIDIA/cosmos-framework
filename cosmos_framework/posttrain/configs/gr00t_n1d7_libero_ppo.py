# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 / LIBERO-Spatial bindings for the shared Joint PPO entry."""

import os
from dataclasses import asdict
from pathlib import Path

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.posttrain.config.config import build_root
from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.model.gr00t.build import CHECKPOINT_REVISION


def make_model_config(
    checkpoint_path: str, execution_horizon: int = 16, *, backbone_buffer_dtype: str | None = None
) -> Gr00tN1d7ModelConfig:
    return Gr00tN1d7ModelConfig(
        checkpoint_path=checkpoint_path,
        backbone_buffer_dtype=backbone_buffer_dtype,
        embodiment_tag="libero_sim",
        model_action_horizon=40,
        training_prefix=16,
        execution_horizon=execution_horizon,
        action_dim=7,
        denoising_steps=4,
        camera_map={"image": "image", "wrist_image": "wrist_image"},
        state_map={key: key for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")},
        language_key="annotation.human.action.task_description",
    )


def make_config() -> PosttrainJointPPOConfig:
    config = build_root(
        PosttrainJointPPOConfig, "joint_ppo", register_experiment_configs=False, register_base_configs=False
    )
    # Preserve the pinned RLinf N1.7 timeout targets, independently of simulator ending facts.
    config.rl.time_limit_bootstrap = "zero"
    resources = Path(os.environ.get("POSTTRAIN_RESOURCE_ROOT", "/resources"))
    manifests = Path(os.environ.get("LIBERO_MANIFEST_ROOT", "/opt/posttrain/libero/manifests"))
    checkpoint = resources / "checkpoints" / CHECKPOINT_REVISION / "libero_spatial"
    config.model = L(Gr00tN1d7ModelConfig)(**asdict(make_model_config(str(checkpoint))))
    config.policy = L("cosmos_framework.posttrain.model.gr00t.build.build_joint_policy")(config="${model}")
    config.rollout = L("cosmos_framework.posttrain.model.gr00t.build.build_joint_rollout")(
        config="${model}", evaluation_mode="ode"
    )
    config.environment = L("cosmos_framework.posttrain.embodied.sims.libero.adapter.LiberoEnvironment")(
        config=LiberoConfig(
            sim_python=os.environ.get("POSTTRAIN_SIM_PYTHON", "/opt/venv/libero/bin/python"),
            train_manifest=str(manifests / "libero_spatial-train.json"),
            eval_manifest=str(manifests / "libero_spatial-eval.json"),
        )
    )
    config.reward = L("cosmos_framework.posttrain.algorithm.reward.decision.build_decision_reward_provider")(
        transform=L("cosmos_framework.posttrain.embodied.sims.libero.reward.LiberoDecisionReward")(
            suite="${environment.config.suite}"
        )
    )
    config.recipe_check = L("cosmos_framework.posttrain.model.gr00t.build.validate_joint_recipe")(
        model="${model}", environment="${environment.config}"
    )
    config.job.project = "cosmos3_action"
    config.job.name = "libero_spatial"
    config.job.wandb_mode = "disabled"
    return config
