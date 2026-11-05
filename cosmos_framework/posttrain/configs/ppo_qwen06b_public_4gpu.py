# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Self-contained four-GPU text PPO recipe using public Qwen3-0.6B weights."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_ppo import PosttrainPPOConfig
from cosmos_framework.posttrain.config.config_ppo import make_config as make_root_config
from cosmos_framework.posttrain.config.rl import OptimizerConfig

_MODEL_ID = "Qwen/Qwen3-0.6B"


def make_config() -> PosttrainPPOConfig:
    """Build a 2-GPU policy/rollout, 1-GPU reference, and 1-GPU critic topology."""
    config = make_root_config()
    config.defaults = ["_self_"]
    config.job.group = "posttrain_ppo_public"
    config.job.name = "qwen06b_public_4gpu"
    config.job.wandb_mode = "disabled"

    config.grid.ranks = 2
    config.grid.reference_ranks = 1
    config.grid.critic_ranks = 1

    for model_config in (config.model_hf, config.reference_model_hf, config.critic_model_hf):
        model_config.name_or_path = _MODEL_ID
        model_config.attn_impl = "eager"
        model_config.grad_ckpt = model_config is not config.reference_model_hf
        model_config.parallelism.dp_replicate = 1
        model_config.parallelism.dp_shard = -1

    config.rollout.name_or_path = _MODEL_ID
    config.rollout.tensor_parallel_size = 2
    config.rollout.gpu_memory_utilization = 0.2
    config.rollout.max_model_len = 1024
    config.rollout.max_tokens = 64
    config.rollout.seed = 42

    config.rl.master_dtype = "bfloat16"
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.transfer_dtype = "bfloat16"
    config.rl.prompts_per_iteration = 4
    config.rl.ppo_epochs = 1
    # The PPO default (1) writes a policy+critic checkpoint after every step.
    config.rl.save_every = 100
    config.rl.optimizer = OptimizerConfig(name="adamw", lr=1e-6, train_keys=[])
    config.rl.critic_optimizer = OptimizerConfig(name="adamw", lr=1e-5, train_keys=[])
    return config
