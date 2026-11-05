# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four-GPU public profile for Qwen2.5-3B GRPO on GSM8K."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k import build_config


def make_config() -> PosttrainGRPOConfig:
    """Reuse the qualified GSM8K recipe with a four-rank colocated world."""
    config = build_config()
    # This HF-only recipe owns every field it consumes. Do not compose generator model,
    # tokenizer, scheduler, checkpoint, or cluster defaults from the host repository.
    config.defaults = ["_self_"]
    config.grid.ranks = 4
    config.rollout.tensor_parallel_size = 4
    config.job.name = "qwen3b_gsm8k_public_4gpu"
    config.job.wandb_mode = "disabled"
    return config
