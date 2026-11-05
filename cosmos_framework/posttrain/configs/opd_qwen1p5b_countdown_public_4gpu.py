# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four-GPU public OPD recipe with disaggregated policy, rollout, and teacher pools."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig
from cosmos_framework.posttrain.configs.opd_qwen1p5b_countdown import build_config


def make_config() -> PosttrainOPDConfig:
    """Reuse the Countdown recipe on a 2-GPU policy, 1-GPU rollout, and 1-GPU teacher."""
    config = build_config()
    config.defaults = ["_self_"]
    config.grid.backend = "ray"
    config.grid.grouping = "per_role"
    config.grid.gpu_placement = "disaggregated"
    config.grid.shapes = {"policy": [1, 2], "rollout": [1, 1]}
    config.rollout.tensor_parallel_size = 1
    config.rl.teacher.tensor_parallel_size = 1
    config.rl.weight_sync_transport = "nixl"
    config.rl.optimizer.name = "adamw"
    config.job.name = "qwen1p5b_countdown_public_4gpu"
    config.job.wandb_mode = "disabled"
    return config
