# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dr-GRPO variant of the Qwen2.5-3B GSM8K recipe.

Dr-GRPO changes two choices within the existing clipped GRPO objective: group advantages are
mean-centred without standard-deviation scaling, and every response loss is normalized by the
fixed ``rollout.max_tokens`` bound. The shared GRPO entry, trainer, and objective builder consume
both as resolved config; no algorithm-specific entrypoint is required.

Run from the cosmos-framework repository root:

.. code-block:: bash

    PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.grpo \
        --config cosmos_framework/posttrain/configs/dr_grpo_qwen3b_gsm8k.py
"""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k import build_config as _build_grpo_config


def build_config() -> PosttrainGRPOConfig:
    """Build the Dr-GRPO GSM8K variant with distinct run provenance."""
    cfg = _build_grpo_config()
    cfg.algorithm = "dr_grpo"
    cfg.job.name = "qwen3b_bf16_dr_grpo_seed42"
    cfg.rl.unbiased_advantage = True
    cfg.rl.loss_type = "seq-mean-token-sum-fixed"
    return cfg


def make_config() -> PosttrainGRPOConfig:
    """Return the config consumed by the shared GRPO entrypoint."""
    return build_config()
