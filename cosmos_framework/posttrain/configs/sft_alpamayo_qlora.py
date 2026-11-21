# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native Alpamayo-R1-10B PhysicalAI trajectory flow QLoRA; supply staged local paths."""

from cosmos_framework.posttrain.config.action_sft import make_config as action_config
from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig
from cosmos_framework.posttrain.model.alpamayo.training import AlpamayoDataConfig


def make_config():
    config = action_config()
    config.job.name = "alpamayo_r1_nf4_flow"
    config.policy_factory = "cosmos_framework.posttrain.model.alpamayo.policy.build_policy"
    config.dense_policy_factory = "cosmos_framework.posttrain.model.alpamayo.dense.build_dense_policy"
    config.dense_sidecars_factory = "cosmos_framework.posttrain.model.alpamayo.dense.dense_sidecars"
    config.data_factory = "cosmos_framework.posttrain.model.alpamayo.training.build_data"
    config.objective_factory = "cosmos_framework.posttrain.model.alpamayo.training.build_objective"
    config.evaluation_callback = "cosmos_framework.posttrain.model.alpamayo.training.evaluate_trajectories"
    config.model = AlpamayoSFTModelConfig(
        checkpoint_path="/path/to/Alpamayo-R1-10B-dd4a24c", processor_path="/path/to/Qwen3-VL-8B-processor-0c351dd"
    )
    config.data = AlpamayoDataConfig(train_path="/path/to/train.jsonl", validation_path="/path/to/validation.jsonl")
    config.rl.max_steps = 100
    config.rl.evaluation_every = 25
    config.rl.optimizer.lr = 1e-4
    config.rl.optimizer.impl = "fused"
    return config
