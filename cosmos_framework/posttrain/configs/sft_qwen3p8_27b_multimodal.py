# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Functional Qwen3.8-27B full-parameter SFT example, not a benchmark recipe.

Generate the dataset with scripts/prepare_sft_example.py. Prefer SFT_MODEL_PATH pointing at an
immutable local HF snapshot. Torchrun chooses the DP world; no GPU allocation is made by this file.
For Obj1k Blender code, use sft_qwen3p8_27b_obj1k.py, which inherits this recipe and increases the
sequence budget for its longer text-only targets.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.config_sft import make_config as make_root_config
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTDatasetConfig, SFTValidationConfig


def build_config() -> PosttrainSFTConfig:
    config = make_root_config()
    config.job.project = "cosmos3_reasoner"
    config.job.group = "qwen3p8_sft_example"
    config.job.name = "multimodal_multiturn_seed42"
    config.job.wandb_mode = "disabled"
    config.model_hf.name_or_path = os.environ.get("SFT_MODEL_PATH", "Qwen/Qwen3.8-27B")
    config.model_hf.attn_impl = "sdpa"
    config.model_hf.grad_ckpt = True
    config.model_hf.parallelism.dp_replicate = 1
    config.model_hf.parallelism.dp_shard = -1
    config.rl.seed = 42
    config.rl.master_dtype = "float32"
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.mixed_precision_mode = "autocast"
    config.rl.max_steps = 20
    config.rl.micro_batch_size = 1
    config.rl.grad_accum = 4
    config.rl.max_length = 4096
    config.rl.checkpoint_format = "dcp"
    config.rl.save_async = False
    config.rl.save_every = 10
    config.rl.optimizer = OptimizerConfig(lr=1e-5, weight_decay=0.01, grad_clip_norm=1.0, warmup_steps=2)
    data_root = os.environ.get("SFT_DATA_ROOT", ".ene/scratch/sft-example")
    config.rl.data = SFTDatasetConfig(path=os.path.join(data_root, "train.jsonl"))
    config.rl.validation = SFTValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=10,
        batch_size=1,
        seed=42,
        data=SFTDatasetConfig(path=os.path.join(data_root, "validation.jsonl"), shuffle=False),
    )
    config.rl.processor_kwargs = {
        "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
        "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 65536}, "num_frames": 8, "fps": None},
    }
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
