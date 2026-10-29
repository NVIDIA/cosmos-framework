# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native dense Cosmos3-Nano-VLM adapters through the existing posttrain SFT loop.

Set SFT_MODEL_PATH to a complete immutable local snapshot, SFT_DATA_ROOT to
train.jsonl/validation.jsonl (prepare_sft_example generates a small mixed-media
fixture), and optionally SFT_ADAPTER_METHOD=lora|dora|pissa. Standard config
overrides remain available; Python recipes can give each component different
rank, alpha, dropout, method, initialization, LR multiplier and weight decay.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.config_sft import make_config as make_root_config
from cosmos_framework.posttrain.config.model import (
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTDatasetConfig, SFTValidationConfig


def make_config():
    config = make_root_config()
    config.job.project = "cosmos3_reasoner"
    config.job.group = "native_adapter_sft"
    config.job.name = "cosmos3_nano_adapters_seed42"
    config.job.wandb_mode = "disabled"
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder="cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy",
        model_id=os.environ.get("SFT_MODEL_PATH", "models/Cosmos3-Nano-VLM"),
        model_config=Qwen3VLModelConfig(dtype="bfloat16", attn_impl="sdpa", activation_checkpointing="full"),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
        requires_collated_policy_forward=True,
    )
    method = os.environ.get("SFT_ADAPTER_METHOD", "lora")
    common = dict(method=method, rank=8, alpha=16, use_rslora=os.environ.get("SFT_ADAPTER_RSLORA", "0") == "1")
    config.rl.adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language",
                targets=[
                    "model.language_model.layers.*.self_attn.q_proj",
                    "model.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            AdapterComponentConfig(
                component="projector",
                targets=[
                    "model.visual.merger.linear_fc1",
                    "model.visual.merger.linear_fc2",
                    "model.visual.deepstack_merger_list.*.linear_fc1",
                    "model.visual.deepstack_merger_list.*.linear_fc2",
                ],
                lr_multiplier=0.5,
                **common,
            ),
            AdapterComponentConfig(
                component="vision", targets=["model.visual.blocks.*.attn.qkv"], layers=[26], lr_multiplier=0.1, **common
            ),
        ]
    )
    config.rl.seed = 42
    config.rl.master_dtype = "bfloat16"  # frozen base; adapter storage is independently FP32
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.mixed_precision_mode = "fsdp"
    config.rl.max_steps = 6
    config.rl.micro_batch_size = 1
    config.rl.grad_accum = 1
    config.rl.max_length = 4096
    config.rl.checkpoint_format = "dcp"
    config.rl.save_async = False
    config.rl.save_every = 3
    config.rl.optimizer = OptimizerConfig(name="adamw", lr=2e-5, weight_decay=0, grad_clip_norm=1)
    data = os.environ.get("SFT_DATA_ROOT", ".ene/scratch/sft-adapters")
    config.rl.data = SFTDatasetConfig(path=os.path.join(data, "train.jsonl"))
    config.rl.validation = SFTValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=3,
        batch_size=1,
        data=SFTDatasetConfig(path=os.path.join(data, "validation.jsonl"), shuffle=False),
    )
    config.rl.processor_kwargs = {
        "images_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 81920}},
        "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 81920}, "num_frames": 4, "fps": None},
    }
    return config
