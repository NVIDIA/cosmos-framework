# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T N1.7 full-prefix/action adapters using native LIBERO supervision."""

import os

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, NativePolicyParallelismConfig
from cosmos_framework.posttrain.configs.sft_cosmos3_nano_adapters import make_config as root_config
from cosmos_framework.posttrain.model.gr00t.sft import BUILDER, Gr00tSFTModelConfig


def make_config():
    config = root_config()
    config.job.project, config.job.group, config.job.name = "robot_adapters", "gr00t_n1d7", "gr00t_adapters_seed42"
    config.model = NativePolicyBuildSpec(
        builder=BUILDER,
        model_id=os.environ["SFT_MODEL_PATH"],
        model_config=Gr00tSFTModelConfig(os.environ["SFT_BACKBONE_PATH"]),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
    )
    config.rl.task, config.rl.flow_training_prefix, config.rl.flow_action_dim = "flow_bc", 10, 7
    config.rl.processor_kwargs = {}
    config.rl.mixed_precision_mode = "autocast"
    common = dict(
        method=os.environ.get("SFT_ADAPTER_METHOD", "lora"),
        use_rslora=os.environ.get("SFT_ADAPTER_RSLORA", "0") == "1",
        rank=8,
        alpha=16,
    )
    config.rl.adapters = AdapterConfig(
        components=[
            Component(
                component="language",
                targets=[
                    "backbone.model.model.language_model.layers.*.self_attn.q_proj",
                    "backbone.model.model.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            Component(
                component="projector",
                targets=[
                    "backbone.model.model.visual.merger.linear_fc1",
                    "backbone.model.model.visual.merger.linear_fc2",
                    "backbone.model.model.visual.deepstack_merger_list.*.linear_fc1",
                    "backbone.model.model.visual.deepstack_merger_list.*.linear_fc2",
                ],
                lr_multiplier=0.5,
                **common,
            ),
            Component(
                component="vision",
                targets=["backbone.model.model.visual.blocks.*.attn.qkv"],
                layers=[26],
                lr_multiplier=0.1,
                **common,
            ),
            Component(
                component="action",
                targets=[
                    "action_head.model.transformer_blocks.*.attn1.to_q",
                    "action_head.model.transformer_blocks.*.attn1.to_v",
                ],
                **common,
            ),
        ]
    )
    return config
