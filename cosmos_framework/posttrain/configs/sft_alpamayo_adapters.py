# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public Alpamayo-R1 physical trajectory demonstrations through native Cosmos SFT."""

import os

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, NativePolicyParallelismConfig
from cosmos_framework.posttrain.configs.sft_cosmos3_nano_adapters import make_config as root_config
from cosmos_framework.posttrain.model.alpamayo.sft import BUILDER, AlpamayoSFTModelConfig


def make_config():
    config = root_config()
    config.job.project = "driving_adapters"
    config.job.group = "alpamayo_r1_trajectory"
    config.job.name = "alpamayo_adapters_seed42"
    config.model = NativePolicyBuildSpec(
        builder=BUILDER,
        model_id=os.environ["SFT_MODEL_PATH"],
        model_config=AlpamayoSFTModelConfig(os.environ["SFT_PROCESSOR_PATH"]),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
    )
    config.rl.task = "flow_bc"
    config.rl.flow_training_prefix = 64
    config.rl.flow_action_dim = 2
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
                    "vlm.model.language_model.layers.*.self_attn.k_proj",
                    "vlm.model.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            Component(
                component="projector",
                targets=[
                    "vlm.model.visual.merger.linear_fc1",
                    "vlm.model.visual.merger.linear_fc2",
                    "vlm.model.visual.deepstack_merger_list.*.linear_fc1",
                    "vlm.model.visual.deepstack_merger_list.*.linear_fc2",
                ],
                lr_multiplier=0.5,
                **common,
            ),
            Component(
                component="vision",
                targets=["vlm.model.visual.blocks.*.attn.qkv"],
                layers=[26],
                lr_multiplier=0.1,
                **common,
            ),
            Component(
                component="action",
                targets=[
                    "expert.layers.*.self_attn.q_proj",
                    "expert.layers.*.self_attn.v_proj",
                    "action_in_proj.encoder.trunk.0",
                    "action_in_proj.encoder.trunk.3",
                    "action_in_proj.encoder.trunk.6",
                ],
                **common,
            ),
            # The native output has only two channels; rank eight cannot fit it.
            Component(component="action", targets=["action_out_proj"], **{**common, "rank": 2, "alpha": 4}),
        ]
    )
    return config
