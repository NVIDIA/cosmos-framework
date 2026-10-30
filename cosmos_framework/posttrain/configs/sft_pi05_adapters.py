# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""π0.5 LIBERO physical demonstrations through the shared SFT entrypoint."""

import os

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, NativePolicyParallelismConfig
from cosmos_framework.posttrain.configs.sft_cosmos3_nano_adapters import make_config as root_config
from cosmos_framework.posttrain.model.pi05.sft import BUILDER, Pi05SFTModelConfig


def make_config():
    config = root_config()
    config.job.project = "robotics_adapters"
    config.job.group = "pi05_libero"
    config.job.name = "pi05_adapters_seed42"
    config.model = NativePolicyBuildSpec(
        builder=BUILDER,
        model_id=os.environ["SFT_MODEL_PATH"],
        model_config=Pi05SFTModelConfig(os.environ["SFT_TOKENIZER_PATH"], os.environ["SFT_NORM_STATS_PATH"]),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
    )
    config.rl.task = "flow_bc"
    config.rl.flow_training_prefix = 10
    config.rl.flow_action_dim = 7
    config.rl.processor_kwargs = {}
    common = dict(
        rank=8,
        alpha=16,
        method=os.environ.get("SFT_ADAPTER_METHOD", "lora"),
        use_rslora=os.environ.get("SFT_ADAPTER_RSLORA", "0") == "1",
    )
    prefix = "paligemma_with_expert.paligemma.model"
    config.rl.adapters = AdapterConfig(
        components=[
            Component(
                component="language",
                targets=[
                    f"{prefix}.language_model.layers.*.self_attn.k_proj",
                    f"{prefix}.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            Component(
                component="projector", targets=[f"{prefix}.multi_modal_projector.linear"], lr_multiplier=0.5, **common
            ),
            Component(
                component="vision",
                targets=[
                    f"{prefix}.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
                    f"{prefix}.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
                ],
                layers=[26],
                lr_multiplier=0.1,
                **common,
            ),
            Component(
                component="action",
                targets=[
                    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.q_proj",
                    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.v_proj",
                    "action_in_proj",
                    "action_out_proj",
                ],
                **common,
            ),
        ]
    )
    return config
