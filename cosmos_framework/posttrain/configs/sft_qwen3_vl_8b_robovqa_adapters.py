# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen3-VL-8B-Instruct adapter SFT on RoboVQA through the native policy builder.

The data, schedule and precision of ``sft_qwen3_vl_8b_robovqa.py`` with the base frozen in bfloat16 and float32
adapters on ``q_proj``/``k_proj``/``v_proj``/``o_proj`` of every language layer (rank 16), the ``linear_fc1``/
``linear_fc2`` layers of the vision merger and deepstack mergers (rank 16, half the learning rate) and the fused
``qkv`` of the last vision block (rank 8, a tenth of the learning rate).

``SFT_ADAPTER_METHOD`` selects lora (default), rslora, dora, rsdora, pissa or rspissa; ``SFT_MODEL_PATH`` and
``SFT_DATA_ROOT`` are as in the full-parameter recipe. ``$SFT_PYTHON`` is the posttrain SFT environment's interpreter
(``docs/posttrain/guides/sft.md``, section 2). Launch::

    SFT_ADAPTER_METHOD=rsdora "$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$N" \
        -m cosmos_framework.posttrain.entrypoints.sft \
        --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa_adapters.py

FSDP2 by default; ``-- model.parallelism.dp_replicate=<R>`` selects HSDP. ``-- rl.adapter_warm_start=<dir>`` starts
from a portable adapter. Each rank first loads the full bfloat16 base (about 17 GB of device memory) before FSDP
shards it; a base that does not fit in 90% of one device is refused. Checkpoints are complete DCP saves that include
the frozen base (about 17 GB each). Export::

    "$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.export_adapters --checkpoint <iter_dir> --out <adapter>
    "$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.export_adapters --adapter <adapter> --merged \
        --base "$SFT_MODEL_PATH" --out <merged_model> [--dtype float32]
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_robovqa import EPOCH_STEPS
from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_robovqa import build_config as build_full_config
from cosmos_framework.utils.generator.adapter_config import (
    AdapterComponentConfig,
    AdapterConfig,
    adapter_method_settings,
)


def build_config(method: str | None = None) -> PosttrainSFTConfig:
    method = method or os.environ.get("SFT_ADAPTER_METHOD", "lora")
    settings = adapter_method_settings(method)
    config = build_full_config()
    config.job.group = "qwen3_vl_8b_robovqa_adapters"
    config.job.name = f"{method}_seed0"
    config.rl.adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language",
                targets=[f"model.language_model.layers.*.self_attn.{p}_proj" for p in ("q", "k", "v", "o")],
                rank=16,
                alpha=32.0,
                **settings,
            ),
            AdapterComponentConfig(
                component="projector",
                targets=[
                    "model.visual.merger.linear_fc1",
                    "model.visual.merger.linear_fc2",
                    "model.visual.deepstack_merger_list.*.linear_fc1",
                    "model.visual.deepstack_merger_list.*.linear_fc2",
                ],
                rank=16,
                alpha=32.0,
                lr_multiplier=0.5,
                **settings,
            ),
            AdapterComponentConfig(
                component="vision",
                targets=["model.visual.blocks.*.attn.qkv"],
                layers=[-1],
                rank=8,
                alpha=16.0,
                lr_multiplier=0.1,
                **settings,
            ),
        ]
    )
    config.rl.optimizer = OptimizerConfig(
        name="adamw", lr=1e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0, warmup_steps=0
    )
    config.rl.checkpoint_format = "dcp"
    config.rl.checkpoint_components = ["model", "optimizer", "training_state"]
    config.rl.save_every = EPOCH_STEPS
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
