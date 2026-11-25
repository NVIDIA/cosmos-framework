# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Functional Qwen3.8-27B adapter SFT example through the HF backend, not a benchmark recipe.

The data and schedule of ``sft_qwen3p8_27b_multimodal.py`` with the base frozen in bfloat16
(``rl.master_dtype == rl.param_dtype``) and float32 adapters on ``mlp.up_proj``/``mlp.down_proj`` of every language
layer and on the vision merger's ``linear_fc1``/``linear_fc2`` (half the learning rate).

``SFT_ADAPTER_METHOD`` selects lora (default), rslora, dora, rsdora, pissa or rspissa; ``SFT_MODEL_PATH`` and
``SFT_DATA_ROOT`` are as in the full-parameter recipe. ``$SFT_PYTHON`` is the posttrain SFT environment's interpreter
(``docs/posttrain/guides/sft.md``, section 2). Launch::

    SFT_ADAPTER_METHOD=pissa "$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$N" \
        -m cosmos_framework.posttrain.entrypoints.sft \
        --config cosmos_framework/posttrain/configs/sft_qwen3p8_27b_adapters.py

FSDP2 by default; ``-- model_hf.parallelism.dp_replicate=<R>`` selects HSDP. ``-- rl.adapter_warm_start=<dir>``
starts from a portable adapter. Each rank first loads the full bfloat16 base (about 54 GB of weights in device
memory) before FSDP shards it, so use 80 GB devices; a base that does not fit in 90% of one device is refused.
Checkpoints include the frozen base weights (about 54 GB each). Export::

    "$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.export_adapters --checkpoint <iter_dir> --out <adapter>
    "$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.export_adapters --adapter <adapter> --merged \
        --base "$SFT_MODEL_PATH" --out <merged_model> [--dtype float32]
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.configs.sft_qwen3p8_27b_multimodal import build_config as build_full_config
from cosmos_framework.utils.generator.adapter_config import (
    AdapterComponentConfig,
    AdapterConfig,
    adapter_method_settings,
)


def build_config(method: str | None = None) -> PosttrainSFTConfig:
    method = method or os.environ.get("SFT_ADAPTER_METHOD", "lora")
    settings = adapter_method_settings(method)
    config = build_full_config()
    config.job.group = "qwen3p8_sft_adapters_example"
    config.job.name = f"{method}_seed42"
    config.rl.master_dtype = "bfloat16"
    config.rl.param_dtype = "bfloat16"
    config.rl.mixed_precision_mode = "fsdp"
    config.rl.adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language",
                targets=["model.language_model.layers.*.mlp.up_proj", "model.language_model.layers.*.mlp.down_proj"],
                rank=16,
                alpha=32.0,
                **settings,
            ),
            AdapterComponentConfig(
                component="projector",
                targets=["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"],
                rank=16,
                alpha=32.0,
                lr_multiplier=0.5,
                **settings,
            ),
        ]
    )
    config.rl.optimizer = OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.0, grad_clip_norm=1.0, warmup_steps=2)
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
