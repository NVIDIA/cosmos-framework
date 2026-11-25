# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-device SFT; ordinary FP32 trainables and BF16 autocast, no weight quantization.

SFT_MODEL_PATH and SFT_DATA_ROOT identify immutable local artifacts. Override
rl.optimizer.name=adamw rl.optimizer.impl=fused
rl.optimizer.bnb_min_8bit_size=4096 for the high-precision control,
or rl.optimizer.name=paged_adamw_8bit for the capacity mode. Train the eight selected
decoder blocks (about 1.54B parameters), with frozen ordinary base weights. The
measured A100 80GB workload uses microbatch four and eight 256x480 video frames.
This recipe is not a speed or quality-gain claim. Enable rl.save_every explicitly
when durable checkpoints are needed; consolidated saves include the dense model.
"""

import dataclasses
import os

from cosmos_framework.posttrain.configs.sft_qwen3p8_27b_multimodal import build_config


def make_config():
    config = build_config()
    config.model_hf.name_or_path = os.environ.get("SFT_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    config.model_hf.parallelism.dp_shard = 1
    config.rl.checkpoint_format = "consolidated"
    # Rebuild (not mutate) the already-constructed OptimizerConfig so its validation runs.
    config.rl.optimizer = dataclasses.replace(
        config.rl.optimizer,
        name="adamw_bnb_8bit",
        train_keys=[f"layers.{index}." for index in range(24, 32)],
        bnb_min_8bit_size=16384,
    )
    config.rl.micro_batch_size = 4
    config.rl.grad_accum = 1
    config.rl.max_steps = 40
    config.rl.save_every = 0
    config.rl.validation.every_steps = 5
    config.rl.validation.max_samples = 16
    config.rl.validation.batch_size = 4
    config.rl.processor_kwargs["videos_kwargs"]["size"] = {"shortest_edge": 262144, "longest_edge": 1048576}
    config.job.group = "quantized_optimizer"
    return config
