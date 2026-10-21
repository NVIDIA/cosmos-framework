# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Obj1k v2 text-to-Blender SFT with thinking traces: 960/40 split, three epochs at eight DP ranks.

Publish the splits externally using docs/howto/sft-obj1k-data.md. SFT_DATA_ROOT defaults
to the versioned GCS S3 prefix; SFT_S3_CREDENTIALS selects the mounted credential file for
data and checkpoint uploads. Set SFT_MODEL_PATH to a complete immutable local HF snapshot. The paired
sft_qwen3p8_27b_obj1k_gcp.json launch profile selects two four-GPU nodes.
This is an initial experiment, not a GPU-qualified memory or throughput claim.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.configs.sft_qwen3p8_27b_multimodal import build_config as build_base_config


def build_config() -> PosttrainSFTConfig:
    config = build_base_config()
    config.job.wandb_mode = "disabled"
    config.job.group = "qwen3p8_sft_obj1k"
    config.job.name = "qwen3d827b_sft_obj1k_v2data_8gpu_3epoch_v001"
    data_root = os.environ.get(
        "SFT_DATA_ROOT",
        "s3://<dataset-bucket>/raw/3dgen/obj1k/v2-seed42",
    )
    credentials = os.environ.get("SFT_S3_CREDENTIALS", "credentials/<credential>.secret")
    config.rl.data.path = os.path.join(data_root, "train.jsonl")
    config.rl.data.credentials = credentials
    config.rl.validation.data.path = os.path.join(data_root, "validation.jsonl")
    config.rl.validation.data.credentials = credentials
    # V1's longest templated conversation was 8,792 tokens; v2 adds reasoning.
    # Validate every v2 conversation against this budget before GPU training.
    config.rl.max_length = 12288
    config.rl.processor_kwargs = {}
    # 960 / (8 ranks * 1 conversation * 4 accumulation steps) = 30 updates/epoch.
    config.rl.max_steps = 90
    config.rl.optimizer.warmup_steps = 5
    config.rl.validation.every_steps = 30
    config.rl.save_every = 30
    # The HF materializer consumes model.pt, not DCP. Budget rank-0 host memory
    # for the immutable model snapshot while async writing/upload overlaps training.
    # Model gathering and sharded optimizer writes remain synchronous.
    config.rl.checkpoint_format = "consolidated"
    config.rl.save_async = True
    config.rl.upload.enabled = True
    config.rl.upload.bucket = "<checkpoint-experiments-bucket>"
    config.rl.upload.credentials = credentials
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
