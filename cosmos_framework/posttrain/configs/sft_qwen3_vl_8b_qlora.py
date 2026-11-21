# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NF4 arm of the matched native Qwen3-VL-8B adapter recipes.

Requires the posttrain environment's sft, cu128-runtime and qlora groups, a supported
CUDA device, and torchrun --nproc-per-node=1. Capability validation refuses fallback
to a dense base. Real-kernel and quality qualification remain release requirements.
"""

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_lora import build_config


def make_config() -> PosttrainSFTConfig:
    return build_config(quantized=True)
