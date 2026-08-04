#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_lora_super (LoRA SFT on VideoPhy-2 via
# CosmosDataLoader, Cosmos3-Super tier / Qwen3-VL-32B). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_lora_super.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/reasoner/config.py as the base config.
#
# Freezing the 32B backbone and training only rank-16 adapters is what makes this
# tier comfortable on a 4-GPU allocation: optimizer state is adapter-sized.
#
# Required env:
#   VIDEOPHYSICS_ROOT      dir containing videophy2_train/ and videophy2_val/
#                          (each with meta.json + media/ + text/). Populate via
#                          `python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf`.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL-32B-Instruct downloads.
#   VLM_SAFETENSORS_PATH   local directory of pre-converted Qwen3-VL-32B safetensors.
#                          When unset the framework downloads the public
#                          Qwen/Qwen3-VL-32B-Instruct snapshot (~64 GB on first run).
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4 node.
#   WANDB_API_KEY          the TOML sets wandb_mode="online"; export a key or set
#                          EXTRA_TAIL_OVERRIDES='job.wandb_mode=offline'.
#
# Usage (from the repo root, inside the training container):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_super.sh
#   # on a 4-GPU node (e.g. GB200x4):
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_super.sh

TOML_FILE="examples/toml/sft_config/videophy2_lora_super.toml"

# Super-variant allocator tweak: expandable_segments so the 32B backbone fits
# without OOM. (Unlike launch_sft_vision_super.sh we do NOT clear
# LD_LIBRARY_PATH — this reasoner recipe decodes VideoPhy-2 clips with torchcodec,
# which dlopen()s the CUDA NPP + FFmpeg libs off LD_LIBRARY_PATH.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
