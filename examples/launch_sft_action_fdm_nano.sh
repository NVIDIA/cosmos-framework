#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_fdm_sft_nano (Bridge LeRobot FDM SFT,
# 8-GPU FSDP). Drives cosmos_framework.scripts.train against the pydantic schema
# at examples/toml/sft_config/action_fdm_sft_nano.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/bridge_lerobot_v3
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_action_fdm_nano.sh

TOML_FILE="examples/toml/sft_config/action_fdm_sft_nano.toml"
: "${DATASET_PATH:=examples/data/bridge_lerobot_v3}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
