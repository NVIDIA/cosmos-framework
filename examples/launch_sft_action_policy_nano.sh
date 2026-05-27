#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_sft_nano (LIBERO action-policy SFT,
# 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_sft_nano.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/LIBERO_LeRobot_v3
#                         (must contain libero_10/, libero_object/,
#                         libero_spatial/, libero_goal/ subdirs)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_action_policy_nano.sh

TOML_FILE="examples/toml/sft_config/action_policy_sft_nano.toml"
: "${DATASET_PATH:=examples/data/LIBERO_LeRobot_v3}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='for suite in libero_10 libero_object libero_spatial libero_goal; do
    [[ -d "$DATASET_PATH/$suite" ]] || { echo "ERROR: LIBERO suite missing: $DATASET_PATH/$suite" >&2; exit 1; }
done'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
