#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_edge (T2V / I2V / V2V vision-only
# SFT on Nemotron-2B-Dense-VL / Cosmos3-Edge, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_edge.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              not needed for nvidia/Cosmos3-Edge (the repo is
#                         ungated); set only if another download requires it
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_edge.sh

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
