#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Dataflow-loader mirror of the VFM vision_sft_nano recipe (vision_sft_nano_mapstyle_dataloader)
# for loss-curve regression vs the baseline launched by launch_sft_vision_nano.sh.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   RUN_NAME              default: vision_sft_nano_mapstyle_dataloader_<timestamp>

TOML_FILE="examples/toml/sft_config/vision_sft_nano_mapstyle_dataloader.toml"
: "${DATASET_PATH:=examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

: "${RUN_NAME:=vision_sft_nano_mapstyle_dataloader_$(date +%Y%m%d_%H%M%S)}"

TAIL_OVERRIDES=(
    "trainer.logging_iter=1" "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment" "job.wandb_mode=online" "job.name=${RUN_NAME}"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
