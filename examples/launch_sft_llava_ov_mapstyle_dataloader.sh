#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# MapDistributor-backed VLM SFT (llava_ov_mapstyle_dataloader) with
# per-worker (epoch, index) checkpoint/resume via CosmosDataLoaderStateCallback.
#
# Optional env vars:
#   RUN_NAME           — job name, also used to derive the checkpoint output path.
#                        Default: llava_ov_mapstyle_dataloader_<timestamp>
#   RESUME_FROM_CKPT   — if set, resumes from this checkpoint directory (supplies
#                        checkpoint.load_path and sets load_training_state=true).
#                        Expected format:
#                          <output_root>/<project>/<group>/<run_name>/checkpoints/iter_<N>
#
# Usage — fresh run:
#   bash examples/launch_sft_llava_ov_mapstyle_dataloader.sh
#
# Usage — resume:
#   RESUME_FROM_CKPT=<path_to_iter_N_ckpt> \
#   RUN_NAME=<same_name_as_fresh_run> \
#       bash examples/launch_sft_llava_ov_mapstyle_dataloader.sh

TOML_FILE="examples/toml/sft_config/llava_ov_mapstyle_dataloader.toml"
: "${RUN_NAME:=llava_ov_mapstyle_dataloader_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-50016}"

TAIL_OVERRIDES=(
    "data_setting.max_tokens=16000"
    "trainer.max_iter=60"
    "checkpoint.save_iter=50"
    "job.wandb_mode=online"
    "job.name=${RUN_NAME}"
)

if [[ -n "${RESUME_FROM_CKPT:-}" ]]; then
    echo ">>> Resuming from checkpoint: ${RESUME_FROM_CKPT}"
    TAIL_OVERRIDES+=(
        "checkpoint.load_path=${RESUME_FROM_CKPT}"
        "checkpoint.load_training_state=true"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
