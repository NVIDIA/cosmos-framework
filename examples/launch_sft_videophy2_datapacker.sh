#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_sft_nano_v2 (VLM dialog SFT on VideoPhy-2
# via CosmosDataLoader + four-role dataflow). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/videophy2_sft_nano_v2.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/vlm/config.py as the base config.
#
# Required env:
#   VIDEOPHYSICS_ROOT  dir containing videophy2_train/ and videophy2_val/
#                      (each with meta.json + media/ + text/). Populate via
#                      `python -m cosmos_framework.scripts.vlm.prepare_videophy2_from_hf`.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL-8B-Instruct downloads.
#   VLM_SAFETENSORS_PATH   local directory of pre-converted Qwen3-VL safetensors.
#                          When set, plumbed to backbone.safetensors_path via a
#                          tail override.
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_datapacker.sh

TOML_FILE="examples/toml/sft_config/videophy2_sft_nano_v2.toml"

: "${RUN_NAME:=videophy2_datapacker_v2_$(date +%Y%m%d_%H%M%S)}"

TAIL_OVERRIDES=(
    "data_setting.max_tokens=16000"
    "trainer.logging_iter=1" "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment" "job.wandb_mode=online" "job.name=${RUN_NAME}"
    ${EXTRA_TAIL_OVERRIDES:-}
)

# When VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path so the
# framework loads weights from the local snapshot while keeping the public HF
# model_name for tokenizer/architecture discovery.
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
