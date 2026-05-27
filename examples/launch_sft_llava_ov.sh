#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for llava_ov_datapacker (VLM SFT on
# lmms-lab/LLaVA-OneVision-Data via DataPackerDataLoader). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/llava_ov_datapacker.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/vlm/config.py as the base config.
#
# The dataset streams from the HuggingFace Hub, so DATASET_PATH /
# WAN_VAE_PATH / BASE_CHECKPOINT_PATH are NOT required; only HF_TOKEN may
# be needed for gated tokenizer downloads. Two model knobs that the
# SFTExperimentConfig dataclass does not model live in TAIL_OVERRIDES:
#
#   model.config.policy.backbone.model_name=<HF or local path>
#   data_setting.max_tokens=<int>
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_llava_ov.sh

TOML_FILE="examples/toml/sft_config/llava_ov_datapacker.toml"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
