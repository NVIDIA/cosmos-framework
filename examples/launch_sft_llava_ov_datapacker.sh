#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Dataflow-loader mirror of the VLM llava_ov recipe (pre_exp012_llava_ov_datapacker_v2)
# for loss-curve regression vs the baseline launched by launch_sft_llava_ov.sh.
TOML_FILE="examples/toml/sft_config/llava_ov_datapacker_v2.toml"
: "${RUN_NAME:=llava_ov_datapacker_v2_$(date +%Y%m%d_%H%M%S)}"
TAIL_OVERRIDES=(
    "data_setting.max_tokens=16000"
    "trainer.logging_iter=1"
    "trainer.max_iter=500"
    "job.project=cosmos_oss_alignment"
    "job.wandb_mode=online"
    "job.name=${RUN_NAME}"
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
