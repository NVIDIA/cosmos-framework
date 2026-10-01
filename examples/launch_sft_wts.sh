#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -u

: "${WTS_TRAIN_ANNOTATION:?Set WTS_TRAIN_ANNOTATION to the training JSON file}"
: "${WTS_TRAIN_MEDIA:?Set WTS_TRAIN_MEDIA to the training video root}"
: "${WTS_VAL_ANNOTATION:?Set WTS_VAL_ANNOTATION to the validation JSON file}"
: "${WTS_VAL_MEDIA:?Set WTS_VAL_MEDIA to the validation video root}"
: "${VLM_SAFETENSORS_PATH:?Set VLM_SAFETENSORS_PATH to converted Cosmos3-Nano safetensors}"

export WTS_TRAIN_ANNOTATION WTS_TRAIN_MEDIA
export WTS_VAL_ANNOTATION WTS_VAL_MEDIA
export VLM_SAFETENSORS_PATH
export WTS_TRAIN_LIMIT="${WTS_TRAIN_LIMIT:-}"
export WTS_VAL_LIMIT="${WTS_VAL_LIMIT:-}"

TOML_FILE="examples/toml/sft_config/wts_vlm.toml"
EXTRA_DATASET_CHECK='
[[ -f "$WTS_TRAIN_ANNOTATION" ]] || { echo "ERROR: training annotations not found: $WTS_TRAIN_ANNOTATION" >&2; exit 1; }
[[ -d "$WTS_TRAIN_MEDIA" ]] || { echo "ERROR: training media root not found: $WTS_TRAIN_MEDIA" >&2; exit 1; }
[[ -f "$WTS_VAL_ANNOTATION" ]] || { echo "ERROR: validation annotations not found: $WTS_VAL_ANNOTATION" >&2; exit 1; }
[[ -d "$WTS_VAL_MEDIA" ]] || { echo "ERROR: validation media root not found: $WTS_VAL_MEDIA" >&2; exit 1; }
[[ -d "$VLM_SAFETENSORS_PATH" ]] || { echo "ERROR: converted checkpoint not found: $VLM_SAFETENSORS_PATH" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
