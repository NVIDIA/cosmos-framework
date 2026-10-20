#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -u

: "${COSMOS_VIDEO_TRAIN_ANNOTATION:?Set COSMOS_VIDEO_TRAIN_ANNOTATION to the training JSON file}"
: "${COSMOS_VIDEO_TRAIN_MEDIA:?Set COSMOS_VIDEO_TRAIN_MEDIA to the training video root}"
: "${COSMOS_VIDEO_VAL_ANNOTATION:?Set COSMOS_VIDEO_VAL_ANNOTATION to the validation JSON file}"
: "${COSMOS_VIDEO_VAL_MEDIA:?Set COSMOS_VIDEO_VAL_MEDIA to the validation video root}"
: "${VLM_SAFETENSORS_PATH:?Set VLM_SAFETENSORS_PATH to converted Cosmos3-Nano safetensors}"

export COSMOS_VIDEO_TRAIN_ANNOTATION COSMOS_VIDEO_TRAIN_MEDIA
export COSMOS_VIDEO_VAL_ANNOTATION COSMOS_VIDEO_VAL_MEDIA
export VLM_SAFETENSORS_PATH
export COSMOS_VIDEO_TRAIN_LIMIT="${COSMOS_VIDEO_TRAIN_LIMIT:-}"
export COSMOS_VIDEO_VAL_LIMIT="${COSMOS_VIDEO_VAL_LIMIT:-}"

TOML_FILE="examples/toml/sft_config/video_sft_nano.toml"
EXTRA_DATASET_CHECK='
[[ -f "$COSMOS_VIDEO_TRAIN_ANNOTATION" ]] || { echo "ERROR: training annotations not found: $COSMOS_VIDEO_TRAIN_ANNOTATION" >&2; exit 1; }
[[ -d "$COSMOS_VIDEO_TRAIN_MEDIA" ]] || { echo "ERROR: training media root not found: $COSMOS_VIDEO_TRAIN_MEDIA" >&2; exit 1; }
[[ -f "$COSMOS_VIDEO_VAL_ANNOTATION" ]] || { echo "ERROR: validation annotations not found: $COSMOS_VIDEO_VAL_ANNOTATION" >&2; exit 1; }
[[ -d "$COSMOS_VIDEO_VAL_MEDIA" ]] || { echo "ERROR: validation media root not found: $COSMOS_VIDEO_VAL_MEDIA" >&2; exit 1; }
[[ -d "$VLM_SAFETENSORS_PATH" ]] || { echo "ERROR: converted checkpoint not found: $VLM_SAFETENSORS_PATH" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
