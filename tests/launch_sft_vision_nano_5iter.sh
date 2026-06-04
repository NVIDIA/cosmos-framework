#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# SMOKE wrapper (test fixture) for tests/nano_training_smoke_test.py — mirrors
# examples/launch_sft_vision_nano.sh but points at tests/vision_sft_nano_5iter.toml
# (max_iter=5, save_iter=5). Reuses the shared launcher helper from examples/.
# Paths below are resolved relative to the repo root by _sft_launcher_common.sh.

TOML_FILE="tests/vision_sft_nano_5iter.toml"
: "${DATASET_PATH:=examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/../examples/_sft_launcher_common.sh"
