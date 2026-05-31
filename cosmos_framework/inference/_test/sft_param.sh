# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

torchrun $TORCHRUN_ARGS -m cosmos_framework.scripts.train \
    -o $OUTPUT_DIR/train \
    --config-file $CONFIG_FILE \
    $TRAIN_ARGS \
    --config-overrides \
    "checkpoint.load_path=$BASE_CHECKPOINT_PATH" \
    $TRAIN_OVERRIDES
