# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# HF -> DCP
# Use temporary directory, since output is large.
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path $BASE_CHECKPOINT_NAME \
    -o $TMP_DIR/checkpoint_base

# Train
torchrun $TORCHRUN_ARGS -m cosmos_framework.scripts.train \
    -o $OUTPUT_DIR/train \
    --config-file $CONFIG_FILE \
    $TRAIN_ARGS \
    --config-overrides \
    "checkpoint.load_path=$TMP_DIR/checkpoint_base" \
    $TRAIN_OVERRIDES

CHECKPOINT_ITER=$(cat $OUTPUT_DIR/train/$JOB_NAME/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$OUTPUT_DIR/train/$JOB_NAME/job/checkpoints/$CHECKPOINT_ITER

# DCP -> HF
# Use temporary directory, since output is large.
python -m cosmos_framework.scripts.export_model \
    -o $TMP_DIR/model \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file $OUTPUT_DIR/train/$JOB_NAME/config.yaml

# Exported model inference is already tested in 'cosmos_framework/scripts/_test/convert_model_to_dcp.sh'
