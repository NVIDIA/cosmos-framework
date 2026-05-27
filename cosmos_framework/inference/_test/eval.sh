# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

torchrun $TORCHRUN_ARGS -m cosmos3.scripts.eval \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $BASE_CHECKPOINT_NAME \
    --dataset.config-file $CONFIG_FILE \
    --dataset.num-samples 1 \
    $INFERENCE_ARGS
