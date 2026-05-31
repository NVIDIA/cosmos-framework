# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

COSMOS_TRAINING=1 torchrun $TORCHRUN_ARGS -m cosmos_framework.scripts.inference \
    -i "$INPUT_DIR/omni/t2v.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $CHECKPOINT_PATH \
    --experiment $EXPERIMENT \
    --no-use-ema-weights \
    $INFERENCE_ARGS
