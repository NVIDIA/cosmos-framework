# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

torchrun $TORCHRUN_ARGS -m cosmos3.scripts.inference \
    -i "$INPUT_DIR/omni/*.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path Cosmos3-Super \
    --parallelism-preset=latency \
    $INFERENCE_ARGS
