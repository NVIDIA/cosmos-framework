# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

torchrun $TORCHRUN_ARGS -m cosmos_framework.scripts.inference \
    -i "$INPUT_DIR/reasoner/*.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path Cosmos3-Nano \
    --parallelism-preset=throughput \
    $INFERENCE_ARGS
