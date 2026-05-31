#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# HF -> Diffusers
# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.convert_model_to_diffusers \
    -o $TMP_DIR/model \
    --checkpoint-path Cosmos3-Nano-GA


# Inference
# python -m cosmos_framework.scripts.inference \
#     -i "$INPUT_DIR/omni/t2i.json" \
#     -o $OUTPUT_DIR/inference \
#     --checkpoint-path $TMP_DIR/model/transformer \
#     $INFERENCE_ARGS
