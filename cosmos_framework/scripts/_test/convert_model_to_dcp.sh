#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# HF -> DCP
# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o $TMP_DIR/checkpoint \
    --checkpoint-path Cosmos3-Nano

# DCP -> HF
# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.export_model \
    -o $TMP_DIR/model \
    --checkpoint-path $TMP_DIR/checkpoint/model \
    --config-file $TMP_DIR/checkpoint/model/config.json \
    --no-use-ema-weights

# HF Inference
torchrun $TORCHRUN_ARGS -m cosmos_framework.scripts.inference \
    -i "$INPUT_DIR/omni/t2i.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $TMP_DIR/model \
    $INFERENCE_ARGS
