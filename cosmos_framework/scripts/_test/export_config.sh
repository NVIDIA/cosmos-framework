#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.export_config \
    -o $OUTPUT_DIR/config.yaml \
    --experiment cosmos3_ga_16bm8b_v1_midtrain
