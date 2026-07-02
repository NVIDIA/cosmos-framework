# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

IGNORE_INDEX = -100

PROCESSOR_KEYS_TO_ADD_QWEN = [
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
]
PROCESSOR_KEYS_TO_ADD_EAGLE = ["input_ids", "attention_mask", "pixel_values", "image_sizes"]

PROCESSOR_KEYS_TO_ADD = list(set(PROCESSOR_KEYS_TO_ADD_QWEN + PROCESSOR_KEYS_TO_ADD_EAGLE))
