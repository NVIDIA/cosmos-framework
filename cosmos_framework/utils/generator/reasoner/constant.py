# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# PyTorch CrossEntropyLoss treats targets equal to -100 as positions to
# exclude from the loss (its upstream default for ignore_index). Used
# wherever label tensors are assembled or CE losses are computed.
IGNORE_INDEX: int = -100

# Per-processor keys that downstream augmentors / collators pass through
# from the HuggingFace processor output into the model batch.
PROCESSOR_KEYS_TO_ADD_QWEN: list[str] = [
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
]
PROCESSOR_KEYS_TO_ADD_EAGLE: list[str] = ["input_ids", "attention_mask", "pixel_values", "image_sizes"]

PROCESSOR_KEYS_TO_ADD: list[str] = list(set(PROCESSOR_KEYS_TO_ADD_QWEN + PROCESSOR_KEYS_TO_ADD_EAGLE))

# Conversation roles whose content can carry images, videos and audio: user turns, and tool results (the
# Qwen3-VL chat template renders a tool message's media inside its <tool_response> block).
MEDIA_ROLES: tuple[str, ...] = ("user", "tool")
