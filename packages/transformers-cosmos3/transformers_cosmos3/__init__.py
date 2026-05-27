# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""transformers shim: load Cosmos3 checkpoints into Qwen3-VL understanding tower."""

from transformers_cosmos3.model import Cosmos3ForConditionalGeneration

__all__ = ["Cosmos3ForConditionalGeneration"]
