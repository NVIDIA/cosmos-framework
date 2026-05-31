# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""transformers shim: load Cosmos3 checkpoints into Qwen3-VL understanding tower."""

from transformers import AutoConfig

from transformers_cosmos3.config import Cosmos3OmniConfig
from transformers_cosmos3.model import Cosmos3ForConditionalGeneration

AutoConfig.register("cosmos3_omni", Cosmos3OmniConfig, exist_ok=True)

__all__ = ["Cosmos3ForConditionalGeneration", "Cosmos3OmniConfig"]
