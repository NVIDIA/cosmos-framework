# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-light configuration for native GR00T flow-BC backbone adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass

HEAD_ADAPTER_PROFILE = "gr00t_n1d7_dit_attention_lora_v1"


@dataclass
class Gr00tBackboneAdaptation:
    """Explicit Cosmos-Reason2-2B language adaptation, independent of action-head training.

    ``*_frozen`` quantizes or casts the feature extractor but does not train language adapters.
    ``*_lora`` propagates the original action loss into FP32 ordinary LoRA parameters.
    The action projector/DiT retain the checkpoint's native finetune selection when enabled.
    ``action_head_adapter_profile`` opts into ordinary DiT attention LoRA, with the same
    rank/alpha/dropout as the language adapters, and requires ``train_action_head=False``:
    only original head parameters are frozen. Native tune flags, training modes and dropout
    remain those of full-head tuning.
    """

    backbone_path: str = ""
    mode: str = "nf4_lora"
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    seed: int = 17
    train_action_head: bool = True
    action_head_adapter_profile: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"nf4_lora", "bf16_lora", "nf4_frozen", "bf16_frozen"}:
            raise ValueError("GR00T backbone mode must be nf4_lora, bf16_lora, nf4_frozen or bf16_frozen")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("GR00T adapter rank must be a positive integer")
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("GR00T adapter alpha must be finite and positive")
        if isinstance(self.dropout, bool) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("GR00T adapter dropout must lie in [0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("GR00T adapter seed must be an unsigned 32-bit integer")
        if type(self.train_action_head) is not bool:
            raise TypeError("GR00T train_action_head must be a boolean")
        if self.mode.endswith("_frozen") and not self.train_action_head:
            raise ValueError("GR00T frozen-backbone training requires a trainable action head")
        if self.action_head_adapter_profile is not None:
            if self.action_head_adapter_profile != HEAD_ADAPTER_PROFILE:
                raise ValueError("GR00T action-head adaptation requires its explicit DiT attention profile")
            if self.train_action_head or self.mode not in {"nf4_lora", "bf16_lora"}:
                raise ValueError("GR00T head adapters require language LoRA and frozen original head parameters")
