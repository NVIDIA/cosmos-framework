# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit existing GR00T finetune control, separate from matched QLoRA recipes."""

from dataclasses import dataclass

from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig

OFFICIAL_HEAD_PROFILE = "gr00t_n1d7_official_head_v1"
NATIVE_ADAPTER_OPTIMIZER_PROFILE = "native_fused_all_v1"


def validate_official_head_optimizer(config):
    if (
        config.name != "adamw"
        or config.impl != ""
        or config.decay_groups != "hf"
        or config.train_keys
        or config.lr_multipliers
    ):
        raise ValueError("official GR00T head control requires adamw_torch, HF decay groups and exact head membership")


@dataclass
class Gr00tOfficialHeadSchedule(SFTLRScheduleConfig):
    """HF cosine's zero endpoints; the generic SFT schedule remains unchanged."""

    kind: str = "cosine"
    f_start: float = 0.0
    f_min: float = 0.0
    profile: str = OFFICIAL_HEAD_PROFILE

    def __post_init__(self):
        if (
            self.profile != OFFICIAL_HEAD_PROFILE
            or self.kind != "cosine"
            or self.f_start != 0.0
            or self.f_min != 0.0
            or type(self.cycle_steps) is not int
            or self.cycle_steps < 0
        ):
            raise ValueError("official GR00T control requires the pinned HF cosine schedule with zero endpoints")
