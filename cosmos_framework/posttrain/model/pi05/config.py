# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The supported stateless RLinf OpenPI π0.5 LIBERO model contract."""

from dataclasses import dataclass

RLINF_REVISION = "ce4d821780d0e431d90f43d028fc93cac61f8fe0"
SAMPLER_IDENTITY = "pi05-openpi-selected-flow-sde-v1"
SAMPLER_NOISE_LEVEL = 0.5
PALIGEMMA_VOCAB_SIZE = 257152


@dataclass(frozen=True)
class Pi05ModelConfig:
    checkpoint_path: str
    tokenizer_path: str
    norm_stats_path: str
    action_chunk: int = 5
    denoise_steps: int = 3

    def __post_init__(self) -> None:
        for name in ("checkpoint_path", "tokenizer_path", "norm_stats_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"pi05.{name} must be a nonempty path")
        if type(self.action_chunk) is not int or not 1 <= self.action_chunk <= 10:
            raise ValueError("pi05.action_chunk must be an integer in [1, 10]")
        if type(self.denoise_steps) is not int or self.denoise_steps < 2:
            raise ValueError("pi05.denoise_steps must be an integer >= 2")
