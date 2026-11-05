# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed HF-only root: SFT does not depend on generator model/callback catalogs."""

from __future__ import annotations

from typing import Any

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig
from cosmos_framework.posttrain.config.sft import SFTConfig


@attrs.define(slots=False)
class PosttrainSFTConfig(PosttrainConfig):
    algorithm: str = "sft"
    model_backend: str = "hf"
    rl: SFTConfig = attrs.field(factory=SFTConfig)
    defaults: list[Any] = attrs.field(factory=lambda: ["_self_"])


def make_config() -> PosttrainSFTConfig:
    # build_root registers diffusion callbacks/tokenizers (and their CUDA dependencies).
    # An HF-only recipe supplies every runtime section and needs none of those groups.
    config = PosttrainSFTConfig(model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None)
    config.job.group = "posttrain_sft"
    config.trainer.callbacks = None
    config.trainer.run_validation = False
    config.upload_reproducible_setup = False
    return config
