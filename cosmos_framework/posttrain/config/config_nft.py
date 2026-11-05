# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Root config for the colocated Cosmos3 online DiffusionNFT entrypoint."""

from __future__ import annotations

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig, build_root
from cosmos_framework.posttrain.config.rl import NFTConfig


@attrs.define(slots=False)
class PosttrainNFTConfig(PosttrainConfig):
    algorithm: str = "nft"
    rl: NFTConfig = attrs.field(factory=NFTConfig)


def make_config(*, register_experiment_configs: bool = True) -> PosttrainNFTConfig:
    """Build the root, optionally omitting experiment catalogs for a concrete recipe."""
    return build_root(PosttrainNFTConfig, "posttrain_nft", register_experiment_configs=register_experiment_configs)
