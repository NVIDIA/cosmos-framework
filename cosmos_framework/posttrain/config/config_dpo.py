# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Root config for the DPO entrypoint: `PosttrainConfig` with `rl: DPOConfig`.

The default `--config` for `entrypoints/dpo.py`. Registration + job/trainer defaults come from
`config.build_root`; this module only pins the typed `rl` section (and inherits `model_backend="i4"`,
the i4 OmniMoT path).
"""

from __future__ import annotations

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig, build_root
from cosmos_framework.posttrain.config.rl import DPOConfig


@attrs.define(slots=False)
class PosttrainDPOConfig(PosttrainConfig):
    algorithm: str = "dpo"
    rl: DPOConfig = attrs.field(factory=DPOConfig)


def make_config(*, register_experiment_configs: bool = True) -> PosttrainDPOConfig:
    """Build the root, optionally omitting experiment catalogs for a concrete recipe."""
    return build_root(PosttrainDPOConfig, "posttrain_dpo", register_experiment_configs=register_experiment_configs)
