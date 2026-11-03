# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Root config for the OPD entrypoint: GRPO's root with ``rl: OPDConfig``.

OPD is GRPO plus a teacher, so it inherits the same model and rollout sections. This typed root
changes the algorithm identity and concrete RL schema while preserving OPD's established HF
default; native OPD recipes select ``model_backend="i4"`` explicitly.

The teacher lives at `rl.teacher`, NOT as a top-level section beside `rollout`. That asymmetry is
deliberate: `config.rollout` is top-level because the student's engine is shared machinery — the
weight-sync plane, the checkpoint/resume story and the sampling knobs all reach it, and a GRPO run
has one whether or not it distils. The teacher exists only because the ALGORITHM asks for one, so it
belongs to the algorithm's config, next to `jsd_beta` and `kl_penalty_coef`.
"""

from __future__ import annotations

import attrs

from cosmos_framework.posttrain.config.config import build_root
from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.rl import OPDConfig


@attrs.define(slots=False)
class PosttrainOPDConfig(PosttrainGRPOConfig):
    algorithm: str = "opd"
    # Preserve the established text-only OPD demo/default. Native OPD recipes select the native backend explicitly.
    model_backend: str = "hf"
    rl: OPDConfig = attrs.field(factory=OPDConfig)


def make_config(*, register_experiment_configs: bool = True) -> PosttrainOPDConfig:
    """Build the root, optionally omitting experiment catalogs for a concrete recipe."""
    return build_root(PosttrainOPDConfig, "posttrain_opd", register_experiment_configs=register_experiment_configs)
