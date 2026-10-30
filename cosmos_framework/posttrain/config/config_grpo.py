# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Root config for the GRPO entrypoint: `PosttrainConfig` with `rl: GRPOConfig`.

The default `--config` for `entrypoints/grpo.py`. Registration + job/trainer defaults come from
`config.build_root`; this module pins the typed `rl` section and keeps the generic native backend as the
default. HF recipes must select ``model_backend="hf"`` explicitly.
"""

from __future__ import annotations

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig, build_root
from cosmos_framework.posttrain.config.model import DataPackerConfig
from cosmos_framework.posttrain.config.rl import GRPOConfig, RolloutConfig


@attrs.define(slots=False)
class PosttrainGRPOConfig(PosttrainConfig):
    algorithm: str = "grpo"
    model_backend: str = "i4"
    # The arithmetic quick-start deliberately demonstrates the minimal policy-gradient objective.
    # GRPOConfig itself defaults to the production clipped surrogate.
    rl: GRPOConfig = attrs.field(factory=lambda: GRPOConfig(grpo_surrogate="legacy_pg"))
    # vLLM rollout engine + sampling (top-level, GRPO-only — like model/grid, not an rl objective
    # field). The selected native/HF policy owns the FSDP trainer weights; config.rollout is the separate
    # inference engine that pulls them each step (build.build_vllm_rollout / control.loop.weight_sync).
    rollout: RolloutConfig = attrs.field(factory=RolloutConfig)
    data_packer: DataPackerConfig = attrs.field(factory=DataPackerConfig)


def make_config(*, register_experiment_configs: bool = True) -> PosttrainGRPOConfig:
    """Build the GRPO root, optionally omitting experiment catalogs for a concrete recipe."""
    return build_root(
        PosttrainGRPOConfig,
        "posttrain_grpo",
        register_experiment_configs=register_experiment_configs,
    )
