# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native supervised action training: family factories, shared trainer and recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig, RunConfig
from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig


@dataclass
class ActionSFTConfig(RunConfig):
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(impl="fused"))
    optimizer_profile: str = "native_fused_all_v1"
    max_steps: int = 100
    micro_batch_size: int = 1
    evaluation_every: int = 10
    evaluation_seed: int = 123
    evaluation_max_samples: int = 0
    reuse_evaluation_inputs: bool = False
    reuse_gr00t_evaluation_backbone: bool = False
    lr_schedule: SFTLRScheduleConfig = field(default_factory=SFTLRScheduleConfig)
    pause_after_step: int = 0
    target_metric: str = ""
    target_threshold: float | None = None
    target_direction: str = "min"
    stop_on_target: bool = False
    checkpoint_format: str = "dcp"
    save_async: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.lr_schedule.__post_init__()
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("native action SFT requires a positive max_steps; use action_eval for evaluation only")
        from cosmos_framework.posttrain.config.gr00t_official import (
            NATIVE_ADAPTER_OPTIMIZER_PROFILE,
            OFFICIAL_HEAD_PROFILE,
            validate_official_head_optimizer,
        )

        if self.optimizer_profile == OFFICIAL_HEAD_PROFILE:
            validate_official_head_optimizer(self.optimizer)
        elif self.optimizer_profile == NATIVE_ADAPTER_OPTIMIZER_PROFILE:
            if self.optimizer.name != "adamw" or self.optimizer.impl != "fused" or self.optimizer.decay_groups != "":
                raise ValueError("action SFT adaptation requires fused AdamW with explicit all-parameter decay groups")
        else:
            raise ValueError("unsupported native action optimizer profile")
        if self.optimizer.train_keys:
            raise ValueError("action SFT owns exact adapter/auxiliary trainability; optimizer train_keys must be empty")
        if self.grad_accum != 1:
            raise ValueError("native action SFT currently requires grad_accum=1; task reductions must not be regrouped")
        if self.checkpoint_format != "dcp" or self.save_async:
            raise ValueError("native action SFT requires synchronous DCP checkpoints")
        if set(self.checkpoint_components) != {"model", "optimizer", "training_state"}:
            raise ValueError("native action SFT requires complete checkpoint components")
        if not self.deterministic:
            raise ValueError("native action SFT requires deterministic=True")
        for name in ("micro_batch_size", "evaluation_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("evaluation_max_samples", "pause_after_step"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.reuse_evaluation_inputs) is not bool:
            raise ValueError("reuse_evaluation_inputs must be a boolean")
        if type(self.reuse_gr00t_evaluation_backbone) is not bool:
            raise ValueError("reuse_gr00t_evaluation_backbone must be a boolean")
        if self.pause_after_step > self.max_steps:
            raise ValueError("pause_after_step must not exceed the declared max_steps")
        if self.target_direction not in {"min", "max"}:
            raise ValueError("target_direction must be min or max")
        if bool(self.target_metric) != (self.target_threshold is not None):
            raise ValueError("target_metric and target_threshold must be specified together")
        if self.target_threshold is not None and not math.isfinite(self.target_threshold):
            raise ValueError("target_threshold must be finite")
        if self.target_metric and self.keep_model_last != 0:
            raise ValueError("action target evidence requires keep_model_last=0 to retain its verified checkpoint")
        if self.stop_on_target and not self.target_metric:
            raise ValueError("stop_on_target requires a prospectively specified target")


@attrs.define(slots=False)
class PosttrainActionSFTConfig(PosttrainConfig):
    algorithm: str = "action_sft"
    rl: ActionSFTConfig = attrs.field(factory=ActionSFTConfig)
    data: Any = None
    policy_factory: str = ""
    data_factory: str = ""
    objective_factory: str = ""
    evaluation_callback: str = ""
    dense_policy_factory: str = ""
    dense_sidecars_factory: str = ""
    defaults: list[Any] = attrs.field(factory=lambda: ["_self_"])


def make_config() -> PosttrainActionSFTConfig:
    config = PosttrainActionSFTConfig(
        model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None
    )
    config.job.group = "posttrain_action_sft"
    config.trainer.callbacks = None
    config.trainer.run_validation = False
    config.upload_reproducible_setup = False
    return config
