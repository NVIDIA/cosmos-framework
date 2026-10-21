# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Driver-owned Joint PPO telemetry; one iteration is one published window."""

from __future__ import annotations

import netrc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from cosmos_framework.posttrain.callbacks.one_logger import OneLoggerLifecycleObserver
from cosmos_framework.posttrain.utils.report import resolve_wandb_destination

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.config_joint_ppo import PosttrainJointPPOConfig
    from cosmos_framework.posttrain.config.rl import JointPPOConfig


ITERATION_UNIT = "joint_ppo_window_v1"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value.strip().lower() == "true"


class JointPPOOneLogger(OneLoggerLifecycleObserver):
    """Window counters stay distinct from optimizer-step observers; checkpoint events use the shared seam."""

    fail_fast_start = True
    count_open_intervals_on_train_end = False
    emit_app_end_after_failure = False

    @classmethod
    def create(cls, config: PosttrainJointPPOConfig, recipe: JointPPOConfig, width: int) -> JointPPOOneLogger | None:
        online = resolve_wandb_destination(config.job).mode == "online"
        enabled = _env_bool("ENABLE_ONELOGGER", bool(os.environ.get("SLURM_JOB_ID")) and online)
        if not enabled:
            return None

        # Canonical onboarding requires netrc; the SDK prioritizes WANDB_API_KEY.
        path = Path(os.environ.get("NETRC") or Path.home() / ".netrc").expanduser()
        try:
            auth = netrc.netrc(path).authenticators("api.wandb.ai")
        except (OSError, netrc.NetrcParseError):
            raise RuntimeError(f"OneLogger requires a readable, parseable W&B netrc at {path}") from None
        if auth is None or not auth[2]:
            raise RuntimeError(f"OneLogger requires an api.wandb.ai credential in {path}")
        return cls(config, recipe, width)

    def __init__(self, config: PosttrainJointPPOConfig, recipe: JointPPOConfig, width: int) -> None:
        # Import explicitly: OneLoggerUtils otherwise swallows missing SDK errors.
        from one_logger.core import OneLogger  # noqa: F401

        from cosmos_framework.utils.one_logger import one_logger_utils

        self.updates_per_window = recipe.decisions // recipe.global_minibatch
        self.decisions = recipe.decisions
        metadata = {"enable_one_logger": True, "one_logger_iteration_unit": ITERATION_UNIT}
        destination = resolve_wandb_destination(config.job)
        name = f"{destination.project}/{destination.group}/{destination.name}"
        target = recipe.max_steps // self.updates_per_window
        sdk_config = dict(
            enable_for_current_rank=True,
            one_logger_project="imaginaire4",
            one_logger_run_name=name,
            one_logger_async=_env_bool("ONE_LOGGER_ASYNC", True),
            log_every_n_train_iterations=1,
            barrier=None,
            app_tag=f"{name}/{width}/joint_ppo",
            app_tag_run_name=name,
            app_tag_run_version=ITERATION_UNIT,
            app_start_time=round(time.time() * 1000),
            world_size=width,
            micro_batch_size=recipe.microbatch,
            batch_size=recipe.decisions,
            global_batch_size=recipe.decisions,
            train_iterations_target=target,
            train_samples_target=target * recipe.decisions,
            is_baseline_run=False,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=recipe.validation.enabled,
            is_test_iterations_enabled=False,
            is_save_checkpoint_enabled=True,
            save_checkpoint_strategy="sync",
            is_log_throughput_enabled=False,
            metadata={
                **metadata,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
                "joint_ppo_updates_per_window": self.updates_per_window,
                "training_wandb_entity": destination.entity,
                "training_wandb_run_name": name,
            },
        )
        super().__init__(one_logger_utils, sdk_config)

    @property
    def metadata(self) -> dict[str, object]:
        return {"enable_one_logger": True, "one_logger_iteration_unit": ITERATION_UNIT} if self.initialized else {}

    def _initialize_one_logger(self) -> None:
        # This driver owns its wrapper, including a partially initialized instance.
        wrapper = self._utils.OneLoggerUtils
        self._one_logger = wrapper.__new__(wrapper)
        self._one_logger.__init__(self._config)
        if not getattr(self._one_logger, "one_logger", None):
            raise RuntimeError("OneLogger SDK initialization did not create a telemetry client")

    def _release_one_logger(self) -> None:
        # No imaginaire global wrapper was borrowed or registered by this adapter.
        pass

    def _checkpoint_iteration(self, model_step: int) -> int:
        return model_step // self.updates_per_window

    def on_checkpoint_committed(self, model_step: int, *, location: str) -> None:
        # The coordinator owns the actual location, including partial artifacts
        # and remote stores. Failed writes retain the last successful metadata.
        self.log_metrics(
            {
                "joint_ppo_checkpoint_path": location,
                "joint_ppo_checkpoint_optimizer_step": model_step,
            }
        )
        self.on_checkpoint_save_success(model_step)
