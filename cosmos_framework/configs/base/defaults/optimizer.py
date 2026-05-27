# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical Hydra-group registry for the optimizer and scheduler SKUs."""

from typing import Any

from cosmos_framework.utils.lazy_config import PLACEHOLDER
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.config_helper import ConfigStore
from cosmos_framework.utils.vfm.optimizer import build_lr_scheduler, build_optimizer

OPTIMIZER_KWARGS: dict[str, Any] = dict(
    # Learning rate for the optimizer.
    lr=1e-4,
    # Weight decay for the optimizer.
    weight_decay=0.1,
    # Beta1 and beta2 for the optimizer.
    betas=[0.9, 0.99],
    # Epsilon for the optimizer.
    eps=1e-8,
    # Whether to use fuse updates to all parameters.
    fused=True,
    # Keys to select for the optimizer.
    keys_to_select=[],
    # Per-key LR multipliers. Maps parameter name patterns to LR multipliers.
    # E.g. {"sound2llm": 5.0, "llm2sound": 5.0} gives those params 5x the base LR.
    lr_multipliers={},
    # Whether to disable weight decay for one-dimensional params such as norm weights and biases.
    # Default is False to preserve historical optimizer behavior.
    disable_weight_decay_for_1d_params=False,
)

LAMBDACOSINE_KWARGS: dict[str, Any] = dict(
    warm_up_steps=[2000],
    cycle_lengths=[100000],
    f_start=[0.0],
    f_max=[1.0],
    f_min=[0.0],
    verbosity_interval=0,
)


def register_optimizers(optimizer_kwargs: dict[str, Any]) -> None:
    """Register the ``fusedadamw`` and ``adamw`` SKUs."""
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fusedadamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="FusedAdam",
            **optimizer_kwargs,
        ),
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="adamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="AdamW",
            **optimizer_kwargs,
        ),
    )


def register_schedulers(lambdacosine_kwargs: dict[str, Any]) -> None:
    """Register the ``lambdalinear`` and ``lambdacosine`` SKUs."""
    cs = ConfigStore.instance()
    cs.store(
        group="scheduler",
        package="scheduler",
        name="lambdalinear",
        node=L(build_lr_scheduler)(
            optimizer=PLACEHOLDER,
            lr_scheduler_type="LambdaLinear",
            warm_up_steps=[1000],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
    )
    cs.store(
        group="scheduler",
        package="scheduler",
        name="lambdacosine",
        node=L(build_lr_scheduler)(
            optimizer=PLACEHOLDER,
            lr_scheduler_type="LambdaCosine",
            **lambdacosine_kwargs,
        ),
    )


def register_optimizer() -> None:
    """VFM project-root entry point."""
    register_optimizers(OPTIMIZER_KWARGS)


def register_scheduler() -> None:
    """VFM project-root entry point."""
    register_schedulers(LAMBDACOSINE_KWARGS)
