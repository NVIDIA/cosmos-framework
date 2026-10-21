# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Mirrors the optimizer registration from Cosmos3 VFM configs.
"""

from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_framework.utils.lazy_config import PLACEHOLDER
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.defaults.optimizer import (
    register_optimizer as register_vfm_optimizer,
)
from cosmos_framework.configs.base.defaults.optimizer import (
    register_scheduler as register_vfm_scheduler,
)
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import register_dmd2_optimizer
from cosmos_framework.utils.generator.optimizer import build_lr_scheduler


def register_optimizer() -> None:
    register_vfm_optimizer()
    register_dmd2_optimizer()


def register_scheduler() -> None:
    register_vfm_scheduler()
    cs = ConfigStore.instance()
    cs.store(
        group="scheduler",
        package="scheduler",
        name="constant",
        node=L(build_lr_scheduler)(
            optimizer=PLACEHOLDER,
            lr_scheduler_type="LambdaLinear",
            warm_up_steps=[0],
            cycle_lengths=[10_000_000_000_000],
            f_start=[1.0],
            f_max=[1.0],
            f_min=[1.0],
        ),
    )
