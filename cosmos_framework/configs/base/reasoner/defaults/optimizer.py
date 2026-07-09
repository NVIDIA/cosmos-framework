# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hydra config registrations for VLM optimizer + LR scheduler."""

from typing import Any

from cosmos_framework.configs.base.defaults.optimizer import (
    register_optimizers,
    register_schedulers,
)

# Shared optimizer kwargs for both fusedadamw and adamw registrations.
# ``lr_multipliers``: vision_encoder backbone at 0.1x base LR; everything else
# at 1.0x. Substring match; only entries != 1.0 need to appear. See
# ``_filter_params_grouped`` in ``vfm/utils/optimizer.py``.
VLM_OPTIMIZER_KWARGS: dict[str, Any] = dict(
    lr=2e-6,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    fused=True,
    keys_to_select=[],
    lr_multipliers={"model.visual": 0.1},
)

# ``f_start`` / ``f_min`` are ratios of the optimizer's base ``lr``:
#     effective_init_lr = lr * f_start
#     effective_end_lr  = lr * f_min
# Update these together with ``lr`` if you want absolute LR endpoints to stay fixed.
VLM_LAMBDACOSINE_KWARGS: dict[str, Any] = dict(
    warm_up_steps=[1000],
    cycle_lengths=["${trainer.max_iter}"],
    f_start=[0.01],
    f_max=[1.0],
    f_min=[0.1],
)


def register_optimizer() -> None:
    """VLM project-root entry point."""
    register_optimizers(VLM_OPTIMIZER_KWARGS)


def register_scheduler() -> None:
    """VLM project-root entry point."""
    register_schedulers(VLM_LAMBDACOSINE_KWARGS)
