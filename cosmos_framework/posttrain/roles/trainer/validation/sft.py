# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Side-effect-isolated objective evaluation, not a durable validation transaction."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.roles.trainer.validation.runtime import (
    reshard_fsdp_modules,
    validation_module_inventory,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import run_rank_converged


def evaluate_sft_batch(model: Any, objective: Objective, batch: list[Trajectory]) -> dict[str, Any]:
    """Evaluate one physical batch and return raw objective metrics.

    No optimizer, source cursor, metric aggregation or publication state is owned here.
    Restore each individual module's mode (including deliberately mixed train/eval
    children) and all captured RNG streams even if preparation or evaluation fails.
    """
    roots, modules = run_rank_converged(
        lambda: validation_module_inventory(
            SimpleNamespace(net=model) if isinstance(model, torch.nn.Module) else model
        ),
        "SFT validation module inventory",
    )
    rng_state = collect_rng_state()
    cleanup = ExitStack()
    cleanup.callback(restore_rng_state, rng_state)
    for module in modules:
        cleanup.callback(setattr, module, "training", module.training)
    cleanup.callback(reshard_fsdp_modules, roots)
    try:

        def enter_eval() -> None:
            for root in roots.values():
                root.eval()

        run_rank_converged(enter_eval, "SFT validation eval setup")
        with torch.no_grad():
            _, metrics = objective(model, batch)
        return metrics
    finally:
        # ExitStack runs every restoration even if an earlier cleanup fails.
        run_rank_converged(cleanup.close, "SFT validation state restoration")
