# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Apply a new stage's initial LR after restoring the preceding Adam state."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.utils.callback import Callback


class ResetNewStageLearningRate(Callback):
    """Keep Adam moments but use the fresh scheduler at a reset stage counter.

    Optimizer state includes the preceding run's param-group LR. Restoring it
    after constructing a new scheduler otherwise overrides that scheduler's
    initialized LR for the first update. A real same-stage resume is untouched.
    """

    def on_before_optimizer_step(
        self, model: Any, optimizer: Any, scheduler: Any, grad_scaler: Any, iteration: int = 0
    ) -> None:
        del model, grad_scaler
        if iteration != 0:
            return
        optimizers = getattr(optimizer, "optimizers", [optimizer])
        schedulers = getattr(scheduler, "schedulers", [scheduler])
        groups = [group for inner in optimizers for group in inner.param_groups]
        rates = [
            base * schedule(0)
            for inner in schedulers
            for base, schedule in zip(inner.base_lrs, inner.lr_lambdas, strict=True)
        ]
        if len(groups) != len(rates):
            raise ValueError("Stage handoff optimizer and new scheduler groups differ")
        for group, rate in zip(groups, rates, strict=True):
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(rate)
            else:
                group["lr"] = rate
