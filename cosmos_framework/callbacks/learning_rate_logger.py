# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import wandb

from cosmos_framework.utils.callback import Callback


class LearningRateLogger(Callback):
    """Logs per-model-part learning rate every ``every_n × logging_iter`` steps.

    Designed for VLM training where the optimizer is an
    ``OptimizersContainer`` exposing ``.optimizers`` (list of single-element
    optimizer lists) paired with ``.model_part_names``. Silently no-ops when
    those attributes are absent so it can be registered alongside plain
    ``torch.optim.Optimizer`` setups without harm.
    """

    def __init__(self, every_n: int = 10):
        self.every_n = every_n

    def on_before_optimizer_step(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model, scheduler, grad_scaler
        gate = self.config.trainer.logging_iter * self.every_n
        if not (iteration == 1 or (gate > 0 and iteration % gate == 0)):
            return
        if not wandb.run:
            return
        if not (hasattr(optimizer, "optimizers") and hasattr(optimizer, "model_part_names")):
            return
        unique_lr: dict[str, float] = {}
        for optim_per_model, name in zip(optimizer.optimizers, optimizer.model_part_names):
            if not optim_per_model:
                continue
            for pg in optim_per_model[0].param_groups:
                unique_lr[f"optim/lr_{name}"] = pg["lr"]
        if not unique_lr:
            return
        wandb.log(unique_lr, step=iteration)
