# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import wandb

from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback


class LearningRateLogger(Callback):
    """Log reasoner default and LR-multiplier groups.

    Parameters that do not match a configured ``optimizer.lr_multipliers`` key
    are logged as ``optim/lr_default``. Named multiplier groups, such as
    ``model.visual``, are logged as ``optim/lr_model.visual`` every
    ``every_n × logging_iter`` steps.
    """

    def __init__(self, every_n: int = 10) -> None:
        self.every_n: int = every_n

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
        if not distributed.is_rank0() or not wandb.run:
            return
        if not (hasattr(optimizer, "optimizers") and hasattr(optimizer, "model")):
            return

        lr_multiplier_keys = list(self.config.optimizer.get("lr_multipliers", {}))
        optimizer_net = getattr(optimizer.model, "net", None)
        if optimizer_net is None:
            return

        lr_key_by_param_id: dict[int, str] = {}
        for param_name, param in optimizer_net.named_parameters():
            lr_key_by_param_id[id(param)] = next(
                (lr_key for lr_key in lr_multiplier_keys if lr_key in param_name),
                "default",
            )

        unique_lr: dict[str, float | torch.Tensor] = {}
        for inner_optimizer in optimizer.optimizers:
            for param_group in inner_optimizer.param_groups:
                for param in param_group["params"]:
                    lr_key = lr_key_by_param_id.get(id(param))
                    if lr_key is not None:
                        unique_lr[f"optim/lr_{lr_key}"] = param_group["lr"]
        if not unique_lr:
            return
        wandb.log(unique_lr, step=iteration)
