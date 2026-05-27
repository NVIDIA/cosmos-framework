# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Optional

import torch
import torch.distributed

from cosmos_framework.checkpoint.base import AbstractCheckpointer
from cosmos_framework.model._base import ImaginaireModel


class Checkpointer(AbstractCheckpointer):
    """
    A dummy checkpointer that does not save or load anything. This is useful for debugging jobs or share workload with collobrators.
    """

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        pass

    def load(
        self,
        model: ImaginaireModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        return 0
