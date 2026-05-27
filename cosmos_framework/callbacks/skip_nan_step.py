# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


class SkipNaNStep(Callback):
    """Skip the optimizer step only when ALL ranks produce NaN/Inf loss.

    When only some ranks produce NaN, the existing GradClip callback's
    nan_to_num handling is sufficient (NaN gradients become 0, valid
    gradients from clean ranks are still used). This callback only
    intervenes when every rank has NaN, meaning no useful gradient
    signal exists.

    The all-reduce ensures all ranks agree on skip/no-skip, preventing
    NCCL desync.

    Args:
        max_consecutive_nan: Abort training after this many consecutive
            all-rank-NaN optimizer steps. Set to 0 to disable the limit.
    """

    def __init__(self, max_consecutive_nan: int = 100) -> None:
        super().__init__()
        self.max_consecutive_nan = max_consecutive_nan
        self._nan_detected = False
        self._consecutive_nan_count = 0

    def on_before_backward(
        self,
        model: ImaginaireModel,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self._nan_detected = True

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        nan_flag = torch.tensor([1.0 if self._nan_detected else 0.0], device="cuda")
        dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
        nan_rank_count = int(nan_flag.item())
        world_size = dist.get_world_size()

        if nan_rank_count > 0 and nan_rank_count < world_size:
            self._consecutive_nan_count = 0

        elif nan_rank_count == world_size:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            self._consecutive_nan_count += 1
            log.warning(
                f"ALL ranks NaN/Inf at iteration {iteration}, skipping optimizer step "
                f"(consecutive: {self._consecutive_nan_count})",
            )

            if self.max_consecutive_nan > 0 and self._consecutive_nan_count >= self.max_consecutive_nan:
                raise RuntimeError(
                    f"Training unstable: all-rank NaN/Inf loss for {self._consecutive_nan_count} "
                    f"consecutive optimizer steps at iteration {iteration}. Aborting.",
                )
        else:
            self._consecutive_nan_count = 0

        self._nan_detected = False
