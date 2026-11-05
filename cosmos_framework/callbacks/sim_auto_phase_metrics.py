# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Record SF-DMD student and critic diagnostics without a hosted logging service."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback

LOSS_KEYS: tuple[str, ...] = ("vsd_loss", "total_generator_loss", "loss_fake_score", "total_critic_loss")


class PhaseMetrics(Callback):
    """Append rank-averaged phase losses and gradient diagnostics to rank-0 JSONL."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path: str = output_path

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],  # Values: [...]; not inspected.
        output_batch: dict[str, torch.Tensor],  # Selected loss values: [].
        loss: torch.Tensor,  # []
        iteration: int = 0,
    ) -> None:
        # The trainer advances its counter before dispatching this callback.
        get_phase = getattr(model, "get_phase")
        record: dict[str, str | int | float] = {
            "iteration": iteration,
            "phase": str(get_phase(max(iteration - 1, 0))),
        }
        metric_values: dict[str, torch.Tensor | float] = {
            key: output_batch[key] for key in LOSS_KEYS if key in output_batch
        }
        metric_values.update(getattr(model, "_distillation_parity_grad_metrics", {}))
        distributed_run = dist.is_initialized()
        for key, value in sorted(metric_values.items()):
            if isinstance(value, torch.Tensor):
                to_local = getattr(value, "to_local", None)
                local_value = to_local() if callable(to_local) else value  # [...]
                metric = local_value.detach().to(device=loss.device, dtype=torch.float32).mean()  # []
            else:
                metric = torch.tensor(value, device=loss.device, dtype=torch.float32)  # []
            if distributed_run:
                dist.all_reduce(metric, op=dist.ReduceOp.SUM)  # []
                metric.div_(dist.get_world_size())  # []
            record[key] = metric.item()

        if distributed_run and dist.get_rank() != 0:
            return
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
