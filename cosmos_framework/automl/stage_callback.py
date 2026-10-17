# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Training-only checkpoint lineage and native PBT optimizer-restore proof."""

import json
import math
from pathlib import Path

import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


def verify_optimizer_configuration(optimizer, scheduler):
    """Verify Framework's native restoration, retaining multipliers/no-decay groups.

    OptimizersContainer retains the new recipe's per-parameter weight decay;
    LRSchedulersContainer restores progress with the new per-group base LRs.
    Do not flatten these containers or overwrite all groups with one LR/WD.
    """
    configured_lrs = scheduler._configured_base_lrs
    if not (len(optimizer.optimizers) == len(scheduler.schedulers) == len(configured_lrs)):
        raise ValueError("PBT optimizer/scheduler container topology mismatch")
    for optim, sched, bases in zip(optimizer.optimizers, scheduler.schedulers, configured_lrs):
        if sched.optimizer is not optim or not (
            len(optim.param_groups) == len(bases) == len(sched.base_lrs) == len(sched.lr_lambdas)
        ):
            raise ValueError("PBT parameter-group scheduler topology mismatch")
        for group, base, actual_base, schedule in zip(optim.param_groups, bases, sched.base_lrs, sched.lr_lambdas):
            expected_decay = {optimizer._configured_weight_decay_by_param_id[id(p)] for p in group["params"]}
            if expected_decay != {group["weight_decay"]}:
                raise ValueError("Donor restoration lost the new recipe's per-group weight decay")
            if actual_base != base or not math.isclose(float(group["lr"]), base * schedule(sched.last_epoch)):
                raise ValueError("Donor restoration lost the new recipe's LR or scheduler progress")


class StageCallback(Callback):
    def __init__(self, stage_file: str):
        super().__init__()
        self.path = Path(stage_file)
        self.stage = json.loads(self.path.read_text())
        self.restored = False
        self.verified = False

    def _receipt(self, iteration):
        rank = dist.get_rank() if dist.is_initialized() else 0
        target = self.path.parent / f"resume-rank-{rank}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "source_checkpoint": self.stage["source_checkpoint"],
                    "optimizer_configuration_verified": self.verified,
                }
            )
        )
        temporary.replace(target)

    def on_load_checkpoint_end(self, model, iteration=0, checkpoint_path=None):
        expected = self.stage["expected_start_iteration"]
        if iteration != expected:
            raise ValueError(f"AutoML stage expected resume iteration {expected}, got {iteration}")
        source = self.stage["source_checkpoint"]
        if source and (not checkpoint_path or Path(checkpoint_path).resolve() != Path(source).resolve()):
            raise ValueError("AutoML stage restored an unexpected checkpoint")
        self.restored = True
        self._receipt(iteration)

    def on_train_start(self, model, iteration=0):
        if not self.restored:
            raise ValueError("Native checkpointer did not attest AutoML stage initialization")

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if self.stage["resume_mode"] == "pbt" and not self.verified:
            verify_optimizer_configuration(optimizer, scheduler)
            self.verified = True
            self._receipt(self.stage["expected_start_iteration"])
