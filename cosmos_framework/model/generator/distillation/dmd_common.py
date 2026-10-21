# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared DMD training bookkeeping and metadata helpers, independent of rollout strategy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2OptimizerConfig
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution
from cosmos_framework.model.generator.distillation.optimizer import (
    OptimizerModelView,
    PhaseOptimizer,
    PhaseScheduler,
    iter_torch_optimizers,
)
from cosmos_framework.utils.generator.dtensor_helper import DTensorFastEmaModelUpdater


def generated_vision_item_indices(
    num_vision_items_per_sample: list[int] | None,
    num_vision_items: int,
) -> list[int]:
    """Return flat vision item indices corresponding to generated targets.

    Multi-item vision samples (transfer, image editing) are packed in order, with
    all items except the last treated as fully conditioned controls and the last
    item treated as the generated target. Distillation losses must reduce over the
    generated targets only, otherwise the fully-masked control items contribute
    zeros that dilute the mean.
    """
    if num_vision_items_per_sample is None:
        return list(range(num_vision_items))
    if sum(num_vision_items_per_sample) != num_vision_items:
        raise ValueError(
            "num_vision_items_per_sample must sum to the number of flat vision items, "
            f"got counts={num_vision_items_per_sample} and {num_vision_items} items."
        )

    generated_item_indices: list[int] = []
    item_offset = 0
    for num_items in num_vision_items_per_sample:
        if num_items <= 0:
            raise ValueError(f"Vision item counts must be positive, got {num_vision_items_per_sample}.")
        generated_item_indices.append(item_offset + num_items - 1)
        item_offset += num_items
    return generated_item_indices


def vision_token_count(tokens: torch.Tensor) -> int:  # [C,T,H,W] or [1,C,T,H,W]
    """Return the T*H*W count used by the noise sampler, excluding channels."""
    if tokens.ndim not in (4, 5):
        raise ValueError(f"Expected vision tokens with 4 or 5 dims, got shape {tuple(tokens.shape)}")
    return tokens.shape[-3] * tokens.shape[-2] * tokens.shape[-1]


def get_data_resolutions(data_batch: dict[str, Any], batch_size: int, default_resolution: str) -> list[str]:
    """Return per-sample resolution strings for RF shift lookup."""
    if "image_size" not in data_batch:
        return [default_resolution] * batch_size

    data_resolutions: list[str] = []
    for i in range(batch_size):
        image_size = data_batch["image_size"][i]  # [1,4] or [4]
        if image_size.dim() == 2:
            image_size = image_size[0]  # [4]
        target_h = int(image_size[0].item())
        target_w = int(image_size[1].item())
        data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
    return data_resolutions


class DMDTrainingBase:
    """Share phase/optimizer hooks without constructing networks or changing model initialization.

    The concrete model supplies the networks, configuration, and EMA schedule.
    Rollout, trainable-parameter selection, and checkpoint loading stay with that model.
    """

    config: Any
    net: torch.nn.Module
    net_fake_score: torch.nn.Module
    net_ema: torch.nn.Module
    net_ema_worker: DTensorFastEmaModelUpdater
    ema_beta: Callable[[int], float]

    # ------------------------ Optimizers & schedulers ------------------------

    def init_optimizer_scheduler(
        self, optimizer_config: DMD2OptimizerConfig, scheduler_config: LazyDict
    ) -> tuple[PhaseOptimizer, PhaseScheduler]:
        """Create optimizers/schedulers for student (net) and critic (fake_score)."""
        opt_net = lazy_instantiate(optimizer_config.net, model=OptimizerModelView(self.net))
        sched_net = lazy_instantiate(scheduler_config, optimizer=opt_net)
        opt_fake = lazy_instantiate(optimizer_config.fake_score, model=OptimizerModelView(self.net_fake_score))
        sched_fake = lazy_instantiate(scheduler_config, optimizer=opt_fake)
        return (
            PhaseOptimizer({"net": opt_net, "fake_score": opt_fake}),
            PhaseScheduler({"net": sched_net, "fake_score": sched_fake}),
        )

    def get_phase(self, iteration: int) -> str:
        """Return the current training phase: ``"student"`` or ``"critic"``."""
        assert not (self.config.warmup_student_steps and self.config.warmup_critic_steps)
        if iteration < self.config.warmup_critic_steps:
            return "critic"
        elif iteration < self.config.warmup_student_steps:
            return "student"
        else:
            return "student" if iteration % self.config.student_update_freq == 0 else "critic"

    def get_optimizer_key(self, iteration: int) -> str:
        return "net" if self.get_phase(iteration) == "student" else "fake_score"

    def get_student_iteration(self, iteration: int) -> int:
        """Effective student iteration index used for EMA scheduling."""
        if iteration < self.config.warmup_student_steps:
            return iteration
        else:
            steps_after_warmup = (iteration - self.config.warmup_student_steps) // self.config.student_update_freq
            return self.config.warmup_student_steps + steps_after_warmup

    # ------------------------ training hooks ------------------------

    def on_before_zero_grad(self, optimizer: PhaseOptimizer, scheduler: PhaseScheduler, iteration: int) -> None:
        del scheduler

        if self.get_phase(iteration) == "critic":
            # Critic phase: for BF16 optimizers that maintain FP32 master weights, copy the master params
            # back into the model params before zeroing gradients. This keeps the low-precision model weights
            # in sync with the authoritative FP32 copy.
            opt = optimizer.get("fake_score")
            if opt is not None:
                params, master_params = [], []
                for inner_opt in iter_torch_optimizers(opt):
                    param_groups_master = getattr(inner_opt, "param_groups_master", None)
                    if not getattr(inner_opt, "master_weights", False) or param_groups_master is None:
                        continue
                    for group, group_master in zip(inner_opt.param_groups, param_groups_master):
                        for p, p_master in zip(group["params"], group_master["params"]):
                            params.append(get_local_tensor_if_DTensor(p.data))  # [*parameter_shape]
                            master_params.append(get_local_tensor_if_DTensor(p_master.data))  # [*parameter_shape]
                if params:
                    torch._foreach_copy_(params, master_params)  # list of [*parameter_shape]
        elif self.get_phase(iteration) == "student":
            # Student phase: update the EMA model using the current student weights.
            if self.config.ema.enabled:
                ema_beta = self.ema_beta(self.get_student_iteration(iteration))
                self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def zero_grad_for_phase(self, iteration: int) -> None:
        if self.get_phase(iteration) == "student":
            self.net.zero_grad(set_to_none=True)
        else:
            self.net_fake_score.zero_grad(set_to_none=True)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> torch.Tensor:  # []
        if not self.config.grad_clip:
            max_norm = 1e12
        # Clip each network separately so their individual grad norms are bounded.
        # Return the student norm as the canonical value logged by the trainer.
        clip_grad_norm_(
            self.net_fake_score.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )  # []
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )  # []
