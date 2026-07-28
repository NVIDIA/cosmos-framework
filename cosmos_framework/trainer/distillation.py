# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
from loguru import logger as log

from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed
from cosmos_framework.model.generator.distillation.optimizer import (
    PhaseOptimizer,
    PhaseScheduler,
    iter_torch_optimizers,
)

__all__: tuple[str, ...] = ("DistillationTrainer",)


@contextmanager
def _sync_grad_for_closure(model: torch.nn.Module, enabled: bool) -> Iterator[None]:
    """Toggle per-closure gradient sync for DDP and FSDP2/HSDP models."""
    fsdp_modules = [
        module for module in model.modules() if callable(getattr(module, "set_requires_gradient_sync", None))
    ]
    if fsdp_modules:
        for module in fsdp_modules:
            module.set_requires_gradient_sync(enabled, recurse=False)
        try:
            yield
        finally:
            for module in fsdp_modules:
                module.set_requires_gradient_sync(True, recurse=False)
        return

    with distributed.ddp_sync_grad(model, enabled):
        yield


class DistillationTrainer(ImaginaireTrainer):
    """Trainer for distillation / interactive model training.

    Inherits ImaginaireTrainer verbatim and overrides only the optimizer hooks.
    Routing is resolved through model.get_optimizer_key when available, with
    model.get_phase as the compatibility fallback.
    """

    @staticmethod
    def _merge_output_batches(
        base: dict[str, torch.Tensor],
        update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for key, value in update.items():
            if key in base and isinstance(base[key], torch.Tensor) and isinstance(value, torch.Tensor):
                base[key] = base[key] + value.detach()
            else:
                base[key] = value.detach() if isinstance(value, torch.Tensor) else value
        return base

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: PhaseOptimizer,
        scheduler: PhaseScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        model = model_ddp.module if self.config.trainer.distributed_parallelism == "ddp" else model_ddp
        closure_fn = getattr(model, "training_step_closures", None)
        if not inspect.ismethod(closure_fn):
            return super().training_step(
                model_ddp,
                optimizer,
                scheduler,
                grad_scaler,
                data,
                iteration=iteration,
                grad_accum_iter=grad_accum_iter,
            )

        self.callbacks.on_before_forward(iteration=iteration)
        with self.training_timer("forward"):
            with self.straggler_detector.profile_section(
                "fwd", self.config.trainer.straggler_detection.analyze_forward
            ):
                closures = list(closure_fn(data, iteration))
        if len(closures) == 0:
            raise RuntimeError(f"{type(model).__name__}.training_step_closures yielded no closures.")

        output_batch: dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | None = None
        should_sync_grad = grad_accum_iter == self.config.trainer.grad_accum_iter - 1
        for _closure_name, closure, is_last_closure in closures:
            with _sync_grad_for_closure(model_ddp, should_sync_grad and is_last_closure):
                with self.training_timer("forward"):
                    with self.straggler_detector.profile_section(
                        "fwd", self.config.trainer.straggler_detection.analyze_forward
                    ):
                        closure_output, closure_loss = closure()
                self.callbacks.on_after_forward(iteration=iteration)
                self._merge_output_batches(output_batch, closure_output)
                closure_loss_detached = closure_loss.detach()  # []
                total_loss = closure_loss_detached if total_loss is None else total_loss + closure_loss_detached  # []
                self.callbacks.on_before_backward(model, closure_loss, iteration=iteration)
                with self.training_timer("backward"):
                    with self.straggler_detector.profile_section(
                        "bwd", self.config.trainer.straggler_detection.analyze_backward
                    ):
                        loss_scaled = grad_scaler.scale(closure_loss / self.config.trainer.grad_accum_iter)
                        loss_scaled.backward()
                        model.on_after_backward()
                self.callbacks.on_after_backward(model, iteration=iteration)

        if total_loss is None:
            raise RuntimeError("No closure loss was produced.")
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model, optimizer, scheduler, grad_scaler, iteration=iteration
                    )
                    self._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
                    model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    self._zero_grad(model, optimizer, iteration)
            grad_accum_iter = 0
        return output_batch, total_loss, grad_accum_iter

    def _optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: PhaseOptimizer,
        scheduler: PhaseScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        if not getattr(self, "_eager_init_done", False):
            for opt_key, opt in optimizer.items():
                self._eager_init_optimizer_state(opt, opt_key)
            self._eager_init_done = True
        key = self._optimizer_key(model, iteration)
        optimizer.step(key, grad_scaler)
        scheduler.step(key)

    def _zero_grad(self, model: torch.nn.Module, optimizer: PhaseOptimizer, iteration: int) -> None:
        key = self._optimizer_key(model, iteration)
        optimizer.zero_grad(key)
        model.zero_grad_for_phase(iteration)

    @staticmethod
    def _optimizer_key(model: torch.nn.Module, iteration: int) -> str:
        get_optimizer_key = getattr(model, "get_optimizer_key", None)
        if callable(get_optimizer_key):
            key = get_optimizer_key(iteration)
            if isinstance(key, str):
                return key

        phase = model.get_phase(iteration)
        return "net" if phase == "student" else "fake_score"

    @staticmethod
    def _eager_init_optimizer_state(optimizer: object, key: str) -> None:
        """Pre-allocate Adam-family optimizer state without changing parameters.

        Adam-family optimizers allocate per-parameter exp_avg/exp_avg_sq lazily on
        the first step() call. With `warmup_critic_steps>0` the student/generator
        does not step until after several checkpoints have been saved, so the
        async checkpointer's pinned-memory companion state_dict is built without
        student optimizer state. Once the student starts stepping, the source
        state_dict structure changes and the next save raises CompanionMismatch
        in `_copy_state_dict`. Pre-allocating both opts up front keeps the
        state_dict structure stable from iter 0.
        """
        inner_optimizers = list(iter_torch_optimizers(optimizer))
        if len(inner_optimizers) != 1 or inner_optimizers[0] is not optimizer:
            for inner_idx, inner_optimizer in enumerate(inner_optimizers):
                DistillationTrainer._eager_init_optimizer_state(inner_optimizer, f"{key}:{inner_idx}")
            return

        torch_optimizer = cast(torch.optim.Optimizer, optimizer)
        log.info(f"[eager-init] allocating optimizer state for key='{key}'")
        uses_group_step = DistillationTrainer._uses_group_step(torch_optimizer)
        DistillationTrainer._eager_init_master_weights(torch_optimizer)
        for group in torch_optimizer.param_groups:
            DistillationTrainer._eager_init_group_state(torch_optimizer, group)
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                state = torch_optimizer.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = DistillationTrainer._optimizer_zero_like(param, uses_group_step)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = DistillationTrainer._optimizer_zero_like(param, uses_group_step)
                if group.get("amsgrad", False) and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = DistillationTrainer._optimizer_zero_like(param, uses_group_step)
                if not uses_group_step and "step" not in state:
                    step_device = param.device if group.get("capturable", False) or group.get("fused", False) else "cpu"
                    state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)  # []

    @staticmethod
    def _uses_group_step(optimizer: torch.optim.Optimizer) -> bool:
        """Return whether the optimizer keeps the step counter on each param group."""
        return hasattr(optimizer, "param_groups_master") and hasattr(optimizer, "capturable")

    @staticmethod
    def _optimizer_zero_like(param: torch.nn.Parameter, force_fp32: bool) -> torch.Tensor:
        """Create an optimizer moment tensor matching the target optimizer convention."""
        value = torch.zeros_like(param)  # [*param.shape]
        return value.float() if force_fp32 else value  # [*param.shape]

    @staticmethod
    def _eager_init_master_weights(optimizer: torch.optim.Optimizer) -> None:
        """Create FusedAdam-style FP32 master weights when the optimizer owns them."""
        if not hasattr(optimizer, "param_groups_master") or optimizer.param_groups_master is not None:
            return
        master_weights = bool(getattr(optimizer, "master_weights", False))
        optimizer.param_groups_master = []
        for group in optimizer.param_groups:
            optimizer.param_groups_master.append(
                {
                    "params": [
                        param.clone().detach().float() if master_weights else None  # [*param.shape]
                        for param in group["params"]
                    ]
                }
            )

    @staticmethod
    def _eager_init_group_state(optimizer: torch.optim.Optimizer, group: dict) -> None:
        """Create FusedAdam-style group state without advancing the optimizer."""
        if not DistillationTrainer._uses_group_step(optimizer) or len(group["params"]) == 0:
            return
        device = group["params"][0].device
        if getattr(optimizer, "capturable", False):
            if "step" not in group:
                group["step"] = torch.zeros(1, dtype=torch.int, device=device)  # [1]
            elif isinstance(group["step"], torch.Tensor):
                group["step"] = group["step"].to(device=device)  # [1]
            else:
                group["step"] = torch.tensor([group["step"]], dtype=torch.int, device=device)  # [1]
            if not isinstance(group["lr"], torch.Tensor):
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)  # []
            else:
                group["lr"] = group["lr"].to(device=device)  # []
        elif "step" not in group:
            group["step"] = 0
