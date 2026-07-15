# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from collections.abc import ItemsView, Iterator
from typing import Any, Protocol, TypeGuard, cast

import torch

from cosmos_framework.utils.generator.optimizer import LRSchedulersContainer, OptimizersContainer

__all__: tuple[str, ...] = (
    "OptimizerContainerLike",
    "OptimizerModelView",
    "PhaseOptimizer",
    "PhaseScheduler",
    "is_optimizer_container",
    "iter_torch_optimizers",
)


class OptimizerContainerLike(Protocol):
    """Structural interface shared by i4 and release-mapped optimizer containers."""

    model: torch.nn.Module
    optimizers: list[torch.optim.Optimizer]

    def step(self) -> None: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


_OPTIMIZER_CONTAINER_METHODS = ("step", "zero_grad", "state_dict", "load_state_dict")


def is_optimizer_container(optimizer: object) -> TypeGuard[OptimizerContainerLike]:
    """Recognize native or release-mapped optimizer containers by their full interface."""
    return isinstance(optimizer, OptimizersContainer) or (
        isinstance(getattr(optimizer, "model", None), torch.nn.Module)
        and isinstance(getattr(optimizer, "optimizers", None), list)
        and all(callable(getattr(optimizer, method, None)) for method in _OPTIMIZER_CONTAINER_METHODS)
    )


class OptimizerModelView(torch.nn.Module):
    """Expose a standalone denoiser as a VFM optimizer-compatible ``.net`` model."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net: torch.nn.Module = net


def iter_torch_optimizers(optimizer: object) -> Iterator[torch.optim.Optimizer]:
    """Yield raw torch optimizers from either a raw optimizer or a VFM container."""
    if is_optimizer_container(optimizer):
        yield from optimizer.optimizers
    else:
        yield cast(torch.optim.Optimizer, optimizer)


class PhaseOptimizer:
    """Optimizer container for alternating-phase distillation training.

    Holds a dict of optimizers and exposes step/zero_grad with an explicit key
    argument. Routing logic (which key is active at a given iteration) lives in
    the trainer, not here.
    """

    def __init__(self, optimizer_dict: dict[str, torch.optim.Optimizer | OptimizerContainerLike]) -> None:
        self._optimizers: dict[str, torch.optim.Optimizer | OptimizerContainerLike] = optimizer_dict

    def step(self, key: str, grad_scaler: torch.amp.GradScaler) -> None:
        for optimizer in iter_torch_optimizers(self._optimizers[key]):
            grad_scaler.step(optimizer)
        grad_scaler.update()

    def zero_grad(self, key: str) -> None:
        self._optimizers[key].zero_grad(set_to_none=True)

    def parameters_for_key(self, key: str) -> list[torch.nn.Parameter]:
        return [
            param
            for optimizer in iter_torch_optimizers(self._optimizers[key])
            for group in optimizer.param_groups
            for param in group["params"]
        ]

    def get(self, key: str) -> torch.optim.Optimizer | OptimizerContainerLike | None:
        return self._optimizers.get(key)

    def items(self) -> ItemsView[str, torch.optim.Optimizer | OptimizerContainerLike]:
        return self._optimizers.items()


class PhaseScheduler:
    """Scheduler container for alternating-phase distillation training.

    Mirrors PhaseOptimizer: holds a dict of LR schedulers and exposes
    step with an explicit key argument.
    """

    def __init__(
        self,
        scheduler_dict: dict[str, torch.optim.lr_scheduler.LRScheduler | LRSchedulersContainer],
    ) -> None:
        self._schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler | LRSchedulersContainer] = scheduler_dict

    def step(self, key: str) -> None:
        self._schedulers[key].step()

    def get(self, key: str) -> torch.optim.lr_scheduler.LRScheduler | LRSchedulersContainer | None:
        return self._schedulers.get(key)

    def items(self) -> ItemsView[str, torch.optim.lr_scheduler.LRScheduler | LRSchedulersContainer]:
        return self._schedulers.items()
