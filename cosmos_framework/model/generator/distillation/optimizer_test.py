# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import MagicMock, call

import pytest
import torch

from cosmos_framework.model.generator.distillation import optimizer as optimizer_module
from cosmos_framework.model.generator.distillation.optimizer import OptimizerModelView, PhaseOptimizer, PhaseScheduler
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


def _make_optimizer() -> MagicMock:
    opt = MagicMock()
    opt.param_groups = []
    return opt


def _make_scheduler() -> MagicMock:
    return MagicMock()


def _make_grad_scaler() -> MagicMock:
    return MagicMock()


def _make_optimizer_container(*optimizers: MagicMock) -> OptimizersContainer:
    container = object.__new__(OptimizersContainer)
    container.optimizers = list(optimizers)
    container.zero_grad = MagicMock()
    return container


class _ForeignOptimizerContainer:
    def __init__(self, *optimizers: MagicMock) -> None:
        self.model: torch.nn.Module = torch.nn.Linear(1, 1)
        self.optimizers: list[MagicMock] = list(optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        return {"foreign": True}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        del state_dict


@pytest.mark.L0
def test_optimizer_model_view_exposes_net_submodule() -> None:
    net = torch.nn.Linear(2, 3)
    view = OptimizerModelView(net)
    assert view.net is net
    assert dict(view.net.named_parameters()) == dict(net.named_parameters())


# -------------------------------------------------------------------------
# PhaseOptimizer.step — only the given key's optimizer is stepped
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_step_net_key() -> None:
    opt_net, opt_fake = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": opt_net, "fake_score": opt_fake})
    scaler = _make_grad_scaler()
    po.step("net", scaler)

    scaler.step.assert_called_once_with(opt_net)
    scaler.update.assert_called_once()
    opt_fake.step.assert_not_called()


@pytest.mark.L0
def test_step_fake_score_key() -> None:
    opt_net, opt_fake = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": opt_net, "fake_score": opt_fake})
    scaler = _make_grad_scaler()
    po.step("fake_score", scaler)

    scaler.step.assert_called_once_with(opt_fake)
    scaler.update.assert_called_once()
    opt_net.step.assert_not_called()


@pytest.mark.L0
def test_step_calls_grad_scaler_update_once() -> None:
    opt_net = _make_optimizer()
    po = PhaseOptimizer({"net": opt_net})
    scaler = _make_grad_scaler()
    po.step("net", scaler)
    scaler.update.assert_called_once()


@pytest.mark.L0
def test_step_optimizer_container_steps_inner_optimizers() -> None:
    opt_a, opt_b = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": _make_optimizer_container(opt_a, opt_b)})
    scaler = _make_grad_scaler()
    po.step("net", scaler)
    assert scaler.step.call_args_list == [call(opt_a), call(opt_b)]
    scaler.update.assert_called_once()


# -------------------------------------------------------------------------
# PhaseOptimizer.zero_grad — only the given key's optimizer is zeroed
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_zero_grad_net_key() -> None:
    opt_net, opt_fake = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": opt_net, "fake_score": opt_fake})
    po.zero_grad("net")

    opt_net.zero_grad.assert_called_once_with(set_to_none=True)
    opt_fake.zero_grad.assert_not_called()


@pytest.mark.L0
def test_zero_grad_fake_score_key() -> None:
    opt_net, opt_fake = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": opt_net, "fake_score": opt_fake})
    po.zero_grad("fake_score")

    opt_fake.zero_grad.assert_called_once_with(set_to_none=True)
    opt_net.zero_grad.assert_not_called()


@pytest.mark.L0
def test_zero_grad_optimizer_container_uses_container_method() -> None:
    container = _make_optimizer_container(_make_optimizer())
    po = PhaseOptimizer({"net": container})
    po.zero_grad("net")
    container.zero_grad.assert_called_once_with(set_to_none=True)


@pytest.mark.L0
def test_parameters_for_key_flattens_optimizer_container_param_groups() -> None:
    param_a = MagicMock()
    param_b = MagicMock()
    opt_a, opt_b = _make_optimizer(), _make_optimizer()
    opt_a.param_groups = [{"params": [param_a]}]
    opt_b.param_groups = [{"params": [param_b]}]
    po = PhaseOptimizer({"net": _make_optimizer_container(opt_a, opt_b)})
    assert po.parameters_for_key("net") == [param_a, param_b]


@pytest.mark.L0
def test_parameters_for_key_accepts_structurally_compatible_foreign_container() -> None:
    param_a = MagicMock()
    param_b = MagicMock()
    opt_a, opt_b = _make_optimizer(), _make_optimizer()
    opt_a.param_groups = [{"params": [param_a]}]
    opt_b.param_groups = [{"params": [param_b]}]
    po = PhaseOptimizer({"net": _ForeignOptimizerContainer(opt_a, opt_b)})

    assert po.parameters_for_key("net") == [param_a, param_b]


# -------------------------------------------------------------------------
# PhaseOptimizer.get
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_get_existing_key() -> None:
    opt_net = _make_optimizer()
    po = PhaseOptimizer({"net": opt_net})
    assert po.get("net") is opt_net


@pytest.mark.L0
def test_get_missing_key_returns_none() -> None:
    po = PhaseOptimizer({"net": _make_optimizer()})
    assert po.get("missing") is None


# -------------------------------------------------------------------------
# PhaseOptimizer.items — returns all entries
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_optimizer_items_returns_all_entries() -> None:
    opt_net, opt_fake = _make_optimizer(), _make_optimizer()
    po = PhaseOptimizer({"net": opt_net, "fake_score": opt_fake})
    items = dict(po.items())
    assert items["net"] is opt_net
    assert items["fake_score"] is opt_fake


# -------------------------------------------------------------------------
# PhaseScheduler.step — only the given key's scheduler is stepped
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_scheduler_step_net_key() -> None:
    sched_net, sched_fake = _make_scheduler(), _make_scheduler()
    ps = PhaseScheduler({"net": sched_net, "fake_score": sched_fake})
    ps.step("net")

    sched_net.step.assert_called_once()
    sched_fake.step.assert_not_called()


@pytest.mark.L0
def test_scheduler_step_fake_score_key() -> None:
    sched_net, sched_fake = _make_scheduler(), _make_scheduler()
    ps = PhaseScheduler({"net": sched_net, "fake_score": sched_fake})
    ps.step("fake_score")

    sched_fake.step.assert_called_once()
    sched_net.step.assert_not_called()


# -------------------------------------------------------------------------
# PhaseScheduler.get
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_scheduler_get_existing_key() -> None:
    sched_net = _make_scheduler()
    ps = PhaseScheduler({"net": sched_net})
    assert ps.get("net") is sched_net


@pytest.mark.L0
def test_scheduler_get_missing_key_returns_none() -> None:
    ps = PhaseScheduler({"net": _make_scheduler()})
    assert ps.get("missing") is None


# -------------------------------------------------------------------------
# PhaseScheduler.items — returns all entries
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_scheduler_items_returns_all_entries() -> None:
    sched_net, sched_fake = _make_scheduler(), _make_scheduler()
    ps = PhaseScheduler({"net": sched_net, "fake_score": sched_fake})
    items = dict(ps.items())
    assert items["net"] is sched_net
    assert items["fake_score"] is sched_fake


# -------------------------------------------------------------------------
# Single-key construction (base class scenario: net only)
# -------------------------------------------------------------------------
@pytest.mark.L0
def test_single_key_optimizer() -> None:
    opt_net = _make_optimizer()
    po = PhaseOptimizer({"net": opt_net})
    scaler = _make_grad_scaler()
    po.step("net", scaler)
    po.zero_grad("net")
    scaler.step.assert_called_once_with(opt_net)
    opt_net.zero_grad.assert_called_once_with(set_to_none=True)


@pytest.mark.L0
def test_single_key_scheduler() -> None:
    sched_net = _make_scheduler()
    ps = PhaseScheduler({"net": sched_net})
    ps.step("net")
    sched_net.step.assert_called_once()


@pytest.mark.L0
@pytest.mark.CPU
def test_public_optimizer_exports_are_explicit() -> None:
    assert optimizer_module.__all__ == (
        "OptimizerContainerLike",
        "OptimizerModelView",
        "PhaseOptimizer",
        "PhaseScheduler",
        "is_optimizer_container",
        "iter_torch_optimizers",
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("phase_key", ["net", "fake_score"])
def test_phase_optimizer_and_scheduler_route_the_same_key(phase_key: str) -> None:
    optimizers = {"net": _make_optimizer(), "fake_score": _make_optimizer()}
    schedulers = {"net": _make_scheduler(), "fake_score": _make_scheduler()}
    phase_optimizer = PhaseOptimizer(optimizers)
    phase_scheduler = PhaseScheduler(schedulers)
    grad_scaler = _make_grad_scaler()

    phase_optimizer.step(phase_key, grad_scaler)
    phase_scheduler.step(phase_key)

    grad_scaler.step.assert_called_once_with(optimizers[phase_key])
    schedulers[phase_key].step.assert_called_once_with()
