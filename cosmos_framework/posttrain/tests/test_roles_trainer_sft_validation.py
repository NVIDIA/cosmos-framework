# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A held-out objective call preserves training modes, RNG, gradients and optimizer."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.roles.trainer.validation import sft as validation
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("failure", [None, "objective", "reshard", "eval"])
def test_restores_rng_modes_and_optimizer_even_on_failure(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    packer = MemoryPacker()
    batch = sample_batch(packer)
    policy = Policy()
    objective = make_objective(packer)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    loss, _ = objective(policy, batch)
    loss.backward()
    optimizer.step()  # Materialize optimizer state; leave existing gradients intact.
    policy.train()
    policy.net.dropout.eval()
    modes = [(module, module.training) for module in policy.modules()]
    parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    gradients = [parameter.grad.clone() for parameter in policy.parameters()]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    observed: dict[str, Any] = {}

    def stochastic_objective(model: Any, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        random.random()
        np.random.rand()
        torch.rand(7)
        if failure == "objective":
            raise RuntimeError("objective failed")
        result = objective(model, trajectories)
        observed.update(result[1])
        return result

    if failure == "reshard":

        def broken_reshard(roots: dict[str, torch.nn.Module]) -> None:
            assert roots
            raise RuntimeError("reshard failed")

        monkeypatch.setattr(validation, "reshard_fsdp_modules", broken_reshard)
    elif failure == "eval":

        def broken_eval() -> None:
            policy.training = False
            torch.rand(3)
            raise RuntimeError("eval failed")

        monkeypatch.setattr(policy, "eval", broken_eval)

    evaluated = Objective(stochastic_objective, reduction=Reduction.TOKEN_SUM)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            validation.evaluate_sft_batch(policy, evaluated, batch)
    else:
        metrics = validation.evaluate_sft_batch(policy, evaluated, batch)
        assert metrics == observed
        assert "sft_loss" in metrics and not any(key.startswith("val/") for key in metrics)
        assert torch.isfinite(metrics["sft_loss"])
        assert not metrics["sft_loss"].requires_grad
    for module, mode in modes:
        assert module.training is mode
    _assert_equal([parameter.detach() for parameter in policy.parameters()], parameters)
    _assert_equal([parameter.grad for parameter in policy.parameters()], gradients)
    _assert_equal(optimizer.state_dict(), optimizer_state)
    _assert_equal((random.getstate(), np.random.get_state(), torch.get_rng_state()), rng_before)


def test_validation_uses_eval_loss_and_excludes_dummy_rows() -> None:
    packer = MemoryPacker()
    real = packer.add([1, 2, 3], [-100, 2, 3], weight=1.0)
    dummy = packer.add([4, 5, 6, 7], [-100, 5, 6, 7], weight=0.0)
    policy = Policy()
    policy.net.dropout.p = 0.7
    objective = make_objective(packer)
    policy.eval()
    _, expected = objective(policy, [real])
    policy.train()
    actual = validation.evaluate_sft_batch(policy, objective, [real, dummy])
    torch.testing.assert_close(actual["sft_loss"], expected["sft_loss"])
    assert actual["sft_valid_tokens"] == expected["sft_valid_tokens"]
    assert actual["sft_valid_sequences"] == expected["sft_valid_sequences"]
    assert all(module.training for module in policy.modules())


def test_non_module_adapter_restores_auxiliary_module_modes() -> None:
    from types import SimpleNamespace

    policy = Policy()
    auxiliary = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    auxiliary[0].eval()
    adapter = SimpleNamespace(net=policy, auxiliary=auxiliary)
    modes = [(module, module.training) for root in (policy, auxiliary) for module in root.modules()]

    def objective(model: Any, batch: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        assert batch == []
        assert all(not module.training for module, _ in modes)
        return model.auxiliary(torch.ones(1, 2)).sum(), {"observed": True}

    metrics = validation.evaluate_sft_batch(adapter, Objective(objective, reduction=Reduction.MEAN), [])
    assert metrics == {"observed": True}
    assert all(module.training == was_training for module, was_training in modes)
