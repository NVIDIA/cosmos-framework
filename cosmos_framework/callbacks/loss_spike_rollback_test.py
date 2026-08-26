# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the gradient-norm spike guard.

Natural spikes are not reproducible run to run, so the guard is exercised with an
injected gradient rather than by waiting for training to misbehave.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from cosmos_framework.callbacks.loss_spike_rollback import LossSpikeRollback


def _harness(**kwargs):
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    defaults = dict(enabled=True, window=10, min_observations=5, rollback_depth=2, grad_norm_factor=10.0)
    defaults.update(kwargs)
    return model, optimizer, scheduler, LossSpikeRollback(**defaults)


def _step(model, optimizer, scheduler, callback, grad_scale):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, grad_scale)
    callback.on_after_backward(model)
    optimizer.step()
    callback.on_before_zero_grad(model, optimizer, scheduler)
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def test_no_rollback_on_steady_gradients():
    model, optimizer, scheduler, callback = _harness()
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    assert callback.rollbacks == 0
    assert scheduler.base_lrs == [0.01]


def test_rollback_restores_weights_and_backs_off_lr():
    model, optimizer, scheduler, callback = _harness()
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)

    # The ring holds the last `rollback_depth` snapshots; the guard restores the OLDEST.
    expected = {name: tensor.clone() for name, tensor in callback._ring[0]["params"].items()}

    _step(model, optimizer, scheduler, callback, 10.0)  # ~1000x the steady norm

    assert callback.rollbacks == 1, "guard did not fire on an injected spike"
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), expected[name])
    assert scheduler.base_lrs == [0.005], "learning rate was not halved on rollback"


def test_lr_backoff_survives_scheduler_step():
    """The whole point of rescaling base_lrs rather than param_group['lr'].

    LambdaLR recomputes lr = base_lrs * lambda(step) on every step, so a direct write to
    param_group['lr'] would be discarded on the next iteration and the backoff would be
    silently ineffective.
    """
    model, optimizer, scheduler, callback = _harness()
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)
    lr_right_after = optimizer.param_groups[0]["lr"]
    for _ in range(3):
        _step(model, optimizer, scheduler, callback, 0.01)
    assert optimizer.param_groups[0]["lr"] <= lr_right_after * 1.1
    assert optimizer.param_groups[0]["lr"] < 0.01, "backoff was undone by scheduler.step()"


def test_stands_down_after_sustained_divergence():
    """A run that has truly diverged must not be frozen by an unyielding guard."""
    model, optimizer, scheduler, callback = _harness(max_consecutive=3)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    for _ in range(10):
        _step(model, optimizer, scheduler, callback, 10.0)
    assert callback.rollbacks <= 3, f"guard never stood down ({callback.rollbacks} rollbacks)"


def test_disabled_is_inert():
    model, optimizer, scheduler, callback = _harness(enabled=False)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)
    assert callback.rollbacks == 0
    assert not callback._ring, "disabled guard should not pay the snapshot memory cost"
