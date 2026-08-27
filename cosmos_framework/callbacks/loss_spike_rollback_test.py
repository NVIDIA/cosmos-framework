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


def _step_at(model, optimizer, scheduler, callback, grad_scale, iteration):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, grad_scale)
    callback.on_after_backward(model, iteration=iteration)
    optimizer.step()
    callback.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _spike_at(model, optimizer, scheduler, callback, iteration):
    _step_at(model, optimizer, scheduler, callback, 10.0, iteration)


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
    # Backoff is ceiling-relative: ceiling ratchets 1.0 -> 0.8, then dips by 0.5.
    assert scheduler.base_lrs == [0.01 * 0.8 * 0.5], f"unexpected rate {scheduler.base_lrs}"


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


def test_baseline_does_not_chase_a_deteriorating_run():
    """The failure that let a real run escape the guard.

    Once the window fills with elevated norms the median rises, and a purely relative test
    then demands an ever larger spike to trip. Observed in training: the baseline drifted
    from ~0.4 to 9.03, so firing required a norm above 90 and the guard went quiet exactly
    when it was needed. The baseline is anchored to the healthiest scale the run has shown.
    """
    model, optimizer, scheduler, callback = _harness(baseline_inflation_cap=4.0)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    healthy = callback._baseline()
    assert healthy is not None

    # Feed a sustained elevation an order of magnitude above healthy, short of tripping.
    for _ in range(60):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.05)
        callback.on_after_backward(model)
        callback.on_before_zero_grad(model, optimizer, scheduler)

    inflated = callback._baseline()
    assert inflated <= 4.0 * healthy * 1.001, f"baseline inflated to {inflated} from {healthy}"


def test_repeated_rollbacks_ratchet_the_learning_rate_down():
    """A run that keeps tripping must not climb back to a rate it cannot hold."""
    model, optimizer, scheduler, callback = _harness(max_consecutive=100)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)

    ceilings = []
    for _ in range(3):
        _step(model, optimizer, scheduler, callback, 10.0)  # spike
        for _ in range(40):  # long clean stretch: recovery walks up to the ceiling
            _step(model, optimizer, scheduler, callback, 0.01)
        ceilings.append(callback._lr_ceiling)

    assert ceilings == sorted(ceilings, reverse=True), f"ceiling did not ratchet down: {ceilings}"
    assert ceilings[-1] < 1.0
    assert callback._lr_scale <= callback._lr_ceiling + 1e-9


def test_rollback_keeps_the_healthy_norm_window():
    """Clearing the window on rollback discards the only healthy reference available."""
    model, optimizer, scheduler, callback = _harness()
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    before = len(callback._norms)
    _step(model, optimizer, scheduler, callback, 10.0)
    assert callback.rollbacks == 1
    assert len(callback._norms) == before, "healthy gradient-norm history was discarded"


def test_clustered_rollbacks_do_not_compound_into_the_floor():
    """The failure that rescued a run from divergence but left it undertrained.

    Four rollbacks inside thirteen steps compounded 0.5**4 straight into the learning-rate
    floor; the run then spent its remaining 274 steps at a tenth of the intended rate and
    lost about ten points of accuracy. A burst of spikes is one episode, not four
    escalations, and the dip is measured from the ceiling rather than from wherever the
    scale happens to have landed.
    """
    model, optimizer, scheduler, callback = _harness(max_consecutive=100, backoff_cooldown=50)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)

    iteration = 0
    for burst in range(4):  # four spikes a few steps apart: one episode
        _spike_at(model, optimizer, scheduler, callback, iteration)
        iteration += 1
        for _ in range(3):  # clean steps repopulate the ring a rollback cleared
            _step_at(model, optimizer, scheduler, callback, 0.01, iteration)
            iteration += 1

    assert callback.rollbacks == 4, "every spike should still be rewound"
    assert callback._lr_scale > callback.lr_min_scale, (
        f"clustered rollbacks collapsed the rate to the floor ({callback._lr_scale})"
    )
    assert callback._lr_ceiling > 0.5, f"ceiling over-ratcheted within one episode ({callback._lr_ceiling})"


def test_separated_episodes_still_ratchet():
    """Spikes far apart are genuinely separate episodes and should each cost rate."""
    model, optimizer, scheduler, callback = _harness(max_consecutive=100, backoff_cooldown=10)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)

    ceilings = []
    for episode in range(3):
        it = episode * 100
        _spike_at(model, optimizer, scheduler, callback, it)
        ceilings.append(callback._lr_ceiling)
        for offset in range(1, 6):  # clean steps rebuild the ring for the next episode
            _step_at(model, optimizer, scheduler, callback, 0.01, it + offset)

    assert ceilings == sorted(ceilings, reverse=True) and ceilings[-1] < ceilings[0]
