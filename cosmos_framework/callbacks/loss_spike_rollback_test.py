# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the gradient-norm spike guard.

Natural spikes are not reproducible run to run, so the guard is exercised with an
injected gradient rather than by waiting for training to misbehave.
"""

from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from cosmos_framework.callbacks.loss_spike_rollback import LossSpikeRollback

pytestmark = pytest.mark.L0


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
    callback.on_before_optimizer_step(model, optimizer, scheduler, None)
    optimizer.step()
    scheduler.step()
    callback.on_before_zero_grad(model, optimizer, scheduler)
    optimizer.zero_grad(set_to_none=True)


def _step_at(model, optimizer, scheduler, callback, grad_scale, iteration):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, grad_scale)
    callback.on_before_optimizer_step(model, optimizer, scheduler, None, iteration=iteration)
    optimizer.step()
    scheduler.step()
    callback.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
    optimizer.zero_grad(set_to_none=True)


def _spike_at(model, optimizer, scheduler, callback, iteration):
    _step_at(model, optimizer, scheduler, callback, 10.0, iteration)


def _assert_reported_lrs(scheduler):
    rates = [group["lr"] for group in scheduler.optimizer.param_groups]
    assert scheduler.get_last_lr() == pytest.approx(rates)
    assert scheduler.state_dict()["_last_lr"] == pytest.approx(rates)


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


@pytest.mark.parametrize("decay", [1.0, 0.95])
def test_lr_backoff_applies_to_the_next_update(decay):
    """Use the trainer order: optimizer, scheduler, rollback, then zero_grad."""
    model, optimizer, _, callback = _harness()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: decay**step)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)

    assert callback.rollbacks == 1
    expected = 0.01 * decay**scheduler.last_epoch * 0.8 * 0.5
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    _assert_reported_lrs(scheduler)
    used_rates = []
    hook = optimizer.register_step_pre_hook(lambda opt, args, kwargs: used_rates.append(opt.param_groups[0]["lr"]))
    try:
        _step(model, optimizer, scheduler, callback, 0.01)
    finally:
        hook.remove()
    assert used_rates == pytest.approx([expected])
    # Healthy recovery updates the already-computed rate, not just base_lrs.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * decay**scheduler.last_epoch * callback._lr_scale)
    _assert_reported_lrs(scheduler)


@pytest.mark.parametrize("with_scheduler", [True, False])
def test_scaling_preserves_multiple_optimizers_groups_and_schedule(with_scheduler):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    optimizers = [
        torch.optim.AdamW([{"params": [model[0].weight], "lr": 0.02}, {"params": [model[0].bias], "lr": 0.005}]),
        torch.optim.AdamW(model[1].parameters(), lr=0.007),
    ]
    schedulers = [
        LambdaLR(optimizers[0], lr_lambda=[lambda step: 0.9**step, lambda step: 0.8**step]),
        LambdaLR(optimizers[1], lr_lambda=lambda step: 0.7**step),
    ]
    # Reporting must follow each scheduler's own optimizer, not container order.
    scheduler = SimpleNamespace(schedulers=list(reversed(schedulers))) if with_scheduler else None
    callback = LossSpikeRollback(enabled=True)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        if with_scheduler:
            sched.step()
    initial = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    counters = [(sched.last_epoch, sched._step_count) for sched in schedulers]
    for factor in (0.5, 1.02, 0.5, 1.02):
        callback._scale_lr(optimizers, scheduler, factor)
        for opt, rates in zip(optimizers, initial):
            assert [group["lr"] for group in opt.param_groups] == pytest.approx(
                [rate * callback._lr_scale for rate in rates]
            )
        assert [(sched.last_epoch, sched._step_count) for sched in schedulers] == counters
        if with_scheduler:
            for sched in schedulers:
                _assert_reported_lrs(sched)
    if with_scheduler:
        next_rates = [[0.02 * 0.9**2, 0.005 * 0.8**2], [0.007 * 0.7**2]]
        for opt, sched, rates in zip(optimizers, schedulers, next_rates):
            opt.step()
            sched.step()
            assert [group["lr"] for group in opt.param_groups] == pytest.approx(
                [rate * callback._lr_scale for rate in rates]
            )
            _assert_reported_lrs(sched)


@pytest.mark.parametrize("recover", [False, True])
def test_optimizer_and_scheduler_checkpoint_rates_agree(recover):
    model, optimizer, scheduler, callback = _harness()
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)
    if recover:
        _step(model, optimizer, scheduler, callback, 0.01)
    _assert_reported_lrs(scheduler)

    checkpoint = BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored_model, restored_optimizer, restored_scheduler, _ = _harness()
    restored_model.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    _assert_reported_lrs(restored_scheduler)

    # Check the next optimizer/scheduler update, without claiming to persist the
    # guard's separate rolling history or change its existing resume policy.
    for net, opt, sched in ((model, optimizer, scheduler), (restored_model, restored_optimizer, restored_scheduler)):
        for parameter in net.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        opt.step()
        sched.step()
        _assert_reported_lrs(sched)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for actual, expected in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.GPU
@pytest.mark.parametrize("kind", ["generator", "reasoner"])
def test_native_scheduler_container_reports_backoff_and_recovery(kind):
    model = nn.Sequential()
    model.add_module("net", nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)))
    model.cuda()
    if kind == "generator":
        from cosmos_framework.utils.generator.optimizer import LRSchedulersContainer
        from cosmos_framework.utils.generator.optimizer import OptimizersContainer as GeneratorOptimizersContainer

        optimizer = GeneratorOptimizersContainer(model, "adamw", fused=True, lr=0.01, lr_multipliers={"bias": 0.25})
        scheduler = LRSchedulersContainer(optimizer, "constant")
    else:
        from cosmos_framework.utils.reasoner.optimizer import OptimizersContainer as ReasonerOptimizersContainer
        from cosmos_framework.utils.reasoner.optimizer import SchedulersContainer

        optimizer = ReasonerOptimizersContainer(
            list(model.net),
            {"lr": 0.01, "weight_decay": 0.0, "fused": True},
            "adamw",
            {"first": 1.0, "second": 0.25},
            ["first", "second"],
        )
        scheduler = SchedulersContainer(optimizer, lr_lambda=lambda step: 0.95**step)
    assert [len(opt.param_groups) for opt in optimizer] == ([2] if kind == "generator" else [1, 1])
    callback = LossSpikeRollback(enabled=True, min_observations=5, window=10, rollback_depth=2)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    for gradient, expected_scale in ((10.0, 0.4), (0.01, 0.4 * 1.02)):
        _step(model, optimizer, scheduler, callback, gradient)
        assert callback.rollbacks == 1
        assert callback._lr_scale == pytest.approx(expected_scale)
        rates = [[group["lr"] for group in sched.optimizer.param_groups] for sched in scheduler.schedulers]
        for sched in scheduler.schedulers:
            _assert_reported_lrs(sched)
        assert scheduler.get_last_lr() == ([lr for group in rates for lr in group] if kind == "generator" else rates)
        assert scheduler.state_dict()["_last_lr"] == rates[0]


def test_zero_scale_remains_zero_during_recovery():
    _, optimizer, scheduler, callback = _harness(lr_min_scale=0.0, lr_backoff=0.0)
    callback._scale_lr(optimizer, scheduler, 0.5)
    _assert_reported_lrs(scheduler)
    counters = (scheduler.last_epoch, scheduler._step_count)
    callback._scale_lr(optimizer, scheduler, 1.02)
    assert optimizer.param_groups[0]["lr"] == 0.0
    _assert_reported_lrs(scheduler)
    assert (scheduler.last_epoch, scheduler._step_count) == counters


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
    control_model, control_optimizer, control_scheduler, _ = _harness()
    for gradient in [0.01] * 20 + [10.0]:
        _step(model, optimizer, scheduler, callback, gradient)
        for parameter in control_model.parameters():
            parameter.grad = torch.full_like(parameter, gradient)
        control_optimizer.step()
        control_scheduler.step()
        control_optimizer.zero_grad(set_to_none=True)
    assert callback.rollbacks == 0
    assert not callback._ring, "disabled guard should not pay the snapshot memory cost"
    assert scheduler.state_dict() == control_scheduler.state_dict()
    torch.testing.assert_close(optimizer.state_dict(), control_optimizer.state_dict(), rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), control_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
        callback.on_before_optimizer_step(model, optimizer, scheduler, None)
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


def test_recovery_waits_for_the_run_to_be_healthy_again():
    """A degraded run looks clean between spikes, and must not be handed its rate back.

    Observed in training: the rate climbed back to 0.627 of base while the median loss sat
    at 0.69, roughly five times its healthy value, and stayed there for 280 steps.
    """
    model, optimizer, scheduler, callback = _harness(recovery_health_factor=2.0)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)  # spike -> backoff
    suppressed = callback._lr_scale
    assert suppressed < callback._lr_ceiling

    # Sustained elevation well above healthy but below the spike threshold: no step trips
    # the guard, yet the run is plainly not well.
    for _ in range(80):
        _step(model, optimizer, scheduler, callback, 0.08)
    assert callback.rollbacks == 1, "elevation should not itself trip the guard"
    # A few recovery steps land before the window accumulates enough elevated samples for
    # the median to register the degradation; what matters is that recovery then stops
    # well short of the ceiling, where an ungated 1.02-per-step walk would have arrived.
    assert callback._lr_scale < suppressed * 1.15, (
        f"rate recovered to {callback._lr_scale} while the run was still degraded"
    )
    assert callback._lr_scale < 0.7 * callback._lr_ceiling, (
        f"rate reached {callback._lr_scale}, close to the ceiling {callback._lr_ceiling}"
    )


def test_recovery_resumes_once_gradients_return_to_normal():
    model, optimizer, scheduler, callback = _harness(recovery_health_factor=2.0)
    for _ in range(20):
        _step(model, optimizer, scheduler, callback, 0.01)
    _step(model, optimizer, scheduler, callback, 10.0)
    suppressed = callback._lr_scale
    for _ in range(80):  # genuinely healthy steps
        _step(model, optimizer, scheduler, callback, 0.01)
    assert callback._lr_scale > suppressed, "rate never recovered despite a healthy run"
