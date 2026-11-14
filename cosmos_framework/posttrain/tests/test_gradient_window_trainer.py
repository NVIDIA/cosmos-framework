# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU optimizer/checkpoint lifecycle with an explicit simulated FSDP hook.

The simulation supplies one local native derivative to the production capture
boundary. It does not qualify CUDA FSDP or replace the required two-host test.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.roles.trainer.base import NonFiniteUpdateError, PartialUpdateError
from cosmos_framework.posttrain.roles.trainer.gradient_window import PROFILE, OwnedGradientWindow, Owner
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, x):
        return (x @ self.weight + self.frozen).square()


def build(*, enabled=True, fail_forward=False, **kwargs):
    model = Model()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01, weight_decay=0.1, foreach=False)
    window = None
    if enabled:
        window = OwnedGradientWindow(
            {"root": [Owner("weight", model.weight, (2,), 1, 1, ("weight",))]},
            microbatches=2,
            world_size=1,
            count_sum=lambda count: count,
        )

        def native_capture(gradient):
            window.hook("root")(gradient.detach().double().reshape(-1))
            return torch.zeros_like(gradient)

        model.weight.register_hook(native_capture)

    def forward(net, batch):
        if fail_forward:
            raise RuntimeError("controlled forward failure")
        x, count = batch
        numerator = net(x).sum() * (count > 0)
        loss = numerator / max(count, 1)
        return loss, {"flow_loss": float(loss.detach()), "flow_bc_masked_elements": count, OBJECTIVE_WEIGHT_SUM: count}

    objective = Objective(
        forward,
        Reduction.TOKEN_SUM,
        identity={"algorithm": "flow_bc", "finite_training_padding": "zero_weight_v1"},
        metric_adapter=MetricAdapter(
            internal_prefix="flow_",
            phases=frozenset({"train", "val"}),
            names={"flow_loss": "loss_avg", "flow_bc_masked_elements": "valid_action_count"},
            ignored=frozenset({OBJECTIVE_WEIGHT_SUM}),
        ),
    )
    trainer = LLMTrainer(
        model,
        objective,
        optimizer,
        grad_accum=2,
        max_grad_norm=1,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1)),
        require_gradients=True,
        gradient_window=window,
        **kwargs,
    )
    return trainer


BATCHES = [
    (torch.tensor([[1.0, 2.0], [2.0, -1.0], [3.0, 0.0]]), 3),
    (torch.tensor([[4.0, 5.0]]), 0),
    (torch.tensor([[2.0, 1.0]]), 1),
    (torch.tensor([[3.0, 1.0], [-1.0, 2.0]]), 2),
]


def assert_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree(a, b)
    else:
        assert actual == expected


def test_public_trainer_two_updates_match_independent_objective_and_resume(tmp_path):
    trainer = build()
    reference = Model()
    independent = torch.optim.AdamW([reference.weight], lr=0.01, weight_decay=0.1, foreach=False)
    schedule = torch.optim.lr_scheduler.LambdaLR(independent, lambda step: 1 / (step + 1))
    initial_frozen = trainer.model.frozen.detach().clone()
    checkpoint = tmp_path / "iter_000001"
    for step in range(2):
        derivatives, counts = [], []
        for x, count in BATCHES[2 * step : 2 * step + 2]:
            numerator = reference(x).sum() * (count > 0)
            derivatives.append(torch.autograd.grad(numerator, reference.weight)[0].double())
            counts.append(count)
        gradient = (sum(derivatives) / sum(counts)).float()
        norm = math.sqrt(math.fsum(float(value) ** 2 for value in gradient))
        coefficient = torch.tensor(min(1.0, 1 / (norm + 1e-6)), dtype=torch.float32)
        reference.weight.grad = gradient * coefficient
        independent.step()
        independent.zero_grad(set_to_none=True)
        schedule.step()
        for micro, batch in enumerate(BATCHES[2 * step : 2 * step + 2]):
            metrics = trainer.train(batch)
            assert metrics["optim/update_outcome"] == ("accumulating" if micro == 0 else "applied")
        assert torch.equal(trainer.model.weight, reference.weight)
        assert_tree(trainer.opt.state[trainer.model.weight], independent.state[reference.weight])
        assert trainer.gradient_window.completed_updates == trainer.iteration() == step + 1
        assert trainer.gradient_window.phase == "idle" and not trainer.gradient_window.buffers
        assert torch.equal(trainer.model.frozen, initial_frozen)
        if step == 0:
            trainer.save(str(checkpoint), data_state={"cursor": 3})
    resumed = build()
    assert resumed.load(str(checkpoint)) == 1
    assert resumed.gradient_window.completed_updates == 1
    for batch in BATCHES[2:]:
        resumed.train(batch)
    assert torch.equal(resumed.model.weight, trainer.model.weight)
    assert_tree(resumed.opt.state_dict(), trainer.opt.state_dict())
    assert_tree(resumed.lr_scheduler.state_dict(), trainer.lr_scheduler.state_dict())
    assert_tree(resumed.gradient_window.state_dict(), trainer.gradient_window.state_dict())


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "extra",
        "foreign",
        "clock",
        "bool_clock",
        "partial",
        "legacy_diagnostic",
        "float_world",
        "bool_shards",
    ],
)
def test_checkpoint_policy_mismatch_is_rejected_before_mutation(failure):
    trainer = build()
    state = trainer._trainer_state()
    original_weight = trainer.model.weight.detach().clone()
    original_optimizer = copy.deepcopy(trainer.opt.state_dict())
    if failure == "missing":
        del state["gradient_policy_state"]
    elif failure == "extra":
        state["gradient_policy_state"]["extra"] = 1
    elif failure == "foreign":
        state["gradient_policy_state"]["identity"]["profile"] = "different"
    elif failure == "clock":
        state["gradient_policy_state"]["completed_updates"] = 1
    elif failure == "bool_clock":
        state["gradient_policy_state"]["completed_updates"] = False
    elif failure == "partial":
        state["gradient_policy_state"]["phase"] = "open"
    elif failure == "float_world":
        state["gradient_policy_state"]["identity"]["world_size"] = 1.0
    elif failure == "bool_shards":
        state["gradient_policy_state"]["identity"]["groups"]["root"][0]["shards"] = True
    else:
        del state["gradient_policy_state"]["identity"]["integration"]
    with pytest.raises(ValueError, match="gradient-policy"):
        trainer._restore_trainer_state_local(state)
    assert torch.equal(trainer.model.weight, original_weight)
    assert_tree(trainer.opt.state_dict(), original_optimizer)
    assert trainer.iteration() == trainer.gradient_window.completed_updates == 0


def test_legacy_resume_and_warm_start_are_distinct(tmp_path):
    legacy = build(enabled=False)
    state = legacy._trainer_state()
    assert "gradient_policy_state" not in state and "gradient_policy" not in state["objective_identity"]
    configured = build()
    with pytest.raises(ValueError, match="gradient-policy"):
        configured._restore_trainer_state_local(state)
    with pytest.raises(ValueError, match="gradient policy"):
        legacy._restore_trainer_state_local(configured._trainer_state())
    # A fresh weight-only initialization is allowed; optimizer and policy clock
    # remain fresh. This is deliberately not a call to the resume interface.
    with torch.no_grad():
        configured.model.weight.copy_(legacy.model.weight)
    assert configured.gradient_window.completed_updates == 0 and not configured.opt.state


def test_interrupted_forward_invalidates_until_complete_restore():
    trainer = build(fail_forward=True)
    state = trainer._trainer_state()
    with pytest.raises(RuntimeError, match="controlled forward failure"):
        trainer.train(BATCHES[0])
    with pytest.raises(RuntimeError, match="invalidated"):
        trainer.gradient_window.state_dict()
    with pytest.raises(RuntimeError, match="invalidated"):
        trainer.train(BATCHES[0])
    trainer._restore_trainer_state_local(state)
    assert trainer.gradient_window.failure is None and trainer.gradient_window.phase == "idle"


def test_explicit_abort_discards_only_an_uncommitted_window():
    trainer = build()
    trainer.train(BATCHES[0])
    assert trainer.gradient_window.buffers and trainer._accum == 1
    trainer.abort_update_window()
    assert not trainer.gradient_window.buffers and trainer._accum == 0
    assert trainer.gradient_window.completed_updates == 0 and not trainer.opt.state
    for batch in BATCHES[:2]:
        trainer.train(batch)
    assert trainer.iteration() == trainer.gradient_window.completed_updates == 1


def test_explicit_abort_cannot_clear_a_failed_policy():
    trainer = build(fail_forward=True)
    with pytest.raises(RuntimeError, match="controlled forward"):
        trainer.train(BATCHES[0])
    trainer.abort_update_window()
    with pytest.raises(RuntimeError, match="invalidated"):
        trainer.gradient_window.state_dict()


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


@pytest.mark.parametrize("bad_micro", [0, 1])
@pytest.mark.parametrize("policy", ["skip", "fail_stop"])
def test_nonfinite_native_derivative_uses_existing_skip_or_fail_stop(tmp_path, bad_micro, policy):
    trainer = build(nonfinite_update=policy)
    before = trainer.model.weight.detach().clone()
    scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    forward = trainer.model.forward
    poison = {"enabled": False}

    def controlled_forward(x):
        value = forward(x)
        return _FiniteForwardInfiniteBackward.apply(value) if poison["enabled"] else value

    trainer.model.forward = controlled_forward
    if bad_micro == 1:
        assert trainer.train(BATCHES[0])["optim/update_outcome"] == "accumulating"
    poison["enabled"] = True
    # This is a numerical derivative failure with a finite objective, not an
    # invalid source/shape/owner. Every uncommitted microbatch is discarded.
    assert torch.isfinite(trainer.model(BATCHES[2][0])).all()
    if policy == "skip":
        result = trainer.train(BATCHES[2])
        assert result["optim/update_outcome"] == "skipped"
        assert result["optim/skip_reason"] == "nonfinite_gradient"
        assert result["optim/nonfinite_skips_total"] == 1
        assert trainer._skip_remaining == 1 - bad_micro
        if bad_micro == 0:
            assert trainer.train(BATCHES[1])["optim/update_outcome"] == "discarded"
        assert trainer._attempt_id == 1
    else:
        with pytest.raises(NonFiniteUpdateError) as rejected:
            trainer.train(BATCHES[2])
        assert rejected.value.reason == "nonfinite_gradient"
        assert trainer._nonfinite_skips_total == trainer._skip_remaining == trainer._attempt_id == 0
    assert trainer.gradient_window.failure is None and trainer._partial_update_error is None
    assert trainer.gradient_window.phase == "idle" and not trainer.gradient_window.buffers
    assert trainer.iteration() == trainer.gradient_window.completed_updates == trainer._accum == 0
    assert not trainer.opt.state and torch.equal(trainer.model.weight, before)
    assert_tree(trainer.lr_scheduler.state_dict(), scheduler)
    checkpoint = str(tmp_path / "rejected-boundary")
    trainer.save(checkpoint, data_state={"rejected_update": True})
    restored = build(nonfinite_update=policy)
    assert restored.load(checkpoint) == 0
    assert restored._nonfinite_skips_total == trainer._nonfinite_skips_total
    poison["enabled"] = False
    for batch in BATCHES[:2]:
        trainer.train(batch)
        restored.train(batch)
    assert trainer.iteration() == restored.iteration() == 1
    assert torch.equal(trainer.model.weight, restored.model.weight)
    assert_tree(trainer.opt.state_dict(), restored.opt.state_dict())


@pytest.mark.parametrize("policy", ["skip", "fail_stop"])
def test_nonfinite_final_cast_uses_numerical_rejection(monkeypatch, policy):
    trainer = build(nonfinite_update=policy)
    original = trainer.gradient_window.finalize

    def overflow_cast():
        # A finite FP64 accumulator can overflow the final FP32 materialization.
        trainer.gradient_window.buffers["weight"].fill_(1e300)
        original()

    monkeypatch.setattr(trainer.gradient_window, "finalize", overflow_cast)
    trainer.train(BATCHES[0])
    if policy == "skip":
        result = trainer.train(BATCHES[2])
        assert result["optim/update_outcome"] == "skipped" and result["optim/skip_reason"] == "nonfinite_gradient"
    else:
        with pytest.raises(NonFiniteUpdateError):
            trainer.train(BATCHES[2])
    assert trainer.gradient_window.failure is None and trainer._partial_update_error is None
    assert not trainer.opt.state and trainer.iteration() == 0 and trainer._accum == 0


def test_post_optimizer_failure_keeps_existing_trainer_poison(monkeypatch):
    trainer = build()
    trainer.train(BATCHES[0])
    monkeypatch.setattr(trainer.lr_scheduler, "step", lambda: (_ for _ in ()).throw(RuntimeError("scheduler failed")))
    with pytest.raises(PartialUpdateError, match="scheduler"):
        trainer.train(BATCHES[1])
    assert trainer._partial_update_error is not None and trainer.gradient_window.failure is not None
    with pytest.raises(PartialUpdateError):
        trainer.train(BATCHES[2])
    with pytest.raises(RuntimeError):
        trainer.gradient_window.state_dict()


def test_midwindow_and_empty_tail_cannot_publish_checkpoint():
    trainer = build()
    trainer.train(BATCHES[0])
    with pytest.raises(RuntimeError, match="boundary"):
        trainer.gradient_window.state_dict()
    empty = (BATCHES[0][0], 0)
    trainer = build()
    original = trainer.model.weight.detach().clone()
    trainer.train(empty)
    with pytest.raises(RuntimeError, match="only padding"):
        trainer.train(empty)
    assert not trainer.opt.state and torch.equal(trainer.model.weight, original)
    assert trainer.iteration() == 0 and trainer.gradient_window.failure is not None


@pytest.mark.parametrize("operation", ["offload", "prepare_weight_sync"])
def test_residency_changes_are_rejected(operation):
    with pytest.raises(RuntimeError, match="owned FP64 gradient"):
        getattr(build(), operation)()


def test_autocast_rejected_before_forward_or_backward():
    trainer = build()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError, match="autocast"):
        trainer.train(BATCHES[0])
    assert not trainer.opt.state


def test_uneven_shards_share_checkpoint_identity_but_keep_local_layout():
    first = OwnedGradientWindow(
        {"g": [Owner("p", torch.nn.Parameter(torch.ones(2)), (3,), 2, 2)]}, microbatches=1, world_size=4
    )
    last = OwnedGradientWindow(
        {"g": [Owner("p", torch.nn.Parameter(torch.ones(1)), (3,), 2, 2)]}, microbatches=1, world_size=4
    )
    assert first.owners[0].local_shape != last.owners[0].local_shape
    assert first.identity() == last.identity()
    last.load_state_dict(first.state_dict())


def config(**kwargs):
    values = {
        "task": "flow_bc",
        "adapters": AdapterConfig(
            components=[AdapterComponentConfig(component="language", targets=["model.layers.*.q_proj"])]
        ),
        "gradient_policy": PROFILE,
        "optimizer": OptimizerConfig(grad_clip_norm=1.0),
    }
    values.update(kwargs)
    return SFTConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gradient_policy": "unknown"},
        {"gradient_policy": False},
        {"task": "conversation"},
        {"adapters": None},
        {"optimizer": OptimizerConfig(force_finite_grads=False)},
        {"optimizer": OptimizerConfig(grad_clip_norm=0)},
    ],
)
def test_public_configuration_rejects_unsupported_policy(kwargs):
    with pytest.raises((ValueError, TypeError)):
        config(**kwargs)


def test_public_configuration_is_opt_in():
    assert SFTConfig().gradient_policy is None
    assert config().gradient_policy == PROFILE


@pytest.mark.parametrize("foreign", ["hf", "alpamayo", "gr00t", "qwen3_vl"])
def test_public_structure_rejects_other_families(monkeypatch, foreign):
    # Even GR00T's unregistered build_policy alias is not an admitted SFT builder.
    from cosmos_framework.posttrain.config import build as builders
    from cosmos_framework.posttrain.entrypoints import sft
    from cosmos_framework.utils import lazy_config

    monkeypatch.setattr(sft, "sft_policy_source", lambda config: ("fixture", False))
    monkeypatch.setattr(lazy_config, "instantiate", lambda value: value)
    monkeypatch.setattr(
        builders,
        "select_policy_backend",
        lambda config: SimpleNamespace(
            native_spec=None
            if foreign == "hf"
            else SimpleNamespace(builder=f"cosmos_framework.posttrain.model.{foreign}.sft.build_policy")
        ),
    )
    root = SimpleNamespace(grid=SimpleNamespace(backend="local", grouping="single"), rl=config())
    with pytest.raises(ValueError, match="gradient-window factory"):
        sft.validate_config_structure(root)
