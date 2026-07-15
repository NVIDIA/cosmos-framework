# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections import defaultdict
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import cosmos_framework.trainer.distillation as trainer_module
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.model.generator.distillation.optimizer import PhaseOptimizer, PhaseScheduler
from cosmos_framework.trainer.distillation import DistillationTrainer


def _make_config(grad_accum_iter: int = 1, distributed_parallelism: str = "ddp") -> SimpleNamespace:
    straggler = SimpleNamespace(
        analyze_forward=False,
        analyze_backward=False,
        analyze_optimizer=False,
        analyze_dataloading=False,
        enabled=False,
        report_freq=10,
        profile_freq=10,
        max_diff=0.2,
        raise_error=False,
        save_s3=False,
    )
    trainer = SimpleNamespace(
        grad_accum_iter=grad_accum_iter,
        distributed_parallelism=distributed_parallelism,
        straggler_detection=straggler,
    )
    return SimpleNamespace(trainer=trainer)


def _make_trainer(grad_accum_iter: int = 1, distributed_parallelism: str = "ddp") -> DistillationTrainer:
    trainer = object.__new__(DistillationTrainer)
    trainer.config = _make_config(grad_accum_iter=grad_accum_iter, distributed_parallelism=distributed_parallelism)
    trainer.callbacks = MagicMock()
    trainer.training_timer = MagicMock()
    trainer.training_timer.return_value = MagicMock(
        __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
    )
    trainer.straggler_detector = MagicMock()
    trainer.straggler_detector.profile_section.return_value = MagicMock(
        __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
    )
    return trainer


def _make_optimizer() -> PhaseOptimizer:
    opt_net = MagicMock()
    return PhaseOptimizer({"net": opt_net, "fake_score": MagicMock()})


def _make_scheduler() -> PhaseScheduler:
    return PhaseScheduler({"net": MagicMock(), "fake_score": MagicMock()})


def _make_model(student_phase: bool = True) -> MagicMock:
    model = MagicMock()
    model.get_phase.return_value = "student" if student_phase else "critic"
    return model


def _make_grad_scaler() -> MagicMock:
    scaler = MagicMock(spec=torch.amp.GradScaler)
    scaler.scale.side_effect = lambda x: x
    return scaler


def _make_optimizer_container(*optimizers: torch.optim.Optimizer) -> OptimizersContainer:
    container = object.__new__(OptimizersContainer)
    container.optimizers = list(optimizers)
    return container


class _FakeMasterWeightOptimizer:
    """Minimal FusedAdam-like object for testing eager state allocation."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        capturable: bool = True,
        master_weights: bool = True,
    ) -> None:
        self.param_groups: list[dict[str, object]] = [
            {
                "params": params,
                "lr": lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        ]
        self.state: defaultdict[torch.nn.Parameter, dict[str, torch.Tensor]] = defaultdict(dict)
        self.capturable: bool = capturable
        self.master_weights: bool = master_weights
        self.param_groups_master: list[dict[str, list[torch.Tensor | None]]] | None = None
        self.step_called: bool = False

    def step(self) -> None:
        self.step_called = True
        raise AssertionError("eager init must not call optimizer.step()")


@pytest.mark.L0
@pytest.mark.CPU
def test_public_trainer_exports_are_explicit() -> None:
    assert trainer_module.__all__ == ("DistillationTrainer",)


# ---------------------------------------------------------------------------
# Class-level checks
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_is_subclass_of_imaginaire_trainer() -> None:
    assert issubclass(DistillationTrainer, ImaginaireTrainer)


@pytest.mark.L0
def test_does_not_define_train() -> None:
    assert "train" not in DistillationTrainer.__dict__


@pytest.mark.L0
def test_defines_training_step_for_closures() -> None:
    assert "training_step" in DistillationTrainer.__dict__


@pytest.mark.L0
def test_does_not_define_validate() -> None:
    assert "validate" not in DistillationTrainer.__dict__


# ---------------------------------------------------------------------------
# _optimizer_step — student phase routes to "net"
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_optimizer_step_student_calls_optimizer_step_net() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    model = _make_model(student_phase=True)

    trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=0)

    grad_scaler.step.assert_called_once_with(optimizer.get("net"))
    grad_scaler.update.assert_called_once()
    grad_scaler.step.assert_called_once()  # only once total — not for fake_score


@pytest.mark.L0
def test_optimizer_step_student_calls_scheduler_step_net() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    model = _make_model(student_phase=True)

    trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=0)

    scheduler.get("net").step.assert_called_once()
    scheduler.get("fake_score").step.assert_not_called()


# ---------------------------------------------------------------------------
# _optimizer_step — critic phase routes to "fake_score"
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_optimizer_step_critic_calls_optimizer_step_fake_score() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    model = _make_model(student_phase=False)

    trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=1)

    grad_scaler.step.assert_called_once_with(optimizer.get("fake_score"))
    grad_scaler.update.assert_called_once()


@pytest.mark.L0
def test_optimizer_step_critic_calls_scheduler_step_fake_score() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    model = _make_model(student_phase=False)

    trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=1)

    scheduler.get("fake_score").step.assert_called_once()
    scheduler.get("net").step.assert_not_called()


# ---------------------------------------------------------------------------
# _optimizer_step — passes iteration to get_phase
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_optimizer_step_passes_iteration_to_get_phase() -> None:
    trainer = _make_trainer()
    model = _make_model(student_phase=True)
    trainer._optimizer_step(model, _make_optimizer(), _make_scheduler(), _make_grad_scaler(), iteration=42)
    model.get_phase.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# _zero_grad — routes to correct key
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_zero_grad_student_zeroes_net() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    model = _make_model(student_phase=True)

    trainer._zero_grad(model, optimizer, iteration=5)

    optimizer.get("net").zero_grad.assert_called_once_with(set_to_none=True)
    optimizer.get("fake_score").zero_grad.assert_not_called()


@pytest.mark.L0
def test_zero_grad_critic_zeroes_fake_score() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    model = _make_model(student_phase=False)

    trainer._zero_grad(model, optimizer, iteration=6)

    optimizer.get("fake_score").zero_grad.assert_called_once_with(set_to_none=True)
    optimizer.get("net").zero_grad.assert_not_called()


@pytest.mark.L0
def test_zero_grad_passes_iteration_to_get_phase() -> None:
    trainer = _make_trainer()
    model = _make_model(student_phase=True)
    trainer._zero_grad(model, _make_optimizer(), iteration=11)
    model.get_phase.assert_called_once_with(11)


# ---------------------------------------------------------------------------
# Integration: hooks wired into the inherited training_step
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_eager_init_allocates_adam_state_when_empty() -> None:
    p = torch.nn.Parameter(torch.randn(3, 4))
    opt = torch.optim.AdamW([p], lr=1e-3)
    assert len(opt.state.get(p, {})) == 0
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    state = opt.state[p]
    assert "exp_avg" in state and "exp_avg_sq" in state
    assert torch.all(state["exp_avg"] == 0)
    assert torch.all(state["exp_avg_sq"] == 0)


@pytest.mark.L0
def test_eager_init_resets_step_counter_to_zero() -> None:
    p = torch.nn.Parameter(torch.randn(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    step_val = opt.state[p]["step"]
    step_val = step_val.item() if isinstance(step_val, torch.Tensor) else step_val
    assert step_val == 0


@pytest.mark.L0
def test_eager_init_does_not_change_param_values() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5))
    original = p.data.clone()
    opt = torch.optim.AdamW([p], lr=1e-3)
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert torch.equal(p.data, original)


@pytest.mark.L0
def test_eager_init_preserves_existing_grads() -> None:
    p = torch.nn.Parameter(torch.randn(3))
    sentinel_grad = torch.full_like(p.data, 7.0)
    p.grad = sentinel_grad
    opt = torch.optim.AdamW([p], lr=1e-3)
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert p.grad is sentinel_grad


@pytest.mark.L0
def test_eager_init_skips_when_state_already_populated() -> None:
    p = torch.nn.Parameter(torch.randn(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.ones_like(p.data)
    opt.step()  # populates state with non-zero exp_avg
    p.grad = None
    exp_avg_before = opt.state[p]["exp_avg"].clone()
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert torch.equal(opt.state[p]["exp_avg"], exp_avg_before)


@pytest.mark.L0
def test_eager_init_skips_params_without_requires_grad() -> None:
    p_train = torch.nn.Parameter(torch.randn(2))
    p_frozen = torch.nn.Parameter(torch.randn(2), requires_grad=False)
    opt = torch.optim.AdamW([p_train, p_frozen], lr=1e-3)
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert len(opt.state.get(p_train, {})) > 0
    assert len(opt.state.get(p_frozen, {})) == 0


@pytest.mark.L0
def test_eager_init_creates_master_weights_without_step() -> None:
    p = torch.nn.Parameter(torch.randn(3).to(torch.bfloat16))
    opt = _FakeMasterWeightOptimizer([p])
    original = p.detach().clone()
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert not opt.step_called
    assert torch.equal(p.data, original)
    assert opt.param_groups_master is not None
    master_param = opt.param_groups_master[0]["params"][0]
    assert master_param.dtype == torch.float32
    assert torch.equal(master_param, original.float())


@pytest.mark.L0
def test_eager_init_creates_fused_adam_group_state() -> None:
    p = torch.nn.Parameter(torch.randn(3))
    opt = _FakeMasterWeightOptimizer([p])
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    group = opt.param_groups[0]
    assert isinstance(group["step"], torch.Tensor)
    assert group["step"].shape == (1,)
    assert group["step"].item() == 0
    assert isinstance(group["lr"], torch.Tensor)


@pytest.mark.L0
def test_eager_init_allocates_missing_state_without_overwriting_existing_state() -> None:
    p_existing = torch.nn.Parameter(torch.randn(2))
    p_missing = torch.nn.Parameter(torch.randn(2))
    opt = _FakeMasterWeightOptimizer([p_existing, p_missing])
    opt.state[p_existing]["exp_avg"] = torch.ones_like(p_existing)
    DistillationTrainer._eager_init_optimizer_state(opt, "net")
    assert torch.equal(opt.state[p_existing]["exp_avg"], torch.ones_like(p_existing))
    assert "exp_avg" in opt.state[p_missing]
    assert "exp_avg_sq" in opt.state[p_missing]


@pytest.mark.L0
def test_eager_init_optimizer_container_initializes_inner_optimizers() -> None:
    p_a = torch.nn.Parameter(torch.randn(2))
    p_b = torch.nn.Parameter(torch.randn(2))
    opt_a = torch.optim.AdamW([p_a], lr=1e-3)
    opt_b = torch.optim.AdamW([p_b], lr=1e-3)
    DistillationTrainer._eager_init_optimizer_state(_make_optimizer_container(opt_a, opt_b), "net")
    assert "exp_avg" in opt_a.state[p_a]
    assert "exp_avg" in opt_b.state[p_b]


@pytest.mark.L0
def test_optimizer_step_runs_eager_init_only_once() -> None:
    trainer = _make_trainer()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    model = _make_model(student_phase=True)
    with patch.object(DistillationTrainer, "_eager_init_optimizer_state") as mock_init:
        trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=0)
        trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=1)
        trainer._optimizer_step(model, optimizer, scheduler, grad_scaler, iteration=2)
    # Called once per opt key on first invocation only (2 keys = 2 calls).
    assert mock_init.call_count == 2
    keys_initialized = sorted(call.args[1] for call in mock_init.call_args_list)
    assert keys_initialized == ["fake_score", "net"]


@pytest.mark.L0
def test_eager_init_after_first_step_matches_fresh_first_step() -> None:
    """After eager init + one AdamW step, weights should match a fresh-launch single step."""
    torch.manual_seed(0)
    p_a = torch.nn.Parameter(torch.randn(4))
    p_b = torch.nn.Parameter(p_a.data.clone())
    grad_value = torch.randn_like(p_a.data)

    # Path A: eager init, then real step.
    opt_a = torch.optim.AdamW([p_a], lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    DistillationTrainer._eager_init_optimizer_state(opt_a, "net")
    p_a.grad = grad_value.clone()
    opt_a.step()

    # Path B: fresh first step.
    opt_b = torch.optim.AdamW([p_b], lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    p_b.grad = grad_value.clone()
    opt_b.step()

    assert torch.allclose(p_a.data, p_b.data, atol=1e-6, rtol=1e-6)


@pytest.mark.L0
def test_hooks_called_via_inherited_training_step() -> None:
    """End-to-end: inherited training_step must invoke _optimizer_step and _zero_grad."""
    trainer = _make_trainer(grad_accum_iter=1)
    # In ddp mode, training_step resolves model_ddp.module as the model
    model_ddp = MagicMock()
    model_ddp.module.get_phase.return_value = "student"
    loss = torch.tensor(1.0, requires_grad=True)
    model_ddp.training_step.return_value = ({"x": torch.tensor(0.0)}, loss)
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    data = {"x": torch.zeros(1)}

    with patch("cosmos_framework.utils.distributed.ddp_sync_grad") as mock_ctx:
        mock_ctx.return_value = MagicMock(
            __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
        )
        trainer.training_step(model_ddp, optimizer, scheduler, grad_scaler, data, iteration=5, grad_accum_iter=0)

    # student phase → "net" key
    grad_scaler.step.assert_called_once_with(optimizer.get("net"))
    grad_scaler.update.assert_called_once()
    scheduler.get("net").step.assert_called_once()
    optimizer.get("net").zero_grad.assert_called_once_with(set_to_none=True)
    # grad_scaler.step must not be called directly by the trainer (PhaseOptimizer owns it)
    # — already verified above by assert_called_once (only PhaseOptimizer calls it once)


class _ClosureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.after_backward_calls: int = 0
        self.before_zero_grad_calls: int = 0
        self.zero_grad_calls: int = 0

    def get_phase(self, iteration: int) -> str:
        del iteration
        return "student"

    def zero_grad_for_phase(self, iteration: int) -> None:
        del iteration
        self.zero_grad_calls += 1

    def on_after_backward(self) -> None:
        self.after_backward_calls += 1

    def on_before_zero_grad(self, optimizer: object, scheduler: object, iteration: int) -> None:
        del optimizer, scheduler, iteration
        self.before_zero_grad_calls += 1

    def training_step_closures(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> Iterator[tuple[str, Callable[[], tuple[dict[str, torch.Tensor], torch.Tensor]], bool]]:
        del data, iteration

        def _closure(scale: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            loss = self.weight * scale  # []
            return {"loss": loss.detach()}, loss

        yield "chunk_0", lambda: _closure(1.0), False
        yield "chunk_1", lambda: _closure(2.0), True


class _FakeFSDPGradientSyncModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_sync_calls: list[tuple[bool, bool]] = []

    def set_requires_gradient_sync(self, enabled: bool, *, recurse: bool = True) -> None:
        self.gradient_sync_calls.append((enabled, recurse))


class _ClosureModelWithFSDP(_ClosureModel):
    def __init__(self) -> None:
        super().__init__()
        self.fake_fsdp = _FakeFSDPGradientSyncModule()


@pytest.mark.L0
def test_training_step_runs_model_closures_and_steps_once() -> None:
    trainer = _make_trainer(grad_accum_iter=1, distributed_parallelism="none")
    model = _ClosureModel()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    data = {"x": torch.zeros(1)}

    with patch("cosmos_framework.utils.distributed.ddp_sync_grad") as mock_ctx:
        mock_ctx.return_value = MagicMock(
            __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
        )
        output_batch, loss, grad_accum_iter = trainer.training_step(
            model,
            optimizer,
            scheduler,
            grad_scaler,
            data,
            iteration=5,
            grad_accum_iter=0,
        )

    assert grad_accum_iter == 0
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(output_batch["loss"], torch.tensor(3.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(3.0))
    assert model.after_backward_calls == 2
    assert model.before_zero_grad_calls == 1
    assert model.zero_grad_calls == 1
    grad_scaler.step.assert_called_once_with(optimizer.get("net"))
    scheduler.get("net").step.assert_called_once()


@pytest.mark.L0
def test_training_step_defers_fsdp2_gradient_sync_until_last_closure() -> None:
    trainer = _make_trainer(grad_accum_iter=1, distributed_parallelism="none")
    model = _ClosureModelWithFSDP()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    data = {"x": torch.zeros(1)}

    output_batch, loss, grad_accum_iter = trainer.training_step(
        model,
        optimizer,
        scheduler,
        grad_scaler,
        data,
        iteration=5,
        grad_accum_iter=0,
    )

    assert grad_accum_iter == 0
    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(output_batch["loss"], torch.tensor(3.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(3.0))
    assert model.fake_fsdp.gradient_sync_calls == [
        (False, False),
        (True, False),
        (True, False),
        (True, False),
    ]
    grad_scaler.step.assert_called_once_with(optimizer.get("net"))


@pytest.mark.L0
@pytest.mark.CPU
def test_training_step_accumulates_closures_across_two_microbatches() -> None:
    trainer = _make_trainer(grad_accum_iter=2, distributed_parallelism="none")
    model = _ClosureModelWithFSDP()
    optimizer = _make_optimizer()
    scheduler = _make_scheduler()
    grad_scaler = _make_grad_scaler()
    data = {"x": torch.zeros(1)}

    _, first_loss, grad_accum_iter = trainer.training_step(
        model,
        optimizer,
        scheduler,
        grad_scaler,
        data,
        iteration=5,
        grad_accum_iter=0,
    )

    assert grad_accum_iter == 1
    torch.testing.assert_close(first_loss, torch.tensor(3.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(1.5))
    grad_scaler.step.assert_not_called()
    scheduler.get("net").step.assert_not_called()

    _, second_loss, grad_accum_iter = trainer.training_step(
        model,
        optimizer,
        scheduler,
        grad_scaler,
        data,
        iteration=5,
        grad_accum_iter=grad_accum_iter,
    )

    assert grad_accum_iter == 0
    torch.testing.assert_close(second_loss, torch.tensor(3.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(3.0))
    grad_scaler.step.assert_called_once_with(optimizer.get("net"))
    scheduler.get("net").step.assert_called_once()
    assert model.fake_fsdp.gradient_sync_calls == [
        (False, False),
        (True, False),
        (False, False),
        (True, False),
        (False, False),
        (True, False),
        (True, False),
        (True, False),
    ]
