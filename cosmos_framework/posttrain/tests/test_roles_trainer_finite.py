# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Load-bearing CPU checks for atomic non-finite update outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.callbacks.base import Callback
from cosmos_framework.posttrain.entrypoints.dpo import _RejectedUpdateError, train_step
from cosmos_framework.posttrain.roles.trainer import (
    BaseTrainer,
    LLMTrainer,
    LocalTrainer,
    ObjectiveValidatingTrainerProtocol,
    TrainerProtocol,
)
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_gradients_finite


class _Policy:
    def __init__(self) -> None:
        self.net = torch.nn.Linear(2, 1, bias=False)


class _MovingPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self.net_ref = torch.nn.Linear(2, 1, bias=False)
        self.refresh_calls = 0

    def reference_semantics_identity(self) -> dict[str, int | str]:
        return {"adapter": "moving-policy-test", "version": 1}

    def refresh_ref(self) -> None:
        self.net_ref.load_state_dict(self.net.state_dict())
        self.net_ref.eval()
        self.refresh_calls += 1


class _CallbackProbe(Callback):
    def __init__(self) -> None:
        self.outcomes: list[str] = []
        self.ref_was_current: list[bool] = []

    def on_train_step(self, trainer: Any, metrics: dict[str, Any]) -> None:
        self.outcomes.append(metrics["optim/update_outcome"])
        self.ref_was_current.append(
            all(
                torch.equal(reference, policy)
                for reference, policy in zip(
                    trainer.model.net_ref.parameters(),
                    trainer.model.net.parameters(),
                )
            )
        )


class _DataProbe:
    def __init__(self) -> None:
        self.calls = 0

    def get_samples(self, count: int) -> list[int]:
        self.calls += 1
        return [self.calls] * count


class _RejectedPolicyProbe:
    def __init__(self) -> None:
        self.calls = 0

    def train(self, _batch: list[int]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 2:
            return {
                "train/loss_avg": 2.0,
                "optim/update_outcome": "skipped",
                "optim/skip_reason": "nonfinite_gradient",
            }
        return {"train/loss_avg": 1.0, "optim/update_outcome": "accumulating"}


class _AppliedPolicyProbe:
    def __init__(self) -> None:
        self.calls = 0

    def train(self, _batch: list[int]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "train/loss_avg": 1.0,
                "optim/update_outcome": "accumulating",
                "optim/intermediate_only": 1,
            }
        return {
            "train/loss_avg": 3.0,
            "optim/update_outcome": "applied",
            "optim/terminal_only": 1,
        }


class _BaseTrainerProbe(BaseTrainer):
    def _reduce_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return metrics


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_hierarchy_and_protocol_names_are_explicit() -> None:
    assert LLMTrainer.__bases__ == (BaseTrainer,)
    assert DiffusionTrainer.__bases__ == (BaseTrainer,)

    local = LocalTrainer(model=object())
    assert isinstance(local, TrainerProtocol)
    assert not isinstance(local, BaseTrainer)
    assert not isinstance(local, ObjectiveValidatingTrainerProtocol)

    from cosmos_framework.posttrain.roles import trainer as trainer_module

    assert not hasattr(trainer_module, "Trainer")
    assert not hasattr(trainer_module, "ObjectiveValidatingTrainer")


@pytest.mark.L0
@pytest.mark.CPU
def test_base_trainer_owns_checkpoint_manager_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    upload = {"bucket": "checkpoints", "prefix": "runs/test"}
    trainer = _BaseTrainerProbe(
        torch.optim.SGD(model.parameters(), lr=0.1),
        grad_accum=3,
        force_finite_grads=False,
        checkpoint_format="dcp",
        save_async=True,
        keep_optim_last=2,
        keep_model_last=4,
        upload=upload,
    )

    assert trainer.grad_accum == 3
    assert trainer.force_finite_grads is False
    assert trainer.checkpoint_format == "dcp"
    assert trainer.save_async is True
    assert trainer.keep_optim_last == 2
    assert trainer.keep_model_last == 4
    assert trainer.upload == upload
    assert trainer.upload is not upload

    trainer.checkpoint_format = "consolidated"
    assert trainer.checkpoint_format == "consolidated"
    with pytest.raises(ValueError, match="checkpoint_format"):
        trainer.checkpoint_format = "invalid"

    finalized: list[bool] = []
    monkeypatch.setattr(trainer.checkpoint_manager, "finalize", lambda: finalized.append(True))
    trainer.finalize_checkpoints()
    assert finalized == [True]


@pytest.mark.L0
@pytest.mark.CPU
def test_train_step_stops_at_rejection_and_returns_only_terminal_state() -> None:
    data = _DataProbe()
    rejected_policy = _RejectedPolicyProbe()
    with pytest.raises(_RejectedUpdateError) as rejected:
        train_step(data, rejected_policy, pairs_per_step=2, grad_accum=3)
    assert rejected.value.microbatch == 1
    assert rejected.value.metrics["optim/skip_reason"] == "nonfinite_gradient"
    assert data.calls == rejected_policy.calls == 2

    data = _DataProbe()
    metrics = train_step(data, _AppliedPolicyProbe(), pairs_per_step=2, grad_accum=2)
    assert metrics["train/loss_avg"] == 2.0
    assert metrics["optim/update_outcome"] == "applied"
    assert metrics["optim/terminal_only"] == 1
    assert "optim/intermediate_only" not in metrics


@pytest.mark.L0
@pytest.mark.CPU
def test_gradient_device_mismatch_is_a_setup_error() -> None:
    cpu_parameter = torch.nn.Parameter(torch.ones(1))  # [1]
    cpu_parameter.grad = torch.ones_like(cpu_parameter)  # [1]
    meta_parameter = torch.nn.Parameter(torch.ones(1, device="meta"))  # [1]
    meta_parameter.grad = torch.ones_like(meta_parameter)  # [1]

    with pytest.raises(RuntimeError, match="placement error, not a non-finite update"):
        all_ranks_gradients_finite([cpu_parameter, meta_parameter])


@pytest.mark.L0
@pytest.mark.CPU
def test_nonfinite_windows_are_atomic_and_control_state_resumes(tmp_path: Path) -> None:
    torch.manual_seed(13)
    policy = _Policy()
    optimizer = torch.optim.AdamW(policy.net.parameters(), lr=0.1)
    mode = {"loss": "finite"}

    def objective(model: _Policy, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = model.net.weight.square().sum()  # []
        if mode["loss"] == "nonfinite":
            loss = loss * torch.tensor(float("nan"))  # []
        return loss, {"dpo_loss": loss.detach()}

    trainer = DiffusionTrainer(policy, Objective(objective, Reduction.MEAN), optimizer, grad_accum=2)
    initial_weight = policy.net.weight.detach().clone()  # [1,2]

    assert trainer.train([])["optim/update_outcome"] == "accumulating"
    with pytest.raises(RuntimeError, match="completed update boundary"):
        trainer.save(str(tmp_path / "open-window"))
    mode["loss"] = "nonfinite"
    loss_skipped = trainer.train([])
    assert loss_skipped["optim/skip_reason"] == "nonfinite_loss"
    assert torch.equal(policy.net.weight, initial_weight)
    assert not optimizer.state_dict()["state"]
    assert all(parameter.grad is None for parameter in policy.net.parameters())

    mode["loss"] = "finite"
    assert trainer.train([])["optim/update_outcome"] == "accumulating"
    hook = policy.net.weight.register_hook(lambda gradient: gradient * float("nan"))  # [1,2]
    try:
        gradient_skipped = trainer.train([])
    finally:
        hook.remove()
    assert gradient_skipped["optim/skip_reason"] == "nonfinite_gradient"
    assert torch.equal(policy.net.weight, initial_weight)
    assert not optimizer.state_dict()["state"]

    assert trainer.train([])["optim/update_outcome"] == "accumulating"
    applied = trainer.train([])
    assert applied["optim/update_outcome"] == "applied"
    assert applied["optim/attempt_id"] == 3
    assert applied["optim/optimizer_step"] == 1
    assert applied["optim/nonfinite_skips_total"] == 2
    assert "optim/zero_grad_skips_total" not in applied
    assert not torch.equal(policy.net.weight, initial_weight)

    checkpoint = tmp_path / "iter_000001"
    trainer.save(str(checkpoint))
    restored_policy = _Policy()
    restored = DiffusionTrainer(
        restored_policy,
        Objective(objective, Reduction.MEAN),
        torch.optim.AdamW(restored_policy.net.parameters(), lr=0.1),
        grad_accum=2,
    )
    assert restored.load(str(checkpoint)) == 1
    restored_state = restored._trainer_state()
    assert restored_state["attempt_id"] == 3
    assert restored_state["nonfinite_skips_total"] == 2
    assert restored_state["nonfinite_skips_consecutive"] == 0
    assert "zero_grad_skips_total" not in restored_state  # disabled path keeps the legacy schema


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_boundary_always_uses_consumed_attempts() -> None:
    policy = _Policy()

    def objective(model: _Policy, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = model.net.weight.square().sum()  # []
        return loss, {}

    trainer = DiffusionTrainer(
        policy,
        Objective(objective, Reduction.MEAN),
        torch.optim.AdamW(policy.net.parameters(), lr=0.1),
    )
    trainer.n_mini = 2
    trainer._attempt_id = 2
    trainer._iter = 1
    trainer._nonfinite_skips_total = 1
    trainer._require_checkpoint_boundary()

    # One more successful attempt is the middle of the next collected batch. The optimizer clock
    # is now divisible by n_mini, so the old commit-based test would incorrectly permit this save.
    trainer._attempt_id = 3
    trainer._iter = 2
    with pytest.raises(RuntimeError, match=r"attempts=3, n_mini=2"):
        trainer._require_checkpoint_boundary()


@pytest.mark.L0
@pytest.mark.CPU
def test_abort_rewinds_attempt_opened_before_first_accumulation(tmp_path: Path) -> None:
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.net.parameters(), lr=0.1)
    fail_objective = {"value": True}

    def objective(model: _Policy, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if fail_objective["value"]:
            raise RuntimeError("injected objective failure")
        loss = model.net.weight.square().sum()  # []
        return loss, {"dpo_loss": loss.detach()}

    trainer = DiffusionTrainer(policy, Objective(objective, Reduction.MEAN), optimizer)
    with pytest.raises(RuntimeError, match="injected objective failure"):
        trainer.train([])
    assert trainer._attempt_id == 1
    assert trainer._accum == 0

    assert trainer.abort_accumulation()
    assert trainer._attempt_id == 0
    assert trainer._accum == 0
    assert not trainer.abort_accumulation()

    fail_objective["value"] = False
    applied = trainer.train([])
    assert applied["optim/update_outcome"] == "applied"
    assert applied["optim/attempt_id"] == 1
    assert trainer.iteration() == 1
    trainer.save(str(tmp_path / "iter_000001"))


@pytest.mark.L0
@pytest.mark.CPU
def test_moving_reference_and_callbacks_advance_only_after_applied_updates() -> None:
    policy = _MovingPolicy()
    callback = _CallbackProbe()
    finite = {"value": True}

    def objective(model: _MovingPolicy, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = model.net.weight.square().sum()  # []
        if not finite["value"]:
            loss = loss * torch.tensor(float("nan"))  # []
        return loss, {"dpo_loss": loss.detach()}

    trainer = DiffusionTrainer(
        policy,
        Objective(objective, Reduction.MEAN),
        torch.optim.SGD(policy.net.parameters(), lr=0.1),
        update_ref_every_iter=2,
        callbacks=[callback],
    )
    assert policy.refresh_calls == 1

    assert trainer.train([])["optim/update_outcome"] == "applied"
    finite["value"] = False
    assert trainer.train([])["optim/update_outcome"] == "skipped"
    finite["value"] = True
    assert trainer.train([])["optim/update_outcome"] == "applied"

    assert policy.refresh_calls == 2
    assert callback.outcomes == ["applied", "applied"]
    assert callback.ref_was_current == [False, True]
