# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import cosmos_framework.callbacks.grad_clip_distillation as grad_clip_module
from cosmos_framework.callbacks.grad_clip_distillation import GradClip
from cosmos_framework.model.generator.distillation.optimizer import PhaseOptimizer


class _PhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net: torch.nn.Linear = torch.nn.Linear(2, 1, bias=False)
        self.net_fake_score: torch.nn.Linear = torch.nn.Linear(2, 1, bias=False)
        self.config: SimpleNamespace = SimpleNamespace(grad_clip=True)
        self._distillation_parity_grad_metrics: dict[str, float] = {}
        self.optimizer_key_override: str | None = None

    def get_optimizer_key(self, iteration: int) -> str:
        if self.optimizer_key_override is not None:
            return self.optimizer_key_override
        return "net" if iteration % 5 == 0 else "fake_score"

    def is_image_batch(self, data_batch: dict[str, object]) -> bool:
        return bool(data_batch["is_image"])


class _PartialNorm:
    def __init__(self, local_value: float, global_value: float) -> None:
        self.local_value: float = local_value
        self.global_value: float = global_value
        self.full_tensor_calls: int = 0

    def __float__(self) -> float:
        return self.local_value

    def full_tensor(self) -> torch.Tensor:  # returns: []
        self.full_tensor_calls += 1
        return torch.tensor(self.global_value)  # []


def _callback() -> GradClip:
    callback = GradClip(clip_norm=1.0)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=100))
    return callback


@pytest.mark.L0
@pytest.mark.CPU
def test_public_grad_clip_exports_are_explicit() -> None:
    assert grad_clip_module.__all__ == ("GradClip",)


@pytest.mark.L0
@pytest.mark.CPU
def test_student_phase_clips_only_student_parameters() -> None:
    model = _PhaseModel()
    optimizer = PhaseOptimizer(
        {
            "net": torch.optim.SGD(model.net.parameters(), lr=0.1),
            "fake_score": torch.optim.SGD(model.net_fake_score.parameters(), lr=0.1),
        }
    )
    model.net.weight.grad = torch.tensor([[3.0, 4.0]])  # [1,2]
    model.net_fake_score.weight.grad = torch.tensor([[0.0, 12.0]])  # [1,2]
    critic_grad_before = model.net_fake_score.weight.grad.clone()  # [1,2]
    callback = _callback()
    callback.on_training_step_start(model, {"is_image": True}, iteration=5)

    callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=5)

    torch.testing.assert_close(model.net.weight.grad.norm(), torch.tensor(1.0))
    torch.testing.assert_close(model.net_fake_score.weight.grad, critic_grad_before)
    assert model._distillation_parity_grad_metrics == {
        "clip_grad_norm/net_selected_preclip": 5.0,
        "clip_grad_norm/net_selected_clip_scale": pytest.approx(0.2),
        "clip_grad_norm/net_selected_clip_norm": 1.0,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_critic_phase_clips_only_fake_score_parameters() -> None:
    model = _PhaseModel()
    optimizer = PhaseOptimizer(
        {
            "net": torch.optim.SGD(model.net.parameters(), lr=0.1),
            "fake_score": torch.optim.SGD(model.net_fake_score.parameters(), lr=0.1),
        }
    )
    model.net.weight.grad = torch.tensor([[6.0, 8.0]])  # [1,2]
    model.net_fake_score.weight.grad = torch.tensor([[5.0, 12.0]])  # [1,2]
    student_grad_before = model.net.weight.grad.clone()  # [1,2]
    callback = _callback()
    callback.on_training_step_start(model, {"is_image": False}, iteration=6)

    callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=6)

    torch.testing.assert_close(model.net.weight.grad, student_grad_before)
    torch.testing.assert_close(model.net_fake_score.weight.grad.norm(), torch.tensor(1.0))
    assert model._distillation_parity_grad_metrics == {
        "clip_grad_norm/fake_score_selected_preclip": 13.0,
        "clip_grad_norm/fake_score_selected_clip_scale": pytest.approx(1.0 / 13.0),
        "clip_grad_norm/fake_score_selected_clip_norm": 1.0,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_dtensor_norm_is_reduced_before_recording_parity_metrics() -> None:
    model = _PhaseModel()
    optimizer = PhaseOptimizer(
        {
            "net": torch.optim.SGD(model.net.parameters(), lr=0.1),
            "fake_score": torch.optim.SGD(model.net_fake_score.parameters(), lr=0.1),
        }
    )
    model.net.weight.grad = torch.tensor([[3.0, 4.0]])  # [1,2]
    partial_norm = _PartialNorm(local_value=2.0, global_value=5.0)
    callback = _callback()
    callback.on_training_step_start(model, {"is_image": True}, iteration=5)

    with patch.object(grad_clip_module, "clip_grad_norm_", return_value=partial_norm):
        callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=5)

    assert partial_norm.full_tensor_calls == 1
    assert model._distillation_parity_grad_metrics == {
        "clip_grad_norm/net_selected_preclip": 5.0,
        "clip_grad_norm/net_selected_clip_scale": pytest.approx(1.0 / (5.0 + 1e-6)),
        "clip_grad_norm/net_selected_clip_norm": 1.0,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_logging_with_wandb_disabled_does_not_call_wandb_log(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _PhaseModel()
    optimizer = PhaseOptimizer(
        {
            "net": torch.optim.SGD(model.net.parameters(), lr=0.1),
            "fake_score": torch.optim.SGD(model.net_fake_score.parameters(), lr=0.1),
        }
    )
    model.net.weight.grad = torch.tensor([[3.0, 4.0]])  # [1,2]
    callback = _callback()
    callback.config.trainer.logging_iter = 1
    callback.on_training_step_start(model, {"is_image": True}, iteration=5)
    monkeypatch.setattr(grad_clip_module.wandb, "run", None)
    wandb_log = MagicMock()
    monkeypatch.setattr(grad_clip_module.wandb, "log", wandb_log)

    callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=5)

    wandb_log.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_multiple_optimizer_phases_accumulate_parity_metrics_within_one_training_step() -> None:
    model = _PhaseModel()
    optimizer = PhaseOptimizer(
        {
            "net": torch.optim.SGD(model.net.parameters(), lr=0.1),
            "fake_score": torch.optim.SGD(model.net_fake_score.parameters(), lr=0.1),
        }
    )
    model.net.weight.grad = torch.tensor([[3.0, 4.0]])  # [1,2]
    model.net_fake_score.weight.grad = torch.tensor([[5.0, 12.0]])  # [1,2]
    callback = _callback()
    callback.on_training_step_start(model, {"is_image": True}, iteration=5)

    callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=5)
    model.optimizer_key_override = "fake_score"
    callback.on_before_optimizer_step(model, optimizer, MagicMock(), MagicMock(), iteration=5)

    assert model._distillation_parity_grad_metrics == {
        "clip_grad_norm/net_selected_preclip": 5.0,
        "clip_grad_norm/net_selected_clip_scale": pytest.approx(0.2),
        "clip_grad_norm/net_selected_clip_norm": 1.0,
        "clip_grad_norm/fake_score_selected_preclip": 13.0,
        "clip_grad_norm/fake_score_selected_clip_scale": pytest.approx(1.0 / 13.0),
        "clip_grad_norm/fake_score_selected_clip_norm": 1.0,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_new_training_step_clears_stale_parity_grad_metrics() -> None:
    model = _PhaseModel()
    model._distillation_parity_grad_metrics = {"stale": 1.0}
    callback = _callback()

    callback.on_training_step_start(model, {"is_image": True}, iteration=5)

    assert model._distillation_parity_grad_metrics == {}
