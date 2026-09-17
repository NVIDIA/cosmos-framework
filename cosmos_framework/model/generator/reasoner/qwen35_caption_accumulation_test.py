# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare unified accumulated updates with one full-batch weighted-CE update."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils.config import CheckpointConfig
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.callback import Callback
from cosmos_framework.model.generator.algorithm.loss import cross_entropy as ce
from cosmos_framework.configs.base.reasoner.defaults.policy_config import PolicyConfig, VLMModelConfig
from cosmos_framework.model.generator.reasoner.qwen35_caption import Qwen35CaptionLoss
from cosmos_framework.model.generator.vlm_model import VLMModel
from cosmos_framework.utils.generator import flash_attn

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


class _CaptionHead(nn.Module):
    embedding: nn.Embedding
    model: nn.Linear
    exponent: float

    def __init__(self, exponent: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 5)
        self.model = nn.Linear(5, 17, bias=False)
        self.exponent = exponent

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> Qwen35CaptionLoss:
        return Qwen35CaptionLoss(
            *ce.fused_weighted_cross_entropy_loss(
                self.embedding(input_ids), labels, self.model.weight, exponent=self.exponent
            )
        )


class _GradientProbe(Callback):
    gradients: list[dict[str, torch.Tensor]]

    def __init__(self) -> None:
        super().__init__()
        self.gradients = []

    def on_before_optimizer_step(
        self,
        model: VLMModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        self.gradients.append({name: parameter.grad.clone() for name, parameter in model.named_parameters()})
        # The real trainer must finish window normalization before clipping callbacks run.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)


def _no_profile(*args: Any, **kwargs: Any) -> nullcontext[None]:
    return nullcontext()


def _make_model(monkeypatch: pytest.MonkeyPatch, exponent: float, normalize: bool = True) -> VLMModel:
    def init_vlm(self: VLMModel, config: VLMModelConfig, checkpoint: CheckpointConfig) -> None:
        self.model = _CaptionHead(exponent)
        self.hf_config = SimpleNamespace(model_type="qwen3_5")
        self.parallel_dims = None

    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=ignore_index, reduction="sum")

    def init_flash_attn_meta(deterministic: bool) -> None:
        pass

    monkeypatch.setattr(VLMModel, "_init_vlm", init_vlm)
    monkeypatch.setattr(flash_attn, "init_flash_attn_meta", init_flash_attn_meta)
    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    return VLMModel(
        VLMModelConfig(
            policy=PolicyConfig(
                use_weighted_ce=True,
                enable_fused_weighted_ce=True,
                weighted_ce_exponent=exponent,
                normalize_weighted_ce_over_accumulation_window=normalize,
            )
        ),
        CheckpointConfig(),
    )


@pytest.mark.parametrize("microbatches", [1, 2, 32])
@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_accumulation_matches_full_batch_update(
    monkeypatch: pytest.MonkeyPatch, microbatches: int, exponent: float, all_ignored: bool
) -> None:
    torch.manual_seed(110)
    model = _make_model(monkeypatch, exponent)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    probe = _GradientProbe()
    trainer = ImaginaireTrainer.__new__(ImaginaireTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            grad_accum_iter=microbatches,
            distributed_parallelism="fsdp",
            straggler_detection=SimpleNamespace(analyze_forward=False, analyze_backward=False, analyze_optimizer=False),
        )
    )
    trainer.training_timer = _no_profile
    trainer.straggler_detector = SimpleNamespace(profile_section=_no_profile)
    trainer.callbacks = probe
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    # Two consecutive windows also verify that the denominator is reset after each update.
    for window in range(2):
        ids = torch.randint(0, 17, (32, 9))
        labels = torch.full_like(ids, -100)
        counts = [0] + [1] * 15 + [3] * 8 + [8] * 8
        if window:
            counts.reverse()
        for row, count in enumerate(counts):
            if count and not all_ignored:
                labels[row, -count:] = ids[row, -count:]
        logits = reference.model.model(reference.model.embedding(ids))
        reference_loss = ce.weighted_cross_entropy_loss(logits, labels, exponent)
        reference_loss.backward()
        expected_gradients = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()}
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=0.01)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        accumulated = 0
        numerator = torch.zeros(())
        denominator = torch.zeros(())
        for input_batch, label_batch in zip(ids.chunk(microbatches), labels.chunk(microbatches), strict=True):
            output, _, accumulated = trainer.training_step(
                model,
                optimizer,
                scheduler,
                scaler,
                {"input_ids": input_batch, "labels": label_batch},
                iteration=window,
                grad_accum_iter=accumulated,
            )
            numerator += output["train_objective_numerator"]
            denominator += output["train_objective_denominator"]
        assert accumulated == 0
        assert model._weighted_ce_window_denominator is None
        assert model._weighted_ce_window_microbatches == 0
        torch.testing.assert_close(numerator / denominator.clamp(min=1), reference_loss.detach())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(probe.gradients[window][name], expected_gradients[name], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], atol=2e-6, rtol=2e-5)


def test_window_normalization_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(monkeypatch, exponent=0.5, normalize=False)
    ids = torch.tensor([[1, 2, 3, 4]])
    output, loss = model.training_step({"input_ids": ids, "labels": ids.clone()}, iteration=0)
    assert "_backward_loss" not in output
    assert model._weighted_ce_window_denominator is None
    torch.testing.assert_close(output["train_objective_numerator"], loss.detach())
    assert output["train_objective_denominator"].item() == 1
