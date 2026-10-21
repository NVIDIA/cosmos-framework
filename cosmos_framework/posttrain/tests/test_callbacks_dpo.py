# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the DPO observability callbacks + the trainer's callback wiring."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.algorithm.dpo import make_objective
from cosmos_framework.posttrain.callbacks.dpo import DPOMetricLogger, DPORewardTrendCallback
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.tests.helpers.preference_env import SyntheticPreferenceEnv
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel


@pytest.mark.L0
@pytest.mark.CPU
def test_callbacks_fire_and_record_through_trainer():
    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=1)
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    logger = DPOMetricLogger(log_every=2)
    trend = DPORewardTrendCallback(window=3, warn=False)
    trainer = DiffusionTrainer(model, make_objective(beta=100.0), opt, callbacks=[logger, trend])

    for _ in range(5):
        trainer.train(env.get_samples(4))

    # One record per optimizer step (grad_accum=1 ⇒ per train call).
    assert len(logger.history) == 5
    assert {"train/loss_avg", "train/dpo_reward_margin", "train/dpo_accuracy"} <= set(logger.history[0])
    # Trend means are finite and computed over the (windowed) history.
    assert trend.margin_mean() == trend.margin_mean()  # not NaN
    assert 0.0 <= trend.accuracy_mean() <= 1.0


@pytest.mark.L0
@pytest.mark.CPU
def test_grad_accum_fires_callback_once_per_optimizer_step():
    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=2)
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    logger = DPOMetricLogger(log_every=100)
    trainer = DiffusionTrainer(model, make_objective(beta=100.0), opt, grad_accum=2, callbacks=[logger])

    for _ in range(4):  # 4 micro-steps, grad_accum=2 ⇒ 2 optimizer steps
        trainer.train(env.get_samples(2))
    assert trainer._iter == 2
    assert len(logger.history) == 2  # callback fires only on completed optimizer steps


@pytest.mark.L0
@pytest.mark.CPU
def test_observer_exception_does_not_break_training():
    class Boom(DPOMetricLogger):
        def on_train_step(self, trainer, metrics):
            raise ValueError("boom")

    torch.manual_seed(0)
    env = SyntheticPreferenceEnv(dim=8, seed=3)
    model = ToyFlowModel(dim=8, seed=0)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
    trainer = DiffusionTrainer(model, make_objective(beta=100.0), opt, callbacks=[Boom()])
    # A throwing observer must not crash the loop (best-effort observability, §2.6).
    trainer.train(env.get_samples(2))
    assert trainer._iter == 1
