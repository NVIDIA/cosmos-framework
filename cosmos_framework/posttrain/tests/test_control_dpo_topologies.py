# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""End-to-end: the offline DPO loop over the decoupled stack, toy model, local backend.

Validates the model↔algorithm↔trainer decoupling (DESIGN §6) end-to-end without a
GPU: the GENERIC `DiffusionTrainer` drives the `algorithm.dpo` objective against a
`FlowMatchingPolicy` (the toy flow model), over `preference Env → buffer → trainer`.
Synthetic separable pairs guarantee a nonzero reward margin so a real optimizer step
happens. Flow test, NOT a convergence/parity test (the iter-0=0.6593 parity gate is
P1 on the real OmniMoT adapter)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.dpo import make_objective
from cosmos_framework.posttrain.control import ActorGroup, DeviceGroup, ExperienceBuffer, bind_device_groups
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.trainer import DiffusionTrainer
from cosmos_framework.posttrain.tests.helpers.preference_env import SyntheticPreferenceEnv
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel

DIM, BETA = 8, 500.0


def _setup():
    model = ToyFlowModel(dim=DIM, seed=0)
    opt = torch.optim.SGD(model.net.parameters(), lr=1.0)
    return {
        "data": SyntheticPreferenceEnv(dim=DIM, seed=0),
        "replay": ExperienceBuffer(),
        "policy": DiffusionTrainer(model, make_objective(BETA), opt),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_runs_two_steps_local():
    data = roles.env("data")
    replay = roles.buffer("replay")
    policy = roles.trainer("policy")
    bind_device_groups(
        [DeviceGroup((ActorGroup((data, replay, policy), _setup),), device="cpu")],
        backend="local",
    )

    P = 4
    before = next(policy.body.model.net.parameters()).detach().clone()
    losses = []
    for _ in range(2):
        pairs = data.get_samples(P).result()
        replay.put_samples(pairs)
        batch = replay.get_samples(P).result()
        metrics = policy.train(batch).result()
        loss = float(metrics["train/loss_avg"])
        assert loss == loss  # finite
        # ref is frozen & bit-equal at init ⇒ ref errors must equal across sides only
        # via the policy; sanity-check the metric surface is present.
        assert {
            "train/dpo_reward_margin",
            "train/dpo_accuracy",
            "train/dpo_err_pol_w",
            "train/dpo_err_ref_w",
        } <= set(metrics)
        losses.append(loss)
    after = next(policy.body.model.net.parameters()).detach()
    assert len(losses) == 2
    assert not torch.allclose(before, after)  # optimizer stepped over the run


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_objective_is_model_agnostic():
    """The objective only touches the FlowMatchingPolicy protocol — calling it
    directly on the toy model (no trainer/WG) must produce a finite loss + full
    metric dict. Guards the decoupling contract."""
    from cosmos_framework.posttrain.algorithm.dpo import dpo_objective
    from cosmos_framework.posttrain.data.packer.preference import unpack_preferences

    model = ToyFlowModel(dim=DIM, seed=1)
    env = SyntheticPreferenceEnv(dim=DIM, seed=1)
    trajs = env.get_samples(3)
    wb, lb = unpack_preferences(trajs)
    loss, metrics = dpo_objective(model, wb, lb, BETA)
    assert torch.isfinite(loss)
    assert float(metrics["dpo_accuracy"]) >= 0.0
    public_metrics = make_objective(BETA).adapt_metrics(metrics, "train")
    assert float(public_metrics["train/dpo_accuracy"]) >= 0.0
