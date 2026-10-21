# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Analytical sampler and inference checks without model downloads."""

import pytest
import torch

from cosmos_framework.posttrain.model.gr00t.ppo import (
    Gr00tPPOCondition,
    flow_sde_transition,
)
from cosmos_framework.posttrain.model.ppo import selected_transition_logprobs

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_ode_transition_is_the_euler_step_at_every_index() -> None:
    generator = torch.Generator().manual_seed(19)
    latent = torch.randn((4, 40, 132), generator=generator)
    velocity = torch.randn(latent.shape, generator=generator)
    mean, std = flow_sde_transition(latent, velocity, torch.arange(4), stochastic=False)
    torch.testing.assert_close(mean, latent + 0.25 * velocity)
    assert torch.count_nonzero(std) == 0


def test_first_sde_step_uses_the_reference_zero_time_noise_convention() -> None:
    latent = torch.linspace(-1, 1, 40 * 132).reshape(1, 40, 132)
    mean, std = flow_sde_transition(latent, torch.zeros_like(latent), torch.tensor([0]), stochastic=True)
    torch.testing.assert_close(mean, latent * 0.875)
    torch.testing.assert_close(std, torch.full_like(std, 0.5))


def test_selected_logprob_reduces_only_the_executed_coordinates() -> None:
    generator = torch.Generator().manual_seed(19)
    mean = torch.randn((2, 40, 132), generator=generator, dtype=torch.float64)
    std = torch.rand(mean.shape, generator=generator, dtype=torch.float64) + 0.1
    sample = mean + std * torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    expected = torch.distributions.Normal(mean[:, :16, :7], std[:, :16, :7]).log_prob(sample[:, :16, :7])
    actual = selected_transition_logprobs(sample, mean, std, score_horizon=16)
    torch.testing.assert_close(actual, expected.float().sum(dim=(1, 2)))
    changed = sample.clone()
    changed[:, 16:, :] += 9.0
    changed[:, :, 7:] -= 9.0
    torch.testing.assert_close(
        selected_transition_logprobs(changed, mean, std, score_horizon=16), actual, atol=0, rtol=0
    )


def test_ode_inference_has_no_exploration_noise_and_preserves_rng_and_modes():
    from contextlib import nullcontext

    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import LinearVelocityHeads

    head = LinearVelocityHeads()
    condition = Gr00tPPOCondition(
        torch.linspace(-1, 1, 8).reshape(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 132),
        torch.zeros(1, dtype=torch.long),
    )
    # The unit under test is the adapter's mode guard and real integrator, independent of model loading.
    adapter = object.__new__(Gr00tN1d7TrainPolicy)
    adapter.net, adapter.compute_ctx = head, nullcontext()
    head.train()
    head.velocity.eval()
    modes = [module.training for module in head.modules()]
    rng = torch.get_rng_state().clone()
    actions = adapter.ppo_ode_actions(condition, seed=25)
    again = adapter.ppo_ode_actions(condition, seed=25)
    expected = torch.randn((1, 40, 132), generator=torch.Generator().manual_seed(25))
    velocity = head.velocity(condition.backbone_features.mean(dim=1))[:, None]
    for _ in range(4):
        expected = expected + 0.25 * (velocity + 0.1 * expected)
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(actions, again, rtol=0, atol=0)
    assert not torch.equal(actions, adapter.ppo_ode_actions(condition, seed=26))
    assert [module.training for module in head.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
