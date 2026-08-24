# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler


def test_fixed_step_invokes_callback_per_step() -> None:
    calls: list[tuple[int, int]] = []
    sampler = FixedStepSampler(t_list=[1.0, 0.6, 0.3, 0.0])
    noise = torch.randn(8)
    sampler(
        lambda latent, timestep: torch.zeros_like(latent),
        noise,
        seed=0,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(0, 3), (1, 3), (2, 3)]


def test_unipc_invokes_callback_per_step() -> None:
    calls: list[tuple[int, int]] = []
    sampler = UniPCSampler(tensor_kwargs={"device": torch.device("cpu")})
    noise = torch.randn(4, 1, 2, 2)
    sampler(
        lambda latent, timestep: torch.zeros_like(latent),
        noise,
        num_steps=5,
        seed=0,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(i, 5) for i in range(5)]


def test_edm_invokes_callback_per_step_before_evaluations() -> None:
    calls: list[tuple[int, int]] = []
    evaluations: list[int] = []

    def x0_fn(noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        evaluations.append(len(calls))
        return torch.zeros_like(noisy)

    sampler = EDMSampler()
    sampler(
        x0_fn,
        torch.randn(1, 4),
        num_steps=6,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    num_steps = calls[-1][1]
    assert [c[0] for c in calls] == list(range(num_steps))
    # every model evaluation happened after at least one callback
    assert min(evaluations) >= 1


def test_callback_default_none_keeps_behavior() -> None:
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0])
    noise = torch.randn(8)
    out_a = sampler(lambda latent, timestep: torch.zeros_like(latent), noise, seed=1)
    out_b = sampler(
        lambda latent, timestep: torch.zeros_like(latent), noise, seed=1, step_callback=lambda i, n: None
    )
    torch.testing.assert_close(out_a, out_b)
