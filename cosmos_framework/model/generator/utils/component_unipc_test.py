# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.generator.utils.component_unipc import ComponentSchedule, sample_component_unipc

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("shift", [1.0, 5.0])
@pytest.mark.parametrize("num_steps", [1, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equal_shifts_match_legacy_unipc(shift: float, num_steps: int, dtype: torch.dtype) -> None:
    noise = [torch.linspace(-1, 1, 8, dtype=dtype), torch.linspace(-2, 2, 11, dtype=dtype)]  # [8], [11]
    sampler = UniPCSampler(tensor_kwargs={"device": torch.device("cpu")})

    def velocity(values: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:  # B x [N_i], [1,1]
        # Every component depends on all the others, so this checks joint evolution.
        return [0.2 * value.sin() + 0.1 * value.mean() + timestep.item() / 5000 for value in values]  # B x [N_i]

    def joint_velocity(
        values: list[torch.Tensor],
        timesteps: list[torch.Tensor],  # B x [N_i], K x [1,1]
    ) -> list[torch.Tensor]:  # B x [N_i]
        torch.testing.assert_close(timesteps[0], timesteps[1])
        return velocity(values, timesteps[0])

    expected = sampler(velocity, noise, num_steps=num_steps, shift=shift, seed=[3, 7])  # B x [N_i]
    actual = sample_component_unipc(
        joint_velocity,
        noise,
        cfg=sampler.cfg,
        schedule=ComponentSchedule(lengths=[(3, 5), (7, 4)], shifts=(shift, shift)),
        num_steps=num_steps,
    )  # B x [N_i]
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_unequal_shifts_match_individual_samplers_with_one_joint_call_per_step() -> None:
    lengths = [(2, 3, 4), (5, 2, 3)]
    shifts = (1.0, 3.0, 5.0)
    seeds = [3, 7]
    noise = [torch.linspace(-1, 1, sum(parts), dtype=torch.float64) for parts in lengths]  # B x [N_i]
    sampler = UniPCSampler(tensor_kwargs={"device": torch.device("cpu")})
    callbacks: list[tuple[int, int]] = []
    model_timesteps: list[tuple[int, ...]] = []

    def velocity(value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # [N_ik], [1,1] -> [N_ik]
        return 0.2 * value.sin() + timestep.item() / 5000  # [N_ik]

    def joint_velocity(
        values: list[torch.Tensor],
        timesteps: list[torch.Tensor],  # B x [N_i], K x [1,1]
    ) -> list[torch.Tensor]:  # B x [N_i]
        assert callbacks[-1] == (len(model_timesteps), 7)
        model_timesteps.append(tuple(int(t.item()) for t in timesteps))
        return [
            torch.cat([velocity(part, t) for part, t in zip(value.split(sizes), timesteps, strict=True)])
            for value, sizes in zip(values, lengths, strict=True)
        ]  # B x [N_i]

    actual = sample_component_unipc(
        joint_velocity,
        noise,
        cfg=sampler.cfg,
        schedule=ComponentSchedule(lengths=lengths, shifts=shifts),
        num_steps=7,
        step_callback=lambda step, count: callbacks.append((step, count)),
    )  # B x [N_i]
    assert callbacks == [(step, 7) for step in range(7)]
    assert len(model_timesteps) == 7
    assert any(len(set(timesteps)) == 3 for timesteps in model_timesteps)
    for sample, result, sizes, seed in zip(noise, actual, lengths, seeds, strict=True):
        for source, output, shift in zip(sample.split(sizes), result.split(sizes), shifts, strict=True):
            expected = sampler(velocity, source, num_steps=7, shift=shift, seed=seed)  # [N_ik]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shifts", [(0.0,), (-1.0,), (float("inf"),), (float("nan"),), ()])
def test_invalid_shifts_fail_before_velocity_evaluation(shifts: tuple[float, ...]) -> None:
    def velocity(values: list[torch.Tensor], timesteps: list[torch.Tensor]) -> list[torch.Tensor]:
        pytest.fail("Invalid schedules must fail before evaluating the model")

    with pytest.raises(ValueError, match="positive and finite"):
        sample_component_unipc(
            velocity,
            [torch.zeros(3)],  # 1 x [3]
            cfg=UniPCSamplerConfig(),
            schedule=ComponentSchedule(lengths=[(3,)], shifts=shifts),
            num_steps=3,
        )
