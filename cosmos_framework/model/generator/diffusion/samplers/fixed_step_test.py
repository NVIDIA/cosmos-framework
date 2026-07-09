# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler


@pytest.mark.L0
def test_auto_appends_zero():
    sampler = FixedStepSampler(t_list=[0.999, 0.75, 0.5, 0.25])
    assert sampler.t_list[-1] == 0.0
    assert sampler.t_list == [0.999, 0.75, 0.5, 0.25, 0.0]


@pytest.mark.L0
def test_no_double_append_zero():
    t_list = [0.999, 0.75, 0.5, 0.25, 0.0]
    sampler = FixedStepSampler(t_list=t_list)
    assert sampler.t_list == t_list
    assert sampler.t_list.count(0.0) == 1


@pytest.mark.L0
def test_sde_reproducible_with_seed():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="sde")
    noise = torch.randn(1, 16)

    def velocity_fn(x, _t):
        return torch.ones_like(x) * 0.1

    result_a = sampler(velocity_fn, noise.clone(), seed=123)
    result_b = sampler(velocity_fn, noise.clone(), seed=123)
    assert torch.allclose(result_a, result_b)


@pytest.mark.L0
def test_sde_preserves_conditioned_entries():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="sde")
    noise = [torch.tensor([5.0, 1.0])]  # [N]
    condition_reference = [torch.tensor([5.0, 0.0])]  # [N]
    condition_mask = [torch.tensor([1.0, 0.0])]  # [N]

    def velocity_fn(x, _t):
        return [torch.zeros_like(x_i) for x_i in x]

    result = sampler(
        velocity_fn,
        noise,
        seed=[123],
        condition_reference=condition_reference,
        condition_mask=condition_mask,
    )
    assert torch.allclose(result[0][0], condition_reference[0][0])


@pytest.mark.L0
def test_output_shape_preserved():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.25, 0.0])
    noise = torch.randn(1, 32)

    def velocity_fn(x, _t):
        return torch.zeros_like(x)

    result = sampler(velocity_fn, noise)
    assert result.shape == noise.shape


@pytest.mark.L0
def test_validates_empty_t_list():
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[])


@pytest.mark.L0
def test_validates_single_entry_t_list():
    # t_list=[0.0] has len==1 after construction; 0.0 is already last so no append,
    # but then len(self.t_list)==1 < 2 triggers the second assert
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[0.0])


@pytest.mark.L0
def test_validates_sample_type():
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[1.0, 0.0], sample_type="invalid")
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[1.0, 0.0], sample_type="ode")


@pytest.mark.L0
def test_velocity_fn_called_correct_times():
    t_list = [1.0, 0.75, 0.5, 0.25]  # auto-appends 0.0 → 5 entries, 4 steps
    sampler = FixedStepSampler(t_list=t_list)
    noise = torch.randn(1, 8)
    call_count = [0]

    def velocity_fn(x, _t):
        call_count[0] += 1
        return torch.zeros_like(x)

    sampler(velocity_fn, noise)
    expected_calls = len(sampler.t_list) - 1  # 4 steps
    assert call_count[0] == expected_calls


@pytest.mark.L0
def test_num_steps_and_shift_do_not_override_t_list():
    t_list = [0.9, 0.5, 0.1, 0.0]
    sampler = FixedStepSampler(t_list=t_list)
    noise = torch.zeros(1, 4)
    recorded_sigmas = []

    def velocity_fn(x, timestep):
        recorded_sigmas.append(timestep.item() / 1000.0)
        return torch.zeros_like(x)

    sampler(velocity_fn, noise, num_steps=8, shift=7.0)
    expected = t_list[:-1]  # last step uses sigma=0.0 but loop already excludes it
    assert len(recorded_sigmas) == len(expected)
    for got, exp in zip(recorded_sigmas, expected):
        assert abs(got - exp) < 1e-4


@pytest.mark.L0
def test_timestep_scaling():
    num_train_timesteps = 1000.0
    t_list = [0.8, 0.0]
    sampler = FixedStepSampler(t_list=t_list, num_train_timesteps=num_train_timesteps)
    noise = torch.randn(1, 4)
    recorded_timesteps = []

    def velocity_fn(x, timestep):
        recorded_timesteps.append(timestep.item())
        return torch.zeros_like(x)

    sampler(velocity_fn, noise)
    assert len(recorded_timesteps) == 1
    assert abs(recorded_timesteps[0] - 0.8 * num_train_timesteps) < 1e-5
