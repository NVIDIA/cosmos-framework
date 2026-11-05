# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Numerical contracts of the local OpenPI compatibility and transition kernels."""

import pytest
import torch
from transformers import GemmaConfig

from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaModel, Pi05RMSNorm, additive_attention_mask
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.pi05.ppo import flow_sde_transition, timesteps
from cosmos_framework.posttrain.model.pi05.source import model_identity
from cosmos_framework.posttrain.model.ppo import selected_transition_logprobs

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(31)
        yield


def test_pi05_adaptive_norm_condition_changes_scale_shift_and_gate():
    generator = torch.Generator().manual_seed(83)
    layer = Pi05RMSNorm(8, 1e-6, adaptive=True)
    hidden = torch.randn((2, 3, 8), generator=generator)
    condition = torch.randn((2, 8), generator=generator, requires_grad=True)
    normalized, gate = layer(hidden, condition)
    modulation = condition @ layer.dense.weight.T + layer.dense.bias
    scale, shift, expected_gate = modulation[:, None].chunk(3, dim=-1)
    expected = hidden / torch.sqrt(hidden.square().mean(-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(normalized, expected * (1 + scale) + shift)
    torch.testing.assert_close(gate, expected_gate)
    (normalized.sum() + gate.sum()).backward()
    assert condition.grad is not None and torch.count_nonzero(condition.grad) > 0


def test_pi05_expert_reuses_immutable_prefix_cache_across_forwards():
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        hidden_activation="gelu_pytorch_tanh",
    )
    model = Pi05GemmaModel(config, adaptive=True)
    generator = torch.Generator().manual_seed(17)
    prefix = tuple(
        (torch.randn((2, 1, 5, 8), generator=generator), torch.randn((2, 1, 5, 8), generator=generator))
        for _ in range(config.num_hidden_layers)
    )
    original = tuple((key.clone(), value.clone()) for key, value in prefix)
    hidden = torch.randn((2, 3, 16), generator=generator)
    condition = torch.randn((2, 16), generator=generator)
    positions = torch.arange(5, 8).expand(2, -1)
    mask = torch.zeros((2, 1, 3, 8))
    output, _ = model(hidden, mask, positions, condition=condition, prefix_cache=prefix)
    output.square().sum().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    repeated, _ = model(hidden, mask, positions, condition=condition, prefix_cache=prefix)
    repeated.square().sum().backward()
    torch.testing.assert_close(repeated, output, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    for (key, value), (old_key, old_value) in zip(prefix, original, strict=True):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)


def test_pi05_flow_sde_density_and_ode_direction():
    generator = torch.Generator().manual_seed(19)
    latent = torch.randn((3, 10, 32), generator=generator)
    velocity = torch.randn((3, 10, 32), generator=generator)
    index = torch.arange(3)
    mean, std = flow_sde_transition(latent, velocity, index, stochastic=False)
    torch.testing.assert_close(mean, latent - velocity / 3, rtol=2e-6, atol=5e-7)
    assert torch.count_nonzero(std) == 0
    mean, std = flow_sde_transition(latent, velocity, index, stochastic=True)
    assert torch.isfinite(mean).all() and (std > 0).all()
    sample = mean + torch.randn(mean.shape, generator=generator) * std
    expected = torch.distributions.Normal(mean, std).log_prob(sample)[:, :5, :7].sum((1, 2))
    actual = selected_transition_logprobs(sample, mean, std, score_horizon=5)
    torch.testing.assert_close(actual, expected)
    changed = sample.clone()
    changed[:, 5:, 7:] += 100
    torch.testing.assert_close(
        selected_transition_logprobs(changed, mean, std, score_horizon=5), actual, rtol=0, atol=0
    )
    changed[:, :5, :7] += std[:, :5, :7]
    assert not torch.equal(selected_transition_logprobs(changed, mean, std, score_horizon=5), actual)


@pytest.mark.parametrize("values", [{"action_chunk": 11}, {"denoise_steps": 1}])
def test_pi05_config_rejects_unsupported_sampler(values):
    with pytest.raises(ValueError, match="pi05"):
        Pi05ModelConfig("checkpoint", "tokenizer", "stats", **values)


def test_pi05_fixed_noise_remains_part_of_model_identity():
    config = Pi05ModelConfig("checkpoint", "tokenizer", "stats")
    assert model_identity(config, {})["noise_level"] == 0.5
    with pytest.raises(TypeError, match="noise_level"):
        Pi05ModelConfig("checkpoint", "tokenizer", "stats", noise_level=0.5)


def test_pi05_model_preserves_native_precision_under_omni_default():
    config = Pi05ModelConfig("checkpoint", "tokenizer", "stats")
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        baseline = Pi05Model(config, device="meta")
        expected = {name: parameter.dtype for name, parameter in baseline.named_parameters()}
        assert baseline.action_in_proj.weight.dtype == baseline.value_head.mlp[0].weight.dtype == torch.float32
        assert (
            baseline.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        )
        # Omni enters this dtype context before invoking its pipeline loader.
        torch.set_default_dtype(torch.bfloat16)
        actual = Pi05Model(config, device="meta")
        changed = [name for name, parameter in actual.named_parameters() if parameter.dtype != expected[name]]
        assert not changed, f"Omni changed native parameter dtypes: {changed}"
        assert timesteps(3, torch.device("cpu")).dtype == torch.float32
        mask = additive_attention_mask(torch.tensor([[[True, False]]]))
        assert mask.dtype == torch.float32 and torch.isfinite(mask).all()
    finally:
        torch.set_default_dtype(original)
