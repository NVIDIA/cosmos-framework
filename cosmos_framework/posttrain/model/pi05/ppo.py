# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: OpenMDW-1.1
"""Selected OpenPI Flow-SDE transition and the shared trainable public forward.

Adapted from RLinf ce4d821780d0e431d90f43d028fc93cac61f8fe0 openpi_action_model.py.
This score is a conditional denoising-transition density over the executed
prefix, not a density of the decoded action. The frozen cache belongs to the
original sampling invocation and is never updated by the expert.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.posttrain.model.pi05.compat import KVPair, additive_attention_mask
from cosmos_framework.posttrain.model.pi05.config import SAMPLER_IDENTITY, SAMPLER_NOISE_LEVEL, Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.ppo import selected_transition_logprobs


@dataclass(frozen=True)
class Pi05PPOCondition:
    prefix_cache: tuple[KVPair, ...]  # 18 pairs of [B,1,968,256]
    prefix_mask: torch.Tensor  # [B,968]
    value_features: torch.Tensor  # [B,2048], float32 after native-precision pooling


@dataclass(frozen=True)
class Pi05PPOTrace:
    latent_before: torch.Tensor  # [B,10,32], float32
    latent_after: torch.Tensor  # [B,10,32], float32
    denoise_index: torch.Tensor  # [B]
    sampler_identity: str = SAMPLER_IDENTITY
    score_horizon: int = 5
    denoise_steps: int = 3


@dataclass(frozen=True)
class Pi05PPOScores:
    logprobs: torch.Tensor  # [B]
    values: torch.Tensor  # [B]


@dataclass(frozen=True)
class Pi05PPOSample:
    normalized_actions: torch.Tensor  # [B,10,32]
    trace: Pi05PPOTrace
    scores: Pi05PPOScores


def timesteps(steps: int, device: torch.device) -> torch.Tensor:
    return torch.cat(
        (
            torch.linspace(1, 1 / steps, steps, device=device, dtype=torch.float32),
            torch.zeros(1, device=device, dtype=torch.float32),
        )
    )


def flow_sde_transition(
    latent: torch.Tensor, velocity: torch.Tensor, index: torch.Tensor, *, steps: int = 3, stochastic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    times = timesteps(steps, latent.device)
    time = times[index][:, None, None].expand_as(latent)
    delta = (times[index] - times[index + 1])[:, None, None].expand_as(latent)
    x0 = latent - velocity * time
    x1 = latent + velocity * (1 - time)
    if stochastic:
        denominator = torch.where(times == 1, times[1], times)
        sigma = (
            torch.full((), SAMPLER_NOISE_LEVEL, dtype=latent.dtype, device=latent.device)
            * torch.sqrt(times / (1 - denominator))[:-1]
        )
        sigma = sigma[index][:, None, None].expand_as(latent)
        x0_weight = torch.ones_like(time) - (time - delta)
        x1_weight = time - delta - sigma**2 * delta / (2 * time)
        std = torch.sqrt(delta) * sigma
    else:
        x0_weight = 1 - (time - delta)
        x1_weight = time - delta
        std = torch.zeros_like(time)
    return x0 * x0_weight + x1 * x1_weight, std


def validate_trace(condition: Pi05PPOCondition, trace: Pi05PPOTrace, config: Pi05ModelConfig) -> None:
    if (
        condition.prefix_mask.ndim != 2
        or condition.prefix_mask.shape[1] != 968
        or condition.prefix_mask.dtype != torch.bool
    ):
        raise ValueError("pi05 PPO condition.prefix_mask must be bool [B,968]")
    batch = condition.prefix_mask.shape[0]
    if condition.value_features.shape != (batch, 2048) or condition.value_features.dtype != torch.float32:
        raise ValueError("pi05 PPO condition.value_features must be float32 [B,2048]")
    if len(condition.prefix_cache) != 18:
        raise ValueError("pi05 PPO condition.prefix_cache must contain all 18 layers")
    for index, pair in enumerate(condition.prefix_cache):
        if len(pair) != 2 or any(
            value.shape != (batch, 1, 968, 256) or value.dtype != torch.bfloat16 for value in pair
        ):
            raise ValueError(f"pi05 PPO condition.prefix_cache[{index}] must contain two BF16 [B,1,968,256] tensors")
    validate_selected_trace(trace, config, batch=batch)


def validate_selected_trace(trace: Pi05PPOTrace, config: Pi05ModelConfig, *, batch: int) -> None:
    """Validate a complete recorded invocation independently of prefix storage."""
    if (
        trace.sampler_identity != SAMPLER_IDENTITY
        or trace.score_horizon != config.action_chunk
        or trace.denoise_steps != config.denoise_steps
    ):
        raise ValueError("pi05 PPO trace sampler identity differs from the bound recipe")
    for name in ("latent_before", "latent_after"):
        value = getattr(trace, name)
        if value.shape != (batch, 10, 32) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"pi05 PPO trace.{name} must be finite float32 [B,10,32]")
    if trace.denoise_index.shape != (batch,) or trace.denoise_index.dtype != torch.long:
        raise ValueError("pi05 PPO trace.denoise_index must be int64 [B]")
    if ((trace.denoise_index < 0) | (trace.denoise_index >= config.denoise_steps)).any():
        raise ValueError("pi05 PPO trace.denoise_index is outside the denoising schedule")


class Pi05Heads(nn.Module):
    """Borrow trainable modules from the loaded full tree; the owner outlives this view."""

    def __init__(self, model: Pi05Model) -> None:
        super().__init__()
        self.config = model.config
        self.expert = model.paligemma_with_expert.gemma_expert.model
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.time_mlp_in = model.time_mlp_in
        self.time_mlp_out = model.time_mlp_out
        self.value_head = model.value_head

    def velocity(self, condition: Pi05PPOCondition, latent: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        width = self.action_in_proj.out_features
        fraction = torch.linspace(0, 1, width // 2, dtype=torch.float64, device=time.device)
        periods = 4e-3 * (4.0 / 4e-3) ** fraction
        phases = (1.0 / periods * 2 * math.pi)[None, :] * time[:, None]
        embedding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1).to(time.dtype)
        adaptive = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(embedding))))
        suffix = self.action_in_proj(latent)
        batch, horizon = latent.shape[:2]
        prefix_mask = condition.prefix_mask[:, None, :].expand(batch, horizon, -1)
        suffix_mask = torch.ones((batch, horizon, horizon), dtype=torch.bool, device=latent.device)
        mask = torch.cat((prefix_mask, suffix_mask), dim=2)
        attention = additive_attention_mask(mask)
        positions = condition.prefix_mask.sum(dim=1)[:, None] + torch.arange(horizon, device=latent.device)[None]
        output, _ = self.expert(suffix, attention, positions, condition=adaptive, prefix_cache=condition.prefix_cache)
        return self.action_out_proj(output.float())

    def values(self, condition: Pi05PPOCondition) -> torch.Tensor:
        return self.value_head(condition.value_features)[:, 0].float()

    def ode_actions(self, condition: Pi05PPOCondition, *, seed: int) -> torch.Tensor:
        return self._sample(condition, seed=seed, stochastic=False)[0]

    def _sample(
        self, condition: Pi05PPOCondition, *, seed: int, stochastic: bool
    ) -> tuple[torch.Tensor, Pi05PPOTrace | None, torch.Tensor | None]:
        device = condition.prefix_mask.device
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = (condition.prefix_mask.shape[0], 10, 32)

        def noise() -> torch.Tensor:
            return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device, generator=generator)

        latent = noise()
        selected = random.Random(seed).randint(0, self.config.denoise_steps - 1) if stochastic else -1
        trace = None
        logprobs = None
        times = timesteps(self.config.denoise_steps, device)
        for step in range(self.config.denoise_steps):
            index = torch.full((shape[0],), step, dtype=torch.long, device=device)
            velocity = self.velocity(condition, latent, times[index])
            mean, std = flow_sde_transition(
                latent, velocity, index, steps=self.config.denoise_steps, stochastic=step == selected
            )
            following = mean + noise() * std
            if step == selected:
                trace = Pi05PPOTrace(
                    latent, following, index, SAMPLER_IDENTITY, self.config.action_chunk, self.config.denoise_steps
                )
                logprobs = selected_transition_logprobs(following, mean, std, score_horizon=self.config.action_chunk)
            latent = following
        return latent, trace, logprobs

    def forward(
        self, condition: Pi05PPOCondition, trace: Pi05PPOTrace | None = None, *, seed: int | None = None
    ) -> Pi05PPOScores | Pi05PPOSample:
        values = self.values(condition)
        if trace is not None:
            # The policy boundary validates traces collectively before this FSDP forward.
            time = timesteps(self.config.denoise_steps, trace.latent_before.device)[trace.denoise_index]
            velocity = self.velocity(condition, trace.latent_before, time)
            mean, std = flow_sde_transition(
                trace.latent_before, velocity, trace.denoise_index, steps=self.config.denoise_steps, stochastic=True
            )
            return Pi05PPOScores(
                selected_transition_logprobs(trace.latent_after, mean, std, score_horizon=self.config.action_chunk),
                values,
            )
        if seed is None:
            raise ValueError("pi05 PPO sampling requires an explicit request seed")
        actions, sampled_trace, logprobs = self._sample(condition, seed=seed, stochastic=True)
        return Pi05PPOSample(actions, sampled_trace, Pi05PPOScores(logprobs, values))
