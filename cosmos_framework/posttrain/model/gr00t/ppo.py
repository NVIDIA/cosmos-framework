# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T action/value forwards and the fixed N1.7 selected Flow-SDE transition.

The sampler follows RLinf e0531f4's N1.7 recipe. Its logprob is the conditional
density of one denoising transition, reduced over the sampled 16 x 7 or 8 x 7 prefix.
It is not the exact density of the decoded environment action.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any, TypeVar

import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.joint_ppo.records import sampled_decision
from cosmos_framework.posttrain.model.gr00t.policy import Gr00tHeads
from cosmos_framework.posttrain.model.ppo import selected_transition_logprobs
from cosmos_framework.posttrain.model.precision import cast_floating_buffers
from cosmos_framework.posttrain.schema import Trajectory, Transition
from cosmos_framework.posttrain.utils.dist import run_rank_converged

NATIVE_ODE_REVISION = "nvidia-euler-bf16-buffers-v1"

SAMPLER_IDENTITY = "gr00t-n1d7-selected-flow-sde-v1"
PREFIX_SAMPLER_IDENTITY = "gr00t-n1d7-selected-flow-sde-prefix-v2"


@dataclass(frozen=True)
class Gr00tPPOCondition:
    """Frozen backbone outputs and raw state; trainable transforms are not cached."""

    backbone_features: torch.Tensor  # [B,L,C]
    image_mask: torch.Tensor  # [B,L]
    backbone_attention_mask: torch.Tensor  # [B,L]
    state: torch.Tensor  # [B,1,S]
    embodiment_id: torch.Tensor  # [B]


@dataclass(frozen=True)
class Gr00tPPOTrace:
    latent_before: torch.Tensor  # [B,40,132]
    latent_after: torch.Tensor  # [B,40,132]
    denoise_index: torch.Tensor  # [B]
    time: torch.Tensor  # [B]
    sampler_identity: str = SAMPLER_IDENTITY
    score_horizon: int = 16


@dataclass(frozen=True)
class Gr00tPPOScores:
    logprobs: torch.Tensor  # [B]
    values: torch.Tensor  # [B]


@dataclass(frozen=True)
class Gr00tPPOSample:
    normalized_actions: torch.Tensor  # [B,40,132], prior to processor decoding
    trace: Gr00tPPOTrace
    scores: Gr00tPPOScores
    action_noise: torch.Tensor  # [B,H_exec,7], decoded-action noise from the same RNG stream


class Gr00tPPOValueHead(nn.Module):
    """The reference action-head value MLP, including its parameter names."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for width in (1024, 512, 256):
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                if layer is self.mlp[-1]:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # [B,C] -> [B,1]
        return self.mlp(features)


def flow_sde_transition(
    latent: torch.Tensor, velocity: torch.Tensor, index: torch.Tensor, *, stochastic: bool
) -> tuple[torch.Tensor, torch.Tensor]:  # [B,H,D], [B,H,D], [B] -> mean/std [B,H,D]
    """Reference operation order and input precision, including its t=0 convention."""
    timesteps = torch.linspace(0, 1, 5, device=latent.device, dtype=latent.dtype)
    time = timesteps[index][:, None, None].expand_as(latent)
    delta = (timesteps[index + 1] - timesteps[index])[:, None, None].expand_as(latent)
    initial = latent - velocity * time
    final = latent + velocity * (1 - time)
    if stochastic:
        noise_level = torch.tensor(0.5, device=latent.device)
        sigma = noise_level * torch.sqrt((1 - timesteps) / torch.where(timesteps == 0, timesteps[1], timesteps))[:-1]
        sigma = sigma[index][:, None, None].expand_as(latent)
        initial_weight = torch.ones_like(time) - (time + delta) - sigma**2 * delta / (2 * (1 - time))
        std = torch.sqrt(delta) * sigma
    else:
        initial_weight = 1 - (time + delta)
        std = torch.zeros_like(time)
    return initial * initial_weight + final * (time + delta), std


def validate_trace(condition: Gr00tPPOCondition, trace: Gr00tPPOTrace) -> None:
    """Reject corrupted sampling records before the trainable forward's collectives."""
    expected = SAMPLER_IDENTITY if trace.score_horizon == 16 else PREFIX_SAMPLER_IDENTITY
    if type(trace.score_horizon) is not int or trace.score_horizon not in (8, 16) or trace.sampler_identity != expected:
        raise ValueError("GR00T PPO sampler identity does not match the bound recipe")
    batch = condition.backbone_features.shape[0]
    for name in ("latent_before", "latent_after"):
        value = getattr(trace, name)
        if value.shape != (batch, 40, 132) or value.dtype != condition.backbone_features.dtype:
            raise ValueError(f"GR00T PPO {name} must retain the native [B,40,132] shape and compute dtype")
        if value.device != condition.backbone_features.device or not bool(torch.isfinite(value).all()):
            raise ValueError(f"GR00T PPO {name} has an invalid device or nonfinite value")
    index = trace.denoise_index
    if index.shape != (batch,) or index.dtype != torch.long or index.device != trace.latent_before.device:
        raise ValueError("GR00T PPO denoise_index must be a native-device int64 [B] tensor")
    if bool(((index < 0) | (index >= 4)).any()):
        raise ValueError("GR00T PPO denoise_index must lie in [0,4)")
    expected_time = index.to(condition.backbone_features.dtype) / 4
    if not torch.equal(trace.time, expected_time):
        raise ValueError("GR00T PPO recorded time differs from its denoising index")


class Gr00tPPOHeads(Gr00tHeads):
    """One public FSDP forward for all trainable action and value transformations.

    Wrap this container with FSDP2, including the existing DiT block wrapping policy.
    The original head remains at ``action_head.*`` for checkpoint and weight publication.
    """

    def __init__(self, action_head: nn.Module, *, score_horizon: int = 16) -> None:
        super().__init__(action_head)
        if type(score_horizon) is not int or score_horizon not in (8, 16):
            raise ValueError("GR00T PPO score_horizon must be 8 or 16")
        self.score_horizon = score_horizon
        config = action_head.config
        if (config.action_horizon, config.max_action_dim, action_head.num_inference_timesteps) != (40, 132, 4):
            raise ValueError("GR00T PPO requires native horizon 40, latent width 132 and four denoising steps")
        if not hasattr(action_head, "value_head"):
            parameter = next(action_head.parameters())
            # HF suppresses initialization while loading the reference head; the reference then
            # initializes the missing value MLP once, in the loaded parameter dtype on CPU.
            with torch.device("meta"):
                head = Gr00tPPOValueHead(config.backbone_embedding_dim + config.input_embedding_dim)
            head.to(dtype=parameter.dtype)
            head.to_empty(device=parameter.device)
            head.initialize_weights()
            action_head.value_head = head

    def _features(self, condition: Gr00tPPOCondition) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.action_head
        # New tensors preserve the frozen cache even when a caller retains an exception.
        vl = head.vl_self_attention(head.vlln(condition.backbone_features))
        state = condition.state.reshape(condition.state.shape[0], 1, -1)
        return vl, head.state_encoder(state, condition.embodiment_id)

    def _velocity(
        self,
        condition: Gr00tPPOCondition,
        vl: torch.Tensor,
        state: torch.Tensor,
        latent: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        head = self.action_head
        buckets = (index / 4.0 * head.num_timestep_buckets).to(torch.int64)
        actions = head.action_encoder(latent, buckets, condition.embodiment_id)
        if head.config.add_pos_embed:
            positions = torch.arange(actions.shape[1], device=actions.device)
            actions = actions + head.position_embedding(positions).unsqueeze(0)
        output = head.model(
            hidden_states=torch.cat((state, actions), dim=1),
            encoder_hidden_states=vl,
            timestep=buckets,
            image_mask=condition.image_mask,
            backbone_attention_mask=condition.backbone_attention_mask,
        )
        return head.action_decoder(output[:, -head.action_horizon :], condition.embodiment_id)

    def _values(self, vl: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((vl.mean(dim=1), state.reshape(state.shape[0], -1)), dim=1)
        return self.action_head.value_head(inputs)[:, 0].float()

    def ode_actions(self, condition: Gr00tPPOCondition, *, seed: int, native: bool = False) -> torch.Tensor:
        """Deterministic inference on a published, unsharded rollout head using a local RNG."""
        vl, state = self._features(condition)
        generator = torch.Generator(device=vl.device).manual_seed(seed)
        latent = torch.randn((vl.shape[0], 40, 132), device=vl.device, dtype=vl.dtype, generator=generator)
        for step in range(4):
            index = torch.full((vl.shape[0],), step, device=vl.device, dtype=torch.long)
            velocity = self._velocity(condition, vl, state, latent, index)
            if native:
                # Preserve the native Euler operation order: algebraic rearrangement changes BF16 rounding.
                latent = latent + 0.25 * velocity * torch.ones_like(latent)
            else:
                latent, _ = flow_sde_transition(latent, velocity, index, stochastic=False)
        return latent

    def forward(
        self,
        condition: Gr00tPPOCondition,
        trace: Gr00tPPOTrace | None = None,
        *,
        seed: int | None = None,
    ) -> Gr00tPPOScores | Gr00tPPOSample:
        if trace is not None:

            def check_trace() -> None:
                validate_trace(condition, trace)
                if trace.score_horizon != self.score_horizon:
                    raise ValueError("GR00T PPO trace scoring prefix differs from the bound policy")

            run_rank_converged(check_trace, "GR00T PPO trace validation")
        vl, state = self._features(condition)
        values = self._values(vl, state)
        if trace is not None:
            velocity = self._velocity(condition, vl, state, trace.latent_before, trace.denoise_index)
            mean, std = flow_sde_transition(trace.latent_before, velocity, trace.denoise_index, stochastic=True)
            return Gr00tPPOScores(
                selected_transition_logprobs(trace.latent_after, mean, std, score_horizon=self.score_horizon), values
            )
        if seed is None:
            raise ValueError("GR00T PPO sampling requires an explicit request seed")
        generator = torch.Generator(device=vl.device).manual_seed(seed)
        latent = torch.randn((vl.shape[0], 40, 132), device=vl.device, dtype=vl.dtype, generator=generator)
        selected = random.Random(seed).randint(0, 3)
        sampled_trace: Gr00tPPOTrace | None = None
        selected_logprobs: torch.Tensor | None = None
        for step in range(4):
            index = torch.full((vl.shape[0],), step, device=vl.device, dtype=torch.long)
            velocity = self._velocity(condition, vl, state, latent, index)
            mean, std = flow_sde_transition(latent, velocity, index, stochastic=step == selected)
            following = (
                mean
                + torch.normal(
                    mean=0.0, std=1.0, size=latent.shape, device=vl.device, dtype=latent.dtype, generator=generator
                )
                * std
            )
            if step == selected:
                sampled_trace = Gr00tPPOTrace(
                    latent,
                    following,
                    index,
                    index.to(vl.dtype) / 4,
                    SAMPLER_IDENTITY if self.score_horizon == 16 else PREFIX_SAMPLER_IDENTITY,
                    self.score_horizon,
                )
                selected_logprobs = selected_transition_logprobs(following, mean, std, score_horizon=self.score_horizon)
            latent = following
        assert sampled_trace is not None and selected_logprobs is not None
        action_noise = 0.1 * torch.randn((vl.shape[0], self.score_horizon, 7), device=vl.device, generator=generator)
        return Gr00tPPOSample(latent, sampled_trace, Gr00tPPOScores(selected_logprobs, values), action_noise)


@contextmanager
def native_ode_backbone_buffers(backbone: torch.nn.Module) -> Iterator[None]:
    """Match Gr00tPolicy.to(BF16), including nonpersistent RoPE buffers, for native ODE only.

    The legacy loader keeps those buffers in FP32. Replace registered bindings (as Module.to
    does), preserving plain-attribute aliases such as original_inv_freq and restoring on failure.
    """
    originals = []
    try:
        cast_floating_buffers(backbone, torch.bfloat16, originals=originals)
        yield
    finally:
        for module, name, buffer in originals:
            module._buffers[name] = buffer


CONDITION_KEY = "gr00t_ppo_condition"
TRACE_KEY = "gr00t_ppo_trace"


Record = TypeVar("Record", Gr00tPPOCondition, Gr00tPPOTrace)


def to_device(record: Record, device: torch.device) -> Record:
    """Copy dataclass tensors, preserving the native dtype and the immutable source."""
    return replace(
        record,
        **{
            field.name: getattr(record, field.name).to(device=device)
            for field in fields(record)
            if isinstance(getattr(record, field.name), torch.Tensor)
        },
    )


def validate_ppo_decision(step: Transition, *, score_horizon: int) -> None:
    condition, trace = step.obs.representations.get(CONDITION_KEY), step.policy_info.get(TRACE_KEY)
    if not isinstance(condition, Gr00tPPOCondition) or not isinstance(trace, Gr00tPPOTrace):
        raise TypeError(f"continuous PPO sample {step.source.sample_id!r} lacks its typed condition/trace")
    validate_trace(condition, trace)
    if trace.score_horizon != score_horizon:
        raise ValueError(f"continuous PPO sample {step.source.sample_id!r} has a different scoring prefix")
    sample_index = step.policy_info.get("sample_index")
    if type(sample_index) is not int or not 0 <= sample_index < condition.backbone_features.shape[0]:
        raise ValueError(f"continuous PPO sample {step.source.sample_id!r} has an invalid invocation row")


def sampled_decisions(
    observations: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    seed: int,
    version: int,
) -> list[Trajectory]:
    """Preserve each complete sampling invocation and correlate its selected row with feedback."""
    return [
        sampled_decision(
            observation,
            result,
            source,
            representations={CONDITION_KEY: result["condition"]},
            policy_info={TRACE_KEY: result["trace"]},
            seed=seed,
            version=version,
        )
        for observation, result, source in zip(observations, sampled, sources, strict=True)
    ]
