# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pi0.5 sampling records carried by the existing decision collector."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from typing import TypeVar

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import sampled_decision
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition, Pi05PPOTrace, validate_selected_trace
from cosmos_framework.posttrain.model.pi05.processing import Pi05ReplayInputs, validate_replay_inputs
from cosmos_framework.posttrain.schema import Trajectory, Transition

INPUTS_KEY = "pi05_ppo_inputs"
TRACE_KEY = "pi05_ppo_trace"
Record = TypeVar("Record", Pi05PPOCondition, Pi05PPOTrace, Pi05ReplayInputs)


def map_record(record: Record, transform: Callable[[torch.Tensor], torch.Tensor]) -> Record:
    """Map owned tensors, including immutable per-layer prefix KV pairs."""

    def convert(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return transform(value)
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    return replace(record, **{field.name: convert(getattr(record, field.name)) for field in fields(record)})


def validate_ppo_decision(step: Transition, config: Pi05ModelConfig) -> None:
    inputs, trace = step.obs.representations.get(INPUTS_KEY), step.policy_info.get(TRACE_KEY)
    if (
        "pi05_ppo_condition" in step.obs.representations
        or not isinstance(inputs, Pi05ReplayInputs)
        or not isinstance(trace, Pi05PPOTrace)
    ):
        raise TypeError(
            "pi05 PPO decision requires typed compact inputs and trace; cached-prefix records are unsupported"
        )
    batch = validate_replay_inputs(inputs)
    validate_selected_trace(trace, config, batch=batch)
    row = step.policy_info.get("sample_index")
    if type(row) is not int or not 0 <= row < trace.latent_before.shape[0]:
        raise ValueError("pi05 PPO decision has an invalid original invocation row")


def pack_ppo_decisions(steps: Sequence[Transition], *, device: torch.device) -> tuple[Pi05ReplayInputs, Pi05PPOTrace]:
    """Pack validated selected rows before H2D; original records remain unchanged."""
    inputs: list[Pi05ReplayInputs] = [step.obs.representations[INPUTS_KEY] for step in steps]
    traces: list[Pi05PPOTrace] = [step.policy_info[TRACE_KEY] for step in steps]
    rows = [step.policy_info["sample_index"] for step in steps]

    def combine(tensors: Sequence[torch.Tensor], *, transfer: bool = False) -> torch.Tensor:
        packed = torch.cat([tensor[row : row + 1] for tensor, row in zip(tensors, rows, strict=True)])
        return packed.to(device=device) if transfer else packed

    packed = Pi05ReplayInputs(
        images=tuple(combine([item.images[view] for item in inputs]) for view in range(2)),
        image_masks=combine([item.image_masks for item in inputs]),
        tokens=combine([item.tokens for item in inputs]),
        token_mask=combine([item.token_mask for item in inputs]),
        state=combine([item.state for item in inputs]),
    )
    trace = replace(
        traces[0],
        latent_before=combine([item.latent_before for item in traces], transfer=True),
        latent_after=combine([item.latent_after for item in traces], transfer=True),
        denoise_index=combine([item.denoise_index for item in traces], transfer=True),
    )
    return packed, trace


def sampled_decisions(
    observations: list[dict], sampled: list[dict], sources: list[dict], *, seed: int, version: int
) -> list[Trajectory]:
    return [
        sampled_decision(
            observation,
            result,
            source,
            representations={INPUTS_KEY: result["inputs"]},
            policy_info={TRACE_KEY: result["trace"]},
            seed=seed,
            version=version,
        )
        for observation, result, source in zip(observations, sampled, sources, strict=True)
    ]
