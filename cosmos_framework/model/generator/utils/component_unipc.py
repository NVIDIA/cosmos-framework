# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference coordination for jointly evaluated components with separate UniPC schedules."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cosmos_framework.utils.progress_bar import progress_bar
from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSamplerConfig


@dataclass(frozen=True)
class ComponentSchedule:
    """Shared component shifts and the corresponding sizes in each flat sample."""

    shifts: tuple[float, ...]
    lengths: list[tuple[int, ...]]


@torch.no_grad()
def sample_component_unipc(
    velocity_fn: Callable[[list[torch.Tensor], list[torch.Tensor]], list[torch.Tensor]],
    noise: list[torch.Tensor],  # B x [N_i]
    *,
    cfg: UniPCSamplerConfig,
    schedule: ComponentSchedule,
    num_steps: int,
    step_callback: Callable[[int, int], None] | None = None,
) -> list[torch.Tensor]:  # B x [N_i]
    """Advance generic flat components with one joint velocity evaluation per step.

    The caller defines component boundaries and adapts the component-ordered
    timesteps (each [1,1]) to its velocity function. Each sample/component owns
    an unchanged UniPC scheduler, including its sigma grid and multistep history.
    No common-clock derivative approximation or additional model call is used.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if cfg.use_dynamic_shifting:
        raise ValueError("Component UniPC schedules require fixed shifts")
    if not schedule.shifts or any(not math.isfinite(value) or value <= 0 for value in schedule.shifts):
        raise ValueError("Component shifts must be positive and finite")
    if not noise or len(noise) != len(schedule.lengths):
        raise ValueError("Component sampler sample metadata lengths differ")

    schedulers: list[list[FlowUniPCMultistepScheduler]] = []
    for sample, lengths in zip(noise, schedule.lengths, strict=True):
        if (
            sample.ndim != 1
            or len(lengths) != len(schedule.shifts)
            or any(length <= 0 for length in lengths)
            or sum(lengths) != sample.numel()
        ):
            raise ValueError("Component lengths must partition each flat sample into nonempty components")
        sample_schedulers: list[FlowUniPCMultistepScheduler] = []
        for shift in schedule.shifts:
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=cfg.num_train_timesteps,
                shift=cfg.shift,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(num_steps, device=sample.device, shift=shift)
            sample_schedulers.append(scheduler)
        schedulers.append(sample_schedulers)

    latent = noise  # B x [N_i]
    for step_index in progress_bar(range(num_steps), desc="Sampling", total=num_steps):
        if step_callback is not None:
            step_callback(step_index, num_steps)
        timesteps = [scheduler.timesteps[step_index].reshape(1, 1) for scheduler in schedulers[0]]  # K x [1,1]
        predictions = velocity_fn(latent, timesteps)  # B x [N_i]
        if len(predictions) != len(latent):
            raise ValueError("Component sampler prediction batch length differs")
        updated: list[torch.Tensor] = []
        for sample, prediction, lengths, sample_schedulers in zip(
            latent, predictions, schedule.lengths, schedulers, strict=True
        ):
            if prediction.shape != sample.shape:
                raise ValueError("Component sampler prediction shape differs from its sample")
            sample_parts = sample.split(lengths)  # K x [N_ik]
            prediction_parts = prediction.split(lengths)  # K x [N_ik]
            parts: list[torch.Tensor] = []
            for scheduler, part, velocity in zip(sample_schedulers, sample_parts, prediction_parts, strict=True):
                # Native UniPC's multistep einsums require a leading batch dimension.
                parts.append(
                    scheduler.step(
                        model_output=velocity,
                        timestep=scheduler.timesteps[step_index],
                        sample=part.unsqueeze(0),  # [1,N_ik]
                        return_dict=False,
                    )[0].squeeze(0)  # [N_ik]
                )
            updated.append(torch.cat(parts))  # [N_i]
        latent = updated  # B x [N_i]
    return latent
