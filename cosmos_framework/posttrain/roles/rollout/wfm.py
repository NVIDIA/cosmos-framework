# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous online WFM rollout using the native Cosmos3 UniPC path."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch

from cosmos_framework.posttrain.model.protocol import OnlineFlowPolicy
from cosmos_framework.posttrain.roles.rollout.base import (
    ShardSpec,
    WeightReceiverBase,
    group_ids_for_prompt,
)
from cosmos_framework.posttrain.roles.rollout.validation import normalize_validation_semantics_identity
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    WFMConditioningInput,
)
from cosmos_framework.posttrain.utils.dist import run_rank_converged, world_size_or_1
from cosmos_framework.posttrain.utils.wfm import repeat_conditioning_batch

_MAX_SEED = (1 << 63) - 1


def _rank_or_zero() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


@dataclass(frozen=True)
class _PreparedWFMGroup:
    """One rank-local group validated before distributed sampling begins."""

    obs: ObsBundle
    model_batch: dict[str, Any]
    seeds: tuple[int, ...]
    sample_ids: tuple[str, ...]
    weight_version: int


def _exact_int(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"online WFM {name} must be an integer >= {minimum}, got {value!r}")
    return value


def _finite_real(value: object, name: str, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"WFM rollout {name} must be a finite real value, got {value!r}")
    result = float(value)
    valid = result >= minimum if inclusive else result > minimum
    if not isfinite(result) or not valid:
        relation = ">=" if inclusive else ">"
        raise ValueError(f"WFM rollout {name} must be finite and {relation} {minimum}, got {value!r}")
    return result


def _sampler_semantics_identity(sampler: Any) -> dict[str, Any]:
    """Bind declared sampler behavior or the exact native UniPC schedule config."""
    sampler_class = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
    declared = getattr(sampler, "validation_semantics_identity", None)
    if callable(declared):
        return normalize_validation_semantics_identity(
            {"class": sampler_class, "declared": declared()},
            context="WFM rollout sampler identity",
        )

    native_unipc_class = "cosmos_framework.model.generator.diffusion.samplers.unipc.UniPCSampler"
    if sampler_class != native_unipc_class:
        raise TypeError(f"WFM rollout sampler {sampler_class} must implement validation_semantics_identity()")
    config = getattr(sampler, "cfg", None)
    config_class = "" if config is None else f"{type(config).__module__}.{type(config).__qualname__}"
    native_config_class = "cosmos_framework.model.generator.diffusion.samplers.unipc.UniPCSamplerConfig"
    if config_class != native_config_class:
        raise TypeError(f"native WFM UniPC sampler has unsupported config {config_class!r}")
    num_train_timesteps = _exact_int(
        getattr(config, "num_train_timesteps", None),
        "sampler cfg.num_train_timesteps",
        minimum=1,
    )
    base_shift = _finite_real(
        getattr(config, "shift", None),
        "sampler cfg.shift",
        minimum=0.0,
        inclusive=False,
    )
    use_dynamic_shifting = getattr(config, "use_dynamic_shifting", None)
    if not isinstance(use_dynamic_shifting, bool):
        raise TypeError(f"WFM rollout sampler cfg.use_dynamic_shifting must be bool, got {use_dynamic_shifting!r}")
    return normalize_validation_semantics_identity(
        {
            "class": sampler_class,
            "config": {
                "class": config_class,
                "num_train_timesteps": num_train_timesteps,
                "shift": base_shift,
                "use_dynamic_shifting": use_dynamic_shifting,
            },
        },
        context="WFM rollout native UniPC sampler identity",
    )


class WFMOnlineRollout(WeightReceiverBase):
    """Generate G final latents per conditioning input; no denoising history."""

    def __init__(
        self,
        policy: OnlineFlowPolicy,
        *,
        sampler: Any,
        num_steps: int,
        shift: float,
        guidance: float,
        group_size: int | None = None,
    ) -> None:
        self.num_steps = _exact_int(num_steps, "num_steps", minimum=1)
        self.shift = _finite_real(shift, "shift", minimum=0.0, inclusive=False)
        self.guidance = _finite_real(guidance, "guidance", minimum=0.0, inclusive=True)
        self.group_size = None if group_size is None else _exact_int(group_size, "group_size", minimum=1)
        self.policy = policy
        self.model = policy
        self.sampler = sampler
        self._weight_version = 0

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Declare model and sampler semantics used by native WFM generation."""
        return normalize_validation_semantics_identity(
            {
                "version": 1,
                "backend": "cosmos3-native-wfm",
                "model": self.policy.reference_semantics_identity(),
                "sampler": _sampler_semantics_identity(self.sampler),
                "num_steps": self.num_steps,
                "shift": self.shift,
                "guidance": self.guidance,
                "group_size": self.group_size,
            },
            context="WFM rollout validation semantics identity",
        )

    def apply_weights(self, source: Any | None = None) -> None:
        """Use the shared receiver path; a colocated shared model is stamp-only."""
        from cosmos_framework.posttrain.comm.weight_transfer import pull_stream_from_peer

        pull_stream_from_peer(self, source)

    def prepare_recv(self) -> dict[str, ShardSpec]:
        """Report unsharded destination slices for the adapter's policy network."""
        return {
            name: ShardSpec(0, (tuple(value.shape)[0] if value.dim() else 0), 0)
            for name, value in self.policy.net.state_dict().items()
        }

    def apply_bucket(self, payload: Any, index: int) -> None:
        """Load one producer-driven bucket when a non-shared backend is configured."""
        backend = getattr(self, "_weight_backend", None)
        if backend is not None:
            backend.apply_bucket(self.policy.net, index, payload)

    def generate(
        self,
        prompts: list[ObsBundle],
        **knobs: Any,
    ) -> list[Trajectory]:
        def prepare() -> list[_PreparedWFMGroup]:
            world_size = world_size_or_1()
            if self.group_size is None:
                if "group" not in knobs:
                    raise ValueError("online WFM rollout requires a bound group_size or an explicit group")
                group = _exact_int(knobs["group"], "group", minimum=1)
            else:
                if world_size > 1 and self.group_size != world_size:
                    raise ValueError(
                        "distributed online WFM currently requires one member per rank: "
                        f"group_size={self.group_size}, world_size={world_size}"
                    )
                group = self.group_size if world_size == 1 else 1
                if "group" in knobs and _exact_int(knobs["group"], "group", minimum=1) != group:
                    raise ValueError(
                        f"online WFM group override {knobs['group']!r} differs from bound local group {group}"
                    )
            if world_size > 1 and (len(prompts) != 1 or group != 1):
                raise ValueError(
                    "distributed online WFM requires exactly one prompt and one local member per rank; "
                    f"got prompts={len(prompts)}, group={group}"
                )
            if "seed" not in knobs:
                raise ValueError("online WFM rollout requires an explicit integer seed")
            seed = _exact_int(knobs["seed"], "seed", minimum=0)
            if "weight_version" not in knobs:
                raise ValueError("online WFM rollout requires an explicit weight_version")
            weight_version = _exact_int(knobs["weight_version"], "weight_version", minimum=0)
            if weight_version != self._weight_version:
                raise RuntimeError(
                    f"online WFM requested weight_version={weight_version}, "
                    f"but the rollout holds version {self._weight_version}"
                )

            prepared: list[_PreparedWFMGroup] = []
            for input_index, obs in enumerate(prompts):
                if not isinstance(obs, ObsBundle):
                    raise TypeError(f"WFM rollout prompt {input_index} must be an ObsBundle")
                conditioning = obs.representations.get(WFM_CONDITIONING_KEY)
                if not isinstance(conditioning, WFMConditioningInput):
                    raise TypeError(
                        f"WFM rollout prompt {input_index} needs representations[{WFM_CONDITIONING_KEY!r}] "
                        "as WFMConditioningInput"
                    )
                if obs.id != conditioning.id:
                    raise ValueError(
                        f"WFM rollout prompt id {obs.id!r} does not match conditioning id {conditioning.id!r}"
                    )
                # One caller-owned base seed expands here over prompt, distributed rank,
                # and rank-local member, so callers cannot accidentally collapse replicas.
                rank = _rank_or_zero()
                first_seed = seed + (input_index * world_size + rank) * group
                final_seed = first_seed + group - 1
                if final_seed > _MAX_SEED:
                    raise ValueError(f"online WFM seed expansion exceeds {_MAX_SEED}: final seed {final_seed}")
                seeds = tuple(range(first_seed, final_seed + 1))
                group_id, _ = group_ids_for_prompt(obs, group)
                sample_ids = tuple(
                    f"{group_id}_r{rank}_k{sample_index}_s{sample_seed}"
                    for sample_index, sample_seed in enumerate(seeds)
                )
                prepared.append(
                    _PreparedWFMGroup(
                        obs=obs,
                        model_batch=repeat_conditioning_batch(conditioning, group),
                        seeds=seeds,
                        sample_ids=sample_ids,
                        weight_version=weight_version,
                    )
                )
            return prepared

        prepared_groups = run_rank_converged(prepare, "online WFM rollout preparation")

        trajectories: list[Trajectory] = []
        for input_index, prepared in enumerate(prepared_groups):

            def sample() -> list[torch.Tensor]:
                latents = self.policy.sample_final_latents(
                    prepared.model_batch,
                    sampler=self.sampler,
                    seeds=list(prepared.seeds),
                    num_steps=self.num_steps,
                    shift=self.shift,
                    guidance=self.guidance,
                )
                if not isinstance(latents, list) or len(latents) != len(prepared.seeds):
                    size = len(latents) if isinstance(latents, list) else "n/a"
                    raise ValueError(f"WFM rollout returned {size} latents for group={len(prepared.seeds)}")
                for latent in latents:
                    if not isinstance(latent, torch.Tensor):
                        raise TypeError(f"WFM final latent must be a tensor, got {type(latent).__name__}")
                    if latent.ndim < 2 or latent.shape[0] != 1:
                        shape = tuple(latent.shape)
                        raise ValueError(f"WFM final latent must keep a leading singleton sample axis, got {shape}")
                return latents

            latents = run_rank_converged(sample, f"online WFM rollout group {input_index} generation")

            def finalize() -> list[Trajectory]:
                group_trajectories: list[Trajectory] = []
                for sample_id, sample_seed, latent in zip(
                    prepared.sample_ids,
                    prepared.seeds,
                    latents,
                    strict=True,
                ):
                    transition = Transition(
                        obs=prepared.obs,
                        action=latent,
                        policy_info={"seed": sample_seed, "weight_version": prepared.weight_version},
                        done=True,
                    )
                    group_trajectories.append(
                        Trajectory(
                            transitions=[transition],
                            id=sample_id,
                            lineage=(prepared.obs.id,),
                            status=Status.COMPLETED,
                            group_id=prepared.obs.id,
                        )
                    )
                return group_trajectories

            trajectories.extend(run_rank_converged(finalize, f"online WFM rollout group {input_index} finalization"))
        return trajectories
