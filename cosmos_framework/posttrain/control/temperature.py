# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU-less objective-specific policy/rollout temperature contracts.

Sampling temperature, policy-loss temperature, and teacher-scoring temperature are distinct
configuration fields.  This module does not synchronize them.  It records the resolved semantics
and rejects only combinations that invalidate the estimator implemented by a supported recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

Algorithm = Literal["grpo", "opd"]
_VLLM_MIN_TEMPERATURE = 0.01


@dataclass(frozen=True)
class SamplingSemantics:
    """The behavior distribution controls relevant to policy-gradient correctness."""

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True)
class TemperatureContract:
    """One resolved algorithm contract spanning rollout, policy loss, and optional teacher."""

    algorithm: Algorithm
    objective: str
    rollout: SamplingSemantics
    policy_temperature: float
    use_rollout_logprobs_for_loss: bool = False
    teacher_signal_mode: str | None = None
    teacher_temperature: float | None = None


@dataclass(frozen=True)
class TemperatureCheck:
    """Pure compatibility result; every message is a semantic configuration error."""

    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether the configured distributions satisfy the selected objective."""
        return not self.errors


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-bool real number, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return resolved


def _top_p(value: Any) -> float:
    resolved = _positive_real("rollout.top_p", value)
    if resolved > 1:
        raise ValueError(f"rollout.top_p must be in (0, 1], got {value!r}")
    return resolved


def _top_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"rollout.top_k must be a non-bool integer, got {value!r}")
    return value


def _boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool, got {value!r}")
    return value


def _sampling(rollout: Any) -> SamplingSemantics:
    temperature = _positive_real("rollout.temperature", rollout.temperature)
    if temperature < _VLLM_MIN_TEMPERATURE:
        raise ValueError(
            "rollout.temperature must be 0.01 or greater for the current vLLM backend: "
            f"vLLM rewrites values in (0, 0.01) to 0.01, got {rollout.temperature!r}"
        )
    return SamplingSemantics(
        # Every supported online recipe requests more than one sample per prompt. vLLM rejects
        # greedy generation (temperature=0) when n>1, so zero is not a runnable grouped rollout.
        temperature=temperature,
        top_p=_top_p(rollout.top_p),
        top_k=_top_k(rollout.top_k),
        repetition_penalty=_positive_real("rollout.repetition_penalty", rollout.repetition_penalty),
    )


def grpo_temperature_contract(rl: Any, rollout: Any) -> TemperatureContract:
    """Resolve the GRPO behavior/policy contract without mutating either config tree."""
    return TemperatureContract(
        algorithm="grpo",
        objective=str(rl.grpo_surrogate),
        rollout=_sampling(rollout),
        policy_temperature=_positive_real("rl.temperature", rl.temperature),
        use_rollout_logprobs_for_loss=_boolean("rl.use_rollout_logprobs_for_loss", rl.use_rollout_logprobs_for_loss),
    )


def opd_temperature_contract(rl: Any, rollout: Any) -> TemperatureContract:
    """Resolve the OPD rollout/policy/teacher contract without mutating config."""
    return TemperatureContract(
        algorithm="opd",
        objective=str(rl.distillation_mode),
        rollout=_sampling(rollout),
        policy_temperature=_positive_real("rl.temperature", rl.temperature),
        use_rollout_logprobs_for_loss=_boolean("rl.use_rollout_logprobs_for_loss", rl.use_rollout_logprobs_for_loss),
        teacher_signal_mode=str(rl.teacher.signal_mode),
        teacher_temperature=_positive_real("teacher.scoring_temperature", rl.teacher.scoring_temperature),
    )


def _rollout_warp_errors(rollout: SamplingSemantics, *, owner: str) -> list[str]:
    errors: list[str] = []
    if rollout.top_p != 1.0:
        errors.append(f"{owner} requires rollout.top_p=1.0, got {rollout.top_p:g}")
    if rollout.top_k > 0:
        errors.append(f"{owner} requires rollout.top_k disabled (<=0), got {rollout.top_k}")
    if rollout.repetition_penalty != 1.0:
        errors.append(f"{owner} requires rollout.repetition_penalty=1.0, got {rollout.repetition_penalty:g}")
    return errors


def check_temperature(contract: TemperatureContract) -> TemperatureCheck:
    """Check the objective-specific distribution identities used by supported recipes."""
    errors: list[str] = []
    rollout = contract.rollout
    policy_temperature = contract.policy_temperature

    if contract.algorithm == "grpo":
        if contract.objective not in ("legacy_pg", "clipped"):
            errors.append(f"unsupported GRPO objective {contract.objective!r}")
        if rollout.temperature != policy_temperature:
            errors.append(
                "on-policy GRPO requires rollout.temperature == rl.temperature, got "
                f"{rollout.temperature:g} and {policy_temperature:g}"
            )
        errors.extend(_rollout_warp_errors(rollout, owner="on-policy GRPO"))
        if contract.use_rollout_logprobs_for_loss:
            if contract.objective == "legacy_pg":
                errors.append(
                    "legacy GRPO does not consume rollout logprobs; set rl.use_rollout_logprobs_for_loss=false"
                )
            elif policy_temperature != 1.0:
                errors.append(
                    "clipped GRPO rollout-logprob loss mode requires "
                    "rollout.temperature == rl.temperature == 1.0; the current vLLM rollout backend "
                    "returns raw, untempered logprobs"
                )
        return TemperatureCheck(tuple(errors))

    if contract.algorithm != "opd":
        return TemperatureCheck((f"unsupported temperature-contract algorithm {contract.algorithm!r}",))
    teacher_temperature = contract.teacher_temperature
    if teacher_temperature is None:
        return TemperatureCheck(("OPD temperature contract has no teacher-scoring temperature",))

    if contract.teacher_signal_mode == "topk":
        if teacher_temperature != 1.0 or policy_temperature != 1.0:
            errors.append(
                "top-k OPD requires teacher.scoring_temperature == rl.temperature == 1.0; "
                f"got teacher={teacher_temperature:g}, policy={policy_temperature:g}; "
                "the current forced-sequence teacher path returns untempered prompt logprobs"
            )
        if contract.objective == "advantage":
            if rollout.temperature != policy_temperature:
                errors.append(
                    "advantage OPD requires rollout.temperature == rl.temperature, got "
                    f"{rollout.temperature:g} and {policy_temperature:g}"
                )
            errors.extend(_rollout_warp_errors(rollout, owner="advantage OPD"))
        elif contract.objective != "direct":
            errors.append(f"unsupported top-k OPD objective {contract.objective!r}")
        return TemperatureCheck(tuple(errors))

    if contract.teacher_signal_mode == "dpca":
        for name, value in (
            ("rollout.temperature", rollout.temperature),
            ("rl.temperature", policy_temperature),
            ("teacher.scoring_temperature", teacher_temperature),
        ):
            if value != 1.0:
                errors.append(f"DPCA requires {name}=1.0, got {value:g}")
        errors.extend(_rollout_warp_errors(rollout, owner="DPCA"))
        return TemperatureCheck(tuple(errors))

    return TemperatureCheck((f"unsupported OPD teacher signal mode {contract.teacher_signal_mode!r}",))
