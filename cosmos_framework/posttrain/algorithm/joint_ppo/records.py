# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable joint PPO statistics, targets and decision validation."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, ObsBundle, SampleSource, Status, Trajectory, Transition

STATS_KEY = "joint_ppo"
TARGET_KEY = "joint_ppo_targets"


@dataclass(frozen=True)
class JointPPOStats:
    old_logprob: float
    old_value: float
    bootstrap_value: float = 0.0


@dataclass(frozen=True)
class JointPPOTargets:
    advantage: float
    returns: float
    loss_mask: bool = True
    loss_weight: float = 1.0


@dataclass(frozen=True)
class JointPPOTargetDiagnostic:
    """Scalar facts from target construction, joined to rollout feedback by sample identity."""

    sample_id: str
    old_value: float
    bootstrap_value: float
    reward: float
    returns: float
    raw_advantage: float
    normalized_advantage: float
    terminated: bool
    truncated: bool
    window_end: bool
    executed_steps: int
    loss_mask: bool = True
    loss_weight: float = 1.0


def sampled_decision(
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    representations: Mapping[str, Any],
    policy_info: Mapping[str, Any],
    seed: int,
    version: int,
) -> Trajectory:
    """Own raw observations while retaining the model's shared invocation records.

    ``result`` supplies action, old_logprob, old_value and sample_index. ``source``
    supplies sample_id, episode_id, fragment_id, session and step_idx; all source
    fields are preserved in transition info. Model representations and policy
    metadata retain their original references.
    """
    identifier = source["sample_id"]
    step = Transition(
        obs=ObsBundle(id=identifier, representations={"raw": copy.deepcopy(observation), **representations}),
        action=result["action"],
        done=False,
        step_idx=source["step_idx"],
        source=SampleSource(identifier, source["episode_id"], source["fragment_id"]),
        policy_info={
            **policy_info,
            "weight_version": version,
            "sampling_seed": seed,
            "sample_index": result["sample_index"],
        },
        algo_extra={STATS_KEY: JointPPOStats(result["old_logprob"], result["old_value"])},
        info=dict(source),
    )
    return Trajectory([step], id=identifier, session=source["session"], group_id=identifier, status=Status.COMPLETED)


def validate_decision(trajectory: Trajectory, version: int, *, training: bool = False) -> Transition:
    if len(trajectory.transitions) != 1:
        raise ValueError(f"joint PPO sample {trajectory.id!r} must contain one decision")
    step = trajectory.transitions[0]
    if step.source is None or step.source.sample_id != trajectory.id:
        raise ValueError(f"joint PPO sample {trajectory.id!r} has inconsistent source identity")
    if step.policy_info.get("weight_version") != version:
        raise ValueError(f"joint PPO sample {trajectory.id!r} has stale behavior version")
    if step.feedback is None or step.next_obs is None:
        raise ValueError(f"joint PPO sample {trajectory.id!r} has no environment feedback/next observation")
    if not isinstance(step.reward, Mapping) or FINAL_REWARD_KEY not in step.reward:
        raise ValueError(f"joint PPO sample {trajectory.id!r} is missing training reward {FINAL_REWARD_KEY!r}")
    reward = step.reward[FINAL_REWARD_KEY]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"joint PPO sample {trajectory.id!r} has invalid final reward")
    stats = step.algo_extra.get(STATS_KEY)
    if not isinstance(stats, JointPPOStats) or not all(
        math.isfinite(getattr(stats, f.name)) for f in dataclasses.fields(stats)
    ):
        raise ValueError(f"joint PPO sample {trajectory.id!r} has invalid old statistics")
    if training:
        targets = step.algo_extra.get(TARGET_KEY)
        if (
            not isinstance(targets, JointPPOTargets)
            or type(targets.loss_mask) is not bool
            or isinstance(targets.loss_weight, bool)
            or not all(math.isfinite(value) for value in (targets.advantage, targets.returns, targets.loss_weight))
            or targets.loss_weight <= 0
        ):
            raise ValueError(f"joint PPO sample {trajectory.id!r} has invalid targets")
    return step


class PreparedJointBatch(TypedDict):
    """A validated source window with frozen targets and explicit behavior-check status."""

    batch_id: str
    decisions: int
    policy_version: int
    behavior_verified: bool
    max_logprob_delta: float | None
    initial_clip_fraction: float | None
    target_diagnostics: list[dict[str, str | int | float | bool]]
    target_metrics: dict[str, float]


class JointUpdateReceipt(TypedDict):
    """Already-reduced update results; publication advances separately in the controller."""

    optimizer_step: int
    metrics: dict[str, float]
    optimizer_metrics: list[dict[str, float]]
    learning_rates: dict[str, float]
    parameter_update_l2: dict[str, float]
