# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint actor/value objective over the model-owned decision scoring capability."""

from __future__ import annotations

from typing import Literal

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, TARGET_KEY, validate_decision
from cosmos_framework.posttrain.algorithm.ppo.losses import ppo_policy_loss, ppo_value_loss
from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    MetricAdapter,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.model.protocol import BatchedDecisionPPOPolicy, DecisionPPOPolicy, PPODecisionScores
from cosmos_framework.posttrain.schema import Trajectory, Transition
from cosmos_framework.posttrain.utils.dist import rank_max_count, rank_mean_count, run_rank_converged

WEIGHT_KEY = "train/valid_decisions"
WEIGHTED = frozenset(
    f"train/{name}" for name in ("policy_loss", "value_loss", "joint_loss", "clip_fraction", "ratio_mean")
)


def make_joint_objective(
    *,
    policy_clip: float,
    value_clip: float,
    dual_clip: float | None,
    huber_delta: float,
    identity: dict,
    loss_reduction: Literal["decision_mean", "fixed_episode_mean"] = "decision_mean",
    microbatch: int | None = None,
) -> Objective:
    fixed_episode = loss_reduction == "fixed_episode_mean"
    names = {
        f"ppo_{key}": key
        for key in ("policy_loss", "value_loss", "joint_loss", "clip_fraction", "ratio_mean", "valid_decisions")
    }
    if fixed_episode:
        names["ppo_decisions"] = "decisions"
    adapter = MetricAdapter(
        names={**names, OBJECTIVE_WEIGHT_SUM: "weight_sum"},
        internal_prefix="ppo_",
        phases=frozenset({"train"}),
        weighted_by={
            f"ppo_{key}": ("ppo_decisions" if fixed_episode and key.endswith("loss") else "ppo_valid_decisions")
            for key in ("policy_loss", "value_loss", "joint_loss", "clip_fraction", "ratio_mean")
        },
    )

    def objective(model: DecisionPPOPolicy, trajectories: list[Trajectory]) -> tuple[torch.Tensor, dict]:
        device = next(model.net.parameters()).device
        batched = isinstance(model, BatchedDecisionPPOPolicy)

        def prepare() -> list[Transition]:
            version = trajectories[0].transitions[0].policy_info["weight_version"] if trajectories else 0
            steps = [validate_decision(t, version, training=True) for t in trajectories]
            if batched:
                if type(microbatch) is not int or microbatch < 1 or len(steps) > microbatch:
                    raise ValueError("joint PPO batched scoring requires a positive microbatch covering all decisions")
                if len(steps) < microbatch and model.ppo_padding is None:
                    raise ValueError("joint PPO batched scoring requires a genuine padding record")
            else:
                for step in steps:
                    model.validate_ppo_record(step)
            return steps

        steps = run_rank_converged(prepare, "joint PPO objective inputs")
        mean_count = rank_mean_count(len(steps), device)
        if mean_count <= 0:
            raise ValueError("joint PPO minibatch is globally empty")
        valid_count = sum(step.algo_extra[TARGET_KEY].loss_mask for step in steps)
        # Zero-contribution ranks still enter every FSDP forward. The role binds one
        # genuine cached sample as its padding template before opening the window.
        padding = model.ppo_padding
        loss = torch.zeros((), device=device)
        totals = {key: torch.zeros((), device=device) for key in WEIGHTED}

        def terms(index: int, step: Transition, scores: PPODecisionScores) -> tuple[torch.Tensor, dict]:
            if index >= len(steps) or not step.algo_extra[TARGET_KEY].loss_mask:
                return scores.zero_loss, {}
            stats, targets = step.algo_extra[STATS_KEY], step.algo_extra[TARGET_KEY]
            old = torch.tensor([stats.old_logprob], device=device)
            advantage = torch.tensor([targets.advantage], device=device)
            mask = torch.ones_like(old, dtype=torch.bool)
            actor, metrics = ppo_policy_loss(scores.logprobs, old, advantage, mask, policy_clip, dual_clip=dual_clip)
            value, _ = ppo_value_loss(
                scores.values,
                torch.tensor([stats.old_value], device=device),
                torch.tensor([targets.returns], device=device),
                mask,
                value_clip,
                huber_delta=huber_delta,
            )
            actor = actor * targets.loss_weight
            value = value * targets.loss_weight
            joint = actor + value
            return joint, {
                "train/policy_loss": actor.detach(),
                "train/value_loss": value.detach(),
                "train/joint_loss": joint.detach(),
                "train/clip_fraction": metrics["clip_fraction"],
                "train/ratio_mean": metrics["ratio_mean"],
            }

        if batched:
            batch = steps + [padding] * (microbatch - len(steps))
            scores = model.score_ppo_decisions(batch)

            def batch_terms() -> tuple[torch.Tensor, dict]:
                batch_loss = scores.zero_loss
                batch_totals = dict(totals)
                for index, step in enumerate(steps):
                    selected = PPODecisionScores(
                        scores.logprobs[index : index + 1], scores.values[index : index + 1], scores.zero_loss
                    )
                    value, metrics = terms(index, step, selected)
                    batch_loss = batch_loss + value / mean_count
                    for key, value in metrics.items():
                        batch_totals[key] = batch_totals[key] + value
                return batch_loss, batch_totals

            loss, totals = run_rank_converged(batch_terms, "joint PPO objective terms")
        else:
            maximum = rank_max_count(len(steps), device)
            for index in range(maximum):
                step = steps[index] if index < len(steps) else padding
                scores = model.score_ppo_decision(step)
                value, metrics = run_rank_converged(lambda: terms(index, step, scores), "joint PPO objective terms")
                loss = loss + value / mean_count
                for key, value in metrics.items():
                    totals[key] = totals[key] + value
        return loss, {
            **{
                f"ppo_{key.removeprefix('train/')}": value
                / max(1, len(steps) if fixed_episode and key.endswith("loss") else valid_count)
                for key, value in totals.items()
            },
            "ppo_valid_decisions": valid_count,
            **({"ppo_decisions": len(steps)} if fixed_episode else {}),
            OBJECTIVE_WEIGHT_SUM: mean_count,
        }

    return Objective(objective, reduction=Reduction.TOKEN_SUM, metric_adapter=adapter, identity=identity)
