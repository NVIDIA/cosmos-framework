# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed decision targets and their mergeable scalar diagnostics."""

from collections.abc import Sequence

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import (
    STATS_KEY,
    JointPPOTargetDiagnostic,
    JointPPOTargets,
)
from cosmos_framework.posttrain.algorithm.ppo.gae import compute_gae
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Trajectory, Transition


def fixed_episode_targets(
    samples: Sequence[Trajectory], *, gamma: float, gae_lambda: float
) -> tuple[JointPPOTargets, ...]:
    """Prepare unnormalized targets for complete, fixed-duration physical episodes.

    Raw control endings stay intact. The first ending's action is valid, its
    entire decision chunk participates in GAE, and later chunks contribute zero.
    The complete episode supplies T while raw control facts supply its valid L;
    the objective averages ``loss * mask * T/L`` over all recorded decisions.
    """
    episodes: dict[tuple[str | None, str], list[Transition]] = {}
    for sample in samples:
        step = sample.transitions[0]
        feedback = step.feedback
        if feedback.control_terminated is None or feedback.episode_complete is None:
            raise ValueError(f"fixed episode sample {sample.id!r} requires control endings and episode_complete")
        feedback.__post_init__()
        if step.done != any((*feedback.control_terminated, *feedback.control_truncated)):
            raise ValueError(f"fixed episode sample {sample.id!r} done differs from its control endings")
        if step.truncated != any(feedback.control_truncated):
            raise ValueError(f"fixed episode sample {sample.id!r} truncated differs from its control endings")
        episodes.setdefault((sample.session, step.source.episode_id), []).append(step)

    ordered = [sorted(episode, key=lambda step: step.step_idx) for episode in episodes.values()]
    for episode in ordered:
        if [step.step_idx for step in episode] != list(range(len(episode))):
            raise ValueError(f"fixed episode {episode[0].source.episode_id!r} has missing/duplicate decision indices")
        if any(step.feedback.episode_complete for step in episode[:-1]) or not episode[-1].feedback.episode_complete:
            raise ValueError(f"fixed episode {episode[0].source.episode_id!r} is not one complete physical episode")
        if any(step.collection_end for step in episode[:-1]):
            raise ValueError(f"fixed episode {episode[0].source.episode_id!r} crosses its collection boundary")

    targets = {}
    for episode in ordered:
        counts = [step.feedback.executed_steps for step in episode]
        total = sum(counts)
        endings = [
            terminated or truncated
            for step in episode
            for terminated, truncated in zip(
                step.feedback.control_terminated, step.feedback.control_truncated, strict=True
            )
        ]
        valid_controls = next((index + 1 for index, ended in enumerate(endings) if ended), total)
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        valid = torch.tensor([[offset < valid_controls for offset in offsets[:-1]]], dtype=torch.bool)  # [1,T]
        rewards = torch.tensor([[step.reward[FINAL_REWARD_KEY] for step in episode]], dtype=torch.float32)  # [1,T]
        values = torch.tensor(
            [[step.algo_extra[STATS_KEY].old_value for step in episode] + [0.0]], dtype=torch.float32
        )  # [1,T+1]
        _, returns = compute_gae(
            rewards,
            values,
            valid,
            torch.tensor([True]),
            torch.tensor([False]),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for index, step in enumerate(episode):
            targets[step.source.sample_id] = JointPPOTargets(
                # The reference forms returns first, then subtracts values in FP32.
                float(returns[0, index] - values[0, index]) if valid[0, index] else 0.0,
                float(returns[0, index]),
                bool(valid[0, index]),
                total / valid_controls,
            )
    return tuple(targets[sample.id] for sample in samples)


def target_diagnostic_metrics(records: list[JointPPOTargetDiagnostic]) -> dict[str, float]:
    """Global decision statistics; a constant target has no defined explained variance."""
    records = [row for row in records if row.loss_mask]
    values = torch.tensor([row.old_value for row in records], dtype=torch.float64)
    returns = torch.tensor([row.returns for row in records], dtype=torch.float64)
    raw = torch.tensor([row.raw_advantage for row in records], dtype=torch.float64)
    variance = returns.var(correction=0)
    result = {
        "train/old_value_mean": float(values.mean()),
        "train/target_return_mean": float(returns.mean()),
        "train/raw_advantage_mean": float(raw.mean()),
        "train/raw_advantage_std": float(raw.std(correction=0)),
        "train/bootstrap_value_mean": sum(row.bootstrap_value for row in records) / len(records),
        "train/nonzero_reward_decisions": float(sum(row.reward != 0 for row in records)),
        "train/value_explained_variance_defined": float(variance > 0),
    }
    if variance > 0:
        result["train/value_explained_variance"] = float(1 - (returns - values).var(correction=0) / variance)
    return result
