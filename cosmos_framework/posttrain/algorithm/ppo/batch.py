# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous PPO batch preparation and deterministic scheduling."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from cosmos_framework.posttrain.algorithm.ppo.gae import compute_gae
from cosmos_framework.posttrain.algorithm.ppo.losses import normalize_advantages, sampled_k3_kl
from cosmos_framework.posttrain.algorithm.ppo.records import PPOStepTargets, PPOTargetStep
from cosmos_framework.posttrain.algorithm.ppo.views import (
    CriticScore,
    PPOEpisode,
    PPOTrainingEpisode,
    ReferenceScore,
    strict_join_ppo_results,
)


@dataclass(frozen=True)
class PPOMinibatch:
    """One optimizer minibatch, subdivided into fixed-size microbatches."""

    epoch: int
    minibatch_index: int
    microbatches: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PPOUpdatePlan:
    """The schedule PARAMETERS for one prepared batch — never the schedule itself.

    This is the whole of what the controller hands a trainable role for an update: recipe values it
    read out of `config/rl.py`, with seed derivation owned by the recipe. Text PPO offsets it by
    publication; Joint PPO preserves the reference's fixed per-rank shuffle. Expanding these
    parameters into epochs, global minibatches and this rank's micro-batches is the
    ROLE's job, because only the role knows its live `ParallelDims` and which contiguous shard of
    the global episode order its cache actually holds. A controller that shipped a materialized
    schedule instead would have to guess both, which is exactly how a schedule and a cache come to
    disagree.
    """

    epochs: int
    num_minibatches: int
    microbatch_size: int
    seed: int
    rank_seeded_torch_shuffle: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "num_minibatches", "microbatch_size"):
            _require_positive_integer(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")


def make_ppo_update_plan(config: object, policy_version: int) -> PPOUpdatePlan:
    """Build one iteration's immutable role-owned PPO update parameters."""
    return PPOUpdatePlan(
        epochs=int(getattr(config, "ppo_epochs")),
        num_minibatches=int(getattr(config, "num_minibatches")),
        microbatch_size=int(getattr(config, "microbatch_size")),
        seed=int(getattr(config, "seed")) + policy_version,
    )


def prepare_ppo_training_batch(
    episodes: Sequence[PPOEpisode],
    reference_results: Sequence[ReferenceScore],
    critic_results: Sequence[CriticScore],
    *,
    gamma: float,
    gae_lambda: float,
    reference_kl_coef: float,
    reduce_stats: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[PPOTrainingEpisode, ...]:
    """Strictly join role outputs and construct immutable token-level PPO targets.

    Reference-KL shaping uses rollout-time behavior log-probabilities. Those
    statistics and the old critic estimates remain immutable for every PPO
    epoch; only the returned normalized policy advantages differ from raw GAE.
    """
    if not math.isfinite(reference_kl_coef) or reference_kl_coef < 0.0:
        raise ValueError(f"reference_kl_coef must be finite and >= 0, got {reference_kl_coef}")
    joined = strict_join_ppo_results(list(episodes), list(reference_results), list(critic_results))
    policy_versions = {item.episode.policy_version for item in joined}
    if len(policy_versions) != 1:
        raise ValueError("a PPO on-policy batch must contain exactly one policy_version")

    batch_size = len(joined)
    max_actions = max(len(item.episode.action_token_ids) for item in joined)
    rewards = torch.zeros((batch_size, max_actions), dtype=torch.float64)  # [B,T]
    values = torch.zeros((batch_size, max_actions + 1), dtype=torch.float64)  # [B,T+1]
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)  # [B,T]
    terminated = torch.zeros(batch_size, dtype=torch.bool)  # [B]
    truncated = torch.zeros(batch_size, dtype=torch.bool)  # [B]

    for row, scored in enumerate(joined):
        episode = scored.episode
        action_length = len(episode.action_token_ids)
        row_mask = torch.tensor(episode.action_mask, dtype=torch.bool)  # [Te]
        old_logprobs = torch.tensor(episode.old_logprobs, dtype=torch.float64)  # [Te]
        reference_logprobs = torch.tensor(scored.reference.action_logprobs, dtype=torch.float64)  # [Te]
        process_rewards = torch.tensor(episode.process_rewards, dtype=torch.float64)  # [Te]
        kl_penalties = sampled_k3_kl(reference_logprobs, old_logprobs)  # [Te]
        shaped_rewards = process_rewards - reference_kl_coef * kl_penalties  # [Te]
        shaped_rewards = torch.where(row_mask, shaped_rewards, torch.zeros_like(shaped_rewards))  # [Te]
        final_action = int(row_mask.sum().item()) - 1
        shaped_rewards[final_action] += episode.terminal_reward

        rewards[row, :action_length] = shaped_rewards
        values[row, : action_length + 1] = torch.tensor(scored.critic.values, dtype=torch.float64)  # [Te+1]
        action_mask[row, :action_length] = row_mask
        terminated[row] = episode.terminated
        truncated[row] = episode.truncated

    value_advantages, returns = compute_gae(
        rewards,
        values,
        action_mask,
        terminated,
        truncated,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )  # [B,T], [B,T]
    advantages = normalize_advantages(value_advantages, action_mask, reduce_stats=reduce_stats)  # [B,T]

    training_episodes: list[PPOTrainingEpisode] = []
    for row, scored in enumerate(joined):
        action_length = len(scored.episode.action_token_ids)
        training_episodes.append(
            PPOTrainingEpisode(
                scored=scored,
                action_rewards=tuple(float(value) for value in rewards[row, :action_length].tolist()),
                advantages=tuple(float(value) for value in advantages[row, :action_length].tolist()),
                returns=tuple(float(value) for value in returns[row, :action_length].tolist()),
                value_advantages=tuple(float(value) for value in value_advantages[row, :action_length].tolist()),
            )
        )
    return tuple(training_episodes)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _shard_permutation_seed(seed: int, epoch: int, dp_rank: int) -> int:
    """Derive one stable per-(epoch, DP coordinate) permutation seed.

    A single sequential ``random.Random(seed)`` cannot serve here: each rank draws only its OWN
    shard's permutations, so consuming the stream in a different order would give the ranks
    inconsistent — and, worse, silently overlapping — schedules. A content-derived seed makes each
    (epoch, dp_rank) draw independent of what any other rank drew, and reproducible across runs.
    """
    digest = hashlib.blake2b(f"{seed}:{epoch}:{dp_rank}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def ppo_minibatches(
    batch: Sequence[str],
    *,
    epochs: int,
    num_minibatches: int,
    microbatch_size: int,
    seed: int,
    dp_rank: int = 0,
    dp_world: int = 1,
    shard_sizes: tuple[int, ...] | None = None,
    rank_seeded_torch_shuffle: bool = False,
) -> tuple[PPOMinibatch, ...]:
    """Schedule local cached samples, with equal forward counts on every rank.

    Permutations stay within each cache shard. ``shard_sizes`` comes from the roles'
    actual caches after decision projection; it cannot be inferred from the number of
    transported episode fragments. Uniform ingress retains its existing permutation
    and minibatch membership. Uneven shares receive empty microbatches, whose public
    objective forwards have zero contribution rather than duplicating a real sample.
    """
    if not batch:
        raise ValueError("cannot schedule an empty PPO batch")
    if any(type(episode_id) is not str for episode_id in batch):
        raise TypeError("batch must contain only exact string episode IDs")
    if any(not episode_id for episode_id in batch):
        raise ValueError("episode IDs must be nonempty")
    if len(set(batch)) != len(batch):
        raise ValueError("episode IDs must be unique")
    _require_positive_integer(epochs, "epochs")
    _require_positive_integer(num_minibatches, "num_minibatches")
    _require_positive_integer(microbatch_size, "microbatch_size")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    _require_positive_integer(dp_world, "dp_world")
    if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or not 0 <= dp_rank < dp_world:
        raise ValueError(f"dp_rank {dp_rank!r} is outside model data-parallel world {dp_world}")

    batch_size = len(batch)
    if num_minibatches > batch_size:
        raise ValueError("num_minibatches cannot exceed the number of PPO samples")
    if shard_sizes is None:
        if batch_size % dp_world:
            raise ValueError("uneven PPO cache ownership requires explicit shard_sizes")
        shard_sizes = (batch_size // dp_world,) * dp_world
    if (
        len(shard_sizes) != dp_world
        or any(type(size) is not int or size < 0 for size in shard_sizes)
        or sum(shard_sizes) != batch_size
    ):
        raise ValueError("shard_sizes must exactly partition the global PPO sample order")
    offsets = [sum(shard_sizes[:rank]) for rank in range(dp_world)]
    schedule: list[PPOMinibatch] = []
    for epoch in range(epochs):
        permutations: list[list[int]] = []
        for rank, size in enumerate(shard_sizes):
            if rank_seeded_torch_shuffle:
                generator = torch.Generator().manual_seed(seed + rank + epoch)
                indices = torch.randperm(size, generator=generator).tolist()
            else:
                indices = list(range(size))
                random.Random(_shard_permutation_seed(seed, epoch, rank)).shuffle(indices)
            permutations.append(indices)
        ordered = [
            (rank, offsets[rank] + permutations[rank][position])
            for position in range(max(shard_sizes))
            for rank in range(dp_world)
            if position < shard_sizes[rank]
        ]
        start = 0
        for minibatch_index in range(num_minibatches):
            size = batch_size // num_minibatches + int(minibatch_index < batch_size % num_minibatches)
            selected = ordered[start : start + size]
            start += size
            local = tuple(batch[index] for rank, index in selected if rank == dp_rank)
            maximum = max(sum(rank == candidate for rank, _index in selected) for candidate in range(dp_world))
            chunks = (maximum + microbatch_size - 1) // microbatch_size
            microbatches = tuple(
                local[index * microbatch_size : (index + 1) * microbatch_size] for index in range(chunks)
            )
            schedule.append(PPOMinibatch(epoch, minibatch_index, microbatches))
    return tuple(schedule)


def ppo_training_batch_metrics(batch: Sequence[PPOTrainingEpisode]) -> dict[str, float]:
    """Summarize immutable rollout, reward, and GAE targets for reporting."""
    if not batch:
        raise ValueError("cannot summarize an empty PPO training batch")
    rewards: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    value_advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    reference_kl: list[torch.Tensor] = []
    for item in batch:
        episode = item.scored.episode
        mask = torch.tensor(episode.action_mask, dtype=torch.bool)  # [T]
        rewards.append(torch.tensor(item.action_rewards, dtype=torch.float64)[mask])
        advantages.append(torch.tensor(item.advantages, dtype=torch.float64)[mask])
        value_advantages.append(torch.tensor(item.value_advantages, dtype=torch.float64)[mask])
        returns.append(torch.tensor(item.returns, dtype=torch.float64)[mask])
        values.append(torch.tensor(item.scored.critic.values[:-1], dtype=torch.float64)[mask])
        old = torch.tensor(episode.old_logprobs, dtype=torch.float64)  # [T]
        reference = torch.tensor(item.scored.reference.action_logprobs, dtype=torch.float64)  # [T]
        reference_kl.append(sampled_k3_kl(reference, old)[mask])
    joined = {
        "reward": torch.cat(rewards),
        "advantage": torch.cat(advantages),
        "value_advantage": torch.cat(value_advantages),
        "return": torch.cat(returns),
        "value": torch.cat(values),
        "reference_kl": torch.cat(reference_kl),
    }
    if any(values.numel() == 0 or not bool(torch.isfinite(values).all()) for values in joined.values()):
        raise ValueError("PPO training batch metrics received empty or non-finite values")
    metrics = {
        "batch/reward_mean": float(joined["reward"].mean()),
        "batch/advantage_mean": float(joined["advantage"].mean()),
        "batch/advantage_std": float(joined["advantage"].std(unbiased=False)),
        "batch/value_advantage_mean": float(joined["value_advantage"].mean()),
        "batch/return_mean": float(joined["return"].mean()),
        "batch/value_mean": float(joined["value"].mean()),
        "batch/reference_kl_mean": float(joined["reference_kl"].mean()),
        "batch/valid_action_tokens": float(joined["reward"].numel()),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("PPO training batch summary contains non-finite metrics")
    return metrics


def prepare_sequence_targets(
    steps: Sequence[PPOTargetStep], *, gamma: float, gae_lambda: float
) -> tuple[PPOStepTargets, ...]:
    """Validate the whole input, compute episode GAE, and restore input ordering.

    Each episode's collected portion must be contiguous in absolute decision indices,
    with an explicit environment or collection boundary at its end. Distinct episodes
    are never connected, including when reset occurred inside a storage fragment.
    """
    if not steps:
        raise ValueError("PPO sequence target preparation requires at least one decision")
    for step in steps:
        if type(step.policy_version) is not int or step.policy_version < 0:
            raise ValueError(f"PPO sample {step.source.sample_id!r} has an invalid policy_version")
    if len({step.policy_version for step in steps}) != 1:
        raise ValueError("PPO sequence target preparation requires one policy_version")
    episodes: dict[tuple[str | None, str], list[PPOTargetStep]] = {}
    sources: set[str] = set()
    lengths: dict[str, int] = {}
    for step in steps:
        source = step.source
        if source.sample_id in sources:
            raise ValueError(f"duplicate PPO sample_id {source.sample_id!r}")
        sources.add(source.sample_id)
        width = len(step.rewards)
        if width == 0 or len(step.mask) != width or len(step.values) != width + 1:
            raise ValueError(f"PPO sample {source.sample_id!r} requires rewards/mask [T] and values [T+1]")
        if any(type(valid) is not bool for valid in step.mask):
            raise ValueError(f"PPO sample {source.sample_id!r} mask must contain booleans")
        valid_count = sum(step.mask)
        if not valid_count or step.mask != (True,) * valid_count + (False,) * (width - valid_count):
            raise ValueError(f"PPO sample {source.sample_id!r} valid positions must form a nonempty prefix")
        for name in ("terminated", "truncated", "window_end"):
            if type(getattr(step, name)) is not bool:
                raise ValueError(f"PPO sample {source.sample_id!r} {name} must be a boolean")
        if step.terminated and step.truncated:
            raise ValueError(f"PPO sample {source.sample_id!r} cannot be both terminated and truncated")
        if not bool(torch.isfinite(torch.tensor((*step.rewards, *step.values), dtype=torch.float64)).all()):
            raise ValueError(f"PPO sample {source.sample_id!r} has nonfinite rewards or values")
        lengths[source.sample_id] = valid_count
        episodes.setdefault((source.session, source.episode_id), []).append(step)

    ordered_episodes = [sorted(episode, key=lambda step: step.source.step_idx) for episode in episodes.values()]
    for episode in ordered_episodes:
        for previous, following in zip(episode, episode[1:]):
            if following.source.step_idx != previous.source.step_idx + 1:
                raise ValueError(
                    f"PPO episode {previous.source.episode_id!r} has duplicate or missing decision indices"
                )
            if previous.terminated or previous.truncated or previous.window_end:
                raise ValueError(
                    f"PPO sample {previous.source.sample_id!r} continues after its episode/window boundary"
                )
        last = episode[-1]
        if not (last.terminated or last.truncated or last.window_end):
            raise ValueError(f"PPO sample {last.source.sample_id!r} is missing a final collection boundary")

    width = max(sum(lengths[step.source.sample_id] for step in episode) for episode in ordered_episodes)
    rewards = torch.zeros((len(ordered_episodes), width), dtype=torch.float64)  # [E,T]
    values = torch.zeros((len(ordered_episodes), width + 1), dtype=torch.float64)  # [E,T+1]
    mask = torch.zeros_like(rewards, dtype=torch.bool)  # [E,T]
    terminated = torch.zeros(len(ordered_episodes), dtype=torch.bool)  # [E]
    truncated = torch.zeros_like(terminated)  # [E]
    offsets: dict[str, tuple[int, int, int]] = {}
    for row, episode in enumerate(ordered_episodes):
        offset = 0
        for step in episode:
            count = lengths[step.source.sample_id]
            end = offset + count
            rewards[row, offset:end] = torch.tensor(step.rewards[:count], dtype=torch.float64)
            values[row, offset:end] = torch.tensor(step.values[:count], dtype=torch.float64)
            mask[row, offset:end] = True
            offsets[step.source.sample_id] = (row, offset, end)
            offset = end
        last = episode[-1]
        values[row, offset] = last.values[lengths[last.source.sample_id]]
        terminated[row] = last.terminated
        # Both nonterminal boundary kinds bootstrap, but retain their distinct source facts.
        truncated[row] = last.truncated or last.window_end and not last.terminated
    advantages, returns = compute_gae(
        rewards, values, mask, terminated, truncated, gamma=gamma, gae_lambda=gae_lambda
    )  # [E,T], [E,T]
    result: list[PPOStepTargets] = []
    for step in steps:
        row, start, end = offsets[step.source.sample_id]
        padding = (0.0,) * (len(step.mask) - lengths[step.source.sample_id])
        result.append(
            PPOStepTargets(
                source=step.source,
                advantages=tuple(advantages[row, start:end].tolist()) + padding,
                returns=tuple(returns[row, start:end].tolist()) + padding,
            )
        )
    return tuple(result)


__all__ = [
    "PPOMinibatch",
    "PPOUpdatePlan",
    "make_ppo_update_plan",
    "ppo_minibatches",
    "ppo_training_batch_metrics",
    "prepare_ppo_training_batch",
    "prepare_sequence_targets",
]
