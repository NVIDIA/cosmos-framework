# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-leaf PPO records.

The records are frozen and store cross-role scalar/vector data as tuples. This
prevents a policy minibatch from accidentally replacing rollout log-probabilities
or token identities after collection. Worker groups exchange only shared
``Trajectory`` objects; these records are materialized and validated locally by
the PPO algorithm. This module imports no PPO wire conversion code so its classes
remain directly importable and pickle-safe in fresh worker processes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cosmos_framework.posttrain.schema.trajectory import SampleSource

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PPOPreparedBatchReceipt:
    """Global tensor-free receipt emitted after policy-side join/GAE/cache install."""

    batch_id: str
    episode_ids: tuple[str, ...]
    policy_version: int
    target_digests: tuple[str, ...]
    batch_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.batch_id) is None:
            raise ValueError("prepared PPO batch_id must be an exact SHA256 digest")
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("prepared PPO batch requires unique episode IDs")
        if type(self.policy_version) is not int or self.policy_version < 0:
            raise ValueError("prepared PPO policy_version must be a non-negative exact integer")
        if len(self.target_digests) != len(self.episode_ids) or any(
            _SHA256_RE.fullmatch(digest) is None for digest in self.target_digests
        ):
            raise ValueError("prepared PPO target digests must exactly align with episode IDs")
        if not self.batch_metrics:
            raise ValueError("prepared PPO batch metrics must be non-empty")
        normalized: dict[str, float] = {}
        for name, value in self.batch_metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("prepared PPO metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"prepared PPO metric {name!r} must be a finite scalar")
            normalized[name] = float(value)
        object.__setattr__(self, "batch_metrics", normalized)


@dataclass(frozen=True)
class PPOCriticTargetReceipt:
    """Tensor-free proof that critic caches received the policy targets unchanged."""

    batch_id: str
    episode_ids: tuple[str, ...]
    target_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.batch_id) is None:
            raise ValueError("critic target receipt batch_id must be an exact SHA256 digest")
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("critic target receipt requires unique non-empty episode IDs")
        if any(not episode_id for episode_id in self.episode_ids):
            raise ValueError("critic target receipt contains an empty episode ID")
        if len(self.target_digests) != len(self.episode_ids) or any(
            _SHA256_RE.fullmatch(digest) is None for digest in self.target_digests
        ):
            raise ValueError("critic target receipt digests must exactly align with episode IDs")


@dataclass(frozen=True)
class PPORoleUpdateReceipt:
    """Tensor-free proof of ONE role-owned update window over one prepared batch.

    The role -- not the controller -- expands the update plan into epochs, global minibatches and
    its own model-DP micro-batches, so this receipt is the whole controller-visible surface of that
    window: how many global minibatches were applied, where the role's durable optimizer counter
    ended, and the role's already-DP-reduced metrics. Per-minibatch counters and the role's metric
    schema stay inside the role; the controller reconciles only the boundary.
    """

    role: str
    batch_id: str
    optimizer_steps: int
    optimizer_step: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.role not in ("policy", "critic"):
            raise ValueError("PPO role update receipt role must be policy or critic")
        if _SHA256_RE.fullmatch(self.batch_id) is None:
            raise ValueError("PPO role update receipt batch_id must be an exact SHA256 digest")
        for name in ("optimizer_steps", "optimizer_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"PPO role update receipt {name} must be an exact integer")
        if self.optimizer_steps < 1:
            raise ValueError("PPO role update receipt must apply at least one global minibatch")
        if self.optimizer_step < self.optimizer_steps:
            raise ValueError("PPO role update receipt counter cannot precede the steps it applied")
        if not self.metrics:
            raise ValueError("PPO role update receipt metrics must be non-empty")
        normalized: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("PPO role update receipt metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"PPO role update receipt metric {name!r} must be a finite scalar")
            normalized[name] = float(value)
        counter_key = f"{self.role}_optimizer_step"
        if normalized.get(counter_key) != float(self.optimizer_step):
            raise ValueError(f"PPO role update receipt metric {counter_key!r} must equal the reported counter")
        object.__setattr__(self, "metrics", normalized)


def merge_ppo_step_metrics(
    prepared: PPOPreparedBatchReceipt,
    on_policy: Mapping[str, float],
    policy: PPORoleUpdateReceipt | None,
    critic: PPORoleUpdateReceipt,
) -> dict[str, float]:
    """Build one public PPO metric frame from tensor-free role receipts.

    ``policy`` is ``None`` on a critic-only warmup iteration (``rl.critic_warmup_iterations``),
    where the actor deliberately runs no update window and therefore has no receipt: a
    `PPORoleUpdateReceipt` refuses to describe a window that applied zero global minibatches, so
    absence is the only truthful representation and a fabricated empty one is not available. The
    frame stays self-describing through ``train/actor_warmup``, which is ``1.0`` exactly on those
    iterations, so a reader never has to infer the phase from a missing key.
    """
    frames = (
        (prepared.batch_metrics, on_policy, critic.metrics)
        if policy is None
        else (prepared.batch_metrics, on_policy, policy.metrics, critic.metrics)
    )
    metrics: dict[str, float] = {"train/actor_warmup": float(policy is None)}
    for frame in frames:
        for name, value in frame.items():
            if name.startswith("batch/"):
                key = f"train/{name.removeprefix('batch/')}"
            elif "/" in name:
                key = name
            else:
                key = f"optim/{name}" if name.endswith("_optimizer_step") else f"train/{name}"
            if key in metrics:
                raise ValueError(f"duplicate PPO public metric {key!r}")
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"PPO public metric {key!r} must be finite")
            metrics[key] = scalar
    return metrics


def _require_finite(values: tuple[float, ...], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def _float_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _int_tuple(values: tuple[int, ...]) -> tuple[int, ...]:
    if any(type(value) is not int for value in values):
        raise TypeError("token IDs must be exact integers, not coerced values")
    if any(value < 0 for value in values):
        raise ValueError("token IDs must be non-negative")
    return values


def _validate_score_identity(
    *,
    episode_id: object,
    policy_version: object,
    rendered_prompt: object,
    prompt_token_ids: tuple[int, ...],
    action_token_ids: tuple[int, ...],
    action_mask: tuple[bool, ...],
    media: tuple[object, ...],
) -> None:
    """Validate exact cross-role identity types before Python equality can coerce them."""
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a non-empty string")
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError(f"policy_version must be a non-negative exact integer, got {policy_version!r}")
    if not isinstance(rendered_prompt, str):
        raise TypeError(f"rendered_prompt must be a string, got {type(rendered_prompt).__name__}")
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty for response-token alignment")
    if not action_token_ids:
        raise ValueError("action_token_ids must be non-empty")
    if len(action_mask) != len(action_token_ids):
        raise ValueError("action_mask length must equal the action token length")
    if not all(type(value) is bool for value in action_mask):
        raise TypeError("action_mask must contain exact booleans")
    if not any(action_mask):
        raise ValueError("action_mask must select at least one action token")
    if any(valid and not previous for previous, valid in zip(action_mask, action_mask[1:])):
        raise ValueError("valid actions in action_mask must form a contiguous prefix")
    if not all(isinstance(item, MediaIdentity) for item in media):
        raise TypeError("media must contain MediaIdentity records")
    if tuple(item.ordinal for item in media if isinstance(item, MediaIdentity)) != tuple(range(len(media))):
        raise ValueError("media ordinals must be contiguous and preserve processor ordering")


@dataclass(frozen=True)
class MediaIdentity:
    """Stable identity and order for one image or video consumed by an episode."""

    modality: str
    identity: str
    ordinal: int

    def __post_init__(self) -> None:
        if self.modality not in ("image", "video"):
            raise ValueError(f"media modality must be 'image' or 'video', got {self.modality!r}")
        if not self.identity:
            raise ValueError("media identity must be non-empty")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError(f"media ordinal must be a non-negative integer, got {self.ordinal!r}")


@dataclass(frozen=True)
class PPOStepSource(SampleSource):
    """A training sample's immutable source, independent of its storage fragment."""

    step_idx: int
    session: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.step_idx) is not int or self.step_idx < 0:
            raise ValueError("PPO source step_idx must be a nonnegative integer")
        if self.session is not None and (not isinstance(self.session, str) or not self.session):
            raise ValueError("PPO source session must be a nonempty string or None")


@dataclass(frozen=True)
class PPOEpisode:
    """One complete immutable on-policy rollout and its external rewards."""

    episode_id: str
    policy_version: int
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    terminated: bool
    truncated: bool
    old_logprobs: tuple[float, ...]
    media: tuple[MediaIdentity, ...] = ()
    terminal_reward: float = 0.0
    process_rewards: tuple[float, ...] = ()
    behavior_topk_token_ids: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", _int_tuple(tuple(self.prompt_token_ids)))
        object.__setattr__(self, "action_token_ids", _int_tuple(tuple(self.action_token_ids)))
        object.__setattr__(self, "action_mask", tuple(self.action_mask))
        object.__setattr__(self, "old_logprobs", _float_tuple(tuple(self.old_logprobs)))
        object.__setattr__(self, "media", tuple(self.media))
        object.__setattr__(self, "process_rewards", _float_tuple(tuple(self.process_rewards)))
        object.__setattr__(
            self,
            "behavior_topk_token_ids",
            tuple(_int_tuple(tuple(token_ids)) for token_ids in self.behavior_topk_token_ids),
        )
        object.__setattr__(self, "terminal_reward", float(self.terminal_reward))
        _validate_score_identity(
            episode_id=self.episode_id,
            policy_version=self.policy_version,
            rendered_prompt=self.rendered_prompt,
            prompt_token_ids=self.prompt_token_ids,
            action_token_ids=self.action_token_ids,
            action_mask=self.action_mask,
            media=self.media,
        )
        action_length = len(self.action_token_ids)
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TypeError("terminated and truncated must be booleans")
        if self.terminated == self.truncated:
            raise ValueError("exactly one of terminated or truncated must be true")
        if len(self.old_logprobs) != action_length:
            raise ValueError(
                f"old_logprobs length {len(self.old_logprobs)} must equal action token length {action_length}"
            )
        if not self.process_rewards:
            object.__setattr__(self, "process_rewards", (0.0,) * action_length)
        elif len(self.process_rewards) != action_length:
            raise ValueError(
                f"process_rewards length {len(self.process_rewards)} must equal action token length {action_length}"
            )
        if self.behavior_topk_token_ids and len(self.behavior_topk_token_ids) != action_length:
            raise ValueError("behavior_topk_token_ids length must equal the action token length")
        for token_ids in self.behavior_topk_token_ids:
            if not token_ids or len(set(token_ids)) != len(token_ids) or any(token_id < 0 for token_id in token_ids):
                raise ValueError("behavior top-k token IDs must be unique non-negative integers")
        _require_finite(self.old_logprobs, "old_logprobs")
        _require_finite(self.process_rewards, "process_rewards")
        if not math.isfinite(self.terminal_reward):
            raise ValueError("terminal_reward must be finite")


@dataclass(frozen=True)
class ReferenceScore:
    """Frozen-reference action scores and fixed top-k/residual JSD targets."""

    episode_id: str
    policy_version: int
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    media: tuple[MediaIdentity, ...]
    action_logprobs: tuple[float, ...]
    support_token_ids: tuple[tuple[int, ...], ...]
    support_probabilities: tuple[tuple[float, ...], ...]
    residual_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", _int_tuple(tuple(self.prompt_token_ids)))
        object.__setattr__(self, "action_token_ids", _int_tuple(tuple(self.action_token_ids)))
        object.__setattr__(self, "action_mask", tuple(self.action_mask))
        object.__setattr__(self, "media", tuple(self.media))
        object.__setattr__(self, "action_logprobs", _float_tuple(tuple(self.action_logprobs)))
        object.__setattr__(
            self,
            "support_token_ids",
            tuple(_int_tuple(tuple(token_ids)) for token_ids in self.support_token_ids),
        )
        object.__setattr__(
            self,
            "support_probabilities",
            tuple(_float_tuple(tuple(probabilities)) for probabilities in self.support_probabilities),
        )
        object.__setattr__(
            self,
            "residual_probabilities",
            _float_tuple(tuple(self.residual_probabilities)),
        )
        _validate_score_identity(
            episode_id=self.episode_id,
            policy_version=self.policy_version,
            rendered_prompt=self.rendered_prompt,
            prompt_token_ids=self.prompt_token_ids,
            action_token_ids=self.action_token_ids,
            action_mask=self.action_mask,
            media=self.media,
        )


@dataclass(frozen=True)
class CriticScore:
    """Old critic estimates: one value per action state plus one bootstrap."""

    episode_id: str
    policy_version: int
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    media: tuple[MediaIdentity, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", _int_tuple(tuple(self.prompt_token_ids)))
        object.__setattr__(self, "action_token_ids", _int_tuple(tuple(self.action_token_ids)))
        object.__setattr__(self, "action_mask", tuple(self.action_mask))
        object.__setattr__(self, "media", tuple(self.media))
        object.__setattr__(self, "values", _float_tuple(tuple(self.values)))
        _validate_score_identity(
            episode_id=self.episode_id,
            policy_version=self.policy_version,
            rendered_prompt=self.rendered_prompt,
            prompt_token_ids=self.prompt_token_ids,
            action_token_ids=self.action_token_ids,
            action_mask=self.action_mask,
            media=self.media,
        )


@dataclass(frozen=True)
class ScoredPPOEpisode:
    """A strict episode-ID join of rollout, reference, and old critic results."""

    episode: PPOEpisode
    reference: ReferenceScore
    critic: CriticScore

    def __post_init__(self) -> None:
        validate_reference_score(self.episode, self.reference)
        validate_critic_score(self.episode, self.critic)


@dataclass(frozen=True)
class PPOTrainingEpisode:
    """Policy-role-prepared token rewards, policy advantages, and critic returns.

    ``advantages`` are the globally normalized policy advantages. The optional
    ``value_advantages`` retain raw GAE deltas for the critic identity
    ``returns = old_value + value_advantages``; when omitted they are derived
    losslessly from the supplied returns and immutable old critic values.
    """

    scored: ScoredPPOEpisode
    action_rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]
    value_advantages: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_rewards", _float_tuple(tuple(self.action_rewards)))
        object.__setattr__(self, "advantages", _float_tuple(tuple(self.advantages)))
        object.__setattr__(self, "returns", _float_tuple(tuple(self.returns)))
        object.__setattr__(self, "value_advantages", _float_tuple(tuple(self.value_advantages)))
        action_length = len(self.scored.episode.action_token_ids)
        for name in ("action_rewards", "advantages", "returns"):
            values = getattr(self, name)
            if len(values) != action_length:
                raise ValueError(f"{name} length {len(values)} must equal action token length {action_length}")
            _require_finite(values, name)
        if not self.value_advantages:
            object.__setattr__(
                self,
                "value_advantages",
                tuple(
                    target - old_value if valid else 0.0
                    for target, old_value, valid in zip(
                        self.returns,
                        self.scored.critic.values[:-1],
                        self.scored.episode.action_mask,
                    )
                ),
            )
        elif len(self.value_advantages) != action_length:
            raise ValueError(
                f"value_advantages length {len(self.value_advantages)} must equal action token length {action_length}"
            )
        _require_finite(self.value_advantages, "value_advantages")
        expected_returns = tuple(
            old_value + advantage for old_value, advantage in zip(self.scored.critic.values[:-1], self.value_advantages)
        )
        if any(
            valid and not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected, valid in zip(self.returns, expected_returns, self.scored.episode.action_mask)
        ):
            raise ValueError("returns must equal old critic value + value advantage at every action token")
        for name in ("action_rewards", "advantages", "returns", "value_advantages"):
            if any(
                not valid and value != 0.0 for value, valid in zip(getattr(self, name), self.scored.episode.action_mask)
            ):
                raise ValueError(f"{name} must be zero at invalid action positions")


def _unique_by_episode_id(results: list[object], role: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for result in results:
        episode_id = getattr(result, "episode_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{role} result has an invalid episode_id")
        if episode_id in indexed:
            raise ValueError(f"duplicate {role} episode_id {episode_id!r}")
        indexed[episode_id] = result
    return indexed


def _validate_common_identity(episode: PPOEpisode, score: ReferenceScore | CriticScore, role: str) -> None:
    if score.episode_id != episode.episode_id:
        raise ValueError(f"{role} episode_id mismatch: expected {episode.episode_id!r}, got {score.episode_id!r}")
    if type(score.policy_version) is not type(episode.policy_version) or score.policy_version != episode.policy_version:
        raise ValueError(
            f"{role} stale policy_version for episode {episode.episode_id!r}: "
            f"expected {episode.policy_version}, got {score.policy_version}"
        )
    for field_name in (
        "rendered_prompt",
        "prompt_token_ids",
        "action_token_ids",
        "action_mask",
        "media",
    ):
        if getattr(score, field_name) != getattr(episode, field_name):
            raise ValueError(f"{role} {field_name} mismatch for episode {episode.episode_id!r}")


def _validate_reference(episode: PPOEpisode, result: ReferenceScore) -> None:
    action_length = len(episode.action_token_ids)
    if len(result.action_logprobs) != action_length:
        raise ValueError("reference action_logprobs length must equal the action token length")
    _require_finite(result.action_logprobs, "reference action_logprobs")
    for name in ("support_token_ids", "support_probabilities", "residual_probabilities"):
        if len(getattr(result, name)) != action_length:
            raise ValueError(f"reference {name} length must equal the action token length")
    for token_index, (token_ids, probabilities, residual) in enumerate(
        zip(result.support_token_ids, result.support_probabilities, result.residual_probabilities)
    ):
        if not token_ids:
            raise ValueError("reference support_token_ids must be non-empty")
        if len(probabilities) != len(token_ids):
            raise ValueError("reference support probabilities must align with support token IDs")
        if len(set(token_ids)) != len(token_ids) or any(token_id < 0 for token_id in token_ids):
            raise ValueError("reference support token IDs must be unique non-negative integers")
        _require_finite(probabilities, "reference support_probabilities")
        _require_finite((residual,), "reference residual_probabilities")
        if any(probability < 0.0 for probability in probabilities) or residual < 0.0:
            raise ValueError("reference support and residual probabilities must be non-negative")
        if not math.isclose(sum(probabilities) + residual, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"reference probabilities at action index {token_index} must sum to 1")
        if episode.behavior_topk_token_ids:
            missing_behavior_ids = set(episode.behavior_topk_token_ids[token_index]) - set(token_ids)
            if missing_behavior_ids:
                raise ValueError(f"reference support at action index {token_index} is missing behavior top-k token IDs")


def _validate_critic(episode: PPOEpisode, result: CriticScore) -> None:
    expected = len(episode.action_token_ids) + 1
    if len(result.values) != expected:
        raise ValueError(f"critic values length {len(result.values)} must equal T+1 ({expected})")
    _require_finite(result.values, "critic values")


def validate_reference_score(episode: PPOEpisode, result: ReferenceScore) -> None:
    """Validate one reference result against its exact rollout identity."""
    _validate_common_identity(episode, result, "reference")
    _validate_reference(episode, result)


def validate_critic_score(episode: PPOEpisode, result: CriticScore) -> None:
    """Validate one critic result against its exact rollout identity."""
    _validate_common_identity(episode, result, "critic")
    _validate_critic(episode, result)


def strict_join_ppo_results(
    episodes: list[PPOEpisode],
    reference_results: list[ReferenceScore],
    critic_results: list[CriticScore],
) -> tuple[ScoredPPOEpisode, ...]:
    """Join cross-role outputs by ID and reject every parity or numerical mismatch."""
    if not episodes:
        raise ValueError("strict PPO join received an empty episode batch")
    episode_index = _unique_by_episode_id(list(episodes), "episode")
    reference_index = _unique_by_episode_id(list(reference_results), "reference")
    critic_index = _unique_by_episode_id(list(critic_results), "critic")
    expected_ids = set(episode_index)
    for role, index in (("reference", reference_index), ("critic", critic_index)):
        missing = expected_ids - set(index)
        extra = set(index) - expected_ids
        if missing:
            raise ValueError(f"missing {role} results for episode IDs {sorted(missing)!r}")
        if extra:
            raise ValueError(f"extra {role} results for episode IDs {sorted(extra)!r}")

    joined: list[ScoredPPOEpisode] = []
    for episode in episodes:
        reference = reference_index[episode.episode_id]
        critic = critic_index[episode.episode_id]
        if not isinstance(reference, ReferenceScore) or not isinstance(critic, CriticScore):
            raise TypeError("strict PPO join received an unexpected score type")
        joined.append(ScoredPPOEpisode(episode=episode, reference=reference, critic=critic))
    return tuple(joined)


@dataclass(frozen=True)
class PPOTargetStep:
    """Fixed rewards and values for one decision's effective learning positions.

    The final value is used only at an episode/window boundary. An interior decision
    continues with the next decision's first value, computed after environment feedback.
    Reward transformation and aggregation have already happened in the recipe.
    """

    source: PPOStepSource
    rewards: tuple[float, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]
    terminated: bool
    truncated: bool
    window_end: bool = False
    policy_version: int = 0


@dataclass(frozen=True)
class PPOStepTargets:
    """Fixed per-position advantages and returns for one decision, in its input order."""

    source: PPOStepSource
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


__all__ = [
    "CriticScore",
    "MediaIdentity",
    "PPOCriticTargetReceipt",
    "PPOEpisode",
    "PPOPreparedBatchReceipt",
    "PPORoleUpdateReceipt",
    "PPOTrainingEpisode",
    "ReferenceScore",
    "ScoredPPOEpisode",
    "merge_ppo_step_metrics",
    "strict_join_ppo_results",
    "validate_critic_score",
    "validate_reference_score",
    "PPOTargetStep",
    "PPOStepTargets",
]
