# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed contracts and framework-owned request construction for group rewards."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError, _completion_text
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult
from cosmos_framework.posttrain.schema import FINISH_REASON_KEY, Status, Trajectory, is_length_verdict

_GROUP_INPUT_NAMES = frozenset({"prompt", "reference"})


def _nonempty_identifier(value: object, *, field: str) -> str:
    """Validate one correlation identifier without silently coercing it."""
    if not isinstance(value, str) or not value.strip():
        raise RewardContractError(f"group reward {field} must be a non-empty string, got {value!r}")
    return value


def _tuple_from_sequence(value: object, *, field: str) -> tuple[object, ...]:
    """Freeze a typed sequence while rejecting strings masquerading as sequences."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RewardContractError(f"group reward {field} must be a sequence, got {type(value).__name__}")
    return tuple(value)


def _freeze_context_value(value: object, *, path: str, active: set[int]) -> object:
    """Deep-freeze one finite JSON-compatible provider input without retaining trajectory aliases."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RewardContractError(f"group reward common input at {path} must be finite")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise RewardContractError(f"group reward common input contains a cycle at {path}")
        if any(not isinstance(key, str) for key in value):
            raise RewardContractError(f"group reward common input mapping keys at {path} must be strings")
        active.add(marker)
        try:
            frozen = {
                key: _freeze_context_value(item, path=f"{path}.{key}", active=active) for key, item in value.items()
            }
        finally:
            active.remove(marker)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        marker = id(value)
        if marker in active:
            raise RewardContractError(f"group reward common input contains a cycle at {path}")
        active.add(marker)
        try:
            return tuple(
                _freeze_context_value(item, path=f"{path}[{index}]", active=active) for index, item in enumerate(value)
            )
        finally:
            active.remove(marker)
    raise RewardContractError(
        f"group reward common input at {path} must contain only finite JSON-compatible values, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class RewardContext:
    """Immutable inputs shared by every candidate in one reward group."""

    prompt: object | None = None
    reference: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _freeze_context_value(self.prompt, path="prompt", active=set()))
        object.__setattr__(
            self,
            "reference",
            _freeze_context_value(self.reference, path="reference", active=set()),
        )


@dataclass(frozen=True)
class RewardCandidate:
    """One immutable candidate identity and its terminal rollout outputs.

    ``finish_reason`` is the rollout engine's verbatim verdict on how the completion ended
    (``policy_info[schema.FINISH_REASON_KEY]``, ``"stop"`` | ``"length"`` | another engine string).
    ``None`` means the producer recorded none and must read as "not known to be truncated", never as
    "stopped". Providers that ignore the field are unaffected by it.
    """

    trajectory_id: str
    completion: str
    completion_token_count: int | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _nonempty_identifier(self.trajectory_id, field="trajectory_id")
        if not isinstance(self.completion, str):
            raise RewardContractError(f"group reward completion must be a string, got {type(self.completion).__name__}")
        if self.completion_token_count is not None and (
            isinstance(self.completion_token_count, bool)
            or not isinstance(self.completion_token_count, int)
            or self.completion_token_count < 0
        ):
            raise RewardContractError(
                "group reward completion_token_count must be a non-negative integer or None, "
                f"got {self.completion_token_count!r}"
            )
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise RewardContractError(
                f"group reward finish_reason must be a string or None, got {self.finish_reason!r}"
            )

    @property
    def is_length_truncated(self) -> bool:
        """Whether the rollout engine recorded that it cut this candidate at its token budget.

        Delegates to ``schema.is_length_verdict``, the one truncation comparison shared with
        ``schema.is_length_truncated(policy_info)``: only the exact ``"length"`` verdict answers True.
        ``None`` (no verdict recorded) is "not known to be truncated", so a provider that gates on this
        never penalizes completions from a producer that does not stamp the key.
        """
        return is_length_verdict(self.finish_reason)


@dataclass(frozen=True)
class RewardGroupRequest:
    """Every ordered completion for one group plus its declared common context."""

    group_id: str
    candidates: tuple[RewardCandidate, ...]
    context: RewardContext = field(default_factory=RewardContext)

    def __post_init__(self) -> None:
        _nonempty_identifier(self.group_id, field="group_id")
        raw_candidates = _tuple_from_sequence(self.candidates, field="candidates")
        if not raw_candidates:
            raise RewardContractError("group reward candidates must not be empty")
        candidates: tuple[RewardCandidate, ...] = tuple(
            candidate for candidate in raw_candidates if isinstance(candidate, RewardCandidate)
        )
        if len(candidates) != len(raw_candidates):
            invalid = next(candidate for candidate in raw_candidates if not isinstance(candidate, RewardCandidate))
            raise RewardContractError(
                f"group reward candidates must contain RewardCandidate values, got {type(invalid).__name__}"
            )
        trajectory_ids = tuple(candidate.trajectory_id for candidate in candidates)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise RewardContractError(f"group reward candidate trajectory_ids must be unique, got {trajectory_ids!r}")
        if not isinstance(self.context, RewardContext):
            raise RewardContractError(
                f"group reward context must be a RewardContext, got {type(self.context).__name__}"
            )
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class RewardCandidateResult:
    """One explicitly correlated candidate result."""

    trajectory_id: str
    result: RewardProviderResult

    def __post_init__(self) -> None:
        _nonempty_identifier(self.trajectory_id, field="result trajectory_id")
        if not isinstance(self.result, RewardProviderResult):
            raise RewardContractError(
                f"group reward candidate result must be a RewardProviderResult, got {type(self.result).__name__}"
            )


@dataclass(frozen=True)
class RewardGroupResult:
    """One ordered, explicitly correlated result for every candidate in a group."""

    group_id: str
    items: tuple[RewardCandidateResult, ...]

    def __post_init__(self) -> None:
        _nonempty_identifier(self.group_id, field="result group_id")
        raw_items = _tuple_from_sequence(self.items, field="result items")
        if not raw_items:
            raise RewardContractError("group reward result items must not be empty")
        items: tuple[RewardCandidateResult, ...] = tuple(
            item for item in raw_items if isinstance(item, RewardCandidateResult)
        )
        if len(items) != len(raw_items):
            invalid = next(item for item in raw_items if not isinstance(item, RewardCandidateResult))
            raise RewardContractError(
                f"group reward result items must contain RewardCandidateResult values, got {type(invalid).__name__}"
            )
        trajectory_ids = tuple(item.trajectory_id for item in items)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise RewardContractError(f"group reward result trajectory_ids must be unique, got {trajectory_ids!r}")
        object.__setattr__(self, "items", items)


def _same_frozen_context_value(left: object, right: object) -> bool:
    """Compare JSON-compatible values without collapsing bool, integer, or float semantics."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            return False
        return all(_same_frozen_context_value(left[key], right[key]) for key in left)
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            return False
        return all(_same_frozen_context_value(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _common_declared_value(trajectories: tuple[Trajectory, ...], *, name: str) -> object:
    """Read, freeze, and prove one required terminal observation value is common to the group."""
    values: list[object] = []
    for index, trajectory in enumerate(trajectories):
        representations = trajectory.transitions[-1].obs.representations
        if name not in representations or representations[name] is None:
            raise RewardContractError(
                f"group reward needs terminal obs.representations[{name!r}] for candidate {index}"
            )
        values.append(
            _freeze_context_value(
                representations[name],
                path=f"{name}[candidate={index}]",
                active=set(),
            )
        )

    expected = values[0]
    for index, value in enumerate(values[1:], start=1):
        if not _same_frozen_context_value(value, expected):
            raise RewardContractError(
                f"group reward declared {name!r} must be identical for every candidate; "
                f"candidate {index} differs from candidate 0"
            )
    return expected


def _completion_token_count(trajectory: Trajectory, *, index: int) -> int | None:
    """Read the rollout-authored completion length without inventing a tokenizer fallback."""
    value = trajectory.transitions[-1].policy_info.get("completion_token_count")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RewardContractError(
            "group reward needs policy_info['completion_token_count'] to be a non-negative integer "
            f"when present for candidate {index}, got {value!r}"
        )
    return value


def _finish_reason(trajectory: Trajectory, *, index: int) -> str | None:
    """Read the rollout engine's verbatim end-of-completion verdict; None when the producer recorded none."""
    value = trajectory.transitions[-1].policy_info.get(FINISH_REASON_KEY)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RewardContractError(
            f"group reward needs policy_info[{FINISH_REASON_KEY!r}] to be a string when present "
            f"for candidate {index}, got {value!r}"
        )
    return value


def build_reward_group_request(
    trajectories: Sequence[Trajectory],
    *,
    needs: tuple[str, ...],
) -> RewardGroupRequest:
    """Validate complete correlated trajectories and construct the provider-facing request."""
    if isinstance(trajectories, (str, bytes)) or not isinstance(trajectories, Sequence):
        raise RewardContractError(f"group reward trajectories must be a sequence, got {type(trajectories).__name__}")
    group = tuple(trajectories)
    if not group:
        raise RewardContractError("group reward needs at least one trajectory")
    invalid_need = next((name for name in needs if name not in _GROUP_INPUT_NAMES), None)
    if invalid_need is not None:
        raise RewardContractError(
            f"group reward cannot construct declared input {invalid_need!r}; supported inputs are "
            f"{sorted(_GROUP_INPUT_NAMES)}"
        )

    group_id: str | None = None
    candidates: list[RewardCandidate] = []
    for index, trajectory in enumerate(group):
        if not isinstance(trajectory, Trajectory):
            raise RewardContractError(
                f"group reward candidate {index} must be a Trajectory, got {type(trajectory).__name__}"
            )
        if trajectory.status != Status.COMPLETED:
            raise RewardContractError(
                f"group reward candidate {index} must be completed, got status={trajectory.status!r}"
            )
        if not trajectory.transitions:
            raise RewardContractError(f"group reward candidate {index} has no terminal transition")
        terminal = trajectory.transitions[-1]
        if not terminal.done:
            raise RewardContractError(f"group reward candidate {index} terminal transition is not done")

        candidate_group_id = _nonempty_identifier(trajectory.group_id, field=f"candidate {index} group_id")
        if group_id is None:
            group_id = candidate_group_id
        elif candidate_group_id != group_id:
            raise RewardContractError(
                f"group reward received mixed group_ids: candidate 0 has {group_id!r}, "
                f"candidate {index} has {candidate_group_id!r}"
            )
        if terminal.obs.id != candidate_group_id:
            raise RewardContractError(
                f"group reward candidate {index} obs.id {terminal.obs.id!r} does not match "
                f"group_id {candidate_group_id!r}"
            )
        candidates.append(
            RewardCandidate(
                trajectory_id=_nonempty_identifier(trajectory.id, field=f"candidate {index} trajectory_id"),
                completion=_completion_text(trajectory),
                completion_token_count=_completion_token_count(trajectory, index=index),
                finish_reason=_finish_reason(trajectory, index=index),
            )
        )

    if group_id is None:
        raise AssertionError("non-empty group did not establish a group_id")
    candidate_ids = tuple(candidate.trajectory_id for candidate in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RewardContractError(f"group reward trajectory_ids must be unique, got {candidate_ids!r}")

    prompt = _common_declared_value(group, name="prompt") if "prompt" in needs else None
    reference = _common_declared_value(group, name="reference") if "reference" in needs else None
    return RewardGroupRequest(
        group_id=group_id,
        candidates=tuple(candidates),
        context=RewardContext(prompt=prompt, reference=reference),
    )
