# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for the rollout ``finish_reason`` stamp on ``RewardCandidate``.

The vLLM rollout records how the engine ended every completion in ``policy_info["finish_reason"]``
(``schema.FINISH_REASON_KEY``). The framework request builder copies that verdict onto the prepared
candidate next to ``completion_token_count`` so a provider can gate on the exact cap-cut signal
instead of inferring it from a token count. ``RewardCandidate.is_length_truncated`` and
``schema.is_length_truncated(policy_info)`` both delegate to ``schema.is_length_verdict``, so the two
readers can never disagree. Providers that ignore the field see no change.
"""

from __future__ import annotations

import dataclasses

import pytest

from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardContext,
    RewardContractError,
    RewardProviderResult,
    bind_candidate_provider,
)
from cosmos_framework.posttrain.algorithm.reward.group import build_reward_group_request
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.schema import (
    FINISH_REASON_KEY,
    FINISH_REASON_LENGTH,
    FINISH_REASON_STOP,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    is_length_truncated,
    is_length_verdict,
)

_ABSENT = object()
"""Sentinel: leave ``policy_info["finish_reason"]`` unset (an explicit ``None`` is a distinct, stamped case)."""


def _trajectory(
    trajectory_id: str,
    completion: str,
    *,
    finish_reason: object = _ABSENT,
    group_id: str = "g0",
) -> Trajectory:
    """One completed single-transition trajectory whose token count is the completion length."""
    policy_info: dict[str, object] = {"completion_text": completion, "completion_token_count": len(completion)}
    if finish_reason is not _ABSENT:
        policy_info[FINISH_REASON_KEY] = finish_reason
    transition = Transition(
        obs=ObsBundle(id=group_id, representations={"prompt": "common prompt"}),
        policy_info=policy_info,
        done=True,
    )
    return Trajectory(transitions=[transition], id=trajectory_id, group_id=group_id, status=Status.COMPLETED)


def _length_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """A provider that never looks at ``finish_reason``."""
    del context
    return RewardProviderResult(terms={"task": float(len(candidate.completion))})


def _cap_cut_gated_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """A provider that zeroes exactly the completions the engine cut at its token budget."""
    del context
    return RewardProviderResult(terms={"task": 0.0 if candidate.is_length_truncated else 1.0})


@pytest.mark.L0
@pytest.mark.CPU
def test_candidate_finish_reason_defaults_to_none_and_stays_frozen() -> None:
    candidate = RewardCandidate(trajectory_id="t", completion="abc", completion_token_count=3)

    assert candidate.finish_reason is None
    assert not candidate.is_length_truncated
    assert candidate == RewardCandidate(
        trajectory_id="t", completion="abc", completion_token_count=3, finish_reason=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.finish_reason = FINISH_REASON_LENGTH  # type: ignore[misc]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [(FINISH_REASON_LENGTH, True), (FINISH_REASON_STOP, False), ("abort", False), (None, False)],
    ids=("length", "stop", "abort", "none"),
)
def test_is_length_truncated_answers_only_to_the_exact_length_verdict(
    finish_reason: str | None, truncated: bool
) -> None:
    candidate = RewardCandidate(trajectory_id="t", completion="abc", finish_reason=finish_reason)

    assert candidate.finish_reason == finish_reason
    assert candidate.is_length_truncated is truncated
    assert dataclasses.replace(candidate, completion_token_count=3).is_length_truncated is truncated


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "invalid", [1, True, b"length", ["length"], 0.0], ids=("int", "bool", "bytes", "list", "float")
)
def test_candidate_rejects_a_non_string_finish_reason(invalid: object) -> None:
    with pytest.raises(RewardContractError, match="finish_reason must be a string or None"):
        RewardCandidate(trajectory_id="t", completion="abc", finish_reason=invalid)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "finish_reason", [FINISH_REASON_LENGTH, FINISH_REASON_STOP, "abort", None], ids=("length", "stop", "abort", "none")
)
def test_candidate_property_and_schema_predicate_share_one_verdict(finish_reason: str | None) -> None:
    candidate = RewardCandidate(trajectory_id="t", completion="abc", finish_reason=finish_reason)

    expected = is_length_verdict(finish_reason)
    assert candidate.is_length_truncated is expected
    assert is_length_truncated({FINISH_REASON_KEY: finish_reason}) is expected
    assert expected is (finish_reason == FINISH_REASON_LENGTH)


@pytest.mark.L0
@pytest.mark.CPU
def test_request_builder_stamps_finish_reason_from_the_terminal_policy_info() -> None:
    trajectories = [
        _trajectory("g0_k0", "cut", finish_reason=FINISH_REASON_LENGTH),
        _trajectory("g0_k1", "done", finish_reason=FINISH_REASON_STOP),
        _trajectory("g0_k2", "unknown"),  # key absent: HF/local rollouts, batches from before the stamp
        _trajectory("g0_k3", "unreported", finish_reason=None),  # the engine reported no verdict
    ]

    request = build_reward_group_request(trajectories, needs=("prompt",))

    assert tuple(candidate.finish_reason for candidate in request.candidates) == ("length", "stop", None, None)
    assert tuple(candidate.completion_token_count for candidate in request.candidates) == (3, 4, 7, 10)
    assert tuple(candidate.is_length_truncated for candidate in request.candidates) == (True, False, False, False)
    assert request.context.prompt == "common prompt"


@pytest.mark.L0
@pytest.mark.CPU
def test_request_builder_rejects_a_non_string_finish_reason_stamp() -> None:
    trajectories = [
        _trajectory("g0_k0", "ok", finish_reason=FINISH_REASON_STOP),
        _trajectory("g0_k1", "bad", finish_reason=1),
    ]

    with pytest.raises(RewardContractError, match=r"finish_reason.*string when present for candidate 1"):
        build_reward_group_request(trajectories, needs=())


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_that_ignores_finish_reason_scores_exactly_as_before() -> None:
    trajectories = [
        _trajectory("g0_k0", "same text", finish_reason=FINISH_REASON_LENGTH),
        _trajectory("g0_k1", "same text", finish_reason=FINISH_REASON_STOP),
        _trajectory("g0_k2", "same text"),
    ]
    worker = RewardWorker([bind_candidate_provider("length", _length_reward)], training_group_size=3)

    scored = worker.score(trajectories)

    assert [trajectory.final_reward for trajectory in scored] == [9.0, 9.0, 9.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_can_gate_on_the_stamped_verdict_through_the_worker() -> None:
    trajectories = [
        _trajectory("g0_k0", "same text", finish_reason=FINISH_REASON_LENGTH),
        _trajectory("g0_k1", "same text", finish_reason=FINISH_REASON_STOP),
        _trajectory("g0_k2", "same text"),
    ]
    worker = RewardWorker([bind_candidate_provider("gated", _cap_cut_gated_reward)], training_group_size=3)

    scored = worker.score(trajectories)

    assert [trajectory.final_reward for trajectory in scored] == [0.0, 1.0, 1.0]
