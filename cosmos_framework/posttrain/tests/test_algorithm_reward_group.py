# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contracts for the group-native reward-worker boundary."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.algorithm.reward import (
    BoundRewardProvider,
    RewardCandidate,
    RewardCandidateResult,
    RewardContext,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderCatalog,
    RewardProviderResult,
    RewardProviderSpec,
    bind_candidate_provider,
)
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


class _RecordingGroupReward:
    """Deterministic request recorder whose completion text is the candidate score."""

    def __init__(self, *, term_name: str = "rubric", fail_group_id: str | None = None) -> None:
        self.term_name = term_name
        self.fail_group_id = fail_group_id
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        trajectory_ids = tuple(candidate.trajectory_id for candidate in request.candidates)
        self.calls.append((request.group_id, trajectory_ids))
        if request.group_id == self.fail_group_id:
            raise RewardContractError(f"configured failure for group {request.group_id!r}")
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=RewardProviderResult(terms={self.term_name: float(candidate.completion)}),
                )
                for candidate in request.candidates
            ),
        )


def _completion_length_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """A pointwise component lifted into the same group-native runtime contract."""
    del context
    return RewardProviderResult(terms={"length": float(len(candidate.completion))})


def _bound_group_reward(
    provider: _RecordingGroupReward,
    *,
    name: str = "rubric",
    needs: tuple[str, ...] = ("prompt",),
) -> BoundRewardProvider:
    catalog = RewardProviderCatalog(
        {
            name: RewardProviderSpec(
                factory=lambda: provider,
                needs=needs,
                term_names=(provider.term_name,),
            )
        }
    )
    return catalog.build(name)


def _trajectory(trajectory_id: str, group_id: str, score: str) -> Trajectory:
    obs = ObsBundle(
        id=group_id,
        representations={"prompt": f"prompt for {group_id}", "reference": {"group": group_id}},
    )
    transition = Transition(obs=obs, policy_info={"completion_text": score}, done=True)
    return Trajectory(
        transitions=[transition],
        id=trajectory_id,
        group_id=group_id,
        status=Status.COMPLETED,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_group_reward_executes_interleaved_groups_once_in_first_seen_order() -> None:
    provider = _RecordingGroupReward()
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g1_k0", "g1", "3"),
        _trajectory("g0_k1", "g0", "2"),
        _trajectory("g1_k1", "g1", "4"),
    ]
    worker = RewardWorker([_bound_group_reward(provider)], training_group_size=2)

    scored = worker.score(trajectories)

    assert provider.calls == [
        ("g0", ("g0_k0", "g0_k1")),
        ("g1", ("g1_k0", "g1_k1")),
    ]
    assert [trajectory.final_reward for trajectory in scored] == [1.0, 3.0, 2.0, 4.0]
    assert [trajectory.transitions[-1].reward["rubric"] for trajectory in scored] == [1.0, 3.0, 2.0, 4.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_incomplete_group_fails_before_any_group_provider_call() -> None:
    provider = _RecordingGroupReward()
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "2"),
        _trajectory("g1_k0", "g1", "3"),
    ]
    worker = RewardWorker([_bound_group_reward(provider)], training_group_size=2)

    with pytest.raises(RewardContractError, match="expected 2 trajectories for group 'g1', got 1"):
        worker.score(trajectories)

    assert provider.calls == []
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("invalid_identity", ["empty", "duplicate", "obs_mismatch"])
def test_group_identity_failures_are_preflighted_before_calls(invalid_identity: str) -> None:
    provider = _RecordingGroupReward()
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "2"),
    ]
    if invalid_identity == "empty":
        trajectories[1].id = ""
    elif invalid_identity == "duplicate":
        trajectories[1].id = trajectories[0].id
    else:
        trajectories[1].transitions[-1].obs.id = "another-group"
    worker = RewardWorker([_bound_group_reward(provider)], training_group_size=2)

    with pytest.raises(RewardContractError):
        worker.score(trajectories)

    assert provider.calls == []
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_all_groups_are_prepared_before_any_provider_execution() -> None:
    first_provider = _RecordingGroupReward(term_name="first")
    second_provider = _RecordingGroupReward(term_name="second")
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "2"),
        _trajectory("g1_k0", "g1", "3"),
        _trajectory("g1_k1", "g1", "4"),
    ]
    trajectories[-1].transitions[-1].obs.representations["prompt"] = "different prompt"
    worker = RewardWorker(
        [
            _bound_group_reward(first_provider, name="first", needs=()),
            _bound_group_reward(second_provider, name="second"),
        ],
        training_group_size=2,
    )

    with pytest.raises(RewardContractError, match="prompt.*identical"):
        worker.score(trajectories)

    assert first_provider.calls == []
    assert second_provider.calls == []
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_group_uses_n_generation_and_accepts_a_singleton() -> None:
    provider = _RecordingGroupReward()
    sample_id = "validation:stream:2:0"
    trajectory = _trajectory(f"{sample_id}_k0", sample_id, "0.75")
    trajectory.transitions[-1].policy_info.update(validation_step=6, weight_version=6)
    worker = RewardWorker(
        [
            bind_candidate_provider(
                "constant_test",
                lambda _candidate, _context: RewardProviderResult(terms={"task": 0.0}),
            )
        ],
        validation_providers=[_bound_group_reward(provider)],
        training_group_size=1,
    )

    worker.score_validation_with_report([trajectory], model_step=6, n_generation=1)
    report = worker.validation_batch_stats()

    assert provider.calls == [(sample_id, (f"{sample_id}_k0",))]
    assert trajectory.final_reward == pytest.approx(0.75)
    assert report.count == 1
    assert report.reward_moments.mean == pytest.approx(0.75)


@pytest.mark.L0
@pytest.mark.CPU
def test_late_group_failure_leaves_the_entire_batch_unmodified() -> None:
    provider = _RecordingGroupReward(fail_group_id="g1")
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "2"),
        _trajectory("g1_k0", "g1", "3"),
        _trajectory("g1_k1", "g1", "4"),
    ]
    worker = RewardWorker([_bound_group_reward(provider)], training_group_size=2)

    with pytest.raises(RewardContractError, match="configured failure for group 'g1'"):
        worker.score(trajectories)

    assert provider.calls == [
        ("g0", ("g0_k0", "g0_k1")),
        ("g1", ("g1_k0", "g1_k1")),
    ]
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_pointwise_and_native_group_providers_compose_by_trajectory_identity() -> None:
    provider = _RecordingGroupReward()
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "20"),
    ]
    worker = RewardWorker(
        [
            bind_candidate_provider(
                "length",
                _completion_length_reward,
                term_names=("length",),
            ),
            _bound_group_reward(provider),
        ],
        training_group_size=2,
    )

    scored = worker.score(trajectories)

    assert [trajectory.transitions[-1].reward for trajectory in scored] == [
        {"length": 1.0, "rubric": 1.0, "final": 2.0},
        {"length": 2.0, "rubric": 20.0, "final": 22.0},
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_terms_across_canonical_providers_fail_atomically() -> None:
    provider = _RecordingGroupReward(term_name="task")
    trajectories = [
        _trajectory("g0_k0", "g0", "1"),
        _trajectory("g0_k1", "g0", "2"),
    ]
    with pytest.raises(RewardContractError, match="training providers.*duplicate optimized term 'task'"):
        RewardWorker(
            [
                bind_candidate_provider(
                    "pointwise_task",
                    lambda _candidate, _context: RewardProviderResult(terms={"task": 0.5}),
                ),
                _bound_group_reward(provider),
            ],
            training_group_size=2,
        )

    assert provider.calls == []
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_terms_in_validation_provider_inventory_fail_at_construction() -> None:
    validation_provider = _RecordingGroupReward(term_name="judge")
    with pytest.raises(RewardContractError, match="validation providers.*duplicate optimized term 'judge'"):
        RewardWorker(
            [
                bind_candidate_provider(
                    "training",
                    lambda _candidate, _context: RewardProviderResult(terms={"task": 0.0}),
                )
            ],
            validation_providers=[
                bind_candidate_provider(
                    "pointwise_judge",
                    lambda _candidate, _context: RewardProviderResult(terms={"judge": 0.5}),
                    term_names=("judge",),
                ),
                _bound_group_reward(validation_provider, name="group_judge"),
            ],
            training_group_size=1,
        )

    assert validation_provider.calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_provider_names_fail_at_construction_even_when_terms_are_disjoint() -> None:
    first_provider = _RecordingGroupReward(term_name="first")
    second_provider = _RecordingGroupReward(term_name="second")
    with pytest.raises(RewardContractError, match="training providers.*duplicate provider name 'duplicate'"):
        RewardWorker(
            [
                _bound_group_reward(first_provider, name="duplicate"),
                _bound_group_reward(second_provider, name="duplicate"),
            ],
            training_group_size=2,
        )

    assert first_provider.calls == []
    assert second_provider.calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_bare_provider_cannot_bypass_catalog_binding() -> None:
    provider = _RecordingGroupReward()
    with pytest.raises(RewardContractError, match="catalog-bound BoundRewardProvider"):
        RewardWorker([provider], training_group_size=2)  # type: ignore[list-item]

    assert provider.calls == []
