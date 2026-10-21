# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: strict, mergeable reward-boundary validation reporting."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.algorithm.annotators.advantage import GROUP_REWARD_SPREAD, group_relative
from cosmos_framework.posttrain.algorithm.reward import RewardContractError
from cosmos_framework.posttrain.comm.transfer.mover import BulkRef
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.utils.batch_stats import (
    IdleRewardReport,
    RewardBatchStats,
    resolve_reward_report,
    reward_batch_stats,
)
from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle


def _traj(group_id: str, reward: float, trajectory_id: str) -> Trajectory:
    transition = Transition(
        obs=ObsBundle(representations={"reference": f"reference-{trajectory_id}"}),
        policy_info={"completion_text": f"completion-{trajectory_id}"},
        reward={"final": reward},
        done=True,
    )
    return Trajectory(
        transitions=[transition],
        id=trajectory_id,
        group_id=group_id,
        status=Status.COMPLETED,
    )


def _scored(groups: list[tuple[str, list[float]]]) -> list[Trajectory]:
    trajectories = [
        _traj(group_id, reward, f"{group_id}-{index}")
        for group_id, rewards in groups
        for index, reward in enumerate(rewards)
    ]
    group_relative()(trajectories)
    return trajectories


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_report_requires_explicit_idle_cells_and_nonempty_active_inventory() -> None:
    active = reward_batch_stats(_scored([("g", [0.0, 1.0])]))
    reports = CollectiveHandle({(0, 0): IdleRewardReport(), (1, 0): active}, (2, 1))
    assert resolve_reward_report(reports, None, idle_cells=(0,)) == active
    with pytest.raises(RewardContractError, match="active reward cell 0"):
        resolve_reward_report(reports, None)
    with pytest.raises(RewardContractError, match="idle reward cell 1"):
        resolve_reward_report(reports, None, idle_cells=(0, 1))
    for indices in ((0, 0), (2,), (-1,), (True,)):
        with pytest.raises(ValueError, match="invalid idle reward cell"):
            resolve_reward_report(reports, None, idle_cells=indices)
    missing = CollectiveHandle({(0, 0): None, (1, 0): active}, (2, 1))
    with pytest.raises(RewardContractError, match="idle reward cell 0"):
        resolve_reward_report(missing, None, idle_cells=(0,))
    all_idle = CollectiveHandle({(0, 0): IdleRewardReport()}, (1, 1))
    with pytest.raises(ValueError, match="empty reward report inventory"):
        resolve_reward_report(all_idle, None, idle_cells=(0,))


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_distinguishes_signal_ties_and_degenerate_groups() -> None:
    report = reward_batch_stats(_scored([("live", [0.0, 1.0]), ("zero", [0.0, 0.0]), ("one", [1.0, 1.0])]))

    assert report.trajectories == 6
    assert report.mean_reward == pytest.approx(0.5)
    assert report.reward_std == pytest.approx(0.5)
    assert (report.reward_min, report.reward_max) == (0.0, 1.0)
    assert report.groups == 3
    assert report.informative_groups == 1
    assert report.tied_groups == 2
    assert report.all_zero_groups == 1
    assert report.all_one_groups == 1
    assert [preview.trajectory_id for preview in report.previews] == ["live-0", "live-1"]
    assert report.metrics() == {
        "train/reward_mean": 0.5,
        "train/reward_std": 0.5,
        "train/reward_min": 0.0,
        "train/reward_max": 1.0,
        "train/rollout_count": 6,
        "train/grpo_group_count": 3,
        "train/grpo_informative_group_count": 1,
        "train/grpo_tied_group_count": 2,
        "train/grpo_all_zero_group_count": 1,
        "train/grpo_all_one_group_count": 1,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_preserves_zero_but_rejects_missing_or_nonfinite_reward() -> None:
    zero_report = reward_batch_stats(_scored([("zero", [0.0, 0.0])]))
    assert zero_report.mean_reward == 0.0
    assert zero_report.all_zero_groups == 1

    for reward in (None, {}, {"task": 0.0}):
        malformed = _scored([("g", [0.0, 1.0])])
        malformed[0].transitions[-1].reward = reward
        with pytest.raises(RewardContractError, match=r"reward\['final'\]"):
            reward_batch_stats(malformed)

    malformed = _scored([("g", [0.0, 1.0])])
    malformed[0].transitions[-1].reward = {"final": float("nan")}
    with pytest.raises(RewardContractError, match="finite"):
        reward_batch_stats(malformed)


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_rejects_absent_or_inconsistent_group_annotations() -> None:
    unstamped = [_traj("g", 0.0, "a"), _traj("g", 1.0, "b")]
    with pytest.raises(RewardContractError, match="group-relative estimator"):
        reward_batch_stats(unstamped)

    inconsistent = _scored([("g", [0.0, 1.0])])
    inconsistent[0].transitions[-1].algo_extra[GROUP_REWARD_SPREAD] = 0.0
    with pytest.raises(RewardContractError, match="inconsistent group annotations"):
        reward_batch_stats(inconsistent)

    with pytest.raises(RewardContractError, match="non-empty"):
        reward_batch_stats([])


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_merge_is_sample_weighted_and_partition_invariant() -> None:
    first = reward_batch_stats(_scored([("a", [0.0, 1.0])]))
    second = reward_batch_stats(_scored([("b", [1.0, 1.0]), ("c", [0.0, 0.0, 0.0])]))
    merged = RewardBatchStats.merge([first, second])
    whole = reward_batch_stats(_scored([("a", [0.0, 1.0]), ("b", [1.0, 1.0]), ("c", [0.0, 0.0, 0.0])]))

    assert merged.metrics() == whole.metrics()
    assert merged.mean_reward == pytest.approx(3.0 / 7.0)
    assert merged.reward_std == pytest.approx(whole.reward_std)
    assert len(merged.previews) == 2
    assert [preview.trajectory_id for preview in merged.previews] == ["a-0", "b-0"]


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_merges_provider_provenance_and_rejects_revision_conflicts() -> None:
    first = reward_batch_stats(
        _scored([("a", [0.0, 1.0])]),
        provider_provenance=(("custom", "r1"),),
    )
    second = reward_batch_stats(
        _scored([("b", [0.0, 1.0])]),
        provider_provenance=(("custom", "r1"), ("other", "v2")),
    )

    merged = RewardBatchStats.merge([first, second])

    assert merged.provider_provenance == (("custom", "r1"), ("other", "v2"))

    conflicting = reward_batch_stats(
        _scored([("c", [0.0, 1.0])]),
        provider_provenance=(("custom", "r2"),),
    )
    with pytest.raises(ValueError, match="custom.*changed revision.*r1.*r2"):
        RewardBatchStats.merge([first, conflicting])


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_matches_cosmos_rl_token_diagnostics() -> None:
    trajectories = _scored([("a", [0.0, 1.0]), ("b", [0.0, 0.0])])
    completion_ids = ([1], [2, 3, 4], [5, 6], [])
    for trajectory, token_ids in zip(trajectories, completion_ids):
        trajectory.transitions[0].action = list(token_ids)

    report = reward_batch_stats(trajectories)
    metrics = report.metrics()

    assert metrics["rollout/completion_length_mean"] == pytest.approx(1.5)
    assert metrics["rollout/completion_length_std"] == pytest.approx(5**0.5 / 2)
    assert metrics["rollout/completion_length_min"] == 0.0
    assert metrics["rollout/completion_length_max"] == 3.0
    # Group a has advantages [-scale,+scale], where scale includes group_relative's epsilon,
    # repeated by completion-token count [1,3]. Group b is tied.
    scale = 0.5 / (0.5 + 1e-6)
    assert metrics["rollout/advantage_mean"] == pytest.approx(2.0 * scale / 6.0)
    assert metrics["rollout/advantage_min"] == pytest.approx(-scale)
    assert metrics["rollout/advantage_max"] == pytest.approx(scale)


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_uses_stamped_length_without_iterating_bulkref() -> None:
    trajectories = _scored([("a", [0.0, 1.0])])
    for trajectory, length in zip(trajectories, (2, 4)):
        trajectory.transitions[0].action = BulkRef.for_bulk("rollout", key=length)
        trajectory.transitions[0].policy_info["completion_token_count"] = length

    metrics = reward_batch_stats(trajectories).metrics()

    assert metrics["rollout/completion_length_mean"] == pytest.approx(3.0)
    assert metrics["rollout/completion_length_min"] == 2.0
    assert metrics["rollout/completion_length_max"] == 4.0


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_batch_stats_previews_first_response_with_terminal_reward() -> None:
    trajectories: list[Trajectory] = []
    for index, reward in enumerate((0.0, 1.0)):
        first = Transition(
            obs=ObsBundle(representations={"reference": f"source-reference-{index}"}),
            policy_info={"completion_text": f"first-completion-{index}"},
            done=False,
        )
        terminal = Transition(
            obs=ObsBundle(representations={"reference": f"terminal-reference-{index}"}),
            policy_info={"completion_text": f"terminal-completion-{index}"},
            reward={"final": reward},
            done=True,
        )
        trajectories.append(
            Trajectory(
                transitions=[first, terminal],
                id=f"multi-{index}",
                group_id="multi",
                status=Status.COMPLETED,
            )
        )

    group_relative()(trajectories)
    report = reward_batch_stats(trajectories, max_previews=1)

    preview = report.previews[0]
    assert (
        preview.trajectory_id,
        preview.reward,
        preview.reference_tail,
        preview.completion,
    ) == (
        "multi-0",
        0.0,
        "source-reference-0",
        "first-completion-0",
    )
