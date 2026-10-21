# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `algorithm/annotators/advantage.py` — the group-relative estimator and its reward contract.

These live apart from `test_algorithm_grpo.py` because the estimator is a separate stage from the
surrogate: it runs at REWARD time, over a whole group, and writes `transition.advantage`; the
objective only reads that number. Keeping the two test files apart is what keeps the two concerns
from being tested as one, which is how the group-atomicity constraint leaked into the objective in
the first place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmos_framework.posttrain.algorithm.annotators.advantage import (
    GROUP_REWARD_MEAN,
    GROUP_REWARD_NONZERO_FRACTION,
    GROUP_REWARD_SPREAD,
    INFORMATIVE_GROUP_GATE,
    distributed_group_relative,
    group_relative,
    is_informative,
)
from cosmos_framework.posttrain.algorithm.reward import RewardContractError
from cosmos_framework.posttrain.algorithm.reward.contract import _terminal_reward
from cosmos_framework.posttrain.control.buffer import MetadataGate
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.utils.batch_stats import reward_batch_stats


def _traj(group_id: str, reward: float, tid: str) -> Trajectory:
    tr = Transition(obs=ObsBundle(), reward={"final": reward}, done=True)
    return Trajectory(transitions=[tr], id=tid, group_id=group_id, status=Status.COMPLETED)


# --------------------------------------------------------------------- the reward contract


@pytest.mark.L0
@pytest.mark.CPU
def test_terminal_reward_requires_an_explicit_finite_final_value() -> None:
    empty = Trajectory(transitions=[], id="empty", group_id="g", status=Status.COMPLETED)
    with pytest.raises(RewardContractError, match="at least one transition"):
        _terminal_reward(empty)

    for reward in (None, {}, {"task": 1.0}):
        trajectory = _traj("g", 0.0, "missing")
        trajectory.transitions[-1].reward = reward
        with pytest.raises(RewardContractError, match=r"reward\['final'\]"):
            _terminal_reward(trajectory)

    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(RewardContractError, match="finite"):
            _terminal_reward(_traj("g", value, "nonfinite"))

    for value in (None, True, "1"):
        trajectory = _traj("g", 0.0, "nonnumeric")
        trajectory.transitions[-1].reward = {"final": value}
        with pytest.raises(RewardContractError, match="to be numeric"):
            _terminal_reward(trajectory)


@pytest.mark.L0
@pytest.mark.CPU
def test_group_advantage_normalizes_within_group():
    batch = [_traj("g0", 1.0, "a"), _traj("g0", 0.0, "b"), _traj("g1", 1.0, "c"), _traj("g1", 1.0, "d")]
    group_relative()(batch)
    assert batch[0].transitions[0].advantage == pytest.approx(1.0, abs=1e-3)
    assert batch[1].transitions[0].advantage == pytest.approx(-1.0, abs=1e-3)
    assert batch[2].transitions[0].advantage == pytest.approx(0.0, abs=1e-3)
    assert batch[3].transitions[0].advantage == pytest.approx(0.0, abs=1e-3)


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_isolated_per_prompt_across_replicas():
    """Reviewer probe: >1 rollout replica must keep each prompt's mean/std ISOLATED after fan-in.
    Two replicas each emit their prompt-index-0 prompt; the group_id is the prompt's own id
    (obs.id), not the batch-local index, so the two groups never merge. Replica 1's completions are
    all-equal (std 0 -> advantage 0); under the old batch-local `f"g0"` group_id the two groups
    would collide into [1,1,0,5,5,5] and these would get a large POSITIVE advantage.

    Two REAL replicas are told apart by their `shard_index` (`roles/env/prompt.py::data_shard`
    threads it from the data role's own worker-group replica count into `ArithmeticEnv`/`DatasetEnv`
    — see `roles/env/dataset.py::_stamp_prompt_id`), so that is what this probe gives each replica
    here. Before ids became deterministic, a plain `uuid.uuid4()` was globally unique regardless of
    config, so an earlier version of this test told the two envs apart with distinct SEEDS instead
    and omitted shard coordinates entirely; that shortcut no longer distinguishes replicas because
    the id is now a pure function of `(shard_index, emitted)`, not of the seed.
    """
    from cosmos_framework.posttrain.roles.env import ArithmeticEnv
    from cosmos_framework.posttrain.roles.rollout.base import group_ids_for_prompt

    G = 3
    p0 = ArithmeticEnv(seed=0, shard_index=0, shard_count=2).reset()
    p1 = ArithmeticEnv(seed=0, shard_index=1, shard_count=2).reset()  # a second, real replica
    assert p0.id != p1.id  # distinct prompts -> distinct group namespace (the fix)
    gid0, tids0 = group_ids_for_prompt(p0, G)
    gid1, tids1 = group_ids_for_prompt(p1, G)
    r0, r1 = [1.0, 1.0, 0.0], [5.0, 5.0, 5.0]  # replica-0 has variance; replica-1 all-equal
    batch = [_traj(gid0, r0[k], tids0[k]) for k in range(G)]
    batch += [_traj(gid1, r1[k], tids1[k]) for k in range(G)]  # fan-in concatenation
    group_relative()(batch)
    # Replica-1's group is all-equal in ISOLATION -> advantage ~0 (would be large + if merged).
    for t in batch[G:]:
        assert t.transitions[0].advantage == pytest.approx(0.0, abs=1e-3)
    # Replica-0 normalizes within its OWN group: reward 1.0 -> positive, 0.0 -> negative.
    assert batch[0].transitions[0].advantage > 0.5
    assert batch[2].transitions[0].advantage < -0.5


# ------------------------------------------------------------- the `unbiased` variant (Dr-GRPO)


@pytest.mark.L0
@pytest.mark.CPU
def test_unbiased_matches_numpy_reference_both_branches():
    """Both branches reproduce cosmos-rl's numpy formulas exactly, including np.std's population
    denominator (÷N, not ÷(N-1)) — the detail that would silently shift every advantage."""
    np = pytest.importorskip("numpy")
    rewards = [1.0, 0.0, 0.0, 0.5]
    eps = 1e-6

    biased = [_traj("g0", r, f"b{i}") for i, r in enumerate(rewards)]
    group_relative(eps=eps, unbiased=False)(biased)
    arr = np.array(rewards, dtype=np.float32)
    want_biased = ((arr - arr.mean()) / (arr.std() + eps)).tolist()  # np.std => ddof=0
    assert [t.transitions[0].advantage for t in biased] == pytest.approx(want_biased, rel=1e-5)

    unb = [_traj("g0", r, f"u{i}") for i, r in enumerate(rewards)]
    group_relative(eps=eps, unbiased=True)(unb)
    assert [t.transitions[0].advantage for t in unb] == pytest.approx((arr - arr.mean()).tolist(), rel=1e-6)


@pytest.mark.L0
@pytest.mark.CPU
def test_unbiased_rejects_unused_standard_deviation_correction() -> None:
    with pytest.raises(ValueError, match="correction is undefined"):
        group_relative(unbiased=True, correction=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_requires_positive_finite_epsilon() -> None:
    for eps in (0.0, -1e-4, float("nan"), float("inf"), True, "1e-4"):
        with pytest.raises(ValueError, match="eps must be finite and > 0"):
            distributed_group_relative(eps=eps, group_size=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        distributed_group_relative(group_size=1)
    for correction in (-1, 2):
        with pytest.raises(ValueError, match="correction must be 0 or 1"):
            distributed_group_relative(group_size=2, correction=correction)
        with pytest.raises(ValueError, match="correction must be 0 or 1"):
            group_relative(correction=correction)


def _distributed_advantage_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch
    import torch.distributed as dist

    reward_values = [-2.8660824, 1e-50, 4.0]
    local_trajectories = [_traj("group", reward, f"local-{index}") for index, reward in enumerate(reward_values)]
    local_estimator = distributed_group_relative(group_size=world_size, correction=1)
    local_estimator(local_trajectories)
    local_advantages = [trajectory.transitions[0].advantage for trajectory in local_trajectories]
    local_metrics = local_estimator.metrics()

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        trajectories = [_traj("group", reward_values[rank], f"rank-{rank}")]
        estimator = distributed_group_relative(group_size=world_size, correction=1)
        estimator(trajectories)
        global_rewards = torch.tensor(reward_values, dtype=torch.float32)  # [G]
        expected_mean = float(global_rewards.mean().item())
        expected_spread = float((global_rewards.max() - global_rewards.min()).item())
        expected_std = float(global_rewards.std(correction=1).item())
        expected = (float(global_rewards[rank].item()) - expected_mean) / (expected_std + 1e-4)
        transition = trajectories[0].transitions[0]
        assert transition.advantage == pytest.approx(expected)
        assert transition.advantage == local_advantages[rank]
        assert transition.algo_extra[GROUP_REWARD_MEAN] == expected_mean
        assert transition.algo_extra[GROUP_REWARD_SPREAD] == expected_spread
        assert transition.algo_extra[GROUP_REWARD_NONZERO_FRACTION] == pytest.approx(2 / 3)
        metrics = estimator.metrics()
        assert metrics == local_metrics
        assert metrics["rollout/reward_group_mean"] == expected_mean
        assert metrics["rollout/reward_group_spread"] == expected_spread
        assert metrics["rollout/reward_nonzero_fraction"] == pytest.approx(2 / 3)
        assert metrics["rollout/reward_group_size"] == world_size
        assert metrics["rollout/reward_group_informative"] == 1
        assert metrics["rollout/reward_group_all_zero"] == 0
        expected_advantages = (global_rewards - expected_mean) / (expected_std + 1e-4)  # [G]
        assert metrics["rollout/advantage_group_std"] == pytest.approx(
            float(expected_advantages.std(correction=1).item())
        )
        assert metrics["rollout/advantage_abs_max"] == pytest.approx(float(expected_advantages.abs().max().item()))
        if rank == 1:
            assert abs(float(expected_advantages[rank].item())) < metrics["rollout/advantage_abs_max"]
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_matches_one_group_across_ranks(tmp_path: Path) -> None:
    """The complete-local-group and one-member-per-rank call shapes produce identical results."""
    import torch.multiprocessing as mp

    world_size = 3
    rendezvous_path = str(tmp_path / "distributed-advantage-rendezvous")
    mp.spawn(_distributed_advantage_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_reports_tied_group_reward_facts() -> None:
    """Without a process group the WFM estimator accepts one complete local group."""
    for reward, nonzero_fraction, all_zero in ((0.0, 0.0, 1), (1.0, 1.0, 0)):
        estimator = distributed_group_relative(group_size=2, correction=1)
        trajectories = [_traj("group", reward, "a"), _traj("group", reward, "b")]

        estimator(trajectories)

        metrics = estimator.metrics()
        assert metrics["rollout/reward_group_mean"] == pytest.approx(reward)
        assert metrics["rollout/reward_group_spread"] == pytest.approx(0.0)
        assert metrics["rollout/reward_nonzero_fraction"] == pytest.approx(nonzero_fraction)
        assert metrics["rollout/reward_group_size"] == 2
        assert metrics["rollout/reward_group_informative"] == 0
        assert metrics["rollout/reward_group_all_zero"] == all_zero
        assert metrics["rollout/advantage_group_std"] == pytest.approx(0.0)
        assert metrics["rollout/advantage_abs_max"] == pytest.approx(0.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_complete_local_distributed_advantage_annotations_match_reward_reporter() -> None:
    trajectories = [
        _traj("group", 0.0, "zero"),
        _traj("group", 1.0, "one-a"),
        _traj("group", 1.0, "one-b"),
    ]
    estimator = distributed_group_relative(group_size=3, correction=1)

    estimator(trajectories)
    report = reward_batch_stats(trajectories)

    assert estimator.metrics()["rollout/reward_group_mean"] == pytest.approx(2.0 / 3.0)
    assert report.mean_reward == pytest.approx(2.0 / 3.0)
    assert report.groups == 1
    assert report.informative_groups == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_invalidates_metrics_before_failure() -> None:
    estimator = distributed_group_relative(group_size=2, correction=1)
    estimator([_traj("group", 0.0, "a"), _traj("group", 1.0, "b")])
    assert estimator.metrics()["rollout/reward_group_informative"] == 1

    with pytest.raises(ValueError, match="incomplete group"):
        estimator([_traj("group", 0.0, "only-member")])

    assert estimator.metrics() == {}


def _distributed_advantage_world_size_mismatch_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        with pytest.raises(RuntimeError, match="group_size=4 must equal world_size=2"):
            distributed_group_relative(group_size=4, correction=1)([_traj("group", float(rank), f"rank-{rank}")])
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_rejects_world_size_mismatch(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "distributed-advantage-group-size-rendezvous")
    mp.spawn(_distributed_advantage_world_size_mismatch_worker, args=(2, rendezvous_path), nprocs=2, join=True)


def _distributed_advantage_multiple_local_members_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        trajectories = [_traj("group", float(rank), f"rank-{rank}-a")]
        if rank == 1:
            trajectories.append(_traj("group", float(rank), f"rank-{rank}-b"))
        with pytest.raises(
            RuntimeError, match="rank 1: ValueError: distributed advantage expected 1 local member, got 2"
        ):
            distributed_group_relative(group_size=world_size, correction=1)(trajectories)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_converges_multiple_local_members(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "distributed-advantage-local-members-rendezvous")
    mp.spawn(_distributed_advantage_multiple_local_members_worker, args=(2, rendezvous_path), nprocs=2, join=True)


def _distributed_advantage_malformed_rank_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        trajectory = _traj("group", float(rank), f"rank-{rank}")
        if rank == 1:
            trajectory.transitions[-1].reward = None
        with pytest.raises(RuntimeError, match="preparation failed on one or more ranks"):
            distributed_group_relative(group_size=world_size, correction=1)([trajectory])
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_distributed_advantage_converges_rank_local_contract_failure(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "distributed-advantage-malformed-rendezvous")
    mp.spawn(
        _distributed_advantage_malformed_rank_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_estimator_catches_short_groups_where_the_objective_no_longer_can():
    """The completeness check moved to the ESTIMATOR, which is the only place it can still mean something.

    It used to live in the objective (`_require_complete_groups`), where it observed a CONSEQUENCE:
    by then the batch had already been through dispatch, so a short group could mean either "the
    rollout lost samples" or "dispatch split the group across ranks" — and the objective had to
    forbid the second to detect the first. That forced group atomicity on every layer downstream.

    At reward time the two are distinguishable. A group arrives intact BY CONSTRUCTION (one rollout
    call produces a prompt's G completions and the reward stage sees exactly that), so a short group
    can only mean samples were dropped. The check survives; the constraint it used to impose on
    dispatch does not.
    """

    def _g(gid: str, n: int) -> list[Trajectory]:
        return [_traj(gid, float(i), f"{gid}_{i}") for i in range(n)]

    group_relative(group_size=4)(_g("a", 4) + _g("b", 4))  # whole groups: fine
    group_relative()(_g("a", 1))  # group_size=None disables the check

    with pytest.raises(ValueError, match="incomplete group"):
        group_relative(group_size=4)(_g("a", 2) + _g("b", 2))  # evenly split -> internally consistent
    with pytest.raises(ValueError, match="incomplete group"):
        group_relative(group_size=4)(_g("a", 4) + _g("b", 3))
    with pytest.raises(ValueError, match="incomplete group"):
        group_relative(group_size=4)(_g("a", 5))  # oversized: two groups merged at fan-in


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_group_id_is_rejected_even_when_the_count_looks_right() -> None:
    """The hole `group_size` cannot see: exactly G id-less trajectories form ONE valid-looking group.

    `Trajectory.group_id` defaults to None, so every trajectory a producer forgot to key lands in the
    same bucket. At exactly `group_size` of them the completeness check passes and reports nothing --
    and G completions of G DIFFERENT prompts get normalized against each other, making the advantage
    measure question difficulty rather than completion quality. Nothing raises, the loss looks
    healthy, and the run is wrong.

    So the guard must fire BEFORE grouping, and must not be satisfiable by supplying `group_size`.
    """
    orphans = [_traj(None, float(i), f"p{i}") for i in range(4)]

    # The failure mode: without the guard these four would be one "complete" group of 4.
    with pytest.raises(RewardContractError, match="group_id=None"):
        group_relative(group_size=4)(orphans)
    # ...and declaring no group_size must not make it acceptable either.
    with pytest.raises(RewardContractError, match="group_id=None"):
        group_relative()(orphans)
    # A single orphan mixed into otherwise-keyed traffic is caught too.
    mixed = [_traj("g", 1.0, "a"), _traj("g", 0.0, "b"), _traj(None, 1.0, "stray")]
    with pytest.raises(RewardContractError, match="group_id=None"):
        group_relative()(mixed)
    # No advantage was written on the way out.
    assert all(t.transitions[0].advantage is None for t in orphans + mixed)


# ------------------------------------------------------------------ the informative-group stamp


@pytest.mark.L0
@pytest.mark.CPU
def test_is_informative_reads_the_estimator_stamp() -> None:
    """The stamp is the one definition of informative, shared with the buffer gate; unstamped is refused."""
    mixed = [_traj("m", 0.0, "m0"), _traj("m", 1.0, "m1")]
    tied = [_traj("t", 1.0, "t0"), _traj("t", 1.0, "t1")]
    group_relative(group_size=2)(mixed + tied)

    verdicts = [is_informative(trajectory) for trajectory in mixed + tied]
    assert verdicts == [True, True, False, False]
    assert [MetadataGate(*INFORMATIVE_GROUP_GATE)(trajectory) for trajectory in mixed + tied] == verdicts
    with pytest.raises(RewardContractError, match="informative-group check"):
        is_informative(_traj("u", 1.0, "unstamped"))
