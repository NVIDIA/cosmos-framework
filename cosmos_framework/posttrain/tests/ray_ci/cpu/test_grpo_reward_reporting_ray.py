# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GRPO reward reporting and scored-handle routing on two real Ray workers."""

from __future__ import annotations

from typing import Any

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardCombiner,
    RewardContext,
    RewardProviderResult,
    bind_candidate_provider,
)
from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.utils.batch_stats import resolve_reward_report
from cosmos_framework.posttrain.utils.entry import replica_lead_values, resolve_all
from cosmos_framework.posttrain.workers.worker_group import Scatter


def _declared_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Derive a deterministic candidate-local score without declaring group context."""
    del context
    return RewardProviderResult(terms={"task": float(candidate.trajectory_id.endswith("t1"))})


class _BulkSink:
    """Bulk-consuming test role proving routed tensors remain materializable."""

    def inspect(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        return {
            "ids": [trajectory.id for trajectory in trajectories],
            "actions": [trajectory.transitions[-1].action.tolist() for trajectory in trajectories],
        }


def _setup() -> dict[str, Any]:
    return {
        "rm": RewardWorker(
            [bind_candidate_provider("declared", _declared_reward)],
            RewardCombiner(),
            annotators=(group_relative(group_size=2),),
            training_group_size=2,
        ),
        "replay": ExperienceBuffer(),
        "sink": _BulkSink(),
    }


def _corrupt_second_replica(trajectories: list[Trajectory]) -> None:
    """Corrupt one replica's estimator stamp after the real estimator ran."""
    if trajectories and str(trajectories[0].group_id).startswith("r1-"):
        trajectories[0].transitions[-1].algo_extra["group_reward_spread"] = 0.0


def _setup_with_corrupt_report() -> dict[str, Any]:
    return {
        "rm": RewardWorker(
            [bind_candidate_provider("declared", _declared_reward)],
            RewardCombiner(),
            annotators=(group_relative(group_size=2), _corrupt_second_replica),
            training_group_size=2,
        ),
        "replay": ExperienceBuffer(),
    }


def _batch(replica: int) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for index, reward in enumerate((0.0, 1.0)):
        transition = Transition(
            obs=ObsBundle(representations={"reference": reward}, id=f"r{replica}-g"),
            action=torch.tensor([replica, index]),  # shape: [2]; values are replica id, sample id
            policy_info={"completion_text": f"replica {replica} sample {index}"},
            done=True,
        )
        trajectories.append(
            Trajectory(
                transitions=[transition],
                id=f"r{replica}-t{index}",
                group_id=f"r{replica}-g",
                status=Status.COMPLETED,
            )
        )
    return trajectories


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("idle_first", [False, True])
def test_reward_report_is_barrier_without_materializing_scored_batches_on_driver(idle_first: bool) -> None:
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/c3-grpo-report",
    )
    # Production GSM8K enables shared-prefix filtering, which makes the reward WG bulk-consuming.
    # This stronger shape proves reporting does not re-ingress and consume the scored BulkRefs.
    rm = roles.reward("rm", replicas=2, needs_bulk=True)
    replay = roles.buffer("replay", replicas=2)
    sink = roles.reward("sink", replicas=2, needs_bulk=True)
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rm, replay, sink), setup=_setup),),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
        )

        scored = rm.score_with_report(
            Scatter([[] if idle_first else _batch(0), _batch(1)]), idle=Scatter([idle_first, False])
        )
        report = resolve_reward_report(rm.batch_stats(), scored, idle_cells=(0,) if idle_first else ())
        assert report.trajectories == (2 if idle_first else 4)
        assert report.groups == (1 if idle_first else 2)
        assert report.informative_groups == report.groups
        assert report.mean_reward == 0.5

        # Reuse the original scored handle after the report barrier. The controller never turns it
        # into a concrete trajectory list; replica alignment and BulkRef ownership remain intact.
        resolve_all(replay.put_samples(scored))
        inspected = replica_lead_values(sink.inspect(replay.get_samples(2)))
        assert inspected == [
            {"ids": [], "actions": []} if idle_first else {"ids": ["r0-t0", "r0-t1"], "actions": [[0, 0], [0, 1]]},
            {"ids": ["r1-t0", "r1-t1"], "actions": [[1, 0], [1, 1]]},
        ]
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_one_bad_reward_replica_fails_the_report_barrier_before_replay() -> None:
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups

    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/c3-grpo-report-bad",
    )
    rm = roles.reward("rm", replicas=2, needs_bulk=True)
    replay = roles.buffer("replay", replicas=2)
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rm, replay), setup=_setup_with_corrupt_report),),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
        )

        scored = rm.score_with_report(Scatter([_batch(0), _batch(1)]))
        with pytest.raises(Exception, match="inconsistent group annotations"):
            resolve_reward_report(rm.batch_stats(), scored)

        # The driver stops at the report barrier, so no replica receives a replay write.
        assert replica_lead_values(replay.get_samples(2)) == [[], []]
    finally:
        ray.shutdown()
