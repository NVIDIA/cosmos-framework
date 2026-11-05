# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L1 @CPU: FAN-OUT — a producer feeding MORE consumer replicas must split, not broadcast.

The `("rollout", "generate", "prompts")` edge is registered `Sharded(split=split_prompts)`, which
only fires when consumer replicas EXCEED producer units (`op="split"`). Every multi-replica run to
date declared data and rollout at equal replica counts, so that branch never executed — the fan-in
(`reduce`) and equal (`one`) branches did. These run it.

Ray-backed but assigned to a CPU DeviceGroup with stub bodies: the property under test is ROUTING,
not generation.
"""

from __future__ import annotations

import pytest


class _Data:
    def get_samples(self, n):
        return [f"p{i}" for i in range(n)]


class _Rollout:
    def __init__(self):
        self._seen = None

    def generate(self, prompts, group=1):
        self._seen = list(prompts)
        return list(prompts)

    def received(self):
        return self._seen


@pytest.fixture
def bind():
    """Bind a (1 data unit -> N rollout replicas) mesh and ALWAYS shut Ray down afterwards.

    Each test starts a cluster; without teardown the actors from one test outlive it and the next
    test binds against a cluster already holding them, which shows up as a hang rather than a
    failure. The rest of this suite uses try/finally + ray.shutdown() for the same reason; a fixture
    keeps it from depending on each test remembering — and, unlike a success flag, still tears down
    when the bind itself raises.
    """

    def _make(rollout_replicas: int, rollout_ranks: int):
        from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
        from cosmos_framework.posttrain.roles import roles

        data = roles.env("data", replicas=1, ranks=1)  # ONE producer unit
        rollout = roles.rollout("rollout", replicas=rollout_replicas, ranks=rollout_ranks)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(
                        ActorGroup(
                            members=(data, rollout),
                            setup=lambda: {"data": _Data(), "rollout": _Rollout()},
                        ),
                    ),
                    device="cpu",
                )
            ],
            backend="ray",
            threshold=0,
        )
        return data, rollout

    try:
        yield _make
    finally:
        # Gate on Ray's ACTUAL state, not on whether the bind succeeded. An earlier version set a
        # flag after bind_device_groups returned, so a bind that raised PART WAY -- after starting
        # the cluster, before returning -- skipped teardown entirely and leaked a cluster holding
        # live actors. That is the same hang this fixture exists to prevent, reachable only on the
        # failure path, which is exactly where a test fixture has to be most reliable.
        try:
            import ray
        except ImportError:  # ray optional; nothing could have been started
            return
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.L1
@pytest.mark.CPU
def test_fan_out_splits_prompts_disjointly_and_exhaustively(bind):
    """1 data unit -> 2 rollout replicas: each replica gets its own half, nothing duplicated or lost.

    Broadcasting instead of splitting is the failure this catches, and it would NOT raise: both
    replicas would generate on the full prompt list, so the step produces twice the trajectories it
    should from the same prompts — duplicated work reported as throughput.
    """
    pytest.importorskip("ray")
    data, rollout = bind(rollout_replicas=2, rollout_ranks=1)

    rollout.generate(data.get_samples(6), group=1)
    per_cell = rollout.received().all()
    slices = [got or [] for _, got in sorted(per_cell.items())]
    flat = [p for s in slices for p in s]

    assert all(len(s) == 3 for s in slices), f"each replica should get half of 6: {slices}"
    assert len(flat) == len(set(flat)) == 6, f"prompts duplicated or lost: {slices}"
    assert sorted(flat) == [f"p{i}" for i in range(6)], f"union must be the whole batch: {slices}"


@pytest.mark.L1
@pytest.mark.CPU
def test_fan_out_gives_every_spmd_rank_of_a_replica_the_SAME_slice(bind):
    """The ranks WITHIN a replica are SPMD mirrors — they must land the identical slice.

    `split_prompts` is deterministic in `(i, n)` precisely so this holds. If two ranks of one replica
    received different slices they would generate on divergent prompts, and because each rank returns
    the full output under SPMD, the result is corrupted trajectories rather than an error.
    """
    pytest.importorskip("ray")
    data, rollout = bind(rollout_replicas=2, rollout_ranks=2)

    rollout.generate(data.get_samples(6), group=1)
    by_replica: dict = {}
    for (r, k), got in sorted(rollout.received().all().items()):
        by_replica.setdefault(r, []).append(got or [])
    assert len(by_replica) == 2
    for r, ranks in by_replica.items():
        assert all(s == ranks[0] for s in ranks), f"replica {r} ranks disagree: {ranks}"
    firsts = [v[0] for v in by_replica.values()]
    assert sorted(p for s in firsts for p in s) == [f"p{i}" for i in range(6)]
