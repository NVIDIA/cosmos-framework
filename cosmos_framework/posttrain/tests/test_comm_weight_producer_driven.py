# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import weakref
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.comm.weight_pairing import full_tensor_actor_targets
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, WeightTargetSet

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _assert_bijection_and_colocated(policy_shape: tuple[int, int], rollout_shape: tuple[int, int]) -> None:
    R_p, K_p = policy_shape
    R_r, K_r = rollout_shape
    assert R_p * K_p == R_r * K_r  # equal product precondition
    targets = full_tensor_actor_targets(policy_shape, rollout_shape)
    # every policy cell present exactly once; every rollout cell targeted exactly once (bijection)
    assert len(targets) == R_p * K_p
    assert all(len(route) == 1 for route in targets.values())
    assert len({route[0] for route in targets.values()}) == R_r * K_r
    # co-located: policy cell (rp,kp) at flat g == rollout cell (rr,kr) at the same flat g
    for (rp, kp), ((rr, kr),) in targets.items():
        assert rp * K_p + kp == rr * K_r + kr


@pytest.mark.parametrize(
    ("rollout_shape", "expected"),
    [
        ((1, 8), (0, 5)),
        ((4, 2), (2, 1)),
        ((2, 4), (1, 1)),
    ],
)
def test_equal_product_pairing_is_bijective_and_colocated(
    rollout_shape: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    policy_shape = (1, 8)
    targets = full_tensor_actor_targets(policy_shape, rollout_shape)

    assert targets[(0, 5)] == (expected,)
    _assert_bijection_and_colocated(policy_shape, rollout_shape)


def test_actor_pairing_accepts_unequal_products() -> None:
    assert full_tensor_actor_targets((1, 2), (1, 4)) == {
        (0, 0): ((0, 0), (0, 2)),
        (0, 1): ((0, 1), (0, 3)),
    }


class _RecordingPeer:  # stands in for the co-located rollout cell's PeerClient (passive receiver)
    def __init__(self) -> None:
        self.plan = None
        self.destination_names = None
        self.coverage_checked = False
        self.applied = []  # (i, payload) in push order
        self.stamped = []  # versions stamped (must be exactly one, AFTER every bucket)
        self.events = []

    def set_bucket_plan(self, plan: object) -> None:
        self.plan = plan
        self.events.append("plan")

    def set_weight_destination_names(self, names: tuple[str, ...]) -> None:
        self.destination_names = names
        self.events.append("destinations")

    def check_plan_coverage(self) -> None:
        self.coverage_checked = True
        self.events.append("coverage")

    def apply_bucket(self, payload, i):
        assert self.plan is not None, "plan must be shipped before buckets"
        assert self.coverage_checked, "strict coverage must be checked before any tensor moves"
        assert self.stamped == [], "version stamped before the stream finished"  # ordering guard
        self.applied.append((i, payload))
        self.events.append(f"apply:{i}")

    def set_weight_witness_spec(self, spec: object) -> None:
        self.events.append(f"witness-spec:{spec}")

    def require_weight_witness_verified(self, spec: object) -> None:
        self.events.append(f"witness-verified:{spec}")

    def mark_weight_version_ready(self, version: int) -> None:
        self.stamped.append(version)
        self.events.append("ready")


class _FakeMover:  # any non-SharedMover mover (push_stream_to_peer rejects SharedMover)
    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""


class _FakeStagingBackend:
    def __init__(
        self,
        n: int,
        destination_names: tuple[str, ...] | None = None,
        witness: object | None = None,
    ) -> None:
        self._n = n
        self._plan = [[("w", (1,), "float32")]] * n
        self._destination_names = destination_names
        self._witness = witness
        self.mover = _FakeMover()
        self.materialized = []
        self.staged = []

    def has_plan(self):
        return True

    def ensure_plan(self, model: object) -> list[list[tuple[str, tuple[int, ...], str]]]:
        del model
        return self._plan

    def plan(self):
        return self._plan

    def bucket_count(self):
        return self._n

    def destination_names(self) -> tuple[str, ...] | None:
        return self._destination_names

    def weight_witness_spec(self, model: object) -> object | None:
        del model
        return self._witness

    def materialize_bucket(self, model: object, i: int) -> dict:
        self.materialized.append(i)
        return {"bucket": i}

    def stage_bucket_state(self, state: dict, version: int) -> dict:
        payload = {**state, "version": version, "stage": len(self.staged)}
        self.staged.append(payload)
        return payload


class _FakeProducer:
    def __init__(self, it, publication=None):
        self.model = object()
        self._iter = it
        self._publication = publication

    def publication_version(self) -> int:
        """Mirror `LLMTrainer`: the weight-publication clock, falling back to optimizer lockstep."""
        return self._iter if self._publication is None else self._publication


def test_producer_ships_plan_pushes_buckets_then_stamps_version():
    """Producer-driven stream (design 2026-07-27 §2), the mirror of pull_stream_from_peer: the policy
    ships its bucket plan to the rollout, PUSHES each staged payload (carrying its training _iter) via
    apply_bucket in fixed order, then stamps the rollout's version EXACTLY ONCE at the end."""
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    producer = _FakeProducer(it=7)
    be = _FakeStagingBackend(3, destination_names=("engine.w",))
    peer = _RecordingPeer()
    wt.push_stream_to_peer(producer, peer, be)
    assert peer.plan == be.plan()  # plan shipped to the rollout first (producer is the plan's source)
    assert peer.destination_names == ("engine.w",)
    assert peer.events[:3] == ["plan", "destinations", "coverage"]
    assert peer.coverage_checked  # strict receiver coverage verified before any tensor moves
    assert [i for i, _ in peer.applied] == [0, 1, 2]  # every bucket, fixed order, pushed via apply_bucket
    assert all(p["version"] == 7 for _, p in peer.applied)  # each payload carries the producer _iter
    assert peer.stamped == [7]  # version stamped ONCE, only after the whole stream succeeded


def test_selected_actor_materializes_once_and_stages_once_per_manual_target() -> None:
    """One full-tensor collective result may feed several single-destination NIXL publications."""
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    first = _RecordingPeer()
    second = _RecordingPeer()
    backend = _FakeStagingBackend(2)
    wt.push_stream_to_peer(_FakeProducer(it=9), WeightTargetSet((first, second)), backend)

    assert backend.materialized == [0, 1]
    assert len(backend.staged) == 4  # two buckets × two single-destination targets
    assert [index for index, _payload in first.applied] == [0, 1]
    assert [index for index, _payload in second.applied] == [0, 1]
    assert first.stamped == [9]
    assert second.stamped == [9]


@pytest.mark.parametrize("reject_witness", [False, True])
def test_live_witness_runs_after_coverage_and_gates_readiness(reject_witness: bool) -> None:
    class _RejectingPeer(_RecordingPeer):
        def require_weight_witness_verified(self, spec: object) -> None:
            raise ValueError(f"bad witness {spec}")

    peer = _RejectingPeer() if reject_witness else _RecordingPeer()
    backend = _FakeStagingBackend(2, witness="digest")

    from cosmos_framework.posttrain.comm import weight_transfer as wt

    if reject_witness:
        with pytest.raises(ValueError, match="bad witness"):
            wt.push_stream_to_peer(_FakeProducer(it=5), peer, backend)
    else:
        wt.push_stream_to_peer(_FakeProducer(it=5), peer, backend)

    assert [index for index, _payload in peer.applied] == [0, 1]
    expected_tail = [] if reject_witness else ["witness-verified:digest", "ready"]
    assert peer.events == [
        "plan",
        "witness-spec:digest",
        "coverage",
        "apply:0",
        "apply:1",
        *expected_tail,
    ]
    assert peer.stamped == ([] if reject_witness else [5])


def test_materialized_state_is_released_before_the_next_bucket() -> None:
    """The next gather starts only after the previous full-tensor state loses its last reference."""
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    class _State(dict):
        pass

    class _LifetimeBackend(_FakeStagingBackend):
        def __init__(self) -> None:
            super().__init__(2)
            self.previous: weakref.ReferenceType[_State] | None = None

        def materialize_bucket(self, model: object, i: int) -> dict:
            if self.previous is not None:
                assert self.previous() is None, "previous full bucket survived into the next gather"
            state = _State(bucket=i)
            self.previous = weakref.ref(state)
            self.materialized.append(i)
            return state

    backend = _LifetimeBackend()
    wt.push_stream_to_peer(_FakeProducer(it=9), _RecordingPeer(), backend)
    assert backend.previous is not None and backend.previous() is None


def test_idle_policy_actor_materializes_and_discards_every_bucket() -> None:
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    backend = _FakeStagingBackend(3)
    wt.push_stream_to_peer(_FakeProducer(it=4), WeightTargetSet(()), backend)
    assert backend.materialized == [0, 1, 2]
    assert backend.staged == []


def test_check_plan_coverage_raises_on_incomplete_plan():
    # Symmetric with pull's pre-stream _strict_plan_coverage: a shipped plan that does NOT cover the
    # rollout's model 1:1 must fail loud BEFORE any tensor moves (else import_state(strict=False)
    # silently leaves the uncovered params stale while weight-version publication advertises fresh).
    pytest.importorskip("torch")
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    r = LocalRollout(nn.Linear(2, 2))  # model params: weight, bias

    class _BE:  # plan covers only "weight", omitting "bias" ⇒ not 1:1 with the model
        def plan(self):
            return [[("weight", (2, 2), "float32")]]

    r._weight_backend = _BE()
    with pytest.raises(ValueError, match="does not cover the consumer model"):
        r.check_plan_coverage()


def test_producer_push_raises_on_none_peer():
    # No co-located rollout peer ⇒ bind wiring bug, fail loud (not a silent skip).
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    with pytest.raises(ValueError, match="no routed rollout target"):
        wt.push_stream_to_peer(_FakeProducer(it=1), None, _FakeStagingBackend(2))


def test_producer_push_raises_on_empty_plan():
    # Empty bucket plan ⇒ nothing to sync (empty manifest?), fail loud rather than a no-op push.
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    with pytest.raises(ValueError, match="empty bucket plan"):
        wt.push_stream_to_peer(_FakeProducer(it=1), _RecordingPeer(), _FakeStagingBackend(0))


def test_producer_push_raises_on_shared_mover():
    # SharedMover (single-process shared instance) is not a producer-driven path — the shared case must
    # bind no backend so push_weights no-ops; a self-targeting per-bucket push would deadlock. Fail loud.
    from cosmos_framework.posttrain.comm import weight_transfer as wt
    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

    be = _FakeStagingBackend(2)
    be.mover = SharedMover.__new__(SharedMover)  # an instance without running __init__ (isinstance hit)
    with pytest.raises(ValueError, match="SharedMover"):
        wt.push_stream_to_peer(_FakeProducer(it=1), _RecordingPeer(), be)


def test_producer_does_not_stamp_if_a_bucket_push_raises():
    # A stream that fails partway must leave the OLD version: the stamp happens ONLY after every bucket
    # landed, so a half-synced model never advertises fresh weights (mirror of pull_stream_from_peer).
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    class _RaisingPeer(_RecordingPeer):
        def apply_bucket(self, payload, i):
            if i == 1:
                raise RuntimeError("bucket 1 push failed")
            super().apply_bucket(payload, i)

    peer = _RaisingPeer()
    with pytest.raises(RuntimeError, match="bucket 1 push failed"):
        wt.push_stream_to_peer(_FakeProducer(it=7), peer, _FakeStagingBackend(3))
    assert peer.stamped == []  # never stamped fresh over a half-synced model


def test_actor_fanout_failure_marks_no_target_ready() -> None:
    """A late direct-target failure must precede readiness on every target of the policy actor."""
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    class _RaisingPeer(_RecordingPeer):
        def apply_bucket(self, payload: object, i: int) -> None:
            if i == 1:
                raise RuntimeError("second actor rejected bucket 1")
            super().apply_bucket(payload, i)

    first = _RecordingPeer()
    second = _RaisingPeer()
    with pytest.raises(RuntimeError, match="second actor rejected bucket 1"):
        wt.push_stream_to_peer(
            _FakeProducer(it=7),
            WeightTargetSet((first, second)),
            _FakeStagingBackend(3),
        )
    assert first.stamped == []
    assert second.stamped == []


def test_rollout_mark_weight_version_ready_raises_without_backend():
    # design §5 symmetric fail-loud: a backendless rollout silently drops every pushed bucket in
    # apply_bucket, so it must REFUSE to mark a fresh version ready (mirrors pull's unsynced-model guard).
    pytest.importorskip("torch")
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    r = LocalRollout(nn.Linear(2, 2))  # no _weight_backend bound
    with pytest.raises(ValueError, match="no weight backend bound"):
        r.mark_weight_version_ready(5)


class _FakeRolloutBackend:  # stand-in for the rollout GridBackend the routed arg carries
    def __init__(self, actors, shape=None, wg_shapes=None):
        self.actors = actors
        self.shape = shape  # only the fall-through (general peer form) path reads these two
        self.wg_shapes = wg_shapes or {}


def test_align_one_routes_policy_actor_to_flat_rollout_target_set() -> None:
    """PRODUCER-DRIVEN grid flip (design 2026-07-27 §2): ``_align_one`` for a policy cell over a
    ``push_weights(rollout)`` edge returns a target set containing the flat-index rollout actor.
    Policy (1,8) ⇒ rollout (4,2): policy cell (0,5) (flat g=5) pairs with rollout cell (2,1)
    (5 = 2*2+1). The peer's actor is the ROLLOUT backend's (2,1) actor (the arg's ``_backend``),
    and the peer targets the rollout WG's name."""
    pytest.importorskip("ray")  # GridBackend.__init__ lazily imports workers.replica -> ray
    from cosmos_framework.posttrain.workers.grid import GridBackend
    from cosmos_framework.posttrain.workers.ray import PeerClient
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    rollout_actors = {(r, k): object() for r in range(4) for k in range(2)}  # distinct sentinels
    rollout = WorkerGroup("rollout", "rollout", replicas=4, ranks=2)
    rollout._bind_backend(_FakeRolloutBackend(rollout_actors))

    policy_grid = {(0, k): (f"sid0{k}", object()) for k in range(8)}  # placeholders
    policy_be = GridBackend(
        policy_grid,  # this branch reads the ARG's backend actors, not the policy's own
        shape=(1, 8),
        weight_routes=full_tensor_actor_targets((1, 8), (4, 2)),
    )
    targets = policy_be._align_one(
        rollout,
        0,
        5,
        (1, 8),
        kind="trainer",
        method="push_weights",
        param="rollout",
    )
    assert isinstance(targets, WeightTargetSet) and len(targets.peers) == 1
    pc = targets.peers[0]
    assert isinstance(pc, PeerClient)
    assert pc._host is rollout_actors[(2, 1)]  # co-located rollout cell (flat g=5)
    assert pc._wg_name == "rollout"


def test_align_one_symmetric_push_weights_falls_through_to_dp_aligned() -> None:
    """A SYMMETRIC producer bind ships NO weight_routes (bind gates on shape asymmetry), so
    ``_align_one`` selects rollout (0,5) for policy (0,5) through the general dp_aligned peer form.
    Distinct actor IDs use a remote peer; aliases of the same actor use a local peer even when
    their Python handle objects differ."""
    ray = pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers.grid import GridBackend
    from cosmos_framework.posttrain.workers.ray import PeerClient
    from cosmos_framework.posttrain.workers.worker_group import LocalPeerRef, WorkerGroup

    rollout_actors = {(0, k): SimpleNamespace(_actor_id=ray.ActorID.from_random()) for k in range(8)}
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=8)
    rollout._bind_backend(_FakeRolloutBackend(rollout_actors, shape=(1, 8), wg_shapes={"rollout": (1, 8)}))

    policy_grid = {(0, k): (f"sid0{k}", SimpleNamespace(_actor_id=ray.ActorID.from_random())) for k in range(8)}
    policy_be = GridBackend(policy_grid, shape=(1, 8), weight_routes=None)  # symmetric ⇒ no routes
    pc = policy_be._align_one(rollout, 0, 5, (1, 8), kind="trainer", method="push_weights", param="rollout")
    assert isinstance(pc, PeerClient)
    assert pc._host is rollout_actors[(0, 5)]  # dp_aligned identity — fell through, not routed
    assert pc._wg_name == "rollout"

    policy_be.actors[(0, 5)] = SimpleNamespace(_actor_id=rollout_actors[(0, 5)]._actor_id)
    assert policy_be.actors[(0, 5)] is not rollout_actors[(0, 5)]
    local = policy_be._align_one(rollout, 0, 5, (1, 8), kind="trainer", method="push_weights", param="rollout")
    assert isinstance(local, LocalPeerRef) and local.name == "rollout"


def test_align_one_unequal_route_keeps_idle_actor_explicit() -> None:
    pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers.grid import GridBackend
    from cosmos_framework.posttrain.workers.ray import PeerClient
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    rollout_actors = {(0, lane): object() for lane in range(2)}
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=2)
    rollout._bind_backend(_FakeRolloutBackend(rollout_actors, shape=(1, 2), wg_shapes={"rollout": (1, 2)}))
    policy_grid = {(0, lane): (f"sid0{lane}", object()) for lane in range(4)}
    policy_backend = GridBackend(
        policy_grid,
        shape=(1, 4),
        weight_routes=full_tensor_actor_targets((1, 4), (1, 2)),
    )

    selected = policy_backend._align_one(
        rollout,
        0,
        0,
        (1, 4),
        kind="trainer",
        method="push_weights",
        param="rollout",
    )
    idle = policy_backend._align_one(
        rollout,
        0,
        3,
        (1, 4),
        kind="trainer",
        method="push_weights",
        param="rollout",
    )
    assert isinstance(selected, WeightTargetSet)
    assert len(selected.peers) == 1 and isinstance(selected.peers[0], PeerClient)
    assert isinstance(idle, WeightTargetSet) and idle.peers == ()


def test_align_one_rejects_incomplete_actor_route_map() -> None:
    pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers.grid import GridBackend
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    rollout_actors = {(0, 0): object()}
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=1)
    rollout._bind_backend(_FakeRolloutBackend(rollout_actors, shape=(1, 1), wg_shapes={"rollout": (1, 1)}))
    policy_grid = {(0, rank): (f"sid0{rank}", object()) for rank in range(2)}
    policy_backend = GridBackend(policy_grid, shape=(1, 2), weight_routes={(0, 0): ((0, 0),)})

    with pytest.raises(RuntimeError, match=r"missing policy cell \(0, 1\)"):
        policy_backend._align_one(
            rollout,
            0,
            1,
            (1, 2),
            kind="trainer",
            method="push_weights",
            param="rollout",
        )


def test_align_one_builds_direct_actor_fanout() -> None:
    pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers.grid import GridBackend
    from cosmos_framework.posttrain.workers.ray import RoutedWeightPeerClient
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    rollout_actors = {(0, lane): object() for lane in range(4)}
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=4)
    rollout._bind_backend(_FakeRolloutBackend(rollout_actors, shape=(1, 4), wg_shapes={"rollout": (1, 4)}))
    policy_grid = {(0, lane): (f"sid0{lane}", object()) for lane in range(2)}
    policy_backend = GridBackend(
        policy_grid,
        shape=(1, 2),
        weight_routes=full_tensor_actor_targets((1, 2), (1, 4)),
    )
    routed = policy_backend._align_one(
        rollout,
        0,
        0,
        (1, 2),
        kind="trainer",
        method="push_weights",
        param="rollout",
    )
    assert isinstance(routed, WeightTargetSet)
    assert len(routed.peers) == 1
    peer = routed.peers[0]
    assert isinstance(peer, RoutedWeightPeerClient)
    assert peer._targets == (
        ((0, 0), rollout_actors[(0, 0)]),
        ((0, 2), rollout_actors[(0, 2)]),
    )


def test_push_weights_is_noop_without_backend():
    # Shared-instance / v0 seam: with no _weight_backend bound, push_weights is a no-op returning None
    # (the shared-model fixture's `policy.push_weights()` relies on this). Call the unbound method
    # on a bare object so no heavy trainer construction is needed.
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    class _Bare:
        pass

    assert LLMTrainer.push_weights(_Bare(), None) is None


def test_push_weights_is_noop_with_shared_mover():
    # SharedMover (single-colocation shared instance): trainer + rollout share the model, so a producer
    # push is a semantic no-op — push_weights must return None WITHOUT raising (the self-push guard in
    # push_stream_to_peer would otherwise raise). Mirror of pull_stream_from_peer's SharedMover shortcut.
    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    class _BE:
        mover = SharedMover.__new__(SharedMover)  # an instance without running __init__

    class _Bare:
        _weight_backend = _BE()

    assert LLMTrainer.push_weights(_Bare(), None) is None  # shared instance ⇒ no-op, no raise


def test_weight_sync_drives_producer_push_weights():
    """The loop entry is PRODUCER-driven (design 2026-07-27 §2): weight_sync(rollout, policy) calls
    policy.push_weights(rollout), NOT rollout.apply_weights(policy). Passing the rollout as the arg is
    what the grid's push_weights edge routes to a co-located rollout PeerClient at invoke time."""
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    class _Policy:
        def __init__(self):
            self.pushed_to = None

        def push_weights(self, rollout):
            self.pushed_to = rollout
            return None  # plain body ⇒ nothing to drain

    class _Rollout:
        def apply_weights(self, policy):
            raise AssertionError("apply_weights must not be called under the producer-driven flip")

    rollout, policy = _Rollout(), _Policy()
    weight_sync(rollout, policy)
    assert policy.pushed_to is rollout  # the policy drove the sync, aimed at the rollout


class _RealModelProducer:
    """A producer with a REAL module, so `model_manifest` reports genuine dtypes."""

    def __init__(self, transfer_dtype=None, dtype=None):
        import torch

        self.model = torch.nn.Linear(4, 4).to(dtype or torch.float32)
        self._iter = 1
        if transfer_dtype is not None:
            self.transfer_dtype = transfer_dtype

    def publication_version(self) -> int:
        """Mirror `LLMTrainer`: the weight-publication clock, falling back to optimizer lockstep."""
        return self._iter


class _PlanCapturingBackend(WeightSync):
    """A REAL backend with no cached plan, recording the plan production code installs.

    Subclasses `WeightSync` rather than reimplementing it: the assertion target is the plan that
    `WeightSync.ensure_plan` builds, so a hand-rolled stand-in with its own `ensure_plan` would
    keep passing after the production builder broke. Only the mover and the NIXL staging calls are
    faked; `ensure_plan`, `has_plan`, `plan`, and `bucket_count` are the shipped implementations.
    """

    def __init__(self, transfer_dtype: str | None = None):
        super().__init__(mover=_FakeMover(), bucket_bytes=1 << 20, transfer_dtype=transfer_dtype)
        self.captured = None

    def set_bucket_plan(self, plan):
        self.captured = plan
        super().set_bucket_plan(plan)

    def stage_bucket(self, model, i, version):
        return {"bucket": i, "version": version}

    def materialize_bucket(self, model: object, i: int) -> dict:
        """Required by `push_stream_to_peer` since main's NIXL weight-sync change.

        A stand-in that omits this dies with `AttributeError: ... has no attribute
        'materialize_bucket'` raised from inside `weight_transfer.py`, which reads as a production
        bug rather than an out-of-date fake. Mirrors the other backend fakes in this file.
        """
        return {"bucket": i}

    def stage_bucket_state(self, state: dict, version: int) -> dict:
        """Second half of the same NIXL staging contract; same reasoning as `materialize_bucket`."""
        return {**state, "version": version}


def test_producer_push_plan_declares_the_transfer_dtype_not_the_model_dtype():
    """The PRODUCTION push path must declare the wire dtype in its bucket plan.

    `bucket_full_tensors` casts to whatever the plan declares, and the receiver allocates from the
    same plan — so if the plan reports the model's STORAGE dtype the cast is an identity and the
    feature is inert. That is exactly what shipped: `push_stream_to_peer` — the path
    `push_weights` actually drives — built its plan from a bare `model_manifest(producer.model)`.
    Wiring it there was necessary but left `LLMTrainer.get_bucket_plan` behind on the bare
    manifest, so the two plan builders declared DIFFERENT dtypes. The test below pins that they
    agree: `workers/ray.py` sizes a staging buffer from whichever plan was installed first, and
    that buffer cannot grow after bind.

    Asserting on the PLAN rather than on the trainer's constructor kwargs is the point: a test that
    only checks the kwarg reached `LLMTrainer` passes while the production plan still says fp32.

    The declaration now reaches the backend at BIND (see
    `test_workers_ray.py::test_bind_carries_the_body_transfer_dtype_into_the_backend`); what this
    pins is the second half -- that a declaring backend actually restates the plan.
    """
    import torch

    from cosmos_framework.posttrain.comm import weight_transfer as wt

    be = _PlanCapturingBackend(transfer_dtype="bfloat16")
    wt.push_stream_to_peer(_RealModelProducer(transfer_dtype="bfloat16"), _RecordingPeer(), be)
    assert be.captured is not None, "producer must build and install a plan"
    declared = {dtype for bucket in be.captured for _n, _s, dtype in bucket}
    assert declared == {"torch.bfloat16"}, f"plan must declare the WIRE dtype, got {declared}"

    # And the params really are fp32, so this is a genuine storage/wire difference, not a no-op.
    fp32 = _RealModelProducer(transfer_dtype="bfloat16")
    assert {p.dtype for p in fp32.model.parameters()} == {torch.float32}


def test_both_plan_routes_declare_the_same_transfer_dtype():
    """The two plan-construction ROUTES must not disagree about the wire dtype.

    `push_stream_to_peer` builds the plan on the production `push_weights` route;
    `LLMTrainer.get_bucket_plan` builds it on `push_bucket`'s lazy fallback and on the
    `rollout.apply_weights(policy)` route that pulls `peer.get_bucket_plan()`. `workers/ray.py` sizes
    the NIXL staging allocation from whichever plan was installed first, and that allocation cannot
    grow after bind -- so a disagreement means buckets sized against the wrong buffer. With the
    shipped GSM8K recipe (`master_dtype=float32`, `transfer_dtype=bfloat16`) a bare manifest
    declares fp32 against a buffer sized for bf16, roughly a factor of two.

    Both routes now funnel through the single `WeightSync.ensure_plan`, which is what makes them
    agree by construction rather than by two sites being kept in sync -- fixing one and leaving the
    other is how this shipped once already. Asserting on both keeps that collapse load-bearing: a
    future route that rebuilds a manifest of its own would show up here.
    """
    from cosmos_framework.posttrain.comm import weight_transfer as wt
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    push_backend = _PlanCapturingBackend(transfer_dtype="bfloat16")
    wt.push_stream_to_peer(_RealModelProducer(transfer_dtype="bfloat16"), _RecordingPeer(), push_backend)
    via_push = {dtype for bucket in push_backend.captured for _n, _s, dtype in bucket}

    trainer = _RealModelProducer(transfer_dtype="bfloat16")
    trainer._weight_backend = _PlanCapturingBackend(transfer_dtype="bfloat16")
    LLMTrainer.get_bucket_plan(trainer)
    via_trainer = {dtype for bucket in trainer._weight_backend.captured for _n, _s, dtype in bucket}

    assert via_push == via_trainer == {"torch.bfloat16"}, (
        f"the two plan routes disagree: push={via_push} trainer={via_trainer}"
    )


def test_producer_push_plan_falls_back_to_model_dtype_without_transfer_dtype():
    """No `transfer_dtype` on the producer ⇒ the model's own dtype, exactly as before this feature."""
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    be = _PlanCapturingBackend()
    wt.push_stream_to_peer(_RealModelProducer(), _RecordingPeer(), be)  # no transfer_dtype attribute
    declared = {dtype for bucket in be.captured for _n, _s, dtype in bucket}
    assert declared == {"torch.float32"}, f"expected the model dtype, got {declared}"


@pytest.mark.L0
@pytest.mark.CPU
def test_push_stamps_the_publication_clock_not_the_optimizer_counter() -> None:
    """A role whose publication clock is independent must publish THAT version.

    `control.loop.weight_sync` runs its prepare/commit barrier at `policy.iteration()`. PPO owns a
    separate outer policy version while `_iter` counts every minibatch optimizer step, so stamping
    `_iter` marks a version the barrier never asks for and every sync after the first update fails
    with "this rollout lane is ready for <optimizer step>".
    """
    from cosmos_framework.posttrain.comm import weight_transfer as wt

    producer = _FakeProducer(it=4, publication=1)
    peer = _RecordingPeer()
    wt.push_stream_to_peer(producer, peer, _FakeStagingBackend(2, destination_names=("engine.w",)))

    assert peer.stamped == [1]
    assert all(payload["version"] == 1 for _, payload in peer.applied)


def test_push_records_per_phase_timing() -> None:
    """The producer leaves ONE `WeightPushTiming` per stream on its backend: a phase per converged step, one entry
    per bucket, CUDA spans None where no device work ran, phases that partition the total, and the next stream
    replaces the record rather than accumulating."""
    import torch

    from cosmos_framework.posttrain.comm import weight_transfer as wt

    be = _FakeStagingBackend(3)
    wt.push_stream_to_peer(_FakeProducer(it=7), _RecordingPeer(), be)
    timing = be.last_push_timing
    assert isinstance(timing, wt.WeightPushTiming) and timing.version == 7
    assert len(timing.materialize_s) == len(timing.deliver_s) == len(timing.gather_gpu_s) == 3
    if torch.cuda.is_available():
        assert all(g is None or g >= 0.0 for g in timing.gather_gpu_s)
    else:
        assert timing.gather_gpu_s == (None, None, None)
    phases = (timing.plan_s, timing.coverage_s, *timing.materialize_s, *timing.deliver_s, timing.ready_s)
    assert all(p >= 0.0 for p in phases)
    assert timing.total_s >= sum(phases)
    wt.push_stream_to_peer(_FakeProducer(it=8), _RecordingPeer(), be)
    assert be.last_push_timing.version == 8 and len(be.last_push_timing.deliver_s) == 3
