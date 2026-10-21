# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""WeightSync (mover-backed) + mapper unit tests (no ray, no GPU)."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.comm.transfer.movers import CopyMover, SharedMover
from cosmos_framework.posttrain.comm.weight_transfer import (
    WeightPayload,
    WeightSync,
    make_weight_backend,
    shared_model_identity,
)

# Weight-plane tests never call the loss, but LLMTrainer's ctor reads `.reduction`.
_UNUSED_OBJECTIVE = Objective(lambda *a, **k: None, reduction=Reduction.MEAN)


class _MoverDouble:
    """The parts of the `Mover` protocol the collective doubles below do not exercise.

    `WeightSync` calls `reserve` when a plan is installed and `release_staged` when a stream ends,
    on EVERY mover -- so a double that omits them fails on plan install, long before the behavior
    it was written to check.
    """

    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""


@pytest.mark.L0
@pytest.mark.CPU
def test_weightsync_reserves_the_installed_plan_and_exposes_release() -> None:
    """Plan sizing and stream teardown reach the mechanism through one backend seam."""
    plan = [[("a", (2,), "torch.float32")], [("b", (4,), "torch.float32")]]

    class _TrackingMover(_MoverDouble):
        def __init__(self) -> None:
            self.reserved: list | None = None
            self.release_calls = 0
            self._release_after_sync = True

        def reserve(self, buckets: list) -> None:
            self.reserved = buckets

        def release_staged(self) -> None:
            self.release_calls += 1

    mover = _TrackingMover()
    backend = WeightSync(mover)
    backend.set_bucket_plan(plan)
    assert mover.reserved is plan
    assert backend.releases_staged_after_sync()
    backend.release_staged()
    assert mover.release_calls == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_weightsync_roundtrip_and_version():
    src = torch.nn.Linear(4, 4)
    dst = torch.nn.Linear(4, 4)
    with torch.no_grad():
        dst.weight.add_(1.0)
    ws = make_weight_backend("copy")
    assert isinstance(ws, WeightSync) and isinstance(ws.mover, CopyMover)
    payload = ws.export(src, version=3)
    assert isinstance(payload, WeightPayload) and payload.version == 3 and payload.handle is not None
    assert not torch.allclose(src.weight, dst.weight)
    ws.apply(dst, payload)
    assert torch.allclose(src.weight, dst.weight) and torch.allclose(src.bias, dst.bias)


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_weightsync_is_noop():
    m = torch.nn.Linear(2, 2)
    before = m.weight.detach().clone()
    ws = make_weight_backend("shared")
    assert isinstance(ws.mover, SharedMover)
    ws.apply(m, ws.export(m, version=1))
    assert torch.allclose(m.weight, before)


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_model_identity_rejects_deepcopied_storage() -> None:
    model = torch.nn.Linear(2, 2)
    identity = shared_model_identity(model)

    assert shared_model_identity(model) == identity
    assert shared_model_identity(copy.deepcopy(model)) != identity


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_weightsync_apply_rejects_missing_payload():
    m = torch.nn.Linear(2, 2)
    ws = make_weight_backend("copy")
    with pytest.raises(ValueError):
        ws.apply(m, None)
    with pytest.raises(ValueError):
        ws.apply(m, WeightPayload(version=1))  # handle is None


@pytest.mark.L0
@pytest.mark.CPU
def test_nccl_weightsync_requires_comm():
    # NcclMover is real (COLLECTIVE) now: with no bound WeightComm, export (stage) and apply (fetch)
    # fail loud with ValueError rather than the old NotImplementedError stub. apply reaches the
    # COLLECTIVE branch (builds recv_plan from the model) before fetch raises on the missing comm.
    m = torch.nn.Linear(2, 2)
    ws = make_weight_backend("nccl")
    with pytest.raises(ValueError):
        ws.export(m, 1)
    with pytest.raises(ValueError):
        ws.apply(m, WeightPayload(version=1, handle=object()))


@pytest.mark.L0
@pytest.mark.CPU
def test_make_weight_backend_kinds() -> None:
    for kind in ("shared", "copy", "ipc", "nccl"):
        assert isinstance(make_weight_backend(kind), WeightSync)
    assert make_weight_backend("ipc").mover._single_buffer is True
    with pytest.raises(KeyError):
        make_weight_backend("bogus")


@pytest.mark.L0
@pytest.mark.CPU
def test_weightsync_streaming_bucketed_roundtrip():
    # Streaming path: sync a source model into a zeroed dst ONE BUCKET AT A TIME; every param must
    # land (partial strict=False load per bucket). Tiny bucket_bytes → weight and bias in separate buckets.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest

    src, dst = nn.Linear(8, 8), nn.Linear(8, 8)
    with torch.no_grad():
        nn.init.zeros_(dst.weight)
        nn.init.zeros_(dst.bias)
    plan = bucket_plan(model_manifest(src), bucket_bytes=64)  # weight (256 elems) and bias split
    prod = WeightSync(CopyMover())
    prod.set_bucket_plan(plan)
    cons = WeightSync(CopyMover())
    cons.set_bucket_plan(plan)
    assert cons.bucket_count() == len(plan) >= 2
    assert not torch.equal(dst.weight, src.weight)
    for i in range(prod.bucket_count()):
        cons.apply_bucket(dst, i, prod.stage_bucket(src, i))  # STAGED: payload carries this bucket
    assert torch.equal(dst.weight, src.weight) and torch.equal(dst.bias, src.bias)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("use_model_mapper", [False, True])
def test_weight_mapper_controls_stream_manifest_export_and_destinations(use_model_mapper: bool) -> None:
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import (
        bucket_plan,
        model_manifest,
        resolve_weight_mapper,
    )

    class _Mapper:
        def __init__(self) -> None:
            self.manifest_calls = 0

        def wire_manifest(self, model: nn.Module) -> list[tuple[str, tuple[int, ...], str]]:
            self.manifest_calls += 1
            return [("wire.weight", tuple(model.weight.shape), str(model.weight.dtype))]

        def export_bucket(
            self,
            model: nn.Module,
            bucket: list[tuple[str, tuple[int, ...], str]],
        ) -> dict[str, torch.Tensor]:
            assert bucket == [("wire.weight", tuple(model.weight.shape), str(model.weight.dtype))]
            return {"wire.weight": model.weight.detach()}  # [O,I]

        def destination_names(self, model: nn.Module) -> tuple[str, ...]:
            del model
            return ("engine.packed_weight",)

    model = nn.Linear(3, 2, bias=False)
    mapper = _Mapper()
    if use_model_mapper:
        model.weight_mapper = mapper
    backend = WeightSync(CopyMover(), mapper=resolve_weight_mapper(model))

    expected_plan = (
        [[("wire.weight", (2, 3), str(model.weight.dtype))]] if use_model_mapper else bucket_plan(model_manifest(model))
    )
    assert backend.ensure_plan(model) == expected_plan
    assert backend.destination_names() == (("engine.packed_weight",) if use_model_mapper else None)
    exported = backend.materialize_bucket(model, 0)
    if use_model_mapper:
        assert set(exported) == {"wire.weight"}
        assert torch.equal(exported["wire.weight"], model.weight)
        assert mapper.manifest_calls == 1
    else:
        assert set(exported).issubset(model.state_dict())
        assert mapper.manifest_calls == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_streaming_version_owned_by_seam_not_sinks():
    # Staleness contract, one owner: a stream is N buckets of ONE logical update carrying ONE
    # monotonic policy version. The bodies' apply_bucket are PURE LOADS (no version semantics —
    # previously the last-bucket stamp was copied in three sinks); pull_stream_from_peer (the seam
    # that knows when the WHOLE stream completed) stamps the consumer once, after the loop.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    src, dst = nn.Linear(8, 8), nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(src), bucket_bytes=64)  # weight and bias in separate buckets
    n = len(plan)
    assert n >= 2

    trainer = LLMTrainer(src, _UNUSED_OBJECTIVE, torch.optim.SGD(src.parameters(), lr=0.1))
    trainer._weight_backend = WeightSync(CopyMover())
    trainer.set_bucket_plan(plan)
    sink = LocalRollout(dst)
    sink._weight_backend = WeightSync(CopyMover())
    sink.set_bucket_plan(plan)

    # stage_bucket must stamp the version it is GIVEN, never the bucket index i.
    assert trainer._weight_backend.stage_bucket(src, n - 1, version=7).version == 7

    # Manual per-bucket applies are pure loads: values land, the SINK's version does NOT move,
    # and the PRODUCER's version does not move either — the version IS the training iteration
    # (training owns advancement; push only reads it).
    trainer._iter = 5
    for i in range(n):
        payload = trainer.push_bucket(i)
        assert payload.version == 5, "push_bucket must stamp the training iteration on every bucket"
        sink.apply_bucket(payload, i)
    assert torch.equal(dst.weight, src.weight)
    assert trainer._iter == 5, "push_bucket must never mutate the version (training owns it)"
    assert sink._weight_version == 0, "apply_bucket must be a pure load (version is the seam's job)"

    # The seam stamps: a full peer-pull sync sets the sink to the PRODUCER'S iteration.
    sink.apply_weights(trainer)
    assert sink._weight_version == 5
    # Re-syncing the same iterate is idempotent (retry-safe by construction — no drift).
    sink.apply_weights(trainer)
    assert sink._weight_version == 5
    trainer._iter = 6  # a training step happened
    sink.apply_weights(trainer)
    assert sink._weight_version == 6


@pytest.mark.L0
@pytest.mark.CPU
def test_push_bucket_version_is_iter_and_stable_across_retry():
    # The version IS the training iteration: push_bucket reads it and never mutates it, so a
    # failed stage retried re-sends the SAME version by construction (no retry drift), and the
    # stamp is what a consumer will compare against for staleness.
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    model = nn.Linear(4, 4)
    trainer = LLMTrainer(model, _UNUSED_OBJECTIVE, torch.optim.SGD(model.parameters(), lr=0.1))
    trainer._iter = 3

    class _RecordingBoomBackend:
        def __init__(self):
            self.versions = []

        def set_bucket_plan(self, plan):
            pass

        def has_plan(self) -> bool:
            return True  # plan present → lazy path not taken

        def bucket_count(self) -> int:
            return 1

        def stage_bucket(self, m, i, version=0):
            self.versions.append(version)
            raise RuntimeError("stage failed (full_tensor / IPC)")

    be = _RecordingBoomBackend()
    trainer._weight_backend = be
    with pytest.raises(RuntimeError):
        trainer.push_bucket(0)
    with pytest.raises(RuntimeError):  # the retry
        trainer.push_bucket(0)
    assert trainer._iter == 3, "push must never move the training iteration"
    assert be.versions == [3, 3], "a retry must re-send the SAME version (the current iterate)"


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_is_pure_load_on_collective_path_too():
    # The bodies' apply_bucket must carry NO version semantics on ANY transport — a manual
    # collective-round loop leaves the sink's version untouched (the stamp — the producer version
    # carried by the returned payload metadata — lives in pull_stream_from_peer, asserted by the
    # fire-before-recv and thread-fallback tests).
    import types

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    class _FakeCollectiveBackend:
        def __init__(self, n: int):
            self._n = n
            self.mover = types.SimpleNamespace(lifetime=COLLECTIVE)

        def bucket_count(self) -> int:
            return self._n

        def apply_bucket(self, model, i, payload):  # collective load stubbed (tests version semantics only)
            pass

    n = 3
    sink = LocalRollout(torch.nn.Linear(4, 4))
    sink._weight_backend = _FakeCollectiveBackend(n)
    for i in range(n):
        sink.apply_bucket(None, i)  # COLLECTIVE round: no payload
        assert sink._weight_version == 0, f"apply_bucket stamped a version at bucket {i} (must be pure load)"


@pytest.mark.L0
@pytest.mark.CPU
def test_streaming_collective_restores_full_recv_plan():
    # COLLECTIVE apply_bucket temporarily narrows mover.recv_plan to the current bucket, then MUST restore
    # the injected full-model plan — else a later batch apply_weights would receive only the
    # last bucket while the producer broadcasts the whole model (wrong load / collective hang).
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)  # weight + bias in separate buckets
    n = len(plan)
    assert n >= 2

    class _FakeCollectiveMover(_MoverDouble):
        lifetime = COLLECTIVE

        def __init__(self):
            self.recv_plan = "FULL"  # sentinel: the injected full-model plan the batch path relies on
            self.during = []

        def fetch(self, handle):
            self.during.append(self.recv_plan)  # what the collective saw for THIS bucket
            return {}  # empty state → import is a no-op

    mover = _FakeCollectiveMover()
    ws = WeightSync(mover)
    ws.set_bucket_plan(plan)
    for i in range(n):
        ws.apply_bucket(model, i)
        assert mover.recv_plan == "FULL", f"recv_plan not restored to the full plan after bucket {i}"
    assert len(mover.during) == n and all(p != "FULL" for p in mover.during)  # each fetch saw its narrowed bucket


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_peer_pull_streams_and_lands_values():
    # The weight-channel role edge: rollout.apply_weights(<peer>) — ONE call, no driver Phase-0, no
    # driver stream loop. On CPU the peer can be the REAL LLMTrainer body (duck-typed: it has
    # get_bucket_plan + push_bucket like the injected PeerClient), so this exercises the real
    # producer: lazy plan (never pre-set anywhere), per-bucket stage, per-sync version.
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    src, dst = nn.Linear(8, 8), nn.Linear(8, 8)
    with torch.no_grad():
        nn.init.zeros_(dst.weight)
        nn.init.zeros_(dst.bias)
    trainer = LLMTrainer(src, _UNUSED_OBJECTIVE, torch.optim.SGD(src.parameters(), lr=0.1))
    trainer._weight_backend = WeightSync(CopyMover())
    sink = LocalRollout(dst)
    sink._weight_backend = WeightSync(CopyMover())
    assert not torch.equal(dst.weight, src.weight)

    sink.apply_weights(trainer)  # the ONE edge call: plan pulled from the peer, buckets pulled+loaded

    assert torch.equal(dst.weight, src.weight) and torch.equal(dst.bias, src.bias)
    # plan negotiated lazily and cached on BOTH sides (policy = single source of truth)
    assert trainer._weight_backend._bucket_plan is not None
    assert sink._weight_backend._bucket_plan == trainer._weight_backend._bucket_plan
    # the sink stamps the PRODUCER'S TRAINING ITERATION after the last bucket (0: no step yet)
    assert sink._weight_version == trainer._iter == 0
    trainer._iter = 1  # a training step happened
    sink.apply_weights(trainer)  # sync #2 → sink tracks the producer's iterate
    assert sink._weight_version == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_peer_pull_collective_fires_before_recv():
    # COLLECTIVE branch of pull_stream_from_peer: per bucket the peer's push MUST be fired
    # asynchronously BEFORE this side enters its recv (both sides inside the broadcast — a
    # synchronous pull would deadlock), and resolved after. Fakes record the interleaving.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)
    n = len(plan)
    assert n >= 2
    log: list = []

    class _FakeCollectiveMover(_MoverDouble):
        lifetime = COLLECTIVE

        def __init__(self):
            self.recv_plan = "FULL"

        def fetch(self, handle):
            log.append(("recv", len(self.recv_plan)))  # narrowed to the current bucket here
            return {}

    class _Fut:
        def __init__(self, i):
            self.i = i

        def result(self):
            log.append(("resolved", self.i))
            # bytes ride the broadcast; the RETURN VALUE is metadata-only, carrying the version
            return WeightPayload(version=9)

    class _FakePeer:
        def get_bucket_plan(self):
            return plan

        def get_weight_destination_names(self) -> None:
            return None

        def call_async(self, method, i):
            assert method == "push_bucket"
            log.append(("fired", i))
            return _Fut(i)

    sink = LocalRollout(model)
    sink._weight_backend = WeightSync(_FakeCollectiveMover())  # real WeightSync over the fake mover
    # plan intentionally NOT set → apply_weights must pull it from the peer (lazy negotiation)

    sink.apply_weights(_FakePeer())

    fired = [x[1] for x in log if x[0] == "fired"]
    assert fired == list(range(n)), f"expected one fire per bucket in order, got {log}"
    for i in range(n):  # per bucket: fired BEFORE recv, resolved AFTER recv
        fi = log.index(("fired", i))
        assert log[fi + 1][0] == "recv", f"bucket {i}: recv did not follow the fire: {log}"
        assert log[fi + 2] == ("resolved", i), f"bucket {i}: fire not resolved after recv: {log}"
    assert sink._weight_version == 9  # stamped from the payload METADATA (the producer's iterate)


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_pull_collective_thread_fallback_for_non_ray_peers():
    # A COLLECTIVE peer WITHOUT call_async (plain body / manual NcclMover setup) must still be
    # co-scheduled — pull_stream_from_peer falls back to a local thread. The fakes form a rendezvous
    # per bucket (push waits for recv to START; recv waits for push to FINISH): a sequential
    # implementation (push before recv, or recv before firing push) deadlocks and times out here,
    # so completing at all proves both sides were inside the "broadcast" together.
    import threading

    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)
    n = len(plan)
    assert n >= 2
    recv_started = [threading.Event() for _ in range(n)]
    push_done = [threading.Event() for _ in range(n)]

    class _FakeCollectiveMover(_MoverDouble):
        lifetime = COLLECTIVE

        def __init__(self):
            self.recv_plan = None
            self._i = 0

        def fetch(self, handle):
            i, self._i = self._i, self._i + 1
            recv_started[i].set()  # let the peer's send proceed
            assert push_done[i].wait(10), f"bucket {i}: peer push never ran concurrently with the recv"
            return {}

    class _NonRayPeer:  # push_bucket only — NO call_async
        def get_bucket_plan(self):
            return plan

        def get_weight_destination_names(self) -> None:
            return None

        def push_bucket(self, i):
            assert recv_started[i].wait(10), f"bucket {i}: recv never entered while push was waiting"
            push_done[i].set()
            return WeightPayload(version=4)  # metadata return: the producer's iterate

    sink = LocalRollout(model)
    sink._weight_backend = WeightSync(_FakeCollectiveMover())
    sink.apply_weights(_NonRayPeer())
    assert all(e.is_set() for e in push_done) and sink._weight_version == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_accepts_asymmetric_shapes_and_drains():
    # Topology validation moved to bind-time pairing (comm/weight_pairing.py). Runtime accepts any
    # (R_p,K_p)×(R_r,K_r) topology. Under the producer-driven flip (design 2026-07-27 §2), weight_sync
    # drives policy.push_weights(rollout) and drains the POLICY's returned handle (the drive role, and
    # thus the drained handle, flipped from the rollout to the policy).
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    class _FakeDrainHandle:
        """Fake handle to track draining."""

        def __init__(self):
            self.drained = False

        def all(self):
            self.drained = True
            return {(0, 0): None}

    class _WG:
        def __init__(self, replicas, ranks):
            self.replicas, self.ranks = replicas, ranks
            self.pushed = False
            self.handle = None

        def push_weights(self, peer):
            self.pushed = True
            self.handle = _FakeDrainHandle()
            return self.handle

    # The loop-level call is shape-agnostic; binder-level actor pairing owns topology validation.
    # Policy execution R stays one while rollout engine count and rank widths vary independently.
    asymmetric_shapes = (
        ((1, 2), (1, 1)),
        ((2, 2), (1, 2)),
        ((1, 1), (1, 2)),
        ((2, 1), (1, 4)),
    )
    for rshape, pshape in asymmetric_shapes:
        rollout, policy = _WG(*rshape), _WG(*pshape)
        weight_sync(rollout, policy)
        assert policy.pushed, f"edge {rshape}→{pshape} was wrongly rejected"
        assert policy.handle is not None and policy.handle.drained, f"handle for {rshape}→{pshape} was not drained"

    class _Body:  # plain bodies carry no shape → skip the check (1:1 by construction)
        def push_weights(self, peer):
            self.pushed = True
            return None  # plain bodies return None (nothing to drain)

    policy_body = _Body()
    weight_sync(object(), policy_body)  # rollout arg is unused by the fake policy body
    assert policy_body.pushed


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_settles_late_cells_before_raising_early_failure() -> None:
    """Weight publication cannot return while a later submitted lane is still mutating."""
    from collections.abc import Iterator

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    settled: list[tuple[int, int]] = []

    class _Handle:
        @staticmethod
        def as_completed() -> Iterator[tuple[tuple[int, int], object]]:
            settled.append((0, 0))
            yield (0, 0), RuntimeError("early push failed")
            settled.append((1, 0))
            yield (1, 0), None

    class _Policy:
        @staticmethod
        def push_weights(_rollout: object) -> _Handle:
            return _Handle()

    with pytest.raises(RuntimeError, match=r"cell \(0, 0\).*early push failed"):
        weight_sync(object(), _Policy())

    assert settled == [(0, 0), (1, 0)]


@pytest.mark.L0
@pytest.mark.CPU
def test_version_prepare_failure_prevents_every_commit() -> None:
    from collections.abc import Iterator

    from cosmos_framework.posttrain.control.loop.weight_sync import confirm_weight_version

    class _PrepareHandle:
        @staticmethod
        def as_completed() -> Iterator[tuple[tuple[int, int], object]]:
            yield (0, 0), None
            yield (0, 1), RuntimeError("lane not ready")

    class _Rollout:
        commits = 0

        @staticmethod
        def prepare_weight_version(_version: int, _identity: str | None) -> _PrepareHandle:
            return _PrepareHandle()

        def commit_weight_version(self, _version: int) -> None:
            self.commits += 1

    class _Policy:
        @staticmethod
        def shared_weight_identity() -> None:
            return None

    rollout = _Rollout()
    with pytest.raises(RuntimeError, match="lane not ready"):
        confirm_weight_version(rollout, _Policy(), 7)
    assert rollout.commits == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_publishes_only_after_policy_push_drains() -> None:
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    events: list[str] = []

    class _PushHandle:
        @staticmethod
        def all() -> dict[tuple[int, int], None]:
            events.append("push-drained")
            return {(0, 0): None}

    class _Policy:
        @staticmethod
        def iteration() -> int:
            return 7

        @staticmethod
        def push_weights(_rollout: object) -> _PushHandle:
            events.append("push-started")
            return _PushHandle()

        @staticmethod
        def shared_weight_identity() -> None:
            return None

    class _Rollout:
        @staticmethod
        def prepare_weight_version(version: int, identity: str | None) -> None:
            assert version == 7 and identity is None
            events.append("prepared")

        @staticmethod
        def commit_weight_version(version: int) -> None:
            assert version == 7
            events.append("committed")

    weight_sync(_Rollout(), _Policy())
    assert events == ["push-started", "push-drained", "prepared", "committed"]


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_rejects_divergent_policy_versions_before_push() -> None:
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    class _VersionHandle:
        @staticmethod
        def all() -> dict[tuple[int, int], int]:
            return {(0, 0): 7, (0, 1): 8}

    class _Policy:
        pushed = False

        @staticmethod
        def iteration() -> _VersionHandle:
            return _VersionHandle()

        def push_weights(self, _rollout: object) -> None:
            self.pushed = True

    policy = _Policy()
    with pytest.raises(RuntimeError, match="inconsistent values"):
        weight_sync(object(), policy)
    assert policy.pushed is False


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_bucket_pluggable_sink_owns_mechanics_once():
    # The (b) seam: WeightSync.apply_bucket is the SINGLE owner of the receive mechanics — a custom
    # sink (vLLM's load_weights stand-in) gets the fetched state while the recv_plan narrow/restore
    # and consumer_done ordering happen in the backend, not re-implemented per engine.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)
    n = len(plan)
    assert n >= 2
    log: list = []

    class _FakeCollectiveMover(_MoverDouble):
        lifetime = COLLECTIVE

        def __init__(self):
            self.recv_plan = "FULL"

        def fetch(self, handle):
            log.append(("fetched-with-narrowed-plan", self.recv_plan != "FULL"))
            return {"k": None}

        def consumer_done(self):
            log.append(("consumer_done",))

    ws = WeightSync(_FakeCollectiveMover())
    ws.set_bucket_plan(plan)
    for i in range(n):
        ws.apply_bucket(model, i, sink=lambda state: log.append(("sink", sorted(state))))
        assert ws.mover.recv_plan == "FULL", f"recv_plan not restored after bucket {i}"
    # per bucket, in order: fetch saw the NARROWED plan → sink got the state → consumer_done ran
    per_bucket = [log[j : j + 3] for j in range(0, len(log), 3)]
    assert len(per_bucket) == n
    for chunk in per_bucket:
        assert chunk[0] == ("fetched-with-narrowed-plan", True)
        assert chunk[1] == ("sink", ["k"])
        assert chunk[2] == ("consumer_done",)


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_pull_strict_coverage_fails_fast_on_key_divergence():
    # Streaming loads each bucket strict=False by necessity, so the STRICT contract moved to plan-
    # pull time: a load_state_dict sink whose keys diverge from the plan (wrong checkpoint / resized /
    # adapter keys) must fail BEFORE any tensor moves — no half-mutated model, no version stamp
    # (the old batch path only failed at strict load time, after fetching everything).
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    src = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(src), bucket_bytes=64)
    pulled: list = []

    class _Peer:
        def get_bucket_plan(self):
            return plan

        def get_weight_destination_names(self) -> None:
            return None

        def push_bucket(self, i):
            pulled.append(i)

    sink = LocalRollout(nn.Sequential(nn.Linear(8, 8)))  # keys "0.weight"/"0.bias" ≠ plan's "weight"/"bias"
    sink._weight_backend = WeightSync(CopyMover())
    with pytest.raises(ValueError, match="strict"):
        sink.apply_weights(_Peer())
    assert pulled == [], "buckets were pulled despite the failed strict coverage check"
    assert sink._weight_version == 0, "version stamped despite the failed sync"


@pytest.mark.L0
@pytest.mark.CPU
def test_collective_apply_without_recv_plan_on_engine_sink_raises_clearly():
    # A vLLM-style sink (model with NO state_dict) can't self-derive the collective recv layout; if
    # recv_plan wasn't injected (init_weight_plane not called / no recv_manifest), apply must raise a
    # clear ValueError naming the fix — not the cryptic AttributeError('LLM' has no 'state_dict').
    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE

    class _CollectiveMover(_MoverDouble):
        lifetime = COLLECTIVE
        recv_plan = None

        def fetch(self, handle):  # must never be reached without a plan
            raise AssertionError("fetch called without a recv_plan")

    class _EngineLike:  # vllm.LLM stand-in: no state_dict attribute
        pass

    ws = WeightSync(_CollectiveMover())
    with pytest.raises(ValueError, match="recv_plan.*recv_manifest|recv_manifest"):
        ws.apply(_EngineLike())
    # an nn.Module sink still self-derives (the pre-existing HF path)
    import torch.nn as nn

    class _CollectiveMoverOK(_CollectiveMover):
        def fetch(self, handle):  # a full valid state, as the real broadcast would deliver
            return {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}

    ws2 = WeightSync(_CollectiveMoverOK())
    ws2.apply(nn.Linear(2, 2))
    assert ws2.mover.recv_plan and ws2.mover.recv_plan[0][0] == "bias"  # derived, name-sorted


@pytest.mark.L0
@pytest.mark.CPU
def test_collective_preflight_probe_fails_before_entering_collective():
    # A backendless producer's push_bucket returns None WITHOUT broadcasting — on real NCCL the
    # consumer would hang in the bucket-0 recv until the collective timeout, so the post-loop None
    # check can never fire for this case. The COLLECTIVE branch must probe the peer on the CONTROL
    # plane first: even with the local plan CACHED (so the lazy pull is skipped), get_bucket_plan is
    # re-resolved and the producer's "no weight backend bound" raise surfaces immediately — the
    # local mover's fetch (the collective recv) is never entered.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)

    class _NeverFetchMover(_MoverDouble):
        lifetime = COLLECTIVE
        recv_plan = None

        def fetch(self, handle):
            raise AssertionError("entered the collective recv despite a backendless peer")

    class _BackendlessPeer:  # what LLMTrainer does with no weight backend bound
        def get_bucket_plan(self):
            raise ValueError("no weight backend bound (bind with weight_sync=True before get_bucket_plan)")

        def get_weight_destination_names(self) -> None:
            return None

        def push_bucket(self, i):
            return None

    sink = LocalRollout(model)
    sink._weight_backend = WeightSync(_NeverFetchMover())
    sink._weight_backend.set_bucket_plan(plan)  # plan cached ⇒ the lazy pull alone would NOT probe
    with pytest.raises(ValueError, match="no weight backend"):
        sink.apply_weights(_BackendlessPeer())
    assert sink._weight_version == 0, "version stamped despite the failed pre-flight probe"


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_pull_none_payload_raises_no_fresh_stamp():
    # Fail-quiet edge killed: a peer whose push_bucket returns None (policy with no weight backend
    # bound = wiring bug) must RAISE — the old fallback stamped a fresh version over a model that
    # received zero bytes, advertising weights that were never synced.
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan, model_manifest
    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    model = nn.Linear(8, 8)
    plan = bucket_plan(model_manifest(model), bucket_bytes=64)

    class _BackendlessPeer:
        def get_bucket_plan(self):
            return plan

        def get_weight_destination_names(self) -> None:
            return None

        def push_bucket(self, i):
            return None  # what LLMTrainer.push_bucket returns with no backend bound

    sink = LocalRollout(model)
    sink._weight_backend = WeightSync(CopyMover())
    with pytest.raises(ValueError, match="no weight backend"):
        sink.apply_weights(_BackendlessPeer())
    assert sink._weight_version == 0, "version stamped although nothing was transferred"


class _ExplodingPeer:
    """Any peer RPC is a bug on the degenerate edges (no backend / SharedMover): a real PeerClient
    would SELF-call the very actor executing apply_weights and deadlock (codex: same-actor RPC)."""

    def __getattr__(self, name):
        raise AssertionError(f"peer RPC {name!r} issued on a degenerate edge (would self-deadlock)")


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_pull_degenerate_edges_stamp_without_peer_rpc():
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout

    # No backend (v0 static/shared): peer form keeps the payload-form contract — stamp only.
    sink = LocalRollout(nn.Linear(4, 4))
    sink.apply_weights(_ExplodingPeer())
    assert sink._weight_version == 1

    # SharedMover (single colocation: shared instance AND shared actor): stamp only, no self-RPC.
    shared = LocalRollout(nn.Linear(4, 4))
    shared._weight_backend = WeightSync(SharedMover())
    shared.apply_weights(_ExplodingPeer())
    assert shared._weight_version == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_pull_unwraps_local_backend_handles():
    # backend="local" passes the raw WorkerGroup as the peer, so peer calls return Handle-like
    # objects — pull_stream_from_peer must unwrap .result() for BOTH the plan and each payload
    # (codex: caching a Handle breaks bucket_count / apply_bucket).
    import torch.nn as nn

    from cosmos_framework.posttrain.roles.rollout.local import LocalRollout
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    class _H:  # minimal local-backend Handle: value behind .result()
        def __init__(self, v):
            self._v = v

        def result(self):
            return self._v

    src, dst = nn.Linear(8, 8), nn.Linear(8, 8)
    with torch.no_grad():
        nn.init.zeros_(dst.weight)
        nn.init.zeros_(dst.bias)
    trainer = LLMTrainer(src, _UNUSED_OBJECTIVE, torch.optim.SGD(src.parameters(), lr=0.1))
    trainer._weight_backend = WeightSync(CopyMover())
    trainer._iter = 4  # the version the payloads must carry

    class _LocalPeer:  # wraps every call result in a Handle, like WorkerGroup._submit on local
        def get_bucket_plan(self):
            return _H(trainer.get_bucket_plan())

        def get_weight_destination_names(self) -> _H:
            return _H(trainer.get_weight_destination_names())

        def push_bucket(self, i):
            return _H(trainer.push_bucket(i))

    sink = LocalRollout(dst)
    sink._weight_backend = WeightSync(CopyMover())
    sink.apply_weights(_LocalPeer())
    assert torch.equal(dst.weight, src.weight) and torch.equal(dst.bias, src.bias)
    assert sink._weight_version == 4  # producer's iterate stamped from the UNWRAPPED payload


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_sync_bucket_bytes_defaults_and_overrides():
    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import BUCKET_BYTES, WeightSync

    assert WeightSync(CopyMover()).bucket_bytes == BUCKET_BYTES
    assert WeightSync(CopyMover(), bucket_bytes=1 << 20).bucket_bytes == 1 << 20


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_bytes_actually_changes_the_plan():
    # A smaller cap must produce MORE buckets, or the knob is decorative.
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan

    manifest = [(f"p{i:02d}", (256, 256), "torch.float32") for i in range(8)]  # 256 KiB each
    assert len(bucket_plan(manifest, 1 << 20)) > 1
    assert len(bucket_plan(manifest, 1 << 30)) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_staging_buffer_sizing_covers_every_bucket_of_its_own_plan():
    """The NIXL staging buffer is sized ONCE at bind and cannot grow (registered + pinned), so
    max() over the plan must cover every bucket -- including a single parameter LARGER than the
    cap, which bucket_plan lets form its own oversized bucket. If sizing used the cap constant
    instead of the plan, that bucket would fail mid-sync with 'staging buffer too small'."""
    from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest
    from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan

    cap = 1 << 20
    manifest = [("big", (1024, 1024), "torch.float32")] + [  # 4 MiB: 4x the cap, own bucket
        (f"p{i:02d}", (128, 128), "torch.float32") for i in range(6)
    ]
    manifest.sort()
    plan = bucket_plan(manifest, cap)
    need = max(flat_nbytes_for_manifest(b) for b in plan)
    assert need > cap, "fixture must contain an oversized bucket to be meaningful"

    # Assert the PROPERTY, not `x <= max(x)`: a buffer sized from the plan packs the oversized
    # bucket, one sized from the cap does not. The previous loop compared each bucket against a
    # bound derived from those same buckets, so it could not fail for any input.
    from cosmos_framework.posttrain.comm.transfer.flat import flat_plan, pack

    big = max(plan, key=flat_nbytes_for_manifest)
    state = {n: torch.zeros(sh, dtype=getattr(torch, d.split(".")[-1])) for n, sh, d in big}
    fp = flat_plan(state)
    pack(state, fp, torch.empty(need, dtype=torch.uint8))  # plan-sized: fits
    with pytest.raises(ValueError, match="too small"):
        pack(state, fp, torch.empty(cap, dtype=torch.uint8))  # cap-sized: does not


@pytest.mark.L0
@pytest.mark.CPU
def test_weightsync_terminal_close_uses_the_explicit_mover_contract():
    class TrackingMover(CopyMover):
        def __init__(self):
            self.open = True
            self.released = 0

        def release_staged(self):
            self.released += 1

        def close(self):
            self.open = False

    mover = TrackingMover()
    backend = WeightSync(mover)
    backend.release_staged()
    assert mover.open and mover.released == 1
    backend.close()
    backend.close()
    assert not mover.open and mover.released == 1
