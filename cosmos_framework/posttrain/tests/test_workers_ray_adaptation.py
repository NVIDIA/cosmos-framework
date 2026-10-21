# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actor-side dispatch adaptation in `_Host.invoke` (Task A5, DESIGN §2.2/§2.3/§2.4).

`_resolve_dispatched` (workers/ray.py, module-level) is the piece that actually EXECUTES a
`Dispatched` descriptor (built driver-side by `GridBackend._align_one`, Task A4): resolve its
own producer refs, materialize bulk with `readers` threaded through for refcounted pop, then
apply the reduce/split/one adaptation so the body signature never sees a list/shard.

Helper-level tests below exercise `_resolve_dispatched` directly with a fake in-process
transport backed by a REAL `BulkStore` — no Ray cluster needed (mirrors test_comm_codec.py's
pure-CPU style). A `_Host.invoke` end-to-end test (real local `ray.init`, same pattern as
test_workers_ray.py) closes the loop: the actor recognizes a `Dispatched` arg and resolves it
without a change to non-Dispatched behavior.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.codec import pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.workers.dispatch import select_chunk, select_model_dp

pytest.importorskip("ray")  # workers.ray imports `ray` at module load (below); the CI image has no ray → skip cleanly

from cosmos_framework.posttrain.workers.data_pool import PoolSlice  # noqa: E402
from cosmos_framework.posttrain.workers.ray import RayStoreMover, _HostImpl, _resolve_dispatched  # noqa: E402
from cosmos_framework.posttrain.workers.worker_group import Dispatched  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

# "module:attr" reference to the real select_chunk (workers/dispatch.py), resolved via
# resolve_fn — derived from the import (not a hand-typed string) so a rename can't silently break it.
_SELECT_CHUNK_REF = f"{select_chunk.__module__}:{select_chunk.__name__}"
_SELECT_MODEL_DP_REF = f"{select_model_dp.__module__}:{select_model_dp.__name__}"


def test_same_actor_pool_views_and_invocation_cleanup() -> None:
    producer = SimpleNamespace(owner_id="local-owner", tensor=torch.arange(4, dtype=torch.uint8))
    producer.view_tensor = MagicMock(side_effect=lambda _slice: producer.tensor)
    consumer = SimpleNamespace(warmup=MagicMock(), fetch_tensor=MagicMock())
    mover = RayStoreMover({}, consumer)
    mover.set_local_producer_pool(producer)
    slice_ = PoolSlice("local-owner", "step-1", "cuda", 0, 1, 0, 4, (4,), "uint8", 32)

    tensor = mover.fetch_pool_slice(slice_)
    assert tensor is producer.tensor
    assert mover.source_for_tensor(tensor) is slice_
    producer.view_tensor.assert_called_once_with(slice_)
    consumer.warmup.assert_not_called()
    consumer.fetch_tensor.assert_not_called()

    tensor_ref = weakref.ref(tensor)
    producer.tensor = None
    del tensor
    gc.collect()
    assert tensor_ref() is None
    mover.clear_pool_tensor_sources()
    producer.tensor = torch.arange(4, dtype=torch.uint8)

    host = object.__new__(_HostImpl)
    host._pending_releases = []
    host._consumer_pool = consumer
    host._transport = mover
    host._invoke_impl = lambda *_args: "ok"

    tensor = mover.fetch_pool_slice(slice_)
    assert mover.source_for_tensor(tensor) is slice_
    assert host.invoke("role", "method", (), {}) == "ok"
    assert mover.source_for_tensor(tensor) is None

    def fail(*_args: object) -> None:
        raise RuntimeError("failed role")

    host._invoke_impl = fail
    tensor = mover.fetch_pool_slice(slice_)
    with pytest.raises(RuntimeError, match="failed role"):
        host.invoke("role", "method", (), {})
    assert mover.source_for_tensor(tensor) is None


@dataclass
class _Blob:
    """Stands in for a tensor: has an int `nbytes` (what the codec keys off), mirrors the
    stand-in used by test_comm_codec.py (not imported from there — tests never import tests)."""

    nbytes: int
    tag: str = "x"


def _traj(action) -> Trajectory:
    tr = Transition(obs=ObsBundle(representations={"prompt": "1+2="}), action=action, done=True)
    return Trajectory(transitions=[tr], id="g0_k0", group_id="g0", status=Status.COMPLETED)


def concat_lists(payloads: list) -> list:
    """Test reduce fn (resolved via `"module:attr"` by `resolve_fn`): flatten producer
    payloads into one list — the actor-side stand-in for a real Sharded(reduce=...) fn."""
    return [x for p in payloads for x in p]


def split_shard(payload: list, index: int, n: int) -> list:
    """Test split fn: contiguous shard `index` of `n` from one shared producer payload —
    the actor-side stand-in for a real Sharded(split=...) fn."""
    size = len(payload) // n
    return payload[index * size : (index + 1) * size]


# ── helper-level: _resolve_dispatched, no Ray ────────────────────────────────────────────


def test_op_one_passthrough_identical_to_raw():
    """op="one" (Mirror / bare single-reader wrap): the single payload comes back untouched,
    same as if the raw ref/value had been forwarded directly (no adaptation applied)."""
    d = Dispatched(refs=["raw_payload"], op="one", fn=None, index=0, n=1, readers=1)
    assert _resolve_dispatched(d, transport=None, materialize=False) == "raw_payload"


def test_op_one_rejects_multiple_payloads():
    # ValueError (not bare assert) so the routing-invariant guard survives `python -O`.
    d = Dispatched(refs=["a", "b"], op="one", fn=None, index=0, n=1, readers=1)
    with pytest.raises(ValueError, match="expects exactly 1 payload"):
        _resolve_dispatched(d, transport=None, materialize=False)


def test_rank_partitioned_reduce_selects_one_model_dp_shard_after_reassembly() -> None:
    reduce_ref = f"{__name__}:concat_lists"
    dispatched = Dispatched(
        refs=[[0, 1], [2, 3]],
        op="reduce",
        fn=reduce_ref,
        index=0,
        n=2,
        readers=1,
        select=_SELECT_MODEL_DP_REF,
        select_kw={},
    )

    result = _resolve_dispatched(dispatched, transport=None, materialize=True, dp_rank=1, dp_world=2)

    assert result == [2, 3]


def test_reduce_calls_fn_with_the_payload_list():
    """op="reduce": the resolved fn is called with the FULL list of producer payloads, not
    a single element — the trainer body sees one already-concatenated batch."""
    d = Dispatched(
        refs=[[1, 2], [3, 4]],
        op="reduce",
        fn=f"{__name__}:concat_lists",
        index=0,
        n=2,
        readers=1,
    )
    out = _resolve_dispatched(d, transport=None, materialize=False)
    assert out == [1, 2, 3, 4]


def test_split_hands_index_and_n_each_replica_gets_its_own_shard():
    """op="split": two simulated consumer replicas share ONE producer payload; each
    Dispatched carries its OWN index, so each replica's call gets a distinct, non-overlapping
    shard (fn receives (payload, index, n), never a pre-sliced value)."""
    shared_payload = [10, 20, 30, 40]
    d0 = Dispatched(refs=[shared_payload], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=1)
    d1 = Dispatched(refs=[shared_payload], op="split", fn=f"{__name__}:split_shard", index=1, n=2, readers=1)

    shard0 = _resolve_dispatched(d0, transport=None, materialize=False)
    shard1 = _resolve_dispatched(d1, transport=None, materialize=False)

    assert shard0 == [10, 20]
    assert shard1 == [30, 40]
    assert shard0 != shard1
    assert shard0 + shard1 == shared_payload  # together cover the whole producer payload


def test_unknown_op_fails_loud():
    d = Dispatched(refs=["x"], op="scatter", fn=None, index=0, n=1, readers=1)
    with pytest.raises(ValueError, match="unknown Dispatched op"):
        _resolve_dispatched(d, transport=None, materialize=False)


def test_readers_threaded_store_pops_only_after_the_4th_consumer_fetch():
    """A single producer unit fed to 4 consumer cells (readers=4, e.g. a rank-mirror or
    Mirror fan-out edge): each simulated consumer resolves the SAME Dispatched (materialize
    True) — the refcounted `BulkStore` entry must survive the first 3 fetches and pop only
    on the 4th (DESIGN §2.4), proving `readers` actually reaches `BulkStore.fetch` through
    `materialize_payload` -> transport.fetch."""
    store = BulkStore()
    action = _Blob(2000)  # above threshold=0 -> externalized to a BulkRef
    packed = pack_payload(_traj(action), store, threshold=0)
    assert len(store) == 1  # one bulk entry resident

    transport = LocalStoreMover()
    d = Dispatched(refs=[packed], op="one", fn=None, index=0, n=1, readers=4)

    for i in range(3):  # 1st..3rd simulated consumer: refcount decrements, entry survives
        out = _resolve_dispatched(d, transport, materialize=True)
        assert out.transitions[0].action is action  # zero-copy, same process
        assert len(store) == 1, f"store popped early after consumer #{i + 1}"

    out = _resolve_dispatched(d, transport, materialize=True)  # 4th (last) simulated consumer
    assert out.transitions[0].action is action
    assert len(store) == 0  # refcount hit zero -> popped


def test_materialize_false_leaves_bulkref_unresolved():
    """When the consuming WG doesn't materialize (name not in self._materialize), a
    Dispatched op="one" payload passes through with its BulkRef fields untouched — the same
    non-materializing behavior as a plain (non-Dispatched) arg."""
    store = BulkStore()
    packed = pack_payload(_traj(_Blob(2000)), store, threshold=0)
    d = Dispatched(refs=[packed], op="one", fn=None, index=0, n=1, readers=1)

    out = _resolve_dispatched(d, transport=LocalStoreMover(), materialize=False)
    assert isinstance(out.transitions[0].action, BulkRef)
    assert len(store) == 1  # untouched


def test_split_on_metadata_false_with_bulk_is_still_post_materialize_byte_identical():
    """Guards the new on_metadata branch's gate: with `on_metadata=False` (the dataclass
    default, and the only value that existed before Task A6), a split payload carrying real
    BulkRefs still takes the OLD path — materialize the whole producer payload first, THEN
    apply the split fn to already-resolved values, exactly as Task A5 shipped it. Every
    BulkRef in this payload is fetched -- the OLD path materializes the WHOLE producer payload
    (all 4 samples) before splitting, so with readers=1 all 4 entries pop, not just the 2 the
    split fn eventually keeps. That "wasted" fetch of the dropped half is exactly the traffic
    the on_metadata fast path (next test) exists to avoid."""
    store = BulkStore()
    samples = [_traj(_Blob(2000, tag=f"s{i}")) for i in range(4)]
    packed = [pack_payload(s, store, threshold=0) for s in samples]
    assert len(store) == 4

    transport = LocalStoreMover()
    d = Dispatched(refs=[packed], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=1)

    out = _resolve_dispatched(d, transport, materialize=True)
    assert [t.transitions[0].action.tag for t in out] == ["s0", "s1"]  # materialized, split after
    assert len(store) == 0  # ALL 4 samples fetched (readers=1) before the split ever ran


def test_on_metadata_split_fetches_only_kept_refs_and_releases_the_rest():
    """DESIGN §2.3 opt-in fast path + §2.4 release protocol (Task A6): a list-of-samples
    payload (each sample its own Trajectory carrying its own BulkRef `action`) is split on the
    PACKED metadata skeleton BEFORE materialization. Two simulated replicas (n=2) each keep a
    contiguous half and, per grid.py's split readers formula (readers = Kc·n, Kc=1 here),
    readers=2: every sample-ref is touched by exactly 2 events total across the two replicas
    -- the ONE replica that keeps it fetches (materializes) it, the ONE replica that drops it
    releases it -- so the store only reaches len()==0 once BOTH replicas have run, never after
    just one."""
    store = BulkStore()
    samples = [_traj(_Blob(2000, tag=f"s{i}")) for i in range(4)]
    packed = [pack_payload(s, store, threshold=0) for s in samples]
    assert len(store) == 4  # one bulk entry per sample, all resident
    fetch_count0 = store.fetch_count

    transport = LocalStoreMover()
    d0 = Dispatched(refs=[packed], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=2, on_metadata=True)
    d1 = Dispatched(refs=[packed], op="split", fn=f"{__name__}:split_shard", index=1, n=2, readers=2, on_metadata=True)

    released: list[BulkRef] = []

    def releaser(refs: list, readers: int) -> None:
        released.extend(refs)
        for r in refs:
            store.release(r, readers=readers)

    out0 = _resolve_dispatched(d0, transport, materialize=True, releaser=releaser)
    assert [t.transitions[0].action.tag for t in out0] == ["s0", "s1"]  # replica 0's kept half
    assert len(released) == 2  # replica 0 dropped samples 2,3 -> released, never fetched
    assert store.fetch_count == fetch_count0 + 2  # only the 2 KEPT refs were fetched so far
    assert len(store) == 4  # every entry's first touch (fetch or release) just decrements

    out1 = _resolve_dispatched(d1, transport, materialize=True, releaser=releaser)
    assert [t.transitions[0].action.tag for t in out1] == ["s2", "s3"]  # replica 1's kept half
    assert len(released) == 4  # replica 1 dropped samples 0,1 -> released, never (re)fetched
    assert store.fetch_count == fetch_count0 + 4  # exactly 4 fetches total -- one per sample, never twice
    assert len(store) == 0  # both replicas ran -> every entry's 2nd touch popped it


def test_on_metadata_split_dropped_ref_with_no_releaser_raises():
    """A drop with nowhere to send the release is a caller bug (would silently leak the
    producer-side entry forever) -- fail loud, matching this codebase's release() posture
    (comm/bulk.py: KeyError on double-release/never-put) rather than swallow it."""
    store = BulkStore()
    samples = [_traj(_Blob(2000, tag=f"s{i}")) for i in range(2)]
    packed = [pack_payload(s, store, threshold=0) for s in samples]
    d = Dispatched(refs=[packed], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=2, on_metadata=True)

    with pytest.raises(ValueError, match="no releaser"):
        _resolve_dispatched(d, LocalStoreMover(), materialize=True)


def test_on_metadata_split_without_materialize_falls_through_unresolved():
    """on_metadata=True is inert when this call doesn't materialize (name not in
    self._materialize) -- the split still applies (consistent with the default op="split"
    contract: the fn always runs), but no bulk is touched and nothing is released, matching
    the non-materializing behavior of every other op."""
    store = BulkStore()
    samples = [_traj(_Blob(2000, tag=f"s{i}")) for i in range(4)]
    packed = [pack_payload(s, store, threshold=0) for s in samples]
    d = Dispatched(refs=[packed], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=2, on_metadata=True)

    out = _resolve_dispatched(d, transport=LocalStoreMover(), materialize=False)
    assert len(out) == 2  # split still ran (contiguous half of 4 samples)
    assert all(isinstance(t.transitions[0].action, BulkRef) for t in out)  # never fetched
    assert len(store) == 4  # nothing fetched, nothing released -- fully untouched


# ── ingress selection (Task 8, DESIGN §3.4/§9.5): select on metadata, materialize the kept share ──


class _CountingMover:
    """Wraps a real transport and counts only ACTUAL `fetch()` calls — the direct proof that the
    unselected share of the bulk plane never moves (`test_unselected_bulk_is_never_fetched`).
    Delegates every fetch to `inner` (a real `LocalStoreMover`), so behavior is otherwise identical."""

    def __init__(self, inner: LocalStoreMover, items: list):
        self._inner = inner
        self.items = items  # the full packed skeleton, for tests to build Dispatched refs from
        self.fetched_count = 0

    def fetch(self, ref, readers: int = 1):
        self.fetched_count += 1
        return self._inner.fetch(ref, readers=readers)


def _fake_bulk_plane(n_items: int) -> tuple[BulkStore, _CountingMover]:
    """A real `BulkStore` holding `n_items` distinct bulk-bearing samples (one `BulkRef` action
    each, mirroring `_traj`/`_Blob` above), plus a `_CountingMover` transport over it. All samples
    share the SAME `id`/`group_id` (like `_traj`) — `select_chunk` sorts on those, and a stable
    sort over equal keys preserves this list's construction order, so the chunk math below is
    deterministic without needing distinct ids."""
    store = BulkStore()
    items = [pack_payload(_traj(_Blob(2000, tag=f"s{i}")), store, threshold=0) for i in range(n_items)]
    return store, _CountingMover(LocalStoreMover(), items)


def _dispatched(items: list, *, select_kw: dict, readers: int = 1) -> Dispatched:
    """A `Dispatched` wrapping the full packed `items` skeleton as a single `op="one"` producer
    unit, carrying `select_chunk` (Task 6) parameterised by `select_kw` — the shape
    `GridBackend._dispatch_from_spec` builds for the `("trainer", "train", "batch")` edge once a
    call passes its control kwargs (Task 7), minus the driver-side plumbing this test bypasses."""
    return Dispatched(
        refs=[items], op="one", fn=None, index=0, n=1, readers=readers, select=_SELECT_CHUNK_REF, select_kw=select_kw
    )


def test_unselected_bulk_is_never_fetched():
    """The whole point (§3.4): bulk outside this rank's mini/micro/dp chunk must never move.
    64 items -> micro (n_micro=4) keeps 16 -> dp (dp_world=2) keeps 8: only those 8 BulkRefs may
    ever reach `transport.fetch`."""
    store, transport = _fake_bulk_plane(n_items=64)
    d = _dispatched(transport.items, select_kw={"mini": 0, "micro": 1, "n_mini": 1, "n_micro": 4})

    out = _resolve_dispatched(d, transport, materialize=True, dp_rank=0, dp_world=2)

    assert len(out) == 8
    assert transport.fetched_count == 8  # 64 / (4 micro * 2 dp), NOT 64
    assert len(store) == 64 - 8  # the other 56 entries are untouched (not fetched, not released)


def test_unselected_bulk_is_not_released():
    """Items for LATER micro calls, and for OTHER dp ranks, must still be fetchable afterwards —
    the ingress-selection branch NEVER releases (unlike the on_metadata split path just above,
    which releases every BulkRef its split drops). Micro=0's call must not touch micro=3's share."""
    store, transport = _fake_bulk_plane(n_items=64)
    first = _dispatched(transport.items, select_kw={"mini": 0, "micro": 0, "n_mini": 1, "n_micro": 4})
    _resolve_dispatched(first, transport, materialize=True, dp_rank=0, dp_world=2)

    later = _dispatched(transport.items, select_kw={"mini": 0, "micro": 3, "n_mini": 1, "n_micro": 4})
    got = _resolve_dispatched(later, transport, materialize=True, dp_rank=0, dp_world=2)

    assert len(got) == 8  # still there — a later micro call's share was never released by the first
    assert all(t.transitions[0].action.tag == f"s{48 + 2 * i}" for i, t in enumerate(got))


@pytest.mark.parametrize("k_c,dp_world,expected_readers", [(4, 4, 1), (8, 4, 2), (2, 2, 1)])
def test_reader_refcount_pops_each_entry_exactly_once(k_c, dp_world, expected_readers):
    """readers = K_c / dp_world (§9.5). The pure-DP case (4, 4, 1) and (2, 2, 1) — where
    K_c == dp_world — PASSES even with the WRONG formula (readers=1 either way); (8, 4, 2) is the
    one that actually matters: with TP inside the replica (K_c=8 ranks sharing dp_world=4 DP
    groups, 2 TP-mirror ranks per group), each of the dp_world groups' kept items must be
    materialized exactly `expected_readers` times before popping — too few and a later mirror
    rank's fetch would fault on an already-popped key; too many and the store never empties."""
    store, transport = _fake_bulk_plane(n_items=32)
    select_kw = {"mini": 0, "micro": 0, "n_mini": 1, "n_micro": 1}  # mini/micro inert: dp is the only axis

    for rank in range(k_c):
        d = _dispatched(transport.items, select_kw=select_kw, readers=expected_readers)
        _resolve_dispatched(d, transport, materialize=True, dp_rank=rank % dp_world, dp_world=dp_world)

    assert len(store) == 0  # every dp group's kept items popped exactly once its readers were spent


# ── invoke-level: a real local `_Host` actor (no external Ray cluster) ──────────────────

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.workers.ray import _Host  # noqa: E402


def _adaptation_setup():
    class Trainer:
        def train(self, batch):
            return batch  # body sees the ALREADY-ADAPTED value, never a list/shard

    return {"trainer": Trainer()}


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_resolves_dispatched_reduce_arg_end_to_end():
    """`_Host.invoke` (real Ray actor, local single-process cluster — same pattern as
    test_workers_ray.py) recognizes a `Dispatched` arg nested in the call and resolves it:
    real `ray.ObjectRef`s in `d.refs` get `ray.get`'d actor-side, then `reduce` is applied —
    the body's `train(batch)` receives the single concatenated batch, not a Dispatched or a
    list of raw refs."""
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_adaptation_setup))  # codec=False: no materialize seam

        ref1 = ray.put([1, 2])
        ref2 = ray.put([3, 4])
        d = Dispatched(refs=[ref1, ref2], op="reduce", fn=f"{__name__}:concat_lists", index=0, n=2, readers=1)

        result = ray.get(host.invoke.remote("trainer", "train", (d,), {}))
        assert result == [1, 2, 3, 4]
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_non_dispatched_arg_unaffected():
    """A plain (non-Dispatched) arg through `_Host.invoke` is completely unaffected by the
    Dispatched-recognition branch — byte-identical to before Task A5."""
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_adaptation_setup))
        result = ray.get(host.invoke.remote("trainer", "train", ([9, 9, 9],), {}))
        assert result == [9, 9, 9]
    finally:
        ray.shutdown()


def _materialize_invoke_setup():
    class Producer:
        def produce(self):
            return _traj(_Blob(2000))  # above threshold=0 -> externalized to a BulkRef

    class Consumer:
        def receive(self, x):
            # Body sees the ALREADY-MATERIALIZED value, never a BulkRef — report on it rather
            # than returning the Trajectory itself: `_Host.invoke`'s (unrelated) producer-side
            # `pack_payload` egress step runs on EVERY return value when `self._codec` is set,
            # so a returned Trajectory would get re-externalized on the way out and defeat the
            # assertion. A plain tuple passes `pack_payload` through untouched.
            action = x.transitions[0].action
            return (isinstance(action, BulkRef), getattr(action, "nbytes", None))

    return {"producer": Producer(), "consumer": Consumer()}


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_materializes_dispatched_and_refcount_pops_at_zero():
    """Closes the reviewer-flagged gap: the two tests above run with codec=False, so
    `_Host.invoke`'s real materialize gate (`self._codec and name in self._materialize`) and
    its `self._transport` were never exercised for a Dispatched arg. Here `run_setup` enables
    codec + puts "consumer" in materialize_names, so `invoke` computes `materialize=True` and
    calls `_resolve_dispatched(d, self._transport, True)` for real.

    One actor plays both producer and (twice) consumer: `producer.produce()` returns a
    Trajectory whose big `action` field `_Host.invoke`'s producer-side `pack_payload` (unrelated
    to Dispatched — the existing egress path) externalizes into the actor's OWN `BulkStore`.
    That packed skeleton becomes `Dispatched(refs=[...], readers=2)` and is carried through TWO
    separate `invoke` calls simulating two consumer cells reading the same producer unit — the
    refcounted entry must survive the first and pop only on the second (DESIGN §2.4), proven via
    the actor-introspection `run` test seam (`len(actor._store)`), not the helper-level bypass."""
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(
            host.run_setup.remote(_materialize_invoke_setup, True, 0, ("consumer",))
        )  # codec, threshold, materialize_names

        def store_len() -> int:
            return ray.get(host.run.remote(lambda h: len(h._store), host))

        packed = ray.get(host.invoke.remote("producer", "produce", (), {}))
        assert store_len() == 1  # one bulk entry resident (the producer's own pack_payload egress)

        d = Dispatched(refs=[packed], op="one", fn=None, index=0, n=1, readers=2)

        out1 = ray.get(host.invoke.remote("consumer", "receive", (d,), {}))  # 1st simulated consumer
        assert out1 == (False, 2000)  # (is_bulk_ref, nbytes): materialized, not a BulkRef
        assert store_len() == 1  # refcount 2 -> 1: entry survives

        out2 = ray.get(host.invoke.remote("consumer", "receive", (d,), {}))  # 2nd simulated consumer
        assert out2 == (False, 2000)
        assert store_len() == 0  # refcount 1 -> 0: popped through the REAL invoke gate
    finally:
        ray.shutdown()


def _on_metadata_split_invoke_setup():
    class Producer:
        def produce(self):
            return [_traj(_Blob(2000, tag=f"s{i}")) for i in range(4)]  # list-of-samples payload

    class Consumer:
        def receive(self, batch):
            # Report tags rather than returning the batch: the batch is a plain list (not a
            # Trajectory/ObsBundle _Host.invoke's producer-side `pack_payload` egress
            # recognizes), so it passes through untouched either way -- tags are just easier
            # to assert on.
            return [t.transitions[0].action.tag for t in batch]

    return {"producer": Producer(), "consumer": Consumer()}


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_on_metadata_split_pulls_only_kept_bulk_via_the_release_seam():
    """Closes the loop for Task A6's release-protocol seam (`_Host.invoke`'s `_release_dropped`
    closure over `self._store`/`self._transport`) with a real local `_Host` actor -- mirrors
    A5's `test_invoke_materializes_dispatched_and_refcount_pops_at_zero` pattern.

    One actor plays producer AND (as two simulated replicas) consumer. `producer.produce()`
    returns a list of 4 samples; `_Host.invoke`'s existing egress `pack_payload` externalizes
    each sample's big `action` into the actor's OWN `BulkStore` (4 entries). The SAME packed
    list becomes both replicas' `Dispatched(op="split", on_metadata=True, readers=2)` -- readers
    = Kc·n = 1·2 (grid.py's split formula for a 2-replica, single-rank sim). Replica 0 fetches
    samples 0,1 and releases (never fetches) 2,3 through `_release_dropped`'s same-actor branch
    (`has_local_store` true inside this actor's own process); replica 1 does the mirror image.
    Every entry's first touch (whichever replica reaches it first) just decrements the refcount
    from 2 to 1 -- store stays at 4 until BOTH replicas have run, then every entry's second touch
    pops it to 0."""
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_on_metadata_split_invoke_setup, True, 0, ("consumer",)))

        def store_len() -> int:
            return ray.get(host.run.remote(lambda h: len(h._store), host))

        packed = ray.get(host.invoke.remote("producer", "produce", (), {}))
        assert store_len() == 4  # 4 samples, each externalized by the producer's own egress

        d0 = Dispatched(
            refs=[packed], op="split", fn=f"{__name__}:split_shard", index=0, n=2, readers=2, on_metadata=True
        )
        d1 = Dispatched(
            refs=[packed], op="split", fn=f"{__name__}:split_shard", index=1, n=2, readers=2, on_metadata=True
        )

        out0 = ray.get(host.invoke.remote("consumer", "receive", (d0,), {}))  # 1st simulated replica
        assert out0 == ["s0", "s1"]
        assert store_len() == 4  # first touch on every entry (fetch OR release) just decrements

        out1 = ray.get(host.invoke.remote("consumer", "receive", (d1,), {}))  # 2nd simulated replica
        assert out1 == ["s2", "s3"]
        assert store_len() == 0  # both replicas ran -> every entry's 2nd touch popped it
    finally:
        ray.shutdown()


def split_drop_all(payload: list, index: int, n: int) -> list:
    """Test split fn: keeps NOTHING, regardless of index -- every call's dropped set is the
    FULL payload. Models a duplicate-delivery / accounting-drift bug on purpose (never a real
    split): calling this twice against the same producer payload drops (releases) the same
    refs twice, so the second release targets an already-popped key on the owning store --
    exactly the remote failure the cross-actor release-protocol sweep must surface."""
    return []


def _drain_pending(h) -> None:
    """Test-only synchronization helper run INSIDE the actor via the `run` seam: blocks
    (`ray.wait` with a generous timeout here, unlike the production sweep's `timeout=0`)
    until every currently-pending fire-and-forget release has actually finished executing on
    its owning peer -- success OR failure. Never raises (`ray.wait` only checks readiness, it
    doesn't retrieve/re-raise); used purely to make the next `_release_dropped` sweep's
    `timeout=0` check deterministic in this test instead of racing a background RPC."""
    import ray

    if h._pending_releases:
        ray.wait(h._pending_releases, num_returns=len(h._pending_releases), timeout=30)


def _cross_actor_producer_setup():
    class Producer:
        def produce(self):
            return [_traj(_Blob(2000, tag=f"s{i}")) for i in range(2)]

    return {"producer": Producer()}


def _cross_actor_consumer_setup():
    class Consumer:
        def receive(self, batch):
            return [t.transitions[0].action.tag for t in batch]

    return {"consumer": Consumer()}


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_cross_actor_release_failure_surfaces_on_the_next_sweep():
    """Reviewer-flagged gap: `_release_dropped`'s cross-actor branch fires `release_bulk.remote`
    and discarded the returned ObjectRef -- a remote-side failure (double release / accounting
    drift) would resolve into a Ray object nobody ever inspects, so it failed SILENTLY, unlike
    the local branch which raises inline (DESIGN §2.4's KeyError posture). Fix: stash the
    ObjectRef in `self._pending_releases`; the top of every `_release_dropped` call does a
    bounded, non-blocking sweep (`ray.wait(timeout=0)`) and `ray.get`s anything that finished,
    raising `RuntimeError` on failure. Finding-3: that sweep is hoisted to the TOP of every
    `invoke`, so a bad release surfaces fail-loud on the NEXT actor operation of ANY kind (not
    only the next on_metadata-split) -- a one-operation delay, never silently.

    Two REAL `_Host` actors (a genuine cross-actor edge, not a same-process shortcut): "hostA"
    owns the producer's `BulkStore`; "hostB" hosts the consumer and only knows hostA through
    the peers registry (`set_peers`), so `has_local_store` is False on hostB for hostA's refs
    -> every release from hostB's `_release_dropped` takes the cross-actor branch.

    `split_drop_all` (always kept=[]) with `readers=1` (deliberately WRONG -- not the real
    Kc·n formula -- so the very FIRST release fully pops each ref) means calling the SAME
    on_metadata split invoke twice drops (and releases) the identical refs twice: the 1st
    release succeeds, the 2nd targets an already-popped key on hostA -> KeyError remotely.
    `_drain_pending` (run inside hostB via the `run` test seam) forces each fire-and-forget
    release to actually finish before the next step, so the failure is deterministically
    ready for the 3rd invoke's sweep to discover and raise on."""
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host_a = _Host.remote()
        ray.get(host_a.run_setup.remote(_cross_actor_producer_setup, True, 0, (), "storeA"))
        host_b = _Host.remote()
        ray.get(host_b.run_setup.remote(_cross_actor_consumer_setup, True, 0, ("consumer",), "storeB"))
        ray.get(host_b.set_peers.remote({"storeA": host_a}))

        packed = ray.get(host_a.invoke.remote("producer", "produce", (), {}))

        def d(index: int) -> Dispatched:
            # readers=1 is deliberately wrong (see docstring): forces the 1st release to
            # fully pop each ref so the 2nd release call targets an already-popped key.
            return Dispatched(
                refs=[packed],
                op="split",
                fn=f"{__name__}:split_drop_all",
                index=index,
                n=2,
                readers=1,
                on_metadata=True,
            )

        out0 = ray.get(host_b.invoke.remote("consumer", "receive", (d(0),), {}))  # 1st release: succeeds
        assert out0 == []  # split_drop_all keeps nothing
        ray.get(host_b.run.remote(_drain_pending, host_b))  # force it to actually finish

        ray.get(host_b.invoke.remote("consumer", "receive", (d(1),), {}))  # 2nd release: fails remotely
        ray.get(host_b.run.remote(_drain_pending, host_b))  # force the failure to become "ready"

        # Finding-3: the sweep is hoisted to the TOP of every `invoke`, so the failure now surfaces on
        # the next invoke of ANY method — a PLAIN body call ([] batch, no Dispatched arg, no split) —
        # not only the next on_metadata split. This bounds surfacing to the next actor op of any kind.
        with pytest.raises(RuntimeError, match="cross-actor bulk release failed"):
            ray.get(host_b.invoke.remote("consumer", "receive", ([],), {}))  # plain call's top-of-invoke sweep raises
    finally:
        ray.shutdown()


# ── Task 9: `_Host.invoke` resolves the actor's REAL dp coordinate, not the (0, 1) pin ──────────


@pytest.mark.L0
@pytest.mark.CPU
def test_invoke_threads_the_actors_real_dp_coordinate_into_select_chunk():
    """`dp_group_sharded` is deleted; `select_chunk` is now the sole dp-axis owner, and it must see
    the ACTOR's real `(dp_rank, dp_world)` from `config/build.py::_dp_coords(self._pdims)` — not
    the `(0, 1)` pin `_Host.invoke` used while `dp_group_sharded` was still the live owner on this
    path (Task 8). A pdims reporting `dp_rank=1, dp_world=2` must make this rank keep only its HALF
    of a select-carrying batch; the (0, 1) pin would have handed it all 4.

    `set_mesh` is bypassed (it calls `pdims.build_device_mesh`, which needs a live process group
    this single, un-rendezvoused actor never joins) — `self._pdims` is injected directly via the
    same `run` test-only introspection seam `store_len()` uses elsewhere in this file.
    """
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_adaptation_setup, True, 0, ("trainer",)))  # codec on; trainer materializes

        # A stub ParallelDims reporting dp_rank=1 of a dp_world=2 shard mesh — the real object
        # `config/build.py::_dp_coords` reads (`dp_enabled`/`dp_shard_enabled`/`dp_shard_mesh`),
        # not a live device mesh (mirrors tests/test_config_vllm_rollout.py's `_mesh` stub).
        mesh = SimpleNamespace(ndim=1, get_local_rank=lambda dim: 1, size=lambda dim=None: 2)
        pdims = SimpleNamespace(
            dp_enabled=True, dp_replicate=1, dp_shard_enabled=True, dp_shard_mesh=mesh, dp_mesh=mesh
        )
        ray.get(host.run.remote(lambda h: setattr(h, "_pdims", pdims), host))

        items = [_traj(_Blob(8, tag=f"s{i}")) for i in range(4)]
        d = Dispatched(
            refs=[items],
            op="one",
            fn=None,
            index=0,
            n=1,
            readers=1,
            select=_SELECT_CHUNK_REF,
            select_kw={"mini": 0, "micro": 0, "n_mini": 1, "n_micro": 1},  # mini/micro inert: dp is the only axis
        )
        result = ray.get(host.invoke.remote("trainer", "train", (d,), {}))
        assert len(result) == 2, (
            f"expected THIS rank's half of 4 items (dp_rank=1, dp_world=2), got {len(result)} — the "
            f"(0, 1) pin (or a coordinate not actually read from self._pdims) would hand back all 4"
        )
    finally:
        ray.shutdown()


def _stamped_group(group_id: str, spread: float) -> list[Trajectory]:
    """One complete two-member group carrying the reward stage's spread stamp and a bulk action."""
    members = []
    for k in range(2):
        tr = Transition(obs=ObsBundle(), action=_Blob(2000, tag=f"{group_id}_k{k}"), done=True)
        tr.algo_extra["spread"] = spread
        members.append(Trajectory(transitions=[tr], id=f"{group_id}_k{k}", group_id=group_id))
    return members


def test_buffer_drain_releases_dropped_bulk_without_a_fetch_under_tp2() -> None:
    """DESIGN §5: dropped bulk never moves. The buffer's `drain` is a lease-releasing egress: the host
    releases each drained BulkRef at its store instead of a bulk consumer materializing it. The kept
    group is pulled once per TP rank (readers=2) as usual; the dropped group is never fetched at all."""
    from cosmos_framework.posttrain.control.buffer import ExperienceBuffer, MetadataGate

    class Rollout:  # the bulk owner; `receive` stands in for a TP rank's bulk-consuming ingress
        def generate(self):
            return _stamped_group("tied", 0.0) + _stamped_group("informative", 1.0)

        def receive(self, batch):
            return [t.transitions[0].action.tag for t in batch]

    host = _HostImpl()
    host._bodies = {"rollout": Rollout(), "replay": ExperienceBuffer()}
    host._codec, host._store, host._materialize = True, BulkStore("owner"), {"rollout"}

    packed = host.invoke("rollout", "generate", (), {})  # producer egress: 4 actions -> 4 BulkRefs
    assert len(host._store) == 4
    host.invoke("replay", "put_samples", (packed,), {})  # metadata index: nothing fetched
    batch = host.invoke("replay", "select_prioritized", (2, MetadataGate(key="spread")), {})
    mirrored = Dispatched(refs=[batch], op="one", fn=None, index=0, n=1, readers=2)  # (R,1) -> (R,TP=2)
    for _rank in range(2):
        assert host.invoke("rollout", "receive", (mirrored,), {}) == ["informative_k0", "informative_k1"]
    assert (host._store.fetch_count, len(host._store)) == (4, 2)  # kept group: 2 refs x 2 ranks; tied still live

    dropped = host.invoke("replay", "drain", (), {})

    assert [t.id for t in dropped] == ["tied_k0", "tied_k1"]  # metadata still returns to the driver
    assert (host._store.fetch_count, len(host._store)) == (4, 0)  # released at the store, zero extra fetches
