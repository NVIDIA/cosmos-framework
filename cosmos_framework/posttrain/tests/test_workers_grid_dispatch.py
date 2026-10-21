# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Registry-driven per-edge routing in ``GridBackend`` (Task A4, DESIGN §2.1/§2.2).

Drives ``GridBackend._align_one`` / ``_actor_args`` directly with a fake ``CollectiveHandle``
(sentinel-string refs, identity getter) — CPU-only, no actors ever invoked. ``_Host.invoke``
(Task A5) is what would actually EXECUTE a returned ``Dispatched``; these tests only check
that ``submit``-side routing builds the right descriptor.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

ray = pytest.importorskip("ray")  # GridBackend.__init__ lazily imports workers.replica -> ray

from cosmos_framework.posttrain.roles.core import dispatch as registry  # noqa: E402
from cosmos_framework.posttrain.workers.grid import GridBackend  # noqa: E402
from cosmos_framework.posttrain.workers.worker_group import (  # noqa: E402
    CollectiveHandle,
    Dispatched,
    Mirror,
    Sharded,
)


def _handle(shape: tuple) -> CollectiveHandle:
    """A CollectiveHandle whose refs are sentinel strings ``"ref_{r}_{k}"``, getter=identity —
    lets assertions compare refs by value without ever resolving anything."""
    R, K = shape
    refs = {(r, k): f"ref_{r}_{k}" for r in range(R) for k in range(K)}
    return CollectiveHandle(refs, shape, getter=lambda x: x)


def _gb() -> GridBackend:
    """A GridBackend with one placeholder cell — none of these tests call .submit()/invoke a
    real actor, only the pure routing methods (_align_one / _actor_args / _dispatch_from_spec),
    so the actor handle is never touched."""
    return GridBackend({(0, 0): ("sid0", object())}, shape=(1, 1))


def test_marker_overrides_registry_hit():
    """A call-site Sharded marker wins over a registry entry for the SAME (kind, method,
    param) — DESIGN §2.1 precedence tier 1 over tier 2. Uses the real registered
    ("trainer", "train", "batch") entry (Task A3) so a hit is guaranteed without monkeypatch,
    then wraps the SAME edge in a marker with a DIFFERENT reduce= to prove the marker's own
    spec — not the registry's — was used."""
    assert registry.lookup("trainer", "train", "batch") is not None  # sanity: registry entry exists
    gb = _gb()
    h = _handle((4, 1))  # P=4 producer units
    marker = Sharded(h, reduce="test_sentinel:override_reduce")
    d = gb._align_one(marker, 0, 0, (2, 4), kind="trainer", method="train", param="batch")
    assert isinstance(d, Dispatched)
    assert d.op == "reduce" and d.fn == "test_sentinel:override_reduce"  # marker's fn, not the registry's
    assert d.refs == ["ref_0_0", "ref_1_0"] and d.n == 2 and d.index == 0 and d.readers == 4


def test_registry_hit_reduce_4to2():
    """Bare CollectiveHandle, no marker: (4,1) producer -> (2,4) consumer hits the registered
    ("trainer", "train", "batch") Sharded(reduce=...) spec -> P=4, R_c=2 -> n=2 units/replica."""
    gb = _gb()
    h = _handle((4, 1))
    d = gb._align_one(h, 0, 0, (2, 4), kind="trainer", method="train", param="batch")
    assert isinstance(d, Dispatched)
    assert d.refs == ["ref_0_0", "ref_1_0"]
    assert d.op == "reduce"
    assert d.fn == "cosmos_framework.posttrain.roles.trainer.llm:concat_batches"
    assert d.index == 0 and d.n == 2 and d.readers == 4 and d.on_metadata is False

    d1 = gb._align_one(h, 1, 0, (2, 4), kind="trainer", method="train", param="batch")
    assert d1.refs == ["ref_2_0", "ref_3_0"] and d1.readers == 4


def test_registry_hit_split_1to2(monkeypatch):
    """Bare CollectiveHandle: (1,1) producer -> (2,2) consumer, registered split — R_c=2 =
    2*P=1 -> every replica shares the ONE unit, split() picks its shard by index."""
    monkeypatch.setitem(registry.DISPATCH, ("env", "sample", "seed"), Sharded(split="test_sentinel:split_fn"))
    gb = _gb()
    h = _handle((1, 1))
    d0 = gb._align_one(h, 0, 0, (2, 2), kind="env", method="sample", param="seed")
    d1 = gb._align_one(h, 1, 0, (2, 2), kind="env", method="sample", param="seed")
    assert d0.refs == ["ref_0_0"] == d1.refs  # same ref, both replicas
    assert d0.op == "split" and d0.fn == "test_sentinel:split_fn"
    assert d0.index == 0 and d1.index == 1  # split shard index differs per replica
    assert d0.n == 2 and d0.readers == 4 and d1.readers == 4  # readers = K_c * n = 2*2


def test_sharded_reduce_missing_fn_raises(monkeypatch):
    monkeypatch.setitem(registry.DISPATCH, ("env", "sample", "seed_missing_reduce"), Sharded())  # no reduce=
    gb = _gb()
    h = _handle((4, 1))
    with pytest.raises(ValueError, match="reduce"):
        gb._align_one(h, 0, 0, (2, 4), kind="env", method="sample", param="seed_missing_reduce")


def test_mirror_marker_replicates_single_unit():
    gb = _gb()
    h = _handle((1, 1))
    marker = Mirror(h)
    d = gb._align_one(marker, 1, 0, (3, 2), kind="env", method="broadcast", param="cfg")
    assert isinstance(d, Dispatched)
    assert d.refs == ["ref_0_0"] and d.op == "one" and d.fn is None
    assert d.index == 0 and d.n == 1 and d.readers == 3 * 2  # R_c * K_c


def test_mirror_requires_single_producer_unit():
    gb = _gb()
    h = _handle((2, 1))  # P=2 != 1
    with pytest.raises(ValueError, match="P == 1"):
        gb._align_one(Mirror(h), 0, 0, (2, 2), kind="env", method="broadcast", param="cfg")


# These three pin the BARE-fallback invariants (tier 3): rank-mirror refcount wrap, symmetric
# raw-ref byte-identity, and replica-mismatch raise. They key off ("reward","score","trajs") — a
# genuinely UNREGISTERED edge (reward/buffer edges stay bare / dp_aligned by design). Task C2
# registered ("rollout","generate","prompts") as a Sharded split, so it is no longer bare and
# can't stand in for these bare-path cases (it would produce a Dispatched via tier 2 instead).
def test_bare_equal_replica_rank_mirror_wraps_multireader():
    """Bare (4,1) -> (4,2) rank mirror: producer ranks=1 feeds consumer ranks=2, so producer
    (r, 0) is read by BOTH consumer (r, 0) and (r, 1). That multi-reader ref is wrapped
    ``Dispatched(op="one", readers=2)`` so the refcounted BulkStore pops only after both lanes pull."""
    gb = _gb()
    h = _handle((4, 1))
    consumer_shape = (4, 2)

    # Both consumer cells mapping to producer (0,0) get the same wrapped ref (refcount honored).
    d_k0 = gb._align_one(h, 0, 0, consumer_shape, kind="reward", method="score", param="trajs")
    d_k1 = gb._align_one(h, 0, 1, consumer_shape, kind="reward", method="score", param="trajs")
    for d in (d_k0, d_k1):
        assert isinstance(d, Dispatched)
        assert d.refs == ["ref_0_0"] and d.op == "one" and d.fn is None
        assert d.index == 0 and d.n == 1 and d.readers == 2  # consumer cell count for that ref


def test_bare_equal_shape_single_reader_stays_raw():
    """Symmetric topology (equal shapes on BOTH axes): every producer cell has exactly one
    reader, so it stays a raw ref (zero overhead) — Global Constraints: 'symmetric-topology
    behavior must remain byte-identical'."""
    gb = _gb()
    h = _handle((2, 2))
    for r in range(2):
        for k in range(2):
            out = gb._align_one(h, r, k, (2, 2), kind="reward", method="score", param="trajs")
            assert out == f"ref_{r}_{k}"  # raw ref, no Dispatched wrapper


def test_bare_replica_mismatch_raises_naming_edge():
    gb = _gb()
    h = _handle((1, 1))
    with pytest.raises(ValueError) as ei:
        gb._align_one(h, 0, 0, (2, 2), kind="reward", method="score", param="trajs")
    msg = str(ei.value)
    assert "reward" in msg and "score" in msg and "trajs" in msg
    assert "(1, 1)" in msg and "(2, 2)" in msg


def test_actor_args_binds_positional_param_name_into_registry_lookup():
    """End-to-end through _actor_args: a positional arg's name is resolved via
    bind_param_names((kind, method)) before the registry lookup — proves the wiring from
    submit()'s (args, kw) down to _align_one's `param=`, not just direct _align_one calls."""
    gb = _gb()
    h = _handle((4, 1))
    aa, kk = gb._actor_args(0, 0, (2, 4), (h,), {}, kind="trainer", method="train")
    assert isinstance(aa[0], Dispatched)
    assert aa[0].fn == "cosmos_framework.posttrain.roles.trainer.llm:concat_batches"
    assert kk == {}


def test_nft_profile_exact_rank_override_wins_over_generic_trainer_sharding() -> None:
    gb = _gb()
    h = _handle((1, 2))
    args, kwargs = gb._actor_args(
        0,
        1,
        (1, 2),
        (h,),
        {},
        kind="trainer",
        method="train",
        dispatch_profile="nft",
    )
    assert args == ("ref_0_1",)
    assert kwargs == {}


def test_nft_profile_rejects_shape_drift_at_dispatch_boundary() -> None:
    gb = _gb()
    h = _handle((1, 2))
    with pytest.raises(ValueError, match="dispatch profile 'nft'.*identical producer and consumer shapes"):
        gb._align_one(
            h,
            0,
            0,
            (1, 1),
            kind="reward",
            method="score_with_provider_report",
            param="trajectories",
            dispatch_profile="nft",
        )


def test_actor_args_kwarg_uses_its_own_name(monkeypatch):
    monkeypatch.setitem(registry.DISPATCH, ("env", "sample", "seed"), Mirror())
    gb = _gb()
    h = _handle((1, 1))
    aa, kk = gb._actor_args(0, 0, (2, 2), (), {"seed": h}, kind="env", method="sample")
    assert aa == ()
    assert isinstance(kk["seed"], Dispatched) and kk["seed"].readers == 4


def test_sharded_multi_rank_spmd_producer_raises_naming_ranks1(monkeypatch):
    """Fix (spec 2026-07-22 §2.2): a Sharded edge whose PRODUCER declares K_p > 1 is a rank-mirror
    SPMD helper — its lanes duplicate the same output, so ``flat_units`` double-counts them. Reject
    BEFORE routing, naming the edge + sub-shape + the ranks=1 remedy, rather than silently K-folding
    the reduce (loss corruption) or misfiring a 'reduce= required'. Uses the real registered
    ("trainer","train","batch") reduce spec so the hit is guaranteed."""
    gb = _gb()
    h = _handle((1, 2))  # K_p = 2 > 1: a multi-rank SPMD producer sub-shape
    with pytest.raises(ValueError) as ei:
        gb._align_one(h, 0, 0, (2, 4), kind="trainer", method="train", param="batch")
    msg = str(ei.value)
    assert "trainer" in msg and "train" in msg and "batch" in msg
    assert "(1, 2)" in msg and "ranks=1" in msg and "§2.2" in msg


def test_rank_partitioned_sharded_edge_reduces_distinct_producer_rank_lanes() -> None:
    gb = _gb()
    handle = _handle((1, 2))
    args, kwargs = gb._actor_args(
        0,
        0,
        (1, 4),
        (handle,),
        {},
        kind="critic",
        method="predict",
    )

    dispatched = args[0]

    assert isinstance(dispatched, Dispatched)
    assert dispatched.refs == ["ref_0_0", "ref_0_1"]
    assert dispatched.op == "reduce"
    assert dispatched.fn == "cosmos_framework.posttrain.roles.trainer.llm:concat_batches"
    assert dispatched.select == "cosmos_framework.posttrain.workers.dispatch:select_model_dp"
    assert dispatched.select_kw == {}
    assert kwargs == {}


def test_sharded_spec_both_reduce_and_split_raises(monkeypatch):
    """Fix: a Sharded spec setting BOTH reduce= and split= is a mis-declaration (route_sharded used
    to silently prefer reduce). Fail loud, naming the edge."""
    monkeypatch.setitem(
        registry.DISPATCH,
        ("env", "sample", "both"),
        Sharded(reduce="test_sentinel:r", split="test_sentinel:s"),
    )
    gb = _gb()
    h = _handle((4, 1))
    with pytest.raises(ValueError) as ei:
        gb._align_one(h, 0, 0, (2, 4), kind="env", method="sample", param="both")
    msg = str(ei.value)
    assert "BOTH" in msg and "env" in msg and "sample" in msg and "both" in msg
