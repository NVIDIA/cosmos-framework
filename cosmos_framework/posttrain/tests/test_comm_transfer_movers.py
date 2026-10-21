# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Mover mechanisms: shared/copy round-trips + resolver matrix + guardrails."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.comm.transfer import ipc
from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes, flat_plan, pack
from cosmos_framework.posttrain.comm.transfer.mover import STAGED, MoveHandle


@pytest.fixture(autouse=True)
def _restore_hint_movers():
    """`_HINT_MOVERS` is module-global with registration but NO deregistration, and several tests
    below register a class defined in the test body. On a SECOND run in the same process that is a
    different class object under a live name, so `register_weight_mover` refuses the rebind by
    design and the test fails -- with a message that reads as nonsense (`_A` -> `_A`), since both
    objects share a `__name__`.

    That breaks the repo's own L0 acceptance gate: `--flake-finder --flake-runs=50` duplicates
    items within one process, so these would fail on run 2 of 50. Verified by calling the test
    bodies twice: run 1 PASS, run 2 ValueError."""
    from cosmos_framework.posttrain.comm.transfer import movers as M

    saved = dict(M._HINT_MOVERS)
    yield
    M._HINT_MOVERS.clear()
    M._HINT_MOVERS.update(saved)


from cosmos_framework.posttrain.comm.transfer.movers import (
    CopyMover,
    IpcMover,
    SharedMover,
    make_mover,
    resolve_mover,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_mover_is_noop():
    mv = SharedMover()
    assert mv.lifetime == STAGED and mv.tensors_only is False
    h = mv.stage({"w": torch.zeros(2)})
    assert isinstance(h, MoveHandle) and h.kind == "shared"
    assert mv.fetch(h) is None  # instance-share: nothing to import


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_mover_roundtrip_by_value_on_cpu():
    mv = CopyMover()
    src = {"w": torch.arange(4, dtype=torch.float32), "b": torch.ones(2)}
    h = mv.stage(src)
    assert h.kind == "copy" and set(h.payload) == {"w", "b"}
    got = mv.fetch(h)
    assert torch.equal(got["w"], src["w"]) and torch.equal(got["b"], src["b"])
    # by-value: mutating the source after stage does not change the staged payload
    src["w"].add_(100.0)
    assert not torch.equal(mv.fetch(h)["w"], src["w"])


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_mover_fetch_rejects_empty():
    mv = CopyMover()
    with pytest.raises(ValueError):
        mv.fetch(MoveHandle(kind="copy", payload=None))


@pytest.mark.L0
@pytest.mark.CPU
def test_make_mover_kinds_and_bogus():
    assert isinstance(make_mover("shared"), SharedMover)
    assert isinstance(make_mover("copy"), CopyMover)
    with pytest.raises(KeyError):
        make_mover("bogus")


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_weight_same_process_is_shared():
    mv = resolve_mover(relationship="same_process", plane="weight")
    assert isinstance(mv, SharedMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_guardrail_tensors_only_rejects_non_tensor():
    # capability="object" must never resolve to a tensors_only mover.
    with pytest.raises(ValueError):
        resolve_mover(relationship="same_gpu", plane="weight", capability="object")


@pytest.mark.L1
@pytest.mark.GPU
def test_ipc_mover_stage_pins_and_emits_handle():
    # Same-process can exercise only the STAGE side: CUDA forbids opening your OWN IPC handle
    # in-process ("invalid device context"), so the fetch/round-trip is covered cross-process in
    # test_comm_ipc.py. Here: stage pins the snapshot (IPC lifetime requirement) and emits an
    # ipc-kind handle carrying the reduce_tensor payload.
    if not torch.cuda.is_available():
        pytest.skip("GPU required for CUDA IPC")
    from cosmos_framework.posttrain.comm.transfer.movers import IpcMover, make_mover

    mv = make_mover("ipc")
    assert isinstance(mv, IpcMover) and mv.tensors_only is True
    src = {"w": torch.arange(6, dtype=torch.float32, device="cuda")}
    h = mv.stage({k: v.detach().clone() for k, v in src.items()})
    assert h.kind == "ipc" and h.payload is not None
    assert mv._pinned is not None  # snapshot pinned until the next stage() (CUDA-IPC lifetime)


@pytest.mark.L0
@pytest.mark.CPU
def test_nccl_mover_contract_is_collective():
    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover, make_mover

    mv = make_mover("nccl")
    assert isinstance(mv, NcclMover)
    assert mv.lifetime == COLLECTIVE and mv.tensors_only is True
    # Real mover now: make_mover builds it with no WeightComm, so stage/fetch fail loud (ValueError)
    # rather than the old NotImplementedError stub. A bound comm is injected via resolve_mover(comm=).
    with pytest.raises(ValueError):
        mv.stage({"w": torch.zeros(2)})
    with pytest.raises(ValueError):
        mv.fetch(None)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_host_weight_is_copy():
    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover

    # Host-staged (CPU / in-process separate models) weight sync uses the object-store CopyMover.
    assert isinstance(resolve_mover(relationship="host", plane="weight", capability="tensor"), CopyMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_cross_device_gpu_uses_nccl():
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover

    # NcclMover is implemented now: different-GPU weight sync routes to it through the normal bind
    # flow (no explicit hint needed). The comm is threaded in later at init_weight_plane, so it
    # resolves with comm=None here. hint="collective" is still accepted for back-compat.
    assert isinstance(resolve_mover(relationship="cross_device", plane="weight", capability="tensor"), NcclMover)
    assert isinstance(
        resolve_mover(relationship="cross_device", plane="weight", capability="tensor", hint="collective"),
        NcclMover,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_cross_device_requires_tensor():
    # NcclMover is tensors_only; a non-tensor payload fails loud at resolve, hint or not.
    with pytest.raises(ValueError):
        resolve_mover(relationship="cross_device", plane="weight", capability="object")


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_rejects_unknown_relationship():
    # DESIGN §3.3: an unresolvable relationship fails loud, never silently degrades to CopyMover.
    with pytest.raises(ValueError, match="unknown relationship"):
        resolve_mover(relationship="bogus", plane="weight", capability="tensor")


@pytest.mark.L0
@pytest.mark.CPU
def test_resolver_tensors_only_requires_explicit_tensor_capability():
    from cosmos_framework.posttrain.comm.transfer.movers import IpcMover

    # capability omitted (None) must fail loud at resolve for a tensors_only mover, not fail-late at
    # fetch; an explicit capability="tensor" resolves to the IpcMover.
    with pytest.raises(ValueError, match="tensors_only"):
        resolve_mover(relationship="same_gpu", plane="weight")
    assert isinstance(resolve_mover(relationship="same_gpu", plane="weight", capability="tensor"), IpcMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_same_gpu_weight_single_buffer_returns_owned_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.comm.transfer.movers import IpcMover, make_mover

    # Weight sync uses one reusable IPC allocation and returns owned consumer tensors.
    mv = resolve_mover(relationship="same_gpu", plane="weight", capability="tensor")
    assert isinstance(mv, IpcMover) and mv._borrow is False and mv._single_buffer is True
    # A direct generic constructor retains the safe snapshot/per-tensor contract for consumers
    # outside WeightSync, which do not necessarily copy out synchronously or install a bucket plan.
    generic = make_mover("ipc")
    assert generic._borrow is False and generic._single_buffer is False
    with pytest.raises(ValueError, match="at least one tensor"):
        IpcMover(single_buffer=True).stage({})

    source = {"a": torch.arange(4, dtype=torch.float32), "b": torch.arange(6, dtype=torch.float32)}
    plan = flat_plan(source)
    flat = torch.empty(flat_nbytes(plan), dtype=torch.uint8)
    pack(source, plan, flat)
    monkeypatch.setattr(ipc, "tensor_from_ipc", lambda _handle: flat)
    handle = MoveHandle(kind="ipc", payload={"buffer_id": 1, "buffer": object(), "plan": plan})

    got = IpcMover(borrow=True, single_buffer=True).fetch(handle)

    assert torch.equal(got["a"], source["a"])
    assert torch.equal(got["b"], source["b"])
    assert got["a"].untyped_storage().data_ptr() != flat.untyped_storage().data_ptr()
    assert got["b"].untyped_storage().data_ptr() != flat.untyped_storage().data_ptr()
    assert got["a"].untyped_storage().data_ptr() != got["b"].untyped_storage().data_ptr()


@pytest.mark.L1
@pytest.mark.GPU
def test_ipc_mover_borrow_aliases_stage_vs_clone() -> None:
    # borrow ⇒ stage aliases the producer storage (no clone); default ⇒ stage clones (independent).
    if not torch.cuda.is_available():
        pytest.skip("GPU required for CUDA IPC")
    from cosmos_framework.posttrain.comm.transfer.movers import IpcMover

    src = {"w": torch.arange(6, dtype=torch.float32, device="cuda")}
    borrow = IpcMover(borrow=True)
    borrow.stage({k: v.detach() for k, v in src.items()})
    assert borrow._pinned["w"].data_ptr() == src["w"].data_ptr()  # aliased, no copy
    snap = IpcMover(borrow=False)
    snap.stage({k: v.detach() for k, v in src.items()})
    assert snap._pinned["w"].data_ptr() != src["w"].data_ptr()  # cloned, independent storage


@pytest.mark.L1
@pytest.mark.GPU
def test_ipc_single_buffer_reuses_one_exported_allocation() -> None:
    """Single-buffer mode packs every bucket into one reserved IPC allocation."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required for CUDA IPC")

    mv = IpcMover(borrow=True, single_buffer=True)
    first = mv.stage({"w": torch.arange(6, dtype=torch.float32, device="cuda")})
    buffer_ptr = mv._buffer.data_ptr()
    second = mv.stage({"x": torch.arange(6, dtype=torch.float32, device="cuda")})
    assert mv._buffer.data_ptr() == buffer_ptr  # same buffer, no new allocation
    assert second.payload["buffer_id"] == first.payload["buffer_id"]  # and no new IPC registration
    # A bucket larger than the current buffer grows it; that is the only time a new handle is issued.
    third = mv.stage({"big": torch.zeros(4096, dtype=torch.float32, device="cuda")})
    assert third.payload["buffer_id"] != first.payload["buffer_id"]


@pytest.mark.L0
@pytest.mark.CPU
def test_ipc_mover_reused_buffer_consumer_done_synchronizes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Borrowed aliases and single-buffer clones both read asynchronously from reusable producer
    # storage. Generic per-tensor snapshots own their storage and remain the no-sync case.
    from cosmos_framework.posttrain.comm.transfer.movers import IpcMover

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))
    IpcMover(borrow=True).consumer_done()
    assert calls == [1]  # borrow waits for the consumer copy
    IpcMover(borrow=False).consumer_done()
    assert calls == [1]  # generic per-tensor snapshot: no extra sync
    IpcMover(borrow=False, single_buffer=True).consumer_done()
    assert calls == [1, 1]  # reusable snapshot waits before producer overwrite


@pytest.mark.L0
@pytest.mark.CPU
def test_register_weight_mover_and_select_by_hint():
    from cosmos_framework.posttrain.comm.transfer import movers as M

    class _FakeMover:
        lifetime = "staged"
        tensors_only = True

    M.register_weight_mover("fake", _FakeMover)
    got = M.resolve_mover(relationship="cross_device", plane="weight", capability="tensor", hint="fake")
    assert isinstance(got, _FakeMover)
    # No hint => the relationship table still decides.
    assert isinstance(
        M.resolve_mover(relationship="cross_device", plane="weight", capability="tensor"),
        M.NcclMover,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_register_weight_mover_is_idempotent_but_refuses_rebind():
    from cosmos_framework.posttrain.comm.transfer import movers as M

    class _A:
        lifetime = "staged"
        tensors_only = True

    class _B:
        lifetime = "staged"
        tensors_only = True

    M.register_weight_mover("dup", _A)
    M.register_weight_mover("dup", _A)  # same class: no-op
    with pytest.raises(ValueError, match="already registered"):
        M.register_weight_mover("dup", _B)


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_hint_fails_loud_with_guidance():
    from cosmos_framework.posttrain.comm.transfer import movers as M

    with pytest.raises(ValueError, match="unknown transport hint"):
        M.resolve_mover(relationship="cross_device", plane="weight", capability="tensor", hint="nope")


@pytest.mark.L0
@pytest.mark.CPU
def test_aot_check_and_resolve_agree_on_hint():
    # The bind gate and the runtime build must read the SAME table, or bind passes and the
    # first sync explodes.
    from cosmos_framework.posttrain.comm.transfer import movers as M

    class _TensorsOnly:
        lifetime = "staged"
        tensors_only = True

    M.register_weight_mover("tonly", _TensorsOnly)
    ok, reason = M.transport_capability_ok("cross_device", "weight", None, hint="tonly")
    assert not ok and "tensors_only" in reason
    with pytest.raises(ValueError, match="tensors_only"):
        M.resolve_mover(relationship="cross_device", plane="weight", capability=None, hint="tonly")


@pytest.mark.L0
@pytest.mark.CPU
def test_reserve_takes_the_largest_bucket_of_the_whole_plan() -> None:
    """A bucket that does not fit makes `stage` reallocate and export a SECOND IPC handle while the
    consumer still caches the first -- the per-registration growth the single buffer exists to
    remove. `bucket_plan` accumulates until it EXCEEDS the cap, so an oversized parameter forms its
    own bucket bigger than the others: the max over the plan is what covers that, not the first
    bucket. Sizing goes through the same `flat.py` helpers `stage` packs with.
    """
    from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

    small = [("a", (4, 8), "torch.float32"), ("b", (4,), "torch.float32")]
    large = [("c", (64, 64), "torch.bfloat16")]
    generic = IpcMover()
    generic.reserve([small, large])
    assert generic._reserved == 0  # direct generic mode has no reusable staging allocation

    mv = IpcMover(single_buffer=True)
    mv.reserve([small, large])
    assert mv._reserved == flat_nbytes_for_manifest(large) > flat_nbytes_for_manifest(small)


@pytest.mark.L0
@pytest.mark.CPU
def test_single_buffer_stage_synchronizes_only_where_cuda_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """The post-pack synchronize guards a cross-process read; without CUDA there is no such reader, and the
    guard is what lets the vLLM-Omni second hop run its protocol tests on CPU."""
    calls: list[int] = []
    monkeypatch.setattr(ipc, "tensor_to_ipc", lambda t: ("h", t))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    IpcMover(single_buffer=True).stage({"a": torch.ones(3)})
    assert calls == []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    IpcMover(single_buffer=True).stage({"a": torch.ones(3)})
    assert calls == [1]


@pytest.mark.L0
@pytest.mark.CPU
def test_ipc_close_releases_regardless_of_the_flag_while_release_staged_stays_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ipc, "tensor_to_ipc", lambda t: ("h", t))
    mv = IpcMover(borrow=False, release_after_sync=False, single_buffer=True)
    mv.stage({"a": torch.ones(3)})
    mv._mapped[mv._buffer_id] = mv._buffer  # what a consumer's fetch would cache
    mv.release_staged()
    assert mv._buffer is not None and mv._mapped, "between syncs the single registration must survive"
    mv.close()
    assert mv._buffer is None and mv._buffer_handle is None and mv._buffer_id is None and mv._mapped == {}


@pytest.mark.L0
@pytest.mark.CPU
def test_ipc_stream_release_can_be_followed_by_another_stream_and_final_close(monkeypatch):
    monkeypatch.setattr(ipc, "tensor_to_ipc", lambda tensor: ("ipc", tensor))
    mover = IpcMover(release_after_sync=True, single_buffer=True)
    first = mover.stage({"a": torch.ones(3)})
    first_id = mover._buffer_id
    mover.release_staged()
    assert mover._buffer is None
    second = mover.stage({"a": torch.full((3,), 2.0)})
    assert mover._buffer_id != first_id and second is not first
    mover.close()
    mover.close()
    assert mover._buffer is None and mover._mapped == {}
