# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Mover implementations + selection (design spec §2.3/§2.4).

shared/copy are plane-specific (spec §2.2b); ipc/nccl (added in later tasks) are shared by
both planes. resolve_mover keys on (relationship, plane, capability) with fail-loud guardrails.
Ray-free: the Ray-backed data mover lives in workers/ray.py, not here.
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE, STAGED, MoveHandle, Mover


class SharedMover:
    """Weight tier-3: trainer & rollout share one nn.Module -> nothing to move."""

    lifetime = STAGED
    tensors_only = False

    def stage(self, obj: Any) -> MoveHandle:
        return MoveHandle(kind="shared")

    def fetch(self, handle: MoveHandle) -> Any:
        return None  # WeightSync.apply skips import on None

    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""

    def close(self) -> None:
        """No mover-owned persistent resources; stores and communicators belong to their owners."""


class CopyMover:
    """Weight copy: by-value CPU snapshot embedded in the handle (rides the return value).
    Device-agnostic; the byte move is the object-store hop that carries the handle."""

    lifetime = STAGED
    tensors_only = False

    def stage(self, obj: dict) -> MoveHandle:
        # The mover owns the mechanism-appropriate snapshot (the mapper only detaches, no copy).
        # .to("cpu", copy=True) forces exactly ONE copy for both a GPU input (device transfer) and a
        # CPU input (where a plain .cpu() would alias) — no redundant clone, and Copy never allocates
        # a durable GPU clone (unlike IPC), so the copy path holds a single full-model copy at peak.
        payload = {k: v.detach().to("cpu", copy=True) for k, v in obj.items()}
        return MoveHandle(kind="copy", payload=payload)

    def fetch(self, handle: MoveHandle) -> Any:
        if handle is None or handle.payload is None:
            raise ValueError("CopyMover.fetch requires a handle carrying a payload")
        return handle.payload

    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""

    def close(self) -> None:
        """No mover-owned persistent resources; stores and communicators belong to their owners."""


class IpcMover:
    """Same-GPU cross-process: only CUDA-IPC handles cross; param memory stays on the GPU.

    The weight-plane resolver uses ``single_buffer=True``: every bucket is packed into ONE staging
    buffer exported once. Exporting a handle per tensor can make CUDA IPC registrations outlive the
    last Python reference and grow by a model per sync. The direct constructor retains the legacy
    per-tensor mode for callers outside the synchronous weight stream.

    The single buffer's lifetime is `release_after_sync`. Off by default: one allocation and one IPC
    registration carried through the whole run. On, both sides drop their references after each
    sync and the memory comes back to training and rollout in between -- measured 2.38 GiB, no
    `ipc_collect` needed. Both sides must release: the consumer's mapping holds an IPC reference to
    the producer's storage, so a producer-only drop frees nothing.

    In single-buffer mode it is allocated on the first `stage` at the size `reserve` took from the
    whole plan — sizing per bucket instead would reallocate mid-run and issue a second handle while
    the first is still registered. The size comes from the largest bucket.

    Single-buffer fetch always returns one owned tensor per entry. vLLM V1 pickles its weight-load
    callable into a separate EngineCore process; passing carved views there serializes the shared
    backing allocation once per view and can exceed msgpack's 4 GiB object limit.

    The legacy mode retains its original meaning: ``borrow=True`` aliases producer tensors and
    ``borrow=False`` stages durable clones. Single-buffer staging and fetch each cost one D2D copy;
    they buy one IPC registration and independent consumer storages."""

    lifetime = STAGED
    tensors_only = True
    _buffer_seq: int = 0

    def __init__(
        self,
        borrow: bool = False,
        release_after_sync: bool = False,
        single_buffer: bool = False,
    ) -> None:
        self._borrow = borrow
        self._release_after_sync = release_after_sync
        self._single_buffer = single_buffer
        self._pinned = None  # legacy producer keepalive: the staged per-tensor state dict
        self._buffer = None  # producer: the one staging buffer every bucket is packed into
        self._buffer_handle = None  # its IPC handle, exported once and re-sent with every bucket
        self._buffer_id = None  # identifies the current buffer so the consumer maps it once
        self._reserved = 0  # largest bucket in the installed plan; see reserve()
        self._mapped = {}  # consumer: buffer_id -> the tensor rebuilt from the producer's handle

    def reserve(self, buckets: list) -> None:
        """Size the opt-in staging buffer from the WHOLE plan before the first `stage`.

        The legacy per-tensor mode is a no-op.

        Args:
            buckets: the bucket plan itself — `[[(name, shape, dtype_str), ...], ...]`. Sized from
                METADATA through the same `comm/transfer/flat.py` helpers `stage` packs with, so
                the reservation and the packing cannot drift; `workers/ray.py::init_nixl` sizes the
                NIXL staging buffer the same way.
        """
        if not self._single_buffer:
            return

        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

        self._reserved = max((flat_nbytes_for_manifest(bucket) for bucket in buckets), default=0)

    def stage(self, obj: dict) -> MoveHandle:
        if not self._single_buffer:
            from cosmos_framework.posttrain.comm.transfer.ipc import state_dict_to_ipc

            # This is intentionally the pre-feature implementation. ``borrow`` aliases live
            # tensors; the safe default pins clones until the next stage.
            state = (
                {key: value.detach() for key, value in obj.items()}
                if self._borrow
                else {key: value.detach().clone() for key, value in obj.items()}
            )
            self._pinned = state
            return MoveHandle(kind="ipc", payload=state_dict_to_ipc(state))

        if not obj:
            raise ValueError("IpcMover.stage requires at least one tensor in single-buffer mode")

        import torch

        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes, flat_plan, pack
        from cosmos_framework.posttrain.comm.transfer.ipc import tensor_to_ipc

        plan = flat_plan(obj)
        first = next(iter(obj.values()))
        want = max(flat_nbytes(plan), self._reserved)
        if self._buffer is None or self._buffer.numel() < want:
            from cosmos_framework.posttrain.utils.residency import untracked_allocation

            # The consumer maps this buffer over CUDA IPC, which the memory saver's mappings cannot
            # export, and it must stay mapped while the producer is paused between syncs.
            with untracked_allocation():
                self._buffer = torch.empty(want, dtype=torch.uint8, device=first.device)
            self._buffer_handle = tensor_to_ipc(self._buffer)
            # A counter, not id(): a reallocated buffer can land on a freed address, and the
            # consumer would then serve the stale mapping out of its cache.
            IpcMover._buffer_seq += 1
            self._buffer_id = IpcMover._buffer_seq
        pack({k: v.detach() for k, v in obj.items()}, plan, self._buffer)
        # The packing copies are async on this process's stream, and the consumer reads the buffer
        # from ANOTHER process whose stream is unordered against it. Nothing else in the path waits:
        # `consumer_done` guards the opposite direction (producer must not overwrite before the
        # consumer has copied out). Exporting a handle to a buffer whose writes have not landed
        # hands over whatever was there before -- mostly right, occasionally not, which reads as a
        # sync that corrupts a few tensors. The per-tensor export this replaced needed no such wait:
        # its tensors came straight from `full_tensor`, whose collective already synchronized.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return MoveHandle(
            kind="ipc",
            payload={"buffer_id": self._buffer_id, "buffer": self._buffer_handle, "plan": plan},
        )

    def fetch(self, handle: MoveHandle) -> Any:
        if not self._single_buffer:
            from cosmos_framework.posttrain.comm.transfer.ipc import state_dict_from_ipc

            if handle is None or handle.payload is None:
                raise ValueError("IpcMover.fetch requires a handle carrying IPC handles")
            state = state_dict_from_ipc(handle.payload)
            return state if self._borrow else {key: value.clone() for key, value in state.items()}

        from cosmos_framework.posttrain.comm.transfer.flat import carve
        from cosmos_framework.posttrain.comm.transfer.ipc import tensor_from_ipc

        if handle is None or handle.payload is None:
            raise ValueError("IpcMover.fetch requires a handle carrying IPC handles")
        payload = handle.payload
        # Map the producer's buffer ONCE per run. Re-opening the same handle per bucket would grow
        # the consumer's IPC cache the way per-tensor exports grew the producer's.
        buffer = self._mapped.get(payload["buffer_id"])
        if buffer is None:
            buffer = tensor_from_ipc(payload["buffer"])
            self._mapped[payload["buffer_id"]] = buffer
        state = carve(buffer, payload["plan"])
        return {key: value.clone() for key, value in state.items()}

    def release_staged(self) -> None:
        """Drop the staging buffer between syncs, when built with `release_after_sync`.

        Off by default: the buffer costs its size (the largest bucket) for the whole run but is
        allocated and IPC-registered once. On, training and rollout get that memory back between
        syncs and each sync pays one allocation and one registration.

        BOTH sides must call this: the consumer's mapping holds a CUDA IPC reference to the
        producer's storage, so a producer-only release frees nothing. With both, dropping the
        references is enough -- measured 2.38 GiB back on the producer, no `ipc_collect` needed.
        """
        if not self._release_after_sync:
            return
        self._drop_staging()

    def close(self) -> None:
        """Drop the staging buffer, its IPC handle and every consumer mapping, whatever `release_after_sync` says.

        For the end of the owning body's life (`OmniPolicyBody.close` -> `WeightSync.close`): the producer side
        drops the buffer and its handle, the consumer side drops the mappings it opened. `release_staged` stays
        flag-gated so the single-registration contract between syncs is unchanged.
        """
        self._drop_staging()

    def _drop_staging(self) -> None:
        """Release the IPC resources without making a per-stream release terminate the mover."""
        self._pinned = None
        self._buffer = None
        self._buffer_handle = None
        self._buffer_id = None
        self._mapped.clear()

    def consumer_done(self) -> None:
        # Borrowed views are copied out by the caller, while single-buffer snapshots clone from the
        # reusable IPC buffer in fetch(). Both D2D copies are asynchronous, so wait before the
        # producer overwrites the buffer with the next bucket.
        if self._borrow or self._single_buffer:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()


class NcclMover:
    """Disaggregated cross-device: bytes over an injected weight-plane WeightComm (§2.2).
    COLLECTIVE — stage(broadcast-send)/fetch(broadcast-recv) are a co-scheduled matched pair the
    edge co-schedules send and recv (pull_stream_from_peer's COLLECTIVE branch, §4). v1 moves each tensor whole (policy
    full_tensor() upstream) in SORTED KEY ORDER; the consumer derives its receive layout from its
    OWN model (same architecture as the trainer in v1) via `recv_plan`, set by WeightSync.apply
    before fetch — so send and recv share no data dependency and cannot deadlock."""

    lifetime = COLLECTIVE
    tensors_only = True

    def __init__(self, comm: Any = None, src: int = 0):
        self._comm = comm
        self._src = int(src)
        self.recv_plan = None  # list[(name, shape, dtype_str)] set by WeightSync.apply on the consumer

    def stage(self, obj: dict) -> MoveHandle:
        if self._comm is None:
            raise ValueError(
                "NcclMover has no WeightComm: the cross-role weight plane was never initialized. The "
                "driver must call _Host.init_weight_plane(...) to build and inject the side group "
                "before the first sync. NOTE: bind_device_groups(weight_sync=True) does NOT yet "
                "auto-wire the plane for cross-device (disaggregated) placement — use colocated "
                "placement (IPC) or drive init_weight_plane manually (see tests/test_comm_weight_sync_e2e.py)."
            )
        plan = []
        for name in sorted(obj):
            t = obj[name].contiguous()
            self._comm.broadcast(t, src=self._src)  # producer sends, sorted order
            plan.append((name, tuple(t.shape), str(t.dtype)))
        return MoveHandle(kind="nccl", meta={"plan": plan, "src": self._src})

    def fetch(self, handle: MoveHandle) -> Any:
        if self._comm is None:
            raise ValueError("NcclMover.fetch needs a WeightComm")
        if self.recv_plan is None:
            raise ValueError("NcclMover.fetch needs recv_plan (WeightSync.apply sets it from the consumer model)")
        import torch

        # Receive buffers must land on the transport's own device (TorchDistComm sets `_dev` per its
        # backend: cuda for nccl, cpu for gloo, §2.2) — NOT a global torch.cuda.is_available() guess,
        # which would silently allocate on a CUDA device the comm never touches (wrong for a gloo
        # comm on a GPU-visible host) and would fight loopback/mock comms with no CUDA affinity at
        # all. Fall back to CPU when the comm doesn't expose a device.
        dev = getattr(self._comm, "_dev", None) or torch.device("cpu")
        out = {}
        for name, shape, dtype_str in self.recv_plan:
            buf = torch.empty(shape, dtype=getattr(torch, dtype_str.split(".")[-1]), device=dev)
            self._comm.broadcast(buf, src=self._src)  # consumer receives, sorted order
            out[name] = buf
        return out

    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""

    def close(self) -> None:
        """No mover-owned persistent resources; stores and communicators belong to their owners."""


class LocalStoreMover:
    """Data-plane store-backed mover (Ray-free): same-process -> zero-copy fetch from the
    producer's BulkStore; cross-process (no communicator) -> pickle round-trip (models a real
    IPC/UCX/NCCL move so no-aliasing semantics hold). The Ray-backed sibling (RayStoreMover)
    lives in workers/ray.py to keep comm/ Ray-free. `stage` is the BulkStore.put side."""

    lifetime = STAGED
    tensors_only = False

    def __init__(self, cross_process: bool = False):
        self._cross_process = cross_process

    def stage(self, obj: Any, store=None) -> MoveHandle:
        if store is None:
            raise ValueError("LocalStoreMover.stage needs a BulkStore")
        return store.put(obj)

    def fetch(self, handle: MoveHandle, readers: int = 1) -> Any:
        """Materialize the bulk for `handle`: zero-copy if local, else a pickle round-trip.

        Args:
            handle: The `BulkRef`/`MoveHandle` addressing the producer's stored bulk.
            readers: Expected reader count threaded to `BulkStore.fetch` for refcounting.

        Returns:
            The stored object (a pickle-cloned copy when `cross_process` is set).
        """
        import pickle

        from cosmos_framework.posttrain.comm.bulk import has_local_store, store_for

        if has_local_store(handle):
            obj = store_for(handle).fetch(handle, readers=readers)  # same process -> zero-copy
            return pickle.loads(pickle.dumps(obj)) if self._cross_process else obj
        raise KeyError(f"no local store for {handle!r}; use RayStoreMover for remote refs")

    def reserve(self, buckets: list) -> None:
        """No staging buffer to size."""

    def release_staged(self) -> None:
        """No staging buffer to release."""

    def close(self) -> None:
        """No mover-owned persistent resources; stores and communicators belong to their owners."""


_MOVERS = {"shared": SharedMover, "copy": CopyMover, "ipc": IpcMover, "nccl": NcclMover}


def make_mover(kind: str) -> Mover:
    """Construct a mover by kind. Extended with ipc/nccl in later tasks."""
    return _MOVERS[kind]()


# Weight-plane relationship → the mover CLASS that serves it. Shared by the AOT predicate and
# `resolve_mover`, so the relationship table and `tensors_only` each have exactly one home: adding a
# mover, or flipping one's tensors_only, moves both decisions together.
_WEIGHT_MOVERS: dict[str, type] = {
    "same_process": SharedMover,
    "same_gpu": IpcMover,
    "host": CopyMover,
    "cross_device": NcclMover,
}


# Hint-selectable weight movers, keyed by the `hint` string. Populated by the layer that OWNS the
# dependency — `workers/nixl.py` calls `register_weight_mover("nixl", ...)` at import — so `comm/`
# stays Ray-free (module docstring) while this table and `_WEIGHT_MOVERS` remain the ONE place both
# `resolve_mover` and the AOT bind check read. Mirrors Ray's own `register_tensor_transport`.
_HINT_MOVERS: dict[str, type] = {}

# Hints that predate the registry and are ADVISORY ONLY: they describe what the relationship
# already implies and never override it. Kept so existing callers/tests keep resolving to the
# relationship table instead of tripping the unknown-hint guard below.
_LEGACY_ADVISORY_HINTS = frozenset({"collective"})


def register_weight_mover(name: str, cls: type) -> None:
    """Register a hint-selectable weight mover under `name`.

    Idempotent for the same class (module re-import is normal under Ray's cloudpickle actor
    shipping). Re-registering a DIFFERENT class under a live name is a wiring bug and raises,
    rather than silently swapping the transport under an existing bind.

    Args:
        name: the `hint` value that selects this mover (e.g. "nixl").
        cls: the Mover class; must expose `lifetime` and `tensors_only` like the built-ins.
    """
    prior = _HINT_MOVERS.get(name)
    if prior is not None and prior is not cls:
        raise ValueError(
            f"weight mover hint {name!r} is already registered to {prior.__name__}; refusing to "
            f"rebind it to {cls.__name__} (a live bind may hold the old transport)"
        )
    for attr in ("lifetime", "tensors_only"):
        if not hasattr(cls, attr):
            raise ValueError(f"{cls.__name__} must define `{attr}` to serve as a weight mover")
    _HINT_MOVERS[name] = cls


def _weight_mover_cls(relationship: str, hint: str | None) -> tuple[type | None, str]:
    """(relationship, hint) → mover class. ONE home for both the AOT check and the runtime build, so
    they cannot drift. Returns (cls, reason); cls is None on failure."""
    if hint is not None and hint not in _LEGACY_ADVISORY_HINTS:
        cls = _HINT_MOVERS.get(hint)
        if cls is None:
            known = ", ".join(sorted(_HINT_MOVERS)) or "<none registered>"
            return None, (
                f"unknown transport hint {hint!r} (registered: {known}). A hinted mover registers "
                f"itself at import of the module that owns its dependency — hint='nixl' needs "
                f"`cosmos_framework.posttrain.workers.nixl` imported first."
            )
        return cls, ""
    cls = _WEIGHT_MOVERS.get(relationship)
    if cls is None:
        return None, f"unknown relationship {relationship!r} (known: {', '.join(_WEIGHT_MOVERS)})"
    return cls, ""


def transport_capability_ok(
    relationship: str, plane: str, capability: str | None = None, hint: str | None = None
) -> tuple[bool, str]:
    """Pure, GPU-less transport decision shared by `resolve_mover` and the AOT bind check. Can a mover
    be resolved for this (relationship, plane, capability, hint)? Returns (ok, reason); constructs NO
    mover. Data-plane edges are always movable (LocalStoreMover); on the weight plane the answer is
    read off the mover class itself, so it cannot drift from what `resolve_mover` builds (DESIGN §3.3).

    `hint`, when given, OVERRIDES `relationship`: placement is a fact, transport is a choice."""
    if plane == "data":
        return True, ""  # LocalStoreMover, always resolvable (mirrors the runtime early-return)
    cls, reason = _weight_mover_cls(relationship, hint)
    if cls is None:
        return False, f"{reason} for plane={plane!r}"
    if cls.tensors_only and capability != "tensor":
        return False, f"{cls.__name__} is tensors_only; resolve with capability='tensor' (got {capability!r})"
    return True, ""


def resolve_mover(
    *,
    relationship: str,
    plane: str = "data",
    capability: str | None = None,
    hint: str | None = None,
    comm: Any = None,
) -> Mover:
    """Placement-aware selection (spec §2.4). The capability/relationship decision is delegated to
    `transport_capability_ok` (one source of truth with the AOT bind check) and the class comes from the
    same `_weight_mover_cls` lookup; this only supplies each mover's arguments. Fails loud at selection.

    `hint` selects a registered transport (`register_weight_mover`) in preference to the relationship
    table — `cross_device` + hint='nixl' picks the NIXL mover while the default stays NCCL."""
    if plane == "data":
        return LocalStoreMover(cross_process=relationship not in ("same_process",))
    ok, reason = transport_capability_ok(relationship, plane, capability, hint)
    if not ok:
        raise ValueError(reason)
    cls, _ = _weight_mover_cls(relationship, hint)
    if hint is not None and hint in _HINT_MOVERS:
        # REGISTERED hints only — a legacy advisory hint ("collective") must keep falling through
        # to the argument-supplying branches below, which is where NcclMover gets its `comm`.
        # Registered movers own their transport lookup (NixlWeightMover reads the per-process
        # NixlTransport singleton at stage() time), so there is nothing to inject here.
        return cls()
    if cls is IpcMover:
        # One reusable buffer prevents per-tensor IPC registrations. Fetch snapshots every carved
        # tensor before vLLM sends it to its EngineCore process.
        return IpcMover(borrow=False, release_after_sync=False, single_buffer=True)
    if cls is NcclMover:
        # cross_device: different physical GPUs ⇒ GPU↔GPU collective over the weight-plane NCCL
        # communicator, threaded into the mover at init_weight_plane (comm=None here is the normal bind
        # path — the driver builds the plane later).
        return NcclMover(comm=comm)
    # same_process ⇒ SharedMover; host ⇒ CopyMover, a host-staged copy (object store): CPU runs /
    # in-process separate models, where the byte move is the object-store/return-value hop.
    return cls()
