# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NIXL weight transport over Ray Direct Transport (design 2026-08-12-nixl-transport).

EVERY ray / nixl / cuda-python / cupy import is FUNCTION-SCOPED: the i4 container ships none
of them, and this module is reachable from `workers/`, so a module-scope import would break
every posttrain run with a failure far from its cause. `nixl_available()` reports;
`NixlTransport.get()` raises.

The singleton lives HERE rather than in `workers/ray.py` deliberately: Ray ships the actor
class BY VALUE via cloudpickle, rebuilding that module's namespace with the same `__name__`
but a different dict — the documented cause of `current_pdims()` silently returning None for
every Ray-backed setup (`workers/ray.py:376-383`). This module is reached through a normal
import resolved via `sys.modules` inside the actor, so there is exactly one copy of the state.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

MIN_RAY = (2, 56, 1)  # adds safe deregistration to the register/target/wait RDT surface

# RLock, not Lock: get()/warmup()/reserve() all take it, and an RLock keeps a future caller
# that reaches get() from inside one of the others from self-deadlocking.
_lock = threading.RLock()
_singleton: "NixlTransport | None" = None


def _release(version: str) -> tuple[int, int, int]:
    """`"2.55.1rc0"` → `(2, 55, 1)`. Always 3 components, zero-padded.

    Filtering non-numeric components out (`if x.isdigit()`) SHORTENS the tuple instead of
    normalizing it, and tuple comparison is length-sensitive: `(2, 55) < (2, 55, 0)` is True, so
    an rc or local build reported as BELOW the floor. The resulting message contradicted itself
    ("ray 2.55.1rc0 < 2.55.0") and, via `_cross_device_transport`, made the whole disaggregated
    path unreachable at bind. Plain `"2.55"` failed the same way despite meaning 2.55.0.

    Uses `packaging`, which ray itself depends on — this only runs after `import ray` succeeded —
    and falls back to leading-digit extraction if that is somehow absent.
    """
    try:
        from packaging.version import Version

        rel = Version(version).release[:3]
    except Exception:  # noqa: BLE001 — malformed/absent packaging must not disable the transport
        import re

        rel = tuple(int(x) for x in re.findall(r"\d+", version)[:3])
    return tuple(rel) + (0,) * (3 - len(rel))


def nixl_available() -> tuple[bool, str]:
    """Is the whole NIXL stack usable in this process? Never raises; never imports at module
    scope.

    Returns:
        `(ok, reason)` — `reason` is empty when ok, and explains the gap otherwise.
    """
    try:
        importlib.import_module("ray")
    except Exception as e:  # noqa: BLE001 — any import failure is simply unavailability
        return False, f"ray not importable: {e}"
    try:
        from ray.experimental import (  # noqa: F401
            deregister_nixl_memory,
            register_nixl_memory,
            set_target_for_ref,
            wait_tensor_freed,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"ray.experimental RDT API missing: {e}"
    try:
        import nixl  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"nixl not importable: {e}"
    try:
        import cupy  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"cupy not importable: {e}"
    try:
        from cuda.bindings import driver  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"cuda-python driver bindings not importable: {e}"
    return True, ""


def _fabric_buffer(nbytes: int):
    """A uint8 tensor over fabric-exportable VMM memory. Returns `(tensor, keepalive)`.

    Plain cudaMalloc memory — what `torch.empty` returns — exports only LEGACY node-local CUDA
    IPC handles, so cross-node `cuda_ipc` SILENTLY falls back to RDMA: measured 143 GB/s vs
    1686 GB/s, with no error and no warning (design §2.3). Cross-node NVLink requires
    CU_MEM_HANDLE_TYPE_FABRIC + gpuDirectRDMACapable. Mirrors nixlbench's `--enable_vmm` path.

    Args:
        nbytes: buffer size; rounded up to the allocation granularity.

    Returns:
        `(tensor, keepalive)` — the keepalive MUST outlive the tensor or `cuMemUnmap` tears the
        mapping down underneath it.
    """
    import cupy
    import torch
    from cuda.bindings import driver

    # The CUDA driver API needs a context current on THIS thread. The RDT warmup binds one on
    # _ray_system, not here, so cuMemCreate would fail with CUDA_ERROR_INVALID_CONTEXT.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    # Resolve the device ONCE and use it for the allocation, the access descriptor AND the tensor
    # wrap. Hardcoding 0 while wrapping with device="cuda" (i.e. current_device()) agrees only
    # because Ray gives each actor one GPU via CUDA_VISIBLE_DEVICES -- a coupling to the placement
    # policy, not to anything intrinsic. With >1 GPU per actor they diverge, and the symptom is a
    # buffer allocated and access-mapped on device 0 while torch reports another, which cupy's
    # UnownedMemory does not validate: a silent cross-device copy rather than a clean error.
    dev = torch.cuda.current_device()

    def _chk(ret, what):
        """cuda-python returns (err,) for void calls and (err, value) otherwise."""
        if ret[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what} failed: {ret[0]}")
        return ret[1] if len(ret) == 2 else None

    # Fabric handles are what a MULTI-NODE NVLink domain (GB200/MNNVL) needs to export memory
    # across nodes, and they require the IMEX daemon. Ask the driver rather than assume: on a
    # PCIe/SXM node with no /dev/nvidia-caps-imex-channels, cuMemCreate rejects a fabric request
    # with a bare CUDA_ERROR_INVALID_VALUE that names neither fabric nor IMEX. A POSIX file
    # descriptor is shareable within the node, which is all a single-node bind needs.
    fabric_ok = _chk(
        driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev),
        "cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED)",
    )

    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = (
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        if fabric_ok
        else driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev
    prop.allocFlags.gpuDirectRDMACapable = 1

    gran = _chk(
        driver.cuMemGetAllocationGranularity(
            prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        ),
        "cuMemGetAllocationGranularity",
    )
    padded = ((nbytes + gran - 1) // gran) * gran
    handle = _chk(driver.cuMemCreate(padded, prop, 0), "cuMemCreate")
    ptr = _chk(driver.cuMemAddressReserve(padded, gran, 0, 0), "cuMemAddressReserve")
    _chk(driver.cuMemMap(ptr, padded, 0, handle, 0), "cuMemMap")

    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = dev
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _chk(driver.cuMemSetAccess(ptr, padded, [access], 1), "cuMemSetAccess")

    mem = cupy.cuda.UnownedMemory(int(ptr), padded, owner=None)
    arr = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(mem, 0))
    return torch.as_tensor(arr, device=f"cuda:{dev}"), (handle, ptr, padded)


def _release_fabric(keepalive: tuple[Any, int, int]) -> None:
    """Release a fabric VMM mapping after its tensor registration and ObjectRefs are gone."""
    import torch
    from cuda.bindings import driver

    handle, ptr, padded = keepalive
    torch.cuda.synchronize()

    def _check(result: tuple, operation: str) -> None:
        if result[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed: {result[0]}")

    _check(driver.cuMemUnmap(ptr, padded), "cuMemUnmap")
    _check(driver.cuMemAddressFree(ptr, padded), "cuMemAddressFree")
    _check(driver.cuMemRelease(handle), "cuMemRelease")


class NixlTransport:
    """Per-process NIXL resources: agent, CUDA warmup, registered fabric staging buffer.

    Shared across movers because Ray's `get_tensor_transport_manager("NIXL")` is itself a
    lock-guarded per-process singleton (`ray/experimental/rdt/util.py:125`) — two movers cannot
    each own one (design §2). Owns no per-plane state beyond the staging buffer.
    """

    def __init__(self):
        self._warm = False
        # Size every mover's staging buffer should be allocated at, computed once at bind from
        # the WHOLE plan (see _Host.init_nixl). Held here rather than on a mover because bind
        # runs before any mover exists; the buffers themselves belong to the movers.
        self._reserved = 0
        # One registered fabric allocation returned by the pre-model capability probe. Recycle into
        # the real model mover when large enough; otherwise Ray deregistration + CUDA VMM teardown
        # frees it without violating one-buffer-per-live-mover ownership.
        self._spare_buffer: tuple[Any, Any] | None = None

    @classmethod
    def get(cls) -> "NixlTransport":
        """The process singleton. Idempotent; raises if the stack is unavailable."""
        global _singleton
        ok, reason = nixl_available()
        if not ok:
            # Derive the floor from MIN_RAY rather than hardcoding it: a stale literal here would
            # contradict the check above and send someone on a pointless upgrade — e.g. a user
            # whose ray already satisfies MIN_RAY, and whose real problem is a missing nixl
            # package, being told to upgrade ray. Enforced by
            # tests/test_workers_nixl.py::test_version_guidance_derives_from_min_ray.
            raise RuntimeError(
                f"NIXL transport unavailable: {reason}. Weight sync with hint='nixl' needs "
                f"ray>={'.'.join(map(str, MIN_RAY))} + nixl (BOTH the `nixl` dispatch shim and the "
                f"matching `nixl-cuXX` backend) + cuda-python + cupy in the image."
            )
        with _lock:
            if _singleton is None:
                _singleton = cls()
            return _singleton

    def warmup(self) -> None:
        """Bind a CUDA context on the RDT transfer thread.

        RDT runs transfers on the actor's `_ray_system` thread, which has no CUDA context
        bound, so without this the FIRST transfer in a fresh cluster dies with
        `NIXL_ERR_BACKEND` (design §2). Idempotent.
        """
        # Under _lock: reading self._warm outside it is a check-then-act, so two callers could
        # both observe False and both run the body. Harmless while the body is idempotent, and
        # silently wrong the day it is not. Once per process, so the lock costs nothing.
        with _lock:
            if self._warm:
                return
            import torch
            from ray.experimental.rdt.util import get_tensor_transport_manager

            torch.cuda.init()
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            get_tensor_transport_manager("NIXL").get_nixl_agent()
            self._warm = True

    def reserve(self, nbytes: int) -> None:
        """Record the staging-buffer size movers in this process should allocate.

        Called at bind with `max(bucket bytes)` over the WHOLE plan (design §2.2). A buffer is
        registered and pinned once allocated and cannot grow, so sizing from the first bucket
        rather than the plan maximum would fail on any later, larger bucket.
        """
        stale: tuple[Any, Any] | None = None
        with _lock:  # read-modify-write; same check-then-act shape as warmup()
            self._reserved = max(self._reserved, int(nbytes))
            if self._spare_buffer is not None:
                buffer, keepalive = self._spare_buffer
                if int(buffer.numel()) < self._reserved:
                    stale = self._spare_buffer
                    self._spare_buffer = None
        if stale is not None:
            buffer, keepalive = stale
            self.release_buffer(buffer, keepalive)

    @property
    def reserved(self) -> int:
        """Bind-time staging-buffer size, or 0 if bind never reserved one."""
        return self._reserved

    def new_buffer(self, nbytes: int) -> tuple:
        """Adopt a sufficient idle probe allocation or register a fresh fabric staging buffer.

        Active buffers are never shared: each mover owns its buffer, so two movers in one actor
        never write the same bytes. Sharing one live buffer would put body B's `pack()`
        on top of bytes a one-sided peer might still be reading for body A — silently, since
        `register_nixl_memory` grants addressability, not access control, and each mover's
        `_wait_free()` only knows about its own publications.

        The pre-model probe returns its idle allocation to the spare slot. The real model mover adopts
        it when large enough; once the model manifest raises the reservation above it, ``reserve``
        calls Ray's deregistration API and releases the CUDA VMM mapping. Live model movers still own
        one private buffer until process teardown; a general mover-retirement lifecycle remains future
        work.

        Args:
            nbytes: buffer size; the caller passes `max(reserved, this bucket's need)`.
        """
        with _lock:
            if self._spare_buffer is not None and int(self._spare_buffer[0].numel()) >= int(nbytes):
                buffer, keepalive = self._spare_buffer
                self._spare_buffer = None
                return buffer, keepalive

        from ray.experimental import register_nixl_memory

        buf, keep = _fabric_buffer(nbytes)
        # Registering the parent covers every view: `_add_tensor_descs` keys on
        # `untyped_storage().data_ptr()`, which views share.
        register_nixl_memory(buf)
        return buf, keep

    def recycle_buffer(self, buffer: Any, keepalive: Any) -> None:
        """Return an idle registered fabric allocation for adoption by a later mover."""
        with _lock:
            if self._spare_buffer is not None:
                raise RuntimeError("NIXL transport already holds a recycled probe buffer")
            self._spare_buffer = (buffer, keepalive)

    def release_buffer(self, buffer: Any, keepalive: Any) -> None:
        """Deregister and free one idle fabric allocation."""
        from ray.experimental import deregister_nixl_memory

        deregister_nixl_memory(buffer)
        _release_fabric(keepalive)


class NixlWeightMover:
    """Weight-plane mover over Ray RDT (design §3).

    STAGED because NIXL is one-sided: `pull_stream_from_peer` takes the same synchronous
    per-bucket branch as Copy/Ipc (`weight_transfer.py:409-421`) and never enters the COLLECTIVE
    branch, sidestepping its documented mid-stream hang where a consumer-side raise strands the
    producer inside a broadcast until the 300 s timeout.

    Publishes ONE flat tensor per bucket rather than one per parameter: `ray.put` is
    per-descriptor (~0.31 ms/tensor measured) and a real 512 MiB bucket holds 100-200 tensors
    (design §3.2). The layout rides `handle.meta["plan"]`, which is also exactly what a
    TP-sharded sink needs — so there is no separate `recv_plan` seam (design §3.4).
    """

    lifetime = "staged"
    tensors_only = True

    def __init__(self):
        # PERSISTENT across syncs: `wait_tensor_freed` keys on id(tensor), and `_wait_free`
        # must target the object published LAST sync (design §4.1).
        self._views: list = []
        # This mover's OWN staging buffer — not shared with any other mover in the process, so
        # `_wait_free()` reasoning over `self._views` covers every publication that can touch
        # these bytes. See NixlTransport.new_buffer for why, and for the lifecycle TODO.
        self._buf = None
        self._keep = None  # fabric mapping keepalive; must outlive self._buf
        self._recv = None  # receive buffer (parent allocation)
        self._target = None  # the VIEW handed to set_target_for_ref; must persist (weak refs)

    def stage(self, obj: dict) -> Any:
        """Copy a bucket into the registered staging buffer and publish it as one tensor.

        MUST run inside a Ray actor: `ray.put(_tensor_transport="nixl")` reaches
        `ray.get_runtime_context().current_actor` (rdt_manager.py:949), which raises
        "This method is only available in an actor" on the driver. Satisfied in production —
        `WeightSync.stage_bucket` runs inside `_Host` — but it means any test must too.

        Args:
            obj: `{param_name: full tensor}` for one bucket, as `bucket_full_tensors` returns.

        Returns:
            A `MoveHandle` carrying the RDT `ObjectRef` and the flat layout.
        """
        import ray

        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes, flat_plan, pack
        from cosmos_framework.posttrain.comm.transfer.mover import MoveHandle

        # MUST precede the self._views reassignment below: it waits on the PREVIOUS sync's
        # published object, and nothing else bounds the window in which a one-sided peer may
        # still be reading the buffer we are about to overwrite (design §4).
        self._wait_free()

        plan = flat_plan(obj)
        need = flat_nbytes(plan)
        transport = NixlTransport.get()
        if self._buf is None:
            # Size from the bind-time reservation (max over the WHOLE plan) so a later, larger
            # bucket still fits: the buffer is pinned once registered and cannot grow. Falling
            # back to this bucket's need covers callers that never went through bind.
            self._buf, self._keep = transport.new_buffer(max(transport.reserved, need))
        elif self._buf.numel() < need:
            raise RuntimeError(
                f"staging buffer is {self._buf.numel()} bytes but this bucket needs {need}; it is "
                f"registered and pinned for the process lifetime, so it cannot grow. Size it from "
                f"max(bucket bytes) over the WHOLE plan at bind, not the first bucket (design §2.2)."
            )
        buf = self._buf
        flat = pack(obj, plan, buf)
        if flat.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr():
            # Guards a refactor that publishes a live parameter instead of a view of the
            # registered buffer. The peer reads the source directly (no pool copy), so that
            # would expose live weights to one-sided reads while the trainer mutates them —
            # silently (design §3.3).
            raise RuntimeError(
                "staged tensor is not a view of the registered staging buffer; publishing a "
                "live parameter would expose it to one-sided reads during training"
            )
        self._views = [flat]
        return MoveHandle(kind="nixl", payload=ray.put([flat], _tensor_transport="nixl"), meta={"plan": plan})

    def prepare_fetch(self, handle: Any) -> int:
        """Allocate and persist the receive target before a timed capability transfer."""
        if handle is None or getattr(handle, "payload", None) is None:
            raise ValueError("NixlWeightMover.fetch requires a handle carrying an ObjectRef")

        import torch

        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes

        need = flat_nbytes(handle.meta["plan"])
        if self._recv is None or self._recv.numel() < need:
            # Plain torch memory on purpose: only the remotely-read producer needs fabric memory.
            self._recv = torch.empty(need, dtype=torch.uint8, device="cuda")  # [N]
            self._target = None
        if self._target is None or self._target.numel() != need:
            # set_target_for_ref stores a weak reference to this exact view object.
            self._target = self._recv[:need]  # [N]
        return need

    def fetch(self, handle: Any) -> dict:
        """Land the published bucket in a registered buffer and carve it into parameters.

        Args:
            handle: the `MoveHandle` produced by the producer's `stage`.

        Returns:
            `{param_name: tensor}` — views into this mover's receive buffer.
        """
        # Validate BEFORE importing the optional deps: a wiring bug (no handle) must report as
        # a wiring bug, not as a missing dependency on a machine that simply lacks ray.
        if handle is None or getattr(handle, "payload", None) is None:
            raise ValueError("NixlWeightMover.fetch requires a handle carrying an ObjectRef")

        import ray
        from ray.experimental import set_target_for_ref

        from cosmos_framework.posttrain.comm.transfer.flat import carve

        plan = handle.meta["plan"]
        self.prepare_fetch(handle)
        set_target_for_ref(handle.payload, [self._target])
        (flat,) = ray.get(handle.payload)
        return carve(flat, plan)

    def reserve(self, buckets: list) -> None:
        """Check the plan fits the buffer bind already registered; this mover cannot size one.

        The same-GPU Ipc mover records its single-buffer requirement here; Copy is a no-op. NIXL
        cannot size here: `init_nixl` already sized the buffer at bind from the max over the WHOLE
        plan and registered it with the fabric, and a registered buffer cannot grow (`new_buffer`).
        So the useful work left is to check the plan the peer sent is the one bind was sized for --
        staging a bucket bigger than the buffer would write past it.

        Args:
            buckets: `list[list[(name, shape, dtype)]]` -- the ordered bucket plan being installed.
        """
        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

        need = max(flat_nbytes_for_manifest(bucket) for bucket in buckets)
        have = NixlTransport.get().reserved
        if need > have:
            raise ValueError(
                f"bucket plan needs a {need} B staging buffer but bind registered {have} B; a "
                "registered NIXL buffer cannot grow, so init_nixl must see this same plan"
            )

    def release_staged(self) -> None:
        """No-op: a registered NIXL buffer is pinned for the process lifetime.

        `register_nixl_memory` bumps a reference count that is never dropped, so there is nothing
        to hand back -- see the lifecycle TODO on `NixlTransport.new_buffer`. Releasing is what
        this method means for Ipc, whose buffer is ordinary memory.
        """

    def close(self) -> None:
        """Registered fabric buffers remain pinned for the process lifetime, as in release_staged.

        The registration API does not balance its owning reference count; dropping the Python keepalives is
        not a supported retirement operation. This mover therefore has no separate terminal deregistration.
        """

    def consumer_done(self) -> None:
        """Drain the consumer's stream before the receive buffer is reused.

        `WeightSync.apply_bucket_state` calls this right after `sink(state)`. That sink loads
        from the tensors `fetch` returned, and `carve` returns VIEWS into `self._recv` — so the
        copies it enqueues READ from the receive buffer and are asynchronous. The next bucket's
        `ray.get` then lands a one-sided RDMA/NVLink write into that same buffer, and that write
        is not ordered against this process's torch stream.

        Ray does not close this for us, which is why the hook has to be implemented here:
          - `wait_fetch_complete` polls NIXL's `check_xfer_state` until DONE — transfer
            completion, not a stream sync;
          - the only `torch.cuda.synchronize` in ray.experimental.rdt is on the SEND path,
            before memory registration;
          - by contrast `cuda_ipc_transport` orders its transfers with CUDA events
            (`record_event` / `wait_event`), so Ray treats stream ordering as required for a
            transport that hands out memory — the NIXL path just leaves it to the caller.

        Device-wide rather than `current_stream()`, matching `IpcMover.consumer_done`: a sink
        may enqueue on a stream of its own (a vLLM engine worker does), and a per-bucket sync is
        negligible against a multi-millisecond transfer.
        """
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def recycle_staging_buffer(self) -> None:
        """Return this mover's idle probe buffer to the process transport for model-sync reuse."""
        if self._buf is None:
            return
        self._wait_free()
        NixlTransport.get().recycle_buffer(self._buf, self._keep)
        self._buf = None
        self._keep = None
        self._views = []

    def _wait_free(self) -> None:
        """Block until the previous sync's `ObjectRef`s are gone.

        NIXL is one-sided and the sender is passive — `send_multiple_tensors` raises
        `NotImplementedError` and the producer never learns its data was read — so overwriting
        the staging buffer mid-read corrupts weights with no error on either side (design §4.3).
        Under the sequential stream this is nearly always already satisfied (~1 ms measured).
        """
        if not self._views:
            return
        from ray.experimental import wait_tensor_freed

        for v in self._views:
            wait_tensor_freed(v, timeout=60)


def _register() -> None:
    """Publish this mover under hint='nixl'. Called at import so that importing this module is
    what makes the hint resolvable — `comm/` cannot import it (Ray-free constraint)."""
    from cosmos_framework.posttrain.comm.transfer.movers import register_weight_mover

    register_weight_mover("nixl", NixlWeightMover)


_register()
