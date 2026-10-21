# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ray backend — validates the Ray seam (DESIGN §1.1, §2.1, §4.2; §3).

Each ActorGroup bound by `bind_device_groups` becomes an R×K mesh with one `_Host` Ray actor per
cell. WorkerGroups listed in the same ActorGroup share the body-construction closure inside each
cell's single OS process; different ActorGroups use different actor processes and may share or split
physical GPUs through placement bundles. The Ray machinery — actor, `.remote()`, ObjectRef → Handle
— is exercised with no change to the control loop.

Each actor runs the metadata/bulk seam (DESIGN §3.2/§3.4):
- egress: `pack`s tensor-bearing Trajectory/ObsBundle results into its local BulkStore,
  returning only BulkRef metadata — so the controller round-trip carries metadata, never
  tensor bytes (#7).
- ingress: `materialize`s BulkRef args ONLY for WGs in `materialize_names` (those that
  compute on bulk — rollout/trainer); reward-on-text / buffer let BulkRefs flow through,
  so bulk moves once where needed (§3.4) and the buffer holds only metadata (§5).
- transport (§3.2 backend / §3.3): `RayStoreMover` — local ref ⇒ zero-copy
  LocalStoreMover; remote ref ⇒ one-sided pull from the producer actor via Ray
  (peer-to-peer A→B, the controller never touches the bulk). The {store_id → actor}
  peer map (the §3.2 connection registry) is built eagerly by `bind` and pushed via
  `set_peers`, so cross-actor (tier-2) edges get the same metadata/bulk treatment.

Placement groups, per-cell actors, peer-to-peer ObjectRef routing, IPC, and NIXL are implemented.
Remaining substrate work is real async overlap and broader data-plane UCX/NCCL transports.
"""

from __future__ import annotations

import atexit
import os
import sys
import time
import uuid
import weakref
from typing import Any, Callable

import ray
import torch

from cosmos_framework.posttrain.comm.bulk import BulkStore, has_local_store, store_for
from cosmos_framework.posttrain.comm.codec import collect_bulk_refs, materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.roles.core.dispatch import resolve_fn
from cosmos_framework.posttrain.utils.profile import ProfileRecord
from cosmos_framework.posttrain.workers.data_pool import (
    ConsumerDataPool,
    PoolSlice,
    ProducerDataPool,
    iter_pool_slices,
)
from cosmos_framework.posttrain.workers.worker_group import Dispatched, Handle, LocalPeerClient, LocalPeerRef

# In-actor accessor for the injected placement dims (the set_mesh seam), read by the
# model build (build_backbone) WITHOUT threading pdims through the no-arg setup_fn()
# contract. run_setup sets this to the actor's _pdims for the duration of the setup call
# and resets it in a finally — per-process safe (each Ray actor is its own process, so
# there is no cross-actor bleed). None outside an actor's setup window ⇒ single-device.
_ACTOR_PDIMS = None
WEIGHT_FANOUT_TIMEOUT_S = 60.0
_CUDA_UUID_V2_REQUIREMENT = "CUDA 11.4+ Driver API (Linux R470.42.01+)"
#: Set in an actor whose peer GPUs are exposed, so `gpu_identity` accepts more than one visible device.
PEER_GPUS_ENV = "POSTTRAIN_PEER_VISIBLE_GPUS"


def _prepend_actor_pythonpath(paths: tuple[str, ...]) -> None:
    """Apply the driver-frozen import path before setup imports external packages."""
    normalized = tuple(dict.fromkeys(os.path.abspath(path or os.curdir) for path in paths))
    sys.path[:] = [*normalized, *(entry for entry in sys.path if os.path.abspath(entry or os.curdir) not in normalized)]


def _expose_peer_gpus() -> None:
    """Make the node's other GPUs visible after the one Ray assigned, which stays ordinal 0.

    vLLM's custom all-reduce opens every TP peer's buffers over CUDA IPC, so their devices must be
    visible, and Ray hides them. Keeping the assigned GPU first leaves ``cuda:0`` (the device
    `init_distributed` selects, the one a colocated trainer's IPC handles name, and the one
    `gpu_identity` proves) the actor's own GPU. NVML counts the node's GPUs without creating the
    CUDA context that would freeze visibility, so this must run before CUDA is initialized.
    Every rank then reports device 0, which PyTorch symmetric memory refuses, so FlashInfer's
    all-reduce + RMSNorm fusion stays off while the custom all-reduce itself runs.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assigned = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(assigned) != 1 or not assigned[0].isdigit():
        raise RuntimeError(
            f"{PEER_GPUS_ENV}=1 needs exactly one Ray-assigned GPU index, got CUDA_VISIBLE_DEVICES={visible!r}"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(f"{PEER_GPUS_ENV}=1 must take effect before this process initializes CUDA")
    import pynvml

    pynvml.nvmlInit()
    try:
        count = int(pynvml.nvmlDeviceGetCount())
    finally:
        pynvml.nvmlShutdown()
    own = int(assigned[0])
    if own >= count:
        raise RuntimeError(f"Ray-assigned GPU {own} is not one of this node's {count} GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in (own, *(i for i in range(count) if i != own)))
    os.environ[PEER_GPUS_ENV] = "1"


def _require_cuda_uuid_v2(cuda: Any) -> Any:
    """Return the MIG-safe UUID entry point or raise an actionable driver requirement."""
    get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", None)
    if get_uuid is None:
        raise RuntimeError(
            f"GPU identity proof requires CUDA driver symbol cuDeviceGetUuid_v2 from the "
            f"{_CUDA_UUID_V2_REQUIREMENT}; the legacy UUID call cannot distinguish MIG compute instances"
        )
    return get_uuid


def current_pdims():
    """Return the ParallelDims injected into THIS actor's current setup, or None.

    Set by _Host.run_setup around the setup_fn() call so the model build can read the
    injected device mesh without changing the setup contract; None when called outside a
    setup window or in a plain (non-Ray) process ⇒ the single-device / torchrun path."""
    return _ACTOR_PDIMS


class PeerClient:
    """Ray-free callable client bound to one producer cell's ``_Host`` actor (DESIGN
    §1.2 body-mode (b): outbound-call). A role body that must call a *peer* WG inline
    during its own step (e.g. a simulator calling the policy per env step) receives one
    of these in place of the peer ``WorkerGroup`` arg — see ``GridBackend._align``.

    ``client.<method>(*args, **kw)`` invokes ``<method>`` on the paired producer body
    and returns the resolved value (``ray.get`` under the hood). The body sees a plain
    synchronous callable — no Ray in the body, no knowledge of the producer's placement.
    Serializable (a Ray actor handle + two strings), so it ships into the consumer actor.
    """

    __slots__ = ("_host", "_wg_name")

    def __init__(self, host: Any, wg_name: str):
        self._host = host  # producer cell's _Host actor handle
        self._wg_name = wg_name  # the body key inside that actor

    def call_async(self, method: str, *args: Any, **kw: Any) -> "_PeerFuture":
        """Non-blocking peer call → future with ``.result()``. Needed by co-scheduled collective
        edges (the COLLECTIVE branch of ``pull_stream_from_peer``): the caller must enter ITS side
        of the collective before resolving the peer's, so a synchronous ``__getattr__`` call would
        deadlock. A real method — normal attribute lookup finds it, so ``__getattr__`` never routes
        the name to the peer body."""
        return _PeerFuture(self._host.invoke.remote(self._wg_name, method, args, kw))

    def __getattr__(self, method: str) -> Callable[..., Any]:
        # Let dunder probes (pickle's __getstate__/__setstate__/__reduce_ex__ lookups,
        # done via getattr on the instance) fall through to the defaults — otherwise
        # they'd resolve to _call and corrupt serialization into the consumer actor.
        if method.startswith("__") and method.endswith("__"):
            raise AttributeError(method)

        def _call(*args: Any, **kw: Any) -> Any:
            future = self.call_async(method, *args, **kw)
            # A retained remote exception must not keep the caller's original bulk arguments
            # alive through this adapter's traceback after its staging has been retired.
            args, kw = (), {}
            return future.result()

        return _call


class _BoundPeerClient(PeerClient):
    """Actor-local codec context for an otherwise serializable peer endpoint."""

    __slots__ = ("_owner", "_materialize")

    def __init__(self, peer: PeerClient, owner: _HostImpl, materialize: bool) -> None:
        super().__init__(peer._host, peer._wg_name)
        self._owner, self._materialize = owner, materialize

    def __reduce__(self):
        # Forwarding a peer transfers its endpoint, never the caller's stores or role bodies.
        return PeerClient, (self._host, self._wg_name)

    def call_async(self, method: str, *args: Any, **kw: Any) -> _PeerFuture:
        owned_refs = set()

        def pack(value: Any) -> Any:
            if not self._owner._codec:
                return value
            existing = set(collect_bulk_refs(value))
            packed = pack_payload(value, self._owner._store, self._owner._threshold)
            owned_refs.update(set(collect_bulk_refs(packed)) - existing)
            return packed

        def discard() -> None:
            if owned_refs:
                self._owner._store.discard_unconsumed(owned_refs)

        try:
            args = tuple(pack(value) for value in args)
            kw = {name: pack(value) for name, value in kw.items()}
            ref = self._host.invoke.remote(self._wg_name, method, args, kw)
        except BaseException:
            discard()
            raise

        def get_result(ref: Any) -> Any:
            try:
                value = ray.get(ref)
                return materialize_payload(value, self._owner._transport) if self._materialize else value
            except BaseException:
                # Inputs consumed by the peer are already gone. Only this call's new staging
                # remains ours to retire; forwarded refs may still belong to another reader.
                discard()
                raise

        return _PeerFuture(ref, getter=get_result)


class RoutedWeightPeerClient:
    """Bounded concurrent calls to all rollout actors routed from one policy actor.

    A single NIXL ``WeightPayload`` may have multiple destination actors. Submitting every target
    before waiting both avoids serial fan-out and lets all vLLM TP lanes enter any engine-local load
    collective together. The producer still treats this adapter as one logical peer and therefore
    stages the bucket exactly once.

    A timeout bounds the policy actor's wait but cannot roll back already-mutated rollout tensors;
    any failure poisons the sync and prevents version publication.
    """

    __slots__ = ("_targets", "_wg_name", "_timeout")

    def __init__(
        self,
        targets: tuple[tuple[tuple[int, int], Any], ...],
        wg_name: str,
        timeout: float = WEIGHT_FANOUT_TIMEOUT_S,
    ) -> None:
        if len(targets) < 2:
            raise ValueError(f"RoutedWeightPeerClient requires at least two targets, got {len(targets)}")
        self._targets = targets
        self._wg_name = wg_name
        self._timeout = float(timeout)

    def _call_all(self, method: str, *args: Any, **kwargs: Any) -> None:
        refs = {host.invoke.remote(self._wg_name, method, args, kwargs): cell for cell, host in self._targets}
        done, pending = ray.wait(list(refs), num_returns=len(refs), timeout=self._timeout)
        failures: list[tuple[tuple[int, int], BaseException]] = []
        for ref in done:
            try:
                ray.get(ref)
            except BaseException as error:  # noqa: BLE001 -- drain every completed target
                failures.append((refs[ref], error))
        for ref in pending:
            failures.append(
                (
                    refs[ref],
                    TimeoutError(f"{method} did not finish within {self._timeout:g}s"),
                )
            )
        if failures:
            details = "; ".join(f"rollout cell {cell}: {type(error).__name__}: {error}" for cell, error in failures)
            raise RuntimeError(
                f"routed weight peer {method} failed on {len(failures)} target(s): {details}"
            ) from failures[0][1]

    def __getattr__(self, method: str) -> Callable[..., None]:
        if method.startswith("__") and method.endswith("__"):
            raise AttributeError(method)

        def _call(*args: Any, **kwargs: Any) -> None:
            self._call_all(method, *args, **kwargs)

        return _call


class _PeerFuture:
    """Future-like handle from ``PeerClient.call_async``: ``.result()`` resolves the remote call
    (``ray.get``). Ray-free surface for the body, mirroring PeerClient itself."""

    __slots__ = ("_ref", "_getter", "_resolved", "_value")

    def __init__(self, ref: Any, getter: Callable[[Any], Any] = ray.get):
        self._ref = ref
        self._getter = getter
        self._resolved = False
        self._value = None

    def result(self) -> Any:
        if not self._resolved:
            self._value = self._getter(self._ref)
            self._resolved = True
            self._ref = None
            self._getter = None
        return self._value


class RayStoreMover:
    """Mover backend over Ray (DESIGN §3.2 backend / §3.3): local refs pull
    zero-copy from the in-process store; remote refs pull one-sided from the producer
    actor, resolved through the `peers` connection registry. Implements the comm
    `Mover` protocol (duck-typed `fetch` + optional `prefetch`); kept here (not in
    comm/) so comm stays Ray-free."""

    lifetime = "staged"
    tensors_only = False

    def __init__(
        self,
        peers: dict,
        consumer_pool: ConsumerDataPool | None = None,
    ):
        self._peers = peers  # {store_id: _Host actor handle}
        self._cache: dict = {}  # ref → bulk, filled by prefetch (batched)
        self._consumer_pool = consumer_pool
        self._local_producer_pool: ProducerDataPool | None = None
        self._pool_tensor_sources: dict[int, tuple[weakref.ReferenceType[torch.Tensor], PoolSlice]] = {}

    def set_local_producer_pool(self, pool: ProducerDataPool) -> None:
        """Install this actor's lazily created source pool."""
        self._local_producer_pool = pool

    def set_consumer_pool(self, pool: ConsumerDataPool) -> None:
        """Install this actor's receive pool after deferred role setup."""
        self._consumer_pool = pool

    def clear_pool_tensor_sources(self) -> None:
        """Clear provenance from the previous invocation."""
        self._pool_tensor_sources.clear()

    def source_for_tensor(self, tensor: torch.Tensor) -> PoolSlice | None:
        """Return the source descriptor when this exact tensor was fetched."""
        key = id(tensor)
        tracked = self._pool_tensor_sources.get(key)
        if tracked is None:
            return None
        tensor_ref, slice_ = tracked
        if tensor_ref() is tensor:
            return slice_
        self._pool_tensor_sources.pop(key, None)
        return None

    def _record_pool_tensor(self, tensor: torch.Tensor, slice_: PoolSlice) -> None:
        self._pool_tensor_sources[id(tensor)] = (weakref.ref(tensor), slice_)

    def prefetch(self, refs, readers: int = 1) -> None:
        """Batch-pull every REMOTE ref in `refs` into `self._cache`, one RPC per producer actor.

        Args:
            refs: BulkRefs to pull; local-store refs are skipped (stay zero-copy, no RPC).
            readers: reader claims (Task A5) threaded to each producer's refcounted `BulkStore.fetch`,
                so a batch shared across consumer cells pops only once every reader has pulled (§3.4).
        """
        # Batch all REMOTE refs into ONE fetch per producer actor (local refs stay
        # zero-copy, no RPC). Turns a per-field pull into one RPC per consumer (§3.4).
        # `readers` (Task A5) threads through to the producer's refcounted BulkStore.fetch
        # so a batch shared by multiple consumer cells pops only once every reader has pulled.
        by_owner: dict[str, list] = {}
        for r in refs:
            if not has_local_store(r):
                by_owner.setdefault(r.store_id, []).append(r)
        for sid, group in by_owner.items():
            for r, v in zip(group, ray.get(self._peers[sid].fetch_bulk_many.remote(group, readers=readers))):
                self._cache[r] = v

    def fetch(self, ref, readers: int = 1) -> Any:
        """Resolve one `ref`: local ⇒ zero-copy store read, prefetched ⇒ pop cache, else single remote pull.

        Args:
            ref: the BulkRef to resolve.
            readers: reader claims threaded to the producer's refcounted `BulkStore.fetch`.

        Returns:
            The materialized bulk value for `ref`.
        """
        if has_local_store(ref):
            return store_for(ref).fetch(ref, readers=readers)  # same process ⇒ zero-copy
        if ref in self._cache:
            return self._cache.pop(ref)  # prefetched (batched)
        return ray.get(self._peers[ref.store_id].fetch_bulk.remote(ref, readers=readers))  # fallback: single pull

    def fetch_pool_slice(self, slice_: PoolSlice) -> torch.Tensor:
        """Return a local source view or pull one tensor through the NIXL target cache."""
        producer_pool = self._local_producer_pool
        if producer_pool is not None and slice_.owner_id == producer_pool.owner_id:
            tensor = producer_pool.view_tensor(slice_)
            self._record_pool_tensor(tensor, slice_)
            return tensor
        if self._consumer_pool is None:
            raise RuntimeError("received a PoolSlice but this actor has no configured consumer data pool")
        self._consumer_pool.warmup()
        tensor, _cache_hit = self._consumer_pool.fetch_tensor(slice_)
        self._record_pool_tensor(tensor, slice_)
        return tensor


def _resolve_dispatched(
    d: Dispatched,
    transport: Any,
    materialize: bool,
    releaser: Callable[[list, int], None] | None = None,
    dp_rank: int = 0,
    dp_world: int = 1,
) -> Any:
    """Actor-side execution of a `Dispatched` descriptor (Task A2/A4 build it; Task A5 runs
    it): resolve THIS arg's own producer refs, materialize bulk if the consuming WG needs it
    (readers threaded through so a fan-out edge's refcounted `BulkStore` entry pops only once
    every reader has pulled — DESIGN §2.4), then apply the reduce/split/one adaptation so the
    body signature never sees a list/shard, only the adapted value.

    Module-level (not a `_Host` method) so it needs no actor/Ray runtime to test — a plain
    fake transport + real `BulkStore` exercises it directly (see
    tests/test_workers_ray_adaptation.py).

    `d.refs` are `ray.ObjectRef`s when this descriptor came over the wire from `GridBackend`
    (built from `CollectiveHandle.ref_at`); tolerate already-resolved plain values too (the
    local/test path, and the CPU helper tests below never construct real Ray refs).

    `d.on_metadata` (Task A6, DESIGN §2.3/§2.4 opt-in fast path): when True, `d.op == "split"`,
    and this call actually materializes (the consuming WG computes on bulk), the split fn runs
    on the PACKED payload — the metadata skeleton, BulkRefs still attached — BEFORE any fetch,
    instead of after. Every BulkRef present in the full skeleton but absent from the kept slice
    is "dropped": it is never fetched; its producer-side refcount is decremented instead via
    `releaser(refs, readers)` (§2.4 release protocol). Only the kept slice is then materialized
    (readers=d.readers, same refcounted pop as the default path). Net effect: an on_metadata
    split edge moves exactly the kept fraction of bulk, never the whole producer payload.

    `releaser` is the seam by which this module-level fn reaches actor state (the local
    `BulkStore` + the peers/connection registry) without importing `_Host` or coupling to its
    internals — `_Host.invoke` builds a small closure over `self._store`/`self._transport`'s
    peers and passes it down; the CPU helper tests below pass a plain fn over a bare
    `BulkStore` instead (no Ray, no `_Host`). Not needed (may stay `None`) unless an
    on_metadata split actually drops a ref through this call — a drop with no releaser is a
    caller bug, so it raises rather than silently leaking the entry.

    When `on_metadata=False` (the default), or `d.op != "split"`, or this call doesn't
    materialize, behavior is byte-identical to Task A5 — no split-before-materialize, no
    release, `releaser` unused.

    `d.select` (Task 7 plumbs it, Task 8 — this task — executes it; DESIGN §3.4, spec
    2026-08-12 §3.4/§9.5): the mini/micro/dp ingress-selection predicate (`select_chunk`),
    carried alongside `d.select_kw` (the resolved `mini`/`micro`/`n_mini`/`n_micro` control
    kwargs). ORDER IS THE WHOLE POINT and mirrors the on_metadata branch's intent but is a
    DISTINCT mechanism (that branch requires `op == "split"`; this one applies to `one`/`reduce`,
    which is what the `("trainer", "train", "batch")` edge actually is):
    resolve refs -> reduce/one on the METADATA skeleton (BulkRefs still attached, nothing
    fetched) -> `select_chunk` -> materialize ONLY the kept slice. Unlike the on_metadata split
    branch, this NEVER releases: nothing here is orphaned — an item the dp predicate drops is
    pulled by a DIFFERENT rank, and one the micro predicate drops is pulled by a LATER call on
    this same rank (the dp/micro predicates just partition the same skeleton differently across
    calls). Only the refcount differs, which is why `d.readers` (not a release) carries the
    correction (§9.5: `readers = K_c / dp_world`, not `K_c`).

    `dp_rank`/`dp_world` are resolved by the CALLER (actor-side — `_Host.invoke`, which is where
    `ParallelDims` lives; the driver knows the cell `(r, k)` but does not own the model-mesh
    factorization). The default `(0, 1)` here is this FUNCTION's no-arg behavior
    (used directly by callers/tests that want an explicit no-op) — `_Host.invoke` (Task 9) passes
    the actor's REAL `_dp_coords(self._pdims)` on every call instead, since `dp_group_sharded` (the
    objective wrapper that used to own the dp axis on this path) is deleted and `select_chunk` is now
    the sole owner end to end.

    Args:
        d: the `Dispatched` descriptor — carries producer `refs`, `op` (`one`/`reduce`/`split`),
            adaptation `fn`, split `index`/`n`, `readers`, the `on_metadata` opt-in flag, and the
            optional `select`/`select_kw` ingress-selection pair.
        transport: the `Mover` (`RayStoreMover`/`LocalStoreMover`) used to materialize BulkRefs.
        materialize: True when the consuming WG computes on bulk ⇒ fetch payloads; else leave packed.
        releaser: `(dropped_refs, readers) -> None` release hook (§2.4); required only when an
            `on_metadata` split actually drops refs, else may stay `None`. Never called by the
            `d.select` branch (it never drops/releases — see above).
        dp_rank: this rank's dp-shard coordinate, consulted only when `d.select is not None`.
        dp_world: the dp-shard extent, consulted only when `d.select is not None`. `<= 1` makes
            `select_chunk`'s dp interleave a no-op (matches `select_chunk`'s own contract).

    Returns:
        The adapted value handed to the body arg — the single payload (`one`), the `fn`-reduced
        result (`reduce`), the kept slice (`split`), or (`d.select` case) the selected+materialized
        chunk.
    """
    payloads = [ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref for ref in d.refs]
    if d.select is not None and materialize:
        # Reduce on the METADATA skeleton first (BulkRefs attached, nothing pulled), then select,
        # then materialize only the kept share. Unlike the split branch below it fires for
        # "reduce"/"one" and NEVER releases — nothing is orphaned: an item the dp predicate drops is
        # pulled by another rank, one the micro predicate drops by a later call here. Only the
        # refcount differs, and `d.readers` already carries it.
        full = payloads[0] if len(payloads) == 1 else resolve_fn(d.fn)(payloads)
        kept = resolve_fn(d.select)(full, dp_rank=dp_rank, dp_world=dp_world, **d.select_kw)
        return materialize_payload(kept, transport, readers=d.readers)
    if d.op == "split" and d.on_metadata and materialize:
        if len(payloads) != 1:
            raise ValueError(f"on_metadata split expects exactly 1 packed payload, got {len(payloads)}")
        full = payloads[0]
        kept = resolve_fn(d.fn)(full, d.index, d.n)
        kept_refs = set(collect_bulk_refs(kept))
        dropped = [r for r in collect_bulk_refs(full) if r not in kept_refs]
        if dropped:
            if releaser is None:
                raise ValueError("on_metadata split dropped BulkRefs but no releaser was supplied")
            releaser(dropped, d.readers)
        return materialize_payload(kept, transport, readers=d.readers)
    if materialize:
        payloads = [materialize_payload(p, transport, readers=d.readers) for p in payloads]
    if d.op == "one":
        if len(payloads) != 1:
            raise ValueError(f"Dispatched op='one' expects exactly 1 payload, got {len(payloads)}")
        return payloads[0]
    if d.op == "reduce":
        return resolve_fn(d.fn)(payloads)
    if d.op == "split":
        return resolve_fn(d.fn)(payloads[0], d.index, d.n)
    raise ValueError(f"unknown Dispatched op {d.op!r} (expected one|reduce|split)")


# max_concurrency=1 PINS the "_default" group to concurrency 1 (named-group sizes are
# independent, so "bulk" stays 2 — verified on-node: bulk runs concurrent with a blocked
# default method). The default group MUST stay single-writer: invoke / set_* / the
# _pending_releases sweep assume it. Pinning guards against a future async-def method on
# this actor flipping Ray's default-group concurrency (async actors default to >1).
class _HostImpl:
    """A single actor process hosting one ActorGroup's role bodies (§4.2 same-process grouping).

    Concurrency model (deadlock fix 2026-07-22): the actor declares one named concurrency
    group, "bulk" (max_concurrency 2). Ray leaves every UNDECLARED method — `invoke`,
    `run_setup`, the set_* injectors, init_distributed, … — in the implicit "_default" group,
    which is pinned to concurrency 1 (max_concurrency=1 above): the
    control loop stays single-threaded exactly as before. Only the bulk-plane pull methods
    (`fetch_bulk`, `fetch_bulk_many`, `release_bulk`) are routed to the "bulk" group so they
    run CONCURRENTLY with a body that is mid-`invoke`.

    Why this is required (not an optimization): a lower-rank producer's bulk lives in ONE
    actor's BulkStore; when that actor's own body is blocked inside a collective (e.g. an FSDP
    forward all-gather) waiting on its SPMD peers, those peers issue `fetch_bulk` RPCs back to
    it to pull the shared payload. With a single-threaded actor the RPC queues behind the
    blocked `invoke` → mutual wait → deadlock (bisected regression: the ranks=1 data shape at
    b75a2b7e07 reintroduced this; see tests/helpers/hf_fsdp_grpo.py's grpo_loop docstring). The
    "bulk" group serves those pulls off a second thread, so a collective-blocked peer is always
    served. Consequence: BulkStore is now touched from >1 thread ⇒ it carries its own lock
    (comm/bulk.py). Actor state touched only from `_default` methods (`_pending_releases` sweep
    in `_sweep_pending_releases`, called from `invoke` + `_release_dropped`; the peers/pairing
    registries) stays single-threaded — unchanged."""

    def __init__(self, pythonpath: tuple[str, ...] = ()):
        _prepend_actor_pythonpath(pythonpath)
        self._bodies: dict[str, Any] = {}
        self._codec = False  # metadata/bulk seam off unless run_setup enables it
        self._store: BulkStore | None = None
        self._transport: Any = LocalStoreMover()  # upgraded to RayStoreMover by set_peers
        self._threshold = 0
        self._materialize: set[str] = set()  # WG names that pull bulk on ingress (§3.4)
        self._data_pool_config: dict[str, int] | None = None
        self._producer_pool: ProducerDataPool | None = None
        self._consumer_pool: ConsumerDataPool | None = None
        self._pdims: Any = None  # model-owned ParallelDims injected by set_mesh (None ⇒ single device)
        self._weight_comm = None  # set by init_weight_plane; the cross-role side group
        self._nixl_probe_source: Any = None  # keeps the pre-model fabric publication alive
        self._nixl_probe_mover: Any = None  # producer or consumer mover backing the capability probe
        self._pending_releases: list = []  # fire-and-forget release_bulk ObjectRefs awaiting a sweep (Task A6 fix)
        # Final blocking drain on actor-process exit: the per-invoke sweep can't cover the LAST batch
        # of a run (no next op), so register a teardown drain that surfaces a last-step release failure.
        atexit.register(self.drain_pending_releases)
        self._peers: dict = {}  # {store_id: _Host actor} connection registry (set_peers); cross-actor bulk pulls

    def run_setup(
        self,
        setup_fn,
        codec: bool = False,
        threshold: int = 0,
        materialize_names: tuple = (),
        store_id: str | None = None,
        replicas: int = 1,
        ranks: int = 1,
        dist_scope: str = "replica",
        replica_index: int = 0,
        rank_index: int = 0,
        model_parallelism: tuple[int, int] | None = None,
        data_pool_config: dict[str, int] | None = None,
    ) -> list:
        # Pass 2 has already formed this execution replica's K-rank default process group. vLLM
        # reuses it directly as one engine's TP world. A trainer declaration additionally supplies
        # the model-owned `(dp_replicate, dp_shard)` factorization of that SAME K-rank world; policy
        # placement R is fixed at one and never becomes a DeviceMesh axis.
        import torch.distributed as dist

        self._pdims = None
        if dist.is_initialized() and (ranks > 1 or (dist_scope == "grid" and replicas > 1)):
            from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

            world = dist.get_world_size()
            if model_parallelism is not None:
                if replicas != 1 or dist_scope != "replica":
                    raise ValueError(
                        "trainer model_parallelism requires one policy execution replica and a "
                        f"per-replica world, got replicas={replicas}, dist_scope={dist_scope!r}"
                    )
                dp_replicate, dp_shard = model_parallelism
                if dp_replicate * dp_shard != world:
                    raise ValueError(
                        f"trainer model_parallelism dp_replicate={dp_replicate} * dp_shard={dp_shard} "
                        f"must equal the live policy world size {world}"
                    )
                self.set_mesh(
                    ParallelDims(
                        world_size=world,
                        dp_replicate=dp_replicate,
                        dp_shard=dp_shard,
                    )
                )
            else:
                # Non-trainer bodies do not consume this mesh, but publishing a pure-FSDP view keeps
                # direct setup/current_pdims callers backward compatible without inventing HSDP.
                self.set_mesh(ParallelDims(world_size=world, dp_replicate=1, dp_shard=world))

            # Warm only the policy model world. vLLM owns initialization of its TP communicator.
            if model_parallelism is not None:
                import torch as _torch

                _warm = _torch.zeros(1, device="cuda" if _torch.cuda.is_available() else "cpu")  # [1]
                dist.all_reduce(_warm)  # [1]
        # Construct ALL role bodies inside the actor (so trainer & rollout share one
        # in-actor model). setup_fn() -> {wg_name: body}. DESIGN §4.2 tier-3 / §9.
        # Publish the injected model dims to the module-global accessor for the
        # duration of the build so the model build (build_backbone -> current_pdims())
        # reads the live mesh WITHOUT changing setup_fn's no-arg contract. Per-process
        # safe; reset in finally so a failed/partial setup never leaks stale pdims.
        #
        # Publish through an EXPLICIT import, not globals(). Ray ships this actor class BY VALUE
        # (cloudpickle), which rebuilds a module namespace carrying the same `__name__` but a
        # DIFFERENT dict. `globals()` here resolved to that reconstructed copy, while a body doing
        # `from ...workers.ray import current_pdims` read the real sys.modules entry — two distinct
        # dicts in ONE process, verified by their differing id(). So the publication was invisible
        # and `current_pdims()` returned None for every Ray-backed setup since it was written. The
        # mesh still reached the model only because every caller had an inference fallback behind
        # this accessor; the documented primary path never once fired.
        # Publish this cell's PLACEMENT COORDINATES for the duration of the build. A role body that
        # owns per-replica state -- a dataset cursor above all -- must know WHICH replica it is, or
        # every replica reads the same rows and the data-parallel axis carries duplicates: R times
        # the compute, one replica's worth of data, and gradients that average copies of themselves.
        # setup_fn() is no-arg by contract, so coordinates use a scoped environment channel while
        # the model mesh uses the explicit current_pdims accessor above.
        import os as _os

        from cosmos_framework.posttrain.config.build import REPLICA_COUNT_ENV, REPLICA_INDEX_ENV
        from cosmos_framework.posttrain.workers import ray as _ray_mod

        _os.environ[REPLICA_INDEX_ENV] = str(replica_index)
        _os.environ[REPLICA_COUNT_ENV] = str(replicas)
        _ray_mod._ACTOR_PDIMS = self._pdims
        try:
            self._bodies = setup_fn()
        finally:
            _ray_mod._ACTOR_PDIMS = None
        self._codec = codec
        self._threshold = threshold
        self._materialize = set(materialize_names)
        self._data_pool_config = dict(data_pool_config) if data_pool_config is not None else None
        if self._data_pool_config is not None:
            self._consumer_pool = ConsumerDataPool(
                cuda_cache_bytes=self._data_pool_config["cuda_cache_bytes"],
                cpu_cache_bytes=self._data_pool_config["cpu_cache_bytes"],
                page_bytes=self._data_pool_config["page_bytes"],
            )
            if isinstance(self._transport, RayStoreMover):
                self._transport.set_consumer_pool(self._consumer_pool)
        if codec:
            # store_id is bind-assigned + globally unique (the per-process auto counter
            # would collide across actors — every actor's first store would be "store0").
            self._store = BulkStore(store_id)  # producer-side bulk residence (§3.4)
        return list(self._bodies.keys())

    def set_peers(self, peers: dict) -> None:
        """Install the connection registry, upgrade transport to `RayStoreMover`, and (re)build any
        deferred weight-pairing lead lanes now that store_ids resolve.

        Args:
            peers: `{store_id: _Host actor handle}` connection registry (§3.2 layer 5), eager at bind.
        """
        # connection registry (§3.2 layer 5): store_id → owning actor, eager at bind.
        self._peers = peers
        self._transport = RayStoreMover(peers, self._consumer_pool)

    def release_all_pool_leases(self) -> tuple[str, ...]:
        """Release this actor's published leases at the synchronous step barrier."""
        if self._producer_pool is None:
            return ()
        return self._producer_pool.release_all_leases()

    def set_weight_backend(self, spec: dict, body_names: tuple[str, ...] | None = None) -> None:
        # Attach the weight-sync backend (DESIGN §8.2 #4) only to the declared WEIGHT endpoint
        # bodies. ``body_names=None`` retains the legacy direct-test behavior (all model owners).
        # spec is either {"relationship": ...} — resolved via the §3.3 resolver,
        # the single source of truth used by resolved grouping — or {"kind": ...} for legacy/direct
        # mover selection. Picked from ActorGroup/DeviceGroup placement at bind; the push/apply seam
        # then rides dp_aligned. No-model
        # bodies (env/reward/buffer) skip. An optional {"hint": ...} selects a REGISTERED transport
        # (register_weight_mover) over the relationship default — cross_device + "nixl" picks the
        # NIXL mover while the default stays NCCL.
        from cosmos_framework.posttrain.comm.transfer.movers import resolve_mover
        from cosmos_framework.posttrain.comm.weight_transfer import (
            WeightSync,
            make_weight_backend,
            resolve_weight_mapper,
        )

        bucket_bytes = spec.get("bucket_bytes")
        if spec.get("hint") == "nixl":
            # Import for the SIDE EFFECT: the mover registers itself under hint='nixl' at import,
            # and this actor process may never have imported it otherwise — resolve_mover would
            # then reject a hint that is genuinely available (movers.py `_weight_mover_cls`).
            import cosmos_framework.posttrain.workers.nixl  # noqa: F401

        def _fresh(mapper: Any, mover_override: Any = None, transfer_dtype: str | None = None) -> WeightSync:
            if "relationship" in spec:
                return WeightSync(
                    mover_override
                    or resolve_mover(
                        relationship=spec["relationship"],
                        plane="weight",
                        capability="tensor",
                        hint=spec.get("hint"),
                    ),
                    mapper=mapper,
                    # SAME value init_nixl sized the staging buffer with (bind passes one knob to
                    # both), so the plan and the buffer cannot disagree.
                    bucket_bytes=bucket_bytes,
                    # Same argument, second axis: init_nixl sizes that buffer from a manifest built
                    # with this dtype, so the plan must declare the same one.
                    transfer_dtype=transfer_dtype,
                )
            return make_weight_backend(spec["kind"], mapper=mapper, transfer_dtype=transfer_dtype)

        selected = None if body_names is None else set(body_names)
        if selected is not None:
            missing = sorted(selected - self._bodies.keys())
            if missing:
                raise ValueError(
                    f"weight endpoint bodies {missing} are absent from actor bodies {sorted(self._bodies)}"
                )
        for name, body in self._bodies.items():
            if selected is not None and name not in selected:
                continue
            if not hasattr(body, "model"):
                if selected is not None:
                    raise ValueError(f"declared weight endpoint body {name!r} exposes no model")
                continue
            # Fresh instance PER body: a backend may hold per-instance mutable state (IpcMover
            # pins the staged snapshot in `_pinned` until the consumer imports it). Sharing one
            # instance across multiple model-bearing bodies in the same actor would let one
            # body's stage() overwrite another's pinned snapshot → GPU use-after-free.
            mover_override = self._nixl_probe_mover if spec.get("hint") == "nixl" else None
            # The wire dtype is declared on the BODY (`LLMTrainer.transfer_dtype`), not on the model
            # it holds, so bind is where it crosses into the backend. `init_nixl` reads the same
            # attribute off the same body when it sizes the staging buffer.
            from cosmos_framework.posttrain.roles.rollout.base import WeightReceiverBase

            backend = _fresh(resolve_weight_mapper(body.model), mover_override, getattr(body, "transfer_dtype", None))
            if isinstance(body, WeightReceiverBase):
                body.bind_weight_backend(backend)
            else:
                body._weight_backend = backend
            if mover_override is not None:
                self._nixl_probe_mover = None

    def set_weight_broadcast(self, spec: dict[str, Any], body_names: tuple[str, ...]) -> None:
        """Inject R2R membership and initialize the communicator in the model-owning process.

        vLLM may own the live model in its engine process, so its adapter sends initialization into
        that process. Ordinary rollout bodies initialize directly in their actor process.
        """
        from cosmos_framework.posttrain.comm.weight_broadcast import ReplicaBroadcastSpec

        parsed = ReplicaBroadcastSpec(**spec)
        selected = set(body_names)
        missing = sorted(selected - self._bodies.keys())
        if missing:
            raise ValueError(f"replica broadcast bodies {missing} are absent from actor bodies {sorted(self._bodies)}")
        for name in body_names:
            body = self._bodies[name]
            if not hasattr(body, "model"):
                raise ValueError(f"replica broadcast body {name!r} exposes no model")
            if not callable(getattr(body, "broadcast_live_weights", None)):
                raise TypeError(f"replica broadcast body {name!r} exposes no broadcast_live_weights()")
            body._weight_broadcast_spec = parsed
            initialize = getattr(body, "initialize_weight_broadcast", None)
            if not callable(initialize):
                raise TypeError(f"replica broadcast body {name!r} exposes no initialize_weight_broadcast()")
            initialize()

    def set_mesh(self, pdims: Any) -> None:
        # Attach the model-owned ParallelDims to THIS actor. Called by run_setup after the K-rank
        # process group exists and before setup_fn builds the model, so role code reads the live mesh
        # through current_pdims(). Placement supplies the world membership only; the trainer's
        # explicit model_parallelism supplies its dp_replicate/dp_shard factorization.
        if pdims is None:
            self._pdims = None
            return
        import torch

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        pdims.build_device_mesh(device_type)
        self._pdims = pdims

    def pdims(self) -> Any:
        # Read back the injected placement dims — the set_mesh seam's counterpart to how
        # set_weight_backend's injection is asserted per actor (backend_id → id(_weight_backend)).
        # Returns a mesh-free ParallelDims CLONE so the driver never serializes the live DeviceMesh
        # (which pins process groups); None when no mesh was injected. Held on the actor, not a role
        # body, because the mesh is injected before the bodies exist.
        if self._pdims is None:
            return None
        from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

        p = self._pdims
        return ParallelDims(
            world_size=p.world_size,
            dp_replicate=p.dp_replicate,
            dp_shard=p.dp_shard,
            tp=p.tp,
            ep=p.ep,
            cp=p.cp,
            pp=p.pp,
        )

    # --- intra-/inter-replica torch.distributed rendezvous (DESIGN §1.1 FSDP) ---
    def rendezvous_endpoint(self) -> tuple[str, int]:
        """Return (node_ip, free_port) for this actor's node — the rank-0 actor's endpoint
        becomes MASTER_ADDR:PORT for the NCCL rendezvous (picked just before init so the
        port is fresh)."""
        ip, ports = self.rendezvous_endpoints(1)
        # TODO(dinghaoy): rendezvous_endpoints closes the socket before init_distributed actually
        # binds the port, so on a busy node another process can claim it in between (NCCL then fails
        # "address already in use" / hangs the whole grid). Harden by keeping the socket open
        # (SO_REUSEPORT) or starting the rank-0 c10d store before returning.
        return ip, ports[0]

    def rendezvous_endpoints(self, count: int) -> tuple[str, tuple[int, ...]]:
        """Allocate distinct currently-free ports on this node in one atomic socket batch."""
        # TODO(dinghaoy): every socket is released before the caller starts its c10d store, so
        # another process can claim a returned port in between. Harden by keeping the sockets open
        # (SO_REUSEPORT) or starting each rank-0 c10d store before returning.
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"rendezvous endpoint count must be positive, got {count!r}")
        import socket

        import ray

        ip = ray.util.get_node_ip_address()
        sockets: list[socket.socket] = []
        try:
            for _index in range(count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((ip, 0))
                sockets.append(sock)
            ports = tuple(int(sock.getsockname()[1]) for sock in sockets)
        finally:
            for sock in sockets:
                sock.close()
        return ip, ports

    def init_distributed(
        self, master_addr: str, master_port: int, rank: int, world_size: int, backend: str | None = None
    ) -> int:
        """Join this replica's process group so an intra-replica collective engine built in
        setup() rendezvous (FSDP self-shard reads world_size at build; vLLM external_launcher
        reuses the world as its TP group). Ray isolates one GPU per actor (CUDA_VISIBLE_DEVICES)
        → the actor's device is cuda:0, also when its group exposes its TP peers' GPUs after it
        (`_expose_peer_gpus`). ``backend`` defaults to nccl on GPU / gloo on CPU; pass it to force
        one (e.g. tests). Idempotent. Returns the rank."""
        import os

        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        return rank

    # --- cross-role weight plane (side group; design §3) ---
    def init_weight_plane(
        self,
        coord_host: str,
        coord_port: int,
        global_rank: int,
        world_size: int,
        backend: str | None = None,
        recv_manifest: list | None = None,
    ) -> int:
        """Join the cross-role weight-plane group — a SIDE ProcessGroup independent of this actor's
        default group (already FSDP or vLLM-TP). Driver-brokered: the driver assigns global ranks
        (policy 0..P-1, rollout P..P+R-1) and passes the rank-0 actor's (host, port) as `coord`.
        Idempotent; returns the global rank.

        NOTE (interim): no production caller yet — the disaggregated cross-device weight plane is
        not auto-wired at bind (control/placement/colocate.py raises NotImplementedError for
        cross_device). Exercised only via tests today; drop this note when bind auto-wires the
        plane (DESIGN §2.4 / §4).

        Wires the live comm into each model-bearing body's already-bound backend via
        ``WeightSync.bind_comm`` (the backend owns its mover's wiring — this layer never touches
        ``mover.*``): ``set_weight_backend`` runs at bind, BEFORE the plane exists, so a collective
        mover starts with ``_comm=None`` — the comm must be injected here or stage/fetch would raise
        (spec §3). ``recv_manifest`` (a ``[(name, shape, dtype_str), ...]`` list the driver computes
        from the policy's ``full_state_dict``) becomes the collective mover's recv layout for
        SHARDED sinks (vLLM TP) that cannot self-derive the full param shapes; an unsharded HF sink
        leaves it None and self-derives from its own model (§2.4)."""
        from cosmos_framework.posttrain.comm.transfer.weight_comm import build_weight_comm

        if self._weight_comm is None:
            self._weight_comm = build_weight_comm(
                coord_host, int(coord_port), int(global_rank), int(world_size), backend or "auto"
            )
        for body in self._bodies.values():
            if not hasattr(body, "model"):
                continue
            be = getattr(body, "_weight_backend", None)
            if be is not None:
                be.bind_comm(self._weight_comm, recv_manifest=recv_manifest)
        return int(global_rank)

    def warm_nixl(self) -> None:
        """Load and warm the NIXL/UCX runtime before role/model setup runs in this actor.

        A GPU-assigned actor binds a CUDA context on the RDT transfer thread through the weight
        transport. A CPU-placed actor only needs Ray's own NIXL agent (UCX, no CUDA) to publish and
        receive CPU pool pages; creating a CUDA context there would fail on a CUDA-less host and
        would otherwise land on a GPU this actor was never assigned.
        """
        from cosmos_framework.posttrain.workers.data_pool import actor_cuda_accessible

        if actor_cuda_accessible():
            from cosmos_framework.posttrain.workers.nixl import NixlTransport

            NixlTransport.get().warmup()
            return
        from ray.experimental.rdt.util import get_tensor_transport_manager

        get_tensor_transport_manager("NIXL").get_nixl_agent()

    def init_nixl(self, bucket_bytes: int | None = None, body_names: tuple[str, ...] | None = None) -> int:
        """Build this actor's NIXL weight transport: warmup + registered fabric staging buffer.

        Called PER ACTOR from bind via ``.remote()``, mirroring ``init_weight_plane``.
        ``bind_device_groups`` runs on the DRIVER and reaches actors only through
        ``.remote()``, so constructing the transport driver-side would leave every actor
        uninitialized and fail at the first sync.

        Sizes the buffer HERE rather than taking it as an argument: the size is max(bucket
        bytes) over the whole plan, which needs the local model's manifest. The driver has no
        model, and shipping a manifest over RPC just to size a local allocation would be
        ceremony. Bodies exist by now (``run_setup`` ran in spawn Pass 3).

        The buffer is registered and PINNED for the process lifetime, so it cannot grow —
        sizing from the WHOLE plan rather than the first bucket is load-bearing. ``bucket_plan``
        accumulates until it EXCEEDS the cap, so a single oversized parameter (a large
        embedding or ``lm_head``) forms its own bucket bigger than ``BUCKET_BYTES``; taking the
        max over the plan is what covers that.

        Returns:
            The staging buffer size in bytes, or 0 if this actor hosts no STAGING body (a
            consumer-only actor — e.g. one holding just a vLLM rollout — is still warmed up).
        """
        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest
        from cosmos_framework.posttrain.comm.weight_transfer import (
            BUCKET_BYTES,
            bucket_plan,
            model_manifest,
            resolve_weight_mapper,
        )
        from cosmos_framework.posttrain.workers.nixl import NixlTransport

        # Warm up UNCONDITIONALLY, before any sizing: bind calls warm_nixl pre-model on every
        # endpoint actor, and this idempotent call preserves safety for direct/test callers. A
        # consumer-only actor allocates no staging buffer but still runs a NIXL receive, and RDT's
        # transfer thread needs a CUDA context bound either way or the first transfer fails.
        t = NixlTransport.get()
        t.warmup()

        need = 0
        selected = None if body_names is None else set(body_names)
        for name, body in self._bodies.items():
            if selected is not None and name not in selected:
                continue
            model = getattr(body, "model", None)
            if model is None:
                continue
            # PRODUCER-side sizing only. `hasattr(body, "model")` is NOT enough: a vLLM rollout
            # body also exposes `.model`, but it is an `LLM` engine handle with no `state_dict`,
            # and `model_manifest` would raise on it. The staging buffer belongs to whoever
            # STAGES; a consumer lands into its own receive buffer (NixlWeightMover.fetch).
            net = getattr(model, "net", model)  # mirrors weight_transfer._net
            if not callable(getattr(net, "state_dict", None)):
                continue
            # Sized from the MANIFEST, not from a state_dict lookup: `model_manifest` unwraps
            # `.net` (a wrapped HFModel exports `net.`-prefixed keys), so joining its names back
            # against `body.model.state_dict()` would silently match nothing and leave `need`
            # at 0 — i.e. no staging buffer, failing only at the first sync. Metadata sizing
            # also needs no tensors, which matters because DTensor params are still sharded here.
            # Same cap the backend will plan with (bind passes one value to both), so the buffer
            # cannot be sized for a different bucketing than the sync actually uses.
            # The staging buffer must be sized from the SAME manifest the plan will declare, or the
            # two disagree and the buffer cannot grow after bind. With fp32 storage and a bf16 wire
            # dtype a model-dtype manifest merely OVERSIZES (safe but 2x); the reverse UNDERSIZES
            # and fails mid-sync.
            mapper = resolve_weight_mapper(model)
            transfer_dtype = getattr(body, "transfer_dtype", None)
            for bucket in bucket_plan(model_manifest(model, mapper, transfer_dtype), bucket_bytes or BUCKET_BYTES):
                need = max(need, flat_nbytes_for_manifest(bucket))
        if need == 0:
            return 0  # consumer-only actor: warm, but nothing to stage
        # RESERVE the size; the buffer itself is allocated by each mover on its first stage
        # (movers own their buffers so two in one actor never write the same bytes). Bind runs
        # before set_weight_backend, so no mover exists here to allocate into.
        t.reserve(need)
        return need

    def nixl_probe_publish(self, nbytes: int, buffer_bytes: int | None = None, fill: int = 0xA5) -> Any:
        """Publish a deterministic fabric-backed buffer before any model body is built."""
        if nbytes < 1:
            raise ValueError(f"NIXL probe bytes must be positive, got {nbytes}")
        import torch

        from cosmos_framework.posttrain.workers.nixl import NixlTransport, NixlWeightMover

        transport = NixlTransport.get()
        transport.warmup()
        transport.reserve(max(nbytes, int(buffer_bytes or nbytes)))
        source = self._nixl_probe_source
        if source is None or int(source.numel()) != nbytes:
            source = torch.full((nbytes,), int(fill), dtype=torch.uint8, device="cuda")  # [N]
        else:
            source.fill_(int(fill))  # [N]
        mover = self._nixl_probe_mover or NixlWeightMover()
        handle = mover.stage({"probe": source})
        self._nixl_probe_source = source
        self._nixl_probe_mover = mover
        return handle

    def nixl_probe_fetch(self, handle: Any, fill: int = 0xA5) -> dict[str, int | float | str]:
        """Fetch and verify one real actor-to-actor NIXL publication, returning wall-clock bandwidth."""
        import ray
        import torch

        from cosmos_framework.posttrain.workers.nixl import NixlTransport, NixlWeightMover

        NixlTransport.get().warmup()
        mover = self._nixl_probe_mover or NixlWeightMover()
        mover.prepare_fetch(handle)
        started = time.perf_counter()
        state = mover.fetch(handle)  # [N]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        tensor = state.get("probe")  # [N]
        if tensor is None or tensor.dtype != torch.uint8:
            raise RuntimeError("NIXL probe did not reconstruct the uint8 'probe' tensor")
        matches = torch.all(tensor == int(fill))  # []
        if not bool(matches.item()):
            raise RuntimeError("NIXL probe value witness failed")
        nbytes = int(tensor.numel())
        self._nixl_probe_mover = mover
        return {
            "bytes": nbytes,
            "seconds": seconds,
            "gbps": nbytes / max(seconds, 1e-12) / 1e9,
            "node_id": str(ray.get_runtime_context().get_node_id()),
        }

    def nixl_probe_release(self) -> None:
        """Drop actor-owned probe state after the consumer and driver release its handle."""
        import torch

        torch.cuda.synchronize()
        wait_free = getattr(self._nixl_probe_mover, "_wait_free", None)
        if callable(wait_free):
            wait_free()
        recycle = getattr(self._nixl_probe_mover, "recycle_staging_buffer", None)
        if callable(recycle):
            recycle()
        self._nixl_probe_source = None

    def reserve_nixl(self, nbytes: int) -> None:
        """Raise this actor's staging-buffer size to `nbytes` (bind's max over all actors).

        `init_nixl` can only size from the LOCAL model, and a consumer lands the PRODUCER's
        bucket, which need not be the same size: FSDP2 mixed precision keeps fp32 master weights
        on the trainer while the rollout holds bf16, so the same parameter is twice the bytes on
        the producing side. Bind levels everyone to the max once all actors have reported.
        """
        from cosmos_framework.posttrain.workers.nixl import NixlTransport

        NixlTransport.get().reserve(nbytes)

    def weight_comm(self):
        """The live weight-plane comm (None if init_weight_plane not called). Read by the driver
        loop's COLLECTIVE dispatch and by movers on the bodies.

        NOTE (interim): no production caller yet — see init_weight_plane; test-only until bind
        auto-wires the cross-device weight plane."""
        return self._weight_comm

    def weight_comm_local(self):
        """Test-only direct accessor (the Ray-serialized handle would drop the live PG)."""
        return self._weight_comm

    def run(self, fn, *a, **k):
        """Test-only: execute fn(self) inside the actor — the LIVE `_Host` instance, not a
        remote handle, so fn can call actor methods (e.g. weight_comm_local()) directly without
        `.remote()` (used to drive an in-actor collective). Callers may pass this actor's own
        ActorHandle as `a[0]` for call-site symmetry (Ray requires an explicit arg to route the
        `.remote()` dispatch to the right actor); a serialized ActorHandle only supports
        `.method.remote()`, never direct calls, even from within its own actor process — so it is
        accepted but not forwarded; `self` is substituted instead."""
        return fn(self, *a[1:], **k)

    def dist_all_reduce_probe(self) -> float:
        """De-risk probe: all-reduce(sum) a rank-valued tensor over the live group. Every
        rank must return sum(0..world-1) if the NCCL rendezvous is real.

        Returns:
            The all-reduced sum — `sum(0..world_size-1)` on a real rendezvous.
        """
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("dist_all_reduce_probe: init_distributed not called")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([float(dist.get_rank())], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def expose_peer_gpus(self) -> str:
        """Show this actor its node's other GPUs, its own first; the spawner's first call on it.

        An actor method rather than a Ray runtime env: in the pinned container an actor started with
        its own runtime env loses import paths, such as the reward package's, that other actors inherit.
        """
        _expose_peer_gpus()
        return os.environ["CUDA_VISIBLE_DEVICES"]

    # --- placement introspection (DESIGN §4.1) ---
    # Node IP / visible GPU belong to the worker process, not the role body: the driver
    # reads them through the backend (GridBackend.hosts()/gpus()) so role bodies stay
    # placement-agnostic (§4.1) and Ray-free.
    def host(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def gpu(self) -> str:
        import os

        return os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    def gpu_identity(self) -> dict[str, Any]:
        """Return authoritative physical-device identity for bind-time DeviceGroup validation.

        CUDA UUID comes directly from the driver for the actor's own device, ordinal 0: its sole
        visible device, or the first one when its group exposes TP peer GPUs. Hostname,
        Ray node id, CUDA_VISIBLE_DEVICES, and Ray accelerator ids are diagnostics only: separate
        raylets may report identical logical ids for different physical GPUs.
        """
        import ctypes
        import os

        class _CudaUuid(ctypes.Structure):
            _fields_ = [("bytes", ctypes.c_ubyte * 16)]

        try:
            cuda = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise RuntimeError(
                "GPU identity proof requires libcuda.so.1 from an NVIDIA driver exposing the "
                f"{_CUDA_UUID_V2_REQUIREMENT}"
            ) from exc

        def _check(result: int, operation: str) -> None:
            if result != 0:
                raise RuntimeError(f"CUDA driver {operation} failed with result {result}")

        _check(int(cuda.cuInit(0)), "cuInit")
        count = ctypes.c_int()
        _check(int(cuda.cuDeviceGetCount(ctypes.byref(count))), "cuDeviceGetCount")
        if count.value != 1 and not (count.value > 1 and os.environ.get(PEER_GPUS_ENV) == "1"):
            raise RuntimeError(
                "GPU DeviceGroup identity requires each Ray actor to see exactly one CUDA device, "
                f"got {count.value} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')!r})"
            )
        device = ctypes.c_int()
        _check(int(cuda.cuDeviceGet(ctypes.byref(device), 0)), "cuDeviceGet")
        get_uuid = _require_cuda_uuid_v2(cuda)
        uuid = _CudaUuid()
        _check(int(get_uuid(ctypes.byref(uuid), device)), "cuDeviceGetUuid_v2")
        raw_uuid = bytes(uuid.bytes)
        if not any(raw_uuid):
            raise RuntimeError("CUDA driver returned an all-zero device UUID")
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        return {
            "cuda_uuid": raw_uuid.hex(),
            "ray_node_id": str(ray.get_runtime_context().get_node_id()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            "ray_accelerator_ids": tuple(accelerator_ids.get("GPU", ())),
        }

    def validate_shared_weight_identity(
        self,
        policy_names: tuple[str, ...],
        rollout_names: tuple[str, ...],
    ) -> None:
        """Fail bind unless same-process policy and rollout bodies share exact model storage."""
        if len(policy_names) != 1 or len(rollout_names) != 1:
            raise RuntimeError(
                "shared model identity validation expects one policy and one rollout body, got "
                f"policy={policy_names}, rollout={rollout_names}"
            )

        def _identity(body_name: str) -> str:
            body = self._bodies[body_name]
            model = getattr(body, "model", None)
            if model is None:
                raise RuntimeError(f"same-process body {body_name!r} exposes no model storage")
            from cosmos_framework.posttrain.comm.weight_transfer import shared_model_identity

            return shared_model_identity(model)

        policy_identity = _identity(policy_names[0])
        rollout_identity = _identity(rollout_names[0])
        if policy_identity != rollout_identity:
            raise RuntimeError(
                "same-process SharedMover requires policy and rollout to share exact model storage; "
                f"policy identity={policy_identity!r}, rollout identity={rollout_identity!r}"
            )

    @ray.method(concurrency_group="bulk")
    def fetch_bulk(self, ref, readers: int = 1) -> Any:
        """Producer side of a remote one-sided pull (§3.4): serve one `ref` from the local `BulkStore`.

        Args:
            ref: the BulkRef to serve to the pulling consumer actor.
            readers: reader claims for the refcounted `BulkStore.fetch`.

        Returns:
            The bulk value for `ref`.
        """
        # producer side of a remote one-sided pull (§3.4): hand the bulk to the
        # consumer actor directly (refcounted); the controller is not in this path.
        # Runs in the "bulk" concurrency group so a peer pulling this actor's bulk is
        # served even while this actor's own body is blocked in a collective (see the
        # _Host docstring — this is what closes the collective-blocked-peer deadlock).
        # BulkStore.fetch is lock-guarded (comm/bulk.py) since it now runs off-thread.
        return self._store.fetch(ref, readers=readers)

    @ray.method(concurrency_group="bulk")
    def fetch_bulk_many(self, refs, readers: int = 1) -> list:
        """Batched producer side: one RPC serves every ref a consumer needs from the local `BulkStore`.

        Args:
            refs: the BulkRefs to serve, in order.
            readers: reader claims for each refcounted `BulkStore.fetch` (per-ref locking).

        Returns:
            The bulk values for `refs`, positionally aligned.
        """
        # batched producer side: one RPC pulls all refs a consumer needs (refcounted).
        # "bulk" concurrency group (see fetch_bulk / the _Host docstring). Per-ref locking
        # (inside BulkStore.fetch) is deliberate: refs are independent keys, so a batch-wide
        # lock would only reduce concurrency without adding safety.
        return [self._store.fetch(r, readers=readers) for r in refs]

    @ray.method(concurrency_group="bulk")
    def release_bulk(self, refs, readers: int = 1) -> None:
        """Drop-path bookkeeping: decrement each ref's reader claims WITHOUT pulling data, freeing
        producer memory once a ref reaches zero (for entries the consumer sliced away).

        Args:
            refs: the BulkRefs to release.
            readers: reader claims to decrement per ref via `BulkStore.release`.
        """
        # drop-path bookkeeping: decrement reader claims WITHOUT pulling data (for
        # entries that reader sliced away); reaches zero and frees producer memory.
        # "bulk" concurrency group (see fetch_bulk / the _Host docstring).
        for r in refs:
            self._store.release(r, readers=readers)

    def node_id(self) -> str:
        """This actor's Ray node id — for tests asserting mesh placement / trainer-rollout
        co-location. Lives on the actor substrate (already Ray/placement-aware, cf.
        rendezvous_endpoint), NOT on the role bodies: a training/inference body must not know Ray
        node placement. Prefer the Ray node id over hostname DNS (loopback in some container setups)."""
        try:
            return ray.get_runtime_context().get_node_id()
        except Exception:
            import socket

            return socket.gethostbyname(socket.gethostname())

    def _sweep_pending_releases(self) -> None:
        """Bounded, NON-blocking sweep of the fire-and-forget cross-actor release ObjectRefs
        (`ray.wait(..., timeout=0)`): `ray.get` anything that finished since the last sweep and RAISE
        on a remote-side failure (double release / accounting drift), otherwise return immediately.
        Single-threaded actor state (only `_default` methods touch `_pending_releases`), so no lock.

        Called from BOTH the top of every `invoke` (so a failed release surfaces on the very next
        actor operation of ANY kind, not only the next `_release_dropped`) and from within
        `_release_dropped` itself (cheap, idempotent). The LAST batch of a run has no next operation
        to sweep it — that residual is covered by `drain_pending_releases`, the blocking teardown
        drain registered via `atexit` in `__init__`. Fail-loud posture mirrors the local release
        branch's inline KeyError."""
        if not self._pending_releases:
            return
        done, rest = ray.wait(self._pending_releases, num_returns=len(self._pending_releases), timeout=0)
        self._pending_releases = rest
        for ref in done:
            try:
                ray.get(ref)
            except Exception as e:
                raise RuntimeError(
                    "cross-actor bulk release failed on a prior sync (accounting drift — "
                    "double release or missing key); failing loud now rather than leaking silently"
                ) from e

    def drain_pending_releases(self, timeout: float = 10.0) -> None:
        """BLOCKING final drain of the fire-and-forget cross-actor release ObjectRefs — the
        actor-teardown counterpart to the non-blocking per-invoke `_sweep_pending_releases`. That
        sweep only runs on a NEXT operation, so the LAST batch of releases in a run (the
        accounting-drift case most likely to appear on a LONG run — a double release / missing key
        `_track_readers` exists to catch) would otherwise be torn down with the ObjectRef still in
        `_pending_releases` and the failure silently discarded. Registered via `atexit` in `__init__`
        so it fires on actor-process exit; a driver may also call it explicitly before shutdown.

        Waits up to `timeout` for every outstanding release, `ray.get`s each, and RAISES on a
        remote-side failure — an uncaught raise in an `atexit` handler prints loudly to stderr rather
        than vanishing. Best-effort under a global `ray.shutdown` (a peer torn down first surfaces as
        a Ray actor error rather than the exact accounting error — still not silent).

        Args:
            timeout: seconds to wait for outstanding releases (typically already resolved by
                teardown, having fired during the run) before giving up.
        """
        if not self._pending_releases:
            return
        pending, self._pending_releases = self._pending_releases, []
        done, _rest = ray.wait(pending, num_returns=len(pending), timeout=timeout)
        for ref in done:
            try:
                ray.get(ref)
            except Exception as e:
                raise RuntimeError(
                    "cross-actor bulk release failed on the final batch of a run (accounting drift — "
                    "double release or missing key); failing loud at teardown rather than discarding"
                ) from e

    def _release_refs(self, refs: list, readers: int) -> None:
        """Release protocol seam (Task A6, DESIGN §2.4): drop reader claims on ``refs`` WITHOUT fetching.

        Passed as `_resolve_dispatched`'s `releaser` for the BulkRefs an on_metadata split drops, and
        run at egress for a body method declared in ``lease_releasing_output_methods`` (the buffer's
        `drain`). Grouped by owning store: same-actor (this process's registry — `has_local_store`,
        the same check `RayStoreMover.fetch` uses) -> release locally, no RPC (raises loudly inline
        on any accounting bug, e.g. double release — see `BulkStore.release`'s KeyError); cross-actor
        -> fire-and-forget `release_bulk.remote` on the owning actor via the transport's peers
        registry (the same {store_id -> actor} connection map `RayStoreMover` pulls through).

        Fail-loud posture: the cross-actor call returns an ObjectRef that can carry a remote-side
        failure. Awaiting it here would turn a deliberately fire-and-forget optimization into a
        blocking RPC per dropped ref, but discarding it made a remote failure fail-SILENT (unlike the
        local branch). Compromise: stash the ObjectRef in `self._pending_releases` and sweep it
        non-blocking, BOTH here (cheap, idempotent) AND at the top of every `invoke`, so a failure
        surfaces on the next actor operation of ANY kind (a one-operation delay), never silently.

        Args:
            refs: the BulkRefs to release (never fetched by this path).
            readers: reader claims to decrement per ref (matches the pull-path pop).
        """
        self._sweep_pending_releases()

        by_owner: dict[str, list] = {}
        for r in refs:
            by_owner.setdefault(r.store_id, []).append(r)
        peers = getattr(self._transport, "_peers", {})
        for sid, group in by_owner.items():
            if has_local_store(group[0]):
                for r in group:
                    store_for(r).release(r, readers=readers)
            else:
                self._pending_releases.append(peers[sid].release_bulk.remote(group, readers=readers))

    def invoke(self, name: str, method: str, args: tuple, kw: dict) -> Any:
        """Actor-side call entrypoint: sweep pending releases, resolve/materialize the args (incl.
        `Dispatched` descriptors and nested ObjectRefs), run `body.method(*args, **kw)`, then pack
        any bulk in the result so only metadata leaves the actor.

        Args:
            name: the body key (WG name) to dispatch on within this actor.
            method: the body method to call (or `"node_id"`, answered by the actor itself).
            args: positional args, each resolved via `_resolve` before the call.
            kw: keyword args, each value resolved via `_resolve` before the call.

        Returns:
            The body method's result, `pack_payload`-externalized to BulkRef metadata when the codec
            seam is on.
        """
        # Finding-3: sweep any pending fire-and-forget releases at the top of EVERY invoke (only when
        # non-empty) so a failure on the final metadata-split of a run surfaces on the next actor
        # operation of any kind, instead of leaking until the next _release_dropped that may never come.
        if self._pending_releases:
            self._sweep_pending_releases()
        if method == "node_id":  # host/placement introspection — answered by the ACTOR, not the body
            return self.node_id()
        try:
            return self._invoke_impl(name, method, args, kw)
        finally:
            if isinstance(self._transport, RayStoreMover):
                self._transport.clear_pool_tensor_sources()

    def _invoke_local_peer(self, name: str, method: str, args: tuple, kw: dict) -> Any:
        """Re-enter a colocated body without RPC, honoring its input materialization contract."""
        body = self._bodies[name]
        if self._codec and name in self._materialize and method not in getattr(body, "metadata_only_input_methods", ()):
            args = tuple(materialize_payload(value, self._transport) for value in args)
            kw = {key: materialize_payload(value, self._transport) for key, value in kw.items()}
        return getattr(body, method)(*args, **kw)

    def _invoke_impl(self, name: str, method: str, args: tuple, kw: dict) -> Any:
        """Resolve, call, and repack one role method."""
        # Resolve ObjectRefs passed nested in args/kw (Ray only auto-resolves refs that are
        # DIRECT .remote() args, not ones nested in a container). Resolving here — on the
        # actor — keeps the controller out of the path: it forwards only the tiny ref and the
        # producer's metadata payload moves object-store→this actor directly (DESIGN §3.4/#7).
        # No-op for RayBackend (it passes already-resolved values, not refs).
        #
        # A `Dispatched` arg (Task A2/A4 asymmetric-mesh routing descriptor) carries its OWN
        # list of producer refs plus a reduce/split adaptation; `_resolve_dispatched` (Task A5)
        # folds THIS arg's ref-resolve + materialize + adapt into a single step (readers
        # threaded through for refcounted bulk pop) so the body never sees a list/shard — only
        # the adapted value. Non-Dispatched args keep the exact same resolve-then-materialize
        # sequence as before this task (byte-identical).
        body = self._bodies[name]
        metadata_only_methods = getattr(body, "metadata_only_input_methods", ())
        # A role can be bulk-consuming for one method and metadata-only for another. GRPO's reward
        # worker is the motivating case: training prefix filtering reads completion token tensors,
        # while held-out rule scoring reads only inline text/reference fields. Treating the whole
        # role as bulk-consuming would consume rollout-owned validation refs and then repack/leak
        # them at RM egress. The method declaration keeps that ownership explicit.
        materialize = self._codec and name in self._materialize and method not in metadata_only_methods

        # The same release seam serves two drop paths: the on_metadata split's dropped refs (via
        # `_resolve_dispatched`'s `releaser`) and the lease-releasing egress below.
        _release_dropped = self._release_refs

        def _resolve(a: Any) -> Any:
            """Resolve one call arg: fold a `Dispatched` through `_resolve_dispatched`, else `ray.get`
            a bare ObjectRef and materialize bulk when this consuming WG computes on it.

            Args:
                a: a single positional/keyword arg value.

            Returns:
                The resolved (and, for a bulk consumer, materialized) value the body will receive.
            """
            if isinstance(a, LocalPeerRef):
                return LocalPeerClient(lambda method, args, kw: self._invoke_local_peer(a.name, method, args, kw))
            if isinstance(a, PeerClient):
                return _BoundPeerClient(a, self, self._codec and name in self._materialize)
            if isinstance(a, Dispatched):
                # Task 9: the actor's REAL dp coordinate. With `dp_group_sharded` deleted from this
                # path, `select_chunk` is the sole dp-axis owner and needs the actor's true
                # `(dp_rank, dp_world)` — computed actor-side because only the actor holds the
                # `ParallelDims` mapping this cell's rank `k` to a dp coordinate (the driver knows only
                # `(r, k)`). `_dp_coords` reports the complete dp_replicate*dp_shard extent and
                # returns `(0, 1)` when `self._pdims` is None, keeping the interleave a no-op there.
                from cosmos_framework.posttrain.config.build import _dp_coords

                dp_rank, dp_world = _dp_coords(self._pdims)
                return _resolve_dispatched(
                    a, self._transport, materialize, releaser=_release_dropped, dp_rank=dp_rank, dp_world=dp_world
                )
            v = ray.get(a) if isinstance(a, ray.ObjectRef) else a
            if materialize:
                # consumer that computes on bulk: pull it (local ⇒ zero-copy, remote ⇒ Ray).
                # TODO(Option B — rank0-pull + intra-replica NCCL broadcast for large bulk): the
                # plain (non-Dispatched) arg path — a mirror/identity edge where every consumer rank
                # independently pulls the SAME payload. The Dispatched trainer-batch edge above no
                # longer has this shape (Task 8: each rank materializes only its own kept share), so
                # this is narrowed to edges that still mirror. For a big multimodal obs there, have
                # rank 0 pull once and broadcast to ranks 1..K-1 over the per-replica torch.distributed
                # group (up since spawn_grid Pass 2). Needs (a) the replica-rank + group handle, (b) a
                # BroadcastTransport materialize_payload dispatches on, gated by bcast_aligned (grid.py).
                v = materialize_payload(v, self._transport)
            return v

        profile = ProfileRecord("ray_invoke")
        with profile.phase("materialize"):
            args = tuple(_resolve(a) for a in args)
            kw = {k: _resolve(v) for k, v in kw.items()}
        input_slices = list(iter_pool_slices((args, kw))) if profile.enabled else []
        with profile.phase("body"):
            out = getattr(body, method)(*args, **kw)
        if self._codec and method in getattr(body, "lease_releasing_output_methods", ()):
            # Terminal egress (DESIGN §5 "evict releases the bulk lease"): a metadata-only body that
            # declares this method returns items whose bulk nobody will consume. Release every BulkRef
            # the payload carries at its owning store — a refcount decrement, never a fetch — so a
            # dropped trajectory's bulk stays where it was produced instead of being mirrored onto a
            # bulk-consuming sink just to be discarded. Refs the buffer holds were never claimed by
            # a reader (its ingress does not materialize), so one release pops each entry.
            self._release_refs(collect_bulk_refs(out), readers=1)
        if self._codec:
            # Producer side: externalize bulk before returning metadata.
            pool_lease_id: str | None = None
            pool_config = self._data_pool_config
            pool_stage: Callable[[Any], Any] | None = None
            if pool_config is not None:

                def _stage_pool_tensor(value: Any) -> Any:
                    nonlocal pool_lease_id
                    if not isinstance(value, torch.Tensor):
                        return None
                    preserved = (
                        self._transport.source_for_tensor(value) if isinstance(self._transport, RayStoreMover) else None
                    )
                    if preserved is not None:
                        return preserved
                    # future(yangyangt): every CUDA tensor is pooled regardless of size, so a 0-D
                    # reward or logprob scalar costs a 256-byte aligned slot plus a PoolSlice
                    # descriptor larger than the value. A `cuda_min_bytes` floor mirroring the CPU
                    # one would keep such tensors inline, but inline CUDA tensors are then serialized
                    # by Ray and may arrive on a different device; verify that before changing this.
                    if not value.is_cuda and value.nbytes < pool_config["cpu_min_bytes"]:
                        return value
                    if self._producer_pool is None:
                        self._producer_pool = ProducerDataPool(
                            self._store.id,
                            cuda_capacity_bytes=pool_config["cuda_capacity_bytes"],
                            cpu_capacity_bytes=pool_config["cpu_capacity_bytes"],
                            page_bytes=pool_config["page_bytes"],
                        )
                        self._transport.set_local_producer_pool(self._producer_pool)
                        self._producer_pool.warmup()
                    if pool_lease_id is None:
                        pool_lease_id = f"{self._store.id}:{uuid.uuid4().hex}"
                    return self._producer_pool.stage_tensor(pool_lease_id, value)

                pool_stage = _stage_pool_tensor

            with profile.phase("pack"):
                out = pack_payload(out, self._store, self._threshold, pool_stage=pool_stage)
            if pool_lease_id is not None:
                with profile.phase("publish"):
                    self._producer_pool.publish_step(pool_lease_id, [])
            else:
                profile.set_phase("publish")
        else:
            profile.set_phase("pack")
            profile.set_phase("publish")
        if profile.enabled:
            output_slices = list(iter_pool_slices(out))
            profile.emit(
                role=name,
                method=method,
                input_pool_slices=len(input_slices),
                input_unpublished=sum(slice_.page_ref is None for slice_ in input_slices),
                output_pool_slices=len(output_slices),
                output_unpublished=sum(slice_.page_ref is None for slice_ in output_slices),
            )
        return out


# Keep the bulk concurrency group for training, where it prevents peer pulls
# from deadlocking behind a collective-blocked invoke.
# NOT `enable_tensor_transport` here: `spawn_grid` sets it per actor, and bind turns it on only
# for the two ActorGroups on a NIXL cross_device weight edge, so ordinary actors keep the optional
# NIXL dependencies lazy. It is what gives an actor the `_ray_system` concurrency group -- the
# background thread RDT runs transfers on (ray/actor.py, guarded by exactly this flag). Ray also
# infers it from a method decorated `@ray.method(tensor_transport=...)`, but NixlWeightMover
# stages through `ray.put(_tensor_transport="nixl")` INSIDE the actor, which Ray cannot see, so it
# has to be asked for. An endpoint actor without it accepts a publication that then never moves:
# the consumer waits out the RDT fetch timeout and reports "not found in RDT object store", the
# producer sits in ray.wait, and NOTHING logs an error -- the same flag also creates
# `_ray_system_error`, so Ray's own error path check-fails before it can report anything.
_Host = ray.remote(concurrency_groups={"bulk": 2}, max_concurrency=1)(_HostImpl)


class _HostMainThreadImpl(_HostImpl):
    """Plain actor body for native-heavy host-affinity eval roles.

    Declaring a Ray concurrency group dispatches actor methods on executor
    threads. Isaac Sim must construct SimulationApp on the process main thread,
    and policy-side native conversions have the same constraint. Override the
    decorated bulk methods so this actor can remain plain and single-threaded
    while preserving their behavior between actor calls.
    """

    def fetch_bulk(self, ref, readers: int = 1) -> Any:
        return super().fetch_bulk(ref, readers)

    def fetch_bulk_many(self, refs, readers: int = 1) -> list:
        return super().fetch_bulk_many(refs, readers)

    def release_bulk(self, refs, readers: int = 1) -> None:
        return super().release_bulk(refs, readers)


_HostMainThread = ray.remote(_HostMainThreadImpl)


class RayBackend:
    """Routes WG method calls to the shared host actor; returns ObjectRef-backed Handles."""

    def __init__(self, host: Any):
        self.host = host

    def submit(self, wg_name: str, body: Any, method: str, args: tuple, kw: dict) -> Handle:
        # body lives in the actor (ignore the local copy); call returns an ObjectRef.
        ref = self.host.invoke.remote(wg_name, method, args, kw)
        return Handle(ref=ref, getter=ray.get)
