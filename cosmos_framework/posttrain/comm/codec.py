# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Codec — payload ↔ metadata/bulk, field-level (DESIGN §3.2 Codec, §3.4 field-level).

Decides inline-vs-lazy per field by size: small fields stay inline (travel with the
metadata); large fields (video/image, and — for this demo — token tensors) go to the
producer's BulkStore as a `BulkRef` (lazy, pulled on demand). The consumer
`materialize`s the bulk; **the controller only ever routes the metadata** (DESIGN §3.4
/ principle #7), so bulk never crosses the control plane.

Entry points:
- `pack_payload` / `materialize_payload` — payload dispatchers that walk a Trajectory
  (or list/tuple of them) and swap its tensor-bearing fields ↔ `BulkRef`, **non-mutating**
  (fresh dataclasses, so producer/buffer objects are untouched). This is the §3.2
  "scatter-gather for nested trajectories (never pickle tensors)" seam.
- `materialize_obs` — consumer-side field-selective pull of a single `ObsBundle`'s bulk.

v0: a size threshold + a registry-backed in-process store. TODO(v1+): tier-aware
packing (compress cross-node, raw GPU surface same-node), real buffer descriptors,
field-level lazy pull (consumer pulls only the fields it reads, not all bulk).
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any, Callable

from cosmos_framework.posttrain.comm.bulk import BulkRef, BulkStore
from cosmos_framework.posttrain.comm.transfer.mover import Mover as Transport
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory, Transition

DEFAULT_THRESHOLD = 1024  # bytes — small inline, large → lazy BulkRef (DESIGN §3.4)


def materialize_obs(bundle: ObsBundle, transport: Transport, fields: set[str] | None = None) -> ObsBundle:
    """Consumer-side: pull BulkRefs (optionally only `fields`) via the transport. A selected value is
    deep-walked, so a container-valued representation (e.g. `frames=[BulkRef, BulkRef]`) resolves and
    not just a top-level BulkRef — symmetric with the recursive pack (a top-level ref is the depth-0
    case). Non-selected fields keep their BulkRef untouched → e.g. raw video never reaches the trainer
    if it only needs encoded features (DESIGN §3.4 / VLA split-worker)."""
    reps: dict[str, object] = {}
    for k, v in bundle.representations.items():
        reps[k] = _unpack_val(v, transport) if (fields is None or k in fields) else v
    return ObsBundle(representations=reps, modality_tags=bundle.modality_tags, id=bundle.id)


# ── Trajectory-level metadata/bulk (§3.2 scatter-gather, §3.4 field-level) ──────────
# A tensor-bearing field (action / mask / targets / tensor values inside
# policy_info|algo_extra|obs.representations) above `threshold` is externalized to the
# producer's BulkStore; everything else (ids, reward/advantage scalars, completion_text,
# weight_version, status) stays inline as metadata. NOTE: §3.4 keeps small fields like
# `action` inline in production (bulk = video/image); here threshold defaults to 0 so the
# LLM example's token tensors exercise the seam — they are the only tensors we have.


def _bulk_nbytes(v: Any) -> int:
    """Byte size if `v` is a tensor/ndarray, else -1 (⇒ inline). Handles torch
    versions without `Tensor.nbytes` via element_size()*nelement()."""
    nb = getattr(v, "nbytes", None)  # numpy / newer torch / _Blob
    if isinstance(nb, int):
        return nb
    es, ne = getattr(v, "element_size", None), getattr(v, "nelement", None)
    if callable(es) and callable(ne):  # older torch.Tensor
        return es() * ne()
    return -1  # not bulk-eligible


# ObsBundles carry their own field-selective materialize path (materialize_obs); one nested inside a
# generic field would silently bypass it, so the walkers make that misuse explicit.
_NESTED_OBS_MSG = (
    "ObsBundle nested inside a generic payload field — pack it via obs/next_obs so the field-selective "
    "materialize_obs path applies (nesting it elsewhere is out of contract)."
)


def _is_pool_slice(v: Any) -> bool:
    """Return whether ``v`` is a pool descriptor without importing the Ray backend."""
    return type(v).__name__ == "PoolSlice" and all(
        hasattr(v, name) for name in ("owner_id", "lease_id", "page_ref", "nbytes")
    )


def _is_bulk_leaf(v: Any) -> bool:
    """A leaf the codec externalizes: an existing `BulkRef` or a tensor/ndarray (bulk-eligible).
    Tested BEFORE container descent so a tensor is never iterated."""
    return isinstance(v, BulkRef) or _is_pool_slice(v) or _bulk_nbytes(v) >= 0


def _walk(v: Any, leaf_fn: Callable[[Any], Any]) -> Any:
    """Recurse a field value, applying `leaf_fn` to each bulk leaf (tensor-like/_Blob/BulkRef)
    and rebuilding the surrounding containers fresh (non-mutating). The leaf check runs BEFORE
    container descent so a tensor is never iterated; scalars (str/bytes/int/float/None) pass
    through. This is what lets a tensor nested at `targets['winner'][media][0]` be externalized,
    not just top-level fields. Assumes ACYCLIC payloads (trajectory trees) — there is no visited-set,
    so a self-referential structure would recurse forever."""
    if _is_bulk_leaf(v):
        return leaf_fn(v)
    if isinstance(v, ObsBundle):
        raise TypeError(_NESTED_OBS_MSG)
    if isinstance(v, dict):
        return {k: _walk(x, leaf_fn) for k, x in v.items()}  # preserves key order
    if isinstance(v, (list, tuple)):
        walked = [_walk(x, leaf_fn) for x in v]
        # NamedTuple subtypes construct positionally — preserve the type across the round trip
        # instead of downgrading it to a plain tuple.
        if isinstance(v, tuple):
            return type(v)(*walked) if hasattr(v, "_fields") else tuple(walked)
        return walked
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # replace() rejects init=False fields (they are recomputed by __post_init__ / defaults),
        # so only pass the init fields through the walker.
        changes = {f.name: _walk(getattr(v, f.name), leaf_fn) for f in dataclasses.fields(v) if f.init}
        return dataclasses.replace(v, **changes)
    return v


def _visit_bulk_leaves(v: Any, fn: Callable[[Any], None]) -> None:
    """Read-only sibling of `_walk`: call `fn` on each bulk leaf in the SAME descent order, WITHOUT
    rebuilding any container — the prefetch fold only gathers refs, so `_walk`'s fresh dict/list/
    `replace` allocation would be pure waste. Same ObsBundle boundary as `_walk`."""
    if _is_bulk_leaf(v):
        fn(v)
        return
    if isinstance(v, ObsBundle):
        raise TypeError(_NESTED_OBS_MSG)
    if isinstance(v, dict):
        for x in v.values():
            _visit_bulk_leaves(x, fn)
    elif isinstance(v, (list, tuple)):  # NamedTuples descend like any tuple (no rebuild needed here)
        for x in v:
            _visit_bulk_leaves(x, fn)
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        for f in dataclasses.fields(v):
            if f.init:  # match _walk: init=False fields aren't externalized, so none to collect
                _visit_bulk_leaves(getattr(v, f.name), fn)


def _pack_val(
    v: Any,
    store: BulkStore,
    threshold: int,
    pool_stage: Callable[[Any], Any] | None = None,
) -> Any:
    def leaf(x: Any) -> Any:
        if isinstance(x, BulkRef) or _is_pool_slice(x):
            return x
        nb = _bulk_nbytes(x)  # tensor/ndarray only (never pickle these)
        if nb <= threshold:
            return x
        if pool_stage is not None:
            staged = pool_stage(x)
            if staged is not None:
                return staged
        return store.put(x, nbytes=nb, dtype=str(getattr(x, "dtype", "")))

    return _walk(v, leaf)


def _unpack_val(v: Any, transport: Transport, readers: int = 1) -> Any:
    """Materialize `v` via `transport`: fetch every `BulkRef` LEAF (recursively, via `_walk`),
    pass inline values through. Threads `readers` to each fetch so a shared ref pops only after
    every consumer has pulled it (refcounted `BulkStore`, DESIGN §2.4).

    Args:
        v: A packed field — a `BulkRef`, an inline value, or a structure nesting them.
        transport: Mover used to fetch the bulk.
        readers: Expected reader count threaded to each `transport.fetch` for refcounting.

    Returns:
        `v` rebuilt with every `BulkRef` leaf materialized; inline values unchanged.
    """

    def fetch(x: Any) -> Any:
        if isinstance(x, BulkRef):
            return transport.fetch(x, readers=readers)
        if _is_pool_slice(x):
            fetch_pool_slice = getattr(transport, "fetch_pool_slice", None)
            if fetch_pool_slice is None:
                raise RuntimeError("received a PoolSlice on a transport without a NIXL target pool")
            return fetch_pool_slice(x)
        return x

    return _walk(v, fetch)


def _pack_obs(
    o: ObsBundle | None,
    store: BulkStore,
    threshold: int,
    pool_stage: Callable[[Any], Any] | None = None,
) -> ObsBundle | None:
    if o is None:
        return None
    reps = {k: _pack_val(v, store, threshold, pool_stage) for k, v in o.representations.items()}
    return ObsBundle(representations=reps, modality_tags=o.modality_tags, id=o.id)


def _unpack_obs(o: ObsBundle | None, transport: Transport, readers: int = 1) -> ObsBundle | None:
    """Materialize every `BulkRef` representation in `o` via `transport`.

    Args:
        o: The bundle to unpack, or None.
        transport: Mover used to fetch bulk representations.
        readers: Expected reader count threaded to each fetch for refcounting.

    Returns:
        A fresh `ObsBundle` with representations materialized, or None if `o` is None.
    """
    if o is None:
        return None
    reps = {k: _unpack_val(v, transport, readers) for k, v in o.representations.items()}
    return ObsBundle(representations=reps, modality_tags=o.modality_tags, id=o.id)


def _map_transition(t: Transition, fn_val, fn_obs) -> Transition:
    # `advantage` is in this set because the channel is no longer scalar-only: it holds a scalar OR
    # a per-token tensor (see `schema/trajectory.py::Transition`). A scalar walks through untouched
    # (`_bulk_nbytes` returns -1 for a float), so GRPO's payload is byte-identical; a per-token
    # tensor that was NOT walked would ride inline through the control plane — no error, just the
    # bloat the bulk seam exists to prevent. `collect_bulk_refs::visit` mirrors this list and must
    # change with it, or prefetch silently misses those refs.
    return replace(
        t,
        obs=fn_obs(t.obs),
        next_obs=fn_obs(t.next_obs),
        action=fn_val(t.action),
        mask=fn_val(t.mask),
        targets=fn_val(t.targets),
        advantage=fn_val(t.advantage),
        policy_info={k: fn_val(v) for k, v in t.policy_info.items()},
        algo_extra={k: fn_val(v) for k, v in t.algo_extra.items()},
    )


def _pack_traj(
    tr: Trajectory,
    store: BulkStore,
    threshold: int,
    pool_stage: Callable[[Any], Any] | None = None,
) -> Trajectory:
    return replace(
        tr,
        transitions=[
            _map_transition(
                t,
                lambda v: _pack_val(v, store, threshold, pool_stage),
                lambda o: _pack_obs(o, store, threshold, pool_stage),
            )
            for t in tr.transitions
        ],
    )


def _unpack_traj(tr: Trajectory, transport: Transport, readers: int = 1) -> Trajectory:
    """Materialize all bulk fields of a trajectory's transitions via `transport`.

    Args:
        tr: The trajectory whose transition fields are unpacked.
        transport: Mover used to fetch bulk fields.
        readers: Expected reader count threaded to each fetch for refcounting.

    Returns:
        A fresh `Trajectory` with its tensor-bearing fields materialized.
    """
    return replace(
        tr,
        transitions=[
            _map_transition(
                t,
                lambda v: _unpack_val(v, transport, readers),
                lambda o: _unpack_obs(o, transport, readers),
            )
            for t in tr.transitions
        ],
    )


def _is_sequence_of(obj: Any, cls: type) -> bool:
    """Return whether ``obj`` is a non-empty list/tuple containing only ``cls`` values."""
    return isinstance(obj, (list, tuple)) and len(obj) > 0 and all(isinstance(x, cls) for x in obj)


def _rebuild_sequence_like(obj: list[Any] | tuple[Any, ...], values: list[Any]) -> list[Any] | tuple[Any, ...]:
    """Rebuild a payload batch without changing its list-vs-tuple contract."""
    return tuple(values) if isinstance(obj, tuple) else values


def collect_bulk_refs(obj: Any) -> list[BulkRef]:
    """All BulkRefs in a payload — lets a transport batch-prefetch them in one shot
    (§3.4 / §3.2 SQ-CQ spirit: one pull per consumer, not per field). A read-only fold
    (`_visit_bulk_leaves`), so the prefetch scan gathers refs without rebuilding anything —
    neither the nested field values NOR the per-transition shell. The field list mirrors
    `_map_transition`; the per-field nesting tests (policy_info + algo_extra) pin both routings,
    so a divergence fails in CI rather than silently.

    Public (not `_`-prefixed): the on_metadata split fast path in `workers/ray.py` (Task A6) also
    uses this ref-walk to diff a full packed payload against its kept slice — the BulkRefs present
    in the former but absent from the latter are exactly what a metadata-only split drops without
    ever fetching (DESIGN §2.3/§2.4). Kept Ray-free (comm/ constraint).

    Args:
        obj: A `Trajectory`, `ObsBundle`, or list/tuple of either to walk (other payloads yield none).

    Returns:
        Every `BulkRef` found in `obj`, in traversal order.
    """
    refs: list[BulkRef] = []

    def collect(x: Any) -> None:
        if isinstance(x, BulkRef):
            refs.append(x)

    def grab_obs(o: ObsBundle | None) -> None:
        if o is not None:
            for v in o.representations.values():
                _visit_bulk_leaves(v, collect)

    def visit(tr: Trajectory) -> None:
        # Mirror _map_transition's field set, but read-only: no replace()/dict rebuild on the
        # prefetch path (see docstring). obs/next_obs go through their bundle walk; the generic
        # tensor-bearing fields + policy_info/algo_extra values go straight to _visit_bulk_leaves.
        for t in tr.transitions:
            grab_obs(t.obs)
            grab_obs(t.next_obs)
            for v in (t.action, t.mask, t.targets, t.advantage):
                _visit_bulk_leaves(v, collect)
            for d in (t.policy_info, t.algo_extra):
                for v in d.values():
                    _visit_bulk_leaves(v, collect)

    if isinstance(obj, Trajectory):
        visit(obj)
    elif isinstance(obj, ObsBundle):
        grab_obs(obj)
    elif _is_sequence_of(obj, Trajectory):
        for x in obj:
            visit(x)
    elif _is_sequence_of(obj, ObsBundle):
        for x in obj:
            grab_obs(x)
    return refs


def _pool_slice_key(slice_: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Return a stable identity for a serializable pool descriptor."""
    return (
        slice_.owner_id,
        slice_.lease_id,
        getattr(slice_, "pool_kind", None),
        getattr(slice_, "page_id", None),
        getattr(slice_, "generation", None),
    )


def _collect_pool_slices(value: Any) -> list[Any]:
    """Collect pooled leaves without importing the Ray/NIXL implementation."""
    slices: list[Any] = []

    def visit(item: Any) -> None:
        if _is_pool_slice(item):
            slices.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field_ in dataclasses.fields(item):
                if field_.init:
                    visit(getattr(item, field_.name))

    visit(value)
    return slices


def _attach_pool_leases(value: Any) -> Any:
    """Keep source descriptors on trajectories through downstream tensor transforms."""
    if isinstance(value, Trajectory):
        found = _collect_pool_slices(value)
        existing = list(value.pool_leases)
        known = {_pool_slice_key(slice_) for slice_ in existing}
        leases = list(existing)
        for slice_ in found:
            key = _pool_slice_key(slice_)
            if key not in known:
                known.add(key)
                leases.append(slice_)
        return replace(value, pool_leases=tuple(leases))
    if _is_sequence_of(value, Trajectory):
        return _rebuild_sequence_like(value, [_attach_pool_leases(item) for item in value])
    return value


def pack_payload(
    obj: Any,
    store: BulkStore,
    threshold: int = 0,
    pool_stage: Callable[[Any], Any] | None = None,
) -> Any:
    """Producer side: externalize bulk fields → BulkRef in `store`. Handles a
    Trajectory or ObsBundle payload (or a list/tuple of either — trajs from rollout, obs
    prompts from env); passes loss scalars / None / anything else through."""
    if isinstance(obj, Trajectory):
        return _attach_pool_leases(_pack_traj(obj, store, threshold, pool_stage))
    if isinstance(obj, ObsBundle):
        return _pack_obs(obj, store, threshold, pool_stage)
    if _is_sequence_of(obj, Trajectory):
        packed = [_pack_traj(x, store, threshold, pool_stage) for x in obj]
        return _attach_pool_leases(_rebuild_sequence_like(obj, packed))
    if _is_sequence_of(obj, ObsBundle):
        return _rebuild_sequence_like(obj, [_pack_obs(x, store, threshold, pool_stage) for x in obj])
    return obj


def materialize_payload(obj: Any, transport: Transport, readers: int = 1) -> Any:
    """Consumer side: pull BulkRefs of Trajectory/ObsBundle payloads back via
    `transport` (a Mover — same-process ⇒ LocalStoreMover zero-copy); pass others through.
    If the transport offers `prefetch`, batch-load all refs first (one RPC per producer
    instead of one per field) — local refs stay zero-copy.

    `readers` (Task A5, DESIGN §2.4): the expected reader count for THIS pull, threaded
    down to `transport.fetch`/`prefetch` -> `BulkStore.fetch` so a bulk entry shared by
    multiple consumer cells (a `Dispatched` fan-out edge) is refcounted rather than popped
    on the first fetch. Default 1 == the old consume-once behavior — every existing caller
    (single-reader edges) is unaffected.

    TODO(Option B — broadcast transport for SPMD bulk fan-in): add a BroadcastTransport that,
    for a low-rank producer → high-rank SPMD consumer edge, delivers the bulk by an intra-replica
    NCCL broadcast instead of N independent pulls: on the replica's rank 0 it pulls the BulkRef
    (as today) then `dist.broadcast(tensor, src=rank0)` to ranks 1..K-1; the other ranks receive
    the broadcast rather than pulling. Same entry point here — it just dispatches on the transport
    type. Metadata stays tiny (the ref reaches every rank via bcast_aligned, see workers/grid.py);
    only Trajectory/ObsBundle tensor fields ride the broadcast. Wins when bulk is large (multimodal
    obs, §3.4).

    Args:
        obj: A `Trajectory`, `ObsBundle`, or list/tuple of either (other payloads pass through).
        transport: Mover pulling each `BulkRef` (same-process ⇒ zero-copy).
        readers: Expected reader count for this pull, threaded to `transport.fetch`/`prefetch`.

    Returns:
        `obj` with its bulk fields materialized (or unchanged when it carries no bulk).
    """
    prefetch = getattr(transport, "prefetch", None)
    if prefetch is not None:
        prefetch(collect_bulk_refs(obj), readers=readers)
    if isinstance(obj, Trajectory):
        return _unpack_traj(obj, transport, readers)
    if isinstance(obj, ObsBundle):
        return _unpack_obs(obj, transport, readers)
    if _is_sequence_of(obj, Trajectory):
        return _rebuild_sequence_like(obj, [_unpack_traj(x, transport, readers) for x in obj])
    if _is_sequence_of(obj, ObsBundle):
        return _rebuild_sequence_like(obj, [_unpack_obs(x, transport, readers) for x in obj])
    return obj
