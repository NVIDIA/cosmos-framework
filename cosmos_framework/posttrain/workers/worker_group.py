# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WorkerGroup / Replica / rank abstraction (DESIGN §1.1, §1.2, spec 2026-07-22 §2.1).

A WorkerGroup is defined by **ports + a body**. v0 degenerate form: 1 replica,
in-process body. The interface is the real one so a Ray-backed, multi-replica
implementation drops in later without touching role/runner code.

**Handle contract (spec 2026-07-22 §2.1/§2.2):** every WG method returns a `Handle`, and any
`Handle` passed as an arg is auto-resolved before the body runs. The SAME recipe
works sync/async/local/ray: only the bound *backend* differs.
- local sync ⇒ eager Handle (value in hand)
- ray       ⇒ Handle wrapping a Ray ObjectRef; `.result()` = `ray.get`
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional


class Handle:
    """Result handle (spec 2026-07-22 §2.1).

    Holds either an eager value (local sync) or a remote ref + getter (e.g. a Ray
    ObjectRef + ray.get). The recipe never cares which — it just gets resolved at
    the next WG boundary.
    """

    __slots__ = ("_value", "_ref", "_get", "_released")

    def __init__(
        self,
        value: Any = None,
        ref: Any = None,
        getter: Optional[Callable] = None,
    ):
        self._value = value
        self._ref = ref
        self._get = getter
        self._released = False

    def result(self) -> Any:
        if self._released:
            raise RuntimeError("result handle has been released")
        if self._ref is not None:
            return self._get(self._ref)  # e.g. ray.get(ref)
        return self._value

    def release(self) -> None:
        """Drop controller-owned result references."""
        self._value = None
        self._ref = None
        self._released = True

    wait = result


@dataclass(frozen=True)
class LocalPeerRef:
    """Binder-selected body in the receiving actor; never an RPC back to that actor."""

    name: str


class LocalPeerClient:
    """Synchronous body peer with an eager local implementation of ``call_async``."""

    def __init__(self, invoke: Callable[[str, tuple, dict], Any]) -> None:
        self._invoke = invoke

    def call_async(self, method: str, *args: Any, **kwargs: Any) -> Handle:
        return Handle(self._invoke(method, args, kwargs))

    def __getattr__(self, method: str) -> Callable[..., Any]:
        if method.startswith("__") and method.endswith("__"):
            raise AttributeError(method)
        return lambda *args, **kwargs: self._invoke(method, args, kwargs)


class CollectiveHandle(Handle):
    """Mesh result handle (DESIGN §1.1 replica×rank / spec 2026-07-22 §2.1). Holds one ObjectRef per
    (replica, rank); at the next WG boundary the backend aligns it onto each consumer
    actor via an AlignPolicy and forwards the producer ObjectRef into the actor, which
    resolves it actor-side. The controller forwards only the tiny ref — never the
    producer's metadata payload — and bulk is pulled peer-to-peer at materialize time
    (DESIGN §3.4/#7)."""

    def __init__(self, refs: dict, shape: tuple, getter=None):
        self._refs = refs  # {(r, k): ref}
        self.shape = shape  # (R, K)
        self._get = getter or (lambda x: x)
        self._released = False

    def ref_at(self, r: int, k: int):
        if self._released:
            raise RuntimeError("result handle has been released")
        return self._refs[(r, k)]

    def all(self) -> dict:
        if self._released:
            raise RuntimeError("result handle has been released")
        # Error semantics (intentionally different from as_completed): all() and result()
        # are **fail-fast** — the first cell that raises propagates out of this call. Use
        # as_completed() for a **fault-tolerant** drain that surfaces per-cell exceptions
        # as values instead (one failed cell doesn't abort the rest).
        return {rk: self._get(ref) for rk, ref in self._refs.items()}

    def as_completed(self):
        """Yield ``((r, k), value)`` as each cell's result resolves — the streaming
        analog of ``all()``, for live per-cell progress without the driver touching the
        backend. **Fault-tolerant (unlike the fail-fast ``all()``/``result()``):** a cell
        whose body raised yields the ``Exception`` as its value, so one failed cell never
        aborts the drain — the caller inspects each value with ``isinstance(v, Exception)``.
        Encapsulates ``ray.wait`` here (Ray-free
        import) so the control loop stays backend-agnostic; if the refs aren't Ray
        ObjectRefs (eager/local getter, or no Ray), falls back to insertion order."""
        if self._released:
            raise RuntimeError("result handle has been released")
        try:
            import ray

            refs = list(self._refs.values())
            use_wait = bool(refs) and all(isinstance(r, ray.ObjectRef) for r in refs)
        except ImportError:
            use_wait = False
        if use_wait:
            rk_by_ref = {ref: rk for rk, ref in self._refs.items()}
            pending = list(self._refs.values())
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                for ref in done:
                    try:
                        val = self._get(ref)
                    except Exception as ex:  # noqa: BLE001 — surface per-cell, don't abort the drain
                        val = ex
                    yield rk_by_ref[ref], val
            return
        for rk, ref in self._refs.items():  # eager / non-ObjectRef refs: resolve in order
            try:
                val = self._get(ref)
            except Exception as ex:  # noqa: BLE001
                val = ex
            yield rk, val

    def result(self):  # debug convenience: replica-0 rank-0; hot path never calls this
        if self._released:
            raise RuntimeError("result handle has been released")
        return self._get(self._refs[(0, 0)])

    wait = result

    def release(self) -> None:
        """Drop every controller-owned cell reference."""
        self._refs.clear()
        self._released = True


def release_result_handles(*values: Any) -> None:
    """Release result handles while ignoring concrete values."""
    for value in values:
        if isinstance(value, Handle):
            value.release()


class Scatter:
    """Dispatch marker (spec 2026-07-22 §2.4): wrap a per-cell sequence so the backend gives
    consumer cell ``(r, k)`` element ``seq[r]`` instead of broadcasting the whole
    sequence. ``len(seq)`` must equal the consumer's replica count. The mesh backend
    (`GridBackend`) interprets it; a local/eager backend gets the raw sequence back
    (single replica ⇒ the one cell receives ``seq[0]``). Ray-free."""

    __slots__ = ("seq",)

    def __init__(self, seq: Any):
        self.seq = list(seq)


class RankAligned:
    """Dispatch marker or registry spec for an exact per-rank producer-to-consumer edge.

    Unlike :class:`Sharded`, this marker declares that every producer rank owns a
    distinct value. The mesh backend therefore requires identical producer and
    consumer shapes and forwards ``(r, k)`` only to ``(r, k)``. This is the
    narrow escape hatch for rank-partitioned SPMD outputs; it must not be used for
    mirrored rank lanes.

    Args:
        arg: Wrapped producer handle or eager local value at a call site, or ``None`` when the
            exact-rank contract is declared in the dispatch registry.
    """

    __slots__ = ("arg",)

    def __init__(self, arg: Any = None) -> None:
        self.arg = arg


class Sharded:
    """Dispatch marker for cross-replica redistribution (spec 2026-07-22 §2.2).

    Declares "the producer's total output, redistributed evenly across my replicas".
    Units are producer cells per its declared sub-shape (row-major). The realization
    (1:1, reduce, or split) is selected at submit time by the routing function
    `route_sharded`, which inspects producer and consumer counts at call time.

    Args:
        arg: wrapped handle (if provided at a call site) or None (if used as a registry spec).
        reduce: optional string reference to a reduction callable (e.g. "module:fn"), required
                when P > R_c (more producer units than consumer replicas).
        split: optional string reference to a split callable (e.g. "module:fn"), required
               when R_c > P (more consumer replicas than producer units).
        on_metadata: if True, declares that split= operates only on the metadata skeleton
                     (fast-path optimization via field-level pulling; requires release protocol).
        select: optional ``"module:fn"`` reference to an ingress-selection predicate (e.g.
                mini/micro/dp chunking, spec 2026-08-12 §3.4). With ``select_params``, selection is
                opt-in when the call supplies one of those kwargs; without them, selection is an
                unconditional part of the registered edge contract.
        select_params: names of the SIBLING call kwargs that parameterise ``select`` (e.g.
                       ``("mini", "micro", "n_mini", "n_micro")``).
        rank_partitioned: whether every producer rank owns a distinct shard. The default rejects
                          multi-rank producers because ordinary SPMD rank lanes are mirrors.
    """

    __slots__ = ("arg", "reduce", "split", "on_metadata", "select", "select_params", "rank_partitioned")

    def __init__(
        self,
        arg: Any = None,
        *,
        reduce: str | None = None,
        split: str | None = None,
        on_metadata: bool = False,
        select: str | None = None,
        select_params: tuple[str, ...] = (),
        rank_partitioned: bool = False,
    ):
        """Construct the ``Sharded`` dispatch marker / registry spec.

        Args:
            arg: wrapped producer handle at a call site, or None as a registry spec.
            reduce: ``"module:fn"`` reduction ref, required when P > R_c (fan-in).
            split: ``"module:fn"`` split ref, required when R_c > P (fan-out).
            on_metadata: True ⇒ ``split=`` operates on the metadata skeleton only.
            select: ``"module:fn"`` ingress-selection predicate ref.
            select_params: names of the sibling call kwargs that parameterise ``select``.
            rank_partitioned: True when producer rank lanes carry distinct shards rather than
                mirrored SPMD results.
        """
        if type(rank_partitioned) is not bool:
            raise TypeError(f"rank_partitioned must be a bool, got {rank_partitioned!r}")
        self.arg = arg
        self.reduce = reduce
        self.split = split
        self.on_metadata = on_metadata
        self.select = select
        self.select_params = select_params
        self.rank_partitioned = rank_partitioned


class Mirror:
    """Dispatch marker for explicit replication (spec 2026-07-22 §2.2).

    Declares "replicate the producer to every consumer replica". Producer must have
    exactly 1 unit (P == 1); explicit use is for eval-style edges where the same
    reference flows to all consumers without redistribution.

    Args:
        arg: wrapped handle (if provided at a call site) or None (if used as a registry spec).
    """

    __slots__ = ("arg",)

    def __init__(self, arg: Any = None):
        """Construct the ``Mirror`` dispatch marker / registry spec.

        Args:
            arg: wrapped producer handle at a call site, or None as a registry spec.
        """
        self.arg = arg


@dataclass
class Dispatched:
    """Wire descriptor for routing decisions, forwarded by `submit` to `_Host.invoke`.

    Picklable, Ray-free. Constructed backend-side in `submit` and consumed in `_Host.invoke`;
    never passed as a call-site argument (see ``_resolve`` for the supported call-site markers).
    Captures the routing decision made at submit time: which producer units (refs), in what
    operation (one/reduce/split), with what adaptation function (if any), for which consumer
    replica's shard.

    Attributes:
        refs: list of ObjectRefs (one or more producer unit refs).
        op: routing operation ("one", "reduce", or "split").
        fn: optional string reference to the adaptation callable (reduce or split function),
            resolved by the actor. None ⇒ identity (direct ref forward).
        index: split shard index this replica keeps (``r % n`` under a split op; 0 otherwise —
            see ``workers.dispatch.split_index``), NOT the consumer replica index.
        n: scaling parameter (1 for "one", count for "reduce" or "split").
        readers: expected reader count (for refcounted bulk pull; spec 2026-07-22 §2.4).
        on_metadata: if True, adaptation operates on metadata only (skip bulk for dropped entries).
        select: optional ``"module:fn"`` ingress-selection predicate ref (spec 2026-08-12 §3.4),
            carried for the actor to apply after materialization. ``None`` ⇒ no selection — this
            is the DPO case, which shares this edge but never passes the control kwargs.
        select_kw: the resolved selection kwargs (e.g. ``{"mini": 0, "micro": 1, "n_mini": 1,
            "n_micro": 4}``) to call ``select`` with, or ``None`` when ``select`` is ``None``.
    """

    refs: list
    op: str
    fn: str | None
    index: int
    n: int
    readers: int
    on_metadata: bool = False
    select: str | None = None
    select_kw: dict | None = None


def take_control_kwargs(spec: Sharded | None, kw: dict) -> tuple[dict, dict | None]:
    """Split a call's kwargs into (body kwargs, selection kwargs) (spec 2026-08-12 §3.4).

    Selection is OPT-IN on the presence of ``spec.select_params``: DPO shares the SAME
    ``("trainer", "train", "batch")`` registry entry as GRPO but never passes mini/micro/n_mini/
    n_micro, so it must fall through to ``(kw, None)`` unchanged — an unconditional predicate
    would silently DP-slice DPO's batch.

    Args:
        spec: the ``Sharded`` registry entry for this edge, or None (unregistered / no hit).
        kw: the call's keyword arguments.

    Returns:
        ``(body_kw, select_kw)``. ``select_kw`` is ``None`` when the spec declares no selection
        or the call passes none of its ``select_params`` — which is what keeps DPO's edge
        untouched. Otherwise ``body_kw`` has the selection kwargs stripped out.
    """
    if spec is None or not getattr(spec, "select", None):
        return kw, None
    if not spec.select_params:
        return kw, {}
    present = {name: kw[name] for name in spec.select_params if name in kw}
    if not present:
        return kw, None
    return {k: v for k, v in kw.items() if k not in present}, present


def scatter(seq: Any) -> Scatter:
    """Mark ``seq`` for per-replica scatter in a WG method call (spec 2026-07-22 §2.4).

    ``sim.run_tasks(scatter(bins), policy=policy)`` hands cell ``r`` its ``bins[r]``.

    Args:
        seq: per-cell sequence; consumer cell ``(r, k)`` receives ``seq[r]``.

    Returns:
        A ``Scatter`` marker wrapping ``seq``.
    """
    return Scatter(seq)


def _local_select(role: str, method: str, args: tuple, kw: dict) -> tuple[tuple, dict]:
    """Apply the local (backend-less) path's ingress selection, if any (spec 2026-08-12 §3.4).

    The local path never reaches ``GridBackend._actor_args``/``_align_one`` — ``WorkerGroup``'s
    ``self._backend is None`` branch invokes the body directly — so it must apply the SAME
    ``take_control_kwargs`` split and consume the selection itself, or the control kwargs
    (``mini``/``micro``/``n_micro``) reach the body and raise ``TypeError`` on the path every
    CPU example and L0 test actually runs. Under torchrun the local processes form the policy's one
    model-DP world, so selection uses the live default-process-group rank/world. This is the same
    flattened ``dp_replicate * dp_shard`` coordinate used by the Ray path; without it every local
    rank would train the complete global batch.

    Args:
        role: this WG's role (the registry ``kind``).
        method: the body method being called.
        args: positional call args (already ``Scatter``/``Handle``-resolved).
        kw: keyword call args (already ``Scatter``/``Handle``-resolved).

    Returns:
        ``(args, kw)`` with the selected slice substituted for the sharded argument (if any) and
        the control kwargs stripped; unchanged (byte-identical) when no edge declares a
        selection or the call passes none of its control kwargs — the DPO case.
    """
    from cosmos_framework.posttrain.roles.core import dispatch as registry  # lazy: avoid a

    # module-level cycle (roles.core.dispatch imports Sharded/Mirror from this module).
    pos_names = registry.bind_param_names(role, method, args, kw)[: len(args)]
    target_name, spec = None, None
    for name in list(pos_names) + list(kw.keys()):
        if name is None:
            continue
        candidate = registry.lookup(role, method, name)
        if candidate is not None and getattr(candidate, "select", None):
            target_name, spec = name, candidate
            break
    body_kw, select_kw = take_control_kwargs(spec, kw)
    if select_kw is None:
        return args, body_kw
    fn = registry.resolve_fn(spec.select)
    import torch.distributed as dist

    dp_rank = dist.get_rank() if dist.is_initialized() else 0
    dp_world = dist.get_world_size() if dist.is_initialized() else 1
    select_kw = dict(select_kw, dp_rank=dp_rank, dp_world=dp_world)
    if target_name in pos_names:
        i = pos_names.index(target_name)
        args = args[:i] + (fn(args[i], **select_kw),) + args[i + 1 :]
    else:
        body_kw = dict(body_kw)
        body_kw[target_name] = fn(body_kw[target_name], **select_kw)
    return args, body_kw


def _resolve(x: Any) -> Any:
    """Resolve a WG-boundary arg: pass dispatch markers through for per-actor alignment,
    materialize a plain ``Handle``, and leave any other value as-is.

    Args:
        x: the argument to resolve.

    Returns:
        The dispatch marker unchanged, the ``Handle``'s resolved value, or ``x`` itself.
    """
    if isinstance(x, (CollectiveHandle, Scatter, RankAligned, Sharded, Mirror)):
        return x  # pass through to the backend for per-actor alignment (not materialized at the WG boundary)
    return x.result() if isinstance(x, Handle) else x


def _resolve_local_dispatch(x: Any) -> Any:
    """Resolve a dispatch marker for the one-cell eager backend."""
    if isinstance(x, Scatter):
        return x.seq[0]
    if isinstance(x, RankAligned):
        inner = x.arg
        return inner.result() if isinstance(inner, Handle) else inner
    if isinstance(x, WorkerGroup):
        if x._backend is not None or x.body is None:
            raise ValueError(f"local peer {x.name!r} must have a locally bound body")
        return LocalPeerClient(lambda method, args, kwargs: getattr(x.body, method)(*args, **kwargs))
    return x


class WorkerGroup:
    """Placement-agnostic proxy for all execution replicas and ranks of one role.

    A bound Ray ``GridBackend`` fans methods over the role's resolved ``(R, K)`` cells; an unbound
    group calls its in-process body eagerly. Model parallelism is separate declaration state and
    never changes the WorkerGroup's execution-replica axis.
    """

    def __init__(
        self,
        name: str,
        role: str,
        body: Any = None,
        replicas: int = 1,
        ranks: int = 1,
        needs_bulk: bool = False,
        model_parallelism: tuple[int, int] | None = None,
        dispatch_profile: str | None = None,
        **placement: Any,
    ) -> None:
        self.name: str = name
        self.role: str = role
        self.body: Any = body
        self.replicas: int = replicas
        self.ranks: int = ranks  # rank width K inside one execution replica
        # Model-owned data-parallel factorization `(dp_replicate, dp_shard)`. It is explicit state
        # on the role declaration, not placement metadata: placement owns `(R, K)` while the model
        # decides how its K-rank Torch world is factored. Only trainer roles set this today.
        self.model_parallelism: tuple[int, int] | None = model_parallelism
        if dispatch_profile is not None and (
            not isinstance(dispatch_profile, str)
            or not dispatch_profile
            or dispatch_profile.strip() != dispatch_profile
        ):
            raise ValueError(
                "WorkerGroup.dispatch_profile must be a non-empty, whitespace-normalized string when set, "
                f"got {dispatch_profile!r}"
            )
        # Algorithm-specific method contracts refine the generic role dispatch registry without
        # putting routing markers in the control loop. Placement remains an independent concern.
        self.dispatch_profile: str | None = dispatch_profile
        # needs_bulk: does this WG's body COMPUTE on bulk fields (so its inputs must be
        # materialized), or does it only route/filter on metadata? A consumer-declared
        # port property (DESIGN §3.4 field-level / move-once-where-needed); the bind/
        # transport layer pulls bulk only at WGs that set this. Default False ⇒ BulkRefs
        # flow through untouched (e.g. reward-on-text, buffer holds metadata, §5).
        self.needs_bulk: bool = needs_bulk
        self.placement: dict[str, Any] = placement  # pin_to / colocate_with — consumed by Placement (§4)
        self._backend: Any = None  # set by bind_device_groups(); None ⇒ local sync-eager

    def _bind_backend(self, backend: Any) -> None:
        self._backend = backend

    # Placement introspection (DESIGN §4.1): real methods (not routed through
    # __getattr__ to a body) so a driver reads node IP / GPU from the bound backend
    # without reaching into `_backend`. Empty for the local/eager backend (one
    # in-process cell, no placement to report).
    def hosts(self) -> dict:
        be = self._backend
        return be.hosts() if be is not None and hasattr(be, "hosts") else {}

    def gpus(self) -> dict:
        be = self._backend
        return be.gpus() if be is not None and hasattr(be, "gpus") else {}

    def connection_registry(self) -> dict:
        """Return the bound data-plane store-owner registry."""
        be = self._backend
        return be.connection_registry() if be is not None and hasattr(be, "connection_registry") else {}

    def _submit(self, method: str, args: tuple, kw: dict) -> Handle:
        """Resolve ``Handle`` args, then dispatch ``method`` to the bound backend
        (``_backend is None`` ⇒ local sync-eager on the in-process body).

        Args:
            method: body method name to call.
            args: positional args (Handles auto-resolved; ``Scatter`` handled per backend).
            kw: keyword args (same handling).

        Returns:
            A ``Handle`` wrapping the call result.
        """
        # Auto-resolve Handle args (v0/ray: eager — for ray this is a controller-side
        # ray.get). TODO(v1+): pass ObjectRefs straight to the actor (Ray resolves them
        # inside) to avoid the controller round-trip + enable real overlap.
        args = tuple(_resolve(a) for a in args)
        kw = {k: _resolve(v) for k, v in kw.items()}
        if self._backend is None:
            # Local sync-eager has one cell: Scatter degenerates to element zero and
            # RankAligned unwraps its one eager producer value.
            args = tuple(_resolve_local_dispatch(a) for a in args)
            kw = {k: _resolve_local_dispatch(v) for k, v in kw.items()}
            args, kw = _local_select(self.role, method, args, kw)  # spec 2026-08-12 §3.4
            return Handle(getattr(self.body, method)(*args, **kw))  # default local sync-eager
        submit = self._backend.submit
        if "kind" in inspect.signature(submit).parameters:
            # GridBackend-only (Task A4, spec 2026-07-22 §2.1): thread this WG's role — the
            # dispatch registry's "kind" in `(kind, method, param)` — through so `submit`
            # can look up per-edge routing specs. RayBackend/ReplicaBackend don't declare
            # a `kind` parameter (untouched by this task) and fall through to the plain
            # call below, unaffected.
            return submit(
                self.name,
                self.body,
                method,
                args,
                kw,
                kind=self.role,
                dispatch_profile=self.dispatch_profile,
            )
        return submit(self.name, self.body, method, args, kw)

    def __getattr__(self, method: str):
        # Only reached for names not on the WG itself → treat as a body method name.
        return lambda *args, **kw: self._submit(method, args, kw)

    def __repr__(self) -> str:
        return f"WorkerGroup(name={self.name!r}, role={self.role!r}, replicas={self.replicas})"
