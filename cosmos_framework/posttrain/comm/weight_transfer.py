# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""WeightTransfer — trainer → rollout weight sync on the weight plane (DESIGN §8.2 #4).

The seam is push_weights()/apply_weights() on the role bodies; this module holds the swappable
pieces: a WeightSync (holds a Mover, the transfer primitive) + a WeightMapper (param key/layout).
The payload rides the existing data plane as a normal arg (CollectiveHandle -> dp_aligned), so no
new transport wiring is needed. Movers are picked from resolved actor/device grouping and attached to the bodies
(workers.ray._Host.set_weight_backend).

v0: shared (tier-3 no-op), copy (CPU/object-store; the CI path), ipc (tier-2 same-GPU, see
comm.transfer.ipc). Hooks: nccl (disaggregated, bind-time communicator) and a resharding WeightMapper
(different policy/rollout parallelisms). Async (mode='buffered' + versioned-buffer lease) is a hook.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.comm.transfer.mover import MoveHandle, Mover
from cosmos_framework.posttrain.comm.transfer.movers import IpcMover, make_mover
from cosmos_framework.posttrain.utils.profile import GpuSpan

WireTensor = tuple[str, tuple[int, ...], str]
WireManifest = list[WireTensor]
BucketPlan = list[list[WireTensor]]
DestinationNames = tuple[str, ...] | None


@dataclass(frozen=True)
class WeightWitnessSpec:
    """One source-to-packed-destination mapping selected for inline live-value proof."""

    wire_name: str
    destination_name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    packed_part: str


def _net(model: Any) -> Any:
    """The trainable module for weight sync. `LLMTrainer` may hold an FSDP `HFModel` WRAPPER whose
    state_dict keys are prefixed (`net.…`), while the rollout loads into the RAW module — so both
    sides must unwrap `.net` or the keys won't match. Plain modules have no `.net` (identity)."""
    return getattr(model, "net", model)


def shared_model_identity(model: Any) -> str:
    """Return a process-local token shared only by the exact same underlying model object.

    The token must not live on the module: deepcopy/pickle would copy such an attribute onto
    independent parameter storage and make a byte-free SharedMover silently skip required transfer.
    Both compared models remain live during bind, so `(pid, id(net))` is an exact object-identity proof.
    """
    net = _net(model)
    return f"{os.getpid()}:{id(net)}"


def full_state_dict(model: Any) -> dict:
    """Materialize a model's params as FULL (unsharded) detached tensors: an FSDP2 DTensor is
    all-gathered via .full_tensor() (a collective — every policy rank must call together, §2.3);
    a plain tensor passes through. No physical clone beyond what full_tensor() allocates; the
    durable snapshot/copy is the mover's job (§2.1). Unwraps `.net` so a wrapped HFModel exports the
    same (raw-module) keys the rollout loads into."""
    out = {}
    for k, v in _net(model).state_dict().items():
        ft = getattr(v, "full_tensor", None)
        full = ft() if callable(ft) else v
        out[k] = full.detach()
    return out


BUCKET_BYTES = 512 * 1024 * 1024  # 512 MiB: bounds the streaming transient while amortizing overhead


def _dtype_bytes(dtype_str: str) -> int:
    return 2 if ("16" in dtype_str) else (8 if "64" in dtype_str else 4)


def declared_wire_dtype(transfer_dtype: str | None) -> str | None:
    """Normalize a transfer-dtype NAME to the manifest's `torch.<name>` spelling, or None.

    The manifest dtype string is consumed by `_dtype_bytes` and by `alloc_recv_buffers` and is
    compared in tests, so accepting either spelling on input while emitting exactly one keeps that
    surface unchanged.
    """
    return f"torch.{transfer_dtype.removeprefix('torch.')}" if transfer_dtype else None


def apply_wire_dtype(model: Any, manifest: WireManifest, transfer_dtype: str | None) -> WireManifest:
    """Restate a manifest's FLOATING PARAMETER dtypes as the declared wire dtype.

    WHY THIS EXISTS. The trainer's STORAGE dtype need not be the inference engine's. With
    `rl.master_dtype="float32"` -- what the reference GRPO recipe runs -- the policy holds fp32
    parameters while the vLLM engine is bf16, so a manifest reporting fp32 sizes every bucket at 2x
    and hands the engine fp32 buffers for bf16 params. The declared dtype is authoritative end to
    end (`bucket_plan` sizes from it, `alloc_recv_buffers` allocates from it, `bucket_full_tensors`
    casts to it), so restating it here is sufficient to make both ends agree. cosmos-rl draws the
    same boundary with `train.transfer_dtype`, applied on both send and receive.

    PARAMETERS ONLY, and only floating ones. A transfer dtype is a statement about how WEIGHTS
    travel; it has nothing to say about a causal mask, a step counter, or BatchNorm running
    statistics, and applying it to them corrupts state with no error anywhere -- an int64 counter
    rounds through bf16, fp32 running stats come back permanently truncated. Excluding non-float
    buffers alone is not enough: floating buffers are exactly the case that survives a dtype check
    while still being wrong.

    `remove_duplicate=False` on the parameter scan: a TIED parameter (classically `lm_head.weight`
    sharing storage with the input embedding) appears once in `named_parameters()` by default but
    under BOTH names in `state_dict()`. With the default the alias falls out of this set, keeps its
    storage dtype in the plan, and is sent uncast -- one wire entry at a different dtype from the
    tensor it aliases, and an unnecessary fp32 transfer of one of the largest matrices in the model.

    Args:
        model: the producer body, used only to enumerate parameter names.
        manifest: the mapper's own sorted manifest.
        transfer_dtype: dtype NAME (e.g. "bfloat16"), or None to leave every entry as reported.

    Returns:
        The manifest with floating-parameter dtypes restated, or `manifest` unchanged when no
        transfer dtype is declared.
    """
    declared = declared_wire_dtype(transfer_dtype)
    if declared is None:
        return manifest
    param_names = {name for name, _ in _net(model).named_parameters(remove_duplicate=False)}
    out: WireManifest = []
    for name, shape, dtype in manifest:
        is_float_param = name in param_names and dtype.startswith(("torch.float", "torch.bfloat", "torch.half"))
        out.append((name, shape, declared if is_float_param else dtype))
    return out


def _identity_wire_manifest(model: Any) -> WireManifest:
    """Identity mapper metadata: sorted raw state-dict names and FULL tensor layouts.

    Padded params are reported at their UNPADDED width, matching what `_identity_export_bucket`
    actually stages: the manifest and the tensors have to describe the same thing or the staging
    buffer is sized for one and filled with the other.
    """
    net = _net(model)
    rows = _unpadded_rows(model)
    manifest: WireManifest = []
    for name, tensor in sorted(net.state_dict().items()):
        shape = tuple(tensor.shape)
        if name in rows:
            shape = (rows[name],) + shape[1:]
        manifest.append((name, shape, str(tensor.dtype)))
    return manifest


def _check_name_sorted(names: list, where: str) -> None:
    """Sorted-name order is LOAD-BEARING for the collective path: NcclMover.stage broadcasts each
    bucket in sorted(name) order while the consumer lays out recv buffers in PLAN order — an unsorted
    plan would silently receive same-shape params into each other's buffers (q/k/v swap, no error).
    Runs once per plan creation/installation (plans are cached), never on the per-bucket hot path.

    Args:
        names: the param names to verify are strictly name-sorted (`a < b` for every adjacent pair).
        where: caller label prefixed to the error message identifying the plan source.
    """
    for a, b in zip(names, names[1:]):
        if not a < b:
            raise ValueError(
                f"{where}: param names must be strictly name-sorted, got {b!r} after {a!r} — "
                f"unsorted plans silently corrupt collective transfers (recv-buffer order is plan "
                f"order but broadcast order is sorted order)"
            )


def bucket_plan(manifest: WireManifest, bucket_bytes: int = BUCKET_BYTES) -> BucketPlan:
    """Partition a sorted manifest into ordered buckets, each ≤ bucket_bytes (except a single param
    larger than bucket_bytes, which forms its own bucket). Deterministic pure function → every rank
    computes the same plan from the same manifest (design §3). Returns list[list[(name, shape, dtype)]].
    Raises on an unsorted manifest (see _check_name_sorted — ordering is a transfer invariant).

    Args:
        manifest: sorted `[(name, shape, dtype_str), ...]` param metadata (e.g. from `model_manifest`).
        bucket_bytes: soft per-bucket byte cap; a single param exceeding it forms its own bucket.

    Returns:
        `list[list[(name, shape, dtype)]]` — ordered buckets partitioning the manifest.
    """
    _check_name_sorted([name for name, _s, _d in manifest], "bucket_plan(manifest)")
    plan, cur, cur_bytes = [], [], 0
    for name, shape, dtype in manifest:
        n = 1
        for d in shape:
            n *= d
        pb = n * _dtype_bytes(dtype)
        if cur and cur_bytes + pb > bucket_bytes:
            plan.append(cur)
            cur, cur_bytes = [], 0
        cur.append((name, tuple(shape), dtype))
        cur_bytes += pb
    if cur:
        plan.append(cur)
    return plan


def _unpadded_rows(model: Any) -> dict:
    """`{param name: dim-0 rows}` for params a model padded to suit its own sharding.

    FSDP pads a Linear whose out_features falls below the shard count so no rank gets an empty
    shard, and a forward hook slices the output back -- invisible to that model, but `state_dict`
    reports the PADDED shape while a consumer that never sharded holds the unpadded one. A model
    that pads declares it by publishing `unpadded_rows`, so the manifest and the staged tensors
    both describe what the consumer actually takes. Absent means nothing was padded, which is
    every model but the VLA expert today.
    """
    return getattr(model, "unpadded_rows", {})


def _identity_export_bucket(model: Any, bucket: list[WireTensor]) -> dict[str, Any]:
    """Identity mapper export for one bucket of raw state-dict tensors."""
    sd = _net(model).state_dict()
    rows = _unpadded_rows(model)
    out: dict[str, Any] = {}
    for name, _shape, _dtype in bucket:
        v = sd[name]
        ft = getattr(v, "full_tensor", None)
        full = (ft() if callable(ft) else v).detach()  # [*weight]
        # Drop the sharding pad AFTER the all-gather: the pad exists to make the shards even, so
        # it has to survive until the tensor is whole again.
        if name in rows:
            full = full[: rows[name]]
        out[name] = full
    return out


class WeightMapper:
    """Trainer param layout/keys -> rollout. v1: gather FSDP/TP shards to FULL tensors on export
    (rollout slices its own shard in its sink, §2.4); identity for plain models. import = name-keyed
    load. TODO(v2): key remap + DimSliceInfo reshard for asymmetric parallelism."""

    # True because nothing here records a dtype it later checks: `wire_manifest` reports whatever the
    # tensors say, the identity `export_bucket` ignores the dtype its bucket carries, and this mapper
    # publishes no witness -- so the cast in `bucket_full_tensors` is free to satisfy whatever the
    # plan declares. Subclasses do NOT inherit this answer (`accepts_wire_dtype` reads the concrete
    # class's own `__dict__`); one that is equally indifferent should repeat the line, and one that
    # records a dtype anywhere -- export guard, manifest, or witness -- should not.
    accepts_wire_dtype = True

    def wire_manifest(self, model: Any) -> WireManifest:
        """Return the sorted full-tensor metadata placed on the wire, at each tensor's own dtype.

        A mapper reports LAYOUT, not wire precision: `model_manifest` restates floating parameter
        dtypes afterwards from the declared transfer dtype, so no mapper has to know that feature
        exists (see `apply_wire_dtype` for why the trainer's storage dtype and the engine's need
        not match).
        """
        return _identity_wire_manifest(model)

    def export_bucket(self, model: Any, bucket: list[WireTensor]) -> dict[str, Any]:
        """Materialize one wire bucket without changing names or tensor layouts."""
        return _identity_export_bucket(model, bucket)

    def destination_names(self, model: Any) -> DestinationNames:
        """Return exact live-sink names when this mapper can prove them.

        The identity mapper deliberately returns ``None``: ordinary HF checkpoint names may be
        packed by vLLM, and preserving the existing post-load coverage behavior is safer than
        pretending raw wire names are live engine names.
        """
        del model
        return None

    def witness_spec(self, model: Any) -> WeightWitnessSpec | None:
        """Return no live-value witness for the legacy identity path."""
        del model
        return None

    def export_state(self, model: Any) -> dict:
        return full_state_dict(model)

    def import_state(self, model: Any, state: dict, strict: bool = True) -> None:
        # Unwrap `.net` to match full_state_dict's raw-module keys (a wrapped HFModel would otherwise
        # reject the un-prefixed keys); identity for a plain module. strict=False for streaming: a
        # single bucket carries only some params, so missing keys are expected (later buckets fill them).
        _net(model).load_state_dict(state, strict=strict)


def resolve_weight_mapper(model: Any) -> Any:
    """Return a model-owned streaming mapper, or the byte-identical identity mapper.

    The seam is structural on purpose: generic transfer code never imports a model family. A model
    that publishes ``weight_mapper`` must implement the complete streaming contract so a typo cannot
    silently fall back to raw names for only one phase of the transfer.
    """
    mapper = getattr(model, "weight_mapper", None)
    if mapper is None:
        return WeightMapper()
    missing = [
        method
        for method in ("wire_manifest", "export_bucket", "destination_names")
        if not callable(getattr(mapper, method, None))
    ]
    if missing:
        raise TypeError(f"model-owned weight_mapper {type(mapper).__name__} is missing streaming hooks {missing}")
    return mapper


def accepts_wire_dtype(mapper: Any) -> bool:
    """Whether `mapper` declares that it can be handed a plan restated to a wire dtype.

    THE DECLARATION MUST BE ON THE MAPPER'S OWN CLASS. Not inherited, not on an instance, not
    inferred. Two earlier versions of this predicate tried to infer the answer -- first from a plain
    `getattr`, then from whether `export_bucket` was still the inherited identity -- and review found
    a way around each: a subclass inherits the base's `True`; a subclass of a compatible subclass
    inherits ITS `True` before any structural check runs; an instance attribute replaces the exported
    callable without changing the class; and a mapper can keep the identity export while overriding
    `wire_manifest` or the witness, whose storage dtype `roles/rollout/vllm_weights.py` re-checks on
    receive. The property being asked about is SEMANTIC -- "does every dtype this mapper records stay
    consistent when the plan restates one" -- and no structural signal implies it.

    So it is declared, once, by the class that knows. The cost is that a subclass which changes
    nothing relevant must still add one line; the benefit is a rule with no way around it. The
    refusal that follows a False is loud and names what to add.

    Args:
        mapper: the mapper to interrogate; may be duck-typed rather than a `WeightMapper`.

    Returns:
        True only when `type(mapper)` itself declares `accepts_wire_dtype` truthy.
    """
    return bool(type(mapper).__dict__.get("accepts_wire_dtype", False))


def model_manifest(model: Any, mapper: Any | None = None, transfer_dtype: str | None = None) -> WireManifest:
    """Return the mapper-owned sorted wire manifest without materializing tensors.

    The wire dtype is applied HERE rather than inside `wire_manifest`, so that a mapper's own
    signature stays `(self, model)` and a mapper that keeps the inherited `export_bucket` honors the
    declaration without knowing it exists. Pushing it into the protocol instead would both break
    existing implementations and make the declaration silently droppable by any mapper that forgot
    to re-apply it. A mapper that REPLACES `export_bucket` is the one case that cannot be handled
    transparently, and it is refused below rather than left to fail later.

    Args:
        model: the producer body, forwarded to the mapper and used to enumerate parameter names.
        mapper: the mapper to consult; resolved from the model when None.
        transfer_dtype: dtype NAME (e.g. "bfloat16") to restate floating PARAMETER dtypes as, or
            None to report each tensor's own storage dtype.

    NOT every mapper can accept a restated plan. A mapper that validates each exported bucket
    against its own recorded metadata -- `Qwen3VLWeightMapper.export_bucket` is the one in-tree --
    reads a changed dtype as corrupted metadata and raises. Such a mapper is refused HERE, where the
    config knob can still be named, rather than several layers down inside an export with a message
    about "wire metadata" that says nothing about `rl.transfer_dtype`. Only an actually-changed
    manifest is refused, so a mapper whose storage dtype already equals the wire dtype is unaffected
    -- which is every recipe today, since `transfer` defaults to `master`.

    Returns:
        The sorted wire manifest, with floating-parameter dtypes restated when a transfer dtype is
        declared.

    Raises:
        ValueError: when a declaration would restate the manifest for a mapper that does not accept
            one.
    """
    resolved = mapper if mapper is not None else resolve_weight_mapper(model)
    manifest = list(resolved.wire_manifest(model))
    restated = apply_wire_dtype(model, manifest, transfer_dtype)
    if restated != manifest and not accepts_wire_dtype(resolved):
        changed = next((n for (n, _s, d), (_n2, _s2, d2) in zip(manifest, restated) if d != d2), "?")
        raise ValueError(
            f"{type(resolved).__name__} does not declare `accepts_wire_dtype`, but "
            f"transfer_dtype={transfer_dtype!r} restates {changed!r} (and others). The declaration "
            f"is required on the mapper's own class and is never inherited, so this may simply be a "
            f"mapper that has not been reviewed for it rather than one that is incompatible. To "
            f"resolve: either leave rl.transfer_dtype equal to rl.master_dtype for this model, or "
            f"check that this mapper never records a dtype it later validates -- an export guard "
            f"comparing the bucket against a stored spec, a manifest it re-checks, or a witness "
            f"whose dtype the rollout sink re-checks -- and then set `accepts_wire_dtype = True` on "
            f"it. Qwen3VLWeightMapper is the in-tree example that does all three and therefore "
            f"cannot declare it without being changed first."
        )
    return restated


def bucket_full_tensors(model: Any, bucket: list[WireTensor], mapper: Any | None = None) -> dict[str, Any]:
    """Materialize one mapper-owned FULL-tensor bucket, cast to the dtype the PLAN declares.

    THE PLAN IS AUTHORITATIVE. `bucket_plan` sized the buckets from these dtypes and the receiver
    allocated its buffers from them, so shipping a tensor at any other dtype writes the wrong number
    of bytes into a buffer that cannot grow. Casting here rather than inside each mapper's
    `export_bucket` is the same choice `model_manifest` makes for the manifest: it is the one layer
    every mapper funnels through, so no mapper has to know the feature exists.

    Per-ENTRY, not one dtype for the bucket: `apply_wire_dtype` already decided which entries the
    wire dtype applies to (floating parameters) and left int/bool counters, masks, and floating
    buffers reporting their own storage dtype. Casting each tensor to the dtype its own entry
    declares therefore reproduces that exclusion exactly, with no second copy of the rule to drift.

    Args:
        model: the producer body.
        bucket: one plan bucket of `(name, shape, dtype_str)` entries.
        mapper: the mapper to export through; resolved from the model when None.

    Returns:
        Name-keyed full tensors, each at its entry's declared dtype.
    """
    import torch

    resolved = mapper if mapper is not None else resolve_weight_mapper(model)
    exported = dict(resolved.export_bucket(model, bucket))
    for name, _shape, dtype_str in bucket:
        tensor = exported.get(name)
        if tensor is None:
            continue
        declared = getattr(torch, dtype_str.split(".")[-1])
        if tensor.dtype != declared:
            exported[name] = tensor.to(declared)
    return exported


@dataclass
class WeightPayload:
    """What export() returns / apply() consumes: a version stamp (staleness, DESIGN §1.4) + the
    shared-primitive MoveHandle carrying the weights."""

    version: int
    handle: MoveHandle | None = None


@dataclass(frozen=True)
class WeightPushTiming:
    """Producer-side wall clock of one pushed stream, kept on the backend as ``last_push_timing``.

    The phases partition ``total_s``. Each is measured around the rank-converged step of
    ``push_stream_to_peer`` that owns it, so it includes the barrier ending that step and, through the
    barrier's ``.item()``, the device work the step launched. ``deliver_s[i]`` is bucket ``i``'s round trip
    through every routed lane (fetch, sink, consumer_done, Ray); ``ready_s`` is the lanes'
    ``mark_weight_version_ready`` round, where a receiver's live-value verification lands, and
    includes release of any mapper export session. ``plan_s`` includes entering that session,
    including Qwen3-VL's one-time shard state-dictionary build.
    ``gather_gpu_s[i]`` is the CUDA span of bucket ``i``'s materialization (all-gather + cast); ``None``
    off-GPU or when the span had not completed by the time the record was built.
    """

    version: int
    plan_s: float
    coverage_s: float
    materialize_s: tuple[float, ...]
    gather_gpu_s: tuple[float | None, ...]
    deliver_s: tuple[float, ...]
    ready_s: float
    total_s: float


@dataclass(frozen=True)
class BucketApplyTiming:
    """Consumer-side wall clock of one received bucket (``WeightSync.apply_timings``): the mover fetch, then
    the sink together with ``consumer_done``."""

    bucket: int
    fetch_s: float
    sink_s: float


@dataclass(frozen=True)
class WeightTargetSet:
    """The routed rollout clients assigned to one policy cell.

    An empty set is an explicit idle route: the policy actor still enters every FSDP
    full-tensor collective but stages no NIXL payload. Keeping that distinct from ``None`` prevents
    a missing bind route from silently degrading to the general dp-aligned peer path.
    """

    peers: tuple[Any, ...]


def _strict_plan_coverage(consumer: Any, be: Any) -> None:
    """STRICT plan/model 1:1 coverage check, run ONCE at plan-install time BEFORE any tensor moves
    (nothing is half-mutated). Buckets load strict=False by necessity (each is partial), so a
    policy/rollout key divergence (resized embedding, adapter keys, wrong checkpoint) would otherwise
    silently half-sync and still stamp a fresh version. A load_state_dict sink must be covered 1:1 by
    the plan; a vLLM engine sink is exempt by construction (its param names differ — fusion — and its
    own load_weights raises on unknown names), detected as ``.model`` without a state_dict.

    Args:
        consumer: the consumer body whose ``.model`` sink is checked (a state_dict-less sink is exempt).
        be: the weight backend supplying ``be.plan()`` — the installed bucket plan to cover.
    """
    net = _net(getattr(consumer, "model", None))
    if net is not None and hasattr(net, "state_dict"):
        plan_names = {name for bucket in be.plan() for (name, _s, _d) in bucket}
        model_names = set(net.state_dict().keys())
        from cosmos_framework.posttrain.comm.weight_validation import check_weight_names

        check_weight_names(
            model_names, plan_names, context="bucket plan does not cover the consumer model 1:1 (strict)"
        )


def _resolved(x: Any) -> Any:  # local-backend peer calls return Handles; PeerClient returns values
    """Unwrap a peer call result: call ``x.result()`` if present (a local-backend Handle), else
    return ``x`` unchanged (a Grid PeerClient already yields resolved values).

    Args:
        x: a peer call result — a duck-typed Handle (has ``.result()``) or an already-resolved value.

    Returns:
        The resolved value.
    """
    r = getattr(x, "result", None)
    return r() if callable(r) else x


def _releases_staged_after_sync(be: Any) -> bool:
    """Return an optional backend's end-of-stream staging-release preference.

    WeightSync implements this hook, while legacy and duck-typed backends do not. Treating a
    missing hook as false preserves their pre-handshake behavior.
    """
    releases_staged = getattr(be, "releases_staged_after_sync", None)
    return bool(releases_staged()) if callable(releases_staged) else False


def pull_stream_from_peer(consumer: Any, peer: Any) -> None:
    """Consumer side of the role-edge weight channel: ``rollout.apply_weights(policy)`` lands here,
    with ``peer`` = the dp-aligned policy cell's PeerClient (injected by workers.grid._align_one when
    a WorkerGroup is passed as an arg). Pulls the sync bucket by bucket so peak memory stays ≈ one
    bucket, with the transport discipline chosen FROM THE BOUND MOVER — the caller never picks a
    per-transport API (peer-pull design §1.1/§1.4):

    - Plan: pulled from the peer once (the policy is the plan's single source of truth) and cached.
    - STAGED (Copy/IPC): synchronous pull per bucket. The per-bucket drain is IMPLICIT per lane —
      the peer call returns only after bucket i is staged, and this lane requests bucket i+1 only
      after apply_bucket(i) finished loading, so the producer's stage(i+1) (which frees pin i) can
      never outrun this lane's fetch of i.
    - COLLECTIVE (NCCL): fire push_bucket(i) asynchronously (peer.call_async) then enter the local
      recv, so both sides are inside the broadcast — a synchronous pull would deadlock (the send
      completes only once the recv side has entered the collective).
    - SharedMover: apply_bucket degenerates per bucket (state is None) and stamps the version.

    THIS function owns the streaming version/staleness protocol (one seam, not copied per sink):
    a stream is N buckets of ONE logical update, so the consumer's ``_weight_version`` is stamped
    ONCE, after the WHOLE loop succeeds, with the version the payloads carry — the PRODUCER'S
    TRAINING ITERATION (training owns advancement; sync is a pure transport of "params at iter k").
    Both transports stamp identically: even on COLLECTIVE the bytes ride the broadcast but the
    payload METADATA (incl. version) rides the control-plane return value (§3.4 metadata/bulk
    split). A ``None`` payload raises — a peer that stages nothing has no weight backend bound
    (wiring bug), and silently stamping "fresh" over an unsynced model was the old fail-quiet
    edge. A stream that fails partway raises before the stamp, leaving the OLD version (a
    half-synced model never advertises fresh weights). The bodies' ``apply_bucket`` are pure
    loads with no version semantics.

    Degenerate edges keep the pre-peer contract, WITHOUT any peer RPC (so their stamp is a LOCAL
    freshness tick, not the producer's iteration — meaningless to compare cross-role, which is fine:
    on a shared edge the weights are literally the same instance):
    - no backend bound (v0 static/shared) ⇒ bump the staleness stamp only (what every payload-form
      sink did);
    - SharedMover (one ActorGroup — trainer and rollout share the instance AND the actor) ⇒ same
      stamp-only shortcut. Issuing a peer RPC here would be a SELF-call: the PeerClient targets the
      very actor currently executing this apply_weights, which cannot serve it → deadlock.
    Peer results are unwrapped via a duck-typed ``.result()`` so a local-backend peer (a raw
    WorkerGroup whose calls return Handles) works the same as a Grid PeerClient (resolved values).

    Args:
        consumer: the rollout body being synced — dispatched on its ``_weight_backend``; its sink
            loads the buckets and its ``_weight_version`` is stamped.
        peer: the dp-aligned policy cell's PeerClient to pull from; ``None`` on a static/shared edge.
    """
    from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE
    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

    be = getattr(consumer, "_weight_backend", None)
    if be is None or isinstance(be.mover, SharedMover):
        consumer._weight_version += 1  # weights already visible (shared) or static: stamp only
        return
    # Fail loud on the RETIRED batch forms (clear message beats an AttributeError deep in the pull):
    # apply_weights is peer-only — a real backend with no peer, or a staged WeightPayload, means the
    # caller wants the old whole-model batch layer, which now lives ONLY on WeightSync.apply/export.
    if peer is None:
        raise ValueError(
            "apply_weights(None) with a bound non-shared backend: the batch/collective-recv form was "
            "removed — pass the policy peer (weight_sync(rollout, policy)), or drive "
            "WeightSync.apply/export directly for a manual whole-model sync."
        )
    if isinstance(peer, WeightPayload):
        raise ValueError(
            "apply_weights(WeightPayload) was removed — pass the policy peer "
            "(weight_sync(rollout, policy)); for a manual whole-model sync use WeightSync.apply(model, "
            "payload, sink=...) directly."
        )

    if not be.has_plan():
        be.set_bucket_plan(_resolved(peer.get_bucket_plan()))
    # Destination metadata is a separate producer-owned proof from the bucket plan. Resolve and
    # validate it even when a direct/manual caller preinstalled the plan: a cached plan must not be
    # an escape hatch around the bucket-0 pre-mutation guard.
    destination_names = _resolved(peer.get_weight_destination_names())
    witness_spec = None
    if destination_names is not None:
        consumer.set_weight_destination_names(destination_names)
        witness_spec = _resolved(peer.get_weight_witness_spec())
        if witness_spec is not None:
            consumer.set_weight_witness_spec(witness_spec)
    # STRICT/semantic coverage before any tensor moves (stronger than the batch path's strict
    # load-time failure: nothing is half-mutated).
    consumer.check_plan_coverage()
    n = be.bucket_count()
    if n == 0:
        raise ValueError("empty bucket plan — nothing to sync (empty model manifest?)")
    if getattr(be.mover, "lifetime", None) == COLLECTIVE:
        # Both sides must be INSIDE the broadcast: fire the peer's send, then enter the local recv.
        # A Ray PeerClient offers call_async (a REAL method — probed on the TYPE, because both
        # PeerClient.__getattr__ and WorkerGroup.__getattr__ would happily manufacture a bogus
        # 'call_async' attribute); any other peer (plain body, manual NcclMover setup) is
        # co-scheduled on a local thread instead.
        from concurrent.futures import ThreadPoolExecutor

        # Pre-flight probe, on the CONTROL plane, BEFORE any collective: a backendless producer's
        # push_bucket returns None WITHOUT broadcasting, which would strand this side inside the
        # bucket-0 recv until the collective timeout — the post-loop None check can never fire
        # first (the hang precedes it). get_bucket_plan raises loudly on the producer when no
        # backend is bound, and is producer-side cached otherwise, so resolving it here (even when
        # OUR plan is already cached) turns that hang into an immediate, attributed error. One
        # metadata RPC per COLLECTIVE stream; the first sync already pays it as the lazy plan pull.
        _resolved(peer.get_bucket_plan())
        # FIXME(@maoshengl): HANG RISK — no abort path for a mid-stream failure. If apply_bucket
        # raises after push_bucket(i) was fired, the producer is stranded INSIDE the broadcast (a
        # raise on one side of a collective cannot interrupt the other): on the call_async path the
        # producer actor blocks until the TorchDistComm timeout (300s); on the thread fallback it is
        # worse — ThreadPoolExecutor.__exit__ JOINS the worker thread, so the exception cannot even
        # leave this block until that timeout. Fail-stop fix: wrap the loop in try/except, abort the
        # communicator (needs a TorchDistComm.abort() that tears down the PG) before re-raising, so
        # both sides fail promptly instead of hanging.
        has_async = callable(getattr(type(peer), "call_async", None))
        payload = None
        with ThreadPoolExecutor(max_workers=1) as ex:
            for i in range(n):
                # TODO: (lms) How to decrease the frequency of this call of `call_async`?
                fut = peer.call_async("push_bucket", i) if has_async else ex.submit(peer.push_bucket, i)
                consumer.apply_bucket(None, i)  # recv layout comes from the cached plan
                # Bytes rode the broadcast; the returned payload is METADATA ONLY (§3.4) — it
                # carries the producer's per-sync version, stamped below after the whole stream.
                payload = _resolved(fut.result())
        if payload is None:
            # Defense-in-depth: the backendless-producer case is caught by the pre-flight probe
            # above (a real NCCL recv would hang before reaching here); this guards duck-typed /
            # fake movers whose fetch doesn't block and any peer violating the metadata protocol.
            raise ValueError(
                "peer.push_bucket returned no payload metadata — the peer is violating the "
                "streaming protocol (every push_bucket must return a WeightPayload carrying the "
                "per-sync version); refusing to stamp a fresh version"
            )
        stream_version = payload.version
    else:
        payload, version = None, None
        for i in range(n):
            payload = _resolved(peer.push_bucket(i))
            if payload is None:
                raise ValueError(
                    f"peer.push_bucket({i}) returned None — the policy peer has no weight backend "
                    f"bound (wiring bug: bind the policy with weight_sync=True); refusing to stamp "
                    f"a fresh version over an unsynced model"
                )
            consumer.apply_bucket(payload, i)
            # Keep the version, DROP the handle, before the next push_bucket triggers the peer's
            # stage(i+1). Same reason as push_stream_to_peer: this local would otherwise still
            # reference bucket i while the producer stages i+1, and a one-sided mover that waits
            # for the previous ObjectRef to be released (NixlWeightMover) would deadlock on it.
            version = payload.version
            payload = None
        stream_version = version
    if witness_spec is not None:
        consumer.require_weight_witness_verified(witness_spec)
    if _releases_staged_after_sync(be):
        # Release consumer mappings before producer storage. The branch is false for every legacy
        # mover, so disabled paths issue no additional peer call.
        be.release_staged()
        release_peer = getattr(peer, "release_weight_buffer", None)
        if callable(release_peer):
            _resolved(release_peer())
    # Every bucket of one sync carries the SAME producer iteration. Publish only after an optional
    # mapped witness verified the live packed/TP destination.
    consumer._weight_version = stream_version


def push_stream_to_peer(producer: Any, peer: Any, be: Any) -> None:
    """Materialize every bucket collectively and push it through the policy cell's routed targets.

    Symmetric paths pass one ordinary ``PeerClient``. Explicit asymmetric routes pass a
    ``WeightTargetSet``; an explicit empty set means gather-and-discard. Grid routing composes multiple
    rollout actors behind one bounded ``RoutedWeightPeerClient``, so one bucket is materialized and
    staged once, then the same NIXL ObjectRef is published concurrently to every routed target. This
    lets all TP lanes enter engine-local load operations together and keeps the policy actor's wait
    bounded. Direct/manual tests may still supply several peer adapters, which are staged sequentially.

    Version publication is deliberately absent here. Successful streams only mark their rollout
    lanes ready; ``control.loop.weight_sync`` drains every policy cell, verifies one producer
    iteration, and performs a rollout-wide prepare/commit barrier.

    Args:
        producer: policy body whose model and publication version define the stream.
        peer: one direct peer, or a bind-created ``WeightTargetSet`` (possibly empty).
        be: this policy actor's weight backend.
    """
    with ExitStack() as export_stack:
        _push_stream_to_peer(producer, peer, be, export_stack)


def _push_stream_to_peer(producer: Any, peer: Any, be: Any, export_stack: ExitStack) -> None:
    """Keep mapper-owned shard references alive only for this producer-driven stream."""
    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    if peer is None:
        raise ValueError("producer push with no routed rollout target (bind wiring bug)")
    if isinstance(be.mover, SharedMover):
        raise ValueError(
            "producer-driven push over a SharedMover is invalid: the shared-model case must bind no "
            "weight backend so push_weights stays a byte-free no-op; a "
            "per-bucket push over a self-targeting PeerClient would deadlock"
        )
    peers = peer.peers if isinstance(peer, WeightTargetSet) else (peer,)
    # ONE version for the whole stream: the payloads the rollout loads and the readiness this
    # marks afterwards must name the same thing, and it must be the version
    # ``control.loop.weight_sync`` runs its prepare/commit barrier at -- ``policy.iteration()``.
    # For a role that owns a separate publication clock (PPO advances its outer policy version
    # once per outer update while ``_iter`` counts every minibatch step) the raw optimizer
    # counter names a version the barrier never asks for. ``publication_version`` falls back to
    # ``_iter``, so lockstep roles are unaffected.
    version = int(producer.publication_version())
    # Sync-cost bookkeeping (`WeightPushTiming`): wall clock around each rank-converged step, so every
    # phase includes its barrier and the device work that barrier's `.item()` waited for.
    started = time.perf_counter()

    def _prepare_plan() -> None:
        # Producer owns the plan. Every policy actor derives the same metadata before any tensor
        # collective; only actors with targets ship it to rollout destinations.
        # be.bucket_bytes, NOT the module default: the transport's staging buffer was sized at bind
        # from a plan built with this same cap, and it cannot grow.
        be.ensure_plan(producer.model)
        # Mappers may reuse local shard views across buckets. Enter under rank convergence so a
        # failed preparation cannot leave peers entering tensor collectives; ExitStack releases
        # the views on success or failure, before the trainer can mutate or offload its storage.
        export_session = getattr(getattr(be, "mapper", None), "export_session", None)
        if callable(export_session):
            export_stack.enter_context(export_session(producer.model))

    run_rank_converged(_prepare_plan, "weight bucket-plan preparation")
    plan_s = time.perf_counter() - started
    n = be.bucket_count()
    if n == 0:
        raise ValueError("empty bucket plan — nothing to sync (empty model manifest?)")
    destination_names_fn = getattr(be, "destination_names", None)
    destination_names = destination_names_fn() if callable(destination_names_fn) else None
    witness_spec_fn = getattr(be, "weight_witness_spec", None)
    witness_spec = witness_spec_fn(producer.model) if callable(witness_spec_fn) else None

    def _prepare_targets() -> None:
        for target in peers:
            _resolved(target.set_bucket_plan(be.plan()))
            if destination_names is not None:
                _resolved(target.set_weight_destination_names(destination_names))
            if witness_spec is not None:
                _resolved(target.set_weight_witness_spec(witness_spec))
            # Strict receiver-side coverage BEFORE any tensor moves. A vLLM engine sink is exempt
            # only when the producer cannot prove mapped live destinations. A semantic mapper sends
            # those destinations above, so vLLM can compare them exactly before bucket 0 mutates it.
            _resolved(target.check_plan_coverage())

    targets_started = time.perf_counter()
    run_rank_converged(_prepare_targets, "weight target plan/coverage preparation")
    coverage_s = time.perf_counter() - targets_started

    materialize_s: list[float] = []
    gather_spans: list[GpuSpan] = []
    deliver_s: list[float] = []
    for i in range(n):
        # The all-gather only makes the current stream wait; the host returns as soon as the work is
        # launched. The span measures the gather + cast on the device, the wall clock around the
        # converged step measures launch + barrier (which, through `.item()`, absorbs that device time).
        gather_span = GpuSpan()

        def _materialize() -> dict:
            with gather_span:
                return be.materialize_bucket(producer.model, i)  # [N_bucket_bytes]

        bucket_started = time.perf_counter()
        state = run_rank_converged(_materialize, f"weight bucket {i} full-tensor materialization")
        materialized_at = time.perf_counter()

        def _deliver_bucket() -> None:
            payload: WeightPayload | None = None
            for target in peers:
                payload = be.stage_bucket_state(state, version=version)
                _resolved(target.apply_bucket(payload, i))
                # Release this adapter's RDT handle before another stage reuses the persistent
                # fabric buffer. A routed adapter settles every destination before returning.
                payload = None

        run_rank_converged(_deliver_bucket, f"weight bucket {i} delivery")
        materialize_s.append(materialized_at - bucket_started)
        deliver_s.append(time.perf_counter() - materialized_at)
        gather_spans.append(gather_span)
        # Drop the gathered full tensors BEFORE evaluating materialize_bucket(i + 1). Assignment
        # evaluates its RHS before rebinding ``state``; without this explicit release, bucket i and
        # bucket i+1 overlap and every policy actor pays roughly two full buckets of peak memory.
        state = None

    if witness_spec is not None:

        def _verify_targets() -> None:
            for target in peers:
                _resolved(target.require_weight_witness_verified(witness_spec))

        run_rank_converged(_verify_targets, "weight target live-value witness")

    def _release_target_buffers() -> None:
        for target in peers:
            release_target = getattr(target, "release_weight_buffer", None)
            if callable(release_target):
                _resolved(release_target())

    if _releases_staged_after_sync(be):
        # Consumers must drop their IPC mappings before the producer drops its storage. The branch
        # is false for every legacy mover, so disabled paths issue no additional peer calls.
        run_rank_converged(_release_target_buffers, "weight target staging-buffer release")
        run_rank_converged(be.release_staged, "weight producer staging-buffer release")

    def _mark_targets_ready() -> None:
        # Include normal cleanup in stream timing; the outer stack still handles failed streams.
        export_stack.close()
        for target in peers:
            _resolved(target.mark_weight_version_ready(version))

    ready_started = time.perf_counter()
    run_rank_converged(_mark_targets_ready, "weight target readiness")
    ready_s = time.perf_counter() - ready_started
    # Every stage's device work has landed by now (the IPC stage synchronizes, Ray's NIXL put
    # synchronizes, and each converged step's `.item()` waited); the spans are read once, here.
    be.last_push_timing = WeightPushTiming(
        version=version,
        plan_s=plan_s,
        coverage_s=coverage_s,
        materialize_s=tuple(materialize_s),
        gather_gpu_s=tuple(span.seconds() for span in gather_spans),
        deliver_s=tuple(deliver_s),
        ready_s=ready_s,
        total_s=time.perf_counter() - started,
    )


class WeightSync:
    """Weight-plane upper layer: holds a Mover (mechanism) + a WeightMapper (key/layout).
    export = producer/trainer side; apply = consumer/rollout side."""

    def __init__(
        self,
        mover: Mover,
        mapper: Any | None = None,
        bucket_bytes: int | None = None,
        transfer_dtype: str | None = None,
    ) -> None:
        self.mover = mover
        self.mapper = mapper or WeightMapper()
        # Wire dtype for floating PARAMETERS, held beside the bucket cap for the same reason: the
        # plan, the staging buffer sized from it, and the receive buffers must all read ONE value.
        # None ⇒ every entry keeps the dtype the model reports. See `apply_wire_dtype`.
        self.transfer_dtype = transfer_dtype
        self._bucket_plan: BucketPlan | None = None
        self._destination_names: DestinationNames = None
        self._destination_names_ready = False
        # Per-bucket byte cap for THIS backend. Held here so the plan and any transport-side
        # buffer sized from it read ONE value: NixlTransport's staging buffer is sized at bind
        # from the plan and is registered+pinned (cannot grow), so a plan built with a different
        # cap would fail mid-sync with "staging buffer too small". None ⇒ BUCKET_BYTES.
        self._bucket_bytes = bucket_bytes
        # Sync-cost records of the LAST stream: `push_stream_to_peer` fills the producer's,
        # `apply_bucket` the consumer's. Read by the acceptance/bench drivers; nothing in the sync
        # path depends on them.
        self.last_push_timing: WeightPushTiming | None = None
        self.apply_timings: list[BucketApplyTiming] = []

    @property
    def bucket_bytes(self) -> int:
        """The per-bucket byte cap this backend plans with (`BUCKET_BYTES` unless overridden)."""
        return self._bucket_bytes or BUCKET_BYTES

    def bind_comm(self, comm: Any, recv_manifest: list | None = None) -> None:
        """Attach the cross-role weight-plane communicator (the NCCL side group built by
        init_weight_plane) — and, for a SHARDED sink that cannot self-derive full param shapes
        (vLLM: TP-sharded live params), the full-model recv layout — to this backend's mover.
        The backend owns its mover's wiring — callers (workers layer) must not reach into
        ``mover._comm`` / ``mover.recv_plan`` themselves. Both arms are no-ops for movers without
        the corresponding seam (STAGED Copy/IPC have no ``_comm``/``recv_plan``), so callers may
        invoke this unconditionally per body."""
        if hasattr(self.mover, "_comm"):
            self.mover._comm = comm
        if recv_manifest is not None and hasattr(self.mover, "recv_plan"):
            # Sort by param name to match NcclMover.stage's sorted(obj) broadcast order — a manifest
            # in state_dict insertion order (weight before bias) would otherwise receive each sorted
            # broadcast into the wrong buffer (or a shape mismatch).
            self.mover.recv_plan = sorted((tuple(e) for e in recv_manifest), key=lambda e: e[0])

    # --- streaming (per-bucket) path; additive, does NOT touch export/apply ---
    def set_bucket_plan(self, plan: BucketPlan) -> None:
        """Cache the streaming bucket plan, re-validating name-sort order first (the plan may arrive
        over RPC from a peer, not only from ``bucket_plan`` — ordering is the transfer's correctness
        precondition).

        Args:
            plan: `list[list[(name, shape, dtype)]]` — the ordered bucket plan to install.
        """
        # Re-validate here too: the consumer receives the plan over RPC from its peer (not only from
        # bucket_plan), and the ordering invariant is the transfer's correctness precondition.
        _check_name_sorted([name for bucket in plan for (name, _s, _d) in bucket], "set_bucket_plan(plan)")
        self.mover.reserve(plan)
        self._bucket_plan = plan

    def release_staged(self) -> None:
        """Release mechanism-owned staging state after a complete stream."""
        self.mover.release_staged()

    def close(self) -> None:
        """Idempotent final cleanup through the mover contract, distinct from per-stream release."""
        self.mover.close()

    def releases_staged_after_sync(self) -> bool:
        """Whether this mover requests the optional end-of-stream release handshake."""
        return bool(getattr(self.mover, "_release_after_sync", False))

    def set_destination_names(self, names: DestinationNames) -> None:
        """Install mapper-proven live destination names received on the control plane."""
        if names is None:
            self._destination_names = None
            self._destination_names_ready = True
            return
        normalized = tuple(names)
        if any(not isinstance(name, str) or not name for name in normalized):
            raise TypeError(f"weight destination names must be non-empty strings, got {normalized!r}")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"weight destination names must be unique, got {normalized!r}")
        self._destination_names = tuple(sorted(normalized))
        self._destination_names_ready = True

    def destination_names(self) -> DestinationNames:
        """Return mapper-proven live destination names, or ``None`` for legacy identity mapping."""
        return self._destination_names

    def ensure_plan(self, model: Any, bucket_bytes: int | None = None) -> BucketPlan:
        """Build and cache one mapper-owned plan plus its sink-coverage metadata.

        This is the ONE place a plan is built, which is what makes `self.transfer_dtype` sufficient
        to declare the wire dtype: every route -- the producer push, `push_bucket`'s lazy fallback,
        and the `rollout.apply_weights(policy)` pull -- arrives through it. It was previously
        declared at one of two plan builders and not the other, and `workers/ray.py` sizes the
        staging allocation from whichever ran first, an allocation that cannot grow after bind.

        The dtype is read from the BACKEND, not passed per call, for the same reason
        `bucket_bytes` is: bind sizes the staging buffer from it, so the plan and that buffer must
        read one value fixed at bind. Threading it through this signature instead would force every
        backend stand-in to grow the parameter without making the declaration any harder to lose.
        """
        if not self.has_plan():
            manifest = model_manifest(model, self.mapper, self.transfer_dtype)
            self.set_bucket_plan(bucket_plan(manifest, bucket_bytes or self.bucket_bytes))
        if not self._destination_names_ready:
            self.set_destination_names(self.mapper.destination_names(model))
        return self.plan()

    def has_plan(self) -> bool:
        return self._bucket_plan is not None

    def plan(self) -> BucketPlan:
        if self._bucket_plan is None:
            raise ValueError("no bucket plan set (set_bucket_plan, or let the peer form pull it lazily)")
        return self._bucket_plan

    def bucket_count(self) -> int:
        if self._bucket_plan is None:
            raise ValueError("no bucket plan set (set_bucket_plan, or let the peer form pull it lazily)")
        return len(self._bucket_plan)

    def materialize_bucket(self, model: Any, i: int) -> dict:
        """Materialize bucket ``i`` on every policy actor, including actors with no target."""
        if self._bucket_plan is None:
            raise ValueError("no bucket plan set before bucket materialization")
        return bucket_full_tensors(model, self._bucket_plan[i], self.mapper)  # [N_bucket_bytes]

    def weight_witness_spec(self, model: Any) -> WeightWitnessSpec | None:
        """Return the mapper's optional witness selection without entering a tensor collective."""
        witness_spec_fn = getattr(self.mapper, "witness_spec", None)
        spec = witness_spec_fn(model) if callable(witness_spec_fn) else None
        if spec is None:
            return None
        if not isinstance(spec, WeightWitnessSpec):
            raise TypeError(f"weight mapper witness_spec must return WeightWitnessSpec or None, got {spec!r}")
        return spec

    def stage_bucket_state(self, state: dict, version: int = 0) -> WeightPayload:
        """Stage one already-materialized full bucket on a policy actor with a target."""
        return WeightPayload(version=version, handle=self.mover.stage(state))

    def stage_bucket(self, model: Any, i: int, version: int = 0) -> WeightPayload:
        """Producer: materialize + stage ONLY bucket i's full tensors (peak ≈ one bucket). ``version`` is
        the per-SYNC policy version — the producer's TRAINING ITERATION, same for every bucket of one
        stream — the caller owns it (mirrors ``export``); staging must NOT invent a per-bucket stamp
        (would corrupt staleness accounting).

        Args:
            model: the producer model to materialize bucket ``i``'s full tensors from.
            i: bucket index into the installed plan.
            version: the per-sync policy version (producer training iteration), same for every bucket.

        Returns:
            A `WeightPayload` stamping ``version`` and carrying the staged bucket's `MoveHandle`.
        """
        state = self.materialize_bucket(model, i)  # [N_bucket_bytes]
        return self.stage_bucket_state(state, version=version)

    def fetch_bucket_state(self, payload: WeightPayload | None, i: int) -> Any:
        """Consumer FETCH half: receive bucket i's raw state — COLLECTIVE recv_plan narrow/restore or
        STAGED handle fetch — WITHOUT applying it to any model. Returns the fetched state dict (or
        None on a Shared/empty edge). Split out of ``apply_bucket`` so the raw fetched state is
        exposed independently of the load step for engine-specific sinks and direct tests.

        Args:
            payload: the producer's `WeightPayload` (its ``handle`` drives a STAGED fetch); ``None`` on
                a Shared/empty edge or when the COLLECTIVE mover receives live (no handle needed).
            i: bucket index into the installed plan (selects the COLLECTIVE recv layout).

        Returns:
            The fetched bucket-``i`` state dict, or ``None`` on a Shared/empty edge.
        """
        from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE

        bucket = self._bucket_plan[i]
        if getattr(self.mover, "lifetime", None) == COLLECTIVE:
            saved = getattr(self.mover, "recv_plan", None)  # the injected FULL-model plan (batch path needs it)
            self.mover.recv_plan = [(n, s, d) for (n, s, d) in bucket]  # this bucket's recv layout
            try:
                return self.mover.fetch(None)
            finally:
                self.mover.recv_plan = saved  # restore so a later batch apply loads the whole model
        return self.mover.fetch(payload.handle) if payload is not None else None

    def apply_bucket_state(self, model: Any, i: int, state: Any, sink: Any = None) -> None:
        """Consumer SINK half: load an ALREADY-FETCHED bucket-i ``state`` into the model in place and
        run consumer_done — no receive mechanics, no version semantics. ``sink(state)`` is the
        pluggable put-to-model step: default = mapper.import_state(strict=False) (HF/Local); vLLM
        injects its engine sink. Used by ``apply_bucket`` (fetch-then-apply, direct/collective) AND
        by the producer-driven push path, where the rollout body receives an already-imported bucket
        ``state`` and lands it via this sink half. ``i`` is carried for symmetry with the fetch half
        (the sink is bucket-agnostic).

        Args:
            model: the consumer model to load ``state`` into (via the default mapper sink).
            i: bucket index — carried for symmetry with the fetch half; the sink is bucket-agnostic.
            state: the already-fetched bucket state to load; ``None`` (Shared/empty edge) is a no-op.
            sink: optional put-to-model callable (default = ``mapper.import_state(strict=False)``).
        """
        if state is not None:
            if sink is not None:
                sink(state)
            else:
                self.mapper.import_state(model, state, strict=False)
            done = getattr(self.mover, "consumer_done", None)
            if done is not None:
                done()

    def apply_bucket(
        self,
        model: Any,
        i: int,
        payload: WeightPayload | None = None,
        sink: Any = None,
    ) -> None:
        """Consumer: receive + load ONLY bucket i, in place = ``fetch_bucket_state`` + the sink half
        (``apply_bucket_state``) — the SINGLE owner of the direct-path receive mechanics for every
        sink. Behavior is byte-identical to the pre-split body; the split exposes fetched state
        independently of the load step. NO version/staleness semantics here — the
        streaming version protocol is owned by pull_stream_from_peer (the seam that knows when the
        WHOLE stream completed).

        Args:
            model: the consumer model to load bucket ``i`` into.
            i: bucket index into the installed plan.
            payload: the producer's `WeightPayload` (STAGED fetch); ``None`` for COLLECTIVE/Shared.
            sink: optional put-to-model callable (default = ``mapper.import_state(strict=False)``).
        """
        started = time.perf_counter()
        state = self.fetch_bucket_state(payload, i)
        fetched_at = time.perf_counter()
        self.apply_bucket_state(model, i, state, sink=sink)
        if i == 0:
            self.apply_timings = []
        self.apply_timings.append(
            BucketApplyTiming(bucket=i, fetch_s=fetched_at - started, sink_s=time.perf_counter() - fetched_at)
        )

    def export(self, model: Any, version: int) -> WeightPayload:
        state = self.mapper.export_state(model)  # weight concern: snapshot (+ future reshard)
        handle = self.mover.stage(state)  # mechanism
        return WeightPayload(version=version, handle=handle)

    def apply(self, model: Any, payload: WeightPayload | None = None, mode: str = "live", sink: Any = None) -> None:
        """Consumer, batch path: fetch the WHOLE state and load it. Owns the same receive mechanics
        as apply_bucket (collective recv / staged fetch / consumer_done); ``sink(state)`` is the
        pluggable put-to-model step (default = mapper.import_state, strict). Returns None with no
        load when a SharedMover fetch yields None (shared instance)."""
        from cosmos_framework.posttrain.comm.transfer.mover import COLLECTIVE

        if getattr(self.mover, "lifetime", None) == COLLECTIVE:
            # COLLECTIVE (NCCL): no producer payload — the consumer derives its receive layout from
            # its OWN model (same architecture in v1) and receives live inside the broadcast. The
            # SyncPlan is thus control-plane knowledge (the consumer's own state_dict), not shipped
            # from the producer (which would create a broadcast deadlock, design §4). A sink that
            # cannot derive full shapes from its model (vLLM: TP-sharded live params) relies on the
            # bind-injected recv_plan instead of this derivation — so only override when unset.
            if getattr(self.mover, "recv_plan", None) is None:
                net = _net(model)
                if not hasattr(net, "state_dict"):  # e.g. a vllm.LLM engine (TP-sharded live params)
                    raise ValueError(
                        "collective apply: mover.recv_plan is not set and this sink cannot self-derive "
                        "the full param shapes (model has no state_dict — e.g. a vLLM engine). Call "
                        "init_weight_plane with recv_manifest (the policy's full manifest) before syncing."
                    )
                sd = net.state_dict()  # unwrap so recv_plan keys match the raw-module producer keys
                self.mover.recv_plan = [(k, tuple(v.shape), str(v.dtype)) for k, v in sorted(sd.items())]
            state = self.mover.fetch(None)
        else:
            if payload is None:
                raise ValueError("WeightSync.apply requires a payload")
            state = self.mover.fetch(payload.handle)  # mechanism (None => shared instance, nothing to load)
        if state is not None:
            if sink is not None:
                sink(state)
            else:
                self.mapper.import_state(model, state)
            # A borrow-mode mover (IpcMover) returns an ALIAS of the producer's live GPU memory, and
            # import_state only ENQUEUES the D2D copy. consumer_done() waits for it so the producer can
            # safely mutate its weights next step (cross-process CUDA streams are otherwise unordered).
            done = getattr(self.mover, "consumer_done", None)
            if done is not None:
                done()


def make_weight_backend(kind: str, mapper: Any | None = None, transfer_dtype: str | None = None) -> WeightSync:
    # Every IPC backend constructed through WeightSync owns the synchronous copy-out contract, so
    # it uses one reusable staging allocation just like the relationship-based resolver. Keep the
    # per-tensor constructor only at the lower-level make_mover/IpcMover seam for generic callers.
    mover = IpcMover(single_buffer=True) if kind == "ipc" else make_mover(kind)
    return WeightSync(mover, mapper, transfer_dtype=transfer_dtype)
