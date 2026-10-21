# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""THE weight-sync entry point (producer-driven weight channel, design 2026-07-27 §2).

``weight_sync(rollout, policy)`` is the ONE call every training loop makes, whatever the topology.
It drives the role edge's PRODUCER form — ``policy.push_weights(rollout)`` — where the transport
discipline is picked from the bound mover (comm/weight_transfer.py::push_stream_to_peer): STAGED
(Copy/IPC/NIXL) per-bucket push through flat actor-level targets; SharedMover/no-backend stay
byte-free no-ops. The plan is the producer's single source of truth — shipped to
the rollout at the start of each stream — so there is no consumer plan pull and no per-transport
helper to choose.

History: this module once held driver-coordinated helpers and, before the producer-driven flip, drove
the CONSUMER form (``rollout.apply_weights(policy)`` → pull_stream_from_peer). Two load-bearing
invariants carry across every rewrite: every load is drained before a two-phase rollout-wide version
publication, and DRAIN EVERY CELL is ``_drain`` below, owned by ``weight_sync`` itself."""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical
from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    from cosmos_framework.posttrain.roles.rollout.base import Rollout
    from cosmos_framework.posttrain.roles.trainer.base import TrainerProtocol
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

WEIGHT_SYNC_TIMING_ENV = "POSTTRAIN_PROFILE_WEIGHT_SYNC"


def _distribution(values: list[float]) -> dict[str, float | int]:
    """Return latency statistics for one sample: ``p50_s`` is the median (the mean of the two middle
    values of an even-sized sample); ``p90_s`` and ``p99_s`` are nearest-rank sample values."""
    if not values:
        return {"n": 0}
    ordered = sorted(float(value) for value in values)
    non_finite = sum(not math.isfinite(value) for value in ordered)
    if non_finite:
        raise ValueError(f"weight-push timing has {non_finite} non-finite duration(s)")

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "n": len(ordered),
        "min_s": ordered[0],
        "p50_s": statistics.median(ordered),
        "p90_s": percentile(0.90),
        "p99_s": percentile(0.99),
        "max_s": ordered[-1],
    }


def summarize_weight_push_timings(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Aggregate actor-local ``WeightPushTiming`` values without hiding the slowest rank."""
    materialized = [record for record in records if record is not None]
    if not materialized:
        raise ValueError("weight-push timing probe returned no records")
    if len(materialized) != len(records):
        # Every policy cell runs the whole push, idle cells included, so either all record a stream or none does.
        raise ValueError(
            f"{len(records) - len(materialized)} of {len(records)} policy cells recorded no weight-push timing"
        )
    versions = {int(record["version"]) for record in materialized}
    if len(versions) != 1:
        raise ValueError(f"weight-push timing records disagree on version: {sorted(versions)}")
    bucket_counts = (
        {len(record["materialize_s"]) for record in materialized}
        | {len(record["gather_gpu_s"]) for record in materialized}
        | {len(record["deliver_s"]) for record in materialized}
        | {len(record["bucket_meta"]) for record in materialized}
    )
    if len(bucket_counts) != 1:
        raise ValueError(f"weight-push timing records disagree on bucket count: {sorted(bucket_counts)}")
    bucket_count = bucket_counts.pop()
    bucket_meta = list(materialized[0]["bucket_meta"])
    if any(list(record["bucket_meta"]) != bucket_meta for record in materialized[1:]):
        raise ValueError("weight-push timing records disagree on bucket metadata")
    slowest = max(materialized, key=lambda record: float(record["total_s"]))
    summary: dict[str, Any] = {
        "version": versions.pop(),
        "cells": len(materialized),
        "buckets": bucket_count,
        "plan": _distribution([record["plan_s"] for record in materialized]),
        "coverage": _distribution([record["coverage_s"] for record in materialized]),
        "ready": _distribution([record["ready_s"] for record in materialized]),
        "total": _distribution([record["total_s"] for record in materialized]),
        "slowest": {
            "rank": int(slowest["rank"]),
            "host": str(slowest["host"]),
            "total_s": float(slowest["total_s"]),
        },
        "bucket": [],
    }
    for index in range(bucket_count):
        gather = [
            float(record["gather_gpu_s"][index]) for record in materialized if record["gather_gpu_s"][index] is not None
        ]
        summary["bucket"].append(
            {
                "index": index,
                "meta": bucket_meta[index],
                "materialize": _distribution([record["materialize_s"][index] for record in materialized]),
                "gather_gpu": _distribution(gather),
                "deliver": _distribution([record["deliver_s"][index] for record in materialized]),
            }
        )
    return summary


def _emit_weight_push_timings(policy: Any) -> None:
    if os.environ.get(WEIGHT_SYNC_TIMING_ENV) != "1":
        return
    try:
        weight_push_timing = getattr(policy, "weight_push_timing", None)
        if not callable(weight_push_timing):
            raise TypeError(f"{type(policy).__name__} exposes no weight_push_timing()")
        records = resolve_all(weight_push_timing())
        # Shared-model and no-backend publication moves no bytes, so no producer records a stream.
        if all(record is None for record in records):
            return
        summary = summarize_weight_push_timings(records)
    except Exception as error:  # noqa: BLE001 -- opt-in diagnostics run after publication; never end the step
        logger.warning("%s=1, but the weight-sync timing summary was skipped: %r", WEIGHT_SYNC_TIMING_ENV, error)
        return
    logger.info("benchmark weight_sync_detail %s", json.dumps(summary, sort_keys=True, separators=(",", ":")))


def weight_sync(
    rollout: "WorkerGroup | Rollout",
    policy: "WorkerGroup | TrainerProtocol",
    *,
    validation_step: int | None = None,
    verify_replica_values: bool = False,
) -> None:
    """THE weight-sync entry point — use this in every training loop, whatever the topology.

        weight_sync(rollout, policy)   # producer-driven: policy.push_weights(rollout)

    Arguments are the ROLES, not payloads: normally the two bound WorkerGroups (every example), or
    the plain role bodies for in-process use (CPU tests / no grid) — anything satisfying the
    Rollout/Trainer protocols. WeightPayload never appears at this surface; it is the per-bucket
    internal wire format between stage_bucket and apply_bucket.

    One call, transport- and backend-agnostic: it drives the role edge's PRODUCER form
    (``policy.push_weights(rollout)`` → comm/weight_transfer.py::push_stream_to_peer), which already
    dispatches on the bound mover — STAGED (Copy/IPC/NIXL) per-bucket push plus unequal-pool routing,
    SharedMover/no-backend byte-free confirmation — and works for plain bodies (returns None), the
    local backend (Handles, unwrapped), and the Ray grid (CollectiveHandle). The drain lives HERE, not
    in user code: it is REQUIRED on the grid (peer design §1.7 — _Host.invoke blocks the actor while
    resolving nested-ref args, so an undrained follow-up task at the policy WG can deadlock the push)
    and a no-op elsewhere.

    Shape/transport safety belongs to bind. Equal-product placement uses the flat-index route;
    unequal products require a disaggregated NIXL WEIGHT edge whose total flat actor route explicitly
    marks active and idle policy cells. The blocking full_tensor on every policy actor keeps the FSDP
    collective aligned; each rollout actor receives one direct full-tensor stream before this
    function's global prepare/commit publication.

    Args:
        rollout: the weight CONSUMER — the bound rollout WorkerGroup (normal) or a plain `Rollout`
            body (in-process); the policy pushes each bucket into it and this controller publishes
            ``_weight_version`` after the global barrier.
        policy: the weight PRODUCER — the bound trainer WorkerGroup or a plain ``TrainerProtocol``
            body whose
            current weights are pushed; ``policy.push_weights(rollout)`` drives the per-bucket stream.
        verify_replica_values: when R2R is enabled, require an exact-byte digest match between the
            source and every rollout replica after broadcast. Intended for the initial sync only;
            the default keeps steady-state sync free of the extra device-to-host digest pass.
    """
    if not isinstance(verify_replica_values, bool):
        raise TypeError(f"verify_replica_values must be bool, got {verify_replica_values!r}")
    uses_replica_broadcast = _uses_replica_broadcast(rollout)
    if verify_replica_values and not uses_replica_broadcast:
        raise ValueError("verify_replica_values requires rollout replica broadcast to be enabled")
    iteration = getattr(policy, "iteration", None)
    producer_version = (
        int(resolve_identical(iteration(), operation="policy weight-sync iteration"))
        if callable(iteration)
        else (int(validation_step) if validation_step is not None else None)
    )
    if producer_version is not None:
        from cosmos_framework.posttrain.utils.dist import rank_identical

        producer_version = int(rank_identical(producer_version, "policy weight-sync iteration"))
    if validation_step is not None and producer_version != int(validation_step):
        raise RuntimeError(
            f"weight sync expected validation step {validation_step}, but policy reports {producer_version}"
        )

    if producer_version is None and uses_replica_broadcast:
        raise RuntimeError("P2R+R2R weight sync requires the policy to expose a concrete weight version")
    _drain(policy.push_weights(rollout))
    if producer_version is None:
        return
    if uses_replica_broadcast:
        if verify_replica_values:
            _drain(rollout.broadcast_live_weights(producer_version, verify_values=True))
        else:
            _drain(rollout.broadcast_live_weights(producer_version))
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    confirmation_error: BaseException | None = None
    try:
        confirm_weight_version(rollout, policy, producer_version)
    except BaseException as error:  # noqa: BLE001 -- local ranks must advance together
        confirmation_error = error
    raise_if_any_rank_failed(confirmation_error, "rollout weight-version publication")
    _emit_weight_push_timings(policy)


def confirm_weight_version(rollout: Any, policy: Any, model_step: int) -> None:
    """Prepare every rollout lane, then publish one version only after the prepare barrier."""
    producer_identity_fn = getattr(policy, "shared_weight_identity", None)
    if not callable(producer_identity_fn):
        raise RuntimeError("weight publication requires policy weight-identity verification")

    producer_identity = producer_identity_fn()
    identities = resolve_all(producer_identity)
    if not identities:
        raise RuntimeError("policy weight-identity verification returned no rank values")
    if all(identity is None for identity in identities):
        routed_identity: Any = None
    elif any(identity is None for identity in identities):
        raise RuntimeError("policy ranks disagree on whether validation uses a shared weight edge")
    elif hasattr(producer_identity, "shape"):
        producer_shape = tuple(producer_identity.shape)
        consumer_shape = (int(getattr(rollout, "replicas", 1)), int(getattr(rollout, "ranks", 1)))
        if producer_shape != consumer_shape:
            raise RuntimeError(
                "shared validation weight identity requires matching policy/rollout meshes; "
                f"got policy={producer_shape}, rollout={consumer_shape}"
            )
        routed_identity = producer_identity
    else:
        if len(identities) != 1:
            raise RuntimeError("non-mesh policy weight identity returned multiple rank values")
        routed_identity = identities[0]
    publish_weight_version(rollout, model_step, producer_identity=routed_identity)


def r2r_weight_sync(rollout: Any, *, model_step: int, verify_replica_values: bool = False) -> None:
    """Independently broadcast one loaded rollout replica and publish the resulting version.

    This is the eval/bootstrap surface: load the source replica by any engine-supported mechanism,
    then call the same live-tensor R2R operation used by optional online P2R+R2R weight sync.
    """
    if not _uses_replica_broadcast(rollout):
        raise RuntimeError("rollout replica broadcast is not enabled by placement")
    if not isinstance(verify_replica_values, bool):
        raise TypeError(f"verify_replica_values must be bool, got {verify_replica_values!r}")
    if verify_replica_values:
        _drain(rollout.broadcast_live_weights(int(model_step), verify_values=True))
    else:
        _drain(rollout.broadcast_live_weights(int(model_step)))
    publish_weight_version(rollout, int(model_step), producer_identity=None)


def publish_weight_version(rollout: Any, model_step: int, *, producer_identity: Any) -> None:
    """Prepare every rollout lane, then publish one version only after the prepare barrier."""
    prepare = getattr(rollout, "prepare_weight_version", None)
    commit = getattr(rollout, "commit_weight_version", None)
    if not callable(prepare) or not callable(commit):
        raise RuntimeError("rollout does not expose two-phase weight-version publication")
    # Phase 1 is read-only. If any lane is missing its complete-stream readiness marker, resolve_all
    # drains every already-submitted check and raises before a single lane advertises the version.
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    preparation_error: BaseException | None = None
    try:
        resolve_all(prepare(model_step, producer_identity))
    except BaseException as error:  # noqa: BLE001 -- torchrun ranks must decide before commit
        preparation_error = error
    raise_if_any_rank_failed(preparation_error, "rollout weight-version prepare")

    commit_error: BaseException | None = None
    try:
        resolve_all(commit(model_step))
    except BaseException as error:  # noqa: BLE001 -- converge a contract violation
        commit_error = error
    raise_if_any_rank_failed(commit_error, "rollout weight-version commit")


def _instance_attribute(obj: Any, name: str) -> Any:
    """Read an instance attribute without letting a proxy fabricate one.

    `WorkerGroup.__getattr__` turns ANY unknown name into a body-method lambda, so plain
    `getattr(wg, "_weight_broadcast_spec", None)` returns a function rather than None on every
    WorkerGroup. Probing with `getattr` therefore reports "replica broadcast is bound" for every
    unbound (local/torchrun) rollout, which sends `weight_sync` down the R2R branch and fails the
    first sync with "rollout replica broadcast is not bound". Reading `__dict__` is the only probe
    a proxy cannot answer.
    """
    return getattr(obj, "__dict__", {}).get(name)


def _uses_replica_broadcast(rollout: Any) -> bool:
    """Read the bind-owned mode flag without issuing a fabricated WorkerGroup method call.

    The local backend never binds R2R — `bind_device_groups` rejects `weight_sync_r2r_broadcast`
    unless the backend is Ray — so an unbound rollout is correctly reported as not broadcasting.
    """
    backend = _instance_attribute(rollout, "_backend")
    if backend is not None:
        return bool(getattr(backend, "weight_broadcast_enabled", False))
    return _instance_attribute(rollout, "_weight_broadcast_spec") is not None


def _drain(handle: object) -> None:
    """Block until EVERY cell of a role call finishes. A GridBackend WorkerGroup submit returns a
    CollectiveHandle spanning all (replica, rank) cells; its ``.result()`` resolves only
    rank-0/replica-0, so a multi-rank sync could return while other ranks are still pushing/loading
    weights — the next generate/train would then race partially-loaded weights (or hide a non-root
    failure). The shared all-settled resolver consumes ``as_completed()`` when available, so an
    early failed lane cannot return while a later already-submitted weight mutation is still running.
    It falls back to ``.all()`` only for legacy/fake handles and to ``.result()`` for a single-cell
    future; a plain-body call returns None (nothing to drain)."""
    resolve_all(handle)
