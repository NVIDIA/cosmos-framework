# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Multi-replica WorkerGroup backend + inter-replica communicator (DESIGN §1.1, §2.4).

Validates that a WorkerGroup grows from "1 body / 1 actor" to "N replicas (each a Ray
actor) + an explicit, optional inter-replica communicator" **without changing the
recipe call or the Handle contract**.

- N replica actors, each holds a body copy (v0.5: replica = 1 actor; the intra-replica
  TP/PP/CP mesh is a separate axis, kept at 1 here).
- A per-method **dispatch** (minimal stand-in for §2.4): selected methods aggregate compact
  per-execution-replica results; other calls retain their per-cell handles.
- The **Communicator** is a first-class object (DESIGN §1.1). v0.5: driver-coordinated
  all-reduce (gather→reduce) under single-controller — validates the seam.
  TODO(v1+): real peer-to-peer NCCL/gloo collective inside the step (e.g.
  ray.util.collective), no driver involvement; intra-replica mesh; weight-broadcast.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean
from typing import Any

import ray

from cosmos_framework.posttrain.schema.metrics import (
    MetricContribution,
    finalize_metric_report,
    merge_metric_reports,
)
from cosmos_framework.posttrain.workers.worker_group import Handle


@ray.remote
class _Replica:
    def __init__(self, body: Any, rank: int, world: int):
        self.body, self.rank, self.world = body, rank, world

    def invoke(self, method: str, args: tuple, kw: dict) -> Any:
        return getattr(self.body, method)(*args, **kw)


class Communicator:
    """Explicit inter-replica communicator (DESIGN §1.1). v0.5: driver-coordinated."""

    def __init__(self, actors: list):
        self.actors = actors
        self.world = len(actors)

    def all_reduce(self, refs: list, op: str = "mean") -> Any:
        vals = ray.get(refs)  # gather per-replica results
        if op == "replica0":
            if not vals:
                raise ValueError("cannot select replica 0 from an empty replica result inventory")
            return vals[0]
        if op == "identical":
            if not vals:
                raise ValueError("cannot check identity of an empty replica result inventory")
            for replica, value in enumerate(vals[1:], start=1):
                if not _metric_values_equal(vals[0], value):
                    raise RuntimeError(f"already-global metric report differs between replica 0 and replica {replica}")
            return vals[0]
        if vals and isinstance(vals[0], dict):
            # Metrics dicts are already reduced across the producing body's Torch model world. For
            # roles with multiple independent execution replicas, replica 0's report still describes
            # only its own data; merge typed contributions before finalizing one global report.
            # A finalized numeric dictionary cannot be reconstructed into exact weighted means,
            # counts, extrema, or sparse observations. Require the producer to retain each value's
            # typed contribution through this boundary, then finalize exactly once after merging.
            if op != "mean":
                raise ValueError(f"unsupported dictionary reduction {op!r}")
            if not all(
                isinstance(contribution, MetricContribution) for report in vals for contribution in report.values()
            ):
                raise TypeError(
                    "cross-replica metric dictionaries must carry MetricContribution values; "
                    "a finalized numeric dictionary has no safe generic reduction"
                )
            return finalize_metric_report(merge_metric_reports(vals))
        return fmean(vals) if op == "mean" else sum(vals)
        # TODO(v1+): real collective (NCCL/gloo) among replicas; no driver gather


def _metric_values_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for already-global metric reports, including tensors."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_metric_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_metric_values_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "shape") and hasattr(left, "dtype"):
        try:
            import torch  # local: this module remains torch-free for driver-only users

            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return bool(torch.equal(left, right))
        except Exception:  # noqa: BLE001 — fall through to primitive equality
            return False
    try:
        equal = left == right
        return equal if type(equal) is bool else bool(equal)
    except Exception:  # noqa: BLE001 — opaque values are not an identity proof
        return False


def _shard(seq, n):
    seq = list(seq)
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        sz = k + (1 if j < r else 0)
        out.append(seq[i : i + sz])
        i += sz
    return out


class ReplicaBackend:
    """WorkerGroup backend: N replica actors + a Communicator. Same `submit` interface
    as the local/ray backends, so the WG/recipe/Handle contract is unchanged."""

    def __init__(self, body: Any, replicas: int, dp_methods=("train",), reduce_op: str = "mean"):
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=max(2, replicas + 1),
                include_dashboard=False,
                log_to_driver=False,
                _temp_dir="/tmp/rcray",
            )
        self.world = replicas
        self.actors = [_Replica.remote(body, r, replicas) for r in range(replicas)]
        self.comm = Communicator(self.actors)
        self.dp_methods = set(dp_methods)
        self.reduce_op = reduce_op

    def submit(self, wg_name: str, body: Any, method: str, args: tuple, kw: dict) -> Handle:
        if method in self.dp_methods:
            # DP: scatter args[0] (batch) across replicas, broadcast the rest, all-reduce
            shards = _shard(args[0], self.world) if args else [[]] * self.world
            refs = [a.invoke.remote(method, (shards[r],) + args[1:], kw) for r, a in enumerate(self.actors)]
            return Handle(self.comm.all_reduce(refs, op=self.reduce_op))  # inter-replica reduce
        # broadcast (e.g. push_weights): run on all replicas, return replica-0's result
        refs = [a.invoke.remote(method, args, kw) for a in self.actors]
        return Handle(ray.get(refs[0]))
