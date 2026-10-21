# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-agnostic half of the weight-sync acceptance and bench drivers.

Shared by `tests/gr00t_weight_sync_cli.py` (GR00T on vLLM-Omni, two hops) and `tests/vllm_weight_sync_cli.py`
(an HF causal LM on plain vLLM, in-process load): binding the `(1, W)` trainer to the `(W, 1)` rollout on the
detected mesh, the trainer-side probes that read the bucket plan and the producer's `WeightPushTiming`, the plain
vLLM bodies of the baseline, and the pure functions that decide a bench run (arm order, statistics, digest and
witness checks). Test-only code: production modules never import it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from statistics import median
from typing import Any

import torch

from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata
from cosmos_framework.posttrain.tests.helpers.gr00t_publication_clock import PublicationClockTrainer

ROLLOUT_NAME = "rollout"
POLICY_NAME = "policy"
#: Bench arms: full-value verification on (the production default) and off.
ARM_ON = "on"
ARM_OFF = "off"


# ── mesh / bind ───────────────────────────────────────────────────────────────────────────────────────────────
def mesh_width(nodes: int, gpus_per_node: int, placement: str) -> int:
    """Cells per side. Colocated: every GPU carries one shard AND one engine; disaggregated: each side gets half."""
    total = nodes * gpus_per_node
    if placement == "colocated":
        return total
    if total % 2:
        raise ValueError(f"{total} GPUs cannot be split evenly between the trainer and the rollout")
    return total // 2


@dataclass(frozen=True)
class BoundPair:
    """The two bound WorkerGroups and the mesh they were placed on."""

    rollout: Any
    policy: Any
    nodes: int
    gpus_per_node: int
    width: int
    bind_seconds: float


def bind_pair(
    rollout_setup: Callable[[], dict[str, Any]],
    policy_setup: Callable[[], dict[str, Any]],
    *,
    placement: str,
    name: str,
    nixl_probe_bytes: int | None = None,
    nixl_min_gbps: float | None = None,
) -> BoundPair:
    """Bind a `(1, W)` trainer WG to a `(W, 1)` rollout WG on the detected mesh, with a weight edge.

    The setups run inside the actors and return ``{ROLLOUT_NAME: body}`` / ``{POLICY_NAME: body}``. Colocated
    pairs cell k of both sides on one GPU (CUDA IPC); disaggregated puts the sides on disjoint GPUs and requires
    the NIXL transport, whose bind-time probe takes the two optional floors.
    """
    from cosmos_framework.posttrain.control.placement import (
        ActorGroup,
        bind_device_groups,
        normalize_gpu_placement,
        trainer_experience_device_groups,
    )
    from cosmos_framework.posttrain.roles import roles
    from cosmos_framework.posttrain.utils.mesh import detect_mesh

    nodes, gpus = detect_mesh()
    width = mesh_width(nodes, gpus, placement)
    rollout = roles.rollout(ROLLOUT_NAME, replicas=width, ranks=1)
    policy = roles.trainer(POLICY_NAME, replicas=1, ranks=width)
    device_groups = trainer_experience_device_groups(
        ActorGroup(members=(rollout,), setup=rollout_setup, name=f"{name}_rollout"),
        ActorGroup(members=(policy,), setup=policy_setup, name=f"{name}_policy"),
        gpu_placement=normalize_gpu_placement(placement, field_name="--placement"),
        name=name,
    )
    bind_kwargs: dict[str, Any] = {
        "backend": "ray",
        "weight_sync": True,
        "weight_sync_transport": "nixl" if placement == "disaggregated" else None,
    }
    if nixl_probe_bytes is not None:
        bind_kwargs["weight_sync_probe_bytes"] = nixl_probe_bytes
    if nixl_min_gbps is not None:
        bind_kwargs["weight_sync_min_gbps"] = nixl_min_gbps
    started = time.perf_counter()
    bind_device_groups(device_groups, **bind_kwargs)
    return BoundPair(rollout, policy, nodes, gpus, width, time.perf_counter() - started)


# ── trainer-side probes ───────────────────────────────────────────────────────────────────────────────────────
def bucket_nbytes(bucket: Sequence[tuple[str, Sequence[int], str]]) -> int:
    """Raw tensor bytes of one plan bucket (numel x element size per entry; no staging alignment)."""
    nbytes = 0
    for _name, shape, dtype in bucket:
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes += numel * torch.empty((), dtype=getattr(torch, str(dtype).split(".")[-1])).element_size()
    return nbytes


class TrainerBenchProbe:
    """Mixin over an `LLMTrainer`: the bucket plan, the publication snapshot's digests and witness values, a
    one-parameter perturbation, and the producer's sync-cost record. Methods that gather FSDP full tensors are
    collective and must be called on every rank together, which a mirrored WorkerGroup call does."""

    def plan_summary(self) -> dict[str, Any]:
        be = self._weight_backend
        be.ensure_plan(self.model)
        plan = be.plan()
        dtypes = sorted({str(dtype) for bucket in plan for _name, _shape, dtype in bucket})
        return {
            "buckets": len(plan),
            "names": sum(len(b) for b in plan),
            "bytes": sum(bucket_nbytes(b) for b in plan),
            "dtypes": dtypes,
        }

    def plan_names(self) -> list[list[str]]:
        """The wire names of every bucket, in plan order."""
        be = self._weight_backend
        be.ensure_plan(self.model)
        return [[name for name, _shape, _dtype in bucket] for bucket in be.plan()]

    def parameter_names(self) -> list[str]:
        """The trainable parameter names (the plan may also carry buffers, which `perturb_parameter` cannot nudge)."""
        return sorted(name for name, _param in self.model.net.named_parameters())

    def publication_digests(self) -> dict[str, dict[str, Any]]:
        """Digests of the publication snapshot: every bucket materialized exactly as `push_stream_to_peer` does.

        Collective (FSDP full_tensor) -- every policy rank must call it together, which a mirrored WG call does.
        """
        be = self._weight_backend
        be.ensure_plan(self.model)
        digests: dict[str, dict[str, Any]] = {}
        for i in range(be.bucket_count()):
            state = be.materialize_bucket(self.model, i)
            digests.update({name: tensor_digest_metadata(t) for name, t in state.items()})
        return digests

    def witness_values(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Full tensors of ``names`` at the wire dtype, on CPU. Collective (FSDP full_tensor): call on every rank."""
        state = self.model.net.state_dict()
        wire = getattr(torch, self.transfer_dtype) if self.transfer_dtype else None
        out: dict[str, torch.Tensor] = {}
        for name in names:
            value = state[name]
            full = (value.full_tensor() if hasattr(value, "full_tensor") else value).detach()  # [*weight]
            if wire is not None and full.is_floating_point():
                full = full.to(wire)
            out[name] = full.cpu()
        return out

    def tied_aliases(self) -> list[list[str]]:
        """Groups of state-dict names sharing one storage (tied weights). The identity manifest ships every name,
        so each extra alias is wire bytes the receiver loads and, on vLLM, discards."""
        by_storage: dict[int, list[str]] = {}
        for name, value in self.model.net.state_dict().items():
            local = value.to_local() if hasattr(value, "to_local") else value
            by_storage.setdefault(local.untyped_storage().data_ptr(), []).append(name)
        return sorted(group for group in by_storage.values() if len(group) > 1)

    def perturb_parameter(self, name: str) -> dict[str, Any]:
        """Add >= 2 bf16 ULP (at the tensor's max magnitude) to one parameter, in place, on every rank."""
        import torch.distributed as dist

        params = dict(self.model.net.named_parameters())
        if name not in params:
            raise KeyError(f"{name!r} is not a parameter; known names start with {sorted(params)[:3]}")
        param = params[name]
        local = param.to_local() if hasattr(param, "to_local") else param
        peak = local.detach().abs().max().to(torch.float32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        delta = 2.0 * torch.finfo(torch.bfloat16).eps * max(float(peak), 1.0)
        with torch.no_grad():
            before = local.detach().to(torch.bfloat16).clone()
            local.add_(delta)
            changed = (local.detach().to(torch.bfloat16) != before).float().sum()
            total = torch.tensor(float(local.numel()), device=changed.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(changed)
            dist.all_reduce(total)
        return {"name": name, "delta": delta, "changed_after_cast_fraction": float(changed / total)}

    def push_timings(self) -> dict[str, Any] | None:
        """The producer's `WeightPushTiming` of the last stream this rank pushed, or None before the first."""
        timing = self._weight_backend.last_push_timing
        return None if timing is None else asdict(timing)


def revisions() -> dict[str, str]:
    """Package versions and the code revision a report was produced with."""
    import importlib.metadata
    import os
    import subprocess

    out: dict[str, str] = {}
    for dist in ("vllm-omni", "vllm", "gr00t", "torch", "ray", "transformers"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "not installed"
    try:
        out["code"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=os.path.dirname(__file__)
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["code"] = "unknown"
    return out


# ── pure functions (unit-tested) ──────────────────────────────────────────────────────────────────────────────
def arm_order(rounds: int) -> list[str]:
    """``rounds`` interleaved on/off pairs, the starting arm alternating so neither arm always runs first."""
    order: list[str] = []
    for r in range(rounds):
        order.extend((ARM_ON, ARM_OFF) if r % 2 == 0 else (ARM_OFF, ARM_ON))
    return order


@lru_cache(maxsize=None)
def _u_count(n1: int, n2: int, u: int) -> int:
    """Arrangements of n1 + n2 distinct ranks whose Mann-Whitney U equals ``u`` (the classic recurrence)."""
    if u < 0:
        return 0
    if n1 == 0 or n2 == 0:
        return 1 if u == 0 else 0
    return _u_count(n1 - 1, n2, u - n2) + _u_count(n1, n2 - 1, u)


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Exact two-sided Mann-Whitney U test (small samples, no ties expected in timings): ``(U, p)``."""
    if not a or not b:
        raise ValueError("mann_whitney_u needs two non-empty samples")
    u_a = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
    u = min(u_a, len(a) * len(b) - u_a)
    total = sum(_u_count(len(a), len(b), k) for k in range(len(a) * len(b) + 1))
    lower = sum(_u_count(len(a), len(b), k) for k in range(int(u) + 1))
    return u, min(1.0, 2.0 * lower / total)


def summarize_bench(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Per-arm n/min/median/max, ``delta_s = median(on) - median(off)`` and whether it clearly exceeds the spread.

    ``significant`` = the two ranges are disjoint AND the delta is at least twice the wider within-arm range; the
    exact Mann-Whitney test is the fallback reading when the ranges overlap. Requires both arms non-empty.
    """
    counts = {arm: len(values) for arm, values in samples.items()}
    if set(samples) != {ARM_ON, ARM_OFF} or any(n == 0 for n in counts.values()):
        raise ValueError(f"summarize_bench needs non-empty samples for both arms {ARM_ON!r}/{ARM_OFF!r}, got {counts}")
    arms = {
        arm: {"n": len(v), "min": min(v), "median": median(v), "max": max(v), "samples": list(v)}
        for arm, v in samples.items()
    }
    on, off = arms[ARM_ON], arms[ARM_OFF]
    delta = on["median"] - off["median"]
    max_range = max(on["max"] - on["min"], off["max"] - off["min"])
    disjoint = on["min"] > off["max"] or off["min"] > on["max"]
    u, p = mann_whitney_u(samples[ARM_ON], samples[ARM_OFF])
    return {
        ARM_ON: on,
        ARM_OFF: off,
        "delta_s": delta,
        "ranges_disjoint": disjoint,
        "delta_over_max_range": abs(delta) / max_range if max_range > 0 else None,
        "mann_whitney_u": u,
        "mann_whitney_p": p,
        "significant": disjoint and (max_range == 0.0 or abs(delta) >= 2.0 * max_range),
    }


def identical(values: Mapping[Any, Any], what: str) -> Any:
    """Every cell must agree; return the shared value."""
    items = list(values.items())
    first = items[0][1]
    for cell, value in items[1:]:
        if value != first:
            raise RuntimeError(f"{what}: cell {cell} disagrees with cell {items[0][0]}")
    return first


def medians(values_by_key: Mapping[str, Sequence[float | None]]) -> dict[str, float | None]:
    """Median per key over the non-None values; None when a key has none."""
    out: dict[str, float | None] = {}
    for key, values in values_by_key.items():
        present = [float(v) for v in values if v is not None]
        out[key] = median(present) if present else None
    return out


def columns(rows: Sequence[Mapping[str, float | None]]) -> dict[str, list[float | None]]:
    """Transpose per-sample rows into per-key columns (missing keys read as None)."""
    keys = sorted({key for row in rows for key in row})
    return {key: [row.get(key) for row in rows] for key in keys}


def paired_lane(policy_cell: tuple[int, int]) -> tuple[int, int]:
    """The rollout cell the flat-index route pairs with a policy cell: policy ``(0, r)`` pushes to lane ``(r, 0)``."""
    return (int(policy_cell[1]), int(policy_cell[0]))


def _sum_or_none(values: Sequence[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return float(sum(values))  # type: ignore[arg-type]


def stage_rows(
    sync_s: float,
    push_by_rank: Mapping[tuple[int, int], Mapping[str, Any] | None],
    lanes_by_cell: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    verify_by_cell: Mapping[tuple[int, int], Mapping[str, Any] | None] | None = None,
) -> tuple[dict[str, float | None], dict[str, dict[str, dict[str, float | None]]]]:
    """The critical-path stage row of one sample, plus every rank's and lane's own totals.

    The critical rank is the policy cell with the largest ``total_s``; its paired lane supplies the receiver rows.
    Sums run over the buckets of that ONE rank/lane (sequential, so additive); lanes are never summed across. Child
    rows nest inside their parents (copy/sync/compare/digest inside the lane sink inside deliver; the worker's own
    time and its digest share inside the finish RPC inside ready) and are not added again. Keys that a path does not
    record (no mapped second hop, verification off) read as None or 0.0.
    """
    ranks = {cell: dict(timing) for cell, timing in push_by_rank.items() if timing}
    if not ranks:
        raise ValueError("stage_rows needs at least one producer timing record")
    critical = max(ranks, key=lambda cell: ranks[cell]["total_s"])
    push = ranks[critical]
    lane_rows = list(lanes_by_cell.get(paired_lane(critical)) or [])
    verify = (verify_by_cell or {}).get(paired_lane(critical))

    def lane_sum(key: str) -> float | None:
        return _sum_or_none([row.get(key) for row in lane_rows])

    materialize = float(sum(push["materialize_s"]))
    deliver = float(sum(push["deliver_s"]))
    fetch, sink = lane_sum("fetch_s"), lane_sum("sink_s")
    rows: dict[str, float | None] = {
        "sync_s": sync_s,
        "push_total_s": push["total_s"],
        "plan_coverage_s": push["plan_s"] + push["coverage_s"],
        "materialize_sum_s": materialize,
        "gather_gpu_sum_s": _sum_or_none(list(push["gather_gpu_s"])),
        "deliver_sum_s": deliver,
        "lane_fetch_sum_s": fetch,
        "lane_sink_sum_s": sink,
        "copy_sum_s": lane_sum("copy_s"),
        "sync_sum_s": lane_sum("sync_s"),
        "compare_sum_s": lane_sum("compare_s"),
        "digest_sum_s": lane_sum("digest_s"),
        "deliver_overhead_sum_s": None if fetch is None or sink is None else deliver - fetch - sink,
        "ready_s": push["ready_s"],
        "finish_rpc_s": 0.0 if not verify else verify["rpc_s"],
        "finish_worker_s": 0.0 if not verify or verify.get("worker_s") is None else verify["worker_s"],
        "finish_digest_s": 0.0 if not verify or verify.get("digest_s") is None else verify["digest_s"],
        "producer_other_s": push["total_s"]
        - push["plan_s"]
        - push["coverage_s"]
        - materialize
        - deliver
        - push["ready_s"],
        "controller_residual_s": sync_s - max(t["total_s"] for t in ranks.values()),
    }
    totals = {
        "ranks": {
            str(cell): {"total_s": t["total_s"], "deliver_sum_s": float(sum(t["deliver_s"])), "ready_s": t["ready_s"]}
            for cell, t in ranks.items()
        },
        "lanes": {
            str(cell): {
                "fetch_sum_s": _sum_or_none([r.get("fetch_s") for r in lane]),
                "sink_sum_s": _sum_or_none([r.get("sink_s") for r in lane]),
            }
            for cell, lane in lanes_by_cell.items()
        },
    }
    return rows, totals


def lane_spread(totals: Sequence[Mapping[str, Mapping[str, Mapping[str, float | None]]]]) -> dict[str, Any]:
    """Per side (``ranks``/``lanes``) and metric: each cell's median over samples, then min/median/max/spread across
    cells -- the "few lanes slow" reading. Lanes are compared, never added."""
    out: dict[str, Any] = {}
    for side in ("ranks", "lanes"):
        per_cell: dict[str, dict[str, list[float | None]]] = {}
        for sample in totals:
            for cell, metrics in sample.get(side, {}).items():
                for key, value in metrics.items():
                    per_cell.setdefault(cell, {}).setdefault(key, []).append(value)
        cell_medians = {cell: medians(metrics) for cell, metrics in per_cell.items()}
        out[side] = {}
        for key in sorted({key for metrics in cell_medians.values() for key in metrics}):
            values = [m[key] for m in cell_medians.values() if m.get(key) is not None]
            out[side][key] = (
                None
                if not values
                else {
                    "per_cell": {cell: m.get(key) for cell, m in cell_medians.items()},
                    "min": min(values),
                    "median": median(values),
                    "max": max(values),
                    "spread": max(values) - min(values),
                }
            )
    return out


def _digest_diff(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if set(expected) != set(actual):
        failures.append(
            f"name sets differ: {len(expected)} vs {len(actual)}; e.g. {sorted(set(expected) ^ set(actual))[:5]}"
        )
    failures.extend(
        f"{name}: {expected[name]} != {actual[name]}"
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    )
    return failures


def check_bench_digests(
    trainer: Mapping[str, Any],
    worker_by_lane: Mapping[Any, Mapping[str, Any]],
    body_by_lane: Mapping[Any, Mapping[str, Any]] | None = None,
) -> list[str]:
    """The trainer's publication digests must equal every lane's worker digests, and its body expectation when one
    is given (verification on); with verification off the body records none and the chain is two-way."""
    failures: list[str] = []
    for lane in sorted(worker_by_lane, key=str):
        failures.extend(f"lane {lane} worker: {f}" for f in _digest_diff(trainer, worker_by_lane[lane]))
        if body_by_lane is not None:
            failures.extend(f"lane {lane} body: {f}" for f in _digest_diff(trainer, body_by_lane[lane]))
    return failures


def check_witness_values(
    expected: Mapping[str, torch.Tensor], actual_by_lane: Mapping[Any, Mapping[str, torch.Tensor]]
) -> list[str]:
    """Every lane's live witness tensors must equal the trainer's bitwise: same shape, dtype and bytes."""
    failures: list[str] = []
    for lane in sorted(actual_by_lane, key=str):
        actual = actual_by_lane[lane]
        if set(actual) != set(expected):
            failures.append(f"lane {lane}: witness names {sorted(actual)} != {sorted(expected)}")
        for name in sorted(set(expected) & set(actual)):
            e, a = expected[name], actual[name]
            if e.shape != a.shape or e.dtype != a.dtype:
                failures.append(
                    f"lane {lane} {name}: rollout {tuple(a.shape)} {a.dtype} != trainer {tuple(e.shape)} {e.dtype}"
                )
            elif not torch.equal(e, a):
                failures.append(f"lane {lane} {name}: values differ from the trainer's")
    return failures


# ── the plain-vLLM baseline bodies ────────────────────────────────────────────────────────────────────────────
class VllmBenchProbe:
    """Mixin over `VLLMRollout`: state read-back, the consumer's sync-cost records, live witness values."""

    def probe_state(self) -> dict[str, Any]:
        backend = getattr(self, "_weight_backend", None)
        used = total = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
        return {
            "weight_version": self.weight_version(),
            "pending": getattr(self, "_pending_weight_version", None),
            "prepared": getattr(self, "_prepared_weight_version", None),
            "mover_kind": type(getattr(backend, "mover", backend)).__name__,
            "device_used_bytes": used,
            "device_total_bytes": total,
        }

    def apply_timings(self) -> list[dict[str, Any]]:
        """The backend's `BucketApplyTiming` rows of the last stream."""
        return [asdict(t) for t in getattr(self._weight_backend, "apply_timings", [])]

    def witness_values(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Live values of ``names`` read inside the engine (in-process V1 ``apply_model``), on CPU."""
        from cosmos_framework.posttrain.roles.rollout import vllm_weights

        wanted = list(names)
        return vllm_weights._apply_to_model(
            self.model, lambda m: {k: dict(m.named_parameters())[k].detach().cpu() for k in wanted}
        )


class HFBenchTrainer(TrainerBenchProbe, PublicationClockTrainer):
    """The baseline's trainer: an FSDP-sharded HF causal LM under the driver-owned publication clock."""


def setup_vllm_rollout(*, model: str, gpu_memory_utilization: float, max_model_len: int) -> dict[str, Any]:
    """Actor setup for the baseline rollout WG: one in-process vLLM V1 engine per cell, bf16, TP 1."""
    from cosmos_framework.posttrain.config.build import _configure_vllm_runtime

    _configure_vllm_runtime()  # in-process EngineCore, so `apply_model`/`load_weights` are direct calls
    from vllm import LLM

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    class _Body(VllmBenchProbe, VLLMRollout):
        """`VLLMRollout` + probes (composed here so importing this module costs no vLLM import)."""

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    body = _Body(llm, packer=init_data_packer(model))
    body.transfer_dtype = "bfloat16"  # read at bind (`workers/ray.py::set_weight_backend`), as on the Omni body
    return {ROLLOUT_NAME: body}


def setup_hf_policy(*, model: str) -> dict[str, Any]:
    """Actor setup for the baseline policy WG: `build_hf_model` over the injected FSDP mesh, bf16 master and wire."""
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.workers.ray import current_pdims

    # `attn_impl="eager"`: the trainer never runs a forward here, and the GR00T venv carries no flash-attn.
    hf = build_hf_model(HFModelConfig(name_or_path=model, dtype="bfloat16", attn_impl="eager"), pdims=current_pdims())
    trainer = HFBenchTrainer(
        hf,
        make_objective(init_data_packer(model)),
        torch.optim.AdamW(hf.net.parameters(), lr=1e-6),
        transfer_dtype="bfloat16",
    )
    return {POLICY_NAME: trainer}
