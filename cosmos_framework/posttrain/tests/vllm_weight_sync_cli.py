# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The plain-vLLM weight-sync baseline: an FSDP-sharded HF causal LM pushed into in-process vLLM engines, timed.

The like-for-like comparison for `tests/gr00t_weight_sync_cli.py --bench-syncs`: the same `(1, W)` trainer to
`(W, 1)` rollout binding, the same `weight_sync` call and the same timing bracket, but the receiver is
`roles/rollout/vllm.py::VLLMRollout` loading in process -- no second hop, no digests -- so it pairs with the GR00T
bench's verification-off arm. Every publication carries a real change (the first witness parameter perturbed by
>= 2 bf16 ULP on the trainer) and is verified OUTSIDE the timed bracket: the rollout's live witness tensors equal the
trainer's bitwise on every lane, the perturbed witness moved, the version advanced, every bucket was applied.

Run inside a Ray cluster on one node (colocated, CUDA IPC); `--bench-warmup` publications precede the
`--bench-syncs` measured ones. The report mirrors the GR00T driver's `bench` section (`stages`, `lane_spread`, raw
samples) so the two can be read side by side; `plan` and `tied_aliases` record the wire volume, including the
tied-weight aliases the identity manifest ships and vLLM's `load_weights` discards.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time
from typing import Any

import torch

from cosmos_framework.posttrain.tests.helpers.weight_sync_bench import (
    bind_pair,
    check_witness_values,
    columns,
    identical,
    lane_spread,
    medians,
    revisions,
    setup_hf_policy,
    setup_vllm_rollout,
    stage_rows,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
#: Non-fused parameters that keep the SAME name on both sides (vLLM fuses q/k/v and gate/up); the first is perturbed.
DEFAULT_WITNESSES = ("model.layers.0.input_layernorm.weight", "model.layers.0.mlp.down_proj.weight")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--bench-syncs", type=int, default=10, help="measured publications")
    parser.add_argument("--bench-warmup", type=int, default=3, help="warmup publications (recorded, not summarized)")
    parser.add_argument(
        "--witness",
        action="append",
        default=None,
        help="parameter read back bitwise on every lane (repeatable); the first one is perturbed before every publication",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--assert", dest="do_assert", action="store_true", help="non-zero exit unless OVERALL PASS")
    args = parser.parse_args(argv)
    if args.bench_syncs < 1 or args.bench_warmup < 0:
        parser.error("--bench-syncs must be >= 1 and --bench-warmup >= 0")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    args.witness = tuple(args.witness) if args.witness else DEFAULT_WITNESSES
    return args


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_root = os.path.abspath(args.out_root)
        os.makedirs(self.out_root, exist_ok=True)
        self.failures: list[str] = []
        self.report: dict[str, Any] = {
            "command": " ".join(sys.argv),
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "revisions": revisions(),
            "model": args.model,
        }
        self.previous_witness: dict[str, torch.Tensor] | None = None

    def fail(self, message: str) -> None:
        print(f"FAIL: {message}")
        self.failures.append(message)

    def execute(self) -> dict[str, Any]:
        args = self.args
        bound = bind_pair(
            functools.partial(
                setup_vllm_rollout,
                model=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            ),
            functools.partial(setup_hf_policy, model=args.model),
            placement="colocated",
            name="vllm_bench",
        )
        self.rollout, self.policy = bound.rollout, bound.policy
        self.report["mesh"] = {
            "nodes": bound.nodes,
            "gpus_per_node": bound.gpus_per_node,
            "width": bound.width,
            "placement": "colocated",
        }
        self.report["bind_seconds"] = bound.bind_seconds
        try:
            self.report["plan"] = identical(self.policy.plan_summary().all(), "bucket plan")
            self.report["tied_aliases"] = identical(self.policy.tied_aliases().all(), "tied aliases")
            movers = sorted({s["mover_kind"] for s in self.rollout.probe_state().all().values()})
            self.report["movers"] = movers
            if movers != ["IpcMover"]:
                self.fail(f"colocated placement must bind IpcMover on every lane, got {movers}")
            print(f"plan: {self.report['plan']} tied aliases: {self.report['tied_aliases']} movers: {movers}")
            samples: list[dict[str, Any]] = []
            for index in range(args.bench_warmup + args.bench_syncs):
                sample = self.publish(index + 1)
                sample["warmup"] = index < args.bench_warmup
                samples.append(sample)
            measured = [s for s in samples if not s["warmup"]]
            seconds = [s["sync_s"] for s in measured]
            self.report["bench"] = {
                "warmup": args.bench_warmup,
                "measured": args.bench_syncs,
                "samples": samples,
                "summary": {
                    "n": len(seconds),
                    "min": min(seconds),
                    "median": statistics.median(seconds),
                    "max": max(seconds),
                    "samples": seconds,
                },
                "stages": medians(columns([s["rows"] for s in measured])),
                "lane_spread": lane_spread([s["totals"] for s in measured]),
            }
            summary = self.report["bench"]["summary"]
            print(f"[bench] vllm median {summary['median']:.3f}s (min {summary['min']:.3f}, max {summary['max']:.3f})")
        finally:
            import ray

            ray.shutdown()  # `VLLMRollout` has no close(): the engines die with their actors
        self.report["failures"] = list(self.failures)
        self.report["overall"] = "PASS" if not self.failures else "FAIL"
        return self.report

    def publish(self, version: int) -> dict[str, Any]:
        """One publication: perturb the first witness, sync (the only timed call), drain the stage records, verify."""
        from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

        witness = list(self.args.witness)
        failures: list[str] = []
        perturbed = identical(self.policy.perturb_parameter(witness[0]).all(), "perturbation")
        self.policy.set_publication_version(version).all()
        started = time.perf_counter()
        weight_sync(self.rollout, self.policy, validation_step=version)
        sync_s = time.perf_counter() - started
        push = self.policy.push_timings().all()
        lanes = self.rollout.apply_timings().all()
        rows, totals = stage_rows(sync_s, push, lanes)
        sample: dict[str, Any] = {
            "version": version,
            "sync_s": sync_s,
            "push": {str(c): t for c, t in push.items()},
            "lanes": {str(c): r for c, r in lanes.items()},
            "rows": rows,
            "totals": totals,
            "perturbation": perturbed,
        }

        expected_by_rank = self.policy.witness_values(witness).all()
        expected = expected_by_rank[sorted(expected_by_rank)[0]]
        failures.extend(check_witness_values(expected, self.rollout.witness_values(witness).all()))
        if self.previous_witness is not None and torch.equal(self.previous_witness[witness[0]], expected[witness[0]]):
            failures.append(f"the perturbation left {witness[0]} unchanged on the trainer")
        self.previous_witness = expected
        if perturbed["changed_after_cast_fraction"] != 1.0:
            failures.append(
                f"only {perturbed['changed_after_cast_fraction']:.3f} of the perturbed elements survive the cast"
            )
        for cell, held in self.rollout.weight_version().all().items():
            if held != version:
                failures.append(f"rollout cell {cell} holds weight_version {held}, expected {version}")
        buckets = self.report["plan"]["buckets"]
        for cell, applied in lanes.items():
            if len(applied) != buckets:
                failures.append(f"rollout cell {cell} applied {len(applied)} buckets, the plan has {buckets}")
        for cell, state in self.rollout.probe_state().all().items():
            if state["pending"] is not None or state["prepared"] is not None:
                failures.append(f"rollout cell {cell} is not cleanly committed: {state}")
        sample["failures"] = failures
        for failure in failures:
            self.fail(f"v{version}: {failure}")
        print(f"[v{version}] weight_sync {sync_s:.3f}s, {len(failures)} failure(s)")
        return sample


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Run(args).execute()
    path = os.path.join(os.path.abspath(args.out_root), "weight_sync_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport: {path}")
    for failure in report["failures"]:
        print(f"FAIL {failure}")
    print("\nOVERALL:", report["overall"])
    if not args.do_assert:
        return 0
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
