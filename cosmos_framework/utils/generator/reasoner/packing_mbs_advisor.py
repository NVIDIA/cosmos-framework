# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""max_batch_size (mbs) advisor for the VLM dynamic batcher.

The best ``max_batch_size`` depends on the *length distribution* of the data: dense data with
many similar-length samples packs well at higher mbs (amortizing the fixed per-step FSDP
collective over more useful tokens), while long-tailed data hits the padding tax / FLOP budget
and is better off near mbs1. This tool answers "what mbs should I train with?" by simulating the
REAL packer (``PackingIterableDataset`` -- the exact code path, no re-implementation/drift) over
an empirical or synthetic length distribution and scoring each candidate mbs with a transparent,
calibratable throughput + memory model.

Why simulate instead of just sweeping on the cluster: a sweep (the validation harness) is the
ground truth but costs GPU-hours per point; this gives a fast, free first cut and a principled
default, and its throughput model is *calibrated from* the harness's mbs1/mbs4 telemetry
(``realized_microbatch_ms`` at two mbs -> overhead + slope), so tool and harness reinforce each other.

Inputs
------
``--lengths-file``  one sample per line, ``"<input_ids_len>"`` or ``"<len>,<modality>"`` with
    modality in {text,image,video}. Modality matters because the packer only co-packs same-modality
    samples, so segregation lowers density; default (no modality column) = all text = optimistic
    upper bound. Produce this file from a short loader dry-run or from logged ``seq_max_len`` values.
``--synthetic``     quick what-if: a mixture of a short-body lognormal and a long tail.

Throughput model (per step, calibratable -- see ``--overhead-ms`` / ``--ms-per-1k-padtok``)::

    step_ms          = overhead_ms + ms_per_1k_padtok * padded_tokens_per_step / 1000
    useful_tok_per_s = useful_tokens_per_step / (step_ms / 1000)

``overhead_ms`` is the fixed per-step cost (FSDP all-gather/reduce-scatter + launch) that bigger
batches amortize; ``ms_per_1k_padtok`` is the marginal compute per 1k PADDED tokens (padding is
paid for). Calibrate both from the harness: with realized_microbatch_ms R1 (mbs1) and R4 (mbs4) and the
simulated padded_tokens_per_step P1,P4 -> slope = (R4-R1)/((P4-P1)/1000), overhead = R1 - slope*P1/1000.

Memory model (exploratory estimate -- calibrate ``--act-mb-per-1k-padtok`` from max-rank reserved memory)::

    est_mean_working_set_gb = weights_gb + act_mb_per_1k_padtok * padded_tokens_per_step / 1000 / 1024

Candidate: the largest modeled-throughput mbs whose ``over_budget_rate == 0`` and rough memory
estimate is below ``--mem-cap-gb``. This is not an OOM-safety gate or a production recommendation.

Caveats (read before trusting a number)
---------------------------------------
This NARROWS the sweep; it does not replace it. (1) The throughput model is calibrated against the
current cost model, which under-predicts (``realized_over_predicted`` ≈0.62 at mbs4, pending the
``is_causal`` refit), so treat tps as RELATIVE (for ranking mbs), not absolute.
(2) The memory model omits vision patch count, attention topology, activation checkpointing,
FSDP residency, fragmentation, and per-rank skew. It is an exploratory ranking feature only; measure
max reserved memory on every rank. (3) The candidate is valid only for the length distribution
fed in -- re-run when the data mix, resolution/fps, or weights change (validity envelope).

Schedule derivation (``max_iter`` for a target number of epochs / samples / tokens)
-----------------------------------------------------------------------------------
Dynamic batching packs a *variable* number of samples per step, so the number of optimizer steps to
reach a fixed number of passes over the data is not ``corpus / const_batch`` -- it depends on the
realized packing density (which is exactly what this tool simulates). The cosine LR schedule's period
is tied to ``trainer.max_iter``, so an mis-set ``max_iter`` either decays the LR before the data target
is reached or never reaches the floor. Given a target, the tool derives the ``max_iter`` that hits it::

    global_batch_per_step = per_step_local * dp_size * grad_accum        # per_step_local from simulation
    max_iter              = ceil(target_total / global_batch_per_step)

where ``per_step_local`` is the simulated mean ``samples_per_step`` (sample basis) or
``useful_tokens_per_step`` (token basis), ``target_total`` is ``epochs * corpus`` (or an explicit
``--target-samples`` / ``--target-tokens``), ``dp_size`` is the data-parallel world size (each replica
consumes a distinct shard), and ``grad_accum`` is micro-batches per optimizer step. This is an
out-of-process, deterministic pre-step: it emits a *number* to pass as ``trainer.max_iter=<N>`` and
never touches the training loop. Pass ``--true-packing`` / ``--flat-budget`` to match the layout the
run will use so the simulated density matches training. Because dynamic batching makes samples/step
vary run-to-run, treat the derived ``max_iter`` as a calibrated estimate -- confirm the simulated
``samples_per_step`` against the realized packing telemetry on a short run before committing a schedule.
This exposure-matched schedule changes the optimizer update count, effective global batch, Adam moment
trajectory, and LR schedule whenever packing density changes. It is therefore a distinct optimization
regime, not a packing-only A/B. For a packing-only comparison, hold logical samples/tokens per optimizer
update, update count, and schedule fixed. For the larger-batch exposure-matched regime, retune LR and do
not attribute quality changes solely to packing.

Run inside the imaginaire4 container (imports torch via the packer)::

    python -m cosmos_framework.utils.generator.reasoner.packing_mbs_advisor --synthetic --max-tokens 16000
    python -m cosmos_framework.utils.generator.reasoner.packing_mbs_advisor --lengths-file lens.txt \
        --mbs-grid 1,2,4,8 --num-steps 4000 --overhead-ms 808 --ms-per-1k-padtok 68.9 --mem-cap-gb 175
    # derive max_iter for 3 epochs over a 4.2M-sample corpus at dp=256, true packing:
    python -m cosmos_framework.utils.generator.reasoner.packing_mbs_advisor --lengths-file lens.txt \
        --true-packing --flat-budget --num-samples 4200000 --epochs 3 --dp-size 256 --grad-accum 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Iterator
from typing import Any

import torch

from cosmos_framework.data.generator.packing_iterable_dataset import PackingIterableDataset, round_up_to_multiple

_MODALITY_KEY = {"image": "pixel_values", "video": "pixel_values_videos", "text": None}


class _LengthStreamDataset(torch.utils.data.IterableDataset):
    """Streams samples drawn (with replacement) from an empirical (length, modality) population.

    Each yielded sample carries only an ``input_ids`` list of the sampled length plus the modality
    marker key the packer keys on (``pixel_values`` / ``pixel_values_videos`` / none) -- enough for
    the packer's length + modality logic without materializing real tensors.
    """

    def __init__(self, population: list[tuple[int, str]], seed: int) -> None:
        self.population = population
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed)
        while True:
            length, modality = self.population[rng.randrange(len(self.population))]
            sample: dict[str, Any] = {"input_ids": [1] * length}
            marker = _MODALITY_KEY.get(modality)
            if marker is not None:
                sample[marker] = 1  # presence-only marker; packer checks ``key in sample``
            yield sample


class _AdvisorPacker(PackingIterableDataset):
    def compute_sample_tokens(self, sample: dict[str, Any]) -> int:
        return len(sample["input_ids"])


def simulate_mbs(
    population: list[tuple[int, str]],
    mbs: int,
    *,
    max_tokens: int,
    pool_size: int,
    long_threshold: int,
    batching_strategy: str,
    num_steps: int,
    seed: int,
    true_packing: bool = False,
    flat_budget: bool = False,
) -> dict[str, float | int | bool]:
    """Run the real packer and aggregate layout-aware packing statistics."""
    if not population:
        raise ValueError("population must not be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if mbs <= 0:
        raise ValueError(f"mbs must be positive, got {mbs}")
    if mbs > pool_size:
        raise ValueError(f"mbs ({mbs}) cannot exceed pool_size ({pool_size}); advisor preserves the requested pool")
    ds = _AdvisorPacker(
        datasets_cfg={"pop": {"dataset": _LengthStreamDataset(population, seed), "ratio": 1.0}},
        max_tokens=max_tokens,
        pool_size=pool_size,
        max_batch_size=mbs,
        long_threshold=long_threshold,
        batching_strategy=batching_strategy,
        seed=seed,
        legacy_budget_gate=False,
        true_packing=true_packing,
        flat_budget=flat_budget,
    )
    packed = ds.true_packing
    it = iter(ds)
    samples = useful = processed = singleton = over_budget = 0
    content_attn = attended_attn = equivalent_padded_attn = packed_dense_attn = 0
    for _ in range(num_steps):
        batch = next(it)
        k = len(batch)
        lens = [len(s["input_ids"]) for s in batch]
        cur_max = max(lens)
        cur_sum = sum(lens)
        samples += k
        useful += cur_sum
        batch_content_attn = sum(length * length for length in lens)
        content_attn += batch_content_attn
        equivalent_padded_length = round_up_to_multiple(cur_max)
        equivalent_padded_attn += k * equivalent_padded_length * equivalent_padded_length
        if packed:
            # True packing pays block-diagonal logical attention plus the alignment tail as its own block.
            row = round_up_to_multiple(cur_sum)
            tail = row - cur_sum
            processed += row
            attended_attn += batch_content_attn + tail * tail
            packed_dense_attn += row * row
        else:
            # Padded: k rows each padded to round16(max); dense per-row attention on all k rows.
            plen = equivalent_padded_length
            processed += plen * k
            attended_attn += plen * plen * k
            packed_dense_attn += plen * plen * k
        singleton += 1 if k == 1 else 0
        over_budget += int(batch[0].get("over_budget", 0))
    return {
        "mbs": mbs,
        "true_packing": packed,
        "samples_per_step": samples / num_steps,
        "useful_tokens_per_step": useful / num_steps,
        # Processed tokens the kernel actually sees (padded: k*round16(max); true packing: round16(sum)).
        # Key name kept as ``padded_tokens_per_step`` so the throughput/memory calibration and JSON
        # schema are unchanged; in the default (padded) layout its value is identical to before.
        "padded_tokens_per_step": processed / num_steps,
        "processed_tokens_per_step": processed / num_steps,
        "useful_util": useful / processed if processed else 0.0,
        "attn_quadratic_waste": 1.0 - content_attn / attended_attn if attended_attn else 0.0,
        "equivalent_padded_attention_waste": (
            1.0 - content_attn / equivalent_padded_attn if equivalent_padded_attn else 0.0
        ),
        "attn_block_sparsity": (1.0 - attended_attn / packed_dense_attn if packed and packed_dense_attn else 0.0),
        "singleton_rate": singleton / num_steps,
        "over_budget_rate": over_budget / num_steps,
    }


def score_throughput(stat: dict[str, float | int], *, overhead_ms: float, ms_per_1k_padtok: float) -> float:
    step_ms = overhead_ms + ms_per_1k_padtok * stat["padded_tokens_per_step"] / 1000.0
    return stat["useful_tokens_per_step"] / (step_ms / 1000.0) if step_ms > 0 else 0.0


def estimate_mean_working_set_gb(
    stat: dict[str, float | int], *, weights_gb: float, act_mb_per_1k_padtok: float
) -> float:
    return weights_gb + act_mb_per_1k_padtok * stat["padded_tokens_per_step"] / 1000.0 / 1024.0


def compute_max_iter(per_step_local: float, target_total: float, dp_size: int, grad_accum: int) -> int:
    """Optimizer steps to process ``target_total`` items given ``per_step_local`` items per DP rank per micro-step.

    One optimizer step consumes ``per_step_local * dp_size * grad_accum`` items: the ``dp_size`` data-parallel
    replicas each process a distinct shard, and ``grad_accum`` micro-batches accumulate into one step. Rounds
    up so the target is fully covered (the LR schedule length must not end before the data target is reached).
    ``target_total`` is a count of samples or useful tokens; ``per_step_local`` must be the matching per-step
    metric (``samples_per_step`` for a sample target, ``useful_tokens_per_step`` for a token target).
    """
    per_step_global = per_step_local * dp_size * grad_accum
    if per_step_global <= 0:
        raise ValueError("per-step throughput must be > 0 to derive max_iter")
    if target_total <= 0:
        raise ValueError("target_total must be > 0 to derive max_iter")
    return math.ceil(target_total / per_step_global)


def resolve_schedule_target(args: argparse.Namespace) -> tuple[float | None, str | None, str | None]:
    """Resolve the schedule target from the CLI into ``(target_total, per_step_key, unit)``.

    Sample basis (``--num-samples`` / ``--target-samples``) and token basis (``--total-tokens`` /
    ``--target-tokens``) are mutually exclusive. An explicit ``--target-samples`` / ``--target-tokens``
    overrides ``epochs * corpus``. Returns ``(None, None, None)`` when no target was requested, in which
    case the tool skips schedule derivation entirely (output identical to the pre-feature advisor).
    """
    sample_basis = args.num_samples is not None or args.target_samples is not None
    token_basis = args.total_tokens is not None or args.target_tokens is not None
    if sample_basis and token_basis:
        raise SystemExit(
            "choose a sample basis (--num-samples/--target-samples) OR a token basis "
            "(--total-tokens/--target-tokens), not both"
        )
    if token_basis:
        total = args.target_tokens if args.target_tokens is not None else args.epochs * args.total_tokens
        return float(total), "useful_tokens_per_step", "useful tokens"
    if sample_basis:
        total = args.target_samples if args.target_samples is not None else args.epochs * args.num_samples
        return float(total), "samples_per_step", "samples"
    return None, None, None


def load_population(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.lengths_file:
        population: list[tuple[int, str]] = []
        with open(args.lengths_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                length = int(float(parts[0]))
                modality = parts[1].strip().lower() if len(parts) > 1 else "text"
                if modality not in _MODALITY_KEY:
                    raise ValueError(f"unknown modality {modality!r} (expected text/image/video)")
                population.append((length, modality))
        if not population:
            raise ValueError(f"no samples parsed from {args.lengths_file}")
        return population
    # Synthetic: a short-body lognormal plus a heavy long tail (a rough VLM-like profile).
    rng = random.Random(args.seed)
    population = []
    for _ in range(args.synthetic_size):
        if rng.random() < args.synthetic_long_frac:
            length = int(rng.uniform(args.long_threshold, max(args.long_threshold + 1, args.max_tokens)))
        else:
            length = int(min(args.max_tokens, max(16, rng.lognormvariate(args.synthetic_mu, args.synthetic_sigma))))
        population.append((length, "text"))
    return population


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--lengths-file", help="one '<len>' or '<len>,<modality>' per line")
    src.add_argument("--synthetic", action="store_true", help="use a synthetic length distribution")
    p.add_argument("--mbs-grid", default="1,2,4,8", help="comma-separated max_batch_size candidates")
    p.add_argument("--num-steps", type=int, default=4000, help="packer steps to simulate per mbs")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--pool-size", type=int, default=16)
    p.add_argument("--long-threshold", type=int, default=6400)
    p.add_argument("--batching-strategy", default="prefer_closest", choices=["prefer_first", "prefer_closest"])
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--flat-budget", action="store_true", help="simulate the explicit fixed flat-budget policy")
    # Throughput model (calibrate from harness realized_microbatch_ms at mbs1/mbs4).
    p.add_argument(
        "--overhead-ms",
        type=float,
        default=808.0,
        help="fixed per-microbatch ms (FSDP comm + launch); calibrated from realized_microbatch_ms",
    )
    p.add_argument(
        "--ms-per-1k-padtok",
        type=float,
        default=68.9,
        help="marginal compute ms per 1k PADDED tokens; calibrated",
    )
    # Memory model (calibrate from harness peak_window_reserved_gb).
    p.add_argument("--weights-gb", type=float, default=40.0, help="static weights+optimizer+ckpt footprint (GB)")
    p.add_argument("--act-mb-per-1k-padtok", type=float, default=900.0, help="activation MB per 1k padded tokens")
    p.add_argument("--mem-cap-gb", type=float, default=180.0, help="exploratory mean-working-set cap")
    # Synthetic distribution knobs.
    p.add_argument("--synthetic-size", type=int, default=20000)
    p.add_argument("--synthetic-mu", type=float, default=6.5, help="lognormal mu (~e^mu median len)")
    p.add_argument("--synthetic-sigma", type=float, default=0.8)
    p.add_argument("--synthetic-long-frac", type=float, default=0.05)
    # Layout: use explicit, hermetic policy knobs instead of inheriting ambient training env vars.
    p.add_argument(
        "--true-packing",
        action="store_true",
        help="simulate true (B=1 varlen) packing",
    )
    # Schedule derivation: given a data target, emit the max_iter (cosine-LR period) that reaches it.
    # A sample basis and a token basis are mutually exclusive; leaving all of these unset skips the
    # schedule block entirely (output identical to the pre-feature advisor).
    p.add_argument("--num-samples", type=int, default=None, help="corpus size (logical samples in one epoch)")
    p.add_argument("--total-tokens", type=float, default=None, help="corpus useful-token count (one epoch)")
    p.add_argument("--epochs", type=float, default=1.0, help="passes over the corpus for the target (default 1)")
    p.add_argument(
        "--target-samples", type=float, default=None, help="explicit total samples (overrides epochs*corpus)"
    )
    p.add_argument(
        "--target-tokens", type=float, default=None, help="explicit total useful tokens (overrides epochs*corpus)"
    )
    p.add_argument("--dp-size", type=int, default=1, help="data-parallel world size (distinct shard per replica)")
    p.add_argument("--grad-accum", type=int, default=1, help="gradient-accumulation micro-batches per optimizer step")
    p.add_argument(
        "--warmup-frac", type=float, default=None, help="if set, also print warm_up_steps = round(frac*max_iter)"
    )
    p.add_argument("--json-out", help="optional path to write the full result table as JSON")
    args = p.parse_args(argv)

    population = load_population(args)
    lengths = [length for length, _ in population]
    lengths_sorted = sorted(lengths)

    def pct(q: float) -> int:
        return lengths_sorted[min(len(lengths_sorted) - 1, int(q * len(lengths_sorted)))]

    print(
        f"population: n={len(population)} | len mean={sum(lengths) / len(lengths):.0f} "
        f"p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={max(lengths)} | "
        f">=long_threshold({args.long_threshold}): {sum(x >= args.long_threshold for x in lengths) / len(lengths):.1%}"
    )

    mbs_grid = [int(x) for x in args.mbs_grid.split(",")]
    if any(mbs > args.pool_size for mbs in mbs_grid):
        raise ValueError(f"every mbs must be <= --pool-size ({args.pool_size}); got {mbs_grid}")
    rows: list[dict[str, float | int | bool]] = []
    for mbs in mbs_grid:
        stat = simulate_mbs(
            population,
            mbs,
            max_tokens=args.max_tokens,
            pool_size=args.pool_size,
            long_threshold=args.long_threshold,
            batching_strategy=args.batching_strategy,
            num_steps=args.num_steps,
            seed=args.seed,
            true_packing=bool(args.true_packing),
            flat_budget=bool(args.flat_budget),
        )
        stat["useful_tokens_per_sec"] = score_throughput(
            stat, overhead_ms=args.overhead_ms, ms_per_1k_padtok=args.ms_per_1k_padtok
        )
        stat["est_mean_working_set_gb"] = estimate_mean_working_set_gb(
            stat, weights_gb=args.weights_gb, act_mb_per_1k_padtok=args.act_mb_per_1k_padtok
        )
        stat["below_exploratory_cap"] = stat["est_mean_working_set_gb"] <= args.mem_cap_gb
        rows.append(stat)

    header = (
        f"{'mbs':>4} {'smpl/stp':>9} {'useful/stp':>11} {'padded/stp':>11} {'util':>6} "
        f"{'attnwaste':>9} {'singl%':>7} {'over%':>6} {'usef_tok/s':>11} {'meanGB':>7} {'cap':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mbs']:>4} {r['samples_per_step']:>9.2f} {r['useful_tokens_per_step']:>11.0f} "
            f"{r['padded_tokens_per_step']:>11.0f} {r['useful_util']:>6.3f} {r['attn_quadratic_waste']:>9.3f} "
            f"{r['singleton_rate'] * 100:>7.1f} {r['over_budget_rate'] * 100:>6.1f} "
            f"{r['useful_tokens_per_sec']:>11.0f} {r['est_mean_working_set_gb']:>7.1f} "
            f"{'yes' if r['below_exploratory_cap'] else 'NO':>5}"
        )

    eligible = [r for r in rows if r["below_exploratory_cap"] and r["over_budget_rate"] == 0.0]
    candidate: dict[str, float | int | bool] | None = None
    unsafe_fallback: dict[str, float | int | bool] | None = None
    if not eligible:
        print("\nNo mbs is below the exploratory cap with a clean budget; no candidate is emitted.")
        unsafe_fallback = min(rows, key=lambda row: row["est_mean_working_set_gb"])
        print(
            f"UNSAFE FALLBACK (informational only): mbs={unsafe_fallback['mbs']} "
            f"(est_mean_working_set_gb={unsafe_fallback['est_mean_working_set_gb']:.1f})."
        )
    else:
        candidate = max(eligible, key=lambda row: row["useful_tokens_per_sec"])
        print(
            f"\nEXPLORATORY CANDIDATE mbs = {candidate['mbs']}  "
            f"(useful_tokens/s~={candidate['useful_tokens_per_sec']:.0f}, "
            f"util={candidate['useful_util']:.3f}, "
            f"est_mean_working_set_gb={candidate['est_mean_working_set_gb']:.1f}, "
            f"singleton_rate={candidate['singleton_rate']:.3f})"
        )
    print(
        "NOTE: this is not an OOM-safety gate or production recommendation. Confirm candidates with "
        "controlled cluster runs and max-rank reserved-memory measurements."
    )

    # Schedule derivation: only when a data target was requested (else output is unchanged).
    target_total, per_step_key, unit = resolve_schedule_target(args)
    schedule: dict[str, float | int | str | None] | None = None
    if target_total is not None:
        for r in rows:
            r["max_iter"] = compute_max_iter(r[per_step_key], target_total, args.dp_size, args.grad_accum)
            if args.warmup_frac is not None:
                r["warm_up_steps"] = round(args.warmup_frac * r["max_iter"])
        basis = (
            f"{args.epochs:g} epoch(s) over {target_total / args.epochs:,.0f} {unit}"
            if (args.target_samples is None and args.target_tokens is None)
            else "explicit target"
        )
        warm_hdr = f" {'warm_up':>8}" if args.warmup_frac is not None else ""
        print(
            f"\n--- schedule: max_iter to reach {target_total:,.0f} {unit} "
            f"({basis}) | dp_size={args.dp_size} grad_accum={args.grad_accum} ---"
        )
        sched_header = f"{'mbs':>4} {'global_batch/stp':>17} {'max_iter':>10}{warm_hdr}"
        print(sched_header)
        print("-" * len(sched_header))
        for r in rows:
            per_step_global = r[per_step_key] * args.dp_size * args.grad_accum
            warm_col = f" {r['warm_up_steps']:>8}" if args.warmup_frac is not None else ""
            print(f"{r['mbs']:>4} {per_step_global:>17.1f} {r['max_iter']:>10}{warm_col}")
        if candidate is not None:
            rec_global = candidate[per_step_key] * args.dp_size * args.grad_accum
            rec_warm = (
                f", warm_up={round(args.warmup_frac * candidate['max_iter'])}" if args.warmup_frac is not None else ""
            )
            print(
                f"At EXPLORATORY CANDIDATE mbs={candidate['mbs']}: trainer.max_iter={candidate['max_iter']} "
                f"(global batch ~= {rec_global:,.1f} {unit}/step{rec_warm})"
            )
        else:
            print("No schedule recommendation is emitted because no candidate passed the exploratory filters.")
        print(
            "NOTE: max_iter derives from the simulated mean effective batch; dynamic batching makes "
            "samples/step vary run-to-run, so confirm the simulated per-step against realized telemetry "
            "before committing the schedule. This is a pre-step: pass the number as trainer.max_iter=<N>."
        )
        schedule = {
            "target_total": target_total,
            "unit": unit,
            "epochs": args.epochs,
            "dp_size": args.dp_size,
            "grad_accum": args.grad_accum,
            "recommended_max_iter": candidate["max_iter"] if candidate is not None else None,
        }

    if args.json_out:
        with open(args.json_out, "w") as fh:
            payload: dict[str, Any] = {
                "population_size": len(population),
                "policy": {
                    "true_packing": bool(args.true_packing),
                    "flat_budget": bool(args.flat_budget),
                    "legacy_budget_gate": False,
                    "pool_size": args.pool_size,
                    "seed": args.seed,
                },
                "rows": rows,
                "candidate_mbs": candidate["mbs"] if candidate is not None else None,
                "unsafe_fallback_mbs": unsafe_fallback["mbs"] if unsafe_fallback is not None else None,
            }
            if schedule is not None:
                payload["schedule"] = schedule
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
