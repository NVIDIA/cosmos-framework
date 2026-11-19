#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare forward-dynamics evaluations of several checkpoints and write ``results/metrics.json``.

Each run directory is the output of ``scripts/evaluate.sh`` (``<run>/<split>/seed_<s>/``). Samples
are matched by name across runs, so the comparison is paired (same window, same seed). Reports
PSNR (and SSIM/LPIPS when present) per split, overall and per SutureBot task.

Example::

    python scripts/summarize_results.py --run base=outputs/eval/base \\
        --run iter_1000=outputs/eval/iter_1000 --run iter_2000=outputs/eval/iter_2000 \\
        --reference base -o results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

TASKS = ("needle pickup", "needle throw", "knot tying")
METRICS = ("psnr", "ssim", "lpips")


def load_run(root: Path) -> dict[tuple[str, str, str], dict]:
    """``{(split, seed, sample): {metric: value, "task": ...}}``"""
    out = {}
    # <split>/seed_<s>/<family>/forward_dynamics/<window>/metrics.json (family: suturebot or suturebot_test)
    for metrics_file in sorted(root.glob("*/seed_*/*/forward_dynamics/*/metrics.json")):
        sample_dir = metrics_file.parent
        split, seed = sample_dir.parents[3].name, sample_dir.parents[2].name.removeprefix("seed_")
        metrics = json.loads(metrics_file.read_text())
        prompt = json.loads((sample_dir / "sample_args.json").read_text()).get("prompt", "")
        # The caption is "<robot context> <task>. <duration/resolution metadata>"; the task is the
        # sentence right after the context, which itself lists every task name.
        task = next((t for t in TASKS if f" {t}." in prompt), "unknown")
        out[(split, seed, sample_dir.name)] = {**{m: metrics[m] for m in METRICS if m in metrics}, "task": task}
    if not out:
        raise FileNotFoundError(f"no metrics.json under {root}")
    return out


def stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "sem": round(std / math.sqrt(n), 4), "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="<label>=<evaluate.sh output dir>")
    parser.add_argument("--reference", help="Label of the control run for paired deltas (e.g. base).")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--metadata", type=Path, help="JSON with revisions/hardware/settings to embed.")
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = load_run(Path(path))
    # Paired comparison: keep the samples every run has.
    common = set.intersection(*(set(r) for r in runs.values()))
    dropped = {label: len(r) - len(common) for label, r in runs.items() if len(r) != len(common)}

    result: dict = {"runs": {}, "paired_samples": len(common), "unpaired_samples_dropped": dropped}
    for label, run in runs.items():
        per_split: dict = defaultdict(dict)
        for split in sorted({k[0] for k in common}):
            keys = sorted(k for k in common if k[0] == split)
            for metric in METRICS:
                if metric not in run[keys[0]]:
                    continue
                entry = {"all": stats([run[k][metric] for k in keys])}
                for task in TASKS:
                    task_keys = [k for k in keys if run[k]["task"] == task]
                    if task_keys:
                        entry[task] = stats([run[k][metric] for k in task_keys])
                if args.reference and label != args.reference:
                    ref = runs[args.reference]
                    entry["delta_vs_" + args.reference] = stats([run[k][metric] - ref[k][metric] for k in keys])
                per_split[split][metric] = entry
        result["runs"][label] = dict(per_split)
    if args.metadata:
        result["metadata"] = json.loads(args.metadata.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    # Markdown table for the README (PSNR, mean +- SEM over samples).
    splits = sorted({k[0] for k in common})
    print("| Run | " + " | ".join(f"{s} PSNR (dB)" for s in splits) + " |")
    print("|---|" + "---:|" * len(splits))
    for label in runs:
        cells = []
        for s in splits:
            e = result["runs"][label][s]["psnr"]["all"]
            cells.append(f"{e['mean']:.2f} ± {e['sem']:.2f} (n={e['n']})")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
