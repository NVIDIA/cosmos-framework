#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# RECONSTRUCTION: the tables and metrics.json of the recipe were assembled by hand from the run records; this script
# recomputes them from the per-episode summaries that eval_fidelity.py writes, so every number has a file behind it.
"""make_results.py - results_tables_heldout.md, results_tables_swap20.md and results/metrics.json.

  make_results.py --eval heldout/eval --outputs outputs --out-dir <assets/sim-bimanual> --meta meta.json
      <eval>/matched, <eval>/frozen, <eval>/swap20_vs_real  : eval_fidelity.py --batch output directories
      meta.json console_logs: inference logs, each a file or a run directory holding console.log (per-rollout seconds;
                                                              a re-used -o directory keeps only the last run's console.log, so pass the shell log of all runs)
      meta.json                                             : hardware, revisions, peak-memory lines (see the run record)

Every statistic is a mean over episodes with the sample standard deviation (ddof = 1). The tables' first line is a `# ` heading.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

FRAMES = [30, 60, 120, 240, 480, 900]
WINDOWS = [("0-300", "frames 0-299"), ("300-600", "300-599"), ("600-end", "600-900"), ("overall", "overall")]
METRICS = (("psnr", "PSNR (dB)", 2), ("ssim", "SSIM", 3), ("lpips", "LPIPS", 3))


def load_summaries(d: Path) -> dict[str, dict]:
    return {p.name[: -len("_summary.json")]: json.loads(p.read_text()) for p in sorted(d.glob("*_summary.json")) if p.name != "aggregate_summary.json"}


def ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")


def subset_stats(summ: dict[str, dict], names: list[str]) -> dict:
    out = {"checkpoints": {}, "windows": {}}
    for f in FRAMES:
        out["checkpoints"][f] = {m: ms([summ[n]["checkpoints"][str(f)][m] for n in names]) for m, _, _ in METRICS}
    for key, _ in WINDOWS:
        out["windows"][key] = {m: ms([summ[n]["windows"][key][m] for n in names]) for m, _, _ in METRICS}
    out["n"] = len(names)
    return out


def fmt(mean_std, nd) -> str:
    return f"{mean_std[0]:.{nd}f} ± {mean_std[1]:.{nd}f}"


def table(rows: list[tuple[str, dict]], kind: str) -> str:
    if kind == "checkpoints":
        head = ["comparison", "metric"] + [f"f{f}" for f in FRAMES]
    else:
        head = ["comparison", "metric"] + [lab for _, lab in WINDOWS]
    body = []
    for label, st in rows:
        for m, mname, nd in METRICS:
            vals = ([st["checkpoints"][f][m] for f in FRAMES] if kind == "checkpoints" else [st["windows"][k][m] for k, _ in WINDOWS])
            body.append([label, mname] + [fmt(v, nd) for v in vals])
    widths = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(head), "| " + " | ".join("-" * w for w in widths) + " |"] + [line(r) for r in body])


def rollout_seconds(console: Path) -> list[tuple[str, float]]:
    """(name, seconds) per sample from the 'Saved sample args' -> 'Saved sample outputs' timestamps of one inference log."""
    t = lambda s: datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")
    starts, res = {}, []
    for line in console.read_text(errors="replace").splitlines():
        m = re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\|.*Saved sample (args|outputs) to '.*/([^/]+)/sample_(?:args|outputs)\.json'", line)
        if not m:
            continue
        ts, kind, name = m.groups()
        if kind == "args":
            starts[name] = t(ts)
        elif name in starts:
            res.append((name, (t(ts) - starts[name]).total_seconds()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--outputs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    a = ap.parse_args()
    meta = json.loads(a.meta.read_text())

    matched, frozen, swap = (load_summaries(a.eval / d) for d in ("matched", "frozen", "swap20_vs_real"))
    assert len(matched) == 20 and len(frozen) == 20 and len(swap) == 20, (len(matched), len(frozen), len(swap))
    t600 = [n for n in matched if n.startswith("task600_")]
    t596 = [n for n in matched if n.startswith("task596_")]
    S = {"matched_all": subset_stats(matched, list(matched)), "matched_600": subset_stats(matched, t600),
         "matched_596": subset_stats(matched, t596), "frozen": subset_stats(frozen, list(frozen))}
    sw600 = [n for n in swap if n.startswith("swap_task600_")]
    sw596 = [n for n in swap if n.startswith("swap_task596_")]
    S |= {"swap_all": subset_stats(swap, list(swap)), "swap_600": subset_stats(swap, sw600), "swap_596": subset_stats(swap, sw596)}

    note = (f"Released checkpoint `nvidia/Cosmos3-Nano-Sim-Bimanual` revision `{meta['checkpoint_revision']}`, 4 steps, KV cache 64, attention sink 8, "
            f"seed {meta['seed']}, {meta['hardware']}. Mean ± std across episodes (ddof=1); real video cropped to its top 352 rows "
            f"(`--align auto` chose `top` for every episode; see `manifests/RECORD.md`).")
    heldout_rows = [("matched actions, all 20 episodes", S["matched_all"]), ("matched, task 600 (10)", S["matched_600"]),
                    ("matched, task 596 (10)", S["matched_596"]), ("frozen first frame vs real (20), static baseline", S["frozen"])]
    swap_rows = [("cross-task action swap vs real, all 20 episodes", S["swap_all"]), ("swap, frame episode task 600 (10)", S["swap_600"]),
                 ("swap, frame episode task 596 (10)", S["swap_596"])]
    (a.out_dir / "results_tables_heldout.md").write_text(
        "# Sim-Bimanual held-out fidelity tables (released checkpoint)\n\n"
        "Matched-action rollouts and the frozen-first-frame control on the 20 held-out AgiBotWorld-Beta episodes, all 20 and per task. " + note +
        "\n\n## Checkpoint frames\n\n" + table(heldout_rows, "checkpoints") + "\n\n## Window means\n\n" + table(heldout_rows, "windows") + "\n")
    (a.out_dir / "results_tables_swap20.md").write_text(
        "# Sim-Bimanual action-swap control tables (released checkpoint)\n\n"
        "Cross-task action swap on all 20 held-out episodes: each rollout keeps its frame episode's first frame and prompt and takes the 900 actions of "
        "the paired episode of the other task. " + note +
        "\n\n## Checkpoint frames\n\n" + table(swap_rows, "checkpoints") + "\n\n### Window means\n\n" + table(swap_rows, "windows") + "\n")

    # paired comparisons and the working-run rule
    agg = lambda d: json.loads((a.eval / d / "aggregate_summary.json").read_text())
    per_frame = lambda d: np.genfromtxt(a.eval / d / "aggregate_per_frame.csv", delimiter=",", names=True)
    pm, pf, ps = per_frame("matched"), per_frame("frozen"), per_frame("swap20_vs_real")
    diff_mf = pm["psnr_mean"] - pf["psnr_mean"]
    frames = pm["frame"].astype(int)
    beats_from = next((int(frames[i]) for i in range(len(frames)) if (diff_mf[i:] > 0).all()), None)
    first_cross = next((int(frames[i]) for i in range(len(frames)) if diff_mf[i] > 0), None)
    paired = np.array([matched[n]["windows"]["overall"]["psnr"] - frozen[n]["windows"]["overall"]["psnr"] for n in matched])
    swap_of = {n: f"swap_{n}_actions_" for n in matched}
    ov = lambda s: s["windows"]["overall"]["psnr"]
    swap_by_frame = {n: next(v for k, v in swap.items() if k.startswith(swap_of[n])) for n in matched}
    paired_swap = np.array([ov(matched[n]) - ov(swap_by_frame[n]) for n in matched])
    rule = {
        "matched_overall_psnr": S["matched_all"]["windows"]["overall"]["psnr"][0],
        "frozen_overall_psnr": S["frozen"]["windows"]["overall"]["psnr"][0],
        "swap_overall_psnr": S["swap_all"]["windows"]["overall"]["psnr"][0],
        "matched_beats_frozen_per_frame_mean_psnr_first_frame": first_cross,
        "matched_beats_frozen_at_every_frame_from": beats_from,
        "matched_minus_frozen_paired_overall_psnr_db": {"mean": float(paired.mean()), "min": float(paired.min()), "max": float(paired.max()),
                                                        "episodes_better": int((paired > 0).sum()), "of": int(paired.size)},
        "matched_minus_swap_paired_overall_psnr_db": {"mean": float(paired_swap.mean()), "min": float(paired_swap.min()), "max": float(paired_swap.max()),
                                                      "episodes_better": int((paired_swap > 0).sum()), "of": int(paired_swap.size)},
    }
    rule["matched_beats_swap_overall"] = rule["matched_overall_psnr"] > rule["swap_overall_psnr"]
    rule["recipe_rule"] = "matched beats frozen from about frame 60 and beats the action swap overall; reported as measured, not tuned"

    # speed
    secs = []
    for run in meta["console_logs"]:
        log = a.outputs / run
        secs += rollout_seconds(log if log.is_file() else log / "console.log")
    sec = np.array([s for _, s in secs])
    speed = {"rollouts_timed": int(sec.size), "generation_window_s": {"min": float(sec.min()), "max": float(sec.max()), "mean": float(sec.mean())},
             "frames_per_second_mean": float(901 / sec.mean()), "peak_torch_memory_gb": meta.get("peak_torch_memory_gb"),
             "peak_gpu_mib_nvidia_smi": meta.get("peak_gpu_mib_nvidia_smi"), "wall_s_by_run": meta.get("wall_s_by_run"),
             "published_agibot_001_example": meta.get("published_example")}

    def cp(st):
        return {m: {"frames": FRAMES, "values": [{"mean": round(st["checkpoints"][f][m][0], 4), "std": round(st["checkpoints"][f][m][1], 4)} for f in FRAMES]}
                for m, _, _ in METRICS}

    def wn(st):
        return {m: {"windows": [k for k, _ in WINDOWS], "values": [{"mean": round(st["windows"][k][m][0], 4), "std": round(st["windows"][k][m][1], 4)} for k, _ in WINDOWS]}
                for m, _, _ in METRICS}

    metrics = {
        "_about": "Every number in the tables of the Sim-Bimanual skill README and result tables, recomputed by tools/make_results.py from the eval_fidelity.py summaries. Means over episodes, std with ddof = 1.",
        "_record": "manifests/RECORD.md and the per-phase run record (command, revision, GPU, wall time, peak memory) of this run",
        "_measured_on": meta["hardware"] + f"; checkpoint {meta['checkpoint_revision']}; framework {meta['framework_revision']}",
        "speed_and_memory": speed,
        "fidelity_checkpoint_frames_20_heldout_episodes": {
            "matched_actions": cp(S["matched_all"]), "frozen_first_frame_control": cp(S["frozen"]),
            "wrong_actions_same_20_episodes_control": cp(S["swap_all"]),
            "per_task_matched_rows": {"task_600_10": cp(S["matched_600"]), "task_596_10": cp(S["matched_596"])}},
        "fidelity_window_means": {
            "matched_20_episodes": wn(S["matched_all"]), "matched_task_600_10": wn(S["matched_600"]), "matched_task_596_10": wn(S["matched_596"]),
            "frozen_first_frame_20": wn(S["frozen"]), "wrong_actions_same_20_episodes": wn(S["swap_all"]),
            "swap_per_task_rows": {"frame_episode_task_600_10": wn(S["swap_600"]), "frame_episode_task_596_10": wn(S["swap_596"])}},
        "derived_in_README_prose": {
            "working_run_rule": rule,
            "frame0_alignment_PSNR_dB": {
                "chosen": sorted({matched[n]["alignment"]["chosen"] for n in matched}),
                "chosen_range": [round(min(matched[n]["alignment"]["chosen_psnr_frame0"] for n in matched), 2), round(max(matched[n]["alignment"]["chosen_psnr_frame0"] for n in matched), 2)],
                "other_crops_range": [round(min(v for n in matched for k, v in matched[n]["alignment"]["candidates_psnr_frame0"].items() if k != "top"), 2),
                                      round(max(v for n in matched for k, v in matched[n]["alignment"]["candidates_psnr_frame0"].items() if k != "top"), 2)],
                "example_task600_ep815362": matched["task600_ep815362"]["alignment"]["candidates_psnr_frame0"]},
        },
        "conversion_gate": meta.get("conversion_gate"),
        "checkpoint_download": meta.get("checkpoint_download"),
    }
    (a.out_dir / "results").mkdir(exist_ok=True)
    (a.out_dir / "results" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"working_run_rule": rule, "speed": speed}, indent=1))


if __name__ == "__main__":
    main()
