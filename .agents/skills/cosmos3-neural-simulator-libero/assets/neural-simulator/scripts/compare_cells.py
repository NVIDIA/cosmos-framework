#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Side-by-side table of evaluation runs (summary.json + gripper_check.json) and a delta row.
Usage: compare_cells.py --out table.md ROW ...  where ROW = label=dir1[,dir2,...] (several dirs = sampling seeds,
reported as mean +- std).  --delta A B  prints a row with (A - B) for the headline metrics."""
import argparse, json, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("rows", nargs="+"); ap.add_argument("--delta", nargs=2, metavar=("A", "B")); ap.add_argument("--out", required=True)
a = ap.parse_args()

def load(d):
    d = Path(d); s = json.loads((d / "summary.json").read_text()); g = json.loads((d / "gripper_check.json").read_text()) if (d / "gripper_check.json").exists() else None
    om = s["overall_mean"]; bc = s["by_chunk_index"]
    r = dict(psnr=om["psnr"], ssim=om["ssim"], lpips=om["lpips"], psnr_left=om["psnr_left"], psnr_right=om["psnr_right"], lpips_left=om["lpips_left"], lpips_right=om["lpips_right"],
             psnr_c0=bc["0"]["psnr"], psnr_c5=bc.get("5", {}).get("psnr", float("nan")), psnr_c10=bc.get("10", {}).get("psnr", float("nan")), lpips_c10=bc.get("10", {}).get("lpips", float("nan")),
             n_frames=s["n_frames_evaluated"], n_eps=s["n_episodes"], s_per_chunk=s["gen_s_per_chunk_mean"], seed=s["run_config"]["seed"], ckpt=s["run_config"]["checkpoint_path"])
    if g: ag = g["aggregate"]; r.update(grip_gen=ag.get("gen_vs_command_recall"), grip_real=ag.get("real_vs_command_recall"), grip_lag_gen=ag.get("mean_lag_frames_gen"), grip_agree=ag.get("frame_agreement_mean"), grip_prec=ag.get("event_precision"), grip_rec=ag.get("event_recall"))
    return r

rows = {}
for spec in a.rows:
    label, dirs = spec.split("=", 1); rs = [load(d) for d in dirs.split(",")]
    agg = {}
    for k in rs[0]:
        vals = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and r.get(k) == r.get(k)]
        if vals and k not in ("seed", "n_frames", "n_eps"): agg[k] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else None, len(vals))
    agg["_meta"] = dict(seeds=[r["seed"] for r in rs], n_frames=rs[0]["n_frames"], n_eps=rs[0]["n_eps"], ckpt=rs[0]["ckpt"])
    rows[label] = agg

def fmt(v, prec=3):
    if v is None: return "n/a"
    m, sd, n = v
    return f"{m:.{prec}f}" + (f" ± {sd:.{prec}f} (n={n})" if sd is not None else "")

cols = [("psnr", "PSNR dB", 2), ("ssim", "SSIM", 3), ("lpips", "LPIPS", 3), ("psnr_c0", "PSNR chunk 0", 2), ("psnr_c5", "PSNR chunk 5", 2), ("psnr_c10", "PSNR chunk 10", 2), ("lpips_c10", "LPIPS chunk 10", 3),
        ("psnr_left", "PSNR 3rd-person", 2), ("psnr_right", "PSNR wrist", 2), ("grip_gen", "gripper cmd -> visible open/close (gen)", 3), ("grip_real", "same, real video", 3), ("grip_lag_gen", "gen lag frames", 1), ("grip_agree", "gripper frame agreement", 3)]
lines = ["| metric | " + " | ".join(rows) + " |", "| --- | " + " | ".join("---" for _ in rows) + " |"]
for k, name, prec in cols:
    lines.append(f"| {name} | " + " | ".join(fmt(rows[l].get(k), prec) for l in rows) + " |")
lines.append("| sampling seeds | " + " | ".join(str(rows[l]["_meta"]["seeds"]) for l in rows) + " |")
lines.append("| frames scored / episodes | " + " | ".join(f"{rows[l]['_meta']['n_frames']} / {rows[l]['_meta']['n_eps']}" for l in rows) + " |")
if a.delta:
    A, B = a.delta
    d = []
    for k, name, prec in cols:
        va, vb = rows[A].get(k), rows[B].get(k)
        if va and vb: d.append(f"{name}: {va[0] - vb[0]:+.{prec}f}")
    lines.append(f"\n**{A} vs {B}** (A - B): " + "; ".join(d))
out = "\n".join(lines); Path(a.out).write_text(out + "\n"); print(out)
