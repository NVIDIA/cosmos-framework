#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Drift curves (PSNR and LPIPS per 16-frame chunk) of several eval runs on one figure.
Usage: drift_compare.py --out fig.png --title "..." "label=eval_dir" ...   (run with the repo venv: uv run --no-sync python)"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="")
a = ap.parse_args()
colors = ["#9aa0a6", "#1f6feb", "#d1242f", "#1a7f37", "#8250df", "#bf8700"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, spec in enumerate(a.runs):
    name, d = spec.split("=", 1)
    bc = json.load(open(d + "/summary.json"))["by_chunk_index"]
    ks = sorted(int(k) for k in bc)
    x = [k * 16 for k in ks]
    ax[0].plot(x, [bc[str(k)]["psnr"] for k in ks], label=name, color=colors[i % len(colors)], lw=2)
    ax[1].plot(x, [bc[str(k)]["lpips"] for k in ks], label=name, color=colors[i % len(colors)], lw=2)
for axis, yl in zip(ax, ["PSNR (dB), mean over views and held-out episodes", "LPIPS (AlexNet), lower is better"]):
    axis.set_xlabel("frames generated since the conditioning frame (16 per chunk)")
    axis.set_ylabel(yl)
    axis.grid(alpha=0.3)
ax[0].legend(frameon=False, fontsize=8)
if a.title:
    fig.suptitle(a.title)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("wrote", a.out)
