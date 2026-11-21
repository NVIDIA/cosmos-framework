# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Score base vs post-trained generations against the ground truth for transfer_multicontrol_lora_nano.

    python examples/transfer_multicontrol_lora_nano/score.py <dataset_dir> <eval_dir> [--models base lora]

Reads ``<eval_dir>/specs/<clip>__<mode>.json`` (from make_specs.py) and ``<eval_dir>/<model>/<clip>__<mode>/vision.mp4``
(from cosmos_framework.scripts.inference); the ground truth is ``<dataset_dir>/clips/<clip>/rgb.mp4``. Generated and
real frames are compared 1:1 over the frames both have:

* PSNR and SSIM on 640x360 grey frames, LPIPS (AlexNet, inputs in [-1, 1]) on 448x256 colour frames, every 2nd frame.
  The controls pin the scene layout, so these are meaningful for the control modes; for seed image -> video they
  mostly measure drift from the true path.
* Saturation and brightness gap: |mean HSV S (or V) of the generation - of the ground truth|, every 5th frame, 0-255.
* Motion ratio: mean absolute frame-to-frame difference, generation / ground truth.

Writes ``<eval_dir>/metrics.csv`` (one row per clip x mode x model) and ``<eval_dir>/metrics.json`` (mean and std
over clips per crop, mode and model, plus how many clips each metric improved on). Needs ``lpips`` and
``scikit-image``: ``uv pip install lpips scikit-image``.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "sat_gap": False, "bright_gap": False}


def load(p: Path) -> list[np.ndarray]:
    c, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = c.read()
        if not ok:
            return fs
        fs.append(f)


def grey(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def to_lpips(fs: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([cv2.cvtColor(cv2.resize(f, (448, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for f in fs])
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    return 10 * np.log10(255**2 / max(np.mean((x.astype(np.float64) - y) ** 2), 1e-10))


def appearance(fs: list[np.ndarray]) -> tuple[float, float, float]:
    a = np.array(fs)
    diff = np.mean([np.abs(a[i + 1].astype(np.float32) - a[i]).mean() for i in range(len(a) - 1)])
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in a[::5]]
    return diff, np.mean([h[..., 1].mean() for h in hsv]), np.mean([h[..., 2].mean() for h in hsv])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("eval_dir")
    ap.add_argument("--models", nargs="+", default=["base", "lora"])
    ap.add_argument("--every", type=int, default=2, help="score every k-th frame for PSNR / SSIM / LPIPS")
    a = ap.parse_args()
    data, ev = Path(a.dataset_dir), Path(a.eval_dir)
    meta = {r["id"]: r for r in csv.DictReader(open(data / "metadata.csv"))}
    net = lpips.LPIPS(net="alex", verbose=False).eval()

    rows = []
    for spec_file in sorted((ev / "specs").glob("*.json")):
        name = spec_file.stem
        clip, mode = name.split("__")
        gen = {m: load(ev / m / name / "vision.mp4") for m in a.models}
        gt = load(data / "clips" / clip / "rgb.mp4")
        n = min(len(gt), *(len(g) for g in gen.values()))
        assert n > 1, f"no frames to compare for {name}"
        idx = list(range(0, n, a.every))
        gt_grey, gt_t = [grey(gt[i]) for i in idx], to_lpips([gt[i] for i in idx])
        gt_motion, gt_sat, gt_bright = appearance(gt[:n])
        for m, g in gen.items():
            x_grey = [grey(g[i]) for i in idx]
            with torch.no_grad():
                lp = torch.cat([net(to_lpips([g[i] for i in idx[k : k + 16]]), gt_t[k : k + 16]).flatten() for k in range(0, len(idx), 16)])
            motion, sat, bright = appearance(g[:n])
            rows.append(
                dict(
                    clip=clip, crop=meta[clip]["crop"], motion_class=meta[clip]["motion_class"], mode=mode, model=m,
                    frames=n,
                    psnr=float(np.mean([psnr(x, y) for x, y in zip(x_grey, gt_grey)])),
                    ssim=float(np.mean([structural_similarity(x, y, data_range=255) for x, y in zip(x_grey, gt_grey)])),
                    lpips=lp.mean().item(),
                    sat_gap=float(abs(sat - gt_sat)), bright_gap=float(abs(bright - gt_bright)),
                    motion_ratio=float(motion / gt_motion),
                )
            )
        print("scored", name, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ev / "metrics.csv", index=False)
    summary = {}
    for (crop, mode, model), g in df.groupby(["crop", "mode", "model"]):
        s = {k: {"mean": round(float(g[k].mean()), 4), "std": round(float(g[k].std(ddof=1)), 4) if len(g) > 1 else None}
             for k in (*HIGHER_IS_BETTER, "motion_ratio")}
        summary.setdefault(crop, {}).setdefault(mode, {})[model] = s | {"clips": len(g)}
    if {"base", "lora"} <= set(a.models):
        w = df.pivot_table(index=["crop", "mode", "clip"], columns="model", values=list(HIGHER_IS_BETTER))
        for (crop, mode), g in w.groupby(level=[0, 1]):
            summary[crop][mode]["lora_better_on"] = {
                k: f"{int(((g[k]['lora'] > g[k]['base']) if hi else (g[k]['lora'] < g[k]['base'])).sum())}/{len(g)}"
                for k, hi in HIGHER_IS_BETTER.items()
            }
    json.dump({"settings": {"every": a.every, "models": a.models}, "summary": summary}, open(ev / "metrics.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(df.groupby(["crop", "mode", "model"])[[*HIGHER_IS_BETTER, "motion_ratio"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
