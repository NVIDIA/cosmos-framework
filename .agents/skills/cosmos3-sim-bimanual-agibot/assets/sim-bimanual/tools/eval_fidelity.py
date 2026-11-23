#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""eval_fidelity.py - how well does a generated rollout match the real recording it was conditioned on?

Single mode
    eval_fidelity.py --real REAL.mp4 --gen GEN.mp4 --name NAME --out OUTDIR
                     [--frames 30 60 120 240 480 900] [--align auto|top|center|bottom|resize]
                     [--device cuda|cpu] [--stride 1] [--lpips-net alex] [--lpips-batch 16] [--workers N]
Batch mode
    eval_fidelity.py --batch ENTRIES.jsonl --out OUTDIR [same options]
    each JSONL line: {"name": ..., "real": ..., "gen": ..., "prompt": optional}

What it does
  1. Decodes both videos fully with PyAV into uint8 RGB arrays [T,H,W,3]; if the frame counts differ the
     first min(T_real, T_gen) frames are compared and a WARNING is printed.
  2. Aligns the real video to the generated dims: vertical crops anchored top/center/bottom (horizontal crop
     centred) when the real frame is at least as large, otherwise an antialiased Pillow bicubic resize.
     --align auto scores every candidate on frame 0 by PSNR and keeps the best.
  3. Per-frame PSNR (skimage, data_range=255), SSIM (skimage, channel_axis=-1, data_range=255) and LPIPS
     (lpips.LPIPS(net=alex), inputs in [-1,1], NCHW float32, batched, model built once, no_grad).
     PSNR/SSIM run in a CPU process pool while LPIPS runs on --device in the main process.
     Identical frame pairs have infinite PSNR; it is capped at PSNR_CAP_DB (100 dB) everywhere (CSV, JSON,
     chart) so that means stay finite, and the number of capped frames is recorded next to every mean.
  4. Writes NAME_per_frame.csv, NAME_summary.json and NAME_drift.png per episode; batch mode adds
     aggregate_per_frame.csv, aggregate_summary.json and aggregate_drift.png.

Failure policy
  single mode : any failure -> ERROR line, exit 1.
  batch mode  : every episode is isolated - decode/alignment errors, frames too small for the metrics, LPIPS
                runtime errors (including CUDA OOM) and any other unexpected exception skip that episode,
                the run continues, and the failure is recorded in aggregate_summary.json.failed_episodes.
                Exit 1 if at least one episode failed. Batch files are validated up front (exit 2): every entry
                must be a JSON object with non-empty string name/real/gen, and no two names may map to the same
                sanitised output stem (NAME_per_frame.csv etc.) because they would overwrite each other.

Exit codes: 0 ok, 1 an episode failed, 2 usage error.
Everything is deterministic: no random numbers are drawn anywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from multiprocessing import shared_memory

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TOOL_VERSION = "1.1"
DEFAULT_FRAMES = [30, 60, 120, 240, 480, 900]
METRICS = ("psnr", "ssim", "lpips")
WINDOWS = [("0-300", 0, 300), ("300-600", 300, 600), ("600-end", 600, None)]
REFERENCE_FRAME = 300
REFERENCE_LABEL = "300-frame recommended limit"
STD_DDOF = 1  # sample standard deviation across episodes
PSNR_CAP_DB = 100.0  # PSNR of identical frames (MSE 0) is +inf; it is reported as this finite value
SSIM_MIN_SIDE = 7  # skimage structural_similarity default win_size

# ----------------------------------------------------------------------------------------------------------
# chart style (mandatory rules)
# ----------------------------------------------------------------------------------------------------------
DPI = 150
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e5e4e0"
REF_LINE = "#a9a8a4"  # neutral gray for the reference line (never a series colour)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: blue, orange, aqua, yellow
# bounds: natural limits the y-range never exceeds by more than a small margin (None = unbounded);
# min_range: y-span used when a series is constant (e.g. identical videos), so the axis stays readable
METRIC_STYLE = {
    "psnr": {"label": "PSNR (dB)\nhigher is better", "fmt": "{:.1f}", "color": SERIES[0],
             "bounds": (None, None), "min_range": 1.0},
    "ssim": {"label": "SSIM (unitless, 0-1)\nhigher is better", "fmt": "{:.3f}", "color": SERIES[1],
             "bounds": (None, 1.0), "min_range": 0.02},
    "lpips": {"label": "LPIPS (unitless distance)\nlower is better", "fmt": "{:.3f}", "color": SERIES[2],
              "bounds": (0.0, None), "min_range": 0.02},
}
# value-label placements tried in this order: (dx pt, dy pt, ha, va) relative to the checkpoint dot
LABEL_PLACEMENTS = [
    (0, 9, "center", "bottom"),  # above
    (0, -9, "center", "top"),  # below
    (10, 7, "left", "bottom"),  # above-right
    (-10, 7, "right", "bottom"),  # above-left
    (10, -7, "left", "top"),  # below-right
    (-10, -7, "right", "top"),  # below-left
    (12, 0, "left", "center"),  # right
    (-12, 0, "right", "center"),  # left
]


def px(n: float) -> float:
    """Device pixels at DPI -> points (matplotlib line/marker units)."""
    return n * 72.0 / DPI


def log(msg: str) -> None:
    print(msg, flush=True)


def brief(exc: BaseException) -> str:
    """First line of an exception message (CUDA errors are paragraphs long)."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


class EvalError(RuntimeError):
    """A per-episode failure (decode, alignment, frames unusable by a metric, metric runtime error)."""


# ----------------------------------------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------------------------------------
def decode_video(path: str):
    """Decode every frame with PyAV -> (uint8 array [T,H,W,3], info dict). Raises EvalError on failure."""
    import av

    if not isinstance(path, str) or not path:
        raise EvalError(f"decode failed: video path must be a non-empty string, got {path!r}")
    if not os.path.isfile(path):
        raise EvalError(f"decode failed: file not found: {path}")
    frames = []
    info = {"path": os.path.abspath(path)}
    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise EvalError(f"decode failed: no video stream in {path}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            info["fps"] = float(rate) if rate else None
            info["declared_frames"] = int(stream.frames or 0)
            info["codec"] = stream.codec_context.name
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                if frames and arr.shape != frames[0].shape:
                    raise EvalError(
                        f"decode failed: frame size changes mid-stream in {path}: {frames[0].shape} -> {arr.shape}"
                    )
                frames.append(arr)
    except EvalError:
        raise
    except Exception as exc:  # av.error.*, OSError, ValueError ...
        raise EvalError(f"decode failed for {path}: {type(exc).__name__}: {exc}") from exc
    if not frames:
        raise EvalError(f"decode failed: no frames decoded from {path}")
    video = np.stack(frames)  # [T,H,W,3] uint8
    info.update({"frames": int(video.shape[0]), "height": int(video.shape[1]), "width": int(video.shape[2])})
    return video, info


def fps_text(v) -> str:
    return f"{v:g}" if v else "?"


# ----------------------------------------------------------------------------------------------------------
# alignment
# ----------------------------------------------------------------------------------------------------------
def alignment_candidates(real_hw, gen_hw):
    """Candidate alignments of real -> gen dims: {name: ("crop", (y0,y1,x0,x1)) | ("resize", (W,H))}."""
    hr, wr = real_hw
    hg, wg = gen_hw
    if hr >= hg and wr >= wg:
        x0 = (wr - wg) // 2
        yc = (hr - hg) // 2
        return {
            "top": ("crop", (0, hg, x0, x0 + wg)),
            "center": ("crop", (yc, yc + hg, x0, x0 + wg)),
            "bottom": ("crop", (hr - hg, hr, x0, x0 + wg)),
        }
    return {"resize": ("resize", (wg, hg))}


def apply_alignment(real: np.ndarray, spec) -> np.ndarray:
    kind, params = spec
    if kind == "crop":
        y0, y1, x0, x1 = params
        return np.ascontiguousarray(real[:, y0:y1, x0:x1])
    # Pillow's convolution resampling (BICUBIC) is antialiased: the filter support scales with the
    # reduction factor, so downscaling does not alias.
    from PIL import Image

    w, h = params
    out = np.empty((real.shape[0], h, w, 3), dtype=np.uint8)
    for i in range(real.shape[0]):
        out[i] = np.asarray(Image.fromarray(real[i]).resize((w, h), resample=Image.BICUBIC))
    return out


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """skimage PSNR (data_range 255); identical frames give +inf which is capped at PSNR_CAP_DB."""
    with np.errstate(divide="ignore"):
        return min(float(peak_signal_noise_ratio(a, b, data_range=255)), PSNR_CAP_DB)


def resolve_alignment(requested: str, real: np.ndarray, gen: np.ndarray):
    """Pick the alignment and return (aligned_real, alignment_info)."""
    real_hw = real.shape[1:3]
    gen_hw = gen.shape[1:3]
    candidates = alignment_candidates(real_hw, gen_hw)
    cand_psnr = {}
    for name, spec in candidates.items():
        cand_psnr[name] = psnr_uint8(apply_alignment(real[:1], spec)[0], gen[0])
    note = None
    if requested == "auto":
        # highest frame-0 PSNR wins; ties resolve in candidate order (top, center, bottom) - deterministic
        chosen = max(candidates, key=lambda k: (cand_psnr[k], -list(candidates).index(k)))
    elif requested == "resize":
        chosen = "resize"
        if "resize" not in candidates:
            candidates["resize"] = ("resize", (gen_hw[1], gen_hw[0]))
            cand_psnr["resize"] = psnr_uint8(apply_alignment(real[:1], candidates["resize"])[0], gen[0])
            note = "resize forced although the real frame is large enough to crop"
    else:
        if requested not in candidates:
            raise EvalError(
                f"alignment '{requested}' impossible: real {real_hw[1]}x{real_hw[0]} is smaller than "
                f"generated {gen_hw[1]}x{gen_hw[0]}; use --align resize or auto"
            )
        chosen = requested
    if tuple(real_hw) == tuple(gen_hw) and candidates[chosen][0] == "crop":
        note = "real and generated dims are equal; every crop is the identity"
    spec = candidates[chosen]
    info = {
        "requested": requested,
        "chosen": chosen,
        "method": spec[0],
        "real_dims_wh": [int(real_hw[1]), int(real_hw[0])],
        "gen_dims_wh": [int(gen_hw[1]), int(gen_hw[0])],
        "candidates_psnr_frame0": {k: round(v, 4) for k, v in cand_psnr.items()},
        "chosen_psnr_frame0": round(cand_psnr[chosen], 4),
        "psnr_cap_db": PSNR_CAP_DB,
    }
    if spec[0] == "crop":
        y0, y1, x0, x1 = spec[1]
        info["real_crop_box"] = {"y0": y0, "y1": y1, "x0": x0, "x1": x1}
    else:
        info["resize_to_wh"] = [int(spec[1][0]), int(spec[1][1])]
        info["resample"] = "Pillow BICUBIC (antialiased)"
    if note:
        info["note"] = note
    return apply_alignment(real, spec), info


# ----------------------------------------------------------------------------------------------------------
# PSNR / SSIM (CPU, optionally in a process pool over shared memory)
# ----------------------------------------------------------------------------------------------------------
def _psnr_ssim_pair(a: np.ndarray, b: np.ndarray):
    return psnr_uint8(a, b), float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def _open_shm(name: str):
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # Python >= 3.13
    except TypeError:  # pragma: no cover - older Pythons
        return shared_memory.SharedMemory(name=name)


def _worker_metrics(name_a: str, name_b: str, shape, idxs):
    """Pool worker: PSNR/SSIM for frame indices `idxs` read from two shared-memory blocks."""
    shm_a = _open_shm(name_a)
    shm_b = _open_shm(name_b)
    try:
        a = np.ndarray(shape, dtype=np.uint8, buffer=shm_a.buf)
        b = np.ndarray(shape, dtype=np.uint8, buffer=shm_b.buf)
        return [(int(i),) + _psnr_ssim_pair(a[i], b[i]) for i in idxs]
    finally:
        del a, b
        shm_a.close()
        shm_b.close()


def psnr_ssim_serial(a: np.ndarray, b: np.ndarray, idxs) -> tuple[np.ndarray, np.ndarray]:
    psnr = np.empty(len(idxs), dtype=np.float64)
    ssim = np.empty(len(idxs), dtype=np.float64)
    for k, i in enumerate(idxs):
        psnr[k], ssim[k] = _psnr_ssim_pair(a[i], b[i])
    return psnr, ssim


class PsnrSsimJob:
    """Asynchronous PSNR/SSIM over a process pool; frames are shared via POSIX shared memory (no pickling).

    Call result() to collect, or abort() if the episode fails meanwhile; both release the shared memory.
    """

    def __init__(self, executor, a: np.ndarray, b: np.ndarray, idxs, n_chunks: int):
        self._shm = []
        self.futures = []
        try:
            names = []
            for arr in (a, b):
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                self._shm.append(shm)
                np.ndarray(arr.shape, dtype=np.uint8, buffer=shm.buf)[:] = arr
                names.append(shm.name)
            idx_list = [int(i) for i in idxs]
            chunks = [idx_list[k::n_chunks] for k in range(n_chunks) if idx_list[k::n_chunks]]
            self.futures = [
                executor.submit(_worker_metrics, names[0], names[1], tuple(a.shape), chunk) for chunk in chunks
            ]
            self._pos = {int(i): k for k, i in enumerate(idxs)}
            self._n = len(idxs)
        except Exception:
            self.close()
            raise

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            psnr = np.full(self._n, np.nan)
            ssim = np.full(self._n, np.nan)
            for fut in self.futures:
                for i, p, s in fut.result():
                    psnr[self._pos[i]] = p
                    ssim[self._pos[i]] = s
            return psnr, ssim
        finally:
            self.close()

    def abort(self) -> None:
        """Cancel whatever is still queued and release the shared memory (safe to call more than once)."""
        for fut in self.futures:
            fut.cancel()
        self.close()

    def close(self) -> None:
        for shm in self._shm:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shm = []


# ----------------------------------------------------------------------------------------------------------
# LPIPS
# ----------------------------------------------------------------------------------------------------------
class LPIPSScorer:
    """lpips.LPIPS built once; scores uint8 RGB frame pairs in batches of `batch` frames on `device`."""

    def __init__(self, net: str, device: str | None, batch: int):
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lpips

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            log("WARNING: --device cuda requested but CUDA is not available; LPIPS falls back to CPU")
            device = "cpu"
        # determinism / device-independence of the convolutions
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torchvision 'pretrained' deprecation chatter
            model = lpips.LPIPS(net=net, verbose=False)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.torch = torch
        self.device = device
        self.net = net
        self.batch = max(1, int(batch))
        self.version = package_version("lpips")

    def _to_tensor(self, x: np.ndarray):
        t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        return t.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)  # uint8 [B,H,W,3] -> float32 NCHW in [-1,1]

    def score(self, a: np.ndarray, b: np.ndarray, idxs) -> np.ndarray:
        out = np.empty(len(idxs), dtype=np.float64)
        idx_arr = np.asarray(idxs, dtype=np.int64)
        with self.torch.no_grad():
            for s in range(0, len(idx_arr), self.batch):
                sel = idx_arr[s : s + self.batch]
                d = self.model(self._to_tensor(a[sel]), self._to_tensor(b[sel]))
                out[s : s + len(sel)] = d.flatten().float().cpu().numpy()
        return out


# ----------------------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------------------
def nanmean(x) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def nanstd(x) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=STD_DDOF)) if x.size > STD_DDOF else None


def window_stats(frames: np.ndarray, values: dict, start: int, end: int | None, n_frames: int):
    """Metric means over the evaluated frames in [start, end); end=None means open-ended, i.e. [start, T).

    `end` in the result is the nominal bound for fixed windows; for the open-ended window it is T, or None when
    the clip ends before `start` (the window is then empty: n == 0, metrics None).
    """
    stop = n_frames if end is None else end
    mask = (frames >= start) & (frames < stop)
    n = int(mask.sum())
    if end is None:
        end_out = int(n_frames) if n_frames > start else None
    else:
        end_out = int(end)
    out = {"start": int(start), "end": end_out, "n": n,
           "n_psnr_capped": int((values["psnr"][mask] >= PSNR_CAP_DB).sum()) if n else 0}
    for m in METRICS:
        out[m] = nanmean(values[m][mask]) if n else None
    return out


def clean_json(obj):
    """Recursively turn NaN/inf floats into None so the JSON is strict."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    return obj


def fmt(v, spec: str, width: int = 0) -> str:
    s = "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)
    return s.rjust(width) if width else s


def safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name).strip())
    return keep or "episode"


def window_label(key: str, w: dict) -> str:
    if key == "overall":
        return "overall"
    end = w.get("end")
    return f"mean[{w['start']},{end if end is not None else 'end'})"


# ----------------------------------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------------------------------
def plot_drift(out_path: str, title: str, subtitle: str, frames: np.ndarray, series: dict, checkpoints: dict,
               fps: float | None, last_frame: int, band: dict | None = None) -> None:
    """Three stacked panels (PSNR, SSIM, LPIPS vs frame index), one y-axis each, following the chart rules.

    series[m]      : values aligned with `frames` (NaN allowed)
    band[m]        : optional std aligned with `frames` -> shaded mean +- 1 std
    checkpoints[m] : list of (frame, value, label_text) drawn as dots with a value label
    The x-axis always reaches the 300-frame reference line, even for shorter clips.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker, transforms

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.8), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    data_max = max(int(frames.max()) if frames.size else 0, int(last_frame))
    xmax = max(data_max, REFERENCE_FRAME)  # the reference line is always drawn
    xpad = max(1.0, 0.02 * xmax)
    annotations = []  # per panel: list of annotation artists, in frame order
    dots = []  # per panel: the checkpoint marker artists, same order

    for ax, metric in zip(axes, METRICS):
        st = METRIC_STYLE[metric]
        colour = st["color"]
        y = np.asarray(series[metric], dtype=np.float64)

        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", axis="both", color=GRID, linewidth=px(1), linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(px(1))
        ax.tick_params(axis="both", which="both", length=0, labelsize=8, labelcolor=TEXT2)

        lo_vals, hi_vals = [y], [y]
        if band is not None and band.get(metric) is not None:
            sd = np.asarray(band[metric], dtype=np.float64)
            ok = np.isfinite(sd) & np.isfinite(y)
            if ok.any():
                ax.fill_between(frames, y - sd, y + sd, where=ok, color=colour, alpha=0.12, linewidth=0, zorder=2)
                lo_vals.append((y - sd)[ok])
                hi_vals.append((y + sd)[ok])
        ax.plot(frames, y, color=colour, linewidth=px(2), solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.axvline(REFERENCE_FRAME, color=REF_LINE, linewidth=px(1.5), linestyle="-", zorder=1)

        panel_ann, panel_dots = [], []
        for f, v, text in sorted(checkpoints.get(metric, []), key=lambda t: t[0]):
            if v is None or not math.isfinite(v):
                continue
            (dot,) = ax.plot([f], [v], linestyle="none", marker="o", markersize=px(12), markerfacecolor=colour,
                             markeredgecolor=SURFACE, markeredgewidth=px(2), zorder=5)
            panel_dots.append(dot)
            dx, dy, ha, va = LABEL_PLACEMENTS[0]
            ann = ax.annotate(text, xy=(f, v), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, fontsize=8, color=TEXT, zorder=6,
                              bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none",
                                    "alpha": 0.85})
            panel_ann.append(ann)
        annotations.append(panel_ann)
        dots.append(panel_dots)

        finite_lo = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in lo_vals]) if lo_vals else np.array([])
        finite_hi = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in hi_vals]) if hi_vals else np.array([])
        if finite_lo.size and finite_hi.size:
            lo, hi = float(finite_lo.min()), float(finite_hi.max())
            rng = hi - lo if hi > lo else max(abs(hi) * 0.1, st["min_range"])
            y0, y1 = lo - 0.24 * rng, hi + 0.24 * rng  # headroom for the value labels
            b0, b1 = st["bounds"]
            if b0 is not None:
                y0 = max(y0, b0 - 0.06 * rng)  # do not waste space beyond the metric's natural range
            if b1 is not None:
                y1 = min(y1, b1 + 0.06 * rng)
            ax.set_ylim(y0, y1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))  # never a '1e-5' offset label
        ax.set_ylabel(st["label"], color=TEXT, fontsize=9)

    axes[-1].set_xlim(-xpad, xmax + xpad)
    step = 150 if xmax > 600 else 60 if xmax > 300 else 30 if xmax > 120 else 10
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(step))
    axes[-1].set_xlabel(f"Frame index (frames, {fps:g} fps)" if fps else "Frame index (frames, fps unknown)",
                        color=TEXT, fontsize=9)

    # reference-line label sits just above the top panel at the top end of the line; it reads to the right
    # unless the line is near the right edge (short clips), then it reads to the left
    ref_label = REFERENCE_LABEL if data_max >= REFERENCE_FRAME else f"{REFERENCE_LABEL} (clip ends at frame {data_max})"
    tr = transforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    if REFERENCE_FRAME <= 0.6 * xmax:
        axes[0].text(REFERENCE_FRAME + 0.006 * xmax, 1.02, ref_label, transform=tr, ha="left", va="bottom",
                     fontsize=8, color=TEXT2)
    else:
        axes[0].text(REFERENCE_FRAME - 0.006 * xmax, 1.02, ref_label, transform=tr, ha="right", va="bottom",
                     fontsize=8, color=TEXT2)

    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=12, color=TEXT, weight="semibold")
    fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=8.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=1.2)

    # collision pass: labels are placed in frame order; each one tries LABEL_PLACEMENTS in turn and keeps the
    # first placement that (a) stays inside its panel - so it can never run into the tick labels, the
    # neighbouring panel or the figure edge -, (b) does not overlap an already placed label or any checkpoint
    # dot, (c) does not cover the series line and (d) does not straddle the reference line. If no placement is
    # clean the least-bad one wins (weights: outside 10, label/dot overlap 3, line cover 1, reference line 1).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, metric, panel_ann, panel_dots in zip(axes, METRICS, annotations, dots):
        ax_bbox = ax.get_window_extent(renderer)
        dot_bbs = [d.get_window_extent(renderer) for d in panel_dots]
        y = np.asarray(series[metric], dtype=np.float64)
        ok = np.isfinite(y)
        line_px = ax.transData.transform(np.column_stack([frames[ok], y[ok]])) if ok.any() else np.empty((0, 2))
        ref_x = ax.transData.transform((REFERENCE_FRAME, 0.0))[0]
        placed = []
        for k, ann in enumerate(panel_ann):
            obstacles = placed + [bb for j, bb in enumerate(dot_bbs) if j != k]

            def score(bb):
                s = 3 * sum(bb.overlaps(o) for o in obstacles)
                if bb.x0 < ax_bbox.x0 or bb.x1 > ax_bbox.x1 or bb.y0 < ax_bbox.y0 or bb.y1 > ax_bbox.y1:
                    s += 10
                if line_px.size:
                    inside = ((line_px[:, 0] >= bb.x0) & (line_px[:, 0] <= bb.x1)
                              & (line_px[:, 1] >= bb.y0) & (line_px[:, 1] <= bb.y1))
                    s += 1 if inside.any() else 0
                if bb.x0 <= ref_x <= bb.x1:
                    s += 1
                return s

            best = None
            for dx, dy, ha, va in LABEL_PLACEMENTS:
                ann.xyann = (dx, dy)
                ann.set_ha(ha)
                ann.set_va(va)
                bb = ann.get_window_extent(renderer).padded(2)
                s = score(bb)
                if best is None or s < best[0]:
                    best = (s, dx, dy, ha, va, bb)
                if s == 0:
                    break
            _, dx, dy, ha, va, bb = best
            ann.xyann = (dx, dy)
            ann.set_ha(ha)
            ann.set_va(va)
            placed.append(bb)

    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------------------
def episode_table(summary: dict) -> str:
    rows = [f"{'frame':>14} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8}"]
    for f, vals in summary["checkpoints"].items():
        rows.append(f"{f:>14} {fmt(vals['psnr'], '.2f', 10)} {fmt(vals['ssim'], '.4f', 8)} {fmt(vals['lpips'], '.4f', 8)}")
    for key, w in summary["windows"].items():
        row = f"{window_label(key, w):>14} {fmt(w['psnr'], '.2f', 10)} {fmt(w['ssim'], '.4f', 8)} {fmt(w['lpips'], '.4f', 8)}"
        if w["n"] == 0:
            row += f"  (empty: only {summary['frames_compared']} frames compared)"
        elif w.get("n_psnr_capped"):
            row += f"  ({w['n_psnr_capped']} identical frame pair(s), PSNR capped at {PSNR_CAP_DB:g} dB)"
        rows.append(row)
    return "\n".join(rows)


def aggregate_table(agg: dict) -> str:
    def ms(d, m, spec):
        if d[m]["mean"] is None:
            return fmt(None, "", 0)
        s = format(d[m]["mean"], spec)
        return s + (" +- " + format(d[m]["std"], spec) if d[m]["std"] is not None else "")

    rows = [f"{'frame':>14} {'PSNR (dB)':>16} {'SSIM':>18} {'LPIPS':>18} {'n':>3}"]
    for f, d in agg["checkpoints"].items():
        rows.append(f"{f:>14} {ms(d, 'psnr', '.2f'):>16} {ms(d, 'ssim', '.4f'):>18} {ms(d, 'lpips', '.4f'):>18} {d['n']:>3}")
    for key, w in agg["windows"].items():
        row = f"{window_label(key, w):>14} {ms(w, 'psnr', '.2f'):>16} {ms(w, 'ssim', '.4f'):>18} {ms(w, 'lpips', '.4f'):>18} {w['n']:>3}"
        if w["n"] == 0:
            row += "  (empty in every episode)"
        rows.append(row)
    return "\n".join(rows)


# ----------------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        os.makedirs(self.out, exist_ok=True)
        self.frames = sorted({int(f) for f in args.frames})
        self._scorer = None
        self._pool = None
        self._pool_broken = False
        self._stems: dict[str, str] = {}  # output stem -> episode name, guards against overwritten outputs

    # lazily built, shared across episodes
    @property
    def scorer(self) -> LPIPSScorer:
        if self._scorer is None:
            t0 = time.time()
            try:
                self._scorer = LPIPSScorer(self.args.lpips_net, self.args.device, self.args.lpips_batch)
            except Exception as exc:
                raise SystemExit(f"ERROR: could not build the LPIPS({self.args.lpips_net}) model: {type(exc).__name__}: {exc}")
            log(f"LPIPS({self.args.lpips_net}) ready on {self._scorer.device} in {time.time() - t0:.1f} s "
                f"(batch {self._scorer.batch})")
        return self._scorer

    def pool(self):
        if self._pool is None and not self._pool_broken and self.args.workers > 1:
            try:
                ctx = mp.get_context("forkserver")
            except ValueError:
                ctx = mp.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.args.workers, mp_context=ctx)
        return self._pool

    def _drop_pool(self, why: str) -> None:
        log(f"WARNING: process pool unavailable ({why}); computing PSNR/SSIM serially from now on")
        self._pool_broken = True
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def evaluate(self, name: str, real_path: str, gen_path: str, prompt: str | None = None) -> dict:
        t_start = time.time()
        stem = safe_name(name)
        if stem in self._stems and self._stems[stem] != name:
            raise EvalError(f"episode name '{name}' sanitises to '{stem}', already used by episode "
                            f"'{self._stems[stem]}'; its outputs would be overwritten - rename one of them")
        if stem in self._stems:
            raise EvalError(f"duplicate episode name '{name}'; its outputs would be overwritten")
        log(f"== {name}")
        if stem != name:
            log(f"note: output files use the sanitised stem '{stem}'")
        real, real_info = decode_video(real_path)
        gen, gen_info = decode_video(gen_path)
        t_decode = time.time() - t_start
        log(f"decoded real {real_info['width']}x{real_info['height']} x{real_info['frames']} @ {fps_text(real_info['fps'])} fps, "
            f"gen {gen_info['width']}x{gen_info['height']} x{gen_info['frames']} @ {fps_text(gen_info['fps'])} fps ({t_decode:.1f} s)")

        n_frames = min(real_info["frames"], gen_info["frames"])
        mismatch = real_info["frames"] != gen_info["frames"]
        if mismatch:
            log(f"WARNING: frame counts differ (real {real_info['frames']}, gen {gen_info['frames']}); "
                f"evaluating the first {n_frames} frames")
        real = real[:n_frames]
        gen = gen[:n_frames]
        hg, wg = gen.shape[1:3]
        if min(hg, wg) < SSIM_MIN_SIDE:
            raise EvalError(f"frames too small for the metrics: generated frame is {wg}x{hg}, SSIM needs at least "
                            f"{SSIM_MIN_SIDE}x{SSIM_MIN_SIDE} (LPIPS needs ~32x32)")

        t0 = time.time()
        real_al, align_info = resolve_alignment(self.args.align, real, gen)
        del real
        t_align = time.time() - t0
        cand = ", ".join(f"{k} {fmt(v, '.2f')}" for k, v in align_info["candidates_psnr_frame0"].items())
        log(f"alignment: {align_info['requested']} -> {align_info['chosen']} ({align_info['method']}); frame-0 PSNR dB: {cand}")

        stride = max(1, int(self.args.stride))
        idx = sorted(set(range(0, n_frames, stride)) | {f for f in self.frames if 0 <= f < n_frames})
        idx_arr = np.asarray(idx, dtype=np.int64)
        missing = [f for f in self.frames if f >= n_frames]
        if missing:
            log(f"WARNING: checkpoint frames beyond the compared range are skipped: {missing}")

        # PSNR/SSIM in the process pool while LPIPS runs on the device in this process. Whatever fails, the
        # shared-memory blocks of the in-flight PSNR/SSIM job are released before the error propagates.
        t0 = time.time()
        scorer = self.scorer  # build first so its time is not hidden inside the metric phase
        t_model = time.time() - t0
        t0 = time.time()
        job = None
        pool = self.pool()
        try:
            if pool is not None and len(idx) >= 32:
                try:
                    job = PsnrSsimJob(pool, real_al, gen, idx, n_chunks=self.args.workers * 4)
                except Exception as exc:
                    self._drop_pool(f"{type(exc).__name__}: {brief(exc)}")
                    job = None
            t1 = time.time()
            try:
                lp = scorer.score(real_al, gen, idx)
            except Exception as exc:  # frame too small for the network, CUDA OOM, ...
                raise EvalError(f"LPIPS({scorer.net}) failed on {wg}x{hg} frames ({scorer.device}): "
                                f"{type(exc).__name__}: {brief(exc)}") from exc
            t_lpips = time.time() - t1
            if job is not None:
                try:
                    ps, ss = job.result()
                except BrokenProcessPool as exc:
                    self._drop_pool(f"{type(exc).__name__}: {brief(exc)}")
                    job = None
                    ps, ss = psnr_ssim_serial(real_al, gen, idx)
                except Exception as exc:
                    raise EvalError(f"PSNR/SSIM failed on {wg}x{hg} frames: {type(exc).__name__}: {brief(exc)}") from exc
            else:
                try:
                    ps, ss = psnr_ssim_serial(real_al, gen, idx)
                except Exception as exc:
                    raise EvalError(f"PSNR/SSIM failed on {wg}x{hg} frames: {type(exc).__name__}: {brief(exc)}") from exc
        except BaseException:
            if job is not None:
                job.abort()
            raise
        t_metrics = time.time() - t0
        values = {"psnr": ps, "ssim": ss, "lpips": lp}
        n_capped = int((ps >= PSNR_CAP_DB).sum())
        if n_capped:
            log(f"note: {n_capped} identical frame pair(s) (MSE 0); their PSNR is reported as {PSNR_CAP_DB:g} dB")

        # summary
        checkpoints = {}
        pos = {int(f): k for k, f in enumerate(idx)}
        for f in self.frames:
            if f in pos:
                k = pos[f]
                checkpoints[str(f)] = {m: float(values[m][k]) for m in METRICS}
                checkpoints[str(f)]["psnr_capped"] = bool(ps[k] >= PSNR_CAP_DB)
        windows = {key: window_stats(idx_arr, values, s, e, n_frames) for key, s, e in WINDOWS}
        windows["overall"] = window_stats(idx_arr, values, 0, None, n_frames)

        t0 = time.time()
        paths = {
            "per_frame_csv": os.path.join(self.out, f"{stem}_per_frame.csv"),
            "summary_json": os.path.join(self.out, f"{stem}_summary.json"),
            "drift_png": os.path.join(self.out, f"{stem}_drift.png"),
        }
        with open(paths["per_frame_csv"], "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "psnr", "ssim", "lpips"])
            for k, f in enumerate(idx):
                w.writerow([int(f), f"{ps[k]:.4f}", f"{ss[k]:.6f}", f"{lp[k]:.6f}"])

        fps = gen_info["fps"] or real_info["fps"]
        cp_plot = {m: [(int(f), v[m], METRIC_STYLE[m]["fmt"].format(v[m])) for f, v in checkpoints.items()] for m in METRICS}
        subtitle = (f"{os.path.basename(real_path)} vs {os.path.basename(gen_path)}  |  {n_frames} frames compared"
                    f"{' (stride ' + str(stride) + ')' if stride > 1 else ''}  |  real aligned by {align_info['chosen']} "
                    f"{align_info['method']} {align_info['real_dims_wh'][0]}x{align_info['real_dims_wh'][1]} -> "
                    f"{align_info['gen_dims_wh'][0]}x{align_info['gen_dims_wh'][1]}  |  LPIPS {scorer.net} on {scorer.device}")
        plot_drift(paths["drift_png"], f"Fidelity drift: {name}", subtitle, idx_arr, values, cp_plot, fps, n_frames - 1)
        t_outputs = time.time() - t0
        t_total = time.time() - t_start

        summary = {
            "name": name,
            "prompt": prompt,
            "real": real_info,
            "gen": gen_info,
            "frame_count_mismatch": mismatch,
            "frames_compared": n_frames,
            "stride": stride,
            "frames_evaluated": len(idx),
            "requested_checkpoints": self.frames,
            "alignment": align_info,
            "psnr_cap_db": PSNR_CAP_DB,
            "n_psnr_capped": n_capped,
            "checkpoints": checkpoints,
            "checkpoints_missing": missing,
            "windows": windows,
            "lpips": {"net": scorer.net, "device": scorer.device, "batch": scorer.batch, "lpips_version": scorer.version},
            "workers": self.args.workers if job is not None else 1,
            "elapsed_seconds": {
                "decode": round(t_decode, 3), "align": round(t_align, 3), "lpips_model_build": round(t_model, 3),
                "psnr_ssim_lpips": round(t_metrics, 3), "lpips_only": round(t_lpips, 3),
                "outputs": round(t_outputs, 3), "total": round(t_total, 3),
            },
            "outputs": paths,
            "tool": tool_info(),
        }
        with open(paths["summary_json"], "w") as fh:
            json.dump(clean_json(summary), fh, indent=2, allow_nan=False)
        self._stems[stem] = name
        log(episode_table(summary))
        log(f"elapsed {t_total:.1f} s (decode {t_decode:.1f}, metrics {t_metrics:.1f} of which LPIPS {t_lpips:.1f}, outputs {t_outputs:.1f})")
        log(f"wrote {paths['per_frame_csv']}\n      {paths['summary_json']}\n      {paths['drift_png']}")
        del real_al, gen
        return summary, values, idx_arr


def package_version(name: str):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            return __import__(name).__version__
        except Exception:
            return None


def tool_info() -> dict:
    info = {"tool": "eval_fidelity.py", "version": TOOL_VERSION, "python": platform.python_version(),
            "numpy": np.__version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for dist in ("torch", "scikit-image", "lpips", "av", "pillow", "matplotlib"):
        info[dist] = package_version(dist)
    return info


def aggregate(evaluator: Evaluator, results: list, failures: list, t_start: float) -> dict:
    """Batch aggregate: per-frame mean/std CSV, summary JSON and the mean +- std drift chart."""
    out = evaluator.out
    names = [r[0]["name"] for r in results]
    frame_union = sorted({int(f) for _, _, idx in results for f in idx})
    n_union = len(frame_union)
    pos = {f: k for k, f in enumerate(frame_union)}
    stack = {m: np.full((len(results), n_union), np.nan) for m in METRICS}
    present = np.zeros((len(results), n_union), dtype=bool)
    for e, (_, values, idx) in enumerate(results):
        cols = [pos[int(f)] for f in idx]
        present[e, cols] = True
        for m in METRICS:
            stack[m][e, cols] = values[m]

    mean = {m: np.array([nanmean(stack[m][:, k]) if np.isfinite(stack[m][:, k]).any() else np.nan for k in range(n_union)]) for m in METRICS}
    std = {m: np.array([(nanstd(stack[m][:, k]) if nanstd(stack[m][:, k]) is not None else np.nan) for k in range(n_union)]) for m in METRICS}
    n_per_frame = present.sum(axis=0)  # episodes that evaluated the frame
    frames_arr = np.asarray(frame_union, dtype=np.int64)

    csv_path = os.path.join(out, "aggregate_per_frame.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "lpips_mean", "lpips_std"])
        for k, f in enumerate(frame_union):
            row = [f, int(n_per_frame[k])]
            for m, spec in (("psnr", ".4f"), ("ssim", ".6f"), ("lpips", ".6f")):
                row.append(format(mean[m][k], spec) if math.isfinite(mean[m][k]) else "")
                row.append(format(std[m][k], spec) if math.isfinite(std[m][k]) else "")
            w.writerow(row)

    def stats(vals):
        vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=STD_DDOF)) if vals.size > STD_DDOF else None,
                "min": float(vals.min()), "max": float(vals.max()), "n": int(vals.size)}

    checkpoints = {}
    for f in evaluator.frames:
        per_ep = [s["checkpoints"].get(str(f)) for s, _, _ in results]
        per_ep = [c for c in per_ep if c is not None]
        if not per_ep:
            continue
        checkpoints[str(f)] = {m: stats([c[m] for c in per_ep]) for m in METRICS}
        checkpoints[str(f)]["n"] = len(per_ep)
        checkpoints[str(f)]["n_psnr_capped"] = int(sum(1 for c in per_ep if c.get("psnr_capped")))
    windows = {}
    for key in [w[0] for w in WINDOWS] + ["overall"]:
        per_ep = [s["windows"][key] for s, _, _ in results]
        windows[key] = {m: stats([w[m] for w in per_ep]) for m in METRICS}
        windows[key]["start"] = per_ep[0]["start"]
        windows[key]["end"] = per_ep[0]["end"] if key != "600-end" and key != "overall" else None
        windows[key]["n"] = int(sum(1 for w in per_ep if w["n"] > 0))
        windows[key]["n_psnr_capped"] = int(sum(w.get("n_psnr_capped", 0) for w in per_ep))
    per_episode = {s["name"]: {**{m: s["windows"]["overall"][m] for m in METRICS},
                               "frames_compared": s["frames_compared"], "alignment": s["alignment"]["chosen"],
                               "frame0_psnr": s["alignment"]["chosen_psnr_frame0"],
                               "n_psnr_capped": s["n_psnr_capped"]} for s, _, _ in results}

    agg = {
        "n_episodes": len(results),
        "episodes": names,
        "failed_episodes": failures,
        "std_ddof": STD_DDOF,
        "psnr_cap_db": PSNR_CAP_DB,
        "settings": {"frames": evaluator.frames, "align": evaluator.args.align, "stride": evaluator.args.stride,
                     "lpips_net": evaluator.args.lpips_net, "lpips_batch": evaluator.args.lpips_batch,
                     "device": results[0][0]["lpips"]["device"], "workers": evaluator.args.workers},
        "checkpoints": checkpoints,
        "windows": windows,
        "per_episode_overall": per_episode,
        "elapsed_seconds": round(time.time() - t_start, 3),
        "outputs": {"aggregate_per_frame_csv": csv_path,
                    "aggregate_summary_json": os.path.join(out, "aggregate_summary.json"),
                    "aggregate_drift_png": os.path.join(out, "aggregate_drift.png")},
        "tool": tool_info(),
    }
    with open(agg["outputs"]["aggregate_summary_json"], "w") as fh:
        json.dump(clean_json(agg), fh, indent=2, allow_nan=False)

    cp_plot = {}
    for m in METRICS:
        pts = []
        for f, d in checkpoints.items():
            mu, sd = d[m]["mean"], d[m]["std"]
            if mu is None:
                continue
            text = METRIC_STYLE[m]["fmt"].format(mu) + (" ± " + METRIC_STYLE[m]["fmt"].format(sd) if sd is not None else "")
            pts.append((int(f), mu, text))
        cp_plot[m] = pts
    fps = results[0][0]["gen"]["fps"]
    ep_text = ", ".join(names) if len(names) <= 3 else f"{len(names)} episodes"
    subtitle = (f"Line: mean across {len(names)} episodes ({ep_text})  |  band: ±1 std (ddof={STD_DDOF})  |  "
                f"{n_union} frames  |  LPIPS {evaluator.args.lpips_net}")
    plot_drift(agg["outputs"]["aggregate_drift_png"], f"Fidelity drift: aggregate of {len(names)} episodes", subtitle,
               frames_arr, mean, cp_plot, fps, int(frames_arr.max()), band=std)
    log(f"\n== aggregate over {len(names)} episode(s) (mean +- std, ddof={STD_DDOF})")
    log(aggregate_table(agg))
    log(f"wrote {csv_path}\n      {agg['outputs']['aggregate_summary_json']}\n      {agg['outputs']['aggregate_drift_png']}")
    return agg


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def usage_error(msg: str) -> "SystemExit":
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    return SystemExit(2)


def read_batch(path: str) -> list:
    """Parse and validate the JSONL batch file; every problem is a usage error (exit 2) with file:line."""
    entries = []
    try:
        with open(path) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise usage_error(f"{path}:{ln}: invalid JSON ({exc})")
                if not isinstance(obj, dict):
                    raise usage_error(f"{path}:{ln}: entry must be a JSON object, got {type(obj).__name__}")
                missing = [k for k in ("name", "real", "gen") if k not in obj]
                if missing:
                    raise usage_error(f"{path}:{ln}: missing keys {missing}")
                bad = [k for k in ("name", "real", "gen") if not isinstance(obj[k], str) or not obj[k].strip()]
                if bad:
                    raise usage_error(f"{path}:{ln}: {bad} must be non-empty strings")
                if obj.get("prompt") is not None and not isinstance(obj["prompt"], str):
                    raise usage_error(f"{path}:{ln}: 'prompt' must be a string when present")
                entries.append({"name": obj["name"], "real": obj["real"], "gen": obj["gen"],
                                "prompt": obj.get("prompt"), "line": ln})
    except OSError as exc:
        raise usage_error(f"cannot read batch file {path}: {exc}")
    if not entries:
        raise usage_error(f"no entries in {path}")
    # names must map to distinct output stems, otherwise NAME_per_frame.csv / _summary.json / _drift.png
    # of one episode would silently overwrite another's
    stems: dict[str, dict] = {}
    for e in entries:
        stem = safe_name(e["name"])
        other = stems.get(stem)
        if other is not None:
            what = ("duplicate episode name" if other["name"] == e["name"]
                    else "episode names collide after sanitising")
            raise usage_error(f"{path}:{e['line']}: {what}: '{e['name']}' and '{other['name']}' (line {other['line']}) "
                              f"would both write {stem}_per_frame.csv / {stem}_summary.json / {stem}_drift.png; "
                              f"rename one of them")
        stems[stem] = e
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--real", help="real recording (mp4)")
    p.add_argument("--gen", help="generated rollout (mp4)")
    p.add_argument("--name", help="episode name used for output files")
    p.add_argument("--batch", help="JSONL with one {name, real, gen, prompt?} object per line")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES, help="checkpoint frame indices")
    p.add_argument("--align", choices=["auto", "top", "center", "bottom", "resize"], default="auto")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None, help="LPIPS device (default: cuda if available)")
    p.add_argument("--stride", type=int, default=1, help="evaluate every k-th frame (checkpoints always included)")
    p.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    p.add_argument("--lpips-batch", type=int, default=16, help="frames per LPIPS batch (default 16)")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="CPU processes for PSNR/SSIM (1 = serial in-process)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch and (args.real or args.gen or args.name):
        raise usage_error("use either --batch or --real/--gen/--name, not both")
    if not args.batch and not (args.real and args.gen and args.name):
        raise usage_error("single mode needs --real, --gen and --name (or use --batch)")
    if not args.batch and not args.name.strip():
        raise usage_error("--name must not be empty")
    if args.stride < 1 or args.lpips_batch < 1 or args.workers < 1:
        raise usage_error("--stride, --lpips-batch and --workers must be >= 1")
    if any(f < 0 for f in args.frames):
        raise usage_error("--frames must be non-negative")

    t_start = time.time()
    evaluator = Evaluator(args)
    try:
        if not args.batch:
            try:
                evaluator.evaluate(args.name, args.real, args.gen)
            except EvalError as exc:
                log(f"ERROR: {exc}")
                return 1
            except Exception as exc:
                log(f"ERROR: unexpected {type(exc).__name__}: {brief(exc)}")
                traceback.print_exc()
                return 1
            return 0

        entries = read_batch(args.batch)
        results, failures = [], []
        for k, e in enumerate(entries, 1):
            log(f"\n[{k}/{len(entries)}]")
            try:
                results.append(evaluator.evaluate(e["name"], e["real"], e["gen"], e["prompt"]))
            except EvalError as exc:
                log(f"ERROR: episode {e['name']}: {exc}")
                failures.append({"name": e["name"], "line": e["line"], "error": str(exc), "kind": "EvalError"})
            except Exception as exc:  # anything else: the episode is skipped, the batch goes on
                log(f"ERROR: episode {e['name']}: unexpected {type(exc).__name__}: {brief(exc)}")
                traceback.print_exc()
                failures.append({"name": e["name"], "line": e["line"],
                                 "error": f"{type(exc).__name__}: {brief(exc)}", "kind": "unexpected"})
        if results:
            aggregate(evaluator, results, failures, t_start)
        else:
            log("\nno episode succeeded; aggregate outputs not written")
        if failures:
            log(f"\n{len(failures)} episode(s) failed: {[f['name'] for f in failures]}")
            return 1
        log(f"batch finished in {time.time() - t_start:.1f} s")
        return 0
    finally:
        evaluator.close()


if __name__ == "__main__":
    sys.exit(main())
