# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Score generated videos against ground truth (PSNR + SSIM), per conditioning mode.

CPU-only "vision" evaluation: pair each predicted ``vision.mp4`` with its ground-truth
video, compute per-clip PSNR and SSIM, and aggregate the means **per conditioning mode**
(``t2v`` / ``i2v`` / ``v2v``). This is a dependency-light port of imaginaire4's
``cosmos3.scripts.eval`` *vision* path (which computes PSNR only); SSIM is added here with
a small Gaussian-window implementation so the released example needs no extra packages —
it reuses the repo's own :func:`cosmos_framework.inference.vision.read_media_frames`.

Predictions must already exist on disk; generate them with
``cosmos_framework.scripts.inference`` (see ``docs/training.md`` / ``docs/dataset_jsonl.md``).
Inference writes ``<output_dir>/<name>/vision.mp4`` and the example's prompt ``name`` is
mode-prefixed (``t2v/<episode>`` …), so the predictions tree is ``<root>/<mode>/<episode>/
vision.mp4`` — the mode becomes the aggregation group and ``<episode>`` the match key paired
to ``<gt_dir>/<episode><gt_extension>``.

Usage
-----
    # score one run
    python -m cosmos_framework.scripts.eval \
        --gt-dir "$DS/val/videos" \
        --predictions-dir outputs/ab/json \
        --output-dir outputs/ab/eval_json

    # score a second run and compare it against the first (prints a per-mode delta table)
    python -m cosmos_framework.scripts.eval \
        --gt-dir "$DS/val/videos" \
        --predictions-dir outputs/ab/dense \
        --output-dir outputs/ab/eval_dense \
        --compare-baseline outputs/ab/eval_json/metrics_aggregate.json
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn.functional as F
import tyro

from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.utils import log

# data range for 8-bit video frames; PSNR/SSIM constants below assume uint8 [0, 255].
_DATA_RANGE = 255.0


# ---------------------------------------------------------------------------
# Metrics — operate on uint8 (C, T, H, W) tensors (the layout read_media_frames returns).
# ---------------------------------------------------------------------------


def compute_psnr(gt_cthw: torch.Tensor, pred_cthw: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB) over the whole clip. Identical clips → 100.0."""
    mse = torch.mean((gt_cthw.to(torch.float64) - pred_cthw.to(torch.float64)) ** 2).item()
    if mse == 0.0:
        return 100.0
    return 10.0 * math.log10(_DATA_RANGE**2 / mse)


def _gaussian_window(window_size: int, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]  # (window_size, window_size)


def compute_ssim(gt_cthw: torch.Tensor, pred_cthw: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """Mean structural similarity, computed per (frame, channel) with a Gaussian window.

    Standard windowed SSIM (Wang et al. 2004) with ``data_range=255`` and the conventional
    ``k1=0.01``, ``k2=0.03``. The Gaussian filter uses ``same`` padding and the SSIM map is
    averaged over all positions/frames/channels — sufficient for a relative A/B comparison.
    """
    channels, _, _, _ = gt_cthw.shape
    x = gt_cthw.permute(1, 0, 2, 3).to(torch.float32)  # (T, C, H, W)
    y = pred_cthw.permute(1, 0, 2, 3).to(torch.float32)
    win = _gaussian_window(window_size, sigma, x.dtype).expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def _filter(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, win, padding=pad, groups=channels)

    mu_x, mu_y = _filter(x), _filter(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _filter(x * x) - mu_x2
    sigma_y2 = _filter(y * y) - mu_y2
    sigma_xy = _filter(x * y) - mu_xy
    c1 = (0.01 * _DATA_RANGE) ** 2
    c2 = (0.03 * _DATA_RANGE) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean().item()


def compute_video_metrics(gt_cthw_uint8: torch.Tensor, pred_path: Path) -> dict[str, float]:
    """Read ``pred_path``, align it to GT, and return ``{"psnr", "ssim"}``.

    Alignment mirrors the imaginaire4 reference: read at most ``T_gt + 1`` frames (so an
    over-long prediction surfaces as a mismatch rather than being silently truncated),
    top-left-crop the prediction to GT's spatial dims, hard-error on any remaining spatial
    mismatch, and trim both to ``min(T_gt, T_pred)`` for the (expected, ``4k+1`` vs raw)
    temporal delta.
    """
    pred, _ = read_media_frames(pred_path, max_frames=gt_cthw_uint8.shape[1] + 1)
    pred = pred[..., : gt_cthw_uint8.shape[-2], : gt_cthw_uint8.shape[-1]]
    gt = gt_cthw_uint8
    if pred.shape != gt.shape:
        if pred.shape[-2:] != gt.shape[-2:]:
            raise ValueError(f"video spatial mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")
        min_t = min(gt.shape[1], pred.shape[1])
        gt, pred = gt[:, :min_t], pred[:, :min_t]
    return {"psnr": compute_psnr(gt, pred), "ssim": compute_ssim(gt, pred)}


# ---------------------------------------------------------------------------
# Pairing + aggregation (ported from imaginaire4 eval_utils).
# ---------------------------------------------------------------------------


def derive_match_key_and_group(pred_path: Path, predictions_dir: Path) -> tuple[str, str]:
    """Path → ``(match_key, group)`` used to pair a prediction with its GT.

    For ``inference``-style outputs (basename ``vision.*``), ``match_key`` is the parent
    directory name and ``group`` is the path between *predictions_dir* and that directory.
    Otherwise ``match_key`` is the filename stem.

    Examples (``predictions_dir=/root``)::

        /root/t2v/episode_0/vision.mp4 → ("episode_0", "t2v")
        /root/sub/foo.mp4              → ("foo", "sub")
    """
    pred_path = pred_path.resolve()
    predictions_dir = predictions_dir.resolve()
    if not pred_path.is_relative_to(predictions_dir):
        raise ValueError(f"pred_path {pred_path} is not under predictions_dir {predictions_dir}")
    parts = pred_path.relative_to(predictions_dir).parts
    if pred_path.name.startswith("vision."):
        if len(parts) < 2:
            raise ValueError(f"expected <group>/<key>/vision.* under predictions_dir, got rel={'/'.join(parts)}")
        return parts[-2], "/".join(parts[:-2])
    return pred_path.stem, "/".join(parts[:-1])


def aggregate_metrics(output_dir: Path) -> dict[str, Any]:
    """Walk ``output_dir`` for per-sample ``metrics.json`` files → per-mode/metric summary.

    Each scalar metric is summarised as ``{mean, count}`` within its mode.
    """
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in output_dir.rglob("metrics.json"):
        m = json.loads(f.read_text())
        mode = m.pop("mode", None)
        m.pop("name", None)
        if mode is None:
            continue
        for k, v in m.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    totals[mode][f"{k}/{sub_k}"].append(float(sub_v))
            else:
                totals[mode][k].append(float(v))
    return {
        mode: {metric: {"mean": float(sum(vals) / len(vals)), "count": len(vals)} for metric, vals in metrics.items()}
        for mode, metrics in totals.items()
    }


# ---------------------------------------------------------------------------
# Optional dense-vs-JSON comparison.
# ---------------------------------------------------------------------------


def format_comparison(
    baseline: dict[str, Any], current: dict[str, Any], baseline_label: str, current_label: str
) -> str:
    """Build a markdown per-mode/metric table: baseline vs current means and Δ(current−baseline)."""
    lines = [
        f"# Eval comparison — `{current_label}` vs baseline `{baseline_label}`",
        "",
        f"| mode | metric | {baseline_label} | {current_label} | Δ ({current_label}−{baseline_label}) | higher |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in sorted(set(baseline) | set(current)):
        b_metrics, c_metrics = baseline.get(mode, {}), current.get(mode, {})
        for metric in sorted(set(b_metrics) | set(c_metrics)):
            b = b_metrics.get(metric, {}).get("mean")
            c = c_metrics.get(metric, {}).get("mean")
            if b is None or c is None:
                lines.append(
                    f"| {mode} | {metric} | {b if b is not None else '—'} | {c if c is not None else '—'} | — | — |"
                )
                continue
            delta = c - b
            higher = current_label if delta > 0 else (baseline_label if delta < 0 else "tie")
            lines.append(f"| {mode} | {metric} | {b:.4f} | {c:.4f} | {delta:+.4f} | {higher} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main(
    gt_dir: Annotated[Path, tyro.conf.arg(help="Directory of ground-truth videos (<match_key><gt-extension>).")],
    predictions_dir: Annotated[Path, tyro.conf.arg(help="Root containing pre-generated prediction videos.")],
    output_dir: Annotated[Path, tyro.conf.arg(help="Where per-sample metrics.json + metrics_aggregate.json land.")],
    predictions_glob: str = "**/vision.mp4",
    gt_extension: str = ".mp4",
    compare_baseline: Annotated[
        Path | None, tyro.conf.arg(help="A prior run's metrics_aggregate.json to compare this run against.")
    ] = None,
) -> None:
    if not gt_dir.exists():
        print(f"ERROR: gt_dir does not exist: {gt_dir}", file=sys.stderr)
        sys.exit(1)
    if not predictions_dir.exists():
        print(f"ERROR: predictions_dir does not exist: {predictions_dir}", file=sys.stderr)
        sys.exit(1)

    pred_paths = sorted(predictions_dir.glob(predictions_glob))
    log.info(f"Found {len(pred_paths)} prediction(s) under {predictions_dir} / {predictions_glob!r}")
    if not pred_paths:
        print(f"ERROR: no predictions matched {predictions_glob!r} under {predictions_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored = skipped_missing_gt = errored = 0
    for i, pred_path in enumerate(pred_paths):
        match_key, group = derive_match_key_and_group(pred_path, predictions_dir)
        bucket = group or "default"
        gt_path = gt_dir / f"{match_key}{gt_extension}"
        if not gt_path.exists():
            log.warning(f"[{i + 1}/{len(pred_paths)}] missing GT for {match_key!r} at {gt_path}; skipping")
            skipped_missing_gt += 1
            continue
        gt_video, _ = read_media_frames(gt_path, max_frames=10**9)
        try:
            metrics = compute_video_metrics(gt_video, pred_path)
        except ValueError as e:
            log.warning(f"[{i + 1}/{len(pred_paths)}] skip {pred_path} ({e})")
            errored += 1
            continue
        sample_dir = output_dir / bucket / match_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "metrics.json").write_text(
            json.dumps({"mode": bucket, "name": match_key, **metrics}, indent=2, sort_keys=True)
        )
        log.info(f"[{i + 1}/{len(pred_paths)}] {bucket}/{match_key}: {metrics}")
        scored += 1

    aggregate = aggregate_metrics(output_dir)
    (output_dir / "metrics_aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    log.success(
        f"scored {scored}/{len(pred_paths)} (skipped {skipped_missing_gt} missing-GT, {errored} mismatch) "
        f"→ {output_dir / 'metrics_aggregate.json'}"
    )
    for mode, metrics in sorted(aggregate.items()):
        log.info(f"  {mode}: " + ", ".join(f"{k}={v['mean']:.4f} (n={v['count']})" for k, v in sorted(metrics.items())))

    if compare_baseline is not None:
        baseline = json.loads(compare_baseline.read_text())
        md = format_comparison(
            baseline,
            aggregate,
            baseline_label=compare_baseline.parent.name or "baseline",
            current_label=predictions_dir.name or "current",
        )
        (output_dir / "comparison.md").write_text(md)
        (output_dir / "comparison.json").write_text(
            json.dumps({"baseline": baseline, "current": aggregate}, indent=2, sort_keys=True)
        )
        log.success(f"Wrote comparison → {output_dir / 'comparison.md'}")
        print("\n" + md)


if __name__ == "__main__":
    tyro.cli(main)
