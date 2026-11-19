#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build one self-contained HTML review page showing real | generated rollouts side by side.

Usage
  make_review_page.py --entries ENTRIES.jsonl --out OUT_DIR/index.html
                      [--frames 30 60 120 240 480 900] [--aggregate aggregate_summary.json]
                      [--aggregate-plot aggregate_drift.png] [--title TEXT] [--intro TEXT]
                      [--generated-at TEXT]

ENTRIES.jsonl (one JSON object per line; blank lines and lines starting with '#' are ignored)
  {"name": str, "prompt": str, "real": path.mp4, "gen": path.mp4,
   "summary": path to NAME_summary.json written by eval_fidelity.py (optional),
   "extra": [{"label": str, "video": path.mp4}, ...] (optional; a single object or a bare path string is
            also accepted; list items may be bare path strings),
   "note": str (optional)}

Behaviour
  * Every referenced video is COPIED (never moved) to OUT_DIR/assets/NAME/<role>__<basename> and every
    src/href in the page is RELATIVE to OUT_DIR, so the directory can be moved or served as-is.
  * Per entry: heading + prompt, a wrapping row of <video controls muted loop preload="metadata"> (real |
    generated | extras, ~480 px wide, aspect preserved), one play-all / pause-all / restart control that
    drives that entry's videos together, a frame strip (real thumbnail above generated, extras below) at the
    requested frame indices, decoded with PyAV and written by Pillow as JPEG q85, width 320, into
    OUT_DIR/assets/NAME/frames/, and a compact table of window means from the summary.
  * Page top: title, intro, generated-at stamp, and (with --aggregate) a checkpoint table of mean +- std (n)
    plus the aggregate drift plot copied into OUT_DIR/assets/.
  * Robust: a missing video renders a visible 'missing: <path>' placeholder and processing continues
    (WARNING on stderr, exit code 0); a missing/unreadable summary just omits the metrics; an entry that
    fails for any other reason gets a placeholder section (WARNING) instead of aborting the page.

Summary schema (read defensively)
  checkpoints: {"30": {"psnr": .., "ssim": .., "lpips": ..}, ...}  or a list of {"frame": 30, ...}
               (label key: frame / frame_idx / frame_index / frame_id / checkpoint / index / idx / step / t)
               or parallel arrays {"frame": [...], "psnr": [...], ...}
  windows:     {"0-300": {"psnr": .., "ssim": .., "lpips": ..}, ...} or a list with window/label/name/range/
               frames or start+end / from+to
  Node aliases are only accepted when the value is a dict or list (a top-level "frames": 901 clip length is
  never taken for the checkpoint table). A row without a recognised label is NEVER shown under a guessed
  label (a list position is not a frame index): it is dropped with a WARNING and a note on the page.
  Metric values may be scalars or {"mean": ..}. Aggregate files use the same layouts with per-metric
  {"mean", "std", "n"} dicts, [mean, std, n] lists, or flat "<metric>_mean" / "<metric>_std" keys; n may also
  live at the checkpoint level or at the top level ("n", "n_episodes", "count", len("episodes")).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import av
from PIL import Image

DEFAULT_FRAMES = [30, 60, 120, 240, 480, 900]
THUMB_WIDTH = 320
JPEG_QUALITY = 85
METRICS = ("psnr", "ssim", "lpips")
METRIC_LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}
METRIC_FMT = {"psnr": "{:.1f}", "ssim": "{:.3f}", "lpips": "{:.3f}"}
DEFAULT_TITLE = "Rollout review: real vs generated"
DEFAULT_INTRO = (
    "Each entry shows the real recording and the generated rollout (same first frame and action sequence) "
    "side by side, a strip of frames at fixed checkpoints, and per-window fidelity metrics where a summary "
    "is available. Videos are muted and loop; use the per-entry controls to drive them together."
)

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- small helpers
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: Any, fallback: str) -> str:
    s = _SAFE.sub("_", str(name or "")).strip("._")
    return s or fallback


def rel(path: Path, out_dir: Path) -> str:
    """Relative POSIX path from OUT_DIR to an asset (never absolute)."""
    return Path(os.path.relpath(path, out_dir)).as_posix()


def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _num(v: Any) -> float | None:
    """float from a scalar, {"mean": ..}, [mean, ...] or numeric string; None otherwise."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    if isinstance(v, dict):
        lv = _lower_keys(v)
        for k in ("mean", "value", "avg", "median"):
            if k in lv:
                return _num(lv[k])
        return None
    if isinstance(v, (list, tuple)):
        return _num(v[0]) if v else None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return None if f is None else int(round(f))


def _lower_keys(d: dict) -> dict:
    return {str(k).lower(): v for k, v in d.items()}


def _frame_int(k: Any) -> int | None:
    """Frame index from a row label: an int, an integral float, or a string containing exactly one run of digits
    ("30", "frame_60", "f0030"). None, bools, ranges ("0-300") and anything else give None (row is dropped)."""
    if k is None or isinstance(k, bool):
        return None
    if isinstance(k, (int, float)):
        return int(k) if float(k).is_integer() else None
    s = str(k).strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else None
    except (ValueError, OverflowError):
        pass
    m = re.fullmatch(r"\D*?(\d+)\D*", s)
    return int(m.group(1)) if m else None


def fmt_metric(metric: str, v: float | None) -> str:
    return "–" if v is None else METRIC_FMT[metric].format(v)


def fmt_mean_std(metric: str, mean: float | None, std: float | None) -> str:
    if mean is None:
        return "–"
    s = METRIC_FMT[metric].format(mean)
    if std is not None:
        s += " ± " + METRIC_FMT[metric].format(std)
    return s


# ----------------------------------------------------------------------------- summary parsing
def load_json(path: Path | None, what: str) -> Any:
    if path is None:
        return None
    if not path.is_file():
        warn(f"{what} not found: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        warn(f"{what} unreadable ({e}): {path}")
        return None


def _find_nodes(summary: Any, keys: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Candidate (key, node) pairs in priority order. Only dict/list values qualify, so a scalar stored under an
    alias (e.g. the clip length "frames": 901 next to a "per_frame" table) is never taken for the table."""
    if not isinstance(summary, dict):
        return []
    ls = _lower_keys(summary)
    return [(k, ls[k]) for k in keys if k in ls and isinstance(ls[k], (dict, list))]


def metrics_of(d: Any) -> dict[str, float | None]:
    """psnr/ssim/lpips means from a dict with flexible key names."""
    out: dict[str, float | None] = {m: None for m in METRICS}
    if not isinstance(d, dict):
        return out
    ld = _lower_keys(d)
    for m in METRICS:
        for k in (m, f"{m}_mean", f"mean_{m}", f"{m}_avg", f"{m}_db"):
            if k in ld:
                out[m] = _num(ld[k])
                break
    return out


# node aliases in priority order; the generic "frames" / "means" aliases come last and, like every alias, only
# count when the value is a dict or list (see _find_nodes)
CHECKPOINT_KEYS = ("checkpoints", "checkpoint", "checkpoint_metrics", "per_checkpoint", "per_frame", "frame_metrics",
                   "frames")
WINDOW_KEYS = ("windows", "window_means", "window_metrics", "window", "segments", "means")
# label keys of one row (list-of-dicts) or of the label array (dict-of-parallel-arrays)
CP_LABEL_KEYS = ("frame", "frame_idx", "frame_index", "frame_id", "checkpoint", "frames", "index", "idx", "step", "t")
WD_LABEL_KEYS = ("window", "label", "name", "range", "frames", "frame_range", "span")
_RANGE_PAIRS = (("start", "end"), ("from", "to"), ("begin", "end"))


def _rows_from_node(node: Any, label_keys: tuple[str, ...]) -> list[tuple[Any, dict]]:
    """Normalise a dict-of-dicts / list-of-dicts / dict-of-parallel-arrays into [(label, record_dict)].

    A list row without a recognised label key gets label None. The list POSITION is never used as a label:
    callers drop None-labelled rows (with a WARNING) instead of showing metrics under a wrong frame number.
    """
    rows: list[tuple[Any, dict]] = []
    if isinstance(node, dict):
        lnode = _lower_keys(node)
        arrays = {k: v for k, v in lnode.items() if isinstance(v, list)}
        label_arr = next((lnode[k] for k in label_keys if k in lnode and isinstance(lnode[k], list)), None)
        if label_arr is not None and arrays:
            for i, lab in enumerate(label_arr):
                rec = {k: v[i] for k, v in arrays.items() if i < len(v) and k not in label_keys}
                rows.append((lab, rec))
        else:
            for k, v in node.items():
                if isinstance(v, dict):
                    rows.append((k, v))
    elif isinstance(node, list):
        for item in node:
            if not isinstance(item, dict):
                continue
            li = _lower_keys(item)
            lab = next((li[k] for k in label_keys if k in li), None)
            if lab is None:
                for a, b in _RANGE_PAIRS:
                    if a in li and b in li:
                        lab = f"{li[a]}-{li[b]}"
                        break
            rows.append((lab, item))
    return rows


def _describe_row(lab: Any, rec: dict) -> str:
    if lab is not None:
        return repr(str(lab))
    return "{" + ", ".join(sorted(str(k) for k in rec)[:6]) + ("...}" if len(rec) > 6 else "}")


def select_rows(summary: Any, node_keys: tuple[str, ...], label_keys: tuple[str, ...], is_frames: bool,
                ctx: str) -> tuple[list[tuple[Any, dict]], int]:
    """(rows, n_dropped) from the first candidate node that yields labelled rows.

    Frame rows carry an int frame index (labels that are not a single frame number are dropped); window rows
    carry a str label (None labels are dropped). Dropped rows are counted and reported with a WARNING so that
    numbers are never displayed under a guessed label. A node that is present but yields nothing also warns.
    """
    kind = "checkpoint" if is_frames else "window"
    what = "frame number" if is_frames else "window label"
    candidates = _find_nodes(summary, node_keys)
    first_dropped: tuple[str, int, list[str]] | None = None
    for key, node in candidates:
        kept: list[tuple[Any, dict]] = []
        bad: list[str] = []
        for lab, rec in _rows_from_node(node, label_keys):
            if is_frames:
                fl: Any = _frame_int(lab)
            else:
                fl = None if lab is None else str(lab).strip() or None
            if fl is None:
                bad.append(_describe_row(lab, rec))
            else:
                kept.append((fl, rec))
        if kept:
            if bad:
                warn(f"{ctx}: {len(bad)} {kind} row(s) under '{key}' without a {what} ignored "
                     f"(e.g. {', '.join(bad[:3])})")
            if is_frames:
                kept.sort(key=lambda r: r[0])
            return kept, len(bad)
        if bad and first_dropped is None:
            first_dropped = (key, len(bad), bad)
    if first_dropped is not None:
        key, n_bad, bad = first_dropped
        warn(f"{ctx}: all {n_bad} {kind} row(s) under '{key}' lack a {what}; ignored (e.g. {', '.join(bad[:3])})")
        return [], n_bad
    if candidates:
        warn(f"{ctx}: {kind} node '{candidates[0][0]}' has no recognisable rows; {kind} metrics omitted")
    return [], 0


def checkpoint_table(summary: Any, ctx: str) -> tuple[dict[int, dict[str, float | None]], int]:
    """({frame: {psnr, ssim, lpips}}, n_dropped_rows) from any of the accepted layouts."""
    rows, dropped = select_rows(summary, CHECKPOINT_KEYS, CP_LABEL_KEYS, True, ctx)
    return {int(fi): metrics_of(rec) for fi, rec in rows}, dropped


def window_table(summary: Any, ctx: str) -> tuple[list[tuple[str, dict[str, float | None]]], int]:
    rows, dropped = select_rows(summary, WINDOW_KEYS, WD_LABEL_KEYS, False, ctx)
    return [(str(lab), metrics_of(rec)) for lab, rec in rows], dropped


def agg_stats(rec: Any, metric: str) -> tuple[float | None, float | None, int | None]:
    """(mean, std, n) for one metric of one aggregate record, tolerant to several layouts."""
    if not isinstance(rec, dict):
        return None, None, None
    ld = _lower_keys(rec)
    mean = std = None
    n: int | None = None
    if metric in ld:
        v = ld[metric]
        if isinstance(v, dict):
            lv = _lower_keys(v)
            mean = _num(next((lv[k] for k in ("mean", "avg", "value") if k in lv), None))
            std = _num(next((lv[k] for k in ("std", "stdev", "sd", "stddev") if k in lv), None))
            n = _int(next((lv[k] for k in ("n", "count", "num") if k in lv), None))
        elif isinstance(v, (list, tuple)):
            mean = _num(v[0]) if len(v) > 0 else None
            std = _num(v[1]) if len(v) > 1 else None
            n = _int(v[2]) if len(v) > 2 else None
        else:
            mean = _num(v)
    if mean is None:
        mean = _num(next((ld[k] for k in (f"{metric}_mean", f"mean_{metric}") if k in ld), None))
    if std is None:
        std = _num(next((ld[k] for k in (f"{metric}_std", f"std_{metric}", f"{metric}_stdev", f"{metric}_sd") if k in ld), None))
    if n is None:
        n = _int(next((ld[k] for k in ("n", "count", "n_episodes", "num_episodes") if k in ld), None))
    return mean, std, n


def agg_top_n(agg: Any) -> int | None:
    if not isinstance(agg, dict):
        return None
    la = _lower_keys(agg)
    for k in ("n", "n_episodes", "num_episodes", "count", "n_entries"):
        if k in la and _int(la[k]) is not None:
            return _int(la[k])
    for k in ("episodes", "entries", "names"):
        if k in la and isinstance(la[k], (list, dict)):
            return len(la[k])
    return None


# ----------------------------------------------------------------------------- video handling
def copy_video(src: str | None, dest_dir: Path, role: str) -> tuple[Path | None, bool]:
    """Copy (never move) a video into dest_dir as <role>__<basename>.

    Returns (dest, copied): dest is None if the source is missing; copied is False when an identical copy
    (same size and mtime, or the source itself) is already in place and was reused.
    """
    if not src:
        return None, False
    sp = Path(str(src)).expanduser()
    if not sp.is_file():
        return None, False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{role}__{safe_name(sp.name, 'video.mp4')}"
    if dest.exists():
        if dest.resolve() == sp.resolve():
            return dest, False
        ds, ss = dest.stat(), sp.stat()
        if ds.st_size == ss.st_size and int(ds.st_mtime) == int(ss.st_mtime):
            return dest, False  # identical copy from a previous run
    shutil.copy2(sp, dest)
    return dest, True


def probe(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"width": None, "height": None, "frames": None, "fps": None}
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            info["width"], info["height"] = s.codec_context.width, s.codec_context.height
            info["frames"] = s.frames or None
            info["fps"] = float(s.average_rate) if s.average_rate else None
    except Exception as e:  # noqa: BLE001
        warn(f"could not probe {path}: {e}")
    return info


def extract_frames(video: Path, indices: list[int], frames_dir: Path, prefix: str) -> dict[int, Path]:
    """Decode sequentially with PyAV, save the requested frame indices as JPEG q85 / width 320 via Pillow."""
    wanted = set(indices)
    out: dict[int, Path] = {}
    if not wanted:
        return out
    last = max(wanted)
    frames_dir.mkdir(parents=True, exist_ok=True)
    decoded = 0
    try:
        with av.open(str(video)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(s)):
                decoded = i + 1
                if i in wanted:
                    img = fr.to_image()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    h = max(1, round(img.height * THUMB_WIDTH / img.width))
                    img = img.resize((THUMB_WIDTH, h), Image.Resampling.LANCZOS)
                    p = frames_dir / f"{prefix}_f{i:04d}.jpg"
                    img.save(p, "JPEG", quality=JPEG_QUALITY, optimize=True)
                    out[i] = p
                if i >= last:
                    break
    except Exception as e:  # noqa: BLE001
        warn(f"decode failed for {video}: {e}")
    missing = sorted(wanted - out.keys())
    if missing:
        warn(f"{video.name}: frames {missing} not available (decoded {decoded} frames)")
    return out


# ----------------------------------------------------------------------------- HTML pieces
CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 32px 64px; background: #fcfcfb; color: #0b0b0b;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.45; }
header { max-width: 1100px; margin-bottom: 28px; }
h1 { font-size: 26px; margin: 0 0 8px; font-weight: 650; letter-spacing: -0.01em; }
h2 { font-size: 20px; margin: 0 0 6px; font-weight: 650; }
h3 { font-size: 12px; margin: 20px 0 8px; font-weight: 600; color: #52514e; text-transform: uppercase; letter-spacing: 0.06em; }
p { margin: 0 0 8px; }
a { color: #0b0b0b; }
.secondary, .meta, .note { color: #52514e; }
.meta { font-size: 13px; }
.note { font-size: 14px; }
.toc { font-size: 14px; margin: 12px 0 0; padding-left: 20px; }
.toc li { margin: 2px 0; }
section { border-top: 1px solid #e5e4e0; padding: 24px 0 16px; }
.prompt { max-width: 1100px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
button { font: inherit; font-size: 13px; padding: 5px 12px; border: 1px solid #e5e4e0; border-radius: 6px;
  background: #ffffff; color: #0b0b0b; cursor: pointer; }
button:hover { background: #f4f3f0; }
.videos { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.videos figure { margin: 0; width: 480px; max-width: 100%; }
.videos video { display: block; width: 100%; height: auto; background: #000; border: 1px solid #e5e4e0; }
figcaption { margin-top: 6px; font-size: 13px; }
figcaption .src { display: block; color: #52514e; font-size: 12px; word-break: break-all; }
.placeholder { display: flex; align-items: center; justify-content: center; text-align: center; padding: 12px;
  width: 100%; aspect-ratio: 16 / 9; background: #f1f0ec; border: 1px solid #e5e4e0; color: #52514e;
  font-size: 13px; word-break: break-all; }
.strip { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.col { width: 240px; max-width: 100%; display: flex; flex-direction: column; gap: 3px; }
.col img { display: block; width: 100%; height: auto; border: 1px solid #e5e4e0; }
.col .placeholder { font-size: 12px; padding: 6px; }
.rl { font-size: 11px; color: #52514e; line-height: 1.2; margin-bottom: 5px; }
.fidx { font-weight: 600; font-size: 13px; margin-top: 4px; }
.fm { font-size: 12px; color: #52514e; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0 12px; }
th, td { border: 1px solid #e5e4e0; padding: 5px 10px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
th { font-weight: 600; background: #f7f6f3; }
th:first-child, td:first-child { text-align: left; }
img.aggplot { display: block; width: 900px; max-width: 100%; height: auto; border: 1px solid #e5e4e0; margin-top: 8px; }
@media (max-width: 700px) { body { padding: 16px; } }
"""

JS = """
(function () {
  function drive(section, action) {
    var vids = section.querySelectorAll('video');
    for (var i = 0; i < vids.length; i++) {
      var v = vids[i];
      if (action === 'restart') { try { v.currentTime = 0; } catch (e) {} }
      if (action === 'pause') { v.pause(); continue; }
      var p = v.play();
      if (p && typeof p.catch === 'function') { p.catch(function () {}); }
    }
  }
  document.addEventListener('click', function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('button[data-action]') : null;
    if (!btn) { return; }
    var section = btn.closest('section.entry');
    if (section) { drive(section, btn.getAttribute('data-action')); }
  });
})();
"""


def render_metrics_line(m: dict[str, float | None] | None) -> str:
    if not m or all(m.get(k) is None for k in METRICS):
        return ""
    parts = [f"PSNR {fmt_metric('psnr', m.get('psnr'))} dB", f"SSIM {fmt_metric('ssim', m.get('ssim'))}",
             f"LPIPS {fmt_metric('lpips', m.get('lpips'))}"]
    return f'<div class="fm">{" · ".join(parts)}</div>'


def render_window_table(rows: list[tuple[str, dict[str, float | None]]]) -> str:
    if not rows:
        return ""
    out = ['<h3>Window means</h3>', '<table class="metrics"><thead><tr><th>Window (frames)</th>']
    out += [f"<th>{esc(METRIC_LABEL[m])}</th>" for m in METRICS]
    out.append("</tr></thead><tbody>")
    for label, m in rows:
        out.append(f"<tr><td>{esc(label)}</td>" + "".join(f"<td>{fmt_metric(k, m.get(k))}</td>" for k in METRICS) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_aggregate_table(rows: list[tuple[Any, dict]], first_col: str, top_n: int | None) -> str:
    """Table of mean ± std (n) per metric for rows already validated/labelled by select_rows()."""
    if not rows:
        return ""
    out = [f'<table class="aggregate"><thead><tr><th>{esc(first_col)}</th>']
    out += [f"<th>{esc(METRIC_LABEL[m])} mean ± std</th>" for m in METRICS]
    out.append("<th>n</th></tr></thead><tbody>")
    for lab, rec in rows:
        cells, ns = [], []
        for m in METRICS:
            mean, std, n = agg_stats(rec, m)
            cells.append(f"<td>{fmt_mean_std(m, mean, std)}</td>")
            if n is not None:
                ns.append(n)
        n_txt = ", ".join(str(n) for n in sorted(set(ns))) if ns else ("" if top_n is None else str(top_n))
        out.append(f"<tr><td>{esc(lab)}</td>{''.join(cells)}<td>{esc(n_txt or '–')}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_aggregate(agg: Any, plot_rel: str | None, plot_src: Path | None) -> str:
    if agg is None and plot_rel is None and plot_src is None:
        return ""
    parts = ['<section class="aggregate" id="aggregate">', "<h2>Aggregate over episodes</h2>"]
    top_n = agg_top_n(agg)
    if top_n is not None:
        parts.append(f'<p class="meta">n = {top_n} episodes</p>')
    if agg is not None:
        cp_rows, cp_dropped = select_rows(agg, CHECKPOINT_KEYS, CP_LABEL_KEYS, True, "aggregate summary")
        wd_rows, wd_dropped = select_rows(agg, WINDOW_KEYS, WD_LABEL_KEYS, False, "aggregate summary")
        cp = render_aggregate_table(cp_rows, "Checkpoint (frame)", top_n)
        wd = render_aggregate_table(wd_rows, "Window (frames)", top_n)
        if cp:
            parts.append("<h3>Checkpoint metrics (mean ± std across episodes)</h3>")
            parts.append(cp)
        if cp_dropped:
            parts.append(f'<p class="note">{cp_dropped} checkpoint row(s) without a frame label ignored.</p>')
        if wd:
            parts.append("<h3>Window means (mean ± std across episodes)</h3>")
            parts.append(wd)
        if wd_dropped:
            parts.append(f'<p class="note">{wd_dropped} window row(s) without a label ignored.</p>')
        if not cp and not wd:
            parts.append('<p class="note">Aggregate file has no recognisable checkpoint or window tables.</p>')
    if plot_rel:
        parts.append(f'<img class="aggplot" src="{esc(plot_rel)}" alt="aggregate drift plot">')
    elif plot_src is not None:
        parts.append(f'<div class="placeholder" style="width:900px;max-width:100%">missing: {esc(plot_src)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- per entry
class Clip:
    def __init__(self, role: str, label: str, src: str | None):
        self.role, self.label, self.src = role, label, src
        self.dest: Path | None = None
        self.info: dict[str, Any] = {}
        self.thumbs: dict[int, Path] = {}


def process_entry(entry: dict, idx: int, total: int, dirname: str, out_dir: Path, frames: list[int], stats: dict) -> str:
    name = str(entry.get("name") or f"entry {idx + 1}")
    asset_dir = out_dir / "assets" / dirname
    frames_dir = asset_dir / "frames"

    clips = [Clip("real", "real", entry.get("real")), Clip("gen", "generated", entry.get("gen"))]
    extras = entry.get("extra")
    if extras is None:
        extras = []
    elif isinstance(extras, (dict, str)):
        extras = [extras]  # one extra given as a single object or as a bare path string
    elif not isinstance(extras, list):
        warn(f"{name}: 'extra' must be a list of {{label, video}} objects (or one object / path string), "
             f"got {type(extras).__name__}; ignored")
        extras = []
    for j, ex in enumerate(extras):
        if isinstance(ex, str):
            if not ex.strip():
                warn(f"{name}: extra #{j} is an empty path, skipped")
                continue
            ex = {"video": ex}
        if not isinstance(ex, dict):
            warn(f"{name}: extra #{j} is not an object, skipped")
            continue
        clips.append(Clip(f"extra{j:02d}", str(ex.get("label") or f"extra {j + 1}"), ex.get("video") or ex.get("path")))

    n_thumbs = n_missing = 0
    for c in clips:
        c.dest, copied = copy_video(c.src, asset_dir, c.role)
        if c.dest is None:
            warn(f"{name}: {c.label} video missing: {c.src if c.src else '(no path given)'}")
            stats["missing"] += 1
            n_missing += 1
            continue
        stats["copied" if copied else "reused"] += 1
        c.info = probe(c.dest)
        c.thumbs = extract_frames(c.dest, frames, frames_dir, c.role)
        stats["thumbs"] += len(c.thumbs)
        n_thumbs += len(c.thumbs)
    print(f"[{idx + 1}/{total}] {name}: {len(clips) - n_missing}/{len(clips)} videos, {n_thumbs} thumbnails"
          + (f", {n_missing} missing" if n_missing else ""), flush=True)

    summary_path = entry.get("summary")
    summary = load_json(Path(str(summary_path)).expanduser(), f"{name}: summary") if summary_path else None
    cps, cps_dropped = checkpoint_table(summary, f"{name}: summary")
    wins, wins_dropped = window_table(summary, f"{name}: summary")

    h: list[str] = [f'<section class="entry" id="entry-{esc(dirname)}" data-entry="{esc(dirname)}">']
    h.append(f"<h2>{esc(name)}</h2>")
    if entry.get("prompt"):
        h.append(f'<p class="prompt"><span class="secondary">Prompt:</span> {esc(entry["prompt"])}</p>')
    if entry.get("note"):
        h.append(f'<p class="note">{esc(entry["note"])}</p>')
    h.append('<div class="controls"><button type="button" data-action="play">Play all</button>'
             '<button type="button" data-action="pause">Pause all</button>'
             '<button type="button" data-action="restart">Restart</button></div>')

    # video row ---------------------------------------------------------------
    h.append('<div class="videos">')
    for c in clips:
        if c.dest is None:
            h.append(f'<figure class="missing"><div class="placeholder">missing: {esc(c.src or "(no path given)")}</div>'
                     f'<figcaption>{esc(c.label)}</figcaption></figure>')
            continue
        dims = ""
        if c.info.get("width") and c.info.get("height"):
            dims = f' · {c.info["width"]}x{c.info["height"]}'
        if c.info.get("frames"):
            dims += f' · {c.info["frames"]} frames'
        if c.info.get("fps"):
            dims += f' · {c.info["fps"]:g} fps'
        h.append(f'<figure><video controls muted loop preload="metadata" playsinline src="{esc(rel(c.dest, out_dir))}" '
                 f'data-role="{esc(c.role)}" title="{esc(c.label)}"></video>'
                 f'<figcaption><strong>{esc(c.label)}</strong><span class="secondary">{esc(dims)}</span>'
                 f'<span class="src">{esc(c.src)}</span></figcaption></figure>')
    h.append("</div>")

    # frame strip -------------------------------------------------------------
    h.append("<h3>Frame strip</h3>")
    h.append('<div class="strip">')
    for f in frames:
        h.append('<div class="col">')
        for c in clips:
            if c.dest is None:
                h.append(f'<div class="placeholder" title="{esc(c.src or "")}">missing: {esc(Path(str(c.src)).name if c.src else "(no path)")}</div>')
            elif f in c.thumbs:
                h.append(f'<img src="{esc(rel(c.thumbs[f], out_dir))}" alt="{esc(c.label)}, frame {f}" loading="lazy">')
            else:
                h.append(f'<div class="placeholder">frame {f} unavailable</div>')
            h.append(f'<div class="rl">{esc(c.label)}</div>')
        h.append(f'<div class="fidx">frame {f}</div>')
        h.append(render_metrics_line(cps.get(f)))
        h.append("</div>")
    h.append("</div>")

    # window means table ------------------------------------------------------
    h.append(render_window_table(wins))
    if summary is not None:
        notes: list[str] = []
        if not cps and not wins:
            notes.append("Summary file has no recognisable checkpoint or window metrics.")
        elif cps and not any(f in cps for f in frames):
            notes.append(f"Summary has checkpoint metrics at frames {', '.join(str(k) for k in sorted(cps))}, none of "
                         f"which is a requested strip frame; per-frame metrics not shown.")
        if cps_dropped:
            notes.append(f"{cps_dropped} checkpoint row(s) without a frame label ignored.")
        if wins_dropped:
            notes.append(f"{wins_dropped} window row(s) without a label ignored.")
        for n in notes:
            h.append(f'<p class="note">{esc(n)}</p>')
    h.append("</section>")
    return "\n".join(h)


# ----------------------------------------------------------------------------- main
def read_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            warn(f"{path.name}:{ln}: invalid JSON skipped ({e})")
            continue
        if not isinstance(obj, dict):
            warn(f"{path.name}:{ln}: not a JSON object, skipped")
            continue
        entries.append(obj)
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a self-contained HTML review page of real | generated rollouts.")
    ap.add_argument("--entries", required=True, type=Path, help="ENTRIES.jsonl (one entry per line)")
    ap.add_argument("--out", required=True, type=Path, help="OUT_DIR/index.html (assets are written under OUT_DIR/assets/)")
    ap.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES, help="frame indices for the frame strip")
    ap.add_argument("--aggregate", type=Path, default=None, help="aggregate_summary.json (checkpoint mean/std/n)")
    ap.add_argument("--aggregate-plot", type=Path, default=None, help="aggregate_drift.png, copied into assets/")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--intro", default=DEFAULT_INTRO)
    ap.add_argument("--generated-at", default=None, help="free text; default is the current local time")
    args = ap.parse_args(argv)

    if not args.entries.is_file():
        print(f"ERROR: entries file not found: {args.entries}", file=sys.stderr)
        return 2
    frames = sorted({int(f) for f in args.frames})
    if any(f < 0 for f in frames):
        print("ERROR: --frames must be non-negative", file=sys.stderr)
        return 2

    out_path: Path = args.out
    if out_path.is_dir() or out_path.suffix.lower() not in (".html", ".htm"):
        out_path = out_path / "index.html"
    out_dir = out_path.parent
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    generated_at = args.generated_at or dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    entries = read_entries(args.entries)
    if not entries:
        warn(f"no entries found in {args.entries}")

    # unique, filesystem-safe directory names ------------------------------------------------
    used: set[str] = set()
    dirnames: list[str] = []
    for i, e in enumerate(entries):
        base = safe_name(e.get("name"), f"entry{i + 1:03d}")
        d, k = base, 2
        while d in used:
            d, k = f"{base}_{k}", k + 1
        used.add(d)
        dirnames.append(d)

    # aggregate --------------------------------------------------------------------------------
    agg = load_json(args.aggregate, "aggregate summary") if args.aggregate else None
    plot_rel: str | None = None
    if args.aggregate_plot is not None:
        if args.aggregate_plot.is_file():
            dest = out_dir / "assets" / safe_name(args.aggregate_plot.name, "aggregate_drift.png")
            if not (dest.exists() and dest.resolve() == args.aggregate_plot.resolve()):
                shutil.copy2(args.aggregate_plot, dest)
            plot_rel = rel(dest, out_dir)
        else:
            warn(f"aggregate plot not found: {args.aggregate_plot}")

    stats = {"copied": 0, "reused": 0, "missing": 0, "thumbs": 0, "failed": 0}
    sections: list[str] = []
    for i, e in enumerate(entries):
        try:
            sections.append(process_entry(e, i, len(entries), dirnames[i], out_dir, frames, stats))
        except Exception as ex:  # noqa: BLE001 - one malformed entry must never lose the whole page
            name = str(e.get("name") or f"entry {i + 1}")
            stats["failed"] += 1
            err = f"{type(ex).__name__}: {ex}"
            warn(f"{name}: entry could not be processed ({err}); placeholder section emitted")
            sections.append(f'<section class="entry" id="entry-{esc(dirnames[i])}" data-entry="{esc(dirnames[i])}">'
                            f'<h2>{esc(name)}</h2><p class="note">Entry could not be processed: {esc(err)}</p></section>')

    # page -------------------------------------------------------------------------------------
    page: list[str] = ["<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
                       '<meta name="viewport" content="width=device-width, initial-scale=1">',
                       f"<title>{esc(args.title)}</title>", f"<style>{CSS}</style>", "</head>", "<body>", "<header>",
                       f"<h1>{esc(args.title)}</h1>", f'<p class="intro">{esc(args.intro)}</p>',
                       f'<p class="meta">Generated {esc(generated_at)} · {len(entries)} '
                       f'{"entry" if len(entries) == 1 else "entries"} · frame checkpoints: '
                       f'{esc(", ".join(str(f) for f in frames))}</p>']
    if len(entries) > 1:
        page.append('<ol class="toc">' + "".join(
            f'<li><a href="#entry-{esc(d)}">{esc(e.get("name") or d)}</a></li>' for e, d in zip(entries, dirnames)) + "</ol>")
    page.append("</header>")
    page.append(render_aggregate(agg, plot_rel, args.aggregate_plot if plot_rel is None else None))
    page.extend(sections)
    if not entries:
        page.append('<section><p class="note">No entries.</p></section>')
    page.append(f"<script>{JS}</script>")
    page.append("</body></html>")
    out_path.write_text("\n".join(page), encoding="utf-8")

    print(f"entries: {len(entries)} | videos copied: {stats['copied']} | reused identical copies: {stats['reused']} | "
          f"missing videos: {stats['missing']} | thumbnails written: {stats['thumbs']} | "
          f"failed entries: {stats['failed']} | warnings: {len(WARNINGS)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
