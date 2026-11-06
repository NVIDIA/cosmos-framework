#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build the evaluation report for one fd_rollout_eval.py output dir:
  - drift curves (PSNR / SSIM / LPIPS vs frame index; mean line + interquartile wash; chunk boundaries)
  - per-episode metric strips
  - review page index.html: real | generated videos for the first N episodes, summary + per-episode table,
    gripper-check table if gripper_check.json exists.
Chart conventions follow the dataviz skill: one axis per panel, single hue (blue) for the one series,
hairline solid gridlines, text in ink tokens, 2px lines, no dual axes.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK2)


def drift_figure(rows, chunk, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_frame = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("psnr", "ssim", "lpips"):
            by_frame[k][int(r["frame"])].append(float(r[k]))
    frames = sorted(by_frame["psnr"].keys())
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, facecolor=SURFACE)
    titles = {"psnr": "PSNR (dB), higher is better", "ssim": "SSIM, higher is better", "lpips": "LPIPS (AlexNet), lower is better"}
    for ax, k in zip(axes, ("psnr", "ssim", "lpips")):
        style(ax)
        med = np.array([np.median(by_frame[k][f]) for f in frames])
        q1 = np.array([np.percentile(by_frame[k][f], 25) for f in frames])
        q3 = np.array([np.percentile(by_frame[k][f], 75) for f in frames])
        n = np.array([len(by_frame[k][f]) for f in frames])
        ax.fill_between(frames, q1, q3, color=BLUE, alpha=0.10, linewidth=0)
        ax.plot(frames, med, color=BLUE, linewidth=2, solid_capstyle="round")
        for b in range(chunk, max(frames) + 1, chunk):
            ax.axvline(b, color=GRID, linewidth=1, zorder=0)
        ax.set_title(titles[k], loc="left", fontsize=11, color=INK, pad=6)
        # direct label at the end + at first frame
        ax.annotate(f"{med[0]:.2f}", (frames[0], med[0]), textcoords="offset points", xytext=(6, 6), fontsize=9, color=INK2)
        ax.annotate(f"{med[-1]:.2f}", (frames[-1], med[-1]), textcoords="offset points", xytext=(6, 0), fontsize=9, color=INK2)
        ax.set_xlim(0, max(frames) + 2)
    axes[-1].set_xlabel("frame index (frame 0 = given conditioning frame; vertical hairlines = 16-frame chunk boundaries)", color=INK2, fontsize=9)
    fig.text(0.01, 0.005, f"median across held-out episodes, shaded = interquartile range; episodes contributing per frame: {n.max()} at frame 1 down to {n.min()} at the last frame", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return {int(f): {"n": int(len(by_frame["psnr"][f])), "psnr": float(np.median(by_frame["psnr"][f])), "ssim": float(np.median(by_frame["ssim"][f])), "lpips": float(np.median(by_frame["lpips"][f]))} for f in frames}


def episode_strip(ep_rows, chunk, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 1.9), facecolor=SURFACE)
    fr = [int(r["frame"]) for r in ep_rows]
    for ax, k in zip(axes, ("psnr", "ssim", "lpips")):
        style(ax)
        ax.plot(fr, [float(r[k]) for r in ep_rows], color=BLUE, linewidth=2)
        for b in range(chunk, max(fr) + 1, chunk):
            ax.axvline(b, color=GRID, linewidth=1, zorder=0)
        ax.set_title(k.upper(), loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    ev = Path(a.eval_dir)
    rep = ev / "report"; rep.mkdir(exist_ok=True)
    summary = json.loads((ev / "summary.json").read_text())
    rows = list(csv.DictReader((ev / "per_frame_all.csv").open()))
    drift = drift_figure(rows, a.chunk, rep / "drift.png")
    (rep / "drift_median_by_frame.json").write_text(json.dumps(drift, indent=1))
    gripper = json.loads((ev / "gripper_check.json").read_text()) if (ev / "gripper_check.json").exists() else None
    grip_by_ep = {r["episode"]: r for r in gripper["per_episode"]} if gripper else {}

    eps = summary["per_episode"]
    om = summary["overall_mean"]
    html = [f"""<!doctype html><html><head><meta charset="utf-8"><title>{a.title or ev.name}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f9f9f7;color:{INK};margin:24px;max-width:1180px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .sub{{color:{INK2};font-size:13px}}
table{{border-collapse:collapse;font-size:13px;background:{SURFACE}}} th,td{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right;font-variant-numeric:tabular-nums}}
th{{color:{INK2};font-weight:600}} td:first-child,th:first-child,td.t,th.t{{text-align:left}}
.ep{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:6px;padding:12px;margin:14px 0}}
video{{width:100%;max-width:966px;display:block;background:#000}} .cap{{color:{INK2};font-size:12px;margin:4px 0 8px}}
.kpi{{display:flex;gap:14px;margin:12px 0}} .tile{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:6px;padding:10px 14px;min-width:150px}}
.tile .l{{color:{INK2};font-size:12px}} .tile .v{{font-size:26px;font-weight:600}}
</style></head><body>
<h1>{a.title or ev.name}</h1>
<div class="sub">checkpoint {summary['run_config']['checkpoint_path']} · EMA weights: {summary['run_config']['use_ema_weights']} · steps {summary['run_config']['num_steps']} · guidance {summary['run_config']['guidance']} · shift {summary['run_config']['shift']} · seed {summary['run_config']['seed']} · {summary['n_episodes']} held-out episodes, {summary['n_frames_evaluated']} generated frames scored at 160x320 (frame 0 given) · {summary['gen_s_per_chunk_mean']:.1f} s per 16-frame chunk · eval wall {summary['wall_s_total']/60:.0f} min</div>
<div class="kpi"><div class="tile"><div class="l">PSNR mean (dB)</div><div class="v">{om['psnr']:.2f}</div></div>
<div class="tile"><div class="l">SSIM mean</div><div class="v">{om['ssim']:.3f}</div></div>
<div class="tile"><div class="l">LPIPS mean (lower better)</div><div class="v">{om['lpips']:.3f}</div></div>"""]
    if gripper:
        g = gripper["aggregate"]
        html.append(f"""<div class="tile"><div class="l">gripper events matched (±{g['tolerance_frames']} fr)</div><div class="v">{g['matched']}/{g['real_transitions']}</div></div>
<div class="tile"><div class="l">gripper frame agreement</div><div class="v">{g['frame_agreement_mean']:.3f}</div></div>""")
    html.append("</div>")
    html.append(f'<h2>Drift over the episode</h2><img src="drift.png" style="max-width:100%;border:1px solid {GRID}" alt="drift curves">')
    # by chunk table
    html.append("<h2>By chunk index (mean over frames in that chunk)</h2><table><tr><th>chunk</th><th>frames</th><th>episodes</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th></tr>")
    for c, v in summary["by_chunk_index"].items():
        html.append(f"<tr><td>{c}</td><td>{v['n_frames']}</td><td>{v['n_episodes']}</td><td>{v['psnr']:.2f}</td><td>{v['ssim']:.3f}</td><td>{v['lpips']:.3f}</td></tr>")
    html.append("</table>")
    html.append("<h2>Per episode</h2><table><tr><th class=t>episode</th><th class=t>task</th><th>frames</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>PSNR L/R</th><th>LPIPS L/R</th>" + ("<th>gripper real/gen/matched</th><th>agree</th>" if gripper else "") + "</tr>")
    for e in eps:
        g = grip_by_ep.get(e["episode"])
        gcell = f"<td>{g['n_real']}/{g['n_gen']}/{g['n_matched']}</td><td>{g['frame_agreement']:.3f}</td>" if g else ""
        html.append(f"<tr><td class=t>{e['episode']}</td><td class=t>{e['task']}</td><td>{e['frames_evaluated']}</td><td>{e['psnr']:.2f}</td><td>{e['ssim']:.3f}</td><td>{e['lpips']:.3f}</td><td>{e['psnr_left']:.1f} / {e['psnr_right']:.1f}</td><td>{e['lpips_left']:.3f} / {e['lpips_right']:.3f}</td>{gcell}</tr>")
    html.append("</table><div class=sub>L = third-person view (left half), R = wrist camera (right half).</div>")
    html.append(f"<h2>Review: real | generated ({min(a.n_videos, len(eps))} episodes, 20 FPS, frame 0 identical by construction)</h2>")
    for e in eps[: a.n_videos]:
        ed = ev / "episodes" / f"ep{e['episode']:03d}"
        ep_rows = [r for r in rows if int(r["episode"]) == e["episode"]]
        strip = rep / f"ep{e['episode']:03d}_strip.png"
        episode_strip(ep_rows, a.chunk, strip)
        rel = Path("..") / "episodes" / ed.name
        g = grip_by_ep.get(e["episode"])
        gtxt = f" · gripper events real {g['real_transitions']} gen {g['gen_transitions']}" if g else ""
        html.append(f"""<div class="ep"><b>Episode {e['episode']}</b> — {e['task']} <span class=sub>({e['frames_evaluated']} generated frames, PSNR {e['psnr']:.2f}, SSIM {e['ssim']:.3f}, LPIPS {e['lpips']:.3f}{gtxt})</span>
<div class="cap">left: real recording · right: generated from frame 0 + real actions</div>
<video controls loop muted preload="metadata" src="{rel}/side_by_side.mp4"></video>
<img src="{strip.name}" style="max-width:100%" alt="per-frame metrics episode {e['episode']}"></div>""")
    html.append("</body></html>")
    (rep / "index.html").write_text("\n".join(html))
    print(f"report: {rep / 'index.html'}")


if __name__ == "__main__":
    main()
