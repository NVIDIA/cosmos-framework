#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# RECONSTRUCTION, not a published run script: the run records describe the frozen-first-frame control, the
# 20-row action-swap manifest and the evaluation JSONL files but do not contain the commands that produced them.
# This script rebuilds the three from the per-episode files that tools/convert_agibot_episode.py writes, following
# the rules the records state (see README.md, "Commands", and CHECKLIST.md). Standard library only; `frozen` needs
# ffmpeg on PATH. Updated for the released checkpoint's schema-5 rows: the row's vision_path is now the lossless
# first-frame PNG, the real 901-frame clip is <name>_640x360.mp4 next to it, and `prompts` fills the fixed prompts in.
"""
make_manifests.py  --  build the control inputs and the evaluation manifests of the Sim-Bimanual recipe.

Layout assumed (the recipe's, all paths relative to the framework root unless given):
  heldout/inputs/<task>/<name>.jsonl            one schema-5 inference row per episode (convert_agibot_episode.py)
  heldout/inputs/<task>/<name>_initial.png      the first frame, the row's vision_path (same tool)
  heldout/inputs/<task>/<name>_640x360.mp4      the real 901-frame clip (same tool)
  heldout/inputs/<task>/actions/<name>.json     schema-5 sidecar: 900x59 normalized_unified_v1 actions (same tool)
  outputs/cosmos3_nano_simulation_humanoid_heldout/<name>/vision.mp4          matched rollouts (run08)
  outputs/cosmos3_nano_simulation_humanoid_heldout_swap20/<swap name>/vision.mp4   swap rollouts (run10)

Sub-commands
  prompts write the fixed prompt of every episode (manifests/prompts_heldout_20.json, {name: prompt}) into its row,
          and concatenate one JSONL per task (heldout/inputs/task<task>/task<task>.jsonl). Run it before any rollout.
  swap    write heldout/inputs/swap20/heldout_swap20.jsonl: the i-th task-600 episode (ids ascending) paired with the
          i-th task-596 episode, both directions, 20 rows. Each row keeps the frame episode's prompt and real clip
          and takes the partner's action file. Row name: swap_<frame episode>_actions_<action episode>.
  frozen  for every episode write <out>/<name>_frozen.mp4: the real frame 0, cropped to the generator's 640x352 top
          rows, held for 901 frames at 30 fps (libx264, crf 8, yuv420p). This is the "frozen first frame" control.
  eval    write entries_heldout.jsonl, entries_frozen.jsonl, entries_swap20_vs_real.jsonl ({"name","real","gen","prompt"},
          the batch format of eval_fidelity.py) and review_entries.jsonl (make_review_page.py format, with the frozen
          and swap clips as extra columns).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASKS = ("600", "596")
MATCHED_OUT = "outputs/cosmos3_nano_simulation_humanoid_heldout"
SWAP_OUT = "outputs/cosmos3_nano_simulation_humanoid_heldout_swap20"


def read_rows(inputs: Path) -> dict[str, list[dict]]:
    """Return {task: [row, ...]} sorted by episode id, from heldout/inputs/<task>/task<task>_ep*.jsonl."""
    rows: dict[str, list[dict]] = {}
    for task in TASKS:
        d = inputs / f"task{task}"
        found = []
        for f in sorted(d.glob(f"task{task}_ep*.jsonl")):
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                r = json.loads(line)
                r["_dir"] = str(d)
                found.append(r)
        if not found:
            sys.exit(f"no task{task}_ep*.jsonl rows under {d}")
        found.sort(key=lambda r: int(re.search(r"_ep(\d+)$", r["name"]).group(1)))
        rows[task] = found
    n = {t: len(v) for t, v in rows.items()}
    if n["600"] != n["596"]:
        sys.exit(f"unequal episode counts per task: {n}; the pairing rule needs 10 + 10")
    return rows


def real_clip(row: dict) -> str:
    """The real 901-frame clip of a row: <dir>/<name>_640x360.mp4 (the row's vision_path is the first-frame PNG)."""
    return str((Path(row["_dir"]) / f"{row['name']}_640x360.mp4").resolve())


def action_file(row: dict) -> str:
    p = Path(row["action_path"])
    if not p.is_absolute():
        p = Path(row["_dir"]) / p
    return str(p.resolve())


def cmd_prompts(a: argparse.Namespace) -> None:
    prompts = json.loads(a.prompts.read_text())
    rows = read_rows(a.inputs)
    n = 0
    for task in TASKS:
        lines = []
        for r in rows[task]:
            if r["name"] not in prompts or not prompts[r["name"]].strip():
                sys.exit(f"no prompt for {r['name']} in {a.prompts}")
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row["prompt"] = prompts[r["name"]].strip()
            (Path(r["_dir"]) / f"{r['name']}.jsonl").write_text(json.dumps(row) + "\n")
            lines.append(json.dumps(row) + "\n")
            n += 1
        (Path(rows[task][0]["_dir"]) / f"task{task}.jsonl").write_text("".join(lines))
    print(f"wrote the prompts of {n} rows")


def cmd_swap(a: argparse.Namespace) -> None:
    rows = read_rows(a.inputs)
    out_rows = []
    for r600, r596 in zip(rows["600"], rows["596"]):
        for frame_row, act_row in ((r600, r596), (r596, r600)):
            row = {k: v for k, v in frame_row.items() if not k.startswith("_")}
            row["name"] = f"swap_{frame_row['name']}_actions_{act_row['name']}"
            row["vision_path"] = str(Path(frame_row["vision_path"]).resolve())  # the frame episode's first frame
            row["action_path"] = action_file(act_row)  # absolute, so the row works from any working directory
            out_rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(f"wrote {len(out_rows)} rows to {a.out}")


def cmd_frozen(a: argparse.Namespace) -> None:
    rows = read_rows(a.inputs)
    a.out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for r in rows[task]:
            src = real_clip(r)
            dst = a.out / f"{r['name']}_frozen.mp4"
            # frame 0 -> top 352 rows -> held for 901 frames at 30 fps
            vf = f"select=eq(n\\,0),crop={a.width}:{a.height}:0:0,loop=loop={a.frames - 1}:size=1:start=0,setpts=N/{a.fps}/TB"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-frames:v", str(a.frames),
                   "-r", str(a.fps), "-an", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(dst)]
            subprocess.run(cmd, check=True)
            print(f"wrote {dst}")


def cmd_eval(a: argparse.Namespace) -> None:
    rows = read_rows(a.inputs)
    a.out.mkdir(parents=True, exist_ok=True)
    heldout, frozen, swap, review = [], [], [], []
    for task in TASKS:
        for r in rows[task]:
            name, prompt = r["name"], r.get("prompt", "")
            real = real_clip(r)
            gen = str((a.outputs / Path(MATCHED_OUT).name / name / "vision.mp4").resolve())
            heldout.append({"name": name, "real": real, "gen": gen, "prompt": prompt})
            fz = str((a.frozen / f"{name}_frozen.mp4").resolve())
            frozen.append({"name": name, "real": real, "gen": fz, "prompt": prompt})
            review.append({"name": name, "prompt": prompt, "real": real, "gen": gen,
                           "summary": str((a.out / "matched" / f"{name}_summary.json").resolve()),
                           "extra": [{"label": "frozen first frame", "video": fz}]})
    for r600, r596 in zip(rows["600"], rows["596"]):
        for frame_row, act_row in ((r600, r596), (r596, r600)):
            name = f"swap_{frame_row['name']}_actions_{act_row['name']}"
            gen = str((a.outputs / Path(SWAP_OUT).name / name / "vision.mp4").resolve())
            swap.append({"name": name, "real": real_clip(frame_row), "gen": gen, "prompt": frame_row.get("prompt", "")})
            for rv in review:
                if rv["name"] == frame_row["name"]:
                    rv["extra"].append({"label": f"action swap ({act_row['name']} actions)", "video": gen})
    for fname, data in (("entries_heldout.jsonl", heldout), ("entries_frozen.jsonl", frozen),
                        ("entries_swap20_vs_real.jsonl", swap), ("review_entries.jsonl", review)):
        (a.out / fname).write_text("".join(json.dumps(r) + "\n" for r in data))
        print(f"wrote {len(data)} rows to {a.out / fname}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--inputs", type=Path, default=Path("heldout/inputs"))
    p.add_argument("--prompts", type=Path, default=Path("manifests/prompts_heldout_20.json")); p.set_defaults(fn=cmd_prompts)
    s = sub.add_parser("swap"); s.add_argument("--inputs", type=Path, default=Path("heldout/inputs"))
    s.add_argument("--out", type=Path, default=Path("heldout/inputs/swap20/heldout_swap20.jsonl")); s.set_defaults(fn=cmd_swap)
    f = sub.add_parser("frozen"); f.add_argument("--inputs", type=Path, default=Path("heldout/inputs"))
    f.add_argument("--out", type=Path, default=Path("heldout/frozen")); f.add_argument("--frames", type=int, default=901)
    f.add_argument("--fps", type=int, default=30); f.add_argument("--width", type=int, default=640)
    f.add_argument("--height", type=int, default=352); f.set_defaults(fn=cmd_frozen)
    e = sub.add_parser("eval"); e.add_argument("--inputs", type=Path, default=Path("heldout/inputs"))
    e.add_argument("--outputs", type=Path, default=Path("outputs")); e.add_argument("--frozen", type=Path, default=Path("heldout/frozen"))
    e.add_argument("--out", type=Path, default=Path("heldout/eval")); e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
