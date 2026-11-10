#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# RECONSTRUCTION: the run records name the picture files but not the script that drew them. This one rebuilds them with the
# layout of the published ones (dark background, one title line, a 640x352 panel per column, one caption line).
"""make_media.py - the hero figure, the strip video/GIF and the stills of the Sim-Bimanual recipe.

Inputs per episode: the real 640x360 clip (convert_agibot_episode.py), the matched rollout vision.mp4, and the frozen-first-frame
clip (make_manifests.py frozen). All three are cropped to the rows the evaluation uses (--crop-top 352: rows 0-351, all columns).

  make_media.py hero   --real R --matched M --frozen F --title T --out-dir D --stem hero_task600_ep815362
        writes <stem>_f30_f60_f900.png, <stem>_strip.mp4 (901 frames, 30 fps, frame counter) and <stem>_strip.gif
        (every 9th frame, 650 px wide, 64 colours)
  make_media.py stills --real R --matched M --frozen F --title T --out FILE.png [--frames 30 900]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (17, 17, 17)
TITLE_FG = (255, 255, 255)
CAP_FG = (220, 220, 220)
PANEL_W, PANEL_H, GAP = 640, 352, 8
TITLE_H, CAP_H = 34, 42
ROW_H = TITLE_H + PANEL_H + CAP_H  # 428
COLS = ["real (AgiBotWorld-Beta)", "matched-action rollout", "frozen first frame (control)"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def read_frames(path: Path, indices: list[int] | None, top: int) -> dict[int, np.ndarray]:
    """uint8 [h,w,3] frames (all, or the requested indices), cropped to the first `top` rows."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                            str(path)], check=True, capture_output=True, text=True).stdout.strip()
    w, h = (int(v) for v in probe.split("x"))
    vf = "" if indices is None else "select='" + "+".join(f"eq(n\\,{i})" for i in indices) + "',"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf + "setpts=N/30/TB", "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"]
    raw = np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.uint8).reshape(-1, h, w, 3)
    keys = list(range(raw.shape[0])) if indices is None else sorted(indices)
    return {k: raw[j, :top, :PANEL_W] for j, k in enumerate(keys)}


def compose_row(title: str, panels: list[np.ndarray], frame: int, f_title, f_cap, labels=COLS) -> Image.Image:
    w = 3 * PANEL_W + 4 * GAP
    img = Image.new("RGB", (w, ROW_H), BG)
    d = ImageDraw.Draw(img)
    d.text((GAP, 6), title, font=f_title, fill=TITLE_FG)
    for c, panel in enumerate(panels):
        x = GAP + c * (PANEL_W + GAP)
        img.paste(Image.fromarray(panel), (x, TITLE_H))
        d.text((x + 6, TITLE_H + PANEL_H + 10), f"{labels[c]}, frame {frame}", font=f_cap, fill=CAP_FG)
    return img


def stack(rows: list[Image.Image], gap: int = 4) -> Image.Image:
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + gap * (len(rows) - 1)), BG)
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height + gap
    return out


def load(args, indices):
    return [read_frames(Path(p), indices, args.crop_top) for p in (args.real, args.matched, args.frozen)]


def cmd_stills(a) -> None:
    clips = load(a, a.frames)
    ft, fc = font(19, True), font(17)
    rows = [compose_row(a.title, [c[f] for c in clips], f, ft, fc) for f in a.frames]
    stack(rows, gap=6).save(a.out)
    print("wrote", a.out)


def cmd_hero(a) -> None:
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ft, fc = font(19, True), font(17)
    frames = [30, 60, 900]
    clips = load(a, frames)
    stack([compose_row(a.title, [c[f] for c in clips], f, ft, fc) for f in frames], gap=4).save(out_dir / f"{a.stem}_f30_f60_f900.png")
    full = load(a, None)
    n = min(len(c) for c in full)
    mp4 = out_dir / f"{a.stem}_strip.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * PANEL_W + 4 * GAP}x{ROW_H}", "-r", "30", "-i", "-",
                          "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
    gif_frames = []
    for i in range(n):
        img = compose_row(a.title, [c[i] for c in full], i, ft, fc)
        p.stdin.write(np.asarray(img).tobytes())
        if i % 9 == 0:
            gif_frames.append(img.resize((650, round(ROW_H * 650 / img.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64))
    p.stdin.close(); assert p.wait() == 0
    gif_frames[0].save(out_dir / f"{a.stem}_strip.gif", save_all=True, append_images=gif_frames[1:], duration=300, loop=0, optimize=True, disposal=2)
    print("wrote", mp4, "and the png and gif next to it;", n, "frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("hero", cmd_hero), ("stills", cmd_stills)):
        s = sub.add_parser(name)
        s.add_argument("--real", required=True); s.add_argument("--matched", required=True); s.add_argument("--frozen", required=True)
        s.add_argument("--title", required=True); s.add_argument("--crop-top", type=int, default=PANEL_H)
        if name == "hero":
            s.add_argument("--out-dir", required=True); s.add_argument("--stem", required=True)
        else:
            s.add_argument("--out", required=True); s.add_argument("--frames", nargs="+", type=int, default=[30, 900])
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
