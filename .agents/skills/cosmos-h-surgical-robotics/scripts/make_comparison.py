#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Side-by-side video of one evaluated sample: ground truth | base checkpoint | fine-tuned checkpoint.

Both runs must have evaluated the same sample (same split, seed and sample name), so the first frame,
actions, prompt, seed and sampler are identical and only the checkpoint differs. Writes an MP4 and a
GIF (for the README). Needs ``ffmpeg`` on PATH.

Example::

    python scripts/make_comparison.py --split test --seed 20260908 --sample 1530 \\
        --base outputs/eval/base --finetuned outputs/eval/iter_000003000 -o outputs/media/hero
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    width, height = out.strip().split(",")
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="evaluate.sh output of the base checkpoint")
    parser.add_argument("--finetuned", type=Path, required=True, help="evaluate.sh output of the fine-tuned checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", default="20260908")
    parser.add_argument("--sample", required=True, help="sample name (window index)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output path without extension")
    parser.add_argument("--fps", type=int, default=10, help="playback fps (the clips are 10 Hz)")
    parser.add_argument("--labels", action="store_true", help="draw labels (needs ffmpeg with drawtext)")
    args = parser.parse_args()

    # <split>/seed_<s>/<family>/forward_dynamics/<window>; the family is suturebot (val) or suturebot_test (test).
    found = sorted((args.base / args.split / f"seed_{args.seed}").glob(f"*/forward_dynamics/{args.sample}"))
    if len(found) != 1:
        raise FileNotFoundError(f"sample {args.sample} not found once under {args.base / args.split}")
    rel = found[0].relative_to(args.base)
    base, ft = args.base / rel, args.finetuned / rel
    videos = [base / "vision_gt.mp4", base / "vision.mp4", ft / "vision.mp4"]
    for v in videos:
        if not v.is_file():
            raise FileNotFoundError(v)
    labels = ["ground truth", "base", "fine-tuned"]
    psnr = [None] + [json.loads((d / "metrics.json").read_text()).get("psnr") for d in (base, ft)]

    # The prediction can be a few rows smaller than the ground truth (latent grid); crop all to
    # the common top-left region, as the PSNR computation does.
    sizes = [_size(v) for v in videos]
    w, h = min(s[0] for s in sizes), min(s[1] for s in sizes)
    chains = []
    for i, (label, p) in enumerate(zip(labels, psnr)):
        chain = f"[{i}:v]crop={w}:{h}:0:0,setpts=N/({args.fps}*TB),fps={args.fps}"
        if args.labels:
            text = label if p is None else f"{label} {p:.1f} dB"
            chain += f",drawtext=text='{text}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.5"
        chains.append(chain + f"[v{i}]")
    graph = ";".join(chains) + ";[v0][v1][v2]hstack=inputs=3[out]"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mp4 = args.output.with_suffix(".mp4")
    gif = args.output.with_suffix(".gif")
    inputs = [x for v in videos for x in ("-i", str(v))]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
         "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
         "-loop", "0", str(gif)],
        check=True,
    )  # fmt: skip
    print(f"wrote {mp4} and {gif}; PSNR base={psnr[1]} fine-tuned={psnr[2]}")


if __name__ == "__main__":
    main()
