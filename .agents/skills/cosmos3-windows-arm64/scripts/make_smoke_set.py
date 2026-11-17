#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Write the synthetic smoke set for the Windows-on-Arm LoRA SFT smoke: ten 33-frame clips plus the annotations.

Clip i (0..9): a grey road with lane markings; a dark car rectangle driving left to right, which stops at
frame 15 when i % 3 == 0 (question q2 "Is a vehicle stopped or braking" = yes); a red pedestrian rectangle
on the road when i % 2 == 0 (q1 "Is there a pedestrian on the road" = yes), on the pavement otherwise.
The labels in assets/smoke/annotations_*.json are the same functions of i, so the set is learnable.

Usage: python make_smoke_set.py --out <dir>
Needs Pillow and one H.264 encoder: an ffmpeg executable (on PATH, FFMPEG=<path> or --ffmpeg) when there is
one, otherwise PyAV, which the dependency file installs and which bundles libx264. A machine set up with a
self-contained TorchCodec wheel has no ffmpeg executable, so PyAV is the normal route there.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

W, H, FRAMES, FPS = 480, 288, 33, 30
HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets" / "smoke"


def draw_frame(i: int, t: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (120, 170, 90))  # grass
    d = ImageDraw.Draw(img)
    d.rectangle([0, 96, W, 192], fill=(70, 70, 70))  # road
    for x in range(0, W, 40):
        d.rectangle([x, 142, x + 20, 146], fill=(230, 230, 230))  # lane marking
    d.rectangle([0, 192, W, 216], fill=(200, 200, 200))  # pavement
    stops = i % 3 == 0
    car_x = 20 + min(t, 15 if stops else FRAMES) * 12
    d.rectangle([car_x, 110, car_x + 56, 134], fill=(30, 30, 160))  # car
    ped_y = 150 if i % 2 == 0 else 196  # on the road or on the pavement
    ped_x = 400 - t * 3
    d.rectangle([ped_x, ped_y, ped_x + 10, ped_y + 22], fill=(220, 40, 40))  # pedestrian
    return img


def encode_with_ffmpeg(ffmpeg: str, i: int, target: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(FRAMES):
            draw_frame(i, t).save(Path(tmp) / f"frame_{t:03d}.png")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(Path(tmp) / "frame_%03d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(target)]
        subprocess.run(cmd, check=True)


def encode_with_pyav(i: int, target: Path) -> None:
    import av

    with av.open(str(target), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "23"}
        for t in range(FRAMES):
            for packet in stream.encode(av.VideoFrame.from_image(draw_frame(i, t))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory; media/ and the two annotation files are written here")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    args = ap.parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        try:
            import av  # noqa: F401
        except ImportError:
            print(
                f"make_smoke_set: no ffmpeg executable ({args.ffmpeg!r} is not on PATH and is not a file) and PyAV "
                "is not installed. Install the dependency file (it brings PyAV), or pass --ffmpeg <ffmpeg.exe> "
                "of a build with libx264.",
                file=sys.stderr,
            )
            return 2
    out = Path(args.out)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        target = media / f"smoke_{i:03d}.mp4"
        if ffmpeg:
            encode_with_ffmpeg(ffmpeg, i, target)
        else:
            encode_with_pyav(i, target)
    for name in ("annotations_train.json", "annotations_val.json"):
        shutil.copyfile(ASSETS / name, out / name)
    n = len(json.loads((out / "annotations_train.json").read_text(encoding="utf-8")))
    encoder = f"ffmpeg ({ffmpeg})" if ffmpeg else "PyAV libx264 (no ffmpeg executable found)"
    print(f"wrote 10 clips with {encoder} to {media} and the annotations ({n} train records, 4 val records) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
