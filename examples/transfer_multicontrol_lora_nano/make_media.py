# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""All media for transfer_multicontrol_lora_nano: hero, before/after (+ synchronized side-by-side), stretch examples.

    python examples/transfer_multicontrol_lora_nano/make_media.py <dataset_dir> <panel_eval_dir> <stretch_eval_dir> <out_dir>

The eval dirs are the ones make_specs.py and cosmos_framework.scripts.inference wrote (``<eval_dir>/{base,lora}/
<clip>__<mode>/vision.mp4``); ground truth and controls come from ``<dataset_dir>/clips/<clip>/``.
Every file: H.264 MP4 (15 fps, same trim and encoding on every panel) + poster JPG (middle frame) + GIF fallback
(480 px wide, 7.5 fps). Panels are 640x360 except before/after, which keep the native 1280x720.
Clips are fixed here: hero vineyard_test_002, before/after vineyard_test_021, stretch orchard_stretch_000 and
strawberry_stretch_000, all seg + depth -> video. Needs ``imageio-ffmpeg`` (``uv pip install imageio-ffmpeg``).
"""
import subprocess, sys
from pathlib import Path
import cv2, imageio_ffmpeg, numpy as np

DATA, PANEL, STRETCH_DIR, OUT = (Path(p) for p in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
HERO = "vineyard_test_002__segdepth"
BEFORE_AFTER = "vineyard_test_021__segdepth"
STRETCH = {"stretch_orchard": "orchard_stretch_000__segdepth", "stretch_bedded": "strawberry_stretch_000__segdepth"}


def load(p, size=(640, 360)):
    c, fs = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = c.read()
        if not ok:
            return fs
        fs.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f)


def lab(f, t, scale=0.6):
    f = f.copy()
    cv2.rectangle(f, (0, 0), (int(12 + 18.5 * scale * len(t)), int(50 * scale)), (0, 0, 0), -1)
    cv2.putText(f, t, (8, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1 if scale < 1 else 2, cv2.LINE_AA)
    return f


def grid(cols, ncol):
    n = min(len(fs) for _, fs in cols)
    out = []
    for i in range(n):
        p = [lab(fs[i], t, 0.6 if fs[i].shape[1] <= 640 else 1.0) for t, fs in cols]
        rows = [np.hstack(p[k:k + ncol]) for k in range(0, len(p), ncol)]
        out.append(np.vstack(rows))
    return out


def write(name, frames):
    h, w = frames[0].shape[:2]
    cv2.imwrite(str(OUT / f"{name}_poster.jpg"), frames[len(frames) // 2], [cv2.IMWRITE_JPEG_QUALITY, 82])
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "15",
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                          "-movflags", "+faststart", str(OUT / f"{name}.mp4")], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(OUT / f"{name}.mp4"), "-vf",
                    "fps=7.5,scale=480:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    str(OUT / f"{name}.gif")], check=True)
    print(name, f"{w}x{h}", len(frames), "frames")


def clip_paths(eval_dir, spec):
    return DATA / "clips" / spec.split("__")[0], eval_dir


# Hero: seg control, ground truth / base, post-trained (seg + depth drive both models).
c, o = clip_paths(PANEL, HERO)
write("hero", grid([("Control (seg)", load(c / "seg.mp4")), ("Ground truth", load(c / "rgb.mp4")),
                    ("Base Cosmos3-Nano", load(o / "base" / HERO / "vision.mp4")),
                    ("Post-trained LoRA", load(o / "lora" / HERO / "vision.mp4"))], 2))

c, o = clip_paths(PANEL, BEFORE_AFTER)

# Before / after on the same clip and settings; checkpoint is the only variable.
before, after = load(o / "base" / BEFORE_AFTER / "vision.mp4", None), load(o / "lora" / BEFORE_AFTER / "vision.mp4", None)
write("before", grid([("Before: base Cosmos3-Nano", before)], 1))
write("after", grid([("After: post-trained LoRA", after)], 1))
write("before-after", grid([("Before: base Cosmos3-Nano", load(o / "base" / BEFORE_AFTER / "vision.mp4")),
                            ("After: post-trained LoRA", load(o / "lora" / BEFORE_AFTER / "vision.mp4"))], 2))

# Stretch: one orchard and one strawberry-bed clip, seg + depth, 2x2 (seg / GT / base / post-trained).
for tag, spec in STRETCH.items():
    c, o = clip_paths(STRETCH_DIR, spec)
    write(tag, grid([("Control (seg)", load(c / "seg.mp4")), ("Ground truth", load(c / "rgb.mp4")),
                     ("Base Cosmos3-Nano", load(o / "base" / spec / "vision.mp4")),
                     ("Post-trained LoRA", load(o / "lora" / spec / "vision.mp4"))], 2))
