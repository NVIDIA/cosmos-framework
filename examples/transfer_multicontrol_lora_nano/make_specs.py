# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Write inference specs for the transfer_multicontrol_lora_nano evaluation.

    python examples/transfer_multicontrol_lora_nano/make_specs.py <dataset_dir> <eval_dir> [--set panel|stretch|smoke]

One spec per clip x mode, named ``<clip>__<mode>``, in ``<eval_dir>/specs/``; seed images (frame 0 of the clip) in
``<eval_dir>/seeds/``. Sets:

* ``panel``: the 6 held-out clips marked ``eval_panel`` in ``metadata.csv`` x i2v, seg, depth, segdepth (24 specs).
* ``stretch``: the 4 ``stretch`` clips (orchard, strawberry beds) x i2v, segdepth (8 specs).
* ``smoke``: ``vineyard_test_021`` x segdepth (1 spec), for the smoke tests.

Settings match the recipe's results: seed 2026, 720p, 15 fps, 35 steps, shift 10. Seed image -> video: 149 frames,
guidance 6. Control modes: one 61-frame chunk, no seed frame, guidance 3, control guidance 0.9, weight 1.0 each.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2

SMOKE_CLIP = "vineyard_test_021"
COMMON = {"seed": 2026, "resolution": "720", "aspect_ratio": "16,9", "fps": 15, "num_steps": 35, "shift": 10.0}
CONTROL = {
    "model_mode": "video2video",
    "num_frames": 61,
    "num_video_frames_per_chunk": 61,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "guidance": 3.0,
    "control_guidance": 0.9,
    "negative_metadata_mode": "none",
    "negative_prompt_keep_metadata": False,
    "emphasize_control_in_prompt": False,
}
MODES = {"seg": ("seg",), "depth": ("depth",), "segdepth": ("seg", "depth")}


def spec(clip_dir: Path, seed_png: Path, clip: str, caption: str, mode: str) -> dict:
    s = {"name": f"{clip}__{mode}", "prompt": caption, **COMMON}
    if mode == "i2v":
        return s | {"model_mode": "image2video", "vision_path": str(seed_png), "num_frames": 149, "guidance": 6.0}
    s |= CONTROL
    for c in MODES[mode]:
        s[c] = {"control_path": str(clip_dir / f"{c}.mp4"), "weight": 1.0}
    return s | {"max_frames": 61}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("eval_dir")
    ap.add_argument("--set", choices=("panel", "stretch", "smoke"), default="panel")
    a = ap.parse_args()
    data, out = Path(a.dataset_dir).resolve(), Path(a.eval_dir).resolve()
    rows = list(csv.DictReader(open(data / "metadata.csv")))
    if a.set == "panel":
        clips, modes = [r for r in rows if r["eval_panel"] == "True"], ("i2v", "seg", "depth", "segdepth")
    elif a.set == "stretch":
        clips, modes = [r for r in rows if r["split"] == "stretch"], ("i2v", "segdepth")
    else:
        clips, modes = [r for r in rows if r["id"] == SMOKE_CLIP], ("segdepth",)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "seeds").mkdir(exist_ok=True)
    for r in clips:
        clip_dir, seed_png = data / "clips" / r["id"], out / "seeds" / f"{r['id']}.png"
        ok, frame = cv2.VideoCapture(str(clip_dir / "rgb.mp4")).read()
        assert ok, f"cannot read {clip_dir / 'rgb.mp4'}"
        cv2.imwrite(str(seed_png), frame)
        for m in modes:
            s = spec(clip_dir, seed_png, r["id"], r["caption"], m)
            json.dump(s, open(out / "specs" / f"{s['name']}.json", "w"), indent=1)
    print(f"{len(clips) * len(modes)} specs ({len(clips)} clips x {', '.join(modes)}) -> {out / 'specs'}")


if __name__ == "__main__":
    main()
