#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert the raw SutureBot dataset (``jchen396/SutureBot``) to LeRobot v3.0, one root per tissue.

The output follows the dVRK layout that ``cosmos3_h_embodied`` reads through ``meta/modality.json``:

* ``observation.state`` (16-D): per PSM, measured pose ``xyz + quaternion xyzw`` and measured jaw angle.
* ``action`` (16-D): per PSM, pose setpoint ``xyz + quaternion xyzw`` and jaw setpoint.
* ``observation.images.endoscope.left`` (960x540, 30 fps, H.264). ``--all-views`` adds the right
  endoscope and both wrist cameras.
* Task text ``needle pickup`` / ``needle throw`` / ``knot tying`` from the subtask folder name.
  Recovery demonstrations (``*_recovery``) keep the task of their subtask.
* Timestamps are ``frame_index / 30``, the convention of the cosmos3_h_embodied data preparation.

Input layout (after unzipping)::

    <raw>/tissue_<N>/<k>_<subtask>[_recovery]/<episode>/{left_img_dir, right_img_dir, endo_psm1, endo_psm2, ee_csv.csv}

Output: ``<out>/tissue_<N>/`` (a LeRobot v3.0 root each). Episodes are ordered by subtask folder,
then episode folder name, so episode indices are deterministic.

Run in the cosmos-framework environment (it pins the lerobot version the loaders use)::

    python scripts/convert_suturebot_to_lerobot.py --raw-root $SUTUREBOT_RAW_ROOT \\
        --out-root $OPENH_DATA_ROOT/suturebot --tissues 3 10 --workers 2
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FPS = 30
POSE_FIELDS = ("position.x", "position.y", "position.z", "orientation.x", "orientation.y", "orientation.z", "orientation.w")
# Feature names equal the CSV column names (same names as the Open-H SutureBot root).
STATE_NAMES = [n for arm in ("psm1", "psm2") for n in (*(f"{arm}_pose.{f}" for f in POSE_FIELDS), f"{arm}_jaw")]
ACTION_NAMES = [n for arm in ("psm1", "psm2") for n in (*(f"{arm}_sp.{f}" for f in POSE_FIELDS), f"{arm}_jaw_sp")]

# Raw image folder -> (LeRobot key, modality.json key, file pattern). endo_psm1 is the right wrist.
VIEWS = {
    "left_img_dir": ("observation.images.endoscope.left", "endoscope_left", "frame{:06d}_left.jpg"),
    "right_img_dir": ("observation.images.endoscope.right", "endoscope_right", "frame{:06d}_right.jpg"),
    "endo_psm2": ("observation.images.wrist.left", "wrist_left", "frame{:06d}_psm2.jpg"),
    "endo_psm1": ("observation.images.wrist.right", "wrist_right", "frame{:06d}_psm1.jpg"),
}
IMAGE_SHAPES = {"left_img_dir": (540, 960, 3), "right_img_dir": (540, 960, 3), "endo_psm1": (480, 640, 3), "endo_psm2": (480, 640, 3)}
TASKS = {"needle_pickup": "needle pickup", "needle_throw": "needle throw", "knot_tying": "knot tying"}


def modality(views: list[str]) -> dict:
    """``meta/modality.json``: same layout as the Open-H SutureBot root."""
    arms = {
        "psm1_pose": {"start": 0, "end": 7},
        "psm1_gripper": {"start": 7, "end": 8},
        "psm2_pose": {"start": 8, "end": 15},
        "psm2_gripper": {"start": 15, "end": 16},
    }
    return {
        "state": arms,
        "action": arms,
        "video": {VIEWS[v][1]: {"original_key": VIEWS[v][0]} for v in views},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def features(views: list[str]) -> dict:
    out = {
        VIEWS[v][0]: {"dtype": "video", "shape": IMAGE_SHAPES[v], "names": ["height", "width", "channel"]} for v in views
    }
    out["observation.state"] = {"dtype": "float32", "shape": (16,), "names": [STATE_NAMES]}
    out["action"] = {"dtype": "float32", "shape": (16,), "names": [ACTION_NAMES]}
    return out


def discover_episodes(tissue_dir: Path) -> list[tuple[Path, str, bool]]:
    """``(episode dir, task text, is_recovery)``, deterministic order."""
    episodes = []
    for subtask_dir in sorted(p for p in tissue_dir.iterdir() if p.is_dir()):
        name = subtask_dir.name.split("_", 1)[1]  # "1_needle_pickup_recovery" -> "needle_pickup_recovery"
        recovery = name.endswith("_recovery")
        task = TASKS[name.removesuffix("_recovery")]
        for ep in sorted(p for p in subtask_dir.iterdir() if p.is_dir()):
            if (ep / "ee_csv.csv").is_file():
                episodes.append((ep, task, recovery))
    return episodes


def read_episode(ep: Path, views: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, list[Path]]]:
    df = pd.read_csv(ep / "ee_csv.csv")
    missing = [c for c in STATE_NAMES + ACTION_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"{ep}: missing CSV columns {missing}")
    n = len(df)
    frames = {}
    for v in views:
        paths = [ep / v / VIEWS[v][2].format(i) for i in range(n)]
        # Use the longest prefix with every image present (a few episodes end early in one view).
        k = next((i for i, p in enumerate(paths) if not p.is_file()), n)
        frames[v] = paths[:k]
        n = min(n, k)
    state = df[STATE_NAMES].to_numpy(np.float32)[:n]
    action = df[ACTION_NAMES].to_numpy(np.float32)[:n]
    return state, action, {v: p[:n] for v, p in frames.items()}


def convert_tissue(raw_root: Path, out_root: Path, tissue: int, views: list[str], crf: int, threads: int) -> dict:
    from lerobot.datasets import lerobot_dataset, video_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # H.264 at CRF ``crf`` (lerobot's default is CRF 30), short GOP for fast random access.
    encode = functools.partial(video_utils.encode_video_frames, crf=crf)
    video_utils.encode_video_frames = encode
    lerobot_dataset.encode_video_frames = encode

    src = raw_root / f"tissue_{tissue}"
    dst = out_root / f"tissue_{tissue}"
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found (download and unzip tissue_{tissue}.zip first)")
    if dst.exists():
        shutil.rmtree(dst)
    episodes = discover_episodes(src)
    ds = LeRobotDataset.create(
        repo_id=f"suturebot/tissue_{tissue}",
        root=dst,
        fps=FPS,
        robot_type="dvrk",
        features=features(views),
        use_videos=True,
        image_writer_threads=threads,
        vcodec="h264",
    )
    t0, frames, skipped, manifest = time.time(), 0, [], []
    for ep, task, recovery in episodes:
        try:
            state, action, images = read_episode(ep, views)
            if len(state) < 2:
                raise ValueError("fewer than 2 frames")
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                raise ValueError("non-finite kinematics")
            for i in range(len(state)):
                frame = {"observation.state": state[i], "action": action[i], "task": task}  # timestamp = i / FPS
                for v in views:
                    frame[VIEWS[v][0]] = np.asarray(Image.open(images[v][i]).convert("RGB"))
                ds.add_frame(frame)
            ds.save_episode()
            manifest.append(
                {"episode_index": len(manifest), "source": str(ep.relative_to(raw_root)), "task": task,
                 "recovery": recovery, "length": len(state)}
            )  # fmt: skip
            frames += len(state)
        except Exception as err:  # keep going; report at the end
            traceback.print_exc()
            skipped.append({"source": str(ep.relative_to(raw_root)), "error": str(err)})
            ds.clear_episode_buffer()
    ds.finalize()
    if not manifest:
        raise RuntimeError(f"tissue_{tissue}: no episode converted ({len(skipped)} failed)")
    meta = dst / "meta"
    (meta / "modality.json").write_text(json.dumps(modality(views), indent=2) + "\n")
    (meta / "suturebot_episodes.json").write_text(json.dumps({"episodes": manifest, "skipped": skipped}, indent=2) + "\n")
    return {"tissue": tissue, "episodes": len(manifest), "frames": frames, "skipped": len(skipped), "seconds": round(time.time() - t0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory holding the unzipped tissue_<N>/ folders.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory; writes <out-root>/tissue_<N>/.")
    parser.add_argument("--tissues", type=int, nargs="+", default=list(range(1, 11)), help="Tissue IDs to convert.")
    parser.add_argument("--all-views", action="store_true", help="Also encode the right endoscope and wrist views.")
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF of the encoded videos.")
    parser.add_argument("--workers", type=int, default=1, help="Tissues converted in parallel.")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    args = parser.parse_args()

    views = list(VIEWS) if args.all_views else ["left_img_dir"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    failed = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_tissue, args.raw_root, args.out_root, t, views, args.crf, args.image_writer_threads): t
            for t in args.tissues
        }
        for fut in as_completed(futures):
            try:
                print(json.dumps(fut.result()), flush=True)
            except Exception:
                failed = True
                print(f"tissue_{futures[fut]} failed", file=sys.stderr)
                traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
