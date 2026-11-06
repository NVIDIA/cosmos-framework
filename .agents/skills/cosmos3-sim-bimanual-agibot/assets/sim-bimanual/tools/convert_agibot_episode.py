#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert one AgiBotWorld-Beta episode into Cosmos3-Nano-Sim-Bimanual forward-dynamics inputs.

Pipeline (all measured/derived from the framework and the checkpoint, no hand-tuned constants):
  1. Read proprio_stats.h5 (state/effector/position, state/joint/position, state/head/position,
     state/waist/position, state/robot/position, state/robot/orientation).
  2. Build per-frame rows with the LeRobot key names the framework's dataset class expects and call
     AgiBotWorldBetaLeRobotDataset._build_fk_action(...) -> raw [T-1, 29] backward-framewise rot6d actions
     (head 9 | right 9 | right gripper 1 | left 9 | left gripper 1), 30 fps, gripper = next frame's open fraction.
  3. Normalise with the checkpoint schema's affine quantile_rot transform: (raw - offset) / scale, float32, no clamp
     (verified bit-exact against the bundled examples' action_raw/action tensors).
  4. Cut the head_color.mp4 to frames [start, start + N + 1), center-crop 640x480 -> 640x360, write mp4 @ 30 fps.
  5. Emit an inference JSONL row in the same shape as inputs/omni/agibot.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

# Recipe copy: the station original hard-coded two local paths here. Set these two variables instead:
#   COSMOS_FRAMEWORK_ROOT   root of the cosmos-framework checkout (the (early access) branch the recipe names)
#   SIM_BIMANUAL_CHECKPOINT the downloaded nvidia/Cosmos3-Nano-Sim-Bimanual directory (default ./Cosmos3-Nano-Sim-Bimanual)
if "COSMOS_FRAMEWORK_ROOT" not in os.environ:
    sys.exit("convert_agibot_episode.py: set COSMOS_FRAMEWORK_ROOT to your cosmos-framework checkout")
FRAMEWORK = Path(os.environ["COSMOS_FRAMEWORK_ROOT"]).expanduser().resolve()
CHECKPOINT = Path(os.environ.get("SIM_BIMANUAL_CHECKPOINT", "./Cosmos3-Nano-Sim-Bimanual")).expanduser().resolve()
sys.path.insert(0, str(FRAMEWORK))

from cosmos_framework.data.generator.action.datasets.agibotworld_beta_lerobot_dataset import (  # noqa: E402
    AgiBotWorldBetaLeRobotDataset,
    _EFFECTOR_KEY, _HEAD_STATE_KEY, _JOINT_KEY, _ROBOT_ORIENTATION_KEY, _ROBOT_POSITION_KEY, _WAIST_KEY,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id  # noqa: E402

H5_KEYS = {
    _EFFECTOR_KEY: "state/effector/position",
    _JOINT_KEY: "state/joint/position",
    _HEAD_STATE_KEY: "state/head/position",
    _WAIST_KEY: "state/waist/position",
    _ROBOT_POSITION_KEY: "state/robot/position",
    _ROBOT_ORIENTATION_KEY: "state/robot/orientation",
}
EXPECTED_DIMS = {_EFFECTOR_KEY: 2, _JOINT_KEY: 14, _HEAD_STATE_KEY: 2, _WAIST_KEY: 2, _ROBOT_POSITION_KEY: 3, _ROBOT_ORIENTATION_KEY: 4}


def load_schema_transform(schema_path: Path, embodiment: str = "agibotworld") -> tuple[np.ndarray, np.ndarray]:
    schema = json.loads(schema_path.read_text())
    emb = schema["embodiments"][embodiment]
    assert emb["layout"]["id"] == "agibot_backward_framewise_rot6d_v1", emb["layout"]["id"]
    assert emb["layout"]["pose_convention"] == "backward_framewise"
    norm = emb["normalizer"]
    assert norm["method"] == "quantile_rot" and norm["transform"]["type"] == "affine" and not norm["transform"]["forward_clamp"]
    assert emb["domain_id"] == get_domain_id(embodiment), (emb["domain_id"], get_domain_id(embodiment))
    offset = np.asarray(norm["transform"]["offset"], dtype=np.float32)
    scale = np.asarray(norm["transform"]["scale"], dtype=np.float32)
    assert offset.shape == (29,) and scale.shape == (29,) and (scale > 0).all()
    return offset, scale


def read_states(h5_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(h5_path, "r") as f:
        for row_key, h5_key in H5_KEYS.items():
            arr = np.asarray(f[h5_key][()], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[1] == EXPECTED_DIMS[row_key], (h5_key, arr.shape)
            out[row_key] = arr
        out["timestamp"] = np.asarray(f["timestamp"][()]) if "timestamp" in f else None
    lengths = {k: v.shape[0] for k, v in out.items() if k != "timestamp"}
    assert len(set(lengths.values())) == 1, lengths
    return out


def raw_actions_from_states(states: dict[str, np.ndarray], start: int, num_actions: int) -> tuple[np.ndarray, dict]:
    """Return raw [num_actions, 29] actions for frames [start, start+num_actions] via the framework's FK conversion."""
    rows = [{k: states[k][i].tolist() for k in H5_KEYS} for i in range(start, start + num_actions + 1)]
    stub = SimpleNamespace(_pose_convention="backward_framewise")
    action, extras = AgiBotWorldBetaLeRobotDataset._build_fk_action(stub, rows)
    raw = action.numpy().astype(np.float32)
    assert raw.shape == (num_actions, 29), raw.shape
    assert np.isfinite(raw).all(), "non-finite raw actions"
    return raw, {k: v.numpy().tolist() for k, v in extras.items()}


def normalise(raw: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    norm = ((raw.astype(np.float32) - offset) / scale).astype(np.float32)
    assert np.isfinite(norm).all(), "non-finite normalised actions"
    return norm


def cut_video(src: Path, dst: Path, start: int, num_frames: int, fps: int = 30) -> dict:
    """Frames [start, start+num_frames) of src, center-cropped to 640x360, re-encoded losslessly enough (crf 8)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(src)],
                           check=True, capture_output=True, text=True)
    st = json.loads(probe.stdout)["streams"][0]
    w, h, n = int(st["width"]), int(st["height"]), int(st["nb_read_frames"])
    assert start + num_frames <= n, f"video has {n} frames, need {start + num_frames}"
    crop_h = int(round(w * 9 / 16))  # 640x480 -> 640x360
    assert crop_h <= h, (w, h)
    y0 = (h - crop_h) // 2
    vf = f"select='between(n\\,{start}\\,{start + num_frames - 1})',setpts=N/{fps}/TB,crop={w}:{crop_h}:0:{y0}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-frames:v", str(num_frames), "-r", str(fps),
                    "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    probe2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(dst)],
                            check=True, capture_output=True, text=True)
    st2 = json.loads(probe2.stdout)["streams"][0]
    assert int(st2["nb_read_frames"]) == num_frames, st2
    return {"src": {"width": w, "height": h, "frames": n, "r_frame_rate": st["r_frame_rate"]},
            "dst": {"width": int(st2["width"]), "height": int(st2["height"]), "frames": int(st2["nb_read_frames"]), "crop_y0": y0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path, help="head_color.mp4")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--start", type=int, default=0, help="first frame index of the window")
    ap.add_argument("--num-actions", type=int, default=900, help="N action rows -> N+1 video frames")
    ap.add_argument("--schema", type=Path, default=CHECKPOINT / "cosmos3_nano_sim_bimanual_action_sources/action_schema.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "actions").mkdir(exist_ok=True)
    offset, scale = load_schema_transform(args.schema)
    states = read_states(args.h5)
    T = states[_JOINT_KEY].shape[0]
    assert args.start + args.num_actions + 1 <= T, f"episode has {T} proprio frames, need {args.start + args.num_actions + 1}"
    raw, extras = raw_actions_from_states(states, args.start, args.num_actions)
    norm = normalise(raw, offset, scale)

    np.save(args.out / "actions" / f"{args.name}_raw.npy", raw)
    action_json = args.out / "actions" / f"{args.name}.json"
    action_json.write_text(json.dumps([[float(v) for v in row] for row in norm]))

    info = {"name": args.name, "h5": str(args.h5), "video": str(args.video), "start": args.start, "num_actions": args.num_actions,
            "proprio_frames": int(T), "raw_action_stats": {"min": raw.min(0).tolist(), "max": raw.max(0).tolist(), "mean": raw.mean(0).tolist()},
            "norm_action_stats": {"min": norm.min(0).tolist(), "max": norm.max(0).tolist(), "abs_mean": np.abs(norm).mean(0).tolist(),
                                  "frac_outside_unit": float((np.abs(norm) > 1).mean())},
            "initial_poses": extras}
    if states["timestamp"] is not None:
        ts = np.asarray(states["timestamp"], dtype=np.float64).ravel()
        if ts.size > 1:
            dt = np.diff(ts[args.start: args.start + args.num_actions + 1])
            info["timestamp_dt_median"] = float(np.median(dt)); info["timestamp_dt_min"] = float(dt.min()); info["timestamp_dt_max"] = float(dt.max())
    if not args.skip_video:
        vid = args.out / f"{args.name}_640x360.mp4"
        info["video_cut"] = cut_video(args.video, vid, args.start, args.num_actions + 1)
        vision_path = str(vid.resolve())
    else:
        vision_path = str(args.video.resolve())
    row = {"model_mode": "forward_dynamics", "name": args.name, "prompt": args.prompt, "vision_path": vision_path,
           "action_path": f"actions/{args.name}.json", "domain_name": "agibotworld", "action_condition": True,
           "raw_action_dim": 29, "fps": 30, "num_frames": args.num_actions + 1, "seed": args.seed}
    (args.out / f"{args.name}.jsonl").write_text(json.dumps(row) + "\n")
    (args.out / f"{args.name}_convert_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("name", "proprio_frames", "num_actions")} | {"frac_outside_unit": info["norm_action_stats"]["frac_outside_unit"],
          "timestamp_dt_median": info.get("timestamp_dt_median"), "video_cut": info.get("video_cut", {}).get("dst")}))


if __name__ == "__main__":
    main()
