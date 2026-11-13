#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert one AgiBotWorld-Beta episode into schema-5 inputs for the released Cosmos3-Nano-Sim-Bimanual.

The released checkpoint takes ``normalized_unified_v1`` actions: 59 values per row, already normalised and masked by the
source (inference does not normalise again). This tool produces them with the framework's own code, not a copy of it:

  1. Read ``proprio_stats.h5`` (effector, joint, head, waist, robot position and orientation) and assemble the FK state
     with ``agibot_semantics.assemble_embodiment_c_state``, exactly as ``LanceAgiBotDataset._build_fk_state`` does.
  2. Robotiq episodes get the dataset's audited gripper calibration (``agibot_grippers.calibrated_gripper_fields``):
     the observed actuator-close range 35..120 is stretched to 0..120 before the open fraction is taken.
  3. ``agibot_semantics.build_agibot_fk_action`` -> 29-D actions in the pose convention the checkpoint pins
     (``backward_chunk_anchored_16f``, read from the checkpoint's action_schema.json).
  4. The AgiBot unified mapping (``LanceAgiBotDataset.get_unified_action_mapping``) scatters the 29-D action into the 59-D
     layout: ego 0-8, right wrist 9-17, right openness 33, left wrist 34-42, left openness 58; every other slot stays zero.
  5. Global-asinh normalisation with the statistics the checkpoint pins (path and sha256 from action_schema.json; the
     file is checked against that hash). The two gripper slots use the metric reference: the statistics are divided by the
     AgiBot gripper width of 0.12 m (``LanceAgiBotDataset.fully_open_gripper_width_m``), as ``LocalBimanualPipeline`` does.
     Invalid slots are set to exactly zero afterwards.
  6. The real clip is cut to frames [start, start + N + 1), resized 640x480 -> 640x360 with the framework's
     ``agibot_video.resize_to_640x360`` (what training does), and written as an mp4 plus a lossless PNG of its first
     frame (the ``vision_path`` of the inference row).
  7. One inference row in the shape of the published ``sim_bimanual_unifiedv1_hf.jsonl`` rows.

Camera masking: training rows where only the head moves are re-labelled ``camera_pose``. This tool writes one scalar
domain, ``agibotworld``, for the whole episode (the recipe's protocol), so every row keeps its robot slots.

``check_conversion_gate.py`` is the acceptance gate: it runs this module's functions against the framework's own data
loader before any rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# COSMOS_FRAMEWORK_ROOT: the cosmos-framework checkout (default: the current directory).
# SIM_BIMANUAL_CHECKPOINT: the downloaded nvidia/Cosmos3-Nano-Sim-Bimanual directory (default ./Cosmos3-Nano-Sim-Bimanual).
FRAMEWORK = Path(os.environ.get("COSMOS_FRAMEWORK_ROOT", ".")).expanduser().resolve()
CHECKPOINT = Path(os.environ.get("SIM_BIMANUAL_CHECKPOINT", "./Cosmos3-Nano-Sim-Bimanual")).expanduser().resolve()
sys.path.insert(0, str(FRAMEWORK))

from cosmos_framework.data.generator.action import agibot_grippers  # noqa: E402
from cosmos_framework.data.generator.action.agibot_semantics import (  # noqa: E402
    assemble_embodiment_c_state,
    build_agibot_fk_action,
)
from cosmos_framework.data.generator.action.agibot_video import resize_to_640x360  # noqa: E402
from cosmos_framework.data.generator.action.lance.lance_agibot_dataset import LanceAgiBotDataset  # noqa: E402
from cosmos_framework.data.generator.action.utils.action_processing import (  # noqa: E402
    load_action_normalization_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import (  # noqa: E402
    LEFT_OPENNESS,
    RIGHT_OPENNESS,
    UNIFIED_ACTION_DIM,
)
from cosmos_framework.inference.bimanual import _invalid_action_slots  # noqa: E402

SOURCE_DATASET = "agibot_world__AgiBotWorld-Beta_20260102"
DOMAIN = "agibotworld"
H5_KEYS = {
    "effector": ("state/effector/position", 2),
    "joint": ("state/joint/position", 14),
    "head": ("state/head/position", 2),
    "waist": ("state/waist/position", 2),
    "robot_position": ("state/robot/position", 3),
    "robot_orientation": ("state/robot/orientation", 4),
}


def load_release_normalizer(checkpoint: Path = CHECKPOINT):
    """Return (normalizer, pose_convention, info) from the checkpoint's pinned action schema."""
    schema_path = checkpoint / "cosmos3_nano_sim_bimanual_action_sources" / "action_schema.json"
    schema = json.loads(schema_path.read_text())
    assert schema["schema_version"] == 5 and schema["input_contract"]["action_space"] == "normalized_unified_v1"
    assert schema["input_contract"]["action_dim"] == UNIFIED_ACTION_DIM
    assert schema["embodiments"][DOMAIN]["domain_id"] == 15
    pose_convention = schema["layout"]["pose_convention"]
    source = schema["normalizer"]["source"]
    stats_path = checkpoint / source["artifact_path"]
    digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    assert digest == source["sha256"], f"pinned statistics hash mismatch: {digest} != {source['sha256']}"
    stats = load_action_normalization_stats("global_asinh", stats_path=stats_path, expected_dim=UNIFIED_ACTION_DIM)
    width = LanceAgiBotDataset.fully_open_gripper_width_m
    q01, q99 = stats["q01"].clone(), stats["q99"].clone()
    for slot in (RIGHT_OPENNESS, LEFT_OPENNESS):  # metric gripper reference -> fraction units of this gripper
        q01[slot] /= width
        q99[slot] /= width
    normalizer = resolve_action_normalization("global_asinh", {**stats, "q01": q01, "q99": q99}, pose_convention=pose_convention)
    info = {"stats_artifact": source["artifact_path"], "stats_sha256": digest, "pose_convention": pose_convention,
            "gripper_width_m": width, "schema_contract_sha256": schema["contract_sha256"]}
    return normalizer, pose_convention, info


def gripper_hardware(task_id: int | str, override: str | None = None) -> str:
    """Hardware of the task's grippers from the audited manifest; one label for the whole task, else an override."""
    if override:
        return override
    shards = agibot_grippers._manifest()["datasets"][SOURCE_DATASET]["shards"].get(f"task_{int(task_id)}")
    if not shards or len(shards) != 1:
        raise SystemExit(f"task {task_id}: the audited manifest has no single gripper hardware label; pass --gripper-hardware")
    return next(iter(shards))


def read_states(h5_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(h5_path, "r") as f:
        for key, (h5_key, width) in H5_KEYS.items():
            arr = np.asarray(f[h5_key][()], dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[1] == width, (h5_key, arr.shape)
            out[key] = arr
    assert len({v.shape[0] for v in out.values()}) == 1, {k: v.shape for k, v in out.items()}
    return out


def fk_state(effector, joint, head, waist, hardware: str) -> np.ndarray:
    """Window [T+1] of observations -> FK state, with the audited gripper calibration applied to the effector."""
    ee = np.zeros((effector.shape[0], 20), dtype=np.float32)  # the Beta state_end_effector layout the dataset reads
    ee[:, 0:2], ee[:, 16:18], ee[:, 18:20] = effector, head, waist
    ee = agibot_grippers.calibrated_gripper_fields(
        {"state_end_effector": ee}, hardware=hardware, state_layout="beta", state_width=None)["state_end_effector"]
    return assemble_embodiment_c_state(ee[:, 0:2], joint, ee[:, 16:18], ee[:, 18:20])


def unified_actions(states, robot_position, robot_orientation, pose_convention: str, normalizer):
    """Observations -> (native 29-D, unified raw 59-D, normalised 59-D, validity mask [59]); all torch float32."""
    native, extras = build_agibot_fk_action(
        states, fk_embodiment_type="embodiment_c_gripper", pose_convention=pose_convention,
        robot_position=robot_position, robot_orientation=robot_orientation)
    mapping = LanceAgiBotDataset.get_unified_action_mapping(None)  # does not use self
    raw, mask = mapping.to_unified(native)
    normalised = torch.where(mask, normalizer.normalize_action(raw), torch.zeros_like(raw))
    return native, raw, normalised, mask, extras


def check_sidecar_rows(norm: np.ndarray, domain: str = DOMAIN) -> None:
    """The release's own rule (inference/bimanual.py): slots without meaning for the domain are exactly zero."""
    assert norm.ndim == 2 and norm.shape[1] == UNIFIED_ACTION_DIM, norm.shape
    assert np.isfinite(norm).all(), "non-finite action"
    invalid = list(_invalid_action_slots(domain, UNIFIED_ACTION_DIM))
    assert not np.any(norm[:, invalid]), f"nonzero values in slots that are invalid for {domain}"


def decode_and_resize(src: Path, start: int, num_frames: int) -> np.ndarray:
    """Frames [start, start+num_frames) of a 640x480 clip, resized 640x480 -> 640x360 as in training: uint8 [T,360,640,3]."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames", "-of", "json", str(src)], check=True, capture_output=True, text=True).stdout)["streams"][0]
    w, h, n = int(probe["width"]), int(probe["height"]), int(probe["nb_read_frames"])
    assert (w, h) == (640, 480), f"expected a 640x480 head camera, got {w}x{h}"
    assert start + num_frames <= n, f"video has {n} frames, need {start + num_frames}"
    vf = f"select='between(n\\,{start}\\,{start + num_frames - 1})',setpts=N/30/TB"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", vf, "-frames:v", str(num_frames), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    video = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(num_frames, h, w, 3).copy()).permute(0, 3, 1, 2)
    out = resize_to_640x360(video)  # [T,3,360,640] uint8
    return out.permute(0, 2, 3, 1).contiguous().numpy()


def write_video(frames: np.ndarray, dst: Path, fps: int = 30) -> None:
    h, w = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", "-an", str(dst)], stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    assert p.returncode == 0, "ffmpeg encode failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, type=Path, help="proprio_stats.h5")
    ap.add_argument("--video", required=True, type=Path, help="head_color.mp4")
    ap.add_argument("--name", required=True)
    ap.add_argument("--task-id", required=True, type=int, help="AgiBot task id (selects the audited gripper hardware)")
    ap.add_argument("--gripper-hardware", choices=["robotiq", "omnipicker"], default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--start", type=int, default=0, help="first frame index of the window")
    ap.add_argument("--num-actions", type=int, default=900, help="N action rows -> N+1 video frames")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance", type=float, default=3.0, help="as in the published rows")
    ap.add_argument("--shift", type=float, default=5.0, help="as in the published rows")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    (args.out / "actions").mkdir(parents=True, exist_ok=True)
    normalizer, pose_convention, norm_info = load_release_normalizer()
    hardware = gripper_hardware(args.task_id, args.gripper_hardware)
    s = read_states(args.h5)
    T = s["joint"].shape[0]
    lo, hi = args.start, args.start + args.num_actions + 1
    assert hi <= T, f"episode has {T} proprio frames, need {hi}"
    states = fk_state(s["effector"][lo:hi], s["joint"][lo:hi], s["head"][lo:hi], s["waist"][lo:hi], hardware)
    native, raw, normalised, mask, _ = unified_actions(
        states, s["robot_position"][lo:hi], s["robot_orientation"][lo:hi], pose_convention, normalizer)
    norm = normalised.numpy().astype(np.float32)
    assert norm.shape == (args.num_actions, UNIFIED_ACTION_DIM), norm.shape
    check_sidecar_rows(norm)

    np.save(args.out / "actions" / f"{args.name}_raw_unified.npy", raw.numpy())
    (args.out / "actions" / f"{args.name}.json").write_text(json.dumps(
        {"action_space": "normalized_unified_v1", "domain_names": DOMAIN, "action": norm.tolist()}))

    info = {"name": args.name, "h5": args.h5.name, "task_id": args.task_id, "gripper_hardware": hardware, "start": args.start,
            "num_actions": args.num_actions, "proprio_frames": int(T), "normalisation": norm_info,
            "valid_slots": [int(i) for i in np.flatnonzero(mask.numpy())],
            "norm_stats": {"abs_max": float(np.abs(norm).max()), "frac_outside_unit": float((np.abs(norm) > 1).mean())}}
    if args.skip_video:
        vision_path = str(args.video.resolve())
    else:
        frames = decode_and_resize(args.video, args.start, args.num_actions + 1)
        clip = args.out / f"{args.name}_640x360.mp4"
        write_video(frames, clip)
        from PIL import Image
        png = args.out / f"{args.name}_initial.png"
        Image.fromarray(frames[0]).save(png)
        info["video"] = {"frames": int(frames.shape[0]), "size": [int(frames.shape[2]), int(frames.shape[1])],
                         "resize": "agibot_video.resize_to_640x360 (bicubic, antialias)"}
        vision_path = str(png.resolve())
    row = {"name": args.name, "model_mode": "forward_dynamics", "prompt": args.prompt, "vision_path": vision_path,
           "domain_name": DOMAIN, "action_condition": True, "raw_action_dim": UNIFIED_ACTION_DIM, "resolution": "480",
           "aspect_ratio": "16,9", "fps": 30, "num_frames": args.num_actions + 1, "action_chunk_size": args.num_actions,
           "num_steps": 4, "guidance": args.guidance, "shift": args.shift, "seed": args.seed, "autoregressive": True,
           "action_path": f"actions/{args.name}.json", "format_prompt_as_json": False}
    (args.out / f"{args.name}.jsonl").write_text(json.dumps(row) + "\n")
    (args.out / f"{args.name}_convert_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"name": args.name, "hardware": hardware, "rows": int(norm.shape[0]), "abs_max": info["norm_stats"]["abs_max"]}))


if __name__ == "__main__":
    main()
