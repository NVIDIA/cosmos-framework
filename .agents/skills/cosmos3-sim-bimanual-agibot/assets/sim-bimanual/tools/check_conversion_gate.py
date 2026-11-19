#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Acceptance gate for convert_agibot_episode.py: run it before any rollout.

Inputs: the downloaded checkpoint (SIM_BIMANUAL_CHECKPOINT) and its training bundle
``examples/training/sim_bimanual_robot_example_v1.tar.zst`` extracted somewhere (``--bundle``). It needs the framework's
``train`` extra (the loader imports it): ``uv sync --locked --group cu130 --extra guardrail --extra train``.

Checks (every one must pass; the exit code is 0 only then):
  A. Loader oracle. For every raw record in the bundle, the conversion in convert_agibot_episode.py (state assembly, audited
     gripper calibration, FK, unified mapping, pinned global-asinh statistics, validity mask) is compared with the
     normalised 59-D action the framework's own data loader (``LocalSimulationDataset``, the loader the release ships for this
     bundle) produces for the same record. On the rows the loader labels with the ``agibotworld`` domain: max |difference|
     <= 1e-5 on the valid slots, exact zeros in the invalid slots and in the 5 padding channels.
  B. Published sidecars. ``assets/training/agibot/actions/agibot_{001,006,034}_normalized_unifiedv1.json``: shape, action
     space, exact zeros outside each row's domain mask (the release's own ``_invalid_action_slots``), and consistency with
     the pinned statistics (de-normalised rot6d halves have unit norm to 1e-3; gripper openness (a fraction of the 0.12 m width) lies in [0, 1]).
  C. The release's own input check, ``tools/check_bimanual_mixed_inputs.py --checkpoint-path`` (run by the caller; see the
     record) and the row-level check of the sidecars this tool writes (``check_sidecar_rows``).

What this gate cannot do: reproduce the three published sidecars from the bundle's records. The bundle holds three
different episodes (different scenes, different camera-mask spans); the published sidecars are sidecars of other
episodes. The comparison below prints that evidence (first-frame RMSE of the bundle clips against the published initial
images) instead of a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_agibot_episode as cv  # noqa: E402  (sets up sys.path for the framework)

from cosmos_framework.data.generator.action.action_binary import decode_action_bin  # noqa: E402
from cosmos_framework.inference.bimanual import _invalid_action_slots  # noqa: E402

TOL = 1e-5


def convert_record(record_bin: Path, hardware: str, normalizer, pose_convention: str) -> np.ndarray:
    """One ACT2 record of the bundle through the same functions the converter uses; normalised 59-D float32."""
    f = decode_action_bin(record_bin.read_bytes())
    ee = f["state_end_effector"]  # [T+1,20]: effector 0:2, head 16:18, waist 18:20
    states = cv.fk_state(ee[:, 0:2], f["state_joint"], ee[:, 16:18], ee[:, 18:20], hardware)
    robot = f["state_robot"]
    _, _, norm, _, _ = cv.unified_actions(states, robot[:, 0:3], robot[:, 3:7], pose_convention, normalizer)
    return norm.numpy()


def check_a(bundle: Path, normalizer, pose_convention: str) -> dict:
    from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset

    samples = json.loads((bundle / "samples.json").read_text())
    ds = LocalSimulationDataset(str(bundle), model="bimanual", stratum="agibot", seed=42, transform_overrides={"resolution": "480"})
    results = []
    for i, rec in enumerate(samples):
        key = next(k for k in rec["assets"] if k.endswith("actions.bin"))
        item = ds[i]
        loader = item["action"].float().numpy()  # [900,64], camera-masked rows projected to the camera domain
        agibot_rows = (item["domain_id"].numpy() == 15)
        best = None
        for hardware in ("robotiq", "omnipicker"):
            mine = convert_record(bundle / rec["assets"][key], hardware, normalizer, pose_convention)
            valid = np.ones(64, bool); valid[list(_invalid_action_slots(cv.DOMAIN, 59))] = False; valid[59:] = False
            diff = np.abs(mine[agibot_rows][:, valid[:59]] - loader[agibot_rows][:, :59][:, valid[:59]]).max()
            zeros = (not np.any(mine[:, ~valid[:59]])) and (not np.any(loader[agibot_rows][:, ~valid]))
            if best is None or diff < best["max_abs_diff_valid"]:
                best = {"identity": rec["identity"], "hardware": hardware, "max_abs_diff_valid": float(diff), "exact_zeros_elsewhere": bool(zeros),
                        "agibotworld_rows": int(agibot_rows.sum()), "camera_pose_rows": int((~agibot_rows).sum())}
        best["pass"] = best["max_abs_diff_valid"] <= TOL and best["exact_zeros_elsewhere"]
        results.append(best)
    return {"records": results, "pass": all(r["pass"] for r in results)}


def first_frame(path: Path, w: int = 640, h: int = 360) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float64)


def check_b(checkpoint: Path, bundle: Path, normalizer) -> dict:
    from PIL import Image

    out = {"sidecars": [], "first_frame_rmse_bundle_vs_published": {}}
    samples = json.loads((bundle / "samples.json").read_text())
    bundle_frames = [first_frame(bundle / next(v for k, v in rec["assets"].items() if k.endswith("video.mp4"))) for rec in samples]
    for name in ("001", "006", "034"):
        p = json.loads((checkpoint / "assets/training/agibot/actions" / f"agibot_{name}_normalized_unifiedv1.json").read_text())
        a = np.asarray(p["action"], dtype=np.float32)
        domains = p["domain_names"] if isinstance(p["domain_names"], list) else [p["domain_names"]] * a.shape[0]
        ok_zero = all(not np.any(a[r, list(_invalid_action_slots(d, 59))]) for r, d in enumerate(domains))
        rows = np.array([d == "agibotworld" for d in domains])
        raw = normalizer.denormalize_action(torch.from_numpy(a)).numpy()
        r6 = raw[rows][:, [12, 13, 14, 15, 16, 17]]
        unit = float(max(abs(np.linalg.norm(r6[:, :3], axis=1) - 1).max(), abs(np.linalg.norm(r6[:, 3:], axis=1) - 1).max()))
        grip = raw[rows][:, [33, 58]]
        out["sidecars"].append({"name": name, "shape": list(a.shape), "action_space": p["action_space"], "agibotworld_rows": int(rows.sum()),
                                "camera_pose_rows": int((~rows).sum()), "exact_zeros_outside_domain_mask": bool(ok_zero),
                                "rot6d_unit_norm_max_dev": unit, "openness_min": float(grip.min()), "openness_max": float(grip.max())})
        img = np.asarray(Image.open(checkpoint / f"assets/training/agibot/images/agibot_{name}_initial.png").convert("RGB").resize((640, 360))).astype(np.float64)
        out["first_frame_rmse_bundle_vs_published"][name] = [round(float(np.sqrt(((img - b) ** 2).mean())), 1) for b in bundle_frames]
    s = out["sidecars"]
    out["pass"] = all(x["exact_zeros_outside_domain_mask"] and x["rot6d_unit_norm_max_dev"] < 1e-3 and -1e-3 <= x["openness_min"] and x["openness_max"] <= 1 + 1e-3 for x in s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path, help="extracted sim_bimanual_robot_example_v1 directory")
    ap.add_argument("--out", type=Path, default=Path("conversion_gate.json"))
    args = ap.parse_args()
    normalizer, pose_convention, info = cv.load_release_normalizer()
    result = {"normalisation": info, "A_loader_oracle": check_a(args.bundle, normalizer, pose_convention),
              "B_published_sidecars": check_b(cv.CHECKPOINT, args.bundle, normalizer)}
    result["pass"] = result["A_loader_oracle"]["pass"] and result["B_published_sidecars"]["pass"]
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
