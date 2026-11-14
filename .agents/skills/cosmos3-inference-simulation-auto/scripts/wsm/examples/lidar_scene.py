# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Create a tiny synthetic RDS-HQ scene with an explicit camera/LiDAR rig.

No measured RGB or LiDAR is synthesized. Use fresh roots for changed geometry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

CAMERA = "camera_front_wide_120fov"
CLIP_ID = "synthetic_drive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pole-x", type=float, default=25.0)
    parser.add_argument(
        "--speed", type=float, default=2.0, help="ego speed in metres/second"
    )
    args = parser.parse_args()
    helper = (
        Path(__file__).resolve().parents[1]
        / "skills/scenario-to-rds-hq/scripts/rds_hq_io.py"
    )
    spec = importlib.util.spec_from_file_location("scene_io", helper)
    assert spec is not None and spec.loader is not None
    scene_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene_io)
    frames = 202  # 201 camera frames plus the end pose of the 67th full sweep
    vehicle = np.tile(np.eye(4), (frames, 1, 1))
    vehicle[:, 0, 3] = np.arange(frames) * args.speed / 30
    camera = np.eye(4)
    camera[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    camera[:3, 3] = [1.8, 0, 1.3]
    # Synthetic equidistant F-theta camera: angle = pixel radius / 600.
    intrinsic = np.array([640, 360, 1280, 720, 0, 1 / 600, 0, 0, 0, 0, 1])
    layers = {layer: {"labels": []} for layer in scene_io.MAP_GEOMETRY}
    layers["3d_poles"]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "polyline3d": {
                        "vertices": [[args.pole_x, 0, 0], [args.pole_x, 0, 4]]
                    }
                }
            }
        }
    ]
    layers["3d_road_boundaries"]["labels"] = [
        {
            "labelData": {
                "shape3d": {
                    "polyline3d": {"vertices": [[-5, y, 0], [60, y, 0]]},
                    "attributes": [{"name": "boundary_type", "text": "curb"}],
                }
            }
        }
        for y in (-3, 3)
    ]
    car = np.eye(4)
    car[:3, 3] = [35, -2, 0.8]
    objects = [
        {
            "car": {
                "object_type": "Car",
                "object_is_moving": False,
                "object_lwh": [4.5, 1.8, 1.6],
                "object_to_world": car.tolist(),
            }
        }
        for _ in range(frames)
    ]
    scene_io.write_clip(
        args.output_root,
        CLIP_ID,
        {CAMERA: vehicle @ camera},
        {CAMERA: intrinsic},
        objects,
        layers,
        {"state_to_legacy_render_state": {}, "traffic_light_states": {}},
        camera_type="ftheta",
    )
    rig_dir = args.output_root / "rig"
    rig_dir.mkdir(exist_ok=True)
    # Explicit synthetic mounting, not calibration claimed for a recorded vehicle.
    rig = {
        "rig": {
            "sensors": [
                {
                    "name": "lidar:gt:top:p128",
                    "nominalSensor2Rig_FLU": {
                        "roll-pitch-yaw": [0, 0, 0],
                        "t": [1.1, 0, 1.9],
                    },
                },
            ]
        }
    }
    (rig_dir / f"{CLIP_ID}.json").write_text(json.dumps(rig, indent=2) + "\n")
    # Vehicle poses remove any ambiguity from camera-to-ego conventions.
    scene_io.write_archive(
        args.output_root / "vehicle_pose" / f"{CLIP_ID}.tar",
        CLIP_ID,
        {f"{i:06d}.vehicle_pose.npy": pose for i, pose in enumerate(vehicle)},
    )
    # Desired appearance for this synthetic example, not a caption of a recording.
    captions = args.output_root / "captions"
    captions.mkdir(exist_ok=True)
    (captions / f"{CAMERA}.json").write_text(
        json.dumps(
            {
                "caption_structured": {
                    "chunk_0_201": {
                        "caption": "Clear daylight on a dry straight road. Two curb boundaries flank the road, "
                        "with an upright pole ahead and a stationary car ahead to its right.",
                        "start_frame": 0,
                        "end_frame": 201,
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            scene_io.validate_clip(args.output_root, CLIP_ID, camera_type="ftheta")
        )
    )


if __name__ == "__main__":
    main()
