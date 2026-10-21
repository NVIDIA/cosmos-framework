# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``cosmos-auto-lidar-rangemap``: render HD-map / LiDAR range maps for RDS-HQ clips.

Example (HD-map control signal from RDS-HQ poses, no LiDAR read)::

    cosmos-auto-lidar-rangemap --input-root /data/rds_hq --output-root /data/out \
        --clip-id <clip_id>

Parity with the Cosmos-Drive-Dreams toolkit (recorded sweep poses, measured LiDAR map too)::

    cosmos-auto-lidar-rangemap --input-root ... --output-root ... --clip-id <clip_id> \
        --pose-source lidar_raw --render-lidar

Measured LiDAR range map only (reads nothing but lidar_raw/<clip>.tar)::

    cosmos-auto-lidar-rangemap --input-root ... --output-root ... --clip-id <clip_id> \
        --skip-hdmap --render-lidar
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

try:
    from cosmos_auto.lidar.hdmap_points import (
        DEFAULT_DENSITY_M,
        DEFAULT_MINIMAP_TYPES,
        DEFAULT_RADIUS_M,
    )
    from cosmos_auto.lidar.rangemap import DEFAULT_MAX_RANGE_M
    from cosmos_auto.lidar.render import RangeMapRenderConfig, render_clip
except ModuleNotFoundError as error:  # standalone renderer dependencies
    raise SystemExit(
        f"cosmos-auto-lidar-rangemap needs the standalone WSM dependencies ({error.name} is not installed): "
        "python -m pip install -e .agents/skills/cookbooks/cosmos3/simulation/auto/wsm"
    ) from error

# Only the strip video needs these; the range-map tars do not.
VIS_DEPENDENCIES = ("matplotlib", "imageio")
CLIP_DIRECTORIES = ("pose", "vehicle_pose")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def vis_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in VIS_DEPENDENCIES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="RDS-HQ root (component/<clip_id>.tar layout)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="where hdmap_rangemap/, lidar_rangemap/, lidar_rangemap_metadata/, render/ are written",
    )
    parser.add_argument(
        "--clip-id", action="append", default=[], help="clip id to render (repeatable)"
    )
    parser.add_argument(
        "--clip-list",
        type=str,
        default=None,
        help="text file with one clip id per line",
    )
    parser.add_argument(
        "--pose-source",
        choices=["rds_hq_pose", "lidar_raw"],
        default="rds_hq_pose",
        help="sweep poses for the HD-map map: RDS-HQ ego poses + rig extrinsic (default) or recorded lidar_raw poses",
    )
    parser.add_argument(
        "--camera-name",
        default="camera_front_wide_120fov",
        help="camera whose pose/ entries define the ego timeline",
    )
    parser.add_argument("--input-pose-fps", type=int, default=30)
    parser.add_argument("--lidar-fps", type=int, default=10)
    parser.add_argument(
        "--sensor-model",
        default="auto",
        help="auto | default | path to a lidar parameters JSON",
    )
    parser.add_argument(
        "--lidar-sensor-name",
        default=None,
        help="rig / sensor_collection lidar name (default: top P128)",
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=None,
        help="firing columns per revolution (default: from the parameters JSON, else 3600)",
    )
    parser.add_argument(
        "--allow-default-extrinsic",
        action="store_true",
        help="fall back to the nominal top-lidar mount when no rig data exists",
    )
    parser.add_argument(
        "--skip-hdmap", action="store_true", help="do not render the HD-map range map"
    )
    parser.add_argument(
        "--export-control",
        action="store_true",
        help="also write physical FP32 frames[3,T,128,1800] lidar_control.safetensors",
    )
    parser.add_argument(
        "--render-lidar",
        action="store_true",
        help="also render the measured LiDAR range map from lidar_raw/",
    )
    parser.add_argument(
        "--minimap-types", nargs="+", default=list(DEFAULT_MINIMAP_TYPES)
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY_M,
        help="point spacing along map geometry (m)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_M,
        help="cylinder radius around polylines (m)",
    )
    parser.add_argument(
        "--fill-in-road",
        action="store_true",
        help="add an estimated drivable surface between road boundaries",
    )
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="thin the map at the sweep-start pose before the shutter solve (several times faster per sweep, "
        "but drops 10-24%% of cells under ego motion); off by default for an exact render",
    )
    parser.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="render only the first N sweeps (debugging)",
    )
    parser.add_argument(
        "--no-vis", action="store_true", help="skip the render/<clip>.mp4 strip video"
    )
    parser.add_argument("--cmap", default="Spectral")
    parser.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def resolve_clip_ids(args: argparse.Namespace) -> list[str]:
    clip_ids = list(args.clip_id)
    if args.clip_list:
        clip_ids += [
            line.strip().replace(".tar", "")
            for line in Path(args.clip_list).read_text().splitlines()
            if line.strip()
        ]
    if not clip_ids:
        # Ego poses may come from either component (see pose_source.load_vehicle_poses).
        found = {
            p.stem
            for name in CLIP_DIRECTORIES
            for p in (Path(args.input_root) / name).glob("*.tar")
        }
        clip_ids = sorted(found)
    if not clip_ids:
        raise SystemExit(
            "no clips given (--clip-id / --clip-list) and none found under "
            + " or ".join(f"<input_root>/{name}" for name in CLIP_DIRECTORIES)
        )
    return clip_ids


def config_from_args(args: argparse.Namespace) -> RangeMapRenderConfig:
    kwargs = dict(
        input_root=args.input_root,
        output_root=args.output_root,
        pose_source=args.pose_source,
        camera_name=args.camera_name,
        input_pose_fps=args.input_pose_fps,
        lidar_fps=args.lidar_fps,
        sensor_model=args.sensor_model,
        lidar_sensor_name=args.lidar_sensor_name,
        n_columns=args.n_columns,
        allow_default_extrinsic=args.allow_default_extrinsic,
        render_hdmap=not args.skip_hdmap,
        render_lidar=args.render_lidar,
        minimap_types=tuple(args.minimap_types),
        density=args.density,
        radius=args.radius,
        fill_in_road=args.fill_in_road,
        prefilter=args.prefilter,
        max_range=args.max_range,
        max_frames=args.max_frames,
        export_control=args.export_control,
        vis=not args.no_vis,
        cmap=args.cmap,
        verbose=not args.quiet,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    return RangeMapRenderConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.vis and not vis_dependencies_available():
        print(
            f"[cosmos-auto-lidar-rangemap] {' and '.join(VIS_DEPENDENCIES)} are not installed; skipping the strip "
            "video (install '.[lidar-vis]' or pass --no-vis to silence this)",
            file=sys.stderr,
        )
        cfg.vis = False
    clip_ids = resolve_clip_ids(args)
    failures = 0
    for clip_id in clip_ids:
        try:
            result = render_clip(clip_id, cfg)
            print(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "frames": len(result.frame_ids),
                        "outputs": result.outputs,
                        "provenance": result.provenance,
                    }
                )
            )
        except (
            Exception
        ) as error:  # keep going through the clip list, report at the end
            failures += 1
            print(f"[{clip_id}] FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
