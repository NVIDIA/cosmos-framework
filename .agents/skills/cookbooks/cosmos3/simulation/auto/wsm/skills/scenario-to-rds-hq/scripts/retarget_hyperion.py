# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Retarget a validated RDS-HQ clip to the bundled eleven-camera Hyperion rig.

Vehicle poses must already use the clip's world frame and pose timeline, with
FLU axes and the origin at the rear-axle ground projection. The saved calibration
uses 1924x1084 for seven cameras and 1720x1080 for the four fisheyes; this helper
does not resize or fit intrinsics.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rds_hq_io import MAP_GEOMETRY, check_clip_id, check_transform, numeric_array, require, validate_clip, write_archive

RIG_PATH: Path = Path(__file__).resolve().parents[1] / "assets" / "hyperion8.json"
ANNOTATION_LAYERS: tuple[str, ...] = ("all_object_info", *MAP_GEOMETRY, "3d_traffic_lights_lightnet_status")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rig() -> dict[str, dict[str, Any]]:
    """Load prepared [14] intrinsics and [4,4] camera-to-vehicle transforms."""
    rig = json.loads(RIG_PATH.read_text())
    require(isinstance(rig, dict) and len(rig) == 11, "The bundled rig must contain eleven cameras")
    for camera, calibration in rig.items():
        require(bool(re.fullmatch(r"camera_[a-z0-9_]+", camera)), f"Invalid camera name: {camera}")
        require("tele_sat" not in camera, "The satellite camera is excluded")
        require(set(calibration) == {"ftheta_intrinsics", "camera_to_vehicle"}, f"{camera}: unexpected rig fields")
        intrinsic = numeric_array(calibration["ftheta_intrinsics"], camera)  # [14]
        require(intrinsic.shape == (14,), f"{camera}: expected fourteen F-theta intrinsic values")
        width, height = (1720, 1080) if "fisheye" in camera else (1924, 1084)
        require(tuple(intrinsic[2:4]) == (width, height), f"{camera}: calibration must already be {width}x{height}")
        require(intrinsic[10] == 1, f"{camera}: expected a backward polynomial")
        check_transform(calibration["camera_to_vehicle"], camera)
    return rig


def retarget_clip(
    source_root: Path,
    output_root: Path,
    clip_id: str,
    vehicle_to_world: np.ndarray,  # [T,4,4]
    *,
    source_camera_type: str = "ftheta",
) -> dict[str, Any]:
    """Write new camera archives, preserve scene archives, and save a report.

    The source remains intact for native-camera validation and RGB comparisons.
    Additional source components remain there and are listed in the report.
    """
    check_clip_id(clip_id)
    require(source_root.resolve() != output_root.resolve(), "Use separate source and retargeted output roots")
    report_path = output_root / "retargeting" / f"{clip_id}.json"
    require(
        not any(output_root.glob(f"*/{clip_id}.tar")) and not report_path.exists(),
        "Output clip already exists; use a fresh root",
    )
    source_report = validate_clip(source_root, clip_id, source_camera_type)
    poses = numeric_array(vehicle_to_world, "vehicle_to_world")  # [T,4,4]
    require(
        poses.shape == (source_report["pose_frames"], 4, 4),
        "Vehicle poses must have shape [T,4,4] with one pose per source pose frame",
    )
    for index, pose in enumerate(poses):  # pose: [4,4]
        check_transform(pose, f"vehicle_to_world[{index}]")
    rig = load_rig()
    camera_members: dict[str, np.ndarray] = {}  # each [4,4]
    intrinsic_members: dict[str, np.ndarray] = {}  # each [14]
    for camera, calibration in rig.items():
        extrinsic = np.asarray(calibration["camera_to_vehicle"], dtype=np.float64)  # [4,4]
        camera_poses = poses @ extrinsic  # [T,4,4], camera -> vehicle -> world
        camera_members.update({f"{index:06d}.pose.{camera}.npy": pose for index, pose in enumerate(camera_poses)})
        intrinsic_members[f"ftheta_intrinsic.{camera}.npy"] = np.asarray(
            calibration["ftheta_intrinsics"], dtype=np.float64
        )  # [14]

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hyperion-", dir=output_root) as temporary:
        staged_root = Path(temporary)
        write_archive(staged_root / "pose" / f"{clip_id}.tar", clip_id, camera_members)
        write_archive(staged_root / "ftheta_intrinsic" / f"{clip_id}.tar", clip_id, intrinsic_members)
        annotation_hashes: dict[str, str] = {}
        for layer in ANNOTATION_LAYERS:
            source = source_root / layer / f"{clip_id}.tar"
            staged = staged_root / layer / f"{clip_id}.tar"
            annotation_hashes[layer] = sha256(source)
            staged.parent.mkdir()
            shutil.copyfile(source, staged)
            require(sha256(staged) == annotation_hashes[layer], f"{layer}: annotations changed during copy")
        report = validate_clip(staged_root, clip_id, camera_type="ftheta")
        report.update(
            {
                "source_root": str(source_root.resolve()),
                "source_camera_type": source_camera_type,
                "source_cameras": source_report["cameras"],
                "output_root": str(output_root.resolve()),
                "rig": RIG_PATH.name,
                "rig_sha256": sha256(RIG_PATH),
                "pose_composition": "vehicle_to_world @ camera_to_vehicle",
                "timeline": "Source pose indices and sample rate are unchanged",
                "annotation_sha256": annotation_hashes,
                "source_only_components": sorted(
                    path.parent.name
                    for path in source_root.glob(f"*/{clip_id}.tar")
                    if path.parent.name not in {*ANNOTATION_LAYERS, "pose", f"{source_camera_type}_intrinsic"}
                ),
                "render_overrides": [
                    "version=v4",
                    "camera_type=ftheta",
                    "camera_spec=internal_11cam",
                    "resolution_spec=source",
                ],
            }
        )
        staged_report = staged_root / "retargeting" / f"{clip_id}.json"
        staged_report.parent.mkdir()
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        files = [Path(layer) / f"{clip_id}.tar" for layer in ("pose", "ftheta_intrinsic", *ANNOTATION_LAYERS)]
        files.append(staged_report.relative_to(staged_root))
        installed: list[Path] = []
        try:
            for relative in files:
                destination = output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Publish complete files without overwriting a concurrent conversion.
                os.link(staged_root / relative, destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink()
            raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("clip_id")
    parser.add_argument("--vehicle-poses", type=Path, required=True, help="NPY array [T,4,4] of vehicle-to-world poses")
    parser.add_argument("--source-camera-type", choices=("pinhole", "ftheta"), default="ftheta")
    args = parser.parse_args()
    vehicle_poses = np.load(args.vehicle_poses, allow_pickle=False)  # [T,4,4]
    report = retarget_clip(
        args.source_root,
        args.output_root,
        args.clip_id,
        vehicle_poses,
        source_camera_type=args.source_camera_type,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
