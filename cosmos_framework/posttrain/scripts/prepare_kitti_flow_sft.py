# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Aligned KITTI raw front-camera/OXTS windows for Alpamayo trajectory SFT.

Pose convention follows KITTI's raw devkit: Mercator translation and Rz(yaw)
Ry(pitch)Rx(roll), then inverse-current-pose coordinates. This profile retains
KITTI's IMU origin/camera domain; it does not claim PhysicalAI calibration parity.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cosmos_framework.posttrain.utils.object_store import file_sha256


def timestamps(path):
    values = np.array(Path(path).read_text().splitlines(), dtype="datetime64[ns]").astype(np.int64)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f"KITTI timestamps must be nonempty and strictly increasing: {path}")
    return values


def oxts_poses(packets):
    packets = np.asarray(packets, dtype=np.float64)
    if packets.ndim != 2 or packets.shape[1] < 6 or not np.isfinite(packets).all():
        raise ValueError("OXTS packets require finite lat/lon/alt/roll/pitch/yaw")
    latitude, longitude, altitude = packets[:, :3].T
    if np.any(np.abs(latitude) >= 90) or np.any(np.abs(longitude) > 180):
        raise ValueError("OXTS latitude/longitude out of bounds")
    scale = np.cos(np.deg2rad(latitude[0]))
    earth_radius = 6378137.0
    positions = np.stack(
        (
            scale * earth_radius * np.deg2rad(longitude),
            scale * earth_radius * np.log(np.tan(np.deg2rad(90 + latitude) / 2)),
            altitude,
        ),
        axis=-1,
    )
    positions -= positions[0]
    rotations = Rotation.from_euler("xyz", packets[:, 3:6]).as_matrix()
    return positions, rotations


def aligned_window(pose_times, positions, rotations, *, current_time):
    """Interpolate to exact 10 Hz, using all six pose coordinates before rebasing."""
    target = current_time + np.arange(-15, 65, dtype=np.float64) * 0.1
    if target[0] < pose_times[0] or target[-1] > pose_times[-1]:
        raise ValueError("KITTI window requires 1.5s history and 6.4s future without extrapolation")
    xyz = np.stack([np.interp(target, pose_times, positions[:, axis]) for axis in range(3)], axis=-1)
    rot = Slerp(pose_times, Rotation.from_matrix(rotations))(target).as_matrix()
    origin, inverse = xyz[15].copy(), rot[15].T.copy()
    xyz = (xyz - origin) @ inverse.T
    rot = inverse[None] @ rot
    return xyz, rot


def prepare(source, out, *, training_drive, validation_drive, stride=2):
    if training_drive == validation_drive:
        raise ValueError("KITTI training and validation must use different drives")
    if stride < 1:
        raise ValueError("KITTI stride must be positive")
    root, destination = Path(source), Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "media").mkdir()
    inventory, counts, alignment = {}, {}, {}
    for split, drive in (("train", training_drive), ("validation", validation_drive)):
        directory = root / drive
        image_paths = sorted((directory / "image_02/data").glob("*.png"))
        pose_paths = sorted((directory / "oxts/data").glob("*.txt"))
        image_ns = timestamps(directory / "image_02/timestamps.txt")
        pose_ns = timestamps(directory / "oxts/timestamps.txt")
        if len(image_paths) != len(image_ns) or len(pose_paths) != len(pose_ns):
            raise ValueError(f"KITTI frame/OXTS inventory differs from timestamp count: {drive}")
        if any(p.stem != f"{i:010d}" for i, p in enumerate(image_paths)) or any(
            p.stem != f"{i:010d}" for i, p in enumerate(pose_paths)
        ):
            raise ValueError(f"KITTI synchronized frame/OXTS indices must be contiguous: {drive}")
        origin = int(min(image_ns[0], pose_ns[0]))
        image_times, pose_times = (image_ns - origin) * 1e-9, (pose_ns - origin) * 1e-9
        if np.max(np.diff(pose_times)) > 0.2 or np.max(np.diff(image_times)) > 0.2:
            raise ValueError(f"KITTI input has a camera/OXTS gap over 0.2s: {drive}")
        packets = [np.loadtxt(path) for path in pose_paths]
        positions, rotations = oxts_poses(packets)
        rows, max_alignment = [], 0.0
        for index in range(0, len(image_times), stride):
            time = image_times[index]
            if time - 1.5 < pose_times[0] or time + 6.4 > pose_times[-1]:
                continue
            desired = time + np.array([-0.3, -0.2, -0.1, 0])
            selected = np.abs(image_times[:, None] - desired[None]).argmin(axis=0)
            error = float(np.abs(image_times[selected] - desired).max())
            if error > 0.03 or len(set(selected)) != 4 or selected[-1] != index:
                raise ValueError(f"KITTI camera history cannot align within 30ms at {drive}:{index}")
            max_alignment = max(max_alignment, error)
            xyz, rot = aligned_window(pose_times, positions, rotations, current_time=time)
            media = []
            for frame in selected:
                name = f"media/{directory.name}-{int(frame):010d}.png"
                target = destination / name
                if not target.exists():
                    shutil.copyfile(image_paths[frame], target)
                    inventory[str(image_paths[frame].relative_to(root))] = file_sha256(str(image_paths[frame]))
                media.append(name)
            rows.append(
                {
                    "id": f"{directory.name}-{index:010d}",
                    "episode_id": directory.name,
                    "observation": {
                        "ego_history_xyz": xyz[:16].tolist(),
                        "ego_history_rot": rot[:16].tolist(),
                        "ego_future_rot": rot[16:].tolist(),
                    },
                    "actions": xyz[16:].tolist(),
                    "media": {"front": media},
                }
            )
        if len(rows) < 8:
            raise ValueError(f"KITTI {split} requires at least eight aligned windows, got {len(rows)}")
        (destination / (split + ".jsonl")).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        counts[split], alignment[split] = len(rows), max_alignment
        for path in [*pose_paths, directory / "oxts/timestamps.txt", directory / "image_02/timestamps.txt"]:
            inventory[str(path.relative_to(root))] = file_sha256(str(path))
    record = {
        "format": "cosmos-kitti-flow-v1",
        "source": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        "license": "CC-BY-NC-SA-3.0",
        "training_drive": training_drive,
        "validation_drive": validation_drive,
        "stride": stride,
        "counts": counts,
        "dt": 0.1,
        "history": 16,
        "future": 64,
        "camera_history": 4,
        "maximum_camera_alignment_seconds": alignment,
        "pose_interpolation": "linear-translation-SLERP-rotation",
        "coordinate_frame": "inverse-current-OXTS-pose; x-forward,y-left,z-up; IMU-origin",
        "camera": "KITTI-rectified-left-color-image_02; original RGB; model processor owns resizing",
        "domain": "KITTI sensor geometry, not PhysicalAI sensor calibration",
        "source_files_sha256": inventory,
    }
    (destination / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out", "training-drive", "validation-drive"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.out,
                training_drive=args.training_drive,
                validation_drive=args.validation_drive,
                stride=args.stride,
            )
        )
    )


if __name__ == "__main__":
    main()
