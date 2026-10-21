# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Where the per-sweep LiDAR poses come from.

The HD-map range map is the control signal for the Cosmos LiDAR generator. The
default source builds it from RDS-HQ ego poses and the rig calibration so it
can be produced for clips without LiDAR. Two sources exist:

``rds_hq_pose`` (default)
    Ego poses from the RDS-HQ ``vehicle_pose`` component (or, when absent, from
    the camera ``pose`` component via the documented convention
    ``vehicle_to_world[0] == I``), composed with the rig LiDAR extrinsic from
    ``rig/<clip>.json`` or ``sensor_collection/<clip>.txt``. Sweeps start at every
    ``input_pose_fps / lidar_fps``-th pose frame and end at the next one.

``lidar_raw``
    The recorded ``lidar_to_world`` / ``starting_timestamp`` of every sweep in
    ``lidar_raw/<clip>.tar``. This is the original toolkit behaviour and is kept
    for parity checks and for rendering the *measured* LiDAR range map, which
    needs the recorded sweep poses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_auto.lidar.sensor_model import (
    RIG_LIDAR_SENSOR_PREFERENCE,
    iter_lidar_blocks,
    parse_sensor_to_vehicle_transform,
    read_sensor_collection_text,
    select_lidar_block,
)
from cosmos_auto.wm.wds_utils import get_sample

PoseSource = Literal["rds_hq_pose", "lidar_raw"]

# Nominal mount of the rig's top LiDAR relative to the vehicle (rear-axle
# ground projection, FLU). Only used when a clip ships neither a rig file nor a
# sensor collection; callers must opt in.
DEFAULT_LIDAR_TO_VEHICLE = np.array(
    [
        [1.0, 0.0, 0.0, 1.14],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.94],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_FRAME_RE = re.compile(r"(\d{6})\.")

# RDS-HQ camera poses are OpenCV (x right, y down, z forward); rig extrinsics are FLU
# (x forward, y left, z up). Columns of this matrix are the OpenCV axes expressed in FLU.
FLU_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


@dataclass
class SweepPoses:
    """Sensor poses bracketing each sweep.

    Attributes:
        frame_ids: pose-frame index at which each sweep starts (``000000``, ``000003``, ...).
        lidar_to_world_start / lidar_to_world_end: ``[T, 4, 4]`` sensor-to-world at sweep start/end.
        start_timestamps_us / end_timestamps_us: ``[T]`` int64 sweep bounds. Only
            differences matter; the epoch follows the source.
        source: which :data:`PoseSource` produced the poses.
    """

    frame_ids: list[int]
    lidar_to_world_start: np.ndarray
    lidar_to_world_end: np.ndarray
    start_timestamps_us: np.ndarray
    end_timestamps_us: np.ndarray
    source: str

    def __post_init__(self) -> None:
        count = len(self.frame_ids)
        for name in ("lidar_to_world_start", "lidar_to_world_end"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, 4, 4):
                raise ValueError(
                    f"{name} must have shape ({count}, 4, 4), got {value.shape}"
                )
            setattr(self, name, value)
        for name in ("start_timestamps_us", "end_timestamps_us"):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {value.shape}"
                )
            setattr(self, name, value)
        if np.any(self.end_timestamps_us <= self.start_timestamps_us):
            raise ValueError("every sweep must end after it starts")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def ego_positions(self) -> np.ndarray:
        """``[T, 3]`` sensor positions at sweep start (used to seed road-surface estimation)."""
        return self.lidar_to_world_start[:, :3, 3]


# ----------------------------------------------------------------------------- extrinsics
def rotation_from_rpy_deg(roll_pitch_yaw_deg: Sequence[float]) -> np.ndarray:
    """Rig-file roll/pitch/yaw (degrees, applied about fixed x, y, z) to a rotation matrix."""
    return Rotation.from_euler(
        "xyz", np.asarray(roll_pitch_yaw_deg, dtype=np.float64), degrees=True
    ).as_matrix()


def select_rig_sensor(
    rig: dict, sensor_name: str | None = None, prefix: str = "lidar:"
) -> dict:
    """Pick a sensor entry from a rig JSON by exact name or by :data:`RIG_LIDAR_SENSOR_PREFERENCE`.

    The same preference list drives :func:`cosmos_auto.lidar.sensor_model.select_lidar_block`,
    so the rig-JSON and sensor_collection paths agree on which lidar they describe.
    """
    sensors = rig.get("rig", rig).get("sensors", [])
    by_name = {sensor.get("name"): sensor for sensor in sensors if sensor.get("name")}
    if sensor_name is not None:
        if sensor_name not in by_name:
            raise KeyError(
                f"sensor {sensor_name!r} not in rig; available: {sorted(by_name)}"
            )
        return by_name[sensor_name]
    for candidate in RIG_LIDAR_SENSOR_PREFERENCE:
        if candidate in by_name:
            return by_name[candidate]
    matching = sorted(name for name in by_name if name.startswith(prefix))
    if len(matching) == 1:
        return by_name[matching[0]]
    raise KeyError(
        f"could not pick a {prefix!r} sensor from rig; candidates: {matching}"
    )


def sensor_to_vehicle_from_rig(
    rig: dict | str | Path, sensor_name: str | None = None
) -> tuple[np.ndarray, str]:
    """Full sensor-to-vehicle extrinsic from an RDS-HQ ``rig/<clip>.json``.

    ``nominalSensor2Rig_FLU`` gives the nominal mount; ``correction_sensor_R_FLU``
    is a small rotation applied in the sensor frame (right-multiplied) and
    ``correction_rig_T`` a translation offset in the rig frame. Verified against
    the recorded ``lidar_to_world`` of RDS-HQ clips (rotation residual < 0.005 deg).

    Returns ``(transform [4, 4], sensor_name)``.
    """
    rig_dict: dict
    if isinstance(rig, dict):
        rig_dict = rig
    else:
        with open(rig) as handle:
            rig_dict = json.load(handle)
    sensor = select_rig_sensor(rig_dict, sensor_name)
    nominal = sensor["nominalSensor2Rig_FLU"]
    rotation = rotation_from_rpy_deg(nominal["roll-pitch-yaw"])
    correction = sensor.get("correction_sensor_R_FLU")
    if correction and "roll-pitch-yaw" in correction:
        rotation = rotation @ rotation_from_rpy_deg(correction["roll-pitch-yaw"])
    translation = np.asarray(nominal["t"], dtype=np.float64) + np.asarray(
        sensor.get("correction_rig_T", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, str(sensor["name"])


def sensor_to_vehicle_from_sensor_collection(
    path_or_text: str | Path, sensor_name: str | None = None
) -> tuple[np.ndarray, str]:
    """LiDAR ``sensor_to_vehicle_transform`` (axis-angle + translation) from ``sensor_collection/<clip>.txt``.

    The lidar block is chosen with the same rules as
    :meth:`cosmos_auto.lidar.sensor_model.SpinningLidarModel.from_sensor_collection`
    (exact ``sensor_name``, else :data:`RIG_LIDAR_SENSOR_PREFERENCE`, else the
    single block present); the extrinsic is read only from inside that block's
    ``sensor_to_vehicle_transform`` sub-message.

    Returns ``(transform [4, 4], sensor_name)`` like :func:`sensor_to_vehicle_from_rig`.
    """
    text = read_sensor_collection_text(path_or_text)
    name, block = select_lidar_block(iter_lidar_blocks(text), sensor_name)
    return parse_sensor_to_vehicle_transform(block), name


def rig_camera_name(camera_name: str) -> str:
    """RDS-HQ camera key (``camera_front_wide_120fov``) -> rig sensor name (``camera:front:wide:120fov``)."""
    return camera_name.replace("_", ":")


def camera_to_vehicle_from_rig(rig: dict | str | Path, camera_name: str) -> np.ndarray:
    """OpenCV camera-to-vehicle extrinsic of an RDS-HQ camera from ``rig/<clip>.json``.

    The rig stores the camera mount in FLU axes; the result is rotated into the
    OpenCV camera frame used by the ``pose`` component, so that
    ``vehicle_to_world = camera_to_world @ inv(camera_to_vehicle)``.
    """
    transform, _ = sensor_to_vehicle_from_rig(rig, rig_camera_name(camera_name))
    transform = transform.copy()
    transform[:3, :3] = transform[:3, :3] @ FLU_FROM_OPENCV
    return transform


def load_lidar_to_vehicle(
    input_root: str | Path,
    clip_id: str,
    sensor_name: str | None = None,
    allow_default: bool = False,
) -> tuple[np.ndarray, str]:
    """Resolve the LiDAR extrinsic for a clip: rig JSON, then sensor collection, then the default.

    Returns ``(lidar_to_vehicle [4, 4], provenance)`` where provenance is
    ``"rig:<sensor_name>"``, ``"sensor_collection:<sensor_name>"`` or ``"default"``.
    """
    rig_path = Path(input_root) / "rig" / f"{clip_id}.json"
    if rig_path.exists():
        transform, name = sensor_to_vehicle_from_rig(rig_path, sensor_name)
        return transform, f"rig:{name}"
    collection_path = Path(input_root) / "sensor_collection" / f"{clip_id}.txt"
    if collection_path.exists():
        transform, name = sensor_to_vehicle_from_sensor_collection(
            collection_path, sensor_name
        )
        return transform, f"sensor_collection:{name}"
    if allow_default:
        return DEFAULT_LIDAR_TO_VEHICLE.copy(), "default"
    raise FileNotFoundError(
        f"no rig/{clip_id}.json or sensor_collection/{clip_id}.txt under {input_root}; "
        "pass allow_default=True to use the nominal top-LiDAR mount"
    )


# ----------------------------------------------------------------------------- ego poses
def _frame_index(key: str) -> int:
    match = _FRAME_RE.search(key)
    if match is None:
        raise ValueError(f"cannot parse a six-digit frame id from {key!r}")
    return int(match.group(1))


def load_vehicle_poses(
    input_root: str | Path,
    clip_id: str,
    camera_name: str = "camera_front_wide_120fov",
    vehicle_pose_dir: str = "vehicle_pose",
    pose_dir: str = "pose",
) -> tuple[np.ndarray, str]:
    """``[T, 4, 4]`` vehicle-to-world poses for every pose frame.

    Sources, in order:

    1. ``<vehicle_pose_dir>/<clip>.tar`` (``<frame>.vehicle_pose.npy``).
    2. Camera poses in ``<pose_dir>/<clip>.tar`` combined with the camera's extrinsic
       from ``rig/<clip>.json``: ``vehicle_to_world[f] = camera_to_world[f] @
       inv(camera_to_vehicle)``. Exact for any world frame (matches ``vehicle_pose``
       to float precision on RDS-HQ exports).
    3. Camera poses alone with the RDS-HQ convention that the vehicle frame is the
       identity at frame 0: ``camera_to_world[f] @ inv(camera_to_world[0])``. A
       dataset that keeps a non-identity initial vehicle pose is then offset by that
       pose relative to the map geometry, so a warning is printed.

    Returns ``(poses, provenance)`` with provenance ``"vehicle_pose"``,
    ``"pose+rig:<camera>"`` or ``"pose:<camera>"``.
    """
    vehicle_tar = Path(input_root) / vehicle_pose_dir / f"{clip_id}.tar"
    if vehicle_tar.exists():
        sample = get_sample(str(vehicle_tar))
        keys = sorted(
            (k for k in sample if k.endswith(".vehicle_pose.npy")), key=_frame_index
        )
        if not keys:
            raise ValueError(f"{vehicle_tar} has no vehicle_pose members")
        poses = np.stack([np.asarray(sample[k], dtype=np.float64) for k in keys])
        _check_contiguous([_frame_index(k) for k in keys], str(vehicle_tar))
        return poses, "vehicle_pose"

    pose_tar = Path(input_root) / pose_dir / f"{clip_id}.tar"
    if not pose_tar.exists():
        raise FileNotFoundError(f"neither {vehicle_tar} nor {pose_tar} exists")
    sample = get_sample(str(pose_tar))
    suffix = f".pose.{camera_name}.npy"
    keys = sorted((k for k in sample if k.endswith(suffix)), key=_frame_index)
    if not keys:
        raise KeyError(f"{pose_tar} has no poses for camera {camera_name!r}")
    camera_to_world = np.stack([np.asarray(sample[k], dtype=np.float64) for k in keys])
    _check_contiguous([_frame_index(k) for k in keys], str(pose_tar))
    rig_path = Path(input_root) / "rig" / f"{clip_id}.json"
    if rig_path.exists():
        try:
            camera_to_vehicle = camera_to_vehicle_from_rig(rig_path, camera_name)
        except KeyError:
            camera_to_vehicle = None
        if camera_to_vehicle is not None:
            return camera_to_world @ np.linalg.inv(camera_to_vehicle)[
                None
            ], f"pose+rig:{camera_name}"
    print(
        f"[{clip_id}] no vehicle_pose and no rig entry for {camera_name!r}: assuming the vehicle frame is the "
        "identity at frame 0 (RDS-HQ convention); a retained initial vehicle offset would shift the sweep poses"
    )
    vehicle_to_world = camera_to_world @ np.linalg.inv(camera_to_world[0])[None]
    return vehicle_to_world, f"pose:{camera_name}"


def _check_contiguous(frame_ids: list[int], what: str) -> None:
    expected = list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))
    if frame_ids != expected or frame_ids[0] != 0:
        raise ValueError(
            f"{what}: pose frames must be contiguous from 000000, got {frame_ids[:3]}...{frame_ids[-2:]}"
        )


def load_frame_timestamps_us(
    input_root: str | Path, clip_id: str, camera_name: str = "camera_front_wide_120fov"
) -> np.ndarray | None:
    """Per-frame timestamps (microseconds) from ``ftheta_<camera>_timestmap/<clip>.timestamps`` if present.

    The RDS-HQ export writes ``<frame>\\t<system_us>`` lines (directory name carries
    the historical ``timestmap`` spelling). Returns ``None`` when unavailable.
    """
    for dir_name in (
        f"ftheta_{camera_name}_timestmap",
        f"ftheta_{camera_name}_timestamp",
        f"{camera_name}_timestamp",
    ):
        path = Path(input_root) / dir_name / f"{clip_id}.timestamps"
        if path.exists():
            rows = [
                line.split("\t")
                for line in path.read_text().strip().splitlines()
                if line.strip()
            ]
            frames = np.asarray([int(r[0]) for r in rows])
            stamps = np.asarray([int(r[1]) for r in rows], dtype=np.int64)
            order = np.argsort(frames)
            return stamps[order]
    return None


# ----------------------------------------------------------------------------- sweep poses
def sweep_poses_from_vehicle_poses(
    vehicle_to_world: np.ndarray,
    lidar_to_vehicle: np.ndarray,
    input_pose_fps: int = 30,
    lidar_fps: int = 10,
    frame_timestamps_us: np.ndarray | None = None,
    frame_ids: Sequence[int] | None = None,
) -> SweepPoses:
    """Build sweep poses from ego poses: one sweep per ``input_pose_fps // lidar_fps`` frames.

    Sweep ``i`` starts at pose frame ``f = i * stride`` and ends at frame ``f + stride``
    (the start of the next sweep), matching the frame ids of recorded
    ``lidar_raw`` sweeps (``000000``, ``000003``, ...). The final partial sweep
    without a successor frame is dropped, as in the toolkit. Sweeps whose end
    timestamp does not advance past the start (duplicated or stalled camera
    timestamps) are skipped with one summary line, mirroring
    :func:`sweep_poses_from_lidar_raw`; only when no sweep survives is an error raised.

    Args:
        vehicle_to_world: ``[T, 4, 4]`` ego poses at ``input_pose_fps``.
        lidar_to_vehicle: ``[4, 4]`` rig extrinsic.
        frame_timestamps_us: optional ``[T]`` per-frame timestamps; synthetic
            ``frame / input_pose_fps`` timestamps are used when ``None``.
        frame_ids: optional explicit sweep start frames (default every ``stride``).
    """
    if input_pose_fps % lidar_fps != 0:
        raise ValueError(
            f"input_pose_fps={input_pose_fps} must be a multiple of lidar_fps={lidar_fps}"
        )
    stride = input_pose_fps // lidar_fps
    total = vehicle_to_world.shape[0]
    if frame_timestamps_us is None:
        frame_timestamps_us = np.round(
            np.arange(total) * (1.0e6 / input_pose_fps)
        ).astype(np.int64)
    frame_timestamps_us = np.asarray(frame_timestamps_us, dtype=np.int64)
    if frame_timestamps_us.shape[0] < total:
        raise ValueError("frame_timestamps_us must cover every pose frame")
    candidates = [
        int(f)
        for f in (frame_ids if frame_ids is not None else range(0, total, stride))
        if f + stride < total
    ]
    if not candidates:
        raise ValueError(
            f"clip has {total} pose frames; need more than {stride} for one sweep"
        )
    starts = [
        f
        for f in candidates
        if frame_timestamps_us[f + stride] > frame_timestamps_us[f]
    ]
    skipped = [
        f
        for f in candidates
        if frame_timestamps_us[f + stride] <= frame_timestamps_us[f]
    ]
    if skipped:
        print(
            f"sweep_poses_from_vehicle_poses: skipped {len(skipped)}/{len(candidates)} sweeps whose end "
            f"timestamp does not advance; start frames {skipped}"
        )
    if not starts:
        raise ValueError(
            f"all {len(candidates)} sweeps have non-advancing timestamps; no usable sweep"
        )
    ends = [f + stride for f in starts]
    lidar_to_vehicle = np.asarray(lidar_to_vehicle, dtype=np.float64)
    return SweepPoses(
        frame_ids=starts,
        lidar_to_world_start=vehicle_to_world[starts] @ lidar_to_vehicle[None],
        lidar_to_world_end=vehicle_to_world[ends] @ lidar_to_vehicle[None],
        start_timestamps_us=frame_timestamps_us[starts],
        end_timestamps_us=frame_timestamps_us[ends],
        source="rds_hq_pose",
    )


def sweep_poses_from_lidar_raw(lidar_sample: dict) -> SweepPoses:
    """Recorded sweep poses from a decoded ``lidar_raw`` WebDataset sample (toolkit behaviour).

    Uses each sweep's ``lidar_to_world`` and ``starting_timestamp`` and the next
    sweep's as the end pose/time. Sweeps whose successor does not advance in
    time, and the final sweep, are dropped.
    """
    keys = sorted(
        (k for k in lidar_sample if k.endswith(".lidar_raw.npz")), key=_frame_index
    )
    frame_ids: list[int] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    t_start: list[int] = []
    t_end: list[int] = []
    for current_key, next_key in zip(keys[:-1], keys[1:]):
        current, following = lidar_sample[current_key], lidar_sample[next_key]
        t0, t1 = (
            int(str(current["starting_timestamp"])),
            int(str(following["starting_timestamp"])),
        )
        if t1 <= t0:
            continue
        frame_ids.append(_frame_index(current_key))
        starts.append(np.asarray(current["lidar_to_world"], dtype=np.float64))
        ends.append(np.asarray(following["lidar_to_world"], dtype=np.float64))
        t_start.append(t0)
        t_end.append(t1)
    if not frame_ids:
        raise ValueError("lidar_raw sample has fewer than two usable sweeps")
    return SweepPoses(
        frame_ids=frame_ids,
        lidar_to_world_start=np.stack(starts),
        lidar_to_world_end=np.stack(ends),
        start_timestamps_us=np.asarray(t_start),
        end_timestamps_us=np.asarray(t_end),
        source="lidar_raw",
    )


def load_sweep_poses(
    input_root: str | Path,
    clip_id: str,
    source: PoseSource = "rds_hq_pose",
    camera_name: str = "camera_front_wide_120fov",
    input_pose_fps: int = 30,
    lidar_fps: int = 10,
    lidar_sensor_name: str | None = None,
    allow_default_extrinsic: bool = False,
    lidar_sample: dict | None = None,
) -> tuple[SweepPoses, dict[str, str]]:
    """Load sweep poses for a clip from the requested source.

    Returns ``(sweep_poses, provenance)`` where ``provenance`` records which files
    supplied the ego poses, extrinsic and timestamps.
    """
    if source == "lidar_raw":
        sample: dict = (
            lidar_sample
            if lidar_sample is not None
            else get_sample(str(Path(input_root) / "lidar_raw" / f"{clip_id}.tar"))
        )
        assert isinstance(sample, dict), f"lidar_raw sample for {clip_id} is not a dict"
        return sweep_poses_from_lidar_raw(sample), {
            "poses": "lidar_raw",
            "extrinsic": "lidar_raw",
        }
    if source != "rds_hq_pose":
        raise ValueError(f"unknown pose source {source!r}")
    vehicle_to_world, pose_provenance = load_vehicle_poses(
        input_root, clip_id, camera_name
    )
    lidar_to_vehicle, extrinsic_provenance = load_lidar_to_vehicle(
        input_root, clip_id, lidar_sensor_name, allow_default=allow_default_extrinsic
    )
    timestamps = load_frame_timestamps_us(input_root, clip_id, camera_name)
    sweeps = sweep_poses_from_vehicle_poses(
        vehicle_to_world,
        lidar_to_vehicle,
        input_pose_fps,
        lidar_fps,
        frame_timestamps_us=timestamps,
    )
    provenance = {
        "poses": pose_provenance,
        "extrinsic": extrinsic_provenance,
        "timestamps": "camera_timestamps"
        if timestamps is not None
        else f"synthetic_{input_pose_fps}hz",
    }
    return sweeps, provenance
