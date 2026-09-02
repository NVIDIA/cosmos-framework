# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Camera-trajectory primitives shared by evaluation and inference.

These were extracted verbatim from ``evaluation/action/eval_camera.py`` so the
inference frontends -- both the native Imaginaire4 one and the released
cosmos-framework one -- can build camera actions without importing the
evaluation harness, which drags in OmegaConf, the sharded camera dataset, and
the visualization stack.

Keeping a single copy here is what makes native and public camera conditioning
bit-identical: both runtimes execute this module, not two transcriptions of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

__all__ = [
    "CameraTrainParams",
    "load_custom_pose_jsons",
    "parse_camera_string",
    "parse_custom_trajectories",
    "trajectory_to_action",
]


@dataclass
class CameraTrainParams:
    """Alignment-critical parameters extracted from a CameraDatasetSharded config."""

    rotation_format: str
    pose_convention: str
    translation_scale: float
    rotation_scale: float
    num_frames: int


def parse_camera_string(
    camera_string: str,
    forward_speed: float = 0.1,  # per frame
    yaw_speed: float = np.deg2rad(0.5),
    pitch_speed: float = np.deg2rad(0.5),
    roll_speed: float = np.deg2rad(0.5),
) -> np.ndarray:
    """
    Parses a camera string into a sequence of camera-to-world poses.
    Format: "<cmd>-<frame>,<cmd>-<frame>,...|<cmd>-<frame>,<cmd>-<frame>,..."
    Example: "w-46,right-46"

    cmd (translation):
    - w: forward (+Z)
    - s: backward (-Z)
    - a: left (-X)
    - d: right (+X)
    - u: up (-Y)
    - n: down (+Y)

    cmd (rotation):
    - up: pitch up
    - down: pitch down
    - left: yaw left
    - right: yaw right
    - cw: roll clockwise
    - ccw: roll counter-clockwise

    cmd (special):
    - stay: hold current pose (no movement)
    - pano: 360° yaw rotation (panorama, rotates right) over the given frames
    - orbit: 360° orbit around a point in front of the current camera over the
        given frames. Moves BOTH camera position and rotation (unlike `pano`,
        which only rotates). Default: counter-clockwise when viewed from above,
        orbit radius = 1.0. Optional radius override: ``orbit-<frames>-<radius>``
        (e.g. ``orbit-150-5.0``).

    frame: duration in number of frames

    Multiple trajectories are separated by '|'.
    """
    commands = [x.strip() for x in camera_string.split(",")]

    # Initial pose - Identity
    current_pose = np.eye(4)
    poses = [current_pose.copy()]

    for cmd in commands:
        if "-" not in cmd:
            continue
        parts = cmd.split("-")
        action_type = parts[0].strip()
        try:
            num_frames = int(parts[1].strip())
        except ValueError:
            continue

        # Local velocity
        trans_vel = np.zeros(3)  # x, y, z
        ang_vel = np.zeros(3)  # x, y, z (euler)

        # Coordinate system assumptions (OpenCV convention):
        # Forward: +Z
        # Right: +X
        # Down: +Y (Pitch axis = Right = +X; Yaw axis = Down = +Y)

        # Translation
        if action_type == "w":
            trans_vel[2] = forward_speed
        elif action_type == "s":
            trans_vel[2] = -forward_speed
        elif action_type == "a":
            trans_vel[0] = -forward_speed
        elif action_type == "d":
            trans_vel[0] = forward_speed
        elif action_type == "u":
            trans_vel[1] = -forward_speed  # up = -Y in OpenCV
        elif action_type == "n":
            trans_vel[1] = forward_speed  # down = +Y in OpenCV
        # Rotation
        elif action_type == "up":
            ang_vel[0] = pitch_speed
        elif action_type == "down":
            ang_vel[0] = -pitch_speed
        elif action_type == "left":
            ang_vel[1] = -yaw_speed
        elif action_type == "right":
            ang_vel[1] = yaw_speed
        elif action_type == "cw":
            ang_vel[2] = roll_speed
        elif action_type == "ccw":
            ang_vel[2] = -roll_speed
        elif action_type == "stay":
            pass  # zero velocity — hold current pose
        elif action_type == "pano":
            # 360° panorama: distribute full yaw rotation evenly across frames
            pano_yaw_per_frame = 2.0 * np.pi / num_frames
            for _ in range(num_frames):
                rot_mat = R.from_euler("xyz", [0, pano_yaw_per_frame, 0]).as_matrix()
                delta_T = np.eye(4)
                delta_T[:3, :3] = rot_mat
                current_pose = current_pose @ delta_T
                poses.append(current_pose.copy())
            continue
        elif action_type == "orbit":
            # 360° orbit around a point in front of the current camera.
            # The orbit center sits at (0, 0, +radius) in the camera's local
            # frame (i.e. straight ahead). Both position and orientation are
            # updated so the camera keeps facing the center.
            #
            # Sign: OpenCV frame has +Y pointing down, so CCW as viewed from
            # above corresponds to a NEGATIVE rotation around +Y.
            orbit_radius = 1.0
            if len(parts) >= 3:
                try:
                    orbit_radius = float(parts[2].strip())
                except ValueError:
                    pass
            dtheta = -2.0 * np.pi / num_frames  # negative => CCW from above
            delta_rot = R.from_euler("xyz", [0.0, dtheta, 0.0]).as_matrix()
            delta_trans = np.array([-orbit_radius * np.sin(dtheta), 0.0, orbit_radius * (1.0 - np.cos(dtheta))])
            for _ in range(num_frames):
                delta_T = np.eye(4)
                delta_T[:3, :3] = delta_rot
                delta_T[:3, 3] = delta_trans
                current_pose = current_pose @ delta_T
                poses.append(current_pose.copy())
            continue

        for _ in range(num_frames):
            # Construct Delta Matrix
            # Rotation
            rot_mat = R.from_euler("xyz", ang_vel).as_matrix()

            delta_T = np.eye(4)
            delta_T[:3, :3] = rot_mat
            delta_T[:3, 3] = trans_vel

            # Update global pose: P_new = P_old @ Delta_T
            current_pose = current_pose @ delta_T
            poses.append(current_pose.copy())

    return np.stack(poses)


def load_custom_pose_jsons(paths_str: str) -> list[np.ndarray]:
    """Load one (T, 4, 4) c2w trajectory per comma-separated JSON path.

    Each JSON must decode to a list of 4x4 camera-to-world pose matrices anchored
    at the first frame (``T_0 = I``). Used to feed real GT trajectories
    (e.g. ``gt_camera_val_idx*.json``) through the ``--custom_pose`` pathway so
    we can decouple the trajectory-OOD axis from the first-frame-OOD axis when
    debugging poor pose following.

    Returns:
        A list of arrays, each with shape ``(T_i, 4, 4)``.
    """
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    trajectories: list[np.ndarray] = []
    for p in paths:
        with open(p) as f:
            arr = np.asarray(json.load(f), dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (4, 4):
            raise ValueError(f"{p}: expected shape (T, 4, 4), got {arr.shape}")
        trajectories.append(arr)
    return trajectories


def parse_custom_trajectories(camera_string: str) -> list[np.ndarray]:
    """Parse a ``|``-separated camera control string into per-trajectory absolute poses.

    Each segment between ``|`` is an independent trajectory parsed by
    :func:`parse_camera_string`.  Returns a list of arrays, each with shape
    ``(T_i, 4, 4)``.
    """
    trajectories = []
    for segment in camera_string.split("|"):
        segment = segment.strip()
        if segment:
            trajectories.append(parse_camera_string(segment))
    return trajectories


def trajectory_to_action(
    poses_abs: np.ndarray,
    num_video_frames: int,
    params: CameraTrainParams,
) -> np.ndarray:
    """Convert absolute poses to relative action vectors, truncating/padding to *num_video_frames*.

    Returns:
        ``action_np`` with shape ``(num_video_frames - 1, D)``.
    """
    T_needed = num_video_frames
    T_have = poses_abs.shape[0]

    if T_have < T_needed:
        pad = np.tile(poses_abs[-1:], (T_needed - T_have, 1, 1))
        poses_abs = np.concatenate([poses_abs, pad], axis=0)
    elif T_have > T_needed:
        poses_abs = poses_abs[:T_needed]

    return pose_abs_to_rel(
        poses_abs,
        rotation_format=params.rotation_format,
        pose_convention=params.pose_convention,
    )
