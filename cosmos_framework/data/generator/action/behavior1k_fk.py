# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, pose_abs_to_rel

BEHAVIOR1K_AGIBOT_ACTION_DIM = 29


BEHAVIOR1K_R1PRO_ACTION_DIM = 23


BEHAVIOR1K_R1PRO_QPOS_DIM = 28


BEHAVIOR1K_R1PRO_HEAD_LINK = "zed_link"


BEHAVIOR1K_R1PRO_LEFT_EEF_LINK = "left_arm_link7"


BEHAVIOR1K_R1PRO_RIGHT_EEF_LINK = "right_arm_link7"


BEHAVIOR1K_R1PRO_GRIPPER_TRAVEL_M = 0.05


BEHAVIOR1K_R1PRO_FULLY_OPEN_GRIPPER_WIDTH_M: float = 2.0 * BEHAVIOR1K_R1PRO_GRIPPER_TRAVEL_M


def _validate_state_prefixed_action(action: np.ndarray, fps: float) -> None:
    """Validate the state-prefixed native R1Pro action contract."""

    if action.ndim != 2 or action.shape[1] != BEHAVIOR1K_R1PRO_ACTION_DIM or action.shape[0] < 2:
        raise ValueError(
            f"BEHAVIOR-1K R1Pro action must have shape [T+1,23] with an initial state row, got {tuple(action.shape)}"
        )
    if not np.isfinite(action).all():
        raise ValueError("BEHAVIOR-1K R1Pro action contains NaN or Inf values")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"BEHAVIOR-1K fps must be positive and finite, got {fps!r}")


def _build_r1pro_qpos(action: np.ndarray) -> np.ndarray:
    """Map state-prefixed R1Pro action rows to MuJoCo qpos ``[T+1,28]``."""

    frame_count = action.shape[0]
    left_gripper = np.clip((action[:, 14] + 1.0) / 2.0, 0.0, 1.0)  # [T+1]
    right_gripper = np.clip((action[:, 22] + 1.0) / 2.0, 0.0, 1.0)  # [T+1]
    qpos = np.zeros((frame_count, BEHAVIOR1K_R1PRO_QPOS_DIM), dtype=np.float32)  # [T+1,28]
    qpos[:, 6:10] = action[:, 3:7]  # [T+1,4]
    qpos[:, 10:17] = action[:, 7:14]  # [T+1,7]
    qpos[:, 17:19] = left_gripper[:, None] * BEHAVIOR1K_R1PRO_GRIPPER_TRAVEL_M  # [T+1,2]
    qpos[:, 19:26] = action[:, 15:22]  # [T+1,7]
    qpos[:, 26:28] = right_gripper[:, None] * BEHAVIOR1K_R1PRO_GRIPPER_TRAVEL_M  # [T+1,2]
    return qpos


def build_behavior1k_r1pro_qpos_from_state(state: np.ndarray) -> np.ndarray:
    """Map measured B1K state rows to the full R1Pro MuJoCo qpos layout.

    The state rows are the authoritative robot configuration for visualization:
    the base virtual joints are handled by the separate absolute base-pose
    trajectory, while the trunk, both arms, and both grippers are copied into
    the fixed-base R1Pro qpos layout.  The returned shape is ``[T+1,28]``.
    """

    state = np.asarray(state, dtype=np.float32)  # [T+1,61]
    if state.ndim != 2 or state.shape[1] != 61 or state.shape[0] < 1:
        raise ValueError(f"BEHAVIOR-1K state must have shape [T+1,61], got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("BEHAVIOR-1K state contains NaN or Inf values")

    from cosmos_framework.data.generator.action.behavior1k_lerobot_dataset_config import (
        state61_to_action23,
    )

    # base_vel_state_scale=1.0: FK reconstructs what the base physically did, from
    # measured proprioception. The 0.75 correction exists to align the state frame with
    # the action's numeric range through the shared action ENCODER -- a model-input
    # concern, not a physical one. Applying it here would integrate a trajectory 1/0.75
    # longer than the robot actually travelled.
    state_action = (
        state61_to_action23(torch.from_numpy(state), base_vel_state_scale=1.0).numpy().astype(np.float32, copy=False)
    )  # [T+1,23]
    return _build_r1pro_qpos(state_action)  # [T+1,28]


def _integrate_r1pro_base(action: np.ndarray, fps: float) -> np.ndarray:
    """Integrate local R1Pro base velocities into normalized base poses."""

    frame_count = action.shape[0]
    base_poses = np.tile(np.eye(4, dtype=np.float32), (frame_count, 1, 1))  # [T+1,4,4]
    dt = 1.0 / fps
    yaw = 0.0
    for frame_idx in range(1, frame_count):
        vx, vy, yaw_velocity = action[frame_idx, 0:3]
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        base_poses[frame_idx, 0, 3] = base_poses[frame_idx - 1, 0, 3] + (cos_yaw * vx - sin_yaw * vy) * dt
        base_poses[frame_idx, 1, 3] = base_poses[frame_idx - 1, 1, 3] + (sin_yaw * vx + cos_yaw * vy) * dt
        yaw += float(yaw_velocity) * dt
        base_poses[frame_idx, :2, :2] = np.asarray(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float32,
        )  # [2,2]
    return base_poses


def _validate_absolute_base_poses(base_poses: np.ndarray, frame_count: int) -> np.ndarray:
    """Validate absolute R1Pro base poses aligned to state frames."""

    base_poses = np.asarray(base_poses, dtype=np.float32)  # [T+1,4,4]
    if base_poses.shape != (frame_count, 4, 4):
        raise ValueError(f"Absolute R1Pro base poses must have shape [{frame_count},4,4], got {base_poses.shape}")
    if not np.isfinite(base_poses).all():
        raise ValueError("Absolute R1Pro base poses contain NaN or Inf values")
    if not np.allclose(base_poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
        raise ValueError("Absolute R1Pro base poses must be homogeneous transforms with bottom row [0,0,0,1]")
    rotations = base_poses[:, :3, :3]  # [T+1,3,3]
    rotation_gram = np.matmul(rotations.transpose(0, 2, 1), rotations)  # [T+1,3,3]
    if not np.allclose(rotation_gram, np.eye(3, dtype=np.float32), atol=1e-3):
        raise ValueError("Absolute R1Pro base poses contain non-orthonormal rotations")
    return np.ascontiguousarray(base_poses)


def _pose_from_mujoco(data: Any, body_id: int) -> np.ndarray:
    """Extract one MuJoCo body pose as a homogeneous transform ``[4,4]``."""

    pose = np.eye(4, dtype=np.float32)  # [4,4]
    pose[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float32)  # [3]
    pose[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float32).reshape(3, 3)  # [3,3]
    return pose


def _rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF fixed-joint RPY values to a rotation matrix."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )  # [3,3]


def _fixed_link_transform(urdf_path: Path, link_name: str) -> tuple[str, np.ndarray] | None:
    """Return the parent and fixed transform for a link fused by MuJoCo."""

    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if (
            joint.get("type") == "fixed"
            and child is not None
            and child.get("link") == link_name
            and parent is not None
            and parent.get("link")
        ):
            origin = joint.find("origin")
            xyz = np.zeros(3, dtype=np.float32)  # [3]
            rpy = np.zeros(3, dtype=np.float32)  # [3]
            if origin is not None:
                xyz = np.asarray(
                    [float(value) for value in origin.get("xyz", "0 0 0").split()], dtype=np.float32
                )  # [3]
                rpy = np.asarray(
                    [float(value) for value in origin.get("rpy", "0 0 0").split()], dtype=np.float32
                )  # [3]
            transform = np.eye(4, dtype=np.float32)  # [4,4]
            transform[:3, :3] = _rpy_to_rotation_matrix(*rpy)
            transform[:3, 3] = xyz
            return str(parent.get("link")), transform
    return None


@dataclass(frozen=True)
class _R1ProForwardKinematicsModel:
    """One compiled R1Pro model plus its resolved FK body frames."""

    model: Any
    # Maps a pose name onto its MuJoCo body id and an optional fixed-link
    # transform applied in that body's frame.
    frame_specs: dict[str, tuple[int, np.ndarray | None]]


def _compile_r1pro_urdf(urdf_path: Path) -> Any:
    """Compile the provisioned URDF directly with the pinned training runtime."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(urdf_path))


@lru_cache(maxsize=1)
def _load_r1pro_fk_model(urdf_path_str: str) -> _R1ProForwardKinematicsModel:
    """Compile the R1Pro URDF once per worker process and cache the result.

    FK runs once per training item, and building a ``MjModel`` reads the URDF,
    parses its XML and compiles the model, which costs far more than the FK
    itself. The compiled model is immutable during FK -- only the per-call
    ``MjData`` is mutated -- so one process-local instance is safe to share
    across windows and across the reader threads of one DataLoader worker.
    This mirrors ``embodiment_c_fk._get_fk_engine``; B1K keys the cache on the
    URDF path because the asset is configured rather than committed.
    """

    import mujoco

    urdf_path = Path(urdf_path_str)
    model = _compile_r1pro_urdf(urdf_path)
    if model.nq < BEHAVIOR1K_R1PRO_QPOS_DIM:
        raise ValueError(f"R1Pro URDF {urdf_path} has nq={model.nq}, expected at least {BEHAVIOR1K_R1PRO_QPOS_DIM}")
    frame_specs: dict[str, tuple[int, np.ndarray | None]] = {}
    for pose_name, body_name in {
        "head_camera": BEHAVIOR1K_R1PRO_HEAD_LINK,
        "left_wrist": BEHAVIOR1K_R1PRO_LEFT_EEF_LINK,
        "right_wrist": BEHAVIOR1K_R1PRO_RIGHT_EEF_LINK,
    }.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            frame_specs[pose_name] = (body_id, None)
            continue
        fixed_link = _fixed_link_transform(urdf_path, body_name)
        if fixed_link is None:
            raise ValueError(f"R1Pro URDF {urdf_path} does not contain body or fixed link {body_name!r}")
        parent_name, transform = fixed_link
        parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
        if parent_id < 0:
            raise ValueError(f"R1Pro URDF fixed link {body_name!r} has missing parent body {parent_name!r}")
        frame_specs[pose_name] = (parent_id, transform)
    return _R1ProForwardKinematicsModel(model=model, frame_specs=frame_specs)


def _compute_r1pro_fk_poses(
    qpos: np.ndarray,
    base_poses: np.ndarray,
    *,
    urdf_path: Path,
) -> dict[str, np.ndarray]:
    """Compute normalized head and bilateral EEF poses from the R1Pro URDF."""

    import mujoco

    fk_model = _load_r1pro_fk_model(str(urdf_path))
    model = fk_model.model
    frame_specs = fk_model.frame_specs
    # MjData holds mutable simulation state and stays per call, unlike the
    # cached model.
    data = mujoco.MjData(model)

    poses: dict[str, list[np.ndarray]] = {name: [] for name in frame_specs}
    for frame_qpos, base_pose in zip(qpos, base_poses, strict=True):
        data.qpos[:] = 0.0
        data.qpos[:BEHAVIOR1K_R1PRO_QPOS_DIM] = frame_qpos
        mujoco.mj_forward(model, data)
        for pose_name, (body_id, fixed_transform) in frame_specs.items():
            local_pose = _pose_from_mujoco(data, body_id)  # [4,4]
            if fixed_transform is not None:
                local_pose = local_pose @ fixed_transform  # [4,4]
            poses[pose_name].append(base_pose @ local_pose)  # [4,4]

    return {
        pose_name: np.stack(frame_poses, axis=0).astype(np.float32, copy=False)  # [T+1,4,4]
        for pose_name, frame_poses in poses.items()
    }


def build_behavior1k_agibot_fk_action(
    action: np.ndarray,
    *,
    urdf_path: str | Path,
    fps: float = 30.0,
    pose_convention: PoseConvention = "backward_framewise",
    base_poses: np.ndarray | None = None,
    base_pose_source: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a shared 29-D SE(3) action from a state-prefixed R1Pro action.

    Args:
        action: Native R1Pro action with shape ``[T+1,23]``. Row zero is the
            conditioning state and rows one through ``T`` are transitions.
        urdf_path: Provisioned calibrated R1Pro URDF used for FK.
        fps: Source sample rate used to integrate base velocity commands.
        pose_convention: Relative-pose convention shared with AgiBot loaders.
        base_poses: Optional absolute R1Pro base poses ``[T+1,4,4]`` from raw
            BEHAVIOR replay states. When provided, these replace velocity
            integration for the base transform used by FK.
        base_pose_source: Provenance label for the supplied absolute poses.

    Returns:
        ``([T,29] action, extras)`` with the shared layout
        ``head(9), right_eef(9), right_gripper(1), left_eef(9), left_gripper(1)``.
    """

    action = np.asarray(action, dtype=np.float32)  # [T+1,23]
    _validate_state_prefixed_action(action, fps)
    urdf_path = Path(urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"R1Pro FK asset does not exist: {urdf_path}")

    qpos = _build_r1pro_qpos(action)  # [T+1,28]
    resolved_base_poses = (
        _validate_absolute_base_poses(base_poses, action.shape[0])
        if base_poses is not None
        else _integrate_r1pro_base(action, fps)
    )  # [T+1,4,4]
    poses = _compute_r1pro_fk_poses(qpos, resolved_base_poses, urdf_path=urdf_path)
    head_relative = pose_abs_to_rel(
        poses["head_camera"], rotation_format="rot6d", pose_convention=pose_convention
    )  # [T,9]
    right_relative = pose_abs_to_rel(
        poses["right_wrist"], rotation_format="rot6d", pose_convention=pose_convention
    )  # [T,9]
    left_relative = pose_abs_to_rel(
        poses["left_wrist"], rotation_format="rot6d", pose_convention=pose_convention
    )  # [T,9]
    right_gripper = np.clip((action[1:, 22] + 1.0) / 2.0, 0.0, 1.0)[:, None]  # [T,1]
    left_gripper = np.clip((action[1:, 14] + 1.0) / 2.0, 0.0, 1.0)[:, None]  # [T,1]
    action_values = np.concatenate(
        [head_relative, right_relative, right_gripper, left_relative, left_gripper],
        axis=-1,
    ).astype(np.float32, copy=False)  # [T,29]
    return torch.from_numpy(action_values), {
        "initial_pose": torch.from_numpy(poses["head_camera"][0].copy()).float(),  # [4,4]
        "initial_pose_right": torch.from_numpy(poses["right_wrist"][0].copy()).float(),  # [4,4]
        "initial_pose_left": torch.from_numpy(poses["left_wrist"][0].copy()).float(),  # [4,4]
        "fk_asset": str(urdf_path),
        "fk_head_link": BEHAVIOR1K_R1PRO_HEAD_LINK,
        "fk_left_eef_link": BEHAVIOR1K_R1PRO_LEFT_EEF_LINK,
        "fk_right_eef_link": BEHAVIOR1K_R1PRO_RIGHT_EEF_LINK,
        "base_poses": torch.from_numpy(resolved_base_poses.copy()).float(),  # [T+1,4,4]
        "initial_base_pose": torch.from_numpy(resolved_base_poses[0].copy()).float(),  # [4,4]
        "base_pose_source": base_pose_source or "integrated_measured_base_velocity",
    }


def build_behavior1k_state_fk_action(
    state: np.ndarray,
    *,
    urdf_path: str | Path,
    fps: float = 30.0,
    pose_convention: PoseConvention = "backward_framewise",
    base_poses: np.ndarray | None = None,
    base_pose_source: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the shared 29-D action from measured R1Pro state frames.

    ``observation.state`` exposes measured R1Pro qpos values and measured base
    velocity. Mapping the full ``[T+1,61]`` state window into the native 23-D
    layout lets the existing FK implementation use measured qpos and measured
    base velocity while preserving the shared AgiBot 29-D output contract. If
    ``base_poses`` are supplied from raw replay states, they are used as the
    absolute base trajectory instead of integrating measured base velocity.
    The native command stream remains separate and is not overwritten.
    """

    state = np.asarray(state, dtype=np.float32)  # [T+1,61]
    if state.ndim != 2 or state.shape[1] != 61 or state.shape[0] < 2:
        raise ValueError(f"BEHAVIOR-1K state must have shape [T+1,61] with T >= 1, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("BEHAVIOR-1K state contains NaN or Inf values")

    from cosmos_framework.data.generator.action.behavior1k_lerobot_dataset_config import (
        state61_to_action23,
    )

    # base_vel_state_scale=1.0: FK reconstructs what the base physically did, from
    # measured proprioception. The 0.75 correction exists to align the state frame with
    # the action's numeric range through the shared action ENCODER -- a model-input
    # concern, not a physical one. Applying it here would integrate a trajectory 1/0.75
    # longer than the robot actually travelled.
    state_action_native = (
        state61_to_action23(torch.from_numpy(state), base_vel_state_scale=1.0).numpy().astype(np.float32, copy=False)
    )  # [T+1,23]
    action, extras = build_behavior1k_agibot_fk_action(
        state_action_native,
        urdf_path=urdf_path,
        fps=fps,
        pose_convention=pose_convention,
        base_poses=base_poses,
        base_pose_source=base_pose_source,
    )
    extras["action_source"] = "measured_observation_state"
    return action, extras
