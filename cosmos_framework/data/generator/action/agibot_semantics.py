# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.embodiment_c_fk import (
    AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
    apply_agibot_gripper_to_opencv,
    apply_robot_base_motion_to_poses,
    compute_fk_transforms_batch,
    compute_link_poses_batch,
    extract_fk_transforms_from_link_poses,
    extract_gripper_states,
)
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, convert_rotation, pose_abs_to_rel

AGIBOT_LEFT_HAND_VIEW = "observation.images.hand_left"


AGIBOT_RIGHT_HAND_VIEW = "observation.images.hand_right"


AGIBOT_ACTION_DIM = 29


AGIBOT_MULTIVIEW_DESCRIPTION = (
    "The top row shows the head-mounted camera view looking down at the workspace. "
    "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
    "the left hand camera on the left and the right hand camera on the right."
)


AgiBotFkEmbodiment = Literal["embodiment_c_gripper", "embodiment_c_gripper_ext"]


_AGIBOTWORLD2026_GRIPPER_OPEN_RAD = 0.91


_AGIBOTWORLD2026_GRIPPER_VALUE_EPS = 1e-4


_AGIBOTWORLD2026_EEF_QUATERNION_SLICE = slice(26, 34)


_AGIBOTWORLD2026_EEF_POSITION_SLICE = slice(34, 40)


@dataclass(frozen=True)
class AgiBotWorld2026StateLayout:
    """Indices required to recover G2A head and wrist poses from one state layout."""

    width: int
    head_rotation_slice: slice
    head_translation_slice: slice


_AGIBOTWORLD2026_STATE_LAYOUTS: dict[int, AgiBotWorld2026StateLayout] = {
    162: AgiBotWorld2026StateLayout(
        width=162,
        head_rotation_slice=slice(126, 135),
        head_translation_slice=slice(156, 159),
    ),
    169: AgiBotWorld2026StateLayout(
        width=169,
        head_rotation_slice=slice(133, 142),
        head_translation_slice=slice(163, 166),
    ),
}


def convert_agibotworld2026_gripper_state_to_open_fraction(values: np.ndarray) -> np.ndarray:
    """Convert G2A gripper radians to the shared ``0=closed, 1=open`` convention."""

    values = np.asarray(values, dtype=np.float32)  # [*]
    if not np.isfinite(values).all():
        raise ValueError("AgiBotWorld2026 gripper values contain NaN or Inf values.")
    if values.size == 0:
        return values

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if (
        min_value < -_AGIBOTWORLD2026_GRIPPER_OPEN_RAD - _AGIBOTWORLD2026_GRIPPER_VALUE_EPS
        or max_value > _AGIBOTWORLD2026_GRIPPER_VALUE_EPS
    ):
        raise ValueError(
            "Unsupported AgiBotWorld2026 gripper value range; "
            f"min={min_value:.4f}, max={max_value:.4f}. Expected radians in "
            f"[-{_AGIBOTWORLD2026_GRIPPER_OPEN_RAD:.2f},0]."
        )
    return np.clip(-values / _AGIBOTWORLD2026_GRIPPER_OPEN_RAD, 0.0, 1.0).astype(np.float32, copy=False)  # [*]


def _build_pose_trajectory(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Build homogeneous poses from aligned position and rotation trajectories."""

    poses = np.tile(np.eye(4, dtype=np.float32), (positions.shape[0], 1, 1))  # [T+1,4,4]
    poses[:, :3, :3] = rotations  # [T+1,3,3]
    poses[:, :3, 3] = positions  # [T+1,3]
    return poses


def build_agibotworld2026_action(
    states: np.ndarray,  # [T+1,162|169]
    *,
    pose_convention: PoseConvention,
) -> tuple[torch.Tensor, dict[str, Any]]:  # [T,29], extras
    """Build the shared 29D AgiBot action directly from recorded G2A poses.

    AgiBotWorld2026 records both arm-end poses and the aligned front-head
    camera transform in every observation state. These poses already include
    the G2A waist, head, arm, and optional mobile-base motion, so no G1 FK or
    external G2A robot asset is used here.
    """

    states = np.asarray(states, dtype=np.float32)  # [T+1,162|169]
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError(f"AgiBotWorld2026 observation.state must have shape [T+1,S] with T >= 1, got {states.shape}.")
    layout = _AGIBOTWORLD2026_STATE_LAYOUTS.get(int(states.shape[1]))
    if layout is None:
        raise ValueError(
            "Unsupported AgiBotWorld2026 observation.state width "
            f"{states.shape[1]}; expected one of {sorted(_AGIBOTWORLD2026_STATE_LAYOUTS)}."
        )
    if not np.isfinite(states).all():
        raise ValueError("AgiBotWorld2026 observation.state contains NaN or Inf values.")

    num_frames = int(states.shape[0])
    eef_quaternions = states[:, _AGIBOTWORLD2026_EEF_QUATERNION_SLICE].reshape(num_frames, 2, 4)  # [T+1,2,4]
    quaternion_norms = np.linalg.norm(eef_quaternions, axis=-1)  # [T+1,2]
    if np.any(quaternion_norms < 1e-8):
        raise ValueError("AgiBotWorld2026 end-effector state contains a zero quaternion.")
    eef_rotations = np.asarray(
        convert_rotation(
            eef_quaternions,
            input_format="quat_xyzw",
            output_format="matrix",
            normalize_matrix=True,
        ),
        dtype=np.float32,
    )  # [T+1,2,3,3]
    eef_positions = states[:, _AGIBOTWORLD2026_EEF_POSITION_SLICE].reshape(num_frames, 2, 3)  # [T+1,2,3]
    left_wrist_poses = _build_pose_trajectory(eef_positions[:, 0], eef_rotations[:, 0])  # [T+1,4,4]
    right_wrist_poses = _build_pose_trajectory(eef_positions[:, 1], eef_rotations[:, 1])  # [T+1,4,4]

    head_rotation_values = states[:, layout.head_rotation_slice].reshape(num_frames, 3, 3)  # [T+1,3,3]
    head_rotations = np.asarray(
        convert_rotation(
            head_rotation_values,
            input_format="matrix",
            output_format="matrix",
            normalize_matrix=True,
        ),
        dtype=np.float32,
    )  # [T+1,3,3]
    head_positions = states[:, layout.head_translation_slice]  # [T+1,3]
    head_poses = _build_pose_trajectory(head_positions, head_rotations)  # [T+1,4,4]

    head_relative = pose_abs_to_rel(
        head_poses,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    right_relative = pose_abs_to_rel(
        right_wrist_poses,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    left_relative = pose_abs_to_rel(
        left_wrist_poses,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    right_gripper = convert_agibotworld2026_gripper_state_to_open_fraction(states[1:, 1])[:, None]  # [T,1]
    left_gripper = convert_agibotworld2026_gripper_state_to_open_fraction(states[1:, 0])[:, None]  # [T,1]
    action_values = np.concatenate(
        [head_relative, right_relative, right_gripper, left_relative, left_gripper],
        axis=-1,
    ).astype(np.float32, copy=False)  # [T,29]
    action = torch.from_numpy(action_values).float()  # [T,29]
    extras: dict[str, Any] = {
        "initial_pose": torch.from_numpy(head_poses[0].copy()).float(),  # [4,4]
        "initial_pose_right": torch.from_numpy(right_wrist_poses[0].copy()).float(),  # [4,4]
        "initial_pose_left": torch.from_numpy(left_wrist_poses[0].copy()).float(),  # [4,4]
    }
    return action, extras


def assemble_embodiment_c_state(
    effector_position: np.ndarray,  # [T+1,2]
    joint_position: np.ndarray,  # [T+1,14]
    head_position: np.ndarray,  # [T+1,2]
    waist_position: np.ndarray,  # [T+1,2]
) -> np.ndarray:  # [T+1,20]
    """Assemble a GEAR-compatible flat state vector from Beta's decomposed fields.

    Args:
        effector_position: ``(N, 2)`` gripper positions as ``[left_gripper, right_gripper]``.
        joint_position: ``(N, 14)`` arm joint positions (7 left + 7 right).
        head_position: ``(N, 2)`` head joints as ``[head_yaw, head_pitch]``.
        waist_position: ``(N, 2)`` waist joints as ``[body_pitch, lift_body]``.

    Returns:
        State array of shape ``(N, 20)`` with GEAR gripper layout:
        ``[arm_joints(14), gripper(2), head_yaw, head_pitch, waist_pitch, waist_lift]``

    Note:
        Beta ``waist_position = [body_pitch, lift_body]`` maps directly into the
        standard GEAR body/head block:
        ``state[16:20] = [head_yaw, head_pitch, body_pitch, lift_body]``.
    """
    # Body/head block: [head_yaw, head_pitch, waist_pitch, waist_lift]
    body_head = np.stack(
        [head_position[:, 0], head_position[:, 1], waist_position[:, 0], waist_position[:, 1]],
        axis=-1,
    )  # [T+1,4]
    return np.concatenate([joint_position, effector_position, body_head], axis=-1).astype(
        np.float32, copy=False
    )  # [T+1,20]


def build_agibot_fk_action(
    states: np.ndarray,  # [T+1,S]
    *,
    fk_embodiment_type: AgiBotFkEmbodiment,
    pose_convention: PoseConvention,
    robot_position: np.ndarray | None = None,  # [T+1,3] | None
    robot_orientation: np.ndarray | None = None,  # [T+1,4] | None
    return_agibot_link_poses: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:  # [T,29], extras
    """Build the shared 29D AgiBot FK-pose action and reconstruction extras.

    Uses the full ``T+1`` observed-state window to compute absolute FK
    transforms, converts them to relative rot6d SE(3) deltas, and concatenates
    scalar gripper values from the next observed frame. The source ``action``
    column is intentionally not used.

    Scalar gripper actions follow the shared viewer/action convention:
    ``0.0`` means closed and ``1.0`` means open. Source URDF-angle and
    actuator-close encodings are normalized by ``extract_gripper_states``.
    Explicit mobile-base position and orientation must be supplied together and
    are only valid for the non-``_ext`` embodiment. The ``_ext`` state already
    contains base motion, which the FK helpers apply internally.
    """

    has_robot_position = robot_position is not None
    has_robot_orientation = robot_orientation is not None
    if has_robot_position != has_robot_orientation:
        raise ValueError("robot_position and robot_orientation must be provided together.")
    if has_robot_position and fk_embodiment_type == "embodiment_c_gripper_ext":
        raise ValueError("embodiment_c_gripper_ext states already include robot base motion.")

    link_poses: dict[str, np.ndarray] | None = None
    if return_agibot_link_poses:
        link_poses = compute_link_poses_batch(states, fk_embodiment_type)  # {name:[T+1,4,4]}
        if robot_position is not None and robot_orientation is not None:
            link_poses = apply_robot_base_motion_to_poses(
                link_poses,
                robot_position,
                robot_orientation,
            )  # {name:[T+1,4,4]}
        native_fk = extract_fk_transforms_from_link_poses(link_poses)  # {name:[T+1,4,4]}
    else:
        native_fk = compute_fk_transforms_batch(states, fk_embodiment_type)  # {name:[T+1,4,4]}
        if robot_position is not None and robot_orientation is not None:
            native_fk = apply_robot_base_motion_to_poses(
                native_fk,
                robot_position,
                robot_orientation,
            )  # {name:[T+1,4,4]}

    fk = apply_agibot_gripper_to_opencv(
        native_fk,
        AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
    )  # {name:[T+1,4,4]}

    # Relative SE(3) deltas.
    head_relative = pose_abs_to_rel(
        fk["head_camera"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    right_relative = pose_abs_to_rel(
        fk["right_wrist"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    left_relative = pose_abs_to_rel(
        fk["left_wrist"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]

    right_gripper_values, left_gripper_values = extract_gripper_states(states[1:], fk_embodiment_type)
    if right_gripper_values is None or left_gripper_values is None:
        raise ValueError(f"{fk_embodiment_type} does not expose scalar gripper states.")
    right_gripper = right_gripper_values[:, None]  # [T,1]
    left_gripper = left_gripper_values[:, None]  # [T,1]
    action_values = np.concatenate(
        [head_relative, right_relative, right_gripper, left_relative, left_gripper],
        axis=-1,
    ).astype(np.float32, copy=False)  # [T,29]
    action = torch.from_numpy(action_values).float()  # [T,29]
    extras: dict[str, Any] = {
        "initial_pose": torch.from_numpy(fk["head_camera"][0].copy()).float(),  # [4,4]
        "initial_pose_right": torch.from_numpy(fk["right_wrist"][0].copy()).float(),  # [4,4]
        "initial_pose_left": torch.from_numpy(fk["left_wrist"][0].copy()).float(),  # [4,4]
    }
    if link_poses is not None:
        extras["agibot_link_poses"] = {
            link_name: torch.from_numpy(poses.copy()).float() for link_name, poses in link_poses.items()
        }  # {name:[T+1,4,4]}
    return action, extras


def build_agibot_action_spec() -> ActionSpec:
    """Describe the shared 29D head/right-EEF/left-EEF pose-action layout."""
    return build_action_spec(
        Pos(prefix="head"),
        Rot("rot6d", prefix="head"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
    )


def agibot_normalizer_filename(
    pose_convention: str,
    rotation_format: str = "rot6d",
) -> str:
    """Return the shared normalizer filename used by all AgiBot pose-action loaders."""
    return f"agibot_{pose_convention}_{rotation_format}.json"
