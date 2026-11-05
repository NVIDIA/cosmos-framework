# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.hand_pose_dataset_config import (
    CAM_POSITION_KEY,
    CAM_ROTATION_KEY,
    NUM_JOINTS,
    QUAT_DIM_PER_JOINT,
    ROTATION_FORMAT_DIM,
    WRIST_JOINT_IDX,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizationMethod,
    ActionNormalizer,
    load_action_normalizer,
)
from cosmos_framework.data.generator.action.utils.pose_utils import (
    RotationConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)
from cosmos_framework.utils import log

_NORMALIZERS_DIR = Path(__file__).parent / "normalizers"


HAND_POSE_JOINT_MODES: tuple[str, ...] = ("forward_dynamics", "inverse_dynamics", "wam")


HAND_POSE_JOINT_MODE_WEIGHTS: tuple[float, ...] = (0.8, 0.1, 0.1)


def compute_raw_action_dim(
    rotation_format: RotationConvention,
    position_joint_indices: tuple[int, ...],
) -> int:
    """Compute the raw action dimensionality for the hand-pose action layout."""
    rot_dim_per_joint = ROTATION_FORMAT_DIM[rotation_format]
    num_pos_joints = len(position_joint_indices)
    num_finger_joints = num_pos_joints - 1
    per_hand_dim = 3 + rot_dim_per_joint + num_finger_joints * 3
    return (3 + rot_dim_per_joint) + 2 * per_hand_dim


def _extract_wrist_pose_components(
    pos_data: np.ndarray,
    rot_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract wrist translation and wrist quaternion from full-hand arrays."""
    wrist_pos = pos_data[:, :3]  # [T+1,3]
    wrist_quat = rot_quat.reshape(pos_data.shape[0], NUM_JOINTS, QUAT_DIM_PER_JOINT)[:, WRIST_JOINT_IDX]  # [T+1,4]
    return wrist_pos, wrist_quat


def _build_fingers_in_wrist_frame(
    pos_data: np.ndarray,
    wrist_poses_abs: np.ndarray,
    position_joint_indices: tuple[int, ...],
) -> np.ndarray:
    """Express selected non-wrist keypoints in the per-frame wrist coordinate frame."""
    future_pos = pos_data[1:].astype(np.float32, copy=False)  # [T,63]
    num_future_frames = future_pos.shape[0]

    selected_non_wrist = [joint_index for joint_index in position_joint_indices if joint_index != WRIST_JOINT_IDX]
    if not selected_non_wrist:
        return np.empty((num_future_frames, 0), dtype=np.float32)  # [T,0]

    pos_3d = future_pos.reshape(num_future_frames, NUM_JOINTS, 3)  # [T,21,3]
    finger_pos = pos_3d[:, selected_non_wrist, :]  # [T,F,3]
    finger_pos_h = np.concatenate(
        [finger_pos, np.ones((*finger_pos.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )  # [T,F,4]
    wrist_inv = np.linalg.inv(wrist_poses_abs[1:])  # [T,4,4]
    finger_pos_wrist = np.einsum("tij,tnj->tni", wrist_inv, finger_pos_h)[..., :3]  # [T,F,3]
    return finger_pos_wrist.reshape(num_future_frames, -1)  # [T,3F]


def build_hand_pose_action(
    sample: dict[str, Any],
    *,
    pose_convention: Literal["backward_anchored", "backward_framewise"],
    rotation_format: RotationConvention,
    position_keys: tuple[str, str],
    rotation_keys: tuple[str, str],
    position_joint_indices: tuple[int, ...],
    wrist_frame_alignment: np.ndarray | None,
) -> torch.Tensor:  # returns [T,D]
    """Build the hand-pose action tensor from a ``T+1`` frame sample.

    Layout: ``[camera, R_wrist, R_fingers, L_wrist, L_fingers]``.
    Both supported pose conventions compute absolute camera/wrist poses first,
    then convert them to anchored or frame-wise relative deltas.
    """

    def _to_np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    cam_pos = _to_np(sample[CAM_POSITION_KEY])  # [T+1,3]
    cam_rot_q = _to_np(sample[CAM_ROTATION_KEY])  # [T+1,4]
    cam_c2w = build_abs_pose_from_components(cam_pos, cam_rot_q, "quat_xyzw")  # [T+1,4,4]

    right_pos_all = _to_np(sample[position_keys[1]])  # [T+1,63]
    right_rot_all = _to_np(sample[rotation_keys[1]])  # [T+1,84]
    left_pos_all = _to_np(sample[position_keys[0]])  # [T+1,63]
    left_rot_all = _to_np(sample[rotation_keys[0]])  # [T+1,84]

    right_wrist_pos, right_wrist_quat = _extract_wrist_pose_components(right_pos_all, right_rot_all)
    left_wrist_pos, left_wrist_quat = _extract_wrist_pose_components(left_pos_all, left_rot_all)

    right_wrist_cam = build_abs_pose_from_components(right_wrist_pos, right_wrist_quat, "quat_xyzw")  # [T+1,4,4]
    left_wrist_cam = build_abs_pose_from_components(left_wrist_pos, left_wrist_quat, "quat_xyzw")  # [T+1,4,4]

    if wrist_frame_alignment is not None:
        right_wrist_cam = right_wrist_cam @ wrist_frame_alignment  # [T+1,4,4]
        left_wrist_cam = left_wrist_cam @ wrist_frame_alignment  # [T+1,4,4]

    right_wrist_world = cam_c2w @ right_wrist_cam  # [T+1,4,4]
    left_wrist_world = cam_c2w @ left_wrist_cam  # [T+1,4,4]

    right_fingers = _build_fingers_in_wrist_frame(right_pos_all, right_wrist_cam, position_joint_indices)  # [T,F]
    left_fingers = _build_fingers_in_wrist_frame(left_pos_all, left_wrist_cam, position_joint_indices)  # [T,F]

    cam_rel = pose_abs_to_rel(cam_c2w, rotation_format=rotation_format, pose_convention=pose_convention)  # [T,Dc]
    right_wrist_rel = pose_abs_to_rel(
        right_wrist_world,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
    )  # [T,Dw]
    left_wrist_rel = pose_abs_to_rel(
        left_wrist_world,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
    )  # [T,Dw]

    action = np.concatenate([cam_rel, right_wrist_rel, right_fingers, left_wrist_rel, left_fingers], axis=-1)  # [T,D]
    return torch.from_numpy(action).float()  # [T,D]


def _normalizer_filename(
    embodiment_type: str,
    pose_convention: str,
    rotation_format: str,
) -> str:
    """Return the bundled normalizer filename for a hand-pose representation."""
    return f"{embodiment_type}_{pose_convention}_{rotation_format}.json"


def _resolve_action_stats_path(
    action_stats_path: str | None,
    *,
    embodiment_type: str,
    pose_convention: str,
    rotation_format: str,
) -> Path:
    """Resolve explicit or bundled hand-pose action statistics."""
    if action_stats_path is None:
        stats_path = _NORMALIZERS_DIR / _normalizer_filename(embodiment_type, pose_convention, rotation_format)
        if stats_path.exists():
            return stats_path
        raise FileNotFoundError(
            f"Could not find bundled hand-pose action stats at {stats_path}. "
            "Pass action_stats_path explicitly or regenerate stats with compute_action_stats.py."
        )

    stats_path = Path(action_stats_path)
    if stats_path.is_absolute():
        if stats_path.exists():
            return stats_path
        raise FileNotFoundError(f"Could not find action_stats_path={action_stats_path!r}.")

    module_dir = Path(__file__).resolve().parent
    candidates = [parent / stats_path for parent in module_dir.parents]
    candidates.extend([_NORMALIZERS_DIR / stats_path.name, module_dir / stats_path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve action_stats_path={action_stats_path!r}; tried: {[str(path) for path in candidates]}"
    )


def load_hand_pose_action_normalizer(
    action_normalization: ActionNormalizationMethod | None,
    *,
    embodiment_type: str,
    pose_convention: str,
    rotation_format: str,
    apply_forward_clamp: bool,
    action_stats_path: str | None,
) -> ActionNormalizer | None:
    """Load the configured hand-pose action normalizer, if requested."""
    if action_normalization is None:
        return None

    stats_path = _resolve_action_stats_path(
        action_stats_path,
        embodiment_type=embodiment_type,
        pose_convention=pose_convention,
        rotation_format=rotation_format,
    )
    normalizer = load_action_normalizer(
        action_normalization,
        stats_path=stats_path,
        apply_forward_clamp=apply_forward_clamp,
    )
    log.info(f"Loaded hand-pose action stats from {stats_path} with action_normalization={action_normalization}")
    return normalizer
