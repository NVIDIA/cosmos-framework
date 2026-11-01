# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bridge native Action windows to the multiview unified-wrist-slot layout.

Camera poses have already been converted from end-effector targets at the
sampler boundary. Reuse the existing slot scatter, metric-gripper normalization and
codec, retaining one camera-major row block per view and all native channels
for the single-view human streams.
"""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.camera_views import (
    CAMERA_VIEW_ACTION_SLOTS,
    camera_action_caption_subject,
)
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    EGO_POSE,
    LEFT_OPENNESS,
    LEFT_WRIST_POSE,
    RIGHT_OPENNESS,
    RIGHT_WRIST_POSE,
    UNIFIED_ACTION_DIM,
)
from cosmos_framework.data.generator.multiview.camera_pose_actions import (
    CameraActionSlot,
    CameraActionSlotBinding,
    CameraActionSlots,
    build_unified_camera_slot_action,
)

_POSE_SLOTS: dict[CameraActionSlot, slice] = {
    "ego": EGO_POSE,
    "right_wrist": RIGHT_WRIST_POSE,
    "left_wrist": LEFT_WRIST_POSE,
}


def _camera_action_slot(
    camera: str,
    camera_pose_actions: Mapping[str, np.ndarray | None] | None,
    valid_mask: torch.Tensor | None,  # [D_raw]
) -> CameraActionSlot | None:
    """Share the target-availability decision between captions and slot projection."""
    if camera not in CAMERA_VIEW_ACTION_SLOTS:
        raise ValueError(f"No audited wrist/ego slot mapping for camera {camera!r}")
    slot = CAMERA_VIEW_ACTION_SLOTS[camera]
    if camera_pose_actions is not None:
        return slot if camera_pose_actions[camera] is not None else None
    if valid_mask is not None and not bool(valid_mask[_POSE_SLOTS[slot]].all()):
        return None
    return slot


def camera_action_caption_subjects(
    sample: dict[str, Any],
    action_valid_mask: torch.Tensor | None,  # [D_raw]
) -> list[str]:
    """Render camera-major subjects from the same bindings used by the action projection."""
    return [
        camera_action_caption_subject(
            camera, _camera_action_slot(camera, sample.get("camera_pose_actions"), action_valid_mask)
        )
        for camera in sample["camera_view_names"]
    ]


def split_action_by_camera(sample: dict[str, Any]) -> dict[str, Any]:
    """Select each camera's existing canonical slots without replicating both arms."""
    camera_names = sample.pop("camera_view_names")
    pose_convention = sample.pop("camera_action_pose_convention")
    gripper_width = sample.pop("camera_action_gripper_width_m")
    camera_pose_actions = sample.pop("camera_pose_actions", None)
    if len(camera_names) == 1:
        # Native human, camera/AV and single-wrist targets already occupy their
        # trained slots, including fingertips, route/history and grippers.
        return sample
    raw = sample["action_raw"]
    if raw.shape[-1] != UNIFIED_ACTION_DIM:
        raise ValueError("Per-camera wrist slots require the unified_v1 Action schema")
    valid_mask = sample["action_valid_mask"][:UNIFIED_ACTION_DIM]
    per_view_poses: list[np.ndarray | None] = []
    bindings: dict[str, CameraActionSlotBinding] = {}
    missing_views: list[int] = []
    for view, camera in enumerate(camera_names):
        slot = _camera_action_slot(camera, camera_pose_actions, valid_mask)
        if slot is None:
            per_view_poses.append(None)
            missing_views.append(view)
            continue
        measured_pose = (
            camera_pose_actions[camera] if camera_pose_actions is not None else raw[:, _POSE_SLOTS[slot]].cpu().numpy()
        )  # [T,9]
        per_view_poses.append(measured_pose)
        if slot == "ego":
            bindings[camera] = CameraActionSlotBinding(slot="ego")
        else:
            openness_slot = RIGHT_OPENNESS if slot == "right_wrist" else LEFT_OPENNESS
            if not bool(valid_mask[openness_slot]):
                raise ValueError(f"Wrist camera {camera!r} has no calibrated gripper openness")
            bindings[camera] = CameraActionSlotBinding(
                slot=slot,
                openness=raw[:, openness_slot].cpu().numpy(),
                openness_frame_offset=0,
            )
    projected = build_unified_camera_slot_action(
        per_view_poses,
        frame_indices=list(range(raw.shape[0] + 1)),
        action_slots=CameraActionSlots(bindings, gripper_width),
        camera_keys=camera_names,
        pose_convention=cast(PoseConvention, pose_convention),
        max_action_dim=sample["action"].shape[-1],
    )
    sample.update(
        action=projected.action,
        action_raw=projected.action_raw,
        raw_action_dim=projected.raw_action_dim,
        action_valid_mask=projected.action_valid_mask,
        action_processing_record=projected.action_processing_record,
        num_views_per_action_item=torch.tensor(len(camera_names), dtype=torch.int64),
    )
    plan = sample["sequence_plan"]
    source_conditions = list(plan.condition_frame_indexes_action)
    rows_per_view = raw.shape[0]
    plan.condition_frame_indexes_action = [
        view * rows_per_view + frame
        for view in range(len(camera_names))
        for frame in (range(rows_per_view) if view in missing_views else source_conditions)
    ]
    return sample
