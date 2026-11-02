# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physical camera names shared by native geometry and per-view Action packing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos_framework.data.generator.multiview.camera_pose_actions import CameraActionSlot

# These are physical camera names from the native Lance adapters, not inferred
# left/right substrings (DROID's left wrist lens is on its single right arm).
CAMERA_VIEW_ACTION_SLOTS: dict[str, CameraActionSlot] = {
    "observation.image.wrist_image_left": "right_wrist",
    "observation.image.exterior_image_1_left": "ego",
    "observation.image.exterior_image_2_left": "ego",
    "observation.images.top_head": "ego",
    "observation.images.head": "ego",
    "observation.images.hand_left": "left_wrist",
    "observation.images.hand_right": "right_wrist",
    "observation.images.top": "ego",
    "observation.images.left": "left_wrist",
    "observation.images.right": "right_wrist",
    "observation.images.top_camera-images-rgb": "ego",
    "observation.images.left_camera-images-rgb": "left_wrist",
    "observation.images.right_camera-images-rgb": "right_wrist",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
    "observation.images.faceImg": "ego",
    "observation.images.leftImg": "left_wrist",
    "observation.images.rightImg": "right_wrist",
    "observation.image.left_main_camera_rgb": "left_wrist",
    "observation.image.right_main_camera_rgb": "right_wrist",
    "observation.rgb.zed_link_camera_0": "ego",
    "observation.rgb.left_realsense_link_camera_0": "left_wrist",
    "observation.rgb.right_realsense_link_camera_0": "right_wrist",
    "observation.images.cam_high": "ego",
    "observation.images.cam_left_wrist": "left_wrist",
    "observation.images.cam_right_wrist": "right_wrist",
    "observation.images.camera_top_left": "ego",
    "observation.images.camera_wrist_left": "left_wrist",
    "observation.images.camera_wrist_right": "right_wrist",
}
_CAMERA_KEY_BY_SLOT: dict[CameraActionSlot, str] = {
    "ego": "camera_head",
    "left_wrist": "camera_wrist_left",
    "right_wrist": "camera_wrist_right",
}

# Physical roles are separate from action slots: exterior, head and fixed top
# cameras can all occupy ego slots without being the same kind of camera.
CAMERA_CAPTION_PREFIXES: dict[str, str] = {
    "camera_wrist_right": "The view is from the camera mounted on the right wrist.",
    "camera_wrist_left": "The view is from the camera mounted on the left wrist.",
    "camera_head": "The view is from the camera mounted on the robot head.",
    "camera_exterior_1": "The view is from the first static exterior camera.",
    "camera_exterior_2": "The view is from the second static exterior camera.",
    "camera_top": "The view is from the fixed camera above the workstation.",
    "camera_front_static": "The view is from the fixed front camera.",
}
_PHYSICAL_CAMERA_KEY_BY_NATIVE_VIEW: dict[str, str] = {
    view: _CAMERA_KEY_BY_SLOT[slot] for view, slot in CAMERA_VIEW_ACTION_SLOTS.items() if slot != "ego"
}
_PHYSICAL_CAMERA_KEY_BY_NATIVE_VIEW.update(
    {
        "observation.image.exterior_image_1_left": "camera_exterior_1",
        "observation.image.exterior_image_2_left": "camera_exterior_2",
        "observation.images.top_head": "camera_head",
        "observation.images.head": "camera_head",
        "observation.images.top": "camera_top",
        "observation.images.top_camera-images-rgb": "camera_top",
        "observation.images.faceImg": "camera_front_static",
        "observation.rgb.zed_link_camera_0": "camera_head",
        "observation.images.cam_high": "camera_top",
        "observation.images.camera_top_left": "camera_head",
    }
)


def native_camera_caption_prefixes(view_names: Sequence[str]) -> dict[str, str]:
    """Describe known physical cameras; never guess a role for an unknown stream."""
    return {
        view: CAMERA_CAPTION_PREFIXES[_PHYSICAL_CAMERA_KEY_BY_NATIVE_VIEW[view]]
        for view in view_names
        if view in _PHYSICAL_CAMERA_KEY_BY_NATIVE_VIEW
    }


def camera_view_key(view_name: str) -> str:
    """Resolve a native observation name to its canonical optical camera key."""
    return _CAMERA_KEY_BY_SLOT[CAMERA_VIEW_ACTION_SLOTS[view_name]]


def camera_action_caption_subject(view_name: str, slot: CameraActionSlot | None) -> str:
    """Describe only the action components allocated to this physical view."""
    if slot is None:
        return "There is no action target."
    if slot == "ego":
        subject = (
            "head ego motion"
            if _PHYSICAL_CAMERA_KEY_BY_NATIVE_VIEW[view_name] == "camera_head"
            else "camera ego motion"
        )
        return f"The action targets are the {subject}, mapped to the ego-pose."
    side = "right" if slot == "right_wrist" else "left"
    return (
        f"The action targets are the {side} wrist-mounted camera motion and gripper openness, "
        f"mapped to the {side} wrist-pose and {side}-openness components."
    )
