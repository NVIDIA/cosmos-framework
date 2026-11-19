# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""RoboLab joint-position interfaces, independent of Isaac and model libraries."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cosmos_framework.posttrain.embodied.embodiment import EmbodimentMap, register_embodiment


def validate_control_freq(robot: str, control_freq: int) -> None:
    if robot != "droid" or control_freq != 15:
        raise ValueError("RoboLab DROID uses joint-position actions at 15 Hz")


def get_embodiment_map(robot: str, control_mode: str = "joint_pos") -> EmbodimentMap:
    if (robot, control_mode) != ("droid", "joint_pos"):
        raise ValueError("Expected RoboLab DROID with joint-position control")
    cameras = {
        "left": "over_shoulder_left_camera",
        "right": "over_shoulder_right_camera",
        "wrist": "wrist_cam",
        "head": "head_camera",
        "viewport": "viewport",
    }
    return register_embodiment(EmbodimentMap("robolab_droid", cameras, "state", 8, 8))


def to_numpy(value: Any) -> np.ndarray:
    # Detach native engine tensors at the simulation boundary; only NumPy crosses it.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_observation(
    raw: Mapping[str, Any], lane: int, robot: str, prompt: str, *, cached_images: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Extract one physical environment's native tensors as independent NumPy arrays.

    ``lane`` indexes the native batch, which may differ from the resumed session's
    logical lane number. Model-specific image and action transforms belong to policy serving code.
    ``cached_images`` reuses owned CPU images between policy queries when no video is
    recorded. Proprioception is always current; native GPU buffers are never retained.
    """
    result = {
        key: cached_images[key] if cached_images is not None else to_numpy(value[lane]).copy()
        for key, value in raw["image_obs"].items()
    }
    proprio = raw["proprio_obs"]
    validate_control_freq(robot, 15)
    joints = to_numpy(proprio["arm_joint_pos"][lane]).reshape(-1)
    state = np.concatenate((joints, to_numpy(proprio["gripper_pos"][lane]).reshape(-1)))
    for key in ("ee_pos", "ee_quat"):
        if key in proprio:
            result[key] = to_numpy(proprio[key][lane]).reshape(-1).copy()
    viewport = raw["viewport_cam"]
    if len(viewport) != 1:
        raise ValueError("RoboLab joint-position interface requires exactly one viewport camera")
    result.update(state=state.astype(np.float32), prompt=prompt)
    result["viewport"] = (
        cached_images["viewport"] if cached_images is not None else to_numpy(next(iter(viewport.values()))[lane]).copy()
    )
    return result
