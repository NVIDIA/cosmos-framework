# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RGB/LiDAR data contracts and transforms; no dataset locations or remote catalog.

Adapted from the corresponding upstream implementation.
"""

from typing import Final

DEFAULT_CAMERAS: Final[tuple[str, ...]] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)


MADS_CAMERAS: Final[tuple[str, ...]] = (
    *DEFAULT_CAMERAS,
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)


DEFAULT_CAMERA_VIEW_MAPPING: Final[dict[str, int]] = dict(zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS))))


INDEX_TO_CAMERA_MAPPING: Final[dict[int, str]] = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_right_70fov",
    5: "camera_rear_tele_30fov",
    6: "camera_front_tele_30fov",
}


STRUCTURED_CAPTION_KEY_PREFIX: Final[str] = "structured_captions_qwen3-vl-8b-lora-v1.5-merged"


STRUCTURED_CAPTION_PROBABILITY: Final[dict[str, float]] = {"caption_structured": 1.0}


_IDX_OF: Final[dict[str, int]] = {cam: idx for idx, cam in INDEX_TO_CAMERA_MAPPING.items()}


TRANSFER_VIDEO_KEY_MAPPING: Final[dict[str, str]] = {cam: f"video_{_IDX_OF[cam]}" for cam in DEFAULT_CAMERAS}


TRANSFER_CAPTION_KEY_MAPPING: Final[dict[str, str]] = {cam: f"metas_{cam}" for cam in DEFAULT_CAMERAS}


TRANSFER_CONTROL_KEY_MAPPING: Final[dict[str, str]] = {cam: f"world_scenario_{_IDX_OF[cam]}" for cam in DEFAULT_CAMERAS}


MADS_V1_VIDEO_KEY_MAPPING: Final[dict[str, str]] = {cam: f"video_{cam}" for cam in MADS_CAMERAS}


MADS_V1_CAPTION_KEY_MAPPING: Final[dict[str, str]] = {
    cam: f"{STRUCTURED_CAPTION_KEY_PREFIX}_{cam}" for cam in MADS_CAMERAS
}


MADS_V1_CONTROL_KEY_MAPPING: Final[dict[str, str]] = {cam: f"world_scenario_v4p0p1_{cam}" for cam in MADS_CAMERAS}


MADS_V1_CAPTION_PROBABILITY: Final[dict[str, float]] = STRUCTURED_CAPTION_PROBABILITY
