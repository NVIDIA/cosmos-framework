# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The MADS rig: what each camera is, where it points, and how wide it sees.

Kept in its own stdlib-only module so it can be released alongside caption_format.py.
The per-view caption header the model is trained on quotes these values verbatim --
camera identity, orientation and field of view -- so a runtime that hand-maintains its
own copy can drift from training silently: the captions stay well formed and merely
describe the wrong rig.
"""

from typing import Final

MADS_CAMERA_ATTRIBUTES: Final[dict[str, dict[str, str | int]]] = {
    "camera_front_wide_120fov": {
        "camera_role": "front",
        "camera_type": "wide",
        "facing": "forward",
        "fov_degrees": 120,
    },
    "camera_cross_right_120fov": {
        "camera_role": "right_side",
        "camera_type": "wide",
        "facing": "right",
        "fov_degrees": 120,
    },
    "camera_rear_right_70fov": {
        "camera_role": "rear_right",
        "camera_type": "standard",
        "facing": "rear_right",
        "fov_degrees": 70,
    },
    "camera_rear_tele_30fov": {
        "camera_role": "rear",
        "camera_type": "telephoto",
        "facing": "backward",
        "fov_degrees": 30,
    },
    "camera_rear_left_70fov": {
        "camera_role": "rear_left",
        "camera_type": "standard",
        "facing": "rear_left",
        "fov_degrees": 70,
    },
    "camera_cross_left_120fov": {
        "camera_role": "left_side",
        "camera_type": "wide",
        "facing": "left",
        "fov_degrees": 120,
    },
    "camera_front_tele_30fov": {
        "camera_role": "front",
        "camera_type": "telephoto",
        "facing": "forward",
        "fov_degrees": 30,
    },
    "camera_front_fisheye_200fov": {
        "camera_role": "front",
        "camera_type": "fisheye",
        "facing": "forward",
        "fov_degrees": 200,
    },
    "camera_left_fisheye_200fov": {
        "camera_role": "left_side",
        "camera_type": "fisheye",
        "facing": "left",
        "fov_degrees": 200,
    },
    "camera_right_fisheye_200fov": {
        "camera_role": "right_side",
        "camera_type": "fisheye",
        "facing": "right",
        "fov_degrees": 200,
    },
    "camera_rear_fisheye_200fov": {
        "camera_role": "rear",
        "camera_type": "fisheye",
        "facing": "backward",
        "fov_degrees": 200,
    },
}
