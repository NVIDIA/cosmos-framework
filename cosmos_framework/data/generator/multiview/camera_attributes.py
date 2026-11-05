# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The pretrained Auto reference rig: camera identities, orientations and fields of view.

Kept in its own stdlib-only module so it can be released alongside caption_format.py.
The per-view caption header the model is trained on quotes these values verbatim --
camera identity, orientation and field of view -- so a runtime that hand-maintains its
own copy can drift from training silently: the captions stay well formed and merely
describe the wrong rig.
"""

from typing import Final

CAMERA_ATTRIBUTES: Final[dict[str, dict[str, str | int]]] = {
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


# The physical rig's view IDs. Written out rather than derived from the ordering of
# CAMERA_ATTRIBUTES above: these index the trained model's view embeddings, so a cosmetic
# reordering of that table must not silently renumber the rig. camera_attributes_test.py pins the
# two against each other.
#
# It lives here, beside the attributes, because this module is the released leaf -- cosmos-framework
# generates its copy from this file. Defining it only in multiview_data_source.py (which is not
# release-mapped) left cosmos-framework's consumers importing a symbol that the release removed on
# every sync, and that it only ever had because someone had hand-added it to the generated file.
CAMERA_VIEW_MAPPING: Final[dict[str, int]] = {
    "camera_front_wide_120fov": 0,
    "camera_cross_right_120fov": 1,
    "camera_rear_right_70fov": 2,
    "camera_rear_tele_30fov": 3,
    "camera_rear_left_70fov": 4,
    "camera_cross_left_120fov": 5,
    "camera_front_tele_30fov": 6,
    "camera_front_fisheye_200fov": 7,
    "camera_left_fisheye_200fov": 8,
    "camera_right_fisheye_200fov": 9,
    "camera_rear_fisheye_200fov": 10,
}

