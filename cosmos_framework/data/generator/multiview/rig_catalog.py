# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RGB/LiDAR data contracts and transforms; no dataset locations or remote catalog.

Adapted from the corresponding upstream implementation.
"""

MADS_AML_ALT_RIG: list[str] = [
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
]


MADS_MODERATE_PLUS_RIGS: list[list[str]] = [
    # 1a: Front-wide only
    ["camera_front_wide_120fov"],
    # 2a: Forward dual
    ["camera_front_wide_120fov", "camera_front_tele_30fov"],
    # 4a: Four-camera fisheye surround
    [
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
    # 4b: Rectilinear mini-surround (approx)
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_tele_30fov",
        "camera_cross_left_120fov",
    ],
    # 5a: Surround without telephoto
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
    ],
    # 5c: Road + around-view
    [
        "camera_front_wide_120fov",
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
    # 6a: Six-camera surround (approximate)
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_rear_fisheye_200fov",
    ],
    # 6b: Surround 6
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_cross_left_120fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
    # 7a: Seven-camera surround
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_rear_fisheye_200fov",
    ],
    # 7b: Alternate seven-camera surround
    MADS_AML_ALT_RIG,
    # 8a: Eight-camera surround
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
    # 8b: Alternate eight-camera surround
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
    ],
    # 10a: 10-cam (6 pinhole + 4 fisheye)
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
    # 10b: 10 (no rear fisheye)
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
    ],
    # 11a: Full eleven-camera rig
    [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    ],
]


MADS_MODERATE_PLUS_RIGS_MAX_6_VIEWS: list[list[str]] = [rig for rig in MADS_MODERATE_PLUS_RIGS if len(rig) <= 6]
