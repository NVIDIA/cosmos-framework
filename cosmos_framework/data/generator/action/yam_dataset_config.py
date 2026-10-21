# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import numpy as np

YAM_TO_OPENCV: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)  # [3,3]


ABC_CAMERA_FEATURES: dict[str, str] = {
    "top": "observation.images.top",
    "left": "observation.images.left_wrist",
    "right": "observation.images.right_wrist",
}


MOLMOACT2_CAMERA_FEATURES: dict[str, str] = {
    "left": "observation.images.left",
    "right": "observation.images.right",
    "top": "observation.images.top",
}


XDOF_CAMERA_FEATURES_480: dict[str, str] = {
    "left": "observation.images.left_camera-images-rgb",
    "right": "observation.images.right_camera-images-rgb",
    "top": "observation.images.top_camera-images-rgb",
}
