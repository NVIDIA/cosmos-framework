# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

import numpy as np

HAND_RIGHT_POSITION_KEY = "observation.state.hand_right_cam"


HAND_RIGHT_ROTATION_KEY = "observation.state.hand_right_cam_rotation"


HAND_LEFT_POSITION_KEY = "observation.state.hand_left_cam"


HAND_LEFT_ROTATION_KEY = "observation.state.hand_left_cam_rotation"


CAM_POSITION_KEY = "observation.state.camera_position"


CAM_ROTATION_KEY = "observation.state.camera_rotation"


NUM_JOINTS = 21


QUAT_DIM_PER_JOINT = 4  # quaternion (qx, qy, qz, qw) as stored in parquet


MAT_DIM_PER_JOINT = 9  # flattened 3x3 rotation matrix (internal working format)


WRIST_JOINT_IDX = 0


FINGERTIP_JOINT_IDXS = (4, 8, 12, 16, 20)


ROTATION_FORMAT_DIM: dict[str, int] = {"rot9d": 9, "rot6d": 6, "euler_xyz": 3}


NO_ACTION_SKIP_LABELS: tuple[str, ...] = ("no action", "no actions")


NO_ACTION_SKIP_LABEL_PREFIXES: tuple[str, ...] = ("hold", "adjust")


NO_ACTION_SKIP_LABEL_SUBSTRINGS: tuple[str, ...] = ("idle",)


WRIST_FRAME_ALIGN_embodiment_a = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


WRIST_FRAME_ALIGN_embodiment_a_IDENTITY = np.eye(4, dtype=np.float32)
