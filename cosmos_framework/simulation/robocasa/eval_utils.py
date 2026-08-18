# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared helpers for RoboCasa closed-loop evaluation.

Action decoders that turn the policy's output into RoboCasa's native 12-D env action, the HTTP
call to the policy server, and the base64 -> frames decoder for the model's generated video.
Imported by ``closed_loop_eval.py``; not a script.
"""

import base64
import io
import json

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation as R



# ----------------------------- action decode -----------------------------

def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """6D rotation (first two columns, Zhou 2019) -> 3x3 via Gram-Schmidt."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def decode_10d_to_env12(a10: np.ndarray, gripper_flip: bool) -> np.ndarray:
    """[pos(3), rot6d(6), gripper(1)] -> env 12D [Δpos, Δrotvec, grip, base4=0, mode=-1]."""
    a10 = np.asarray(a10, dtype=np.float64)
    pos = a10[:3]
    rotvec = R.from_matrix(rot6d_to_matrix(a10[3:9])).as_rotvec()
    grip = -a10[9] if gripper_flip else a10[9]
    env = np.zeros(12, dtype=np.float32)
    env[0:3] = pos
    env[3:6] = rotvec
    env[6] = float(np.clip(grip, -1.0, 1.0))
    env[7:11] = 0.0        # base_motion (fixed base)
    env[11] = -1.0         # control_mode = arm
    return env


# Normalised base command in [-1, 1] maps onto these full-deflection velocities
# (calibrated on NavigateKitchen; see BASE_MAX_VELOCITY in robocasa_lerobot_dataset.py).
BASE_MAX_VELOCITY = (0.601, 0.640, 1.251)  # forward m/s, side m/s, yaw rad/s
CONTROL_DT = 1.0 / 20.0


def decode_20d_to_env12(a20: np.ndarray, gripper_flip: bool) -> np.ndarray:
    """Mobile-base 20D policy output -> env 12D action.

    Policy layout (dataset order, base first)::

        [base_pos(3), base_rot6d(6), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]

    Env layout (arm first — NOT the same order)::

        [eef_pos(3), eef_rotvec(3), gripper(1), base_motion(4), control_mode(1)]

    The base block is an ego-frame pose delta, so it converts straight to the normalised
    velocity command without any world->body rotation: ``cmd = (delta / dt) / v_max``.
    ``base_motion[3]`` (torso height) stays 0 — it is never actuated in target/atomic.
    When the mode channel says arm-only, the base command is forced to zero so a small
    spurious base delta cannot creep in.
    """
    a20 = np.asarray(a20, dtype=np.float64)
    env = decode_10d_to_env12(a20[10:20], gripper_flip)  # arm block reuses the 10D path

    base_mode = float(a20[9]) > 0.0
    env[11] = 1.0 if base_mode else -1.0
    if base_mode:
        d_pos = a20[0:3]  # ego-frame translation delta (forward, side, up)
        yaw = float(R.from_matrix(rot6d_to_matrix(a20[3:9])).as_rotvec()[2])
        cmd = np.array(
            [
                d_pos[0] / CONTROL_DT / BASE_MAX_VELOCITY[0],
                d_pos[1] / CONTROL_DT / BASE_MAX_VELOCITY[1],
                yaw / CONTROL_DT / BASE_MAX_VELOCITY[2],
            ]
        )
        env[7:10] = np.clip(cmd, -1.0, 1.0)
        env[10] = 0.0  # torso height: never actuated
    return env


def decode_15d_to_env12(a15: np.ndarray, gripper_flip: bool) -> np.ndarray:
    """Raw-base 15D policy output -> env 12D action.

    Policy layout (``base_encoding='raw'``)::

        [base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]

    The base block IS the env's native ``base_motion`` command, so this is a straight
    copy — no dt, no velocity calibration, no controller inversion, and replaying a
    recorded demo through this path is exact. Contrast ``decode_20d_to_env12``, which
    has to invert a lagging velocity controller and therefore cannot round-trip.

    ``base_motion`` is already normalised to [-1, 1] in the data; the clip only guards
    against the policy overshooting the valid command range. When the mode channel says
    arm-only the base command is zeroed, matching the 20D path.
    """
    a15 = np.asarray(a15, dtype=np.float64)
    env = decode_10d_to_env12(a15[5:15], gripper_flip)  # arm block reuses the 10D path

    base_mode = float(a15[4]) > 0.0
    env[11] = 1.0 if base_mode else -1.0
    env[7:11] = np.clip(a15[0:4], -1.0, 1.0) if base_mode else 0.0
    return env


# ----------------------------- observation -----------------------------



def b64_png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict(server_url: str, composite: np.ndarray, prompt: str, image_size: int, timeout: float,
            state: list[float] | None = None) -> dict:
    payload = {"image": b64_png(composite), "prompt": prompt,
               "domain_name": "robocasa", "image_size": image_size}
    if state is not None:
        payload["state"] = state  # 10D eef proprioception -> clean conditioning token
    resp = requests.post(f"{server_url}/predict", json=payload,
                         headers={"Content-Type": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    if result.get("error"):
        raise RuntimeError(f"server error: {result['error']}")
    return result


def decode_pred_video(video_b64_list) -> list[np.ndarray]:
    frames = []
    for b in video_b64_list or []:
        raw = base64.b64decode(b.split(",", 1)[-1])
        frames.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
    return frames


# ----------------------------- rollout -----------------------------





