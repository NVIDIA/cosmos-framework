# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Closed-loop evaluation for LIBERO using the Action HTTP inference server.

# Single-view example (agentview camera):
PYTHONPATH=. python cosmos_framework/simulation/libero/closed_loop_eval.py \
  --server_url http://localhost:8000 \
  --task_suite libero_10 \
  --num_trials_per_task 10 \
  --action_horizon 16 \
  --camera agentview \
  --save_gifs --gif_fps 20 \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --action_dim 10 \
  --output_dir results/libero_closed_loop_10_single_view

# Multi-view example (agentview + wrist cameras):
PYTHONPATH=. python cosmos_framework/simulation/libero/closed_loop_eval.py \
  --server_url http://localhost:8000 \
  --task_suite libero_goal \
  --num_trials_per_task 2 \
  --action_horizon 16 \
  --camera agentview,wrist \
  --save_gifs --gif_fps 20 \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --action_dim 10 \
  --output_dir results/libero_closed_loop_goal_multiview
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.libero_pose_utils import (
    libero_rotation_format,
    libero_rotation_space_from_action_dim,
)
from cosmos_framework.data.generator.action.pose_utils import convert_rotation
from cosmos_framework.data.generator.action.viewpoint_utils import DEFAULT_VIEWPOINT_TEMPLATES

benchmark: Any
get_libero_path: Any
OffScreenRenderEnv: Any


TASK_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


_CAMERA_PROMPT_NAMES: dict[str, str] = {
    "agentview": "third-person view",
    "wrist": "wrist-mounted camera",
}


def _append_prompt_sentence(prompt: str, sentence: str) -> str:
    """Append one metadata sentence using the same separator convention as training augmentors."""
    if sentence in prompt:
        return prompt
    prompt = prompt.rstrip()
    if not prompt:
        return sentence.rstrip()
    separator = " " if prompt.rstrip().endswith(".") else ". "
    return prompt + separator + sentence.rstrip()


def _concat_view_layout_description(cameras: list[str]) -> str:
    """Describe the horizontal camera layout sent by ``ActionEnvironmentClient``."""
    camera_names = [_CAMERA_PROMPT_NAMES[camera] for camera in cameras]
    if len(camera_names) == 2:
        return f"The left half shows the {camera_names[0]}; the right half shows the {camera_names[1]}."
    layout = ", ".join(camera_names)
    return f"The views are concatenated horizontally from left to right as: {layout}."


def _augment_task_prompt_with_viewpoint(task_description: str, cameras: list[str]) -> str:
    """Concat-view caption augmentation for closed-loop LIBERO eval."""
    if len(cameras) <= 1:
        return task_description
    prompt = _append_prompt_sentence(task_description, DEFAULT_VIEWPOINT_TEMPLATES["concat_view"])
    return _append_prompt_sentence(prompt, _concat_view_layout_description(cameras))


def _rotation_repr_to_mat(rotation: np.ndarray, rotation_space: str) -> np.ndarray:
    """Convert a single LIBERO rotation block to a 3x3 rotation matrix."""
    matrix = convert_rotation(
        rotation,
        libero_rotation_format(rotation_space),
        "matrix",
        normalize_matrix=rotation_space != "3d",
    )
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected NumPy rotation matrix, got {type(matrix)!r}")
    return matrix


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    error: str | None
    actions: list[list[float]]


class ActionEnvironmentClient:
    """Client for interacting with the Action model server."""

    server_url: str
    domain_name: str
    prompt: str
    image_size: int
    timeout: float

    def __init__(
        self,
        server_url: str,
        domain_name: str,
        prompt: str,
        image_size: int,
        timeout: float,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.domain_name = domain_name
        self.prompt = prompt
        self.image_size = image_size
        self.timeout = timeout

    def check_health(self) -> bool:
        """Check if the model server is healthy."""
        try:
            resp = requests.get(f"{self.server_url}/", timeout=5.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_info(self) -> dict[str, str]:
        """Get model server info."""
        resp = requests.get(f"{self.server_url}/info", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def notify_next_episode(self) -> None:
        """Notify server to advance to next episode (used with dataset action server)."""
        try:
            requests.post(
                f"{self.server_url}/next_episode",
                json={"prompt": self.prompt},
                timeout=5.0,
            )
        except requests.RequestException:
            pass

    def encode_image(self, image: np.ndarray) -> str:
        """Encode a numpy image (H, W, 3) uint8 to base64 PNG, resizing to image_size."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def encode_image_raw(self, image: np.ndarray) -> str:
        """Encode a numpy image (H, W, 3) uint8 to base64 PNG without resizing."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model input size."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        return np.array(pil_img)

    def concatenate_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Resize each image and concatenate horizontally (side-by-side).

        Args:
            images: List of images with shape (H, W, 3).

        Returns:
            Concatenated image with shape (image_size, image_size*num_views, 3).
        """
        resized = [self.resize_image(img) for img in images]
        return np.concatenate(resized, axis=1)

    def predict(self, observation: np.ndarray | list[np.ndarray]) -> dict[str, Any]:
        """Send observation(s) to model server and get predicted actions.

        Args:
            observation: Single image as np.ndarray or list of images for multi-view.
                For multi-view, images are resized and concatenated horizontally before sending.
        """
        if isinstance(observation, list):
            # Multi-view: resize each, concatenate horizontally, and send as single image
            concatenated = self.concatenate_images(observation)
            encoded = self.encode_image_raw(concatenated)
        else:
            # Single view: send single image
            encoded = self.encode_image(observation)

        payload = {
            "image": encoded,
            "prompt": self.prompt,
            "domain_name": self.domain_name,
            "image_size": self.image_size,
        }

        resp = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        result = resp.json()
        if "error" in result and result["error"]:
            raise RuntimeError(f"Model server error: {result['error']}")
        return result

    def predict_batch(self, observations: list[list[np.ndarray]]) -> list[list[list[float]]]:
        """Batched inference: a list of per-env multi-view observations -> ONE
        POST /predict_batch -> a list of action chunks (one per env). Used by the
        vectorized eval so N parallel envs share a single diffusion forward."""
        items = []
        for obs_imgs in observations:
            concat = self.concatenate_images(obs_imgs) if len(obs_imgs) > 1 else self.resize_image(obs_imgs[0])
            items.append(
                {
                    "image": self.encode_image_raw(concat),
                    "prompt": self.prompt,
                    "domain_name": self.domain_name,
                    "image_size": self.image_size,
                }
            )
        resp = requests.post(
            f"{self.server_url}/predict_batch",
            json={"items": items},
            headers={"Content-Type": "application/json"},
            timeout=max(self.timeout, 300.0),
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result and result["error"]:
            raise RuntimeError(f"Model server error: {result['error']}")
        return result["actions"]


def _find_accessible_dri_nodes() -> list[Path]:
    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return []
    nodes = list(dri_path.glob("renderD*")) + list(dri_path.glob("card*"))
    return [node for node in nodes if os.access(node, os.R_OK | os.W_OK)]


def _resolve_mujoco_backend(requested_backend: str) -> tuple[str, str]:
    requested_backend = requested_backend.lower()
    if requested_backend != "auto":
        return requested_backend, "requested"

    env_backend = os.environ.get("MUJOCO_GL")
    if env_backend:
        return env_backend.lower(), "env"

    if _find_accessible_dri_nodes():
        return "egl", "auto-gpu"
    return "osmesa", "auto-cpu"


def _configure_mujoco_env(requested_backend: str) -> str:
    backend, source = _resolve_mujoco_backend(requested_backend)
    if backend not in {"egl", "osmesa", "glfw"}:
        raise ValueError(f"Unsupported MuJoCo GL backend: {backend!r}. Use auto, egl, osmesa, or glfw.")

    os.environ["MUJOCO_GL"] = backend
    if backend == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    return f"{backend} ({source})"


def _import_libero() -> None:
    global benchmark, get_libero_path, OffScreenRenderEnv
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero import get_libero_path as libero_get_libero_path
        from libero.libero.envs import OffScreenRenderEnv as libero_offscreen_render_env
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError(
            "Failed to import LIBERO. Make sure the LIBERO environment is activated. "
            f"python={sys.executable!r}, import_error={exc!r}"
        ) from exc

    benchmark = libero_benchmark
    get_libero_path = libero_get_libero_path
    OffScreenRenderEnv = libero_offscreen_render_env


def _wait_for_server(client: ActionEnvironmentClient, timeout_s: float) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if client.check_health():
            return
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for server at {client.server_url}")


def _get_libero_env(
    task: Any,
    *,
    resolution: int,
    seed: int,
    render_gpu_device_id: int,
) -> tuple[Any, str]:
    task_description = str(task.language)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _get_libero_image(
    obs: dict[str, Any],
    camera: str,
    *,
    flip_images: bool,
    rotate_180: bool,
) -> np.ndarray:
    if camera == "agentview":
        image = obs["agentview_image"]
    elif camera == "wrist":
        image = obs["robot0_eye_in_hand_image"]
    else:
        raise ValueError(f"Unsupported camera={camera!r}. Use 'agentview' or 'wrist'.")

    if rotate_180:
        image = image[::-1, ::-1]
    if flip_images:
        image = np.flipud(image)
    return image


def _get_libero_images(
    obs: dict[str, Any],
    cameras: list[str],
    *,
    flip_images: bool,
    rotate_180: bool,
) -> list[np.ndarray]:
    """Get images from multiple cameras."""
    return [_get_libero_image(obs, camera, flip_images=flip_images, rotate_180=rotate_180) for camera in cameras]


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).round().astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _save_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    if not frames:
        return
    duration_ms = int(1000 / fps) if fps > 0 else 100
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def _decode_b64_frames(b64_frames: list[str]) -> list[Image.Image]:
    """Decode a list of base64-encoded PNG strings into PIL Images."""
    images: list[Image.Image] = []
    for b64 in b64_frames:
        raw = base64.b64decode(b64)
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def _save_comparison_gif(
    comparison_windows: list[tuple[list[Image.Image], list[Image.Image]]],
    output_path: Path,
    fps: int,
    target_height: int = 256,
    separator_width: int = 4,
) -> None:
    """Create and save a side-by-side comparison GIF (Action prediction | env rollout).

    Each window is a (action_frames, env_frames) pair from one prediction call.
    Frames are paired index-by-index; the conditioning frame (index 0) of
    subsequent windows is skipped to avoid duplicating the boundary frame.
    """
    from PIL import ImageDraw

    combined_frames: list[Image.Image] = []
    banner_h = 16

    for window_idx, (action_frames, env_frames) in enumerate(comparison_windows):
        n = min(len(action_frames), len(env_frames))
        start = 1 if window_idx > 0 else 0
        for i in range(start, n):
            action_img = action_frames[i]
            env_img = env_frames[i]

            action_w = int(action_img.width * target_height / action_img.height)
            env_w = int(env_img.width * target_height / env_img.height)
            action_resized = action_img.resize((action_w, target_height), Image.Resampling.BILINEAR)
            env_resized = env_img.resize((env_w, target_height), Image.Resampling.BILINEAR)

            total_w = action_w + separator_width + env_w
            total_h = target_height + banner_h
            combined = Image.new("RGB", (total_w, total_h), color=0)

            draw = ImageDraw.Draw(combined)
            draw.rectangle([(0, 0), (action_w, banner_h)], fill=(30, 30, 60))
            draw.rectangle([(action_w + separator_width, 0), (total_w, banner_h)], fill=(30, 60, 30))
            draw.text((4, 1), "Action Prediction", fill=(100, 180, 255))
            draw.text((action_w + separator_width + 4, 1), "Environment", fill=(100, 255, 100))

            combined.paste(action_resized, (0, banner_h))
            combined.paste(env_resized, (action_w + separator_width, banner_h))
            combined_frames.append(combined)

    if combined_frames:
        _save_gif(combined_frames, output_path, fps)


def _select_action_chunk(actions: list[list[float]], action_horizon: int) -> list[list[float]]:
    if action_horizon <= 0 or action_horizon >= len(actions):
        return actions
    return actions[:action_horizon]


def _format_action(action: list[float], action_dim: int) -> list[float]:
    if len(action) < action_dim:
        raise ValueError(f"Action dimension {len(action)} smaller than expected {action_dim}")
    return action[:action_dim]


def _remap_gripper(action: list[float], mode: str) -> list[float]:
    """Map the model's gripper command to the LIBERO env's [-1, 1] (negative = open).

    The right mapping depends on the gripper convention of the dataset the policy
    was trained on (the server denormalizes back to that raw convention):

    * ``zero_one`` (NVIDIA LIBERO_LeRobot_v3): raw gripper in [0, 1]; the env wants
      [-1, 1] with negative=open. The i4/cosmos-rl reference BINARIZES this to hard
      {-1, +1} via ``-sign(2g - 1)`` (not the continuous ``1 - 2g`` from issue #50).
      For a confident policy the two agree (g~0/1), but an undertrained policy emits
      g~0.5 where continuous ``1-2g``~0 never actuates the gripper -> grasps fail.
      Binarizing matches the reference and is robust to weak checkpoints.
    * ``pm_one`` (community ``lerobot/libero_*``): raw gripper already in {-1, +1}
      (robosuite convention) -> pass through (clamped).
    * ``pm_one_flip``: {-1, +1} but with inverted open/close sign.
    """
    action = list(action)  # avoid mutating the caller's list
    g = action[-1]
    if mode == "zero_one":
        action[-1] = max(-1.0, min(1.0, g * 2.0 - 1.0)) * -1.0  # [0,1] -> [-1,1], negative=open (issue #50)
    elif mode == "pm_one":
        action[-1] = max(-1.0, min(1.0, g))
    elif mode == "pm_one_flip":
        action[-1] = max(-1.0, min(1.0, -g))
    else:
        raise ValueError(f"Unknown gripper_mode={mode!r}. Use zero_one/pm_one/pm_one_flip.")
    return action


def _infer_rotation_space(action_dim: int, rotation_space: str) -> str:
    if rotation_space != "auto":
        return rotation_space
    return libero_rotation_space_from_action_dim(action_dim)


def _obs_to_pose(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    rotation = R.from_quat(quat).as_matrix()
    return position, rotation


def _anchored_action_to_delta(
    anchored_action: np.ndarray,
    base_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    rotation_space: str,
) -> np.ndarray:
    anchored_translation = anchored_action[:3]
    rotation_dim = anchored_action.shape[0] - 4
    anchored_rotation = anchored_action[3 : 3 + rotation_dim]
    gripper = anchored_action[3 + rotation_dim : 4 + rotation_dim]

    base_pos, base_rot = base_pose
    current_pos, current_rot = current_pose

    if rotation_space == "3d":
        anchored_rot = R.from_rotvec(anchored_rotation).as_matrix()
    elif rotation_space == "6d":
        anchored_rot = _rotation_repr_to_mat(anchored_rotation, rotation_space)
    elif rotation_space == "9d":
        anchored_rot = anchored_rotation.reshape(3, 3)
    else:
        raise ValueError(f"Unsupported rotation_space={rotation_space!r}. Use 3d/6d/9d.")
    target_rot = base_rot @ anchored_rot
    target_pos = base_pos + base_rot @ anchored_translation
    delta_pos = target_pos - current_pos
    delta_rot = target_rot @ current_rot.T
    delta_rotvec = R.from_matrix(delta_rot).as_rotvec()

    return np.concatenate([delta_pos, delta_rotvec, gripper], axis=0)


def _framewise_action_to_delta(
    framewise_action: np.ndarray,
    rotation_space: str,
) -> np.ndarray:
    """Convert a frame-wise policy action to LIBERO's 7D simulator command.

    Frame-wise actions are already per-step deltas in the LIBERO controller's
    convention (see ``LiberoDataset`` with ``action_space='frame_wise_relative'``),
    so the only conversion required is decoding the chosen rotation
    representation back to a rotation vector. No anchor/current pose is needed.
    """
    if rotation_space == "3d":
        return framewise_action

    translation = framewise_action[:3]
    rotation_dim = framewise_action.shape[0] - 4
    rotation_repr = framewise_action[3 : 3 + rotation_dim]
    gripper = framewise_action[3 + rotation_dim : 4 + rotation_dim]
    rotation_delta = _rotation_repr_to_mat(rotation_repr, rotation_space)

    delta_pos = translation
    delta_rotvec = R.from_matrix(rotation_delta).as_rotvec()
    return np.concatenate([delta_pos, delta_rotvec, gripper], axis=0)


def _run_episode(
    env: Any,
    client: ActionEnvironmentClient,
    *,
    cameras: list[str],
    flip_images: bool,
    rotate_180: bool,
    action_horizon: int,
    action_dim: int,
    action_space: str,
    rotation_space: str,
    gripper_mode: str,
    max_steps: int,
    warmup_steps: int,
    initial_state: np.ndarray | None,
    gif_path: Path | None,
    gif_fps: int,
    comparison_path: Path | None = None,
) -> EpisodeResult:
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue: list[list[float]] = []
    base_pose: tuple[np.ndarray, np.ndarray] | None = None
    step = 0
    success = False
    gif_frames: list[Image.Image] = []
    action_log: list[list[float]] = []
    is_multi_view = len(cameras) > 1
    resolved_rotation_space = _infer_rotation_space(action_dim, rotation_space)

    comparison_windows: list[tuple[list[Image.Image], list[Image.Image]]] = []

    def record_frame(current_obs: dict[str, Any]) -> None:
        if gif_path is None:
            return
        image = _get_libero_image(
            current_obs,
            cameras[0],
            flip_images=flip_images,
            rotate_180=rotate_180,
        )
        image = _ensure_uint8_image(image)
        gif_frames.append(Image.fromarray(image).convert("RGB"))

    def capture_comparison_frame(current_obs: dict[str, Any]) -> Image.Image:
        """Capture an env frame matching Action's input view (multi-view concatenated if applicable)."""
        if is_multi_view:
            imgs = _get_libero_images(current_obs, cameras, flip_images=flip_images, rotate_180=rotate_180)
            concat = client.concatenate_images(imgs)
            return Image.fromarray(_ensure_uint8_image(concat)).convert("RGB")
        img = _get_libero_image(current_obs, cameras[0], flip_images=flip_images, rotate_180=rotate_180)
        return Image.fromarray(_ensure_uint8_image(img)).convert("RGB")

    record_frame(obs)

    while step < max_steps:
        if step < warmup_steps:
            dummy = _get_libero_dummy_action()
            obs, _, _, _ = env.step(dummy)
            action_log.append(dummy)
            step += 1
            record_frame(obs)
            continue

        if not action_queue:
            if is_multi_view:
                observation_imgs = _get_libero_images(
                    obs,
                    cameras,
                    flip_images=flip_images,
                    rotate_180=rotate_180,
                )
                result = client.predict(observation_imgs)
            else:
                observation_img = _get_libero_image(
                    obs,
                    cameras[0],
                    flip_images=flip_images,
                    rotate_180=rotate_180,
                )
                result = client.predict(observation_img)
            actions = result.get("action", [])
            if not actions:
                return EpisodeResult(False, step, "Empty action chunk from server", action_log)
            action_queue = _select_action_chunk(actions, action_horizon)

            if comparison_path is not None:
                action_video_b64 = result.get("video", [])
                if action_video_b64:
                    action_frames = _decode_b64_frames(action_video_b64)
                    env_comparison_frames = [capture_comparison_frame(obs)]
                    comparison_windows.append((action_frames, env_comparison_frames))

            if action_space == "relative":
                base_pose = _obs_to_pose(obs)

        raw_action = _format_action(action_queue.pop(0), action_dim)
        if action_space == "relative":
            if base_pose is None:
                raise RuntimeError("Missing base pose for relative action conversion")
            current_pose = _obs_to_pose(obs)
            action = _anchored_action_to_delta(
                np.asarray(raw_action, dtype=np.float32),
                base_pose,
                current_pose,
                resolved_rotation_space,
            )
            action_list = action.tolist()
        else:
            action = _framewise_action_to_delta(
                np.asarray(raw_action, dtype=np.float32),
                resolved_rotation_space,
            )
            action_list = action.tolist()

        # Map the model's gripper command to the env's [-1, 1] per the dataset convention.
        action_list = _remap_gripper(action_list, gripper_mode)

        action_log.append(action_list)
        obs, _, done, info = env.step(action_list)
        step += 1
        record_frame(obs)

        if comparison_path is not None and comparison_windows:
            comparison_windows[-1][1].append(capture_comparison_frame(obs))

        if isinstance(info, dict) and info.get("success"):
            success = True
            break
        if done:
            success = True if not isinstance(info, dict) else bool(info.get("success", True))
            break

    if gif_path is not None:
        _save_gif(gif_frames, gif_path, gif_fps)
    if comparison_path is not None and comparison_windows:
        _save_comparison_gif(comparison_windows, comparison_path, gif_fps)
    return EpisodeResult(success, step, None, action_log)


def _load_initial_states(
    task_suite: Any,
    task_id: int,
    *,
    task_description: str,
    initial_states_path: str,
    episode_idx: int,
) -> np.ndarray | None:
    default_initial_states = task_suite.get_task_init_states(task_id)

    if initial_states_path == "DEFAULT":
        return np.array(default_initial_states[episode_idx])

    with open(initial_states_path, "r", encoding="utf-8") as f:
        all_initial_states = json.load(f)

    task_key = task_description.replace(" ", "_")
    episode_key = f"demo_{episode_idx}"
    if not all_initial_states[task_key][episode_key]["success"]:
        return None
    return np.array(all_initial_states[task_key][episode_key]["initial_state"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO closed-loop evaluation via Action HTTP server")
    parser.add_argument(
        "--server_url", type=str, required=True, help="Base URL for Action server (e.g., http://host:8000)"
    )
    parser.add_argument("--task_suite", type=str, default="libero_spatial", choices=sorted(TASK_MAX_STEPS.keys()))
    parser.add_argument("--num_trials_per_task", type=int, default=10)
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated task IDs to evaluate (default: all)")
    parser.add_argument("--image_size", type=int, default=256, help="Model input image size")
    parser.add_argument("--env_image_size", type=int, default=256, help="Environment render resolution")
    parser.add_argument("--action_horizon", type=int, default=0, help="Actions to execute per request (0=full chunk)")
    parser.add_argument("--action_dim", type=int, default=10, help="Action dimension for LIBERO")
    parser.add_argument(
        "--action_space",
        type=str,
        default="frame_wise_relative",
        choices=["relative", "frame_wise_relative"],
        help="Action space expected from the model (relative=anchored, frame_wise_relative=framewise deltas).",
    )
    parser.add_argument(
        "--rotation_space",
        type=str,
        default="auto",
        choices=["auto", "3d", "6d", "9d"],
        help="Rotation representation for anchored actions (auto infers from action_dim).",
    )
    parser.add_argument(
        "--gripper_mode",
        type=str,
        default="zero_one",
        choices=["zero_one", "pm_one", "pm_one_flip"],
        help="Gripper convention of the training data: 'zero_one' = [0,1] (NVIDIA "
        "LIBERO_LeRobot_v3, mapped 1-2g); 'pm_one' = {-1,+1} (community lerobot/libero_*, "
        "pass-through); 'pm_one_flip' = {-1,+1} with inverted sign.",
    )
    parser.add_argument("--domain_name", type=str, default="libero")
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera(s) to use. Single camera: 'agentview' or 'wrist'. Multiple cameras: comma-separated, e.g., 'agentview,wrist'.",
    )
    parser.add_argument("--flip_images", action="store_true", help="Flip images vertically before encoding")
    parser.add_argument(
        "--rotate_180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate images by 180 degrees before encoding (default: True; pass --no-rotate-180 to disable)",
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="Stabilization steps with dummy actions")
    parser.add_argument("--max_steps", type=int, default=0, help="Override max steps per episode (0=default)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument("--wait_timeout", type=float, default=60.0, help="Seconds to wait for server health")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_gifs", action="store_true", help="Save per-episode GIFs of rendered frames")
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison GIFs (Action prediction vs environment rollout)",
    )
    parser.add_argument("--gif_fps", type=int, default=20, help="Frames per second for saved GIFs")
    parser.add_argument(
        "--mujoco_gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
        help="MuJoCo GL backend (auto picks egl if /dev/dri is accessible, else osmesa).",
    )
    parser.add_argument(
        "--render_gpu_device_id",
        type=int,
        default=-1,
        help="GPU device index for EGL rendering (-1 uses default device).",
    )
    parser.add_argument(
        "--initial_states_path",
        type=str,
        default="DEFAULT",
        help='Path to initial states JSON. Use "DEFAULT" for benchmark defaults.',
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel LIBERO envs (SubprocVectorEnv). >1 runs trials in waves "
        "with ONE batched /predict_batch per control step (~num_envs x faster). 1 = serial.",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save evaluation summary JSON")
    return parser.parse_args()


class _LiberoEnvFactory:
    """Picklable env factory for SubprocVectorEnv under the spawn start method.

    spawn pickles each env_fn and re-imports this module in the child, so the
    factory must be a top-level class (lambdas/closures are not picklable). The
    child sets the GL backend and imports OffScreenRenderEnv locally so its EGL
    context is created fresh in the worker process."""

    def __init__(
        self,
        *,
        bddl_file_name: str,
        camera_heights: int,
        camera_widths: int,
        render_gpu_device_id: int,
        mujoco_gl: str,
    ) -> None:
        self.bddl_file_name = bddl_file_name
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.render_gpu_device_id = render_gpu_device_id
        self.mujoco_gl = mujoco_gl

    def __call__(self) -> Any:
        # Resolve to a concrete GPU; -1 (auto) makes EGL device selection race/fail
        # across spawned workers (EGLError / "'EGLGLContext' object has no attribute
        # '_context'"). Set the GL backend + pin the EGL device BEFORE importing
        # OffScreenRenderEnv (which dlopen's the GL stack at import).
        dev = self.render_gpu_device_id if self.render_gpu_device_id >= 0 else 0
        os.environ["MUJOCO_GL"] = self.mujoco_gl
        if self.mujoco_gl == "egl":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
            os.environ["EGL_DEVICE_ID"] = str(dev)
        elif self.mujoco_gl == "osmesa":
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv

        return _OffScreenRenderEnv(
            bddl_file_name=self.bddl_file_name,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
            render_gpu_device_id=dev,
        )


def _run_task_vectorized(
    task: Any,
    task_description: str,
    *,
    num_trials: int,
    num_envs: int,
    env_image_size: int,
    seed: int,
    render_gpu_device_id: int,
    client: ActionEnvironmentClient,
    cameras: list[str],
    flip_images: bool,
    rotate_180: bool,
    action_horizon: int,
    action_dim: int,
    rotation_space: str,
    gripper_mode: str,
    max_steps: int,
    warmup_steps: int,
    init_states: list[np.ndarray | None],
) -> list[dict[str, Any]]:
    """Run all `num_trials` of one task across `num_envs` parallel LIBERO envs
    (SubprocVectorEnv), in waves. Each control step gathers obs from the ACTIVE
    (not-done) envs, issues ONE batched /predict_batch, and steps all active envs;
    done envs are masked out. Returns per-trial result dicts in trial order with the
    same shape as the serial path's episode_results."""
    import multiprocessing as _mp

    from libero.libero.envs.venv import SubprocVectorEnv

    # LIBERO's SubprocVectorEnv defaults to the fork start method; forked children
    # inherit the parent's already-dlopen'd EGL/GL state, which corrupts per-child
    # render-context creation (EGLError / 'EGLGLContext' has no attribute '_context').
    # Force spawn so each env worker starts clean — exactly like the (working) serial
    # single-process path. spawn pickles env_fns, so the factory below is picklable.
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:  # pragma: no cover - already set
        pass

    resolved_rotation_space = _infer_rotation_space(action_dim, rotation_space)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    results: list[dict[str, Any]] = [None] * num_trials  # type: ignore[list-item]
    for t in range(num_trials):
        if init_states[t] is None:
            results[t] = {
                "episode": t,
                "success": False,
                "steps": 0,
                "error": "Skipped due to failed expert demo",
                "elapsed_s": 0.0,
            }
    runnable = [t for t in range(num_trials) if init_states[t] is not None]
    if not runnable:
        return results

    n = min(num_envs, len(runnable))

    mujoco_gl = os.environ.get("MUJOCO_GL", "egl")
    env_fn = _LiberoEnvFactory(
        bddl_file_name=bddl,
        camera_heights=env_image_size,
        camera_widths=env_image_size,
        render_gpu_device_id=render_gpu_device_id,
        mujoco_gl=mujoco_gl,
    )
    venv = SubprocVectorEnv([env_fn for _ in range(n)])
    try:
        venv.seed(seed)
        for w0 in range(0, len(runnable), n):
            wave = runnable[w0 : w0 + n]          # trial indices for this wave
            slots = list(range(len(wave)))        # env slots in use
            t_wave0 = time.perf_counter()
            venv.reset(id=slots)
            states = np.stack([np.asarray(init_states[t], dtype=np.float64) for t in wave])
            obs_arr = venv.set_init_state(states, id=slots)
            obs_by_slot = {s: obs_arr[i] for i, s in enumerate(slots)}
            done = {s: False for s in slots}
            succ = {s: False for s in slots}
            err: dict[int, str | None] = {s: None for s in slots}
            nsteps = {s: max_steps for s in slots}
            step = 0

            for _ in range(warmup_steps):
                act = np.stack([_get_libero_dummy_action() for _ in slots])
                obs_arr, _, _, _ = venv.step(act, id=slots)
                for i, s in enumerate(slots):
                    obs_by_slot[s] = obs_arr[i]
                step += 1

            while step < max_steps:
                active = [s for s in slots if not done[s]]
                if not active:
                    break
                obs_batch = [
                    _get_libero_images(obs_by_slot[s], cameras, flip_images=flip_images, rotate_180=rotate_180)
                    for s in active
                ]
                try:
                    chunks = client.predict_batch(obs_batch)
                except Exception as e:  # noqa: BLE001
                    for s in active:
                        done[s] = True
                        err[s] = f"server error: {e}"
                        nsteps[s] = step
                    break
                if not chunks or len(chunks) != len(active):
                    for s in active:
                        done[s] = True
                        err[s] = "bad batch response from server"
                        nsteps[s] = step
                    break
                chunk_by_slot = {s: chunks[k] for k, s in enumerate(active)}
                horizon = action_horizon if action_horizon > 0 else len(chunks[0])
                for h in range(horizon):
                    cur = [s for s in slots if not done[s]]
                    if not cur or step >= max_steps:
                        break
                    env_actions = []
                    for s in cur:
                        raw = _format_action(chunk_by_slot[s][h], action_dim)
                        a = _framewise_action_to_delta(np.asarray(raw, dtype=np.float32), resolved_rotation_space)
                        env_actions.append(_remap_gripper(a.tolist(), gripper_mode))
                    obs_arr, _, d, info = venv.step(np.stack(env_actions), id=cur)
                    step += 1
                    for i, s in enumerate(cur):
                        obs_by_slot[s] = obs_arr[i]
                        di = bool(d[i])
                        ii = info[i] if isinstance(info, (list, np.ndarray)) else info
                        is_succ = bool(ii.get("success")) if isinstance(ii, dict) else False
                        if is_succ:
                            done[s], succ[s], nsteps[s] = True, True, step
                        elif di:
                            # mirror serial: done w/o explicit success defaults to success
                            done[s] = True
                            succ[s] = ii.get("success", True) if isinstance(ii, dict) else True
                            nsteps[s] = step
            per_ep_elapsed = round((time.perf_counter() - t_wave0) / max(1, len(wave)), 3)
            for s, t in zip(slots, wave):
                results[t] = {
                    "episode": t,
                    "success": bool(succ[s]),
                    "steps": int(nsteps[s]),
                    "error": err[s],
                    "elapsed_s": per_ep_elapsed,
                }
    finally:
        try:
            venv.close()
        except Exception:  # noqa: BLE001
            pass
    return results


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_gifs and not args.output_dir:
        raise ValueError("--save_gifs requires --output_dir to be set")
    if args.save_comparison and not args.output_dir:
        raise ValueError("--save_comparison requires --output_dir to be set")

    # Parse cameras from comma-separated string
    cameras = [c.strip() for c in args.camera.split(",") if c.strip()]
    if not cameras:
        raise ValueError("At least one camera must be specified")
    for cam in cameras:
        if cam not in ("agentview", "wrist"):
            raise ValueError(f"Unsupported camera={cam!r}. Use 'agentview' or 'wrist'.")

    mujoco_backend = _configure_mujoco_env(args.mujoco_gl)
    _import_libero()

    client = ActionEnvironmentClient(
        server_url=args.server_url,
        domain_name=args.domain_name,
        prompt="",
        image_size=args.image_size,
        timeout=args.timeout,
    )
    print(f"MuJoCo GL backend: {mujoco_backend}", flush=True)
    print("Waiting for model server...", flush=True)
    _wait_for_server(client, args.wait_timeout)
    print(f"Connected to model server: {client.get_info()}", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = int(task_suite.n_tasks)

    if args.task_ids:
        selected_task_ids = [int(t) for t in args.task_ids.split(",") if t.strip()]
    else:
        selected_task_ids = list(range(num_tasks))

    max_steps = args.max_steps if args.max_steps > 0 else TASK_MAX_STEPS[args.task_suite]

    total_episodes = 0
    total_successes = 0
    task_results: list[dict[str, Any]] = []

    output_dir = Path(args.output_dir) if args.output_dir else None
    gif_root = output_dir / "gifs" if output_dir and args.save_gifs else None
    comparison_root = output_dir / "comparisons" if output_dir and args.save_comparison else None

    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)

        # ---- Vectorized path: N parallel envs + one batched /predict_batch per step ----
        if args.num_envs > 1:
            task_description = str(task.language)
            client.prompt = _augment_task_prompt_with_viewpoint(task_description, cameras)
            init_states = [
                _load_initial_states(
                    task_suite,
                    task_id,
                    task_description=task_description,
                    initial_states_path=args.initial_states_path,
                    episode_idx=e,
                )
                for e in range(args.num_trials_per_task)
            ]
            episode_results = _run_task_vectorized(
                task,
                task_description,
                num_trials=args.num_trials_per_task,
                num_envs=args.num_envs,
                env_image_size=args.env_image_size,
                seed=args.seed,
                render_gpu_device_id=args.render_gpu_device_id,
                client=client,
                cameras=cameras,
                flip_images=args.flip_images,
                rotate_180=args.rotate_180,
                action_horizon=args.action_horizon,
                action_dim=args.action_dim,
                rotation_space=args.rotation_space,
                gripper_mode=args.gripper_mode,
                max_steps=max_steps,
                warmup_steps=args.warmup_steps,
                init_states=init_states,
            )
            task_episodes = 0
            task_successes = 0
            for er in episode_results:
                task_episodes += 1
                total_episodes += 1
                if er["success"]:
                    task_successes += 1
                    total_successes += 1
                print(
                    f"Task {task_id} | Episode {er['episode'] + 1}/{args.num_trials_per_task} | "
                    f"success={er['success']} steps={er['steps']} elapsed_s={er['elapsed_s']:.1f} | "
                    f"task SR {task_successes}/{task_episodes} ({100.0 * task_successes / max(1, task_episodes):.1f}%) | "
                    f"overall SR {total_successes}/{total_episodes} "
                    f"({100.0 * total_successes / max(1, total_episodes):.1f}%)",
                    flush=True,
                )
            task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
            task_results.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": task_success_rate,
                    "episode_results": episode_results,
                }
            )
            print(
                f"Task {task_id} summary: {task_successes}/{task_episodes} ({task_success_rate * 100:.1f}%)",
                flush=True,
            )
            continue

        env, task_description = _get_libero_env(
            task,
            resolution=args.env_image_size,
            seed=args.seed,
            render_gpu_device_id=args.render_gpu_device_id,
        )

        task_episodes = 0
        task_successes = 0
        episode_results: list[dict[str, Any]] = []

        for episode_idx in range(args.num_trials_per_task):
            episode_t0 = time.perf_counter()
            client.prompt = _augment_task_prompt_with_viewpoint(task_description, cameras)
            initial_state = _load_initial_states(
                task_suite,
                task_id,
                task_description=task_description,
                initial_states_path=args.initial_states_path,
                episode_idx=episode_idx,
            )
            if initial_state is None:
                episode_elapsed_s = time.perf_counter() - episode_t0
                episode_results.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": 0,
                        "error": "Skipped due to failed expert demo",
                        "elapsed_s": round(episode_elapsed_s, 3),
                    }
                )
                print(
                    f"Task {task_id} | Episode {episode_idx + 1}/{args.num_trials_per_task} | "
                    "success=False steps=0 "
                    f"elapsed_s={episode_elapsed_s:.1f} "
                    "error='Skipped due to failed expert demo'",
                    flush=True,
                )
                continue

            gif_path = (
                gif_root / f"task_{task_id:03d}" / f"episode_{episode_idx:03d}.gif" if gif_root is not None else None
            )
            comparison_path = (
                comparison_root / f"task_{task_id:03d}" / f"episode_{episode_idx:03d}.gif"
                if comparison_root is not None
                else None
            )
            try:
                result = _run_episode(
                    env,
                    client,
                    cameras=cameras,
                    flip_images=args.flip_images,
                    rotate_180=args.rotate_180,
                    action_horizon=args.action_horizon,
                    action_dim=args.action_dim,
                    action_space=args.action_space,
                    rotation_space=args.rotation_space,
                    gripper_mode=args.gripper_mode,
                    max_steps=max_steps,
                    warmup_steps=args.warmup_steps,
                    initial_state=initial_state,
                    gif_path=gif_path,
                    gif_fps=args.gif_fps,
                    comparison_path=comparison_path,
                )
            except Exception as exc:
                result = EpisodeResult(False, 0, str(exc), [])
            episode_elapsed_s = time.perf_counter() - episode_t0

            task_episodes += 1
            total_episodes += 1
            if result.success:
                task_successes += 1
                total_successes += 1

            episode_results.append(
                {
                    "episode": episode_idx,
                    "success": result.success,
                    "steps": result.steps,
                    "error": result.error,
                    "elapsed_s": round(episode_elapsed_s, 3),
                }
            )

            # Save per-episode action log as JSON
            if output_dir is not None and result.actions:
                action_log_dir = output_dir / "actions" / f"task_{task_id:03d}"
                action_log_dir.mkdir(parents=True, exist_ok=True)
                action_log_path = action_log_dir / f"episode_{episode_idx:03d}.json"
                action_log_path.write_text(
                    json.dumps(result.actions, indent=2),
                    encoding="utf-8",
                )

            client.notify_next_episode()

            print(
                f"Task {task_id} | Episode {episode_idx + 1}/{args.num_trials_per_task} | "
                f"success={result.success} steps={result.steps} elapsed_s={episode_elapsed_s:.1f} | "
                f"task SR {task_successes}/{task_episodes} ({100.0 * task_successes / max(1, task_episodes):.1f}%) | "
                f"overall SR {total_successes}/{total_episodes} ({100.0 * total_successes / max(1, total_episodes):.1f}%)",
                flush=True,
            )

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
                "episode_results": episode_results,
            }
        )
        print(
            f"Task {task_id} summary: {task_successes}/{task_episodes} ({task_success_rate * 100:.1f}%)",
            flush=True,
        )
        # Close the env (and its EGL/MuJoCo render context) before the next task.
        # Leaving it open leaks one EGL context per task and hangs after ~8 tasks.
        try:
            env.close()
        except Exception:
            pass

    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    summary = {
        "task_suite": args.task_suite,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "num_trials_per_task": args.num_trials_per_task,
        "selected_task_ids": selected_task_ids,
        "action_space": args.action_space,
        "rotation_space": _infer_rotation_space(args.action_dim, args.rotation_space),
        "action_dim": args.action_dim,
        "task_results": task_results,
    }

    print(
        f"Overall success rate: {total_successes}/{total_episodes} ({overall_success_rate * 100:.1f}%)",
        flush=True,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
