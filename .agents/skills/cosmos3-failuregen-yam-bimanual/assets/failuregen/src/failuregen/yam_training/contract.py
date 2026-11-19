# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only contract shared by YAM preparation, training and failure generation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

DOMAIN = "yam_reel_joint_delta"
DOMAIN_ID = 32
SOURCE_FPS = 30
FPS = 5
CHUNK = 16
STRIDE = SOURCE_FPS // FPS
CANVAS = 960
CAPTION = (
    "A bimanual YAM robot transfers cable reels from a box into a tote. "
    "The overhead camera occupies the top half. The left and right wrist cameras occupy "
    "the bottom-left and bottom-right quarters. Black padding preserves camera geometry."
)
CAMERAS = ("observation.images.overhead", "observation.images.wrist_left", "observation.images.wrist_right")
CONTRACT = {
    "version": 2,
    "domain": DOMAIN,
    "domain_id": DOMAIN_ID,
    "raw_action_dim": 14,
    "source": "observation.state",
    "source_fps": SOURCE_FPS,
    "fps": FPS,
    "chunk_length": CHUNK,
    "source_stride": STRIDE,
    "joints": "next measured state minus current measured state, radians",
    "grippers": "absolute current measured state at columns 6 and 13; preserve recorded per-arm polarity",
    "normalization": "2 * (action - q01) / max(q99 - q01, 1e-8) - 1; no clipping",
    "camera_order": list(CAMERAS),
    "canvas_wh": [CANVAS, CANVAS],
    "layout": "overhead top half; left/right wrists bottom quarters; aspect-preserving black padding",
    "timing": "source frame index / 30; measured states at source indices 0,6,12,...",
}

EEF_CONTRACT = {
    "version": 1,
    "domain": "xdof_yam",
    "domain_id": 16,
    "raw_action_dim": 20,
    "source": ".labeling/cosmos_yam_report_5fps/actions_raw",
    "source_fps": SOURCE_FPS,
    "fps": FPS,
    "chunk_length": CHUNK,
    "source_stride": STRIDE,
    "pose": "inverse(T_previous) @ T_current; per-arm tcp_site at gripper mount",
    "rotation": "first two matrix columns concatenated: R00,R10,R20,R01,R11,R21",
    "arm_order": ["left", "right"],
    "position_units": "meters",
    "grippers": "absolute recorded destination state at columns 9 and 19; preserve per-arm polarity",
    "normalization": CONTRACT["normalization"],
    "normalizer_origin": "fit on this fine-tune's TRAIN transitions; not recovered pretrained statistics",
    "pretrained_preprocessing_verified": False,
    "camera_order": list(CAMERAS),
    "canvas_wh": [CANVAS, CANVAS],
    "layout": CONTRACT["layout"],
    "timing": CONTRACT["timing"],
}


def training_contract(action_space: str) -> dict:
    if action_space not in ("joint", "eef"):
        raise ValueError(f"Unknown YAM action space: {action_space}")
    return EEF_CONTRACT if action_space == "eef" else CONTRACT


def action_space_for(contract: dict) -> str:
    for name in ("joint", "eef"):
        if contract == training_contract(name):
            return name
    raise ValueError("Unrecognized YAM action contract")


def episode_actions(arrays, contract: dict) -> np.ndarray:
    """Return intervals exactly once; EEF deltas are already in the export."""
    if action_space_for(contract) == "joint":
        return raw_actions(arrays["states"])
    actions = np.asarray(arrays["actions_raw"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 20 or not np.isfinite(actions).all():
        raise ValueError("Expected finite (T, 20) relative EEF actions")
    if len(actions) + 1 != len(arrays["source_indices"]):
        raise ValueError("EEF actions must have one fewer row than observations")
    return actions


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_actions(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 14 or len(states) < 2 or not np.isfinite(states).all():
        raise ValueError("Expected finite measured states with shape (T+1, 14)")
    actions = np.diff(states, axis=0)
    actions[:, [6, 13]] = states[:-1, [6, 13]]
    return actions


def normalize(actions: np.ndarray, stats: dict) -> np.ndarray:
    width = stats.get("contract", CONTRACT)["raw_action_dim"]
    lo, hi = (np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99"))
    if lo.shape != (width,) or hi.shape != (width,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError(f"Expected finite {width}D q01/q99 with positive range")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != width or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (T, {width}) actions matching the normalizer")
    return (2 * (actions - lo) / np.maximum(hi - lo, 1e-8) - 1).astype(np.float32)


def load_stats(path: Path) -> dict:
    stats = json.loads(path.read_text())
    action_space_for(stats.get("contract"))
    normalize(np.zeros((1, stats["contract"]["raw_action_dim"])), stats)
    return stats


def compose_views(front: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Square 720-tier canvas, with 50/25/25 slots and no camera distortion."""
    images = [np.asarray(im) for im in (front, left, right)]
    if any(im.shape[-3:] != (480, 640, 3) or im.dtype != np.uint8 for im in images):
        raise ValueError("YAM camera frames must be uint8 RGB 640x480")
    if any(im.shape != images[0].shape for im in images):
        raise ValueError("Camera frame counts must match")

    def wrist(im):
        # Bilinear half-pixel coordinates, 640x480 -> 480x360.
        x = (np.arange(480) + 0.5) * 4 / 3 - 0.5
        y = (np.arange(360) + 0.5) * 4 / 3 - 0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        wx, wy = (x - x0)[None, :, None], (y - y0)[:, None, None]
        top = (1 - wx) * im[..., y0[:, None], x0[None, :], :] + wx * im[..., y0[:, None], x0[None, :] + 1, :]
        bottom = (1 - wx) * im[..., y0[:, None] + 1, x0[None, :], :] + wx * im[..., y0[:, None] + 1, x0[None, :] + 1, :]
        return np.rint((1 - wy) * top + wy * bottom).clip(0, 255).astype(np.uint8)

    canvas = np.zeros((*images[0].shape[:-3], CANVAS, CANVAS, 3), dtype=np.uint8)
    canvas[..., :480, 160:800, :] = front
    canvas[..., 540:900, :480, :] = wrist(left)
    canvas[..., 540:900, 480:, :] = wrist(right)
    return canvas


def valid_starts(frames: int, excluded: list[dict], stride: int = 1) -> list[int]:
    """Reject every clip whose closed observation interval touches unknown footage."""
    return [
        start
        for start in range(0, frames - CHUNK, stride)
        if not any(start / FPS < segment["end"] and (start + CHUNK) / FPS >= segment["start"] for segment in excluded)
    ]


def load_prepared(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "prepared":
        raise ValueError("Prepared YAM data is incomplete or uses a different contract; re-run preparation")
    action_space_for(manifest.get("contract"))
    stats_path = root / "action_stats.json"
    if file_sha256(stats_path) != manifest["stats_sha256"]:
        raise ValueError("YAM training normalizer changed after preparation")
    stats = load_stats(stats_path)
    if stats["contract"] != manifest["contract"]:
        raise ValueError("Prepared data and normalizer use different action contracts")
    return manifest, stats


def decode_frames(path: Path, first: int, count: int, stride: int = STRIDE, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode a short CFR window without loading an episode or GPU dependencies."""
    if first < 0 or count < 1 or stride < 1:
        raise ValueError("Invalid frame window")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-threads",
            "1",
            "-ss",
            f"{first / SOURCE_FPS:.12f}",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"select=not(mod(n\\,{stride}))",
            "-frames:v",
            str(count),
            "-fps_mode",
            "passthrough",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    expected = count * 480 * 640 * 3
    if len(result.stdout) != expected:
        raise ValueError(f"{path}: expected {count} RGB 640x480 frames; got {len(result.stdout)} bytes")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, 480, 640, 3)
