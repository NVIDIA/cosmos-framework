# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Output checks and browser-sized previews for native Sim-Auto inference."""

from __future__ import annotations

import json
from pathlib import Path

from helpers.auto_helpers import _ffmpeg, validate_framework_outputs


def validate_sim_auto_outputs(output_dir: Path, request: dict) -> dict:
    """Check camera videos, optional LiDAR, and that the rollout covered the requested horizon."""
    result = validate_framework_outputs(Path(output_dir), request)
    if "lidar" not in request and any(Path(output_dir).rglob("lidar.safetensors")):
        raise ValueError("A camera-only request produced LiDAR output; check the selected output directory.")
    sample_dir = next(iter(result["cameras"].values())).parent
    frames = request["num_frames"]
    rolling = json.loads((sample_dir / "rolling_prompt.json").read_text())
    if rolling.get("num_frames") != frames or rolling.get("source_frame_range") != [0, frames - 1]:
        raise ValueError(
            f"Expected a rolling schedule over frames [0, {frames - 1}], found "
            f"{rolling.get('num_frames')!r} frames over {rolling.get('source_frame_range')!r}."
        )
    chunks = json.loads((sample_dir / "caption_chunks.json").read_text())
    cameras = len(request["multiview"]["views"])
    if not chunks or chunks[0].get("source_frame_range", [None])[0] != 0:
        raise ValueError("The first consumed caption span must start at frame 0.")
    if not all(isinstance(chunk.get("prompt"), list) and len(chunk["prompt"]) == cameras for chunk in chunks):
        raise ValueError("Expected one consumed caption per camera in caption_chunks.json.")
    result.update(sample_dir=sample_dir, source_frame_range=rolling["source_frame_range"], caption_spans=len(chunks))
    return result


def camera_previews(cameras: dict, preview_dir: Path, *, width: int = 480) -> dict:
    """Write small H.264 copies of the generated videos for inline display; outputs stay unchanged."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews = {}
    for camera_key, path in cameras.items():
        target = preview_dir / f"{camera_key}.mp4"
        _ffmpeg(
            "-y", "-i", str(path), "-vf", f"scale={width}:-2", "-an",
            "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target),
        )
        previews[camera_key] = target
    return previews
