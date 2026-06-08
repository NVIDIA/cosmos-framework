# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared ffprobe-based video metadata helper for the captioning / dataset scripts.

Returns ``fps``, ``duration`` (seconds), ``width``, ``height`` and
``total_frames`` for a video file.  Used by both ``caption_from_video.py``
(to fill the structured caption's media fields) and ``captions_to_sft_jsonl.py``
(to build SFT JSONL rows), so the two stay consistent.
"""

import json
import subprocess
from pathlib import Path

_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def probe_video_metadata(video_path: str | Path) -> dict:
    """Return ``{fps, duration, width, height, total_frames}`` via ffprobe.

    Raises:
        RuntimeError: if ffprobe fails or the file has no video stream.
    """
    video_path = str(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
    fps_num, fps_den = (fps_str.split("/") + ["1"])[:2]
    fps_den_f = float(fps_den) or 1.0
    fps = float(fps_num) / fps_den_f

    # Duration: prefer the container format duration, fall back to the stream's.
    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # nb_frames may be absent; fall back to duration * fps.
    total_frames = int(video_stream.get("nb_frames") or round(duration * fps))

    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }
