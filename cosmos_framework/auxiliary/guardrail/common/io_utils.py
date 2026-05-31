# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import glob
from dataclasses import dataclass

import imageio
import numpy as np

from cosmos_framework.utils import log


@dataclass
class VideoData:
    frames: np.ndarray  # Shape: [B, H, W, C]
    fps: int
    duration: int  # in seconds


def get_video_filepaths(input_dir: str) -> list[str]:
    """Get a list of filepaths for all videos in the input directory."""
    paths = glob.glob(f"{input_dir}/**/*.mp4", recursive=True)
    paths += glob.glob(f"{input_dir}/**/*.avi", recursive=True)
    paths += glob.glob(f"{input_dir}/**/*.mov", recursive=True)
    paths = sorted(paths)
    log.debug(f"Found {len(paths)} videos")
    return paths


def read_video(filepath: str) -> VideoData:
    """Read a video file and extract its frames and metadata."""
    try:
        reader = imageio.get_reader(filepath, "ffmpeg")
    except Exception as e:
        raise ValueError(f"Failed to read video file: {filepath}") from e

    # Extract metadata from the video file
    try:
        metadata = reader.get_meta_data()
        fps = metadata.get("fps")
        duration = metadata.get("duration")
    except Exception as e:
        reader.close()
        raise ValueError(f"Failed to extract metadata from video file: {filepath}") from e

    # Extract frames from the video file
    try:
        frames = np.array([frame for frame in reader])
    except Exception as e:
        raise ValueError(f"Failed to extract frames from video file: {filepath}") from e
    finally:
        reader.close()

    return VideoData(frames=frames, fps=fps, duration=duration)


def save_video(filepath: str, frames: np.ndarray, fps: int) -> None:
    """Save a video file from a sequence of frames."""
    try:
        writer = imageio.get_writer(filepath, fps=fps, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
    except Exception as e:
        raise ValueError(f"Failed to save video file to {filepath}") from e
    finally:
        writer.close()
