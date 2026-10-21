# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate one generated video against its resolved inference request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-fps", type=float, required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--fps-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def require_positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video(path: Path) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("output must contain exactly one video stream")
        stream = streams[0]
        if stream.average_rate is None:
            raise ValueError("output has no valid average frame rate")

        decoded_frames = 0
        has_spatial_variance = False
        has_temporal_variance = False
        previous_sample: bytes | None = None
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            decoded_frames += 1
            has_spatial_variance = has_spatial_variance or int(rgb.min()) != int(rgb.max())
            row_stride = max(rgb.shape[0] // 64, 1)
            column_stride = max(rgb.shape[1] // 64, 1)
            sample = rgb[::row_stride, ::column_stride].tobytes()
            if previous_sample is not None and sample != previous_sample:
                has_temporal_variance = True
            previous_sample = sample

        codec_context = stream.codec_context
        pixel_format = codec_context.format.name if codec_context.format is not None else None
        return {
            "codec": codec_context.name,
            "decoded_frames": decoded_frames,
            "fps": float(stream.average_rate),
            "height": codec_context.height,
            "pixel_format": pixel_format,
            "spatial_variance": has_spatial_variance,
            "temporal_variance": has_temporal_variance,
            "width": codec_context.width,
        }


def main() -> None:
    args = parse_args()
    require_positive(args.expected_frames, "expected frames")
    require_positive(args.expected_fps, "expected FPS")
    require_positive(args.expected_width, "expected width")
    require_positive(args.expected_height, "expected height")
    if args.fps_tolerance < 0:
        raise ValueError("FPS tolerance must be nonnegative")

    video = args.video.expanduser().resolve()
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(f"output video is missing or empty: {video}")

    result = decode_video(video)
    saved_dimensions = (result["width"], result["height"])
    expected_dimensions = (args.expected_width, args.expected_height)
    if saved_dimensions != expected_dimensions:
        raise ValueError(f"saved dimensions {saved_dimensions} do not match expected {expected_dimensions}")
    if abs(result["fps"] - args.expected_fps) > args.fps_tolerance:
        raise ValueError(f"saved FPS {result['fps']} does not match expected {args.expected_fps}")
    if result["decoded_frames"] != args.expected_frames:
        raise ValueError(f"decoded {result['decoded_frames']} frames; expected {args.expected_frames}")
    if not result["spatial_variance"]:
        raise ValueError("decoded video has no sampled spatial variance")
    if not result["temporal_variance"]:
        raise ValueError("decoded video has no sampled temporal variance")

    result["sha256"] = sha256_file(video)
    result["video"] = str(video)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
