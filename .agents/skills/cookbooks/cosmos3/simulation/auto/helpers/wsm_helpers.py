# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Inspect caller-supplied RDS-HQ scenes and export native Auto inference requests."""

from __future__ import annotations

import io
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np

from helpers.auto_helpers import CAMERA_ORDER, require_asset


def caption_window(document: dict, frames: int, fps: int) -> str:
    """Select the source caption covering the first window, clipping time labels."""
    chunks = document["caption_structured"].values()
    covering = [c for c in chunks if c["start_frame"] == 0 and c["end_frame"] >= frames]
    if len(covering) != 1:
        raise ValueError(f"Need one source caption covering the first {frames} frames.")
    text = covering[0]["caption"].strip()
    spans = list(re.finditer(r"\[([^\]\n]+?)s\s*-\s*([^\]\n]+?)s\]\s*:\s*", text))
    if spans:
        # Scene-wide appearance often precedes the first timestamp. Keep it.
        introduction = text[: spans[0].start()].strip()
        parts = [introduction] if introduction else []
        duration = frames / fps
        previous_end = 0.0
        selected = 0
        for index, span in enumerate(spans):
            try:
                start, end = map(float, span.groups())
            except ValueError as error:
                raise ValueError("Caption timestamps must be numeric seconds.") from error
            if not (math.isfinite(start) and math.isfinite(end) and previous_end <= start < end):
                raise ValueError("Caption timestamps must be finite, nonnegative, ordered and non-overlapping.")
            previous_end = end
            stop = spans[index + 1].start() if index + 1 < len(spans) else len(text)
            description = text[span.end() : stop].strip()
            if not description:
                raise ValueError("Each caption timestamp needs a nonempty description.")
            if start >= duration:
                continue
            parts.append(f"[{start:g}s-{min(end, duration):g}s]: {description}")
            selected += 1
        if not selected:
            raise ValueError("The selected window has no timestamped caption.")
        text = "\n".join(parts)
    if not text:
        raise ValueError("The selected window has no caption.")
    return text


def inspect_scene(
    input_root: Path,
    sample: str,
    cameras: list[str],
    frames: int,
    fps: int = 30,
    resolution_spec: str = "resize_1280x720",
) -> dict:
    """Check supplied calibration and continuous poses before rendering, without requiring captions."""
    if type(frames) is not int or frames < 1 or (frames - 1) % 4:
        raise ValueError("Choose a positive 4k+1 frame count, such as 297.")
    if fps != 30:
        raise ValueError("This example expects a synchronized 30 FPS source timeline.")
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= set(CAMERA_ORDER):
        raise ValueError(
            "This notebook needs unique supported Auto camera keys. For custom names or pinhole cameras, "
            "render with the standalone CLI and follow the inference input-preparation guide for camera mapping."
        )
    if resolution_spec not in {"source", "resize_1280x720"}:
        raise ValueError("Choose resolution_spec='source' or 'resize_1280x720', matching the render.")
    calibration = {}
    intrinsic_path = input_root / "ftheta_intrinsic" / f"{sample}.tar"
    if not intrinsic_path.is_file() and (input_root / "pinhole_intrinsic" / f"{sample}.tar").is_file():
        raise ValueError(
            "This notebook uses F-theta calibration. Render pinhole controls with camera_type=pinhole "
            "in the standalone CLI, then use the inference input-preparation guide to prepare the native request."
        )
    require_asset(intrinsic_path)
    with tarfile.open(intrinsic_path) as archive:
        for member in archive:
            if not member.isfile() or ".ftheta_intrinsic." not in member.name:
                continue
            camera = member.name.split(".ftheta_intrinsic.")[1].removesuffix(".npy")
            if camera not in cameras:
                continue
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"Missing calibration payload: {member.name}")
            array = np.load(io.BytesIO(payload.read()), allow_pickle=False)
            if array.shape not in {(11,), (14,)} or not np.isfinite(array).all():
                raise ValueError(f"Invalid calibration: {camera}")
            # Source mode rounds each intrinsic dimension up to an even number.
            calibration[camera] = (
                [int(np.ceil(value / 2)) * 2 for value in array[2:4]] if resolution_spec == "source" else [1280, 720]
            )
    if set(calibration) != set(cameras):
        raise ValueError("Supply calibration for every selected camera; no fallback calibration is used.")
    pose_frames = {camera: set() for camera in cameras}
    with tarfile.open(require_asset(input_root / "pose" / f"{sample}.tar")) as archive:
        for member in archive:
            match = re.fullmatch(re.escape(sample) + r"\.(\d{6})\.pose\.(camera_[^.]+)\.npy", member.name)
            if member.isfile() and match and match[2] in pose_frames:
                pose_frames[match[2]].add(int(match[1]))
    for camera, available in pose_frames.items():
        if not set(range(frames)).issubset(available):
            raise ValueError(f"Missing requested pose frames for {camera}.")
    require_asset(input_root / "all_object_info" / f"{sample}.tar")
    # Archive member order must not change the user's selected camera order.
    return {"sample": sample, "resolutions": {camera: calibration[camera] for camera in cameras}}


def export_wsm_request(output_root: Path, scene: dict, sampling: dict, caption_root: Path) -> Path:
    """Validate final-view captions and write a native T2V request with interval-specific copies."""
    output_root = Path(output_root).resolve()
    frames, fps = sampling["num_frames"], sampling["fps"]
    captions = {
        camera: caption_window(json.loads(require_asset(caption_root / f"{camera}.json").read_text()), frames, fps)
        for camera in scene["resolutions"]
    }
    caption_dir = output_root / "captions"
    if caption_dir.resolve() == caption_root.resolve():
        raise ValueError("Keep source captions in a separate directory, such as view_captions, to preserve them.")
    caption_dir.mkdir(exist_ok=True)
    views = []
    for camera, caption in captions.items():
        relative = Path("world_scenario") / f"ftheta_{camera}" / f"{scene['sample']}_0.mp4"
        require_asset(output_root / relative)
        caption_path = caption_dir / f"{camera}.json"
        caption_path.write_text(
            json.dumps(
                {
                    "caption_structured": {
                        f"chunk_0_{frames}": {
                            "caption": caption,
                            "start_frame": 0,
                            "end_frame": frames,
                        }
                    }
                },
                indent=2,
            )
            + "\n"
        )
        views.append(
            {
                "camera_key": camera,
                "control_path": relative.as_posix(),
                "caption_path": caption_path.relative_to(output_root).as_posix(),
            }
        )
    request = {
        **sampling,
        "name": scene["sample"],
        "model_mode": "text2video",
        "resolution": "720",
        "aspect_ratio": "16,9",
        "prompt": "",
        "wsm": {},
        "num_video_frames_per_chunk": frames,
        "num_conditional_frames": 0,
        "multiview": {
            "views": views,
            "load_caption_from_data": True,
            "condition_video_as_image": False,
            "use_first_chunk_only": True,
        },
    }
    target = output_root / "request.json"
    target.write_text(json.dumps(request, indent=2) + "\n")
    return target
