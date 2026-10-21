# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only helpers for the Cosmos Framework and vLLM-Omni Auto notebooks."""

from __future__ import annotations

import base64
import copy
import html
import json
import math
import mimetypes
import shutil
import subprocess
import time
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

# Shared input preparation and media previews.

CAMERA_ORDER = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)


def require_asset(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Unhydrated Git LFS pointer: {path}. Download the actual file contents.")
    return path


def _ffmpeg(*args: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg and ffprobe and add both to PATH.")
    result = subprocess.run(["ffmpeg", "-v", "error", *args], capture_output=True, text=True)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def probe_video(path: Path) -> dict:
    require_asset(path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": Fraction(stream["avg_frame_rate"]),
        "frames": int(stream["nb_read_frames"]),
        "duration": Fraction(stream["duration"]),
    }


def prepare_controls(views: list[dict], base_dir: Path, output_dir: Path, frames: int, fps: float) -> list[Path]:
    """Decode synchronized controls and trim their first window without resizing."""
    if frames < 1 or (frames - 1) % 4 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Use a positive 4k+1 camera frame count and positive finite FPS.")
    sources = [require_asset(base_dir / view["control_path"]) for view in views]
    metadata = [probe_video(path) for path in sources]
    signatures = {(item["fps"], item["frames"], item["duration"]) for item in metadata}
    if len(signatures) != 1:
        raise ValueError("WSM controls must have synchronized FPS, frame counts, and durations.")
    rate, count, duration = next(iter(signatures))
    if rate != fps or count < frames or abs(float(duration - count / rate)) > 1e-5:
        raise ValueError(f"WSM timeline {rate=}, {count=}, {duration=} cannot supply {frames} frames at {fps} FPS.")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for view, source, info in zip(views, sources, metadata, strict=True):
        if info["codec"] != "h264" or info["pixel_format"] != "yuv420p":
            raise ValueError(f"Expected H.264/yuv420p control: {source}")
        _ffmpeg("-i", str(source), "-f", "null", "-")
        target = source
        if count != frames:
            target = output_dir / f"{view['camera_key']}.mp4"
            _ffmpeg(
                "-y",
                "-i",
                str(source),
                "-vf",
                f"trim=start_frame=0:end_frame={frames},setpts=N/({fps}*TB)",
                # setpts can clear frame-rate metadata; explicitly preserve the requested cadence.
                "-r",
                str(fps),
                "-fps_mode",
                "cfr",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                str(frames),
                "-movflags",
                "+faststart",
                str(target),
            )
        actual = probe_video(target)
        if any(
            actual[key] != expected
            for key, expected in {
                "frames": frames,
                "fps": fps,
                "width": info["width"],
                "height": info["height"],
            }.items()
        ):
            raise ValueError(f"Unexpected prepared control metadata: {target}: {actual}")
        prepared.append(target)
    return prepared


def validate_rgb_conditions(views: list[dict], base_dir: Path) -> None:
    """Only the first decoded RGB frame is used by condition_video_as_image."""
    for view in views:
        path = require_asset(base_dir / view["vision_path"])
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            check=True,
            capture_output=True,
        )
        if not result.stdout or result.stderr:
            raise ValueError(f"Cannot decode RGB first frame: {path}: {result.stderr.decode(errors='replace')}")


def validate_lidar(path: Path, sweeps: int, *, exact: bool = False) -> dict:
    """Read numeric sweeps incrementally; never load a GPU library or normalize inputs."""
    import numpy as np
    from safetensors import safe_open

    require_asset(path)
    with safe_open(path, framework="numpy") as handle:
        if list(handle.keys()) != ["frames"]:
            raise ValueError("LiDAR must contain only the tensor 'frames'.")
        tensor = handle.get_slice("frames")
        shape = tensor.get_shape()
        if tensor.get_dtype() != "F32" or len(shape) != 4 or shape[0] != 3 or shape[2:] != [128, 1800]:
            raise ValueError(f"Expected FP32 LiDAR [3,T,128,1800], got {shape} / {tensor.get_dtype()}.")
        if sweeps < 1 or shape[1] < sweeps or (exact and shape[1] != sweeps):
            raise ValueError(f"Expected {'exactly' if exact else 'at least'} {sweeps} sweeps, got {shape[1]}.")
        metadata = json.loads((handle.metadata() or {}).get("lidar", "{}"))
        for index in range(sweeps):
            frame = tensor[:, index]
            if not np.isfinite(frame).all() or (frame[0] < 0).any() or (frame[1:] < 0).any() or (frame[1:] > 1).any():
                raise ValueError(f"Invalid physical LiDAR values in sweep {index}.")
            if metadata.get("apply_validity_mask"):
                if not np.isin(frame[2], [0, 1]).all() or np.any(frame[:2, frame[2] == 0] != 0):
                    raise ValueError(f"Invalid masked LiDAR values in sweep {index}.")
    return {"shape": shape, "dtype": "float32", "metadata": metadata}


def split_cameras(
    path: Path, output_dir: Path, camera_order: tuple, frames: int, fps: float, *, width: int = 832, height: int = 480
) -> dict:
    info = probe_video(path)
    if any(
        info[key] != value
        for key, value in {
            "frames": len(camera_order) * frames,
            "fps": fps,
            "width": width,
            "height": height,
        }.items()
    ):
        raise ValueError(f"Unexpected camera-major response metadata: {info}")
    outputs = {}
    for index, camera_key in enumerate(camera_order):
        target = output_dir / f"vision_view{index:02d}_{camera_key}.mp4"
        _ffmpeg(
            "-y",
            "-i",
            str(path),
            "-vf",
            f"trim=start_frame={index * frames}:end_frame={(index + 1) * frames},setpts=N/({fps}*TB)",
            # Match the timestamp spacing even when FFmpeg cannot infer the filter's frame rate.
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        )
        actual = probe_video(target)
        if any(
            actual[key] != value
            for key, value in {"frames": frames, "fps": fps, "width": width, "height": height}.items()
        ):
            raise ValueError(f"Unexpected split output: {target}: {actual}")
        outputs[camera_key] = target
    return outputs


def camera_grid(outputs: dict) -> str:
    cards = []
    for camera_key, path in outputs.items():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        cards.append(
            "<figure style='margin:0'><video controls loop muted playsinline preload='metadata' "
            "style='width:100%;background:#000'>"
            f"<source src='data:video/mp4;base64,{encoded}' type='video/mp4'></video>"
            f"<figcaption style='font:12px monospace'>{html.escape(camera_key)}</figcaption></figure>"
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px'>"
        + "".join(cards)
        + "</div>"
    )


# Cosmos Framework output validation.

def validate_framework_outputs(output_dir: Path, manifest: dict) -> dict:
    """Check native camera videos and, when requested, physical LiDAR output."""
    records = list(output_dir.rglob("sample_outputs.json"))
    if len(records) != 1 or json.loads(records[0].read_text()).get("status") != "success":
        raise ValueError("Expected one successful native Framework sample record.")
    cameras = {}
    for index, view in enumerate(manifest["multiview"]["views"]):
        key = view["camera_key"]
        paths = list(output_dir.rglob(f"vision_view{index:02d}_{key}.mp4"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one generated video for {key}.")
        info = probe_video(paths[0])
        width, height = {"480": (832, 480), "720": (1280, 720)}[str(manifest["resolution"])]
        expected = {"width": width, "height": height, "frames": manifest["num_frames"], "fps": manifest["fps"]}
        if any(info[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected output video metadata: {paths[0]}: {info}")
        cameras[key] = paths[0]
    result = {"cameras": cameras}
    if "lidar" in manifest:
        import numpy as np
        from safetensors import safe_open

        paths = list(output_dir.rglob("lidar.safetensors"))
        if len(paths) != 1:
            raise ValueError("Expected one generated lidar.safetensors file.")
        sweeps = round(manifest["num_frames"] * 10.0 / manifest["fps"])
        with safe_open(paths[0], framework="numpy") as handle:
            if set(handle.keys()) != {"range_m", "intensity", "frame_ids"}:
                raise ValueError("Expected native LiDAR range_m, intensity, and frame_ids tensors.")
            ranges, intensity = handle.get_slice("range_m"), handle.get_slice("intensity")
            if ranges.get_shape() != [sweeps, 128, 1800] or intensity.get_shape() != ranges.get_shape():
                raise ValueError("Unexpected generated LiDAR dimensions or sweep count.")
            frame_ids = handle.get_tensor("frame_ids")
            expected_ids = np.arange(sweeps, dtype=np.int64) * round(manifest["fps"] / 10.0)
            if not np.array_equal(frame_ids, expected_ids):
                raise ValueError("LiDAR timestamps do not align with the camera timeline.")
            for index in range(sweeps):
                r, i = ranges[index], intensity[index]
                if not np.isfinite(r).all() or not np.isfinite(i).all() or (r < 0).any() or (i < 0).any() or (i > 1).any():
                    raise ValueError(f"Invalid physical LiDAR output at sweep {index}.")
        result.update(lidar=paths[0], lidar_sweeps=sweeps)
    return result


# vLLM-Omni request preparation and HTTP client.

def prepare_request(manifest: dict, base_dir: Path, model: str) -> tuple[dict, list[Path]]:
    """Translate local manifest paths to the existing multipart Videos API contract."""
    for field in ("negative_prompt", "negative_metadata_mode"):
        if manifest.get(field) is not None:
            raise ValueError(
                f"Cosmos3-Nano-Transfer-Auto does not support {field}; "
                "use per_view_negative_prompt to apply one negative caption to every camera."
            )
    # The server maps camera_key to its trained camera ID, so a subset is valid; keep canonical order.
    camera_keys = [view["camera_key"] for view in manifest["multiview"]["views"]]
    if not camera_keys or camera_keys != [key for key in CAMERA_ORDER if key in camera_keys]:
        raise ValueError("List distinct cameras from the canonical camera order, in that order.")
    extra = {
        key: copy.deepcopy(manifest[key])
        for key in (
            "wsm",
            "multiview",
            "lidar",
            "resolution",
            "aspect_ratio",
            "control_guidance",
            "per_view_negative_prompt",
        )
        if key in manifest
    }
    paths = []
    for index, view in enumerate(extra["multiview"]["views"]):
        if not isinstance(view.get("prompt"), str) or not view["prompt"].strip():
            raise ValueError(f"Every camera requires a nonempty plain-text prompt; missing views[{index}].prompt.")
        for role in ("control", "vision"):
            if f"{role}_reference_index" in view:
                raise ValueError("Supply local paths, not upload indexes, in the input manifest.")
            if f"{role}_path" in view:
                path = require_asset(base_dir / view.pop(f"{role}_path"))
                view[f"{role}_reference_index"] = len(paths)
                paths.append(path)
    if "lidar" in extra:
        if extra["lidar"].get("return_output") is not True:
            raise ValueError("The joint walkthrough requires lidar.return_output=true.")
        if any(
            "vision_reference_index" not in view or "control_reference_index" not in view
            for view in extra["multiview"]["views"]
        ):
            raise ValueError("The joint I2V example requires WSM and RGB for every camera.")
        path = require_asset(base_dir / extra["lidar"].pop("control_path"))
        extra["lidar"]["control_reference_index"] = len(paths)
        paths.append(path)
    if len(paths) > 23 or any(path.stat().st_size > 512 * 1024**2 for path in paths):
        raise ValueError("Videos API accepts at most 23 files, each no larger than 512 MiB.")
    form = {
        key: str(manifest[key])
        for key in (
            "prompt",
            "fps",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "flow_shift",
            "seed",
        )
        if key in manifest
    }
    form.update(model=model, size="832x480", extra_params=json.dumps(extra))
    return form, paths


def _post(client, endpoint: str, form: dict, paths: list[Path]):
    with ExitStack() as stack:
        files = [
            (
                "input_references",
                (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                ),
            )
            for path in paths
        ]
        return client.post(endpoint, data=form, files=files)


def _check_response(response, output_dir: Path) -> None:
    if response.is_error:
        (output_dir / "response_error.txt").write_text(response.text)
    response.raise_for_status()


def _download(client, url: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with client.stream("GET", url) as response:
            if response.is_error:
                response.read()
            _check_response(response, path.parent)
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty artifact: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def submit_joint(client, form: dict, paths: list[Path], output_dir: Path, *, timeout: float = 3600) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Polling timeout must be positive and finite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    response = _post(client, "/v1/videos", form, paths)
    _check_response(response, output_dir)
    job = response.json()
    job_url = f"/v1/videos/{job['id']}"
    deadline = time.monotonic() + timeout
    print(f"Submitted {job['id']} at {client.base_url.join(job_url)}")
    while True:
        (output_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        if job.get("status") == "failed":
            (output_dir / "response_error.txt").write_text(json.dumps(job, indent=2))
            raise RuntimeError(f"Video generation failed: {job.get('error')}")
        _check_response(response, output_dir)
        if job.get("status") == "completed":
            break
        if job.get("status") not in {"queued", "in_progress"}:
            raise RuntimeError(f"Unexpected job status: {job}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Polling timed out; job remains at {client.base_url.join(job_url)}. See job.json.")
        time.sleep(min(1, remaining))
        response = client.get(job_url, timeout=max(0.001, min(30, deadline - time.monotonic())))
        try:
            job = response.json()
        except ValueError:
            _check_response(response, output_dir)
            raise
    if not job.get("lidar"):
        raise ValueError("Completed job is missing its requested LiDAR artifact; see job.json.")
    video_path = output_dir / "multiview_camera_major.mp4"
    _download(client, job_url + "/content", video_path)
    return {"video": video_path, "descriptor": job["lidar"], "job_id": job["id"]}


def download_lidar(client, job: dict, output_path: Path) -> tuple[Path, dict]:
    """Stream a completed job's numeric LiDAR and save its descriptor beside it as JSON."""
    if job.get("status") != "completed" or not job.get("lidar"):
        raise RuntimeError("The job must be completed and describe a LiDAR artifact; see job.json.")
    output_path = Path(output_path)
    _download(client, f"/v1/videos/{job['id']}/lidar", output_path)
    output_path.with_suffix(".json").write_text(json.dumps(job["lidar"], indent=2) + "\n")
    return output_path, job["lidar"]


def validate_lidar_output(path: Path, descriptor: dict, sweeps: int, *, lidar_fps: float = 10.0) -> dict:
    """Require exactly ``sweeps`` valid sweeps from time zero, with metadata matching the job descriptor."""
    info = validate_lidar(path, sweeps, exact=True)
    expected = {
        "shape": info["shape"], "dtype": info["dtype"], "fps": lidar_fps,
        "num_frames": sweeps, "start_time_seconds": 0,
    }
    for field, value in expected.items():
        if descriptor.get(field) != value:
            raise ValueError(f"LiDAR descriptor {field}={descriptor.get(field)!r}, expected {value!r}.")
    for field in ("shape", "dtype", "fps", "num_frames", "channels", "units",
                  "apply_validity_mask", "validity_threshold", "range_projection"):
        if info["metadata"].get(field) != descriptor.get(field):
            raise ValueError(f"Embedded LiDAR metadata {field!r} differs from the job descriptor.")
    return info
