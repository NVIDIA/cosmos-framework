# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only helpers for the Cosmos Framework, vLLM-Omni, and SGLang Auto notebooks."""

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


def prepare_controls(
    views: list[dict], base_dir: Path, output_dir: Path, frames: int, fps: float, *, materialize: bool = False
) -> list[Path]:
    """Decode synchronized controls and trim their first window without resizing.

    A control that already has ``frames`` frames is returned in place. With
    ``materialize`` it is copied into ``output_dir`` instead, so every returned path
    lies under ``output_dir``; a server that opens inputs by path sees only what is
    mounted there.
    """
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
        elif materialize:
            target = output_dir / f"{view['camera_key']}.mp4"
            if not (target.exists() and target.samefile(source)):
                shutil.copyfile(source, target)
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


def stage_conditions(views: list[dict], output_dir: Path, *, field: str = "vision_path") -> list[Path]:
    """Copy each camera's condition file into ``output_dir`` as ``<camera_key><suffix>``.

    SGLang opens inputs by path on the server, and a downloaded sample lives in the
    client's Hugging Face cache. Staging the files under the mounted output folder
    gives the server one absolute path that exists on both sides. The copy keeps the
    suffix of the path named in the request: a Hugging Face snapshot entry is a
    symlink into an extension-less blob store, and the server's video reader picks
    its decoder by extension.
    Every call rewrites the copies, so a changed condition is picked up even when
    its size did not change.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for view in views:
        named = Path(view[field])
        source = require_asset(named)
        target = output_dir / f"{view['camera_key']}{named.suffix}"
        shutil.copyfile(source, target)
        staged.append(target)
    return staged


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


def validate_sglang_lidar(path: Path, sweeps: int, *, fps: float = 10.0) -> dict:
    """Validate SGLang's generated range map incrementally on the CPU."""
    import numpy as np
    from safetensors import safe_open

    if sweeps < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Expected positive LiDAR sweep count and finite positive FPS.")
    require_asset(path)
    dtypes = {"range_m": "F32", "intensity": "F16", "validity": "U8"}
    shape = [sweeps, 128, 1800]
    with safe_open(path, framework="numpy") as handle:
        if set(handle.keys()) != set(dtypes):
            raise ValueError("SGLang LiDAR must contain range_m, intensity, and validity tensors.")
        tensors = {name: handle.get_slice(name) for name in dtypes}
        for name, tensor in tensors.items():
            if tensor.get_shape() != shape or tensor.get_dtype() != dtypes[name]:
                raise ValueError(f"Expected {name} with shape {shape} and dtype {dtypes[name]}.")
        metadata = handle.metadata() or {}
        try:
            actual_fps = float(metadata["fps"])
            min_range = float(metadata["min_range_m"])
            max_range = float(metadata["max_range_m"])
            invalid_range = float(metadata["invalid_range_m"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Missing or invalid SGLang LiDAR metadata.") from exc
        if (
            metadata.get("layout") != "[sweeps, beams, azimuth]"
            or actual_fps != fps
            or not math.isfinite(min_range)
            or not math.isfinite(max_range)
            or not 0 <= min_range < max_range
            or invalid_range != 0
        ):
            raise ValueError(f"Unexpected SGLang LiDAR metadata: {metadata}.")
        for index in range(sweeps):
            ranges = tensors["range_m"][index]
            intensity = tensors["intensity"][index]
            validity = tensors["validity"][index]
            if (
                not np.isfinite(ranges).all()
                or not np.isfinite(intensity).all()
                or (ranges < 0).any()
                or (intensity < 0).any()
                or (intensity > 1).any()
                or not np.isin(validity, [0, 1]).all()
            ):
                raise ValueError(f"Invalid SGLang LiDAR values in sweep {index}.")
            valid = validity == 1
            if (ranges[valid] <= 0).any() or (ranges[~valid] != 0).any() or (intensity[~valid] != 0).any():
                raise ValueError(f"Invalid masked SGLang LiDAR values in sweep {index}.")
    return {"shape": shape, "dtypes": dtypes, "metadata": metadata}


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


# vLLM-Omni and SGLang request preparation and HTTP client.

BACKENDS = ("vllm_omni", "sglang")


def _check_backend(backend: str) -> str:
    """Return ``backend`` when it names a supported Videos API server."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; use {' or '.join(map(repr, BACKENDS))}.")
    return backend


def check_model(client, expected: str | None = None) -> str:
    """Return the served model ID from ``GET /v1/models``, requiring ``expected`` when given."""
    response = client.get("/v1/models")
    response.raise_for_status()
    ids = [item["id"] for item in response.json().get("data", []) if item.get("id")]
    if expected:
        if expected not in ids:
            raise ValueError(f"Expected served model {expected!r}; found {ids}.")
        return expected
    if len(ids) != 1:
        raise ValueError(f"Set the expected model explicitly; server returned {ids}.")
    return ids[0]


def prepare_request(
    manifest: dict, base_dir: Path, model: str, *, backend: str = "vllm_omni"
) -> tuple[dict, list[Path]]:
    """Build a Videos API request and its upload list for one of ``BACKENDS``.

    Both backends use multipart form fields. vLLM-Omni uploads every file and
    refers to it by a reference index; SGLang reads each file from an absolute
    path visible to the server, so the upload list stays empty. For SGLang,
    ``lidar.return_output`` is translated to ``lidar.decode``.
    """
    backend = _check_backend(backend)
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
                if backend == "sglang":
                    view[f"{role}_path"] = str(path)
                else:
                    view[f"{role}_reference_index"] = len(paths)
                    paths.append(path)
    if "lidar" in extra:
        lidar = extra["lidar"]
        if backend == "sglang":
            if set(lidar) - {"control_path", "return_output", "decode"}:
                raise ValueError("SGLang LiDAR accepts control_path and decode (or the return_output alias).")
            decode = lidar.get("decode", lidar.get("return_output", False))
            if not isinstance(decode, bool) or (
                "return_output" in lidar
                and (not isinstance(lidar["return_output"], bool) or lidar["return_output"] != decode)
            ):
                raise ValueError("LiDAR decode/return_output must be booleans and agree when both are supplied.")
        elif lidar.get("return_output") is not True:
            raise ValueError("The joint walkthrough requires lidar.return_output=true.")
        reference_suffix = "path" if backend == "sglang" else "reference_index"
        if any(
            f"vision_{reference_suffix}" not in view or f"control_{reference_suffix}" not in view
            for view in extra["multiview"]["views"]
        ):
            raise ValueError("The joint I2V example requires WSM and RGB for every camera.")
        if backend == "sglang":
            lidar.pop("return_output", None)
            lidar["decode"] = decode
            lidar["control_path"] = str(require_asset(base_dir / lidar["control_path"]))
        else:
            path = require_asset(base_dir / lidar.pop("control_path"))
            lidar["control_reference_index"] = len(paths)
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
    """Send multipart form fields with optional file uploads."""
    if not paths:
        # A None filename makes each part a regular field. Passing only data=
        # would use URL encoding, which SGLang's Videos API does not parse.
        return client.post(endpoint, files=[(key, (None, value)) for key, value in form.items()])
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


def submit_wsm(
    client, form: dict, paths: list[Path], output_dir: Path, *, async_mode: bool = False, timeout: float = 3600
) -> Path:
    """Save the WSM video, optionally submitting and polling an async job.

    For SGLang, prepare the request with backend="sglang" and set async_mode=True.
    The call blocks until the video is saved; timeout limits async job polling.
    """
    path = output_dir / "multiview_camera_major.mp4"
    if async_mode:
        job = _submit_video_job(client, form, paths, output_dir, timeout=timeout)
        _download(client, f"/v1/videos/{job['id']}/content", path)
        return path
    output_dir.mkdir(parents=True, exist_ok=True)
    response = _post(client, "/v1/videos/sync", form, paths)
    _check_response(response, output_dir)
    if not response.content:
        raise ValueError("The synchronous video response was empty.")
    path.write_bytes(response.content)
    return path


def _submit_video_job(
    client, form: dict, paths: list[Path], output_dir: Path, *, timeout: float, interval: float = 10
) -> dict:
    """Submit an async Videos API job and poll until it completes.

    Every job response is saved as ``job.json``. A failed or cancelled job saves
    the response as ``response_error.txt`` and raises; a polling timeout raises
    but leaves the job running on the server.
    """
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
        if job.get("status") in {"failed", "cancelled"}:
            (output_dir / "response_error.txt").write_text(json.dumps(job, indent=2))
            raise RuntimeError(f"Video generation {job['status']}: {job.get('error')}")
        _check_response(response, output_dir)
        if job.get("status") == "completed":
            break
        if job.get("status") not in {"queued", "in_progress"}:
            raise RuntimeError(f"Unexpected job status: {job}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Polling timed out; job remains at {client.base_url.join(job_url)}. See job.json.")
        time.sleep(min(interval, remaining))
        response = client.get(job_url, timeout=max(0.001, min(30, deadline - time.monotonic())))
        try:
            job = response.json()
        except ValueError:
            _check_response(response, output_dir)
            raise
    return job


def submit_joint(
    client, form: dict, paths: list[Path], output_dir: Path, *, timeout: float = 3600, backend: str = "vllm_omni"
) -> dict:
    """Save a joint job's video and, for SGLang decode requests, its native LiDAR artifact.

    vLLM-Omni returns only the LiDAR descriptor; download its numeric output from /lidar separately.
    """
    backend = _check_backend(backend)
    return_lidar = True
    if backend == "sglang":
        return_lidar = json.loads(form["extra_params"])["lidar"].get("decode", False)
    job = _submit_video_job(client, form, paths, output_dir, timeout=timeout)
    job_url = f"/v1/videos/{job['id']}"
    if return_lidar and not job.get("lidar"):
        raise ValueError("Completed job is missing its requested LiDAR artifact; see job.json.")
    if return_lidar and backend == "sglang" and not job["lidar"].get("files", {}).get("rangemap"):
        raise ValueError("Completed job is missing its LiDAR rangemap file; see job.json.")
    video_path = output_dir / "multiview_camera_major.mp4"
    _download(client, job_url + "/content", video_path)
    if backend != "sglang":
        return {"video": video_path, "descriptor": job["lidar"], "job_id": job["id"]}
    if not return_lidar:
        return {"video": video_path, "lidar": None, "descriptor": None, "job_id": job["id"]}
    lidar_path = output_dir / "multiview_camera_major.lidar.safetensors"
    _download(client, job_url + "/content?variant=rangemap", lidar_path)
    lidar_path.with_suffix(".json").write_text(json.dumps(job["lidar"], indent=2) + "\n")
    return {"video": video_path, "lidar": lidar_path, "descriptor": job["lidar"], "job_id": job["id"]}


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


def save_outputs(output_dir: Path, video: Path, cameras: dict, manifest: dict, *, lidar: dict | None = None) -> None:
    """Write ``sample_outputs.json`` for a camera-major response split into per-camera files."""
    summary = {
        "camera_major_response": str(video),
        "camera_order": list(cameras),
        "frames_per_view": manifest["num_frames"],
        "fps": manifest["fps"],
        "files_by_camera": {key: str(path) for key, path in cameras.items()},
    }
    if lidar is not None:
        summary["job_id"] = lidar["job_id"]
        if lidar["lidar"] is not None:
            summary.update(lidar_file=str(lidar["lidar"]), lidar=lidar["descriptor"])
    (output_dir / "sample_outputs.json").write_text(json.dumps(summary, indent=2) + "\n")
