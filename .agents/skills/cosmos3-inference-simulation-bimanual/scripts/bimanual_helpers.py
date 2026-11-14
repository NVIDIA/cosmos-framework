# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only input preparation and HTTP client for the SGLang Bimanual notebook.

This client targets the unified (schema-5) Cosmos3-Nano-Sim-Bimanual export: 59-value
rows that are already normalized, with one embodiment name per row. The server
validates the checkpoint contract, pads rows to the model width and routes every row
to its embodiment domain.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Keep the server's order for ties when selecting the closest image aspect.
CANVASES = ((640, 640), (736, 544), (544, 736), (832, 480), (480, 832))
VIEWPOINTS = {"ego_view", "third_person_view", "wrist_view", "concat_view"}
UNIFIED_ACTION_SPACE = "normalized_unified_v1"
UNIFIED_ACTION_DIM = 59
# Keys of the earlier native (29-value) recipe that the unified export does not take.
_RETIRED_KEYS = ("camera_trajectory", "camera_pose_convention", "camera_translation_scale", "camera_action_normalization")


@dataclass(frozen=True)
class BimanualInput:
    prompt: str
    image: Image.Image
    action: np.ndarray
    domain_names: list[str]
    extra_args: dict
    num_frames: int
    fps: int
    seed: int
    height: int
    width: int
    output_height: int
    output_width: int
    metadata: dict


def _integer(value, name: str, minimum: int = 1) -> int:
    """Return ``value`` as an ``int``; integral floats such as ``30.0`` are accepted, booleans are not."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value).is_integer()
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _asset(value, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Supply a local image/video or action file path.")
    path = (base_dir / Path(value).expanduser()).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Unhydrated Git LFS pointer: {path}. Run git lfs pull.")
    return path


def first_media_image(path: Path) -> Image.Image:
    """Read a local image or extract the first video frame with FFmpeg."""
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
            ],
            capture_output=True,
            check=True,
        )
        if not result.stdout:
            raise ValueError(f"No conditioning frame could be decoded from {path}.")
        with Image.open(BytesIO(result.stdout)) as image:
            return image.convert("RGB")


def _numeric_array(value, name: str) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=object)
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.integer, np.floating))
            for item in values.flat
        ):
            raise ValueError(f"{name} must contain real numbers, not booleans or strings.")
        array = values.astype(np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric {name}.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def load_action_sidecar(path: Path, *, default_domain: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Rows and one embodiment name per row from an action file.

    The file is either the upstream unified sidecar object
    ``{"action_space": "normalized_unified_v1", "action": rows, "domain_names": names}``
    or a bare array of rows, in which case every row uses ``default_domain``.
    ``domain_names`` may hold one name for the whole clip or one name per row.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        space = payload.get("action_space")
        if space != UNIFIED_ACTION_SPACE:
            raise ValueError(f"{path}: action_space must be {UNIFIED_ACTION_SPACE!r}, got {space!r}.")
        if "action" not in payload:
            raise ValueError(f"{path}: the sidecar object needs an `action` list.")
        rows = _numeric_array(payload["action"], "actions")
        names = payload.get("domain_names")
        if names is None:
            if default_domain is None:
                raise ValueError(f"{path}: the sidecar has no domain_names and the record sets no domain_name.")
            names = [default_domain]
        if isinstance(names, str):
            names = [names]
    else:
        rows = _numeric_array(payload, "actions")
        if default_domain is None:
            raise ValueError(f"{path}: a bare action array needs domain_name in the record.")
        names = [default_domain]
    if rows.ndim != 2:
        raise ValueError(f"{path}: actions must be a list of rows.")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{path}: domain_names must be a non-empty list of embodiment names.")
    if len(names) == 1:
        names = names * len(rows)
    if len(names) != len(rows):
        raise ValueError(f"{path}: {len(names)} domain names for {len(rows)} action rows; supply one per row.")
    return rows, [name.strip() for name in names]


def domain_runs(names: list[str]) -> list[tuple[str, int, int]]:
    """Consecutive (name, start_row, end_row) segments; end is exclusive."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(names) + 1):
        if index == len(names) or names[index] != names[start]:
            runs.append((names[start], start, index))
            start = index
    return runs


def prepare_bimanual_input(record: dict, *, base_dir: Path) -> BimanualInput:
    """Prepare the notebook's unified-schema example without a runtime package."""
    num_frames = _integer(record.get("num_frames", 901), "num_frames", 2)
    if (num_frames - 1) % 4:
        raise ValueError("The Bimanual recipe requires num_frames = 4k+1.")
    fps = _integer(record.get("fps", 30), "fps")
    seed = _integer(record.get("seed", 42), "seed", 0)
    if str(record.get("resolution", record.get("image_size", 480))) != "480":
        raise ValueError("This Bimanual helper supports the checkpoint's 480p canvases.")
    if record.get("model_mode", "forward_dynamics") != "forward_dynamics":
        raise ValueError("This cookbook uses an initial image and actions for forward dynamics.")
    if record.get("autoregressive") is False:
        raise ValueError("Bimanual generation is autoregressive.")
    retired = [key for key in _RETIRED_KEYS if key in record]
    if retired or "action_space" in record:
        raise ValueError(
            f"{retired or ['action_space']} belong to the earlier 29-value recipe. The unified export takes "
            f"pre-normalized {UNIFIED_ACTION_DIM}-value rows in a {UNIFIED_ACTION_SPACE!r} sidecar; convert camera "
            "trajectories to unified rows before submitting them."
        )
    normalization = record.get("action_normalization", "none")
    if normalization != "none":
        raise ValueError("Unified sidecars are already normalized; action_normalization must be 'none'.")
    if any(record.get(key) is not None for key in ("action", "actions")):
        raise ValueError("Point action_path at the sidecar file; inline rows are not supported here.")

    action_path = _asset(record.get("action_path"), base_dir)
    rows, domain_names = load_action_sidecar(action_path, default_domain=record.get("domain_name"))
    if rows.shape != (num_frames - 1, UNIFIED_ACTION_DIM):
        raise ValueError(f"Expected {num_frames - 1}x{UNIFIED_ACTION_DIM} unified actions, got {rows.shape}.")
    with np.errstate(over="ignore"):
        action = rows.astype(np.float32)
    if not np.isfinite(action).all():
        raise ValueError("Actions must be finite in float32.")
    if record.get("raw_action_dim", UNIFIED_ACTION_DIM) != UNIFIED_ACTION_DIM:
        raise ValueError(f"raw_action_dim must be {UNIFIED_ACTION_DIM} for the unified export.")
    if _integer(record.get("action_chunk_size", num_frames - 1), "action_chunk_size") != num_frames - 1:
        raise ValueError("action_chunk_size must equal num_frames - 1.")
    domain = record.get("domain_name", domain_names[0])
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain_name must be an embodiment name.")

    vision_path = _asset(record.get("vision_path"), base_dir)
    image = first_media_image(vision_path)
    height, width = record.get("height"), record.get("width")
    if height is None and width is None:
        width, height = min(CANVASES, key=lambda size: abs(image.height / image.width - size[1] / size[0]))
    elif height is None or width is None:
        raise ValueError("height and width must be supplied together.")
    else:
        height, width = _integer(height, "height"), _integer(width, "width")
        if (width, height) not in CANVASES:
            raise ValueError(f"Choose a trained 480p canvas (width, height) from {CANVASES}.")
    scale = min(width / image.width, height / image.height, 1.0)
    output_width = max(int(image.width * scale + 0.5) // 16, 1) * 16
    output_height = max(int(image.height * scale + 0.5) // 16, 1) * 16
    viewpoint = record.get("view_point", "ego_view")
    if viewpoint not in VIEWPOINTS:
        raise ValueError(f"Unsupported view_point: {viewpoint!r}.")
    prompt = record.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    extra_args = {
        "action_mode": "forward_dynamics",
        "domain_name": domain,
        "domain_names": domain_names,
        "action_normalization": "none",
        "action_view_point": viewpoint,
    }
    # The server defaults to the export's caption format (plain text for the
    # unified checkpoint); a record may still pin it explicitly.
    if "format_prompt_as_json" in record:
        if not isinstance(record["format_prompt_as_json"], bool):
            raise ValueError("format_prompt_as_json must be a boolean.")
        extra_args["format_prompt_as_json"] = record["format_prompt_as_json"]
    metadata = {
        "vision_path": str(vision_path), "action_source": str(action_path),
        "action_space": UNIFIED_ACTION_SPACE,
        "domain_runs": [{"domain_name": name, "rows": [start, end]} for name, start, end in domain_runs(domain_names)],
        "num_frames": num_frames, "fps": fps, "seed": seed,
        "canvas": {"height": height, "width": width},
        "output": {"height": output_height, "width": output_width},
        **{key: value for key, value in extra_args.items() if key != "domain_names"},
    }
    return BimanualInput(
        prompt=prompt.strip(), image=image, action=action, domain_names=domain_names, extra_args=extra_args,
        num_frames=num_frames, fps=fps, seed=seed, height=height, width=width,
        output_height=output_height, output_width=output_width, metadata=metadata,
    )


MAX_FORM_FIELD_BYTES = 1024 * 1024  # The server parses each multipart form field within this limit.


def encode_extra_params(prepared: BimanualInput) -> str:
    """JSON for the ``extra_params`` form field, with the action rows in their shortest float32 form.

    Rows written with Python's double ``repr`` can exceed the server's 1 MiB form-field
    limit for long rollouts. The shortest decimal that round-trips each float32 value is
    about half the size and reproduces the same tensor on the server; the result is checked
    before it is sent.
    """
    payload = dict(prepared.extra_args)
    try:
        import orjson
    except ImportError:
        payload["action"] = [
            [float(np.format_float_positional(value, unique=True, trim="-")) for value in row]
            for row in prepared.action
        ]
        text = json.dumps(payload, separators=(",", ":"))
    else:
        payload["action"] = prepared.action
        text = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
    decoded = np.asarray(json.loads(text)["action"], dtype=np.float32)
    if decoded.shape != prepared.action.shape or not np.array_equal(decoded, prepared.action):
        raise ValueError("Action serialization must preserve every float32 value.")
    size = len(text.encode("utf-8"))
    if size > MAX_FORM_FIELD_BYTES:
        raise ValueError(
            f"The action payload is {size:,} bytes, above the server's {MAX_FORM_FIELD_BYTES:,}-byte multipart-field "
            "limit; shorten the rollout or raise the server's form-field limit."
        )
    return text


def submit_bimanual(
    client, prepared: BimanualInput, model: str, output_path: Path, *, timeout: float = 3600, poll_interval: float = 5
) -> Path:
    """Submit multipart image/actions, poll an async job, and stream its video.

    The client is an httpx.Client with the server base_url and optional auth.
    timeout bounds job polling; the client's timeout applies to upload/download.
    The step count and guidance are fixed by the distilled checkpoint, so the
    request does not send them.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, poll_interval)):
        raise ValueError("Polling timeout and interval must be positive and finite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(".error.txt")

    def check(response):
        if response.is_error:
            error_path.write_text(response.text)
        response.raise_for_status()

    form = {
        "model": model, "prompt": prepared.prompt,
        "size": f"{prepared.width}x{prepared.height}",
        "num_frames": str(prepared.num_frames), "fps": str(prepared.fps),
        "seed": str(prepared.seed),
        "extra_params": encode_extra_params(prepared),
    }
    output_path.with_suffix(".request.json").write_text(json.dumps(form, indent=2) + "\n")
    output_path.with_suffix(".metadata.json").write_text(json.dumps(prepared.metadata, indent=2) + "\n")
    image_buffer = BytesIO()
    prepared.image.save(image_buffer, format="PNG")
    response = client.post(
        "/v1/videos", data=form,
        files={"input_reference": ("conditioning.png", image_buffer.getvalue(), "image/png")},
    )
    check(response)
    job = response.json()
    job_url = f"/v1/videos/{job['id']}"
    deadline = time.monotonic() + timeout
    print(f"Submitted {job['id']} at {client.base_url.join(job_url)}")
    while True:
        output_path.with_suffix(".job.json").write_text(json.dumps(job, indent=2) + "\n")
        status = job.get("status")
        if status in {"failed", "cancelled"}:
            error_path.write_text(json.dumps(job, indent=2) + "\n")
            raise RuntimeError(f"Video generation {status}: {job.get('error')}")
        if status == "completed":
            break
        if status not in {"queued", "in_progress"}:
            raise RuntimeError(f"Unexpected video job status: {job}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Polling timed out; job remains at {client.base_url.join(job_url)}.")
        time.sleep(min(poll_interval, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Polling timed out; job remains at {client.base_url.join(job_url)}.")
        response = client.get(job_url, timeout=min(30, remaining))
        check(response)
        job = response.json()
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with client.stream("GET", job_url + "/content") as response:
            if response.is_error:
                response.read()
            check(response)
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError("The generated video response was empty.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
