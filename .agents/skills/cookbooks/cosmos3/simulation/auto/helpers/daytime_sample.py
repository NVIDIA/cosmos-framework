# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fetch the shared daytime Auto sample and prepare native and vLLM-Omni requests.

The sample is published with nvidia/Cosmos3-Nano-Transfer-Auto under ``assets/sample_data``
and serves both Transfer-Auto and Sim-Auto: one scene, seven synchronized cameras, each with
an RGB recording, a WSM control, and a caption file. The request files under
``assets/daytime_scene/`` name those files relative to the sample root and keep only
sampling settings in the cookbook. Any vLLM-Omni LiDAR conversion is written locally.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from helpers.auto_helpers import CAMERA_ORDER, require_asset
from helpers.lidar_rangemap import convert_control
from helpers.wsm_helpers import caption_window

SAMPLE_REPO = "nvidia/Cosmos3-Nano-Transfer-Auto"
SAMPLE_REVISION = "b8dcc2f70421e31db86d1b81d194c791d65eaad2"
SAMPLE_PREFIX = "assets/sample_data"
SAMPLE_SCENE = "daytime_scene"
SAMPLE_CAMERAS = CAMERA_ORDER[:7]
CAPTION_FOLDERS = ("captions", "captions_with_timestamp")


def sample_files() -> list[str]:
    """Files used by the native and serving examples, relative to the sample root."""
    return [
        f"{SAMPLE_SCENE}/{folder}/{camera}.{suffix}"
        for camera in SAMPLE_CAMERAS
        for folder, suffix in (("rgb", "mp4"), ("wsm", "mp4"), *((name, "json") for name in CAPTION_FOLDERS))
    ] + ["input-transfer-auto.jsonl", "input-sim-auto.jsonl", f"{SAMPLE_SCENE}/lidar/control.tar"]


def verify_sample(root: Path) -> Path:
    """Check every needed file against the sample's ``SHA256SUMS``; report all problems at once."""
    root = Path(root).expanduser().resolve()
    checksums = require_asset(root / "SHA256SUMS")
    expected = {}
    for line in checksums.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.removeprefix("./")] = digest
    errors = []
    for relative in sample_files():
        try:
            path = require_asset(root / relative)
        except (FileNotFoundError, ValueError) as error:
            errors.append(str(error))
            continue
        if relative not in expected:
            errors.append(f"{relative} is not listed in SHA256SUMS.")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected[relative]:
            errors.append(f"Checksum mismatch: {path}")
    if errors:
        raise ValueError(f"The sample at {root} is incomplete or modified:\n" + "\n".join(errors))
    return root


def fetch_sample(local_root: str | Path | None = None) -> Path:
    """Return a verified sample root: ``local_root`` if given, else a pinned Hugging Face download.

    Downloads inputs, both caption styles and native requests, but no model weights
    or recorded LiDAR targets. Later runs reuse the Hugging Face cache.
    """
    if local_root is None:
        from huggingface_hub import snapshot_download

        patterns = [f"{SAMPLE_PREFIX}/{relative}" for relative in ("SHA256SUMS", *sample_files())]
        snapshot = snapshot_download(SAMPLE_REPO, revision=SAMPLE_REVISION, allow_patterns=patterns)
        local_root = Path(snapshot) / SAMPLE_PREFIX
    return verify_sample(Path(local_root))


def read_caption(path: Path, num_frames: int) -> str:
    """Return the caption of the chunk that starts at frame 0 and covers ``num_frames``."""
    return caption_window(json.loads(require_asset(Path(path)).read_text()), num_frames, 30)


def native_request(request_path: Path, *, mode: str, caption_folder: str | None = None) -> dict:
    """Resolve one native JSON/JSONL request without converting captions or LiDAR.

    Sample caption selection is explicit; custom requests keep their own paths by
    leaving caption_folder unset. Camera keys and their order are preserved.
    Joint requests retain all-view RGB conditioning when supplied; without RGB,
    they generate camera and LiDAR output from geometry and captions alone.
    """
    if mode not in {"camera", "joint", "text2video"}:
        raise ValueError("Choose camera, joint, or text2video.")
    if caption_folder is not None and caption_folder not in CAPTION_FOLDERS:
        raise ValueError(f"Choose a caption folder from {CAPTION_FOLDERS}.")
    # HF snapshots symlink files into a blob store. Relative request paths belong
    # to the snapshot directory, not to the dereferenced blob's parent.
    request_path = Path(request_path).expanduser().absolute()
    request = json.loads(require_asset(request_path).read_text())
    base = request_path.parent
    multiview = request["multiview"]
    has_rgb = [view.get("vision_path") is not None for view in multiview["views"]]
    if mode == "joint" and any(has_rgb) and not all(has_rgb):
        raise ValueError("Joint requests require RGB for every selected camera, or none.")
    conditioned = mode == "camera" or (mode == "joint" and any(has_rgb))
    for view in multiview["views"]:
        if caption_folder is not None:
            view["caption_path"] = f"{SAMPLE_SCENE}/{caption_folder}/{view['camera_key']}.json"
        if mode == "text2video":
            view.pop("vision_path", None)
        for field in ("control_path", "vision_path", "caption_path"):
            if field in view:
                path = (base / view[field]).absolute()
                require_asset(path)
                view[field] = str(path)
        read_caption(Path(view["caption_path"]), request["num_frames"])
    multiview["condition_video_as_image"] = conditioned
    request["model_mode"] = "image2video" if conditioned else "text2video"
    if not conditioned:
        request["num_conditional_frames"] = 0
        request["condition_frame_indexes_vision"] = []
    if mode != "joint":
        request.pop("lidar", None)
    else:
        lidar = request["lidar"]
        for field in ("control_path", "condition_path"):
            if field in lidar:
                path = (base / lidar[field]).absolute()
                require_asset(path)
                lidar[field] = str(path)
    return request


def resolve_request(
    request_path: Path,
    sample_root: Path,
    *,
    caption_folder: str | None = "captions_with_timestamp",
    lidar_output: Path | None = None,
) -> dict:
    """Load a request file and make it self-contained for the vLLM-Omni helpers.

    Each view's ``control_path`` and ``vision_path`` resolve under ``sample_root``, and its
    ``caption_path``, if present, is replaced by the caption text as ``prompt``.
    ``prompt_from_camera`` selects the top-level Videos API prompt. Native range-map
    archives are converted only for vLLM-Omni, into the caller's output directory.
    Requests with inline prompts pass through with only their paths resolved.
    """
    request_path = require_asset(Path(request_path))
    sample_root = Path(sample_root).resolve()
    manifest = copy.deepcopy(json.loads(request_path.read_text()))
    if caption_folder is not None and caption_folder not in CAPTION_FOLDERS:
        raise ValueError(f"Choose a caption folder from {CAPTION_FOLDERS}.")
    views = manifest["multiview"]["views"]
    for view in views:
        for role in ("control", "vision"):
            if f"{role}_path" in view:
                view[f"{role}_path"] = str(sample_root / view[f"{role}_path"])
        if "caption_path" in view:
            caption = Path(view.pop("caption_path"))
            if caption_folder is not None:
                caption = caption.parent.parent / caption_folder / caption.name
            view["prompt"] = read_caption(sample_root / caption, manifest["num_frames"])
    if "prompt_from_camera" in manifest:
        prompts = {view["camera_key"]: view.get("prompt") for view in views}
        source = manifest.pop("prompt_from_camera")
        if not prompts.get(source):
            raise ValueError(f"prompt_from_camera {source!r} is not a captioned camera of this request.")
        manifest["prompt"] = prompts[source]
    if "lidar" in manifest:
        # Preserve the snapshot filename: a blob-store target has no .tar suffix.
        control = sample_root / manifest["lidar"]["control_path"]
        require_asset(control)
        if control.suffix == ".tar":
            if lidar_output is None:
                raise ValueError("Set lidar_output to a local .safetensors path for vLLM-Omni conversion.")
            convert_control(control, lidar_output)
            control = lidar_output.resolve()
        manifest["lidar"]["control_path"] = str(control)
    return manifest
