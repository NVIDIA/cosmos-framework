# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only request helpers for the seven-camera Sim-Auto vLLM-Omni notebook.

Requests use the shared daytime sample: a WSM control, an RGB condition, and a caption per
camera, plus an optional numeric LiDAR control for joint generation.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from uuid import uuid4

from helpers.auto_helpers import require_asset
from helpers.daytime_sample import SAMPLE_CAMERAS, resolve_request

CAMERA_ORDER = SAMPLE_CAMERAS
MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Auto"


def load_request(
    path: Path, sample_root: Path, *, control_dir: Path | None = None,
    caption_folder: str | None = "captions_with_timestamp", lidar_output: Path | None = None,
) -> dict:
    """Resolve media and captions under ``sample_root``; optionally take WSM controls from ``control_dir``.

    ``control_dir`` must hold one ``<camera_key>.mp4`` control per camera, for example a
    different WSM render of the same scene.
    """
    manifest = resolve_request(path, sample_root, caption_folder=caption_folder, lidar_output=lidar_output)
    if control_dir is not None:
        for view in manifest["multiview"]["views"]:
            view["control_path"] = str((Path(control_dir).expanduser() / Path(view["control_path"]).name).resolve())
    _validate_contract(manifest)
    return manifest


def _validate_contract(manifest: dict) -> None:
    if manifest.get("wsm") != {}:
        raise ValueError("The Sim-Auto sample requires wsm: {}.")
    multiview = manifest.get("multiview", {})
    views = multiview.get("views", [])
    if tuple(view.get("camera_key") for view in views) != CAMERA_ORDER:
        raise ValueError("Use the sample's seven cameras in canonical request order.")
    if multiview.get("condition_video_as_image") is not True:
        raise ValueError("Set condition_video_as_image=true for first-frame RGB conditioning.")
    if not isinstance(manifest.get("prompt"), str) or not manifest["prompt"].strip():
        raise ValueError("The Videos API requires a nonempty top-level prompt.")
    frames = manifest.get("num_frames")
    fps = manifest.get("fps")
    if type(frames) is not int or frames < 1 or (frames - 1) % 4:
        raise ValueError("num_frames must be a positive 4k+1 integer.")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    if type(manifest.get("seed")) is not int:
        raise ValueError("Supply an integer seed in the request.")
    if manifest.get("resolution") != "480" or manifest.get("aspect_ratio", "16,9") != "16,9":
        raise ValueError("This showcase uses resolution '480' and aspect_ratio '16,9'.")
    if multiview.get("num_frames", frames) != frames:
        raise ValueError("Top-level and multiview frame counts must agree.")
    if manifest.get("guidance_scale", 1.0) != 1.0 or manifest.get("num_inference_steps", 4) != 4:
        raise ValueError("Sim-Auto requires guidance 1.0 and its fixed four-step schedule.")
    forbidden = (
        "guidance", "num_steps", "shift", "flow_shift", "negative_prompt",
        "negative_metadata_mode", "per_view_negative_prompt", "edge", "blur", "depth", "seg",
    )
    for field in forbidden:
        if field in manifest:
            raise ValueError(f"Remove {field!r} from this Sim-Auto request.")
    if "lidar" in manifest:
        lidar = manifest["lidar"]
        if not isinstance(lidar, dict) or set(lidar) != {"control_path", "return_output"} or lidar["return_output"] is not True:
            raise ValueError("A joint request needs lidar with only control_path and return_output=true.")
        if not isinstance(lidar["control_path"], str) or not lidar["control_path"].endswith(".safetensors"):
            raise ValueError("The LiDAR control must be a .safetensors path.")
        if fps != 30:
            raise ValueError("Joint LiDAR generation requires the checkpoint's 30 FPS camera clock.")
    for view in views:
        if not isinstance(view.get("prompt"), str) or not view["prompt"].strip():
            raise ValueError(f"Missing caption for {view['camera_key']}.")
        for role in ("control", "vision"):
            if not isinstance(view.get(f"{role}_path"), str) or not view[f"{role}_path"]:
                raise ValueError(f"Missing {role}_path for {view['camera_key']}.")
            if f"{role}_reference_index" in view:
                raise ValueError("Supply local paths in the manifest; upload indexes are generated.")


def require_inputs(manifest: dict) -> None:
    """Report every missing or unhydrated asset before any GPU request."""
    _validate_contract(manifest)
    errors = []
    for view in manifest["multiview"]["views"]:
        for role in ("control", "vision"):
            try:
                require_asset(Path(view[f"{role}_path"]))
            except (FileNotFoundError, ValueError) as error:
                errors.append(f"{view['camera_key']} {role}: {error}")
        if Path(view["control_path"]).resolve() == Path(view["vision_path"]).resolve():
            errors.append(f"{view['camera_key']}: WSM and RGB must be separate inputs.")
    if "lidar" in manifest:
        try:
            require_asset(Path(manifest["lidar"]["control_path"]))
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"LiDAR control: {error}")
    if errors:
        raise ValueError(
            "Sim-Auto needs seven matching WSM controls and seven RGB conditions, plus the LiDAR "
            "control for a joint request. Rerun sample download and local LiDAR preparation. "
            "For missing WSM controls, set the notebook's CONTROL_DIR to a "
            "directory with one <camera_key>.mp4 control per camera.\n"
            + "\n".join(errors)
        )


def prepare_request(manifest: dict, model: str = MODEL_ID) -> tuple[dict, list[Path]]:
    """Translate ordered WSM/RGB paths, and an optional LiDAR control, to the multipart Videos API.

    Camera *i* uploads its control as file 2*i* and its RGB condition as file 2*i* + 1; a
    LiDAR control is the last upload. Every call starts a fresh generation session.
    """
    require_inputs(manifest)
    multiview = copy.deepcopy(manifest["multiview"])
    multiview["num_frames"] = manifest["num_frames"]
    sources = []
    for view in multiview["views"]:
        for role in ("control", "vision"):
            view[f"{role}_reference_index"] = len(sources)
            sources.append(view.pop(f"{role}_path"))
    extra = {
        "wsm": {}, "multiview": multiview, "resolution": "480", "aspect_ratio": "16,9",
        "session_id": f"sim-auto-{uuid4().hex}", "reset": True, "close_session": True,
    }
    if "lidar" in manifest:
        extra["lidar"] = {"control_reference_index": len(sources), "return_output": True}
        sources.append(manifest["lidar"]["control_path"])
    paths = []
    for source in sources:
        path = require_asset(Path(source))
        if path.stat().st_size > 512 * 1024**2:
            raise ValueError(f"Upload exceeds the 512 MiB file limit: {path}")
        paths.append(path)
    form = {key: str(manifest[key]) for key in ("prompt", "num_frames", "fps", "seed")}
    form.update(
        model=model, size="832x480", guidance_scale="1.0", num_inference_steps="4",
        enable_frame_interpolation="false", extra_params=json.dumps(extra),
    )
    return form, paths
