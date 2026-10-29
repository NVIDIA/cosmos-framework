# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only request helpers for the seven-camera Sim-Auto SGLang notebook.

SGLang serves the same ``multiview.views[]`` request as vLLM-Omni on the shared daytime
sample, but opens every WSM control, RGB condition, and LiDAR control from a path the
server can read instead of accepting uploads, and runs each rollout as one asynchronous
Videos API job without session fields. Manifest loading and the input checks are shared
with the vLLM-Omni helpers.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from helpers.auto_helpers import require_asset
from helpers.sim_auto_vllm_omni_helpers import CAMERA_ORDER, MODEL_ID, load_request, require_inputs

__all__ = ["CAMERA_ORDER", "MODEL_ID", "load_request", "prepare_request", "require_inputs"]


def _number(value: float) -> str:
    """Multipart form text for a number: ``30.0`` travels as ``30``."""
    return str(int(value)) if float(value).is_integer() else str(value)


def prepare_request(manifest: dict, model: str = MODEL_ID) -> tuple[dict, list[Path]]:
    """Translate a loaded Sim-Auto request to SGLang's multipart Videos API fields.

    Returns the form fields and an empty upload list: every ``control_path`` and
    ``vision_path``, and a joint request's ``lidar.control_path``, travel absolute inside
    the JSON-encoded ``extra_params`` and must be readable at that path on the server.
    The form carries no session fields, so two calls for the same manifest produce the
    same request.
    """
    require_inputs(manifest)
    multiview = copy.deepcopy(manifest["multiview"])
    multiview["num_frames"] = manifest["num_frames"]
    for view in multiview["views"]:
        for role in ("control", "vision"):
            view[f"{role}_path"] = str(require_asset(Path(view[f"{role}_path"])))
    extra = {"wsm": {}, "multiview": multiview, "resolution": "480", "aspect_ratio": "16,9"}
    if "lidar" in manifest:
        extra["lidar"] = {"control_path": str(require_asset(Path(manifest["lidar"]["control_path"]))), "decode": True}
    form = {
        "model": model,
        "prompt": manifest["prompt"],
        "size": "832x480",
        "num_frames": str(manifest["num_frames"]),
        "fps": _number(manifest["fps"]),
        "seed": str(manifest["seed"]),
        "guidance_scale": "1.0",
        "num_inference_steps": "4",
        "extra_params": json.dumps(extra),
    }
    return form, []
