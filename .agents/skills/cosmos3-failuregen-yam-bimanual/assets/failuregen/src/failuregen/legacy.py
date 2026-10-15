# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read run directories written before this package existed.

The scripts this package replaced wrote the same information under different names: the host
episode was ``good``, the recipe was ``category``, and the mode was ``both`` or ``grounded``
rather than transplant or control. Batches rendered with them are the only large corpus of
results that exists, so scoring and exporting have to read them.

The mapping lives here rather than in the modules that consume it, so there is exactly one place
that knows the old vocabulary and it can be deleted once the old batches are gone.
"""

from __future__ import annotations

import json
from pathlib import Path

MODES = {
    "both": "transplant",
    "transplant_only": "transplant",
    "grounded": "control",
    "prompt_only": "prompt_only",
    "gripper_release": "gripper_release",
}


def read_record(run_dir: Path) -> dict | None:
    """The run's record in the current shape, or None if there is no record.

    Records already in the current shape pass through untouched, so callers can point at a mix
    of old and new batches without caring which is which.
    """
    path = Path(run_dir) / "record.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if "host" in raw:
        return raw
    return translate(raw, name=Path(run_dir).name)


def translate(raw: dict, name: str) -> dict:
    host = raw.get("good") or {}
    donor = raw.get("failure_event")
    mode = MODES.get(raw.get("mode", ""), raw.get("mode"))
    if mode == "control":
        donor = None  # a grounded run replays the host, whatever the manifest carried alongside
    return {
        "name": name,
        "recipe": raw.get("category"),
        "scenario_id": raw.get("scenario_id"),
        "mode": mode,
        "arm": raw.get("arm"),
        "target_phase": raw.get("target_phase"),
        "donor_phase": raw.get("donor_phase"),
        "dims": raw.get("transplant_dims"),
        "blend_frames": raw.get("blend_frames"),
        "jaw_amplitude": raw.get("jaw_amplitude"),
        "pose_distance": raw.get("pose_distance"),
        "pre_chunks": raw.get("pre_chunks"),
        "post_chunks": raw.get("post_chunks"),
        "fps": raw.get("fps", 5),
        "failure_time_s": raw.get("failure_time_s", 0.0),
        "caption": raw.get("prompt_after_injection") or raw.get("caption", ""),
        "phases": raw.get("phases") or [],
        "host": {
            "source": host.get("source"),
            "episode_index": host.get("episode_index"),
            "t_start": host.get("t_start", 0.0),
            "t_inject": host.get("t_inject", 0.0),
        },
        "donor": donor,
        "legacy": True,
    }
