# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Render an already encoded action window without joint conversion or normalization.

The notebook uses this path for normalized 20-D EEF actions. It checks the saved
checkpoint contract and preserves the exact conditioning inputs and provenance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(profile, run: dict) -> tuple[Path, list]:
    if profile.model.get("action_encoding") != "precomputed":
        raise ValueError("prepared manifest requires model.action_encoding: precomputed")
    prepared = run["prepared"]
    if profile.model.get("normalization") == "train_quantile":
        from .yam_training.contract import EEF_CONTRACT, load_stats

        stats_path = Path(profile.model["normalizer_stats"])
        contract_path = Path(profile.model["training_contract_file"])
        contract = json.loads(contract_path.read_text())
        stats = load_stats(stats_path)
        if (
            stats["contract"] != EEF_CONTRACT
            or contract["contract"] != EEF_CONTRACT
            or prepared.get("normalization") != "train_quantile"
            or prepared.get("stats_sha256") != sha256(stats_path)
            or contract["stats_sha256"] != prepared["stats_sha256"]
            or prepared.get("training_contract_sha256") != sha256(contract_path)
            or prepared.get("config_sha256") != sha256(Path(profile.model["config_file"]))
        ):
            raise ValueError("Prepared EEF inputs differ from the saved training config/normalizer")
    if run["mode"] not in ("control", "perturbation"):
        raise ValueError("prepared runs must be controls or perturbations")
    for key in ("vision", "action"):
        path = Path(prepared[f"{key}_path"])
        if sha256(path) != prepared[f"{key}_sha256"]:
            raise ValueError(f"Prepared {key} changed after planning: {path}")
    actions = json.loads(Path(prepared["action_path"]).read_text())
    array = np.asarray(actions, dtype=np.float64)
    chunks = prepared.get("chunks", 1)
    if type(chunks) is not int or chunks < 1:
        raise ValueError("Prepared chunks must be a positive integer")
    expected = (chunks * profile.dataset.chunk_length, int(profile.model["action_dim"]))
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"Expected finite prepared actions with shape {expected}, got {array.shape}")
    return Path(prepared["vision_path"]), actions


def render_prepared(
    profile, model, run: dict, out_dir: Path, *, seed: int = 0, is_rank0=lambda: True, donor: dict | None = None
) -> Path:
    image, actions = load_inputs(profile, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    initial = out_dir / "initial_obs.png"
    if is_rank0():
        with Image.open(image) as source:
            source.convert("RGB").save(initial)
        record = {
            "name": run["name"],
            "recipe": run["recipe"],
            "mode": run["mode"],
            "host": {
                "source": run["collection"],
                "episode_index": run["host_episode"],
                "t_start": run["start_seconds"],
                "t_inject": run["inject_at_s"],
            },
            "donor": donor,
            "arm": run["arm"],
            "target_phase": run["target_phase"],
            "pre_chunks": 0,
            "post_chunks": run["prepared"].get("chunks", 1),
            "fps": profile.dataset.target_fps,
            "failure_time_s": round(run["inject_at_s"] - run["start_seconds"], 6),
            "caption": run["caption"],
            "phases": [run["mode"]],
            "seed": seed,
            "checkpoint": str(model.checkpoint),
            "domain_name": profile.model["domain_name"],
            "action_encoding": "precomputed",
            "normalization": run["prepared"].get("normalization", "none; frozen export"),
            "config_file": profile.model.get("config_file"),
            "prepared": run["prepared"],
            "perturbation": run.get("perturbation"),
            "provenance": run["provenance"],
        }
        (out_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    chunk_size = profile.dataset.chunk_length
    action_chunks = [actions[i : i + chunk_size] for i in range(0, len(actions), chunk_size)]
    if is_rank0():
        (out_dir / "actions_precomputed.json").write_text(json.dumps(action_chunks, allow_nan=False) + "\n")
    from .rollout import last_frame, stitch

    current, clips = initial, []
    for index, chunk_actions in enumerate(action_chunks):
        chunk_dir = out_dir / f"chunk_{index:03d}"
        clip = model.rollout_chunk(current, chunk_actions, run["caption"], chunk_dir, seed + index)
        clips.append(clip)
        if index + 1 < len(action_chunks):
            current = chunk_dir / "last_frame.png"
            if is_rank0():
                last_frame(clip, current)
    synth = out_dir / "synth.mp4"
    if is_rank0():
        if len(clips) == 1:
            shutil.copyfile(clips[0], synth)
        else:
            stitch(clips, synth, profile.dataset.target_fps)
    return synth
