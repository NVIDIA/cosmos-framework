# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cosmos_framework.data.generator.action.embodiment_c_fk import convert_gripper_state_to_open_fraction

Gripper = Literal["omnipicker", "robotiq"]


_ASSET_PATH: Path = Path(__file__).parent / "manifests/agibot_episode_grippers.json"


GRIPPER_MANIFEST_SHA256: str = "92cc8e371c2ba5ae6d76834805aba393dd3a69193487b6dc44181dc790675bf1"


_LEGACY_DATASET_ALIASES: dict[str, str] = {
    "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-custom": ("embodiment_c__EmbodimentCWorld_20260208_agibot-custom"),
    "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-offshelf": (
        "embodiment_c__EmbodimentCWorld_20260208_agibot-offshelf"
    ),
    "agibot_gear_ext__AgiBotWorld-GEAR_20260208_agibot-custom-ext": (
        "embodiment_c_ext__EmbodimentCWorld_20260208_agibot-custom-ext"
    ),
}


_LEGACY_EPISODE_SUFFIXES: dict[str, str] = {
    "embodiment_c__EmbodimentCWorld_20260208_agibot-custom": "AgiBotWorld-GEAR_20260208/agibot-custom",
    "embodiment_c__EmbodimentCWorld_20260208_agibot-offshelf": "AgiBotWorld-GEAR_20260208/agibot-offshelf",
    "embodiment_c_ext__EmbodimentCWorld_20260208_agibot-custom-ext": ("AgiBotWorld-GEAR_20260208/agibot-custom-ext"),
}


GRIPPER_CALIBRATION_VERSION: str = "agibot-actuator-close-v1"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    payload = _ASSET_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != GRIPPER_MANIFEST_SHA256:
        raise ValueError("AgiBot gripper manifest checksum mismatch.")
    manifest = json.loads(payload)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported AgiBot gripper manifest version.")
    return manifest


@lru_cache(maxsize=1)
def _episode_grippers() -> dict[str, dict[str, dict[int, Gripper]]]:
    """Cache compact source-suffix/shard/episode hardware labels once per worker."""
    manifest = _manifest()
    result: dict[str, dict[str, dict[int, Gripper]]] = {}
    for dataset_name, dataset in manifest["datasets"].items():
        shards: dict[str, dict[int, Gripper]] = {}
        for shard, hardware_groups in dataset["shards"].items():
            episodes: dict[int, Gripper] = {}
            for hardware, indices in hardware_groups.items():
                if hardware not in ("omnipicker", "robotiq"):
                    raise ValueError(f"Unknown AgiBot gripper hardware {hardware!r}.")
                for index in indices:
                    if not isinstance(index, int) or index < 0 or index in episodes:
                        raise ValueError(f"Invalid or conflicting gripper episode {shard}/episode_{index}.")
                    episodes[index] = hardware
            shards[shard] = episodes
        suffixes = {dataset["source_episode_suffix"].strip("/")}
        if dataset_name in _LEGACY_EPISODE_SUFFIXES:
            suffixes.add(_LEGACY_EPISODE_SUFFIXES[dataset_name])
        for suffix in suffixes:
            if suffix in result:
                raise ValueError(f"Duplicate AgiBot gripper source suffix {suffix!r}.")
            result[suffix] = shards
    return result


def action_episode_gripper(source_dataset: str, shard_id: str, episode_index: int) -> Gripper:
    """Resolve hardware from a physical Action episode without guessing labels."""
    source_dataset = _LEGACY_DATASET_ALIASES.get(source_dataset, source_dataset)
    dataset = _manifest()["datasets"].get(source_dataset)
    if dataset is None:
        raise ValueError(f"No audited AgiBot gripper source {source_dataset!r}.")
    suffix = dataset["source_episode_suffix"].strip("/")
    hardware = _episode_grippers()[suffix].get(shard_id, {}).get(episode_index)
    if hardware is None:
        raise ValueError(f"No audited AgiBot gripper label for {source_dataset}/{shard_id}/episode_{episode_index}.")
    return hardware


def episode_gripper(source_sample_id: str) -> Gripper:
    """Look up the original episode, never infer hardware from a clip's range.

    A partially closed OmniPicker clip can lie entirely within [35,120], so
    observed extrema are not a safe fallback for missing hardware labels.
    """
    episode_path, separator, _window = source_sample_id.rpartition("/")
    if separator:
        normalized = episode_path.strip("/")
        for suffix, shards in _episode_grippers().items():
            marker = f"{suffix}/"
            if normalized.startswith(marker):
                relative = normalized[len(marker) :]
            else:
                embedded_marker = f"/{marker}"
                marker_index = normalized.find(embedded_marker)
                if marker_index < 0:
                    continue
                relative = normalized[marker_index + len(embedded_marker) :]
            shard, separator, episode = relative.rpartition("/episode_")
            if not separator or not episode.isdecimal():
                continue
            hardware = shards.get(shard, {}).get(int(episode))
            if hardware is not None:
                return hardware
    raise ValueError(f"No audited AgiBot gripper label for source_sample_id={source_sample_id!r}.")


def calibrated_gripper_fields(
    fields: dict[str, np.ndarray],  # each value: [T+1,D]
    *,
    hardware: Gripper,
    state_layout: str,
    state_width: int | None,
) -> dict[str, np.ndarray]:  # each value: [T+1,D]
    """Copy only Robotiq's two observed scalars; preserve every other channel.

    Saturation handles observed endpoint jitter (e.g. Beta 34.8 or 120.73),
    not a change to the shared pose/unified normalizers. Source values are
    independently audited before export. The parent's finite/upper-range checks
    are retained, and negative source values are rejected. Observation zero is
    included so idle detection and the camera-mask gripper guard use the same
    calibration.
    """
    if hardware == "omnipicker":
        return fields
    if hardware != "robotiq":
        raise ValueError(f"Unsupported AgiBot gripper hardware {hardware!r}.")
    if state_layout == "beta":
        key, width, indices = "state_end_effector", 20, (0, 1)
    elif state_layout == "gear" and state_width in (32, 94):
        key, width = "state", state_width
        indices = (0, 1) if width == 94 else (14, 15)
    else:
        raise ValueError(f"Unsupported AgiBot gripper state layout {state_layout!r}/{state_width}.")
    values = np.asarray(fields[key])  # [T+1,D]
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Expected AgiBot {key} shape [T+1,{width}], got {values.shape}.")
    grippers = values[:, indices]  # [T+1,2]
    # Retain the parent's existing finite/range validation, including its
    # allowed closed-state overshoot. Calibration must not hide corrupt units.
    convert_gripper_state_to_open_fraction(grippers)  # [T+1,2]
    if not np.isfinite(grippers).all() or (grippers < 0).any():
        raise ValueError("Robotiq calibration requires finite, nonnegative actuator-close observations.")
    adjusted = values.astype(np.float32, copy=True)  # [T+1,D]
    adjusted[:, indices] = np.clip((grippers - 35.0) * (120.0 / 85.0), 0.0, 120.0)  # [T+1,2]
    return {**fields, key: adjusted}
