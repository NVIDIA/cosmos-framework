# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, cast

ABCGripperSubset = Literal["realsense_crank", "realsense_flexible", "realsense_linear", "zed"]


ABC_GRIPPER_MANIFEST_PATH: Path = Path(__file__).parent / "manifests" / "abc_130k_grippers.json"


ABC_GRIPPER_MANIFEST_SCHEMA_VERSION: int = 1


ABC_GRIPPER_DATASET_VERSION: str = "ABC-130k_lerobot_20260725"


ABC_GRIPPER_SUBSETS: tuple[ABCGripperSubset, ...] = (
    "realsense_crank",
    "realsense_flexible",
    "realsense_linear",
    "zed",
)


ABC_GRIPPER_SUBSET_SPECS: Mapping[ABCGripperSubset, tuple[str, float]] = {
    "realsense_crank": ("crank", 0.071),
    "realsense_flexible": ("flexible", 0.096),
    "realsense_linear": ("linear", 0.096),
    "zed": ("linear", 0.096),
}


@dataclass(frozen=True, slots=True)
class ABCGripperBinding:
    """One reviewed gripper calibration assigned to an ABC episode."""

    subset: ABCGripperSubset
    gripper_type: str
    fully_open_width_m: float


@dataclass(frozen=True, slots=True)
class ABCGripperManifest:
    """Validated lookup over one checked-in ABC manifest."""

    path: Path
    sha256: str
    source_metadata_sha256: str
    source_metadata_subset_episode_counts: Mapping[ABCGripperSubset, int]
    unresolved_source_metadata_episodes: tuple[tuple[ABCGripperSubset, str, str], ...]
    bindings: Mapping[tuple[str, int], ABCGripperBinding]
    source_episode_counts: Mapping[str, int]
    unmapped_episode_counts: Mapping[str, int]
    subset_episode_counts: Mapping[ABCGripperSubset, int]

    def binding(self, source_dataset: str, episode_index: int) -> ABCGripperBinding | None:
        """Return the binding for one physical LeRobot episode, if classified."""
        return self.bindings.get((source_dataset, int(episode_index)))


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return one JSON object or fail with a field-specific message."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"ABC gripper manifest field {field_name!r} must be a JSON object with string keys.")
    return cast(dict[str, Any], value)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    """Return one non-negative integer manifest field."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ABC gripper manifest field {field_name!r} must be a non-negative integer.")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    """Return one lowercase SHA-256 hex digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ABC gripper manifest field {field_name!r} must be a lowercase SHA-256 digest.")
    return value


def _validate_episode_indices(value: Any, field_name: str, source_episode_count: int) -> tuple[int, ...]:
    """Return one strictly increasing, in-range episode-index list."""
    if not isinstance(value, list):
        raise ValueError(f"ABC gripper manifest field {field_name!r} must be a JSON list.")
    indices = tuple(_require_nonnegative_int(index, field_name) for index in value)
    if tuple(sorted(set(indices))) != indices:
        raise ValueError(f"ABC gripper manifest field {field_name!r} must be strictly increasing and unique.")
    if indices and indices[-1] >= source_episode_count:
        raise ValueError(
            f"ABC gripper manifest field {field_name!r} contains episode {indices[-1]}, "
            f"outside source episode count {source_episode_count}."
        )
    return indices


@lru_cache(maxsize=None)
def load_abc_gripper_manifest(path: str | Path = ABC_GRIPPER_MANIFEST_PATH) -> ABCGripperManifest:
    """Load and validate the checked-in ABC gripper manifest once per process."""
    resolved_path = Path(path).expanduser().resolve()
    payload = resolved_path.read_bytes()
    raw = _require_mapping(json.loads(payload), "root")
    if raw.get("schema_version") != ABC_GRIPPER_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "ABC gripper manifest schema_version must be "
            f"{ABC_GRIPPER_MANIFEST_SCHEMA_VERSION}, got {raw.get('schema_version')!r}."
        )
    if raw.get("dataset_version") != ABC_GRIPPER_DATASET_VERSION:
        raise ValueError(
            f"ABC gripper manifest dataset_version must be {ABC_GRIPPER_DATASET_VERSION!r}, "
            f"got {raw.get('dataset_version')!r}."
        )
    source_metadata_sha256 = _require_sha256(raw.get("source_metadata_sha256"), "source_metadata_sha256")
    raw_source_metadata_subset_counts = _require_mapping(
        raw.get("source_metadata_subset_episode_counts"),
        "source_metadata_subset_episode_counts",
    )
    if set(raw_source_metadata_subset_counts) != set(ABC_GRIPPER_SUBSETS):
        raise ValueError(
            f"ABC gripper manifest source_metadata_subset_episode_counts must name {list(ABC_GRIPPER_SUBSETS)!r}."
        )
    source_metadata_subset_episode_counts = {
        subset: _require_nonnegative_int(
            raw_source_metadata_subset_counts[subset],
            f"source_metadata_subset_episode_counts.{subset}",
        )
        for subset in ABC_GRIPPER_SUBSETS
    }
    raw_unresolved = raw.get("unresolved_source_metadata_episodes")
    if not isinstance(raw_unresolved, list):
        raise ValueError("ABC gripper manifest field 'unresolved_source_metadata_episodes' must be a JSON list.")
    unresolved_source_metadata_episodes: list[tuple[ABCGripperSubset, str, str]] = []
    for index, raw_episode in enumerate(raw_unresolved):
        record = _require_mapping(raw_episode, f"unresolved_source_metadata_episodes[{index}]")
        subset = record.get("subset")
        relative_path = record.get("relative_path")
        reason = record.get("reason")
        if subset not in ABC_GRIPPER_SUBSETS:
            raise ValueError(f"ABC gripper manifest unresolved episode {index} names unknown subset {subset!r}.")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"ABC gripper manifest unresolved episode {index} must have a non-empty relative_path.")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"ABC gripper manifest unresolved episode {index} must have a non-empty reason.")
        unresolved_source_metadata_episodes.append((cast(ABCGripperSubset, subset), relative_path, reason))
    unresolved_paths = [relative_path for _, relative_path, _ in unresolved_source_metadata_episodes]
    if unresolved_paths != sorted(set(unresolved_paths)):
        raise ValueError("ABC gripper manifest unresolved source-metadata episodes must be unique and path-sorted.")

    raw_source_counts = _require_mapping(raw.get("source_episode_counts"), "source_episode_counts")
    source_episode_counts = {
        source_dataset: _require_nonnegative_int(count, f"source_episode_counts.{source_dataset}")
        for source_dataset, count in raw_source_counts.items()
    }
    if not source_episode_counts:
        raise ValueError("ABC gripper manifest source_episode_counts must not be empty.")

    raw_unmapped_counts = _require_mapping(raw.get("unmapped_episode_counts"), "unmapped_episode_counts")
    if set(raw_unmapped_counts) != set(source_episode_counts):
        raise ValueError("ABC gripper manifest unmapped_episode_counts must name every source dataset exactly once.")
    unmapped_episode_counts = {
        source_dataset: _require_nonnegative_int(count, f"unmapped_episode_counts.{source_dataset}")
        for source_dataset, count in raw_unmapped_counts.items()
    }

    raw_subsets = _require_mapping(raw.get("subsets"), "subsets")
    if set(raw_subsets) != set(ABC_GRIPPER_SUBSETS):
        raise ValueError(f"ABC gripper manifest subsets must be exactly {list(ABC_GRIPPER_SUBSETS)!r}.")

    mutable_bindings: dict[tuple[str, int], ABCGripperBinding] = {}
    subset_episode_counts: dict[ABCGripperSubset, int] = {}
    mapped_by_source = {source_dataset: 0 for source_dataset in source_episode_counts}
    for subset in ABC_GRIPPER_SUBSETS:
        subset_record = _require_mapping(raw_subsets[subset], f"subsets.{subset}")
        expected_gripper_type, expected_width_m = ABC_GRIPPER_SUBSET_SPECS[subset]
        if subset_record.get("gripper_type") != expected_gripper_type:
            raise ValueError(f"ABC gripper manifest subset {subset!r} must use gripper_type={expected_gripper_type!r}.")
        width_m = subset_record.get("fully_open_width_m")
        if isinstance(width_m, bool) or not isinstance(width_m, (int, float)):
            raise ValueError(f"ABC gripper manifest subset {subset!r} fully_open_width_m must be numeric.")
        width_m = float(width_m)
        if not math.isclose(width_m, expected_width_m, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"ABC gripper manifest subset {subset!r} must use fully_open_width_m={expected_width_m}.")

        raw_episode_indices = _require_mapping(
            subset_record.get("episode_indices"), f"subsets.{subset}.episode_indices"
        )
        subset_count = 0
        for source_dataset, raw_indices in raw_episode_indices.items():
            if source_dataset not in source_episode_counts:
                raise ValueError(f"ABC gripper manifest subset {subset!r} names unknown source {source_dataset!r}.")
            expected_camera_token = "_zedx_" if subset == "zed" else "_realsense_"
            if expected_camera_token not in source_dataset:
                raise ValueError(
                    f"ABC gripper manifest subset {subset!r} cannot contain source dataset {source_dataset!r}."
                )
            indices = _validate_episode_indices(
                raw_indices,
                f"subsets.{subset}.episode_indices.{source_dataset}",
                source_episode_counts[source_dataset],
            )
            binding = ABCGripperBinding(subset, expected_gripper_type, width_m)
            for episode_index in indices:
                key = (source_dataset, episode_index)
                if key in mutable_bindings:
                    raise ValueError(f"ABC gripper manifest episode {key!r} is assigned to multiple subsets.")
                mutable_bindings[key] = binding
            subset_count += len(indices)
            mapped_by_source[source_dataset] += len(indices)
        declared_subset_count = _require_nonnegative_int(
            subset_record.get("episode_count"),
            f"subsets.{subset}.episode_count",
        )
        if subset_count != declared_subset_count:
            raise ValueError(
                f"ABC gripper manifest subset {subset!r} declares {declared_subset_count} episodes, found {subset_count}."
            )
        subset_episode_counts[subset] = subset_count

    for source_dataset, source_count in source_episode_counts.items():
        computed_unmapped = source_count - mapped_by_source[source_dataset]
        declared_unmapped = unmapped_episode_counts[source_dataset]
        if computed_unmapped != declared_unmapped:
            raise ValueError(
                f"ABC gripper manifest source {source_dataset!r} declares {declared_unmapped} unmapped episodes, "
                f"computed {computed_unmapped}."
            )

    declared_mapped_count = _require_nonnegative_int(raw.get("mapped_episode_count"), "mapped_episode_count")
    if declared_mapped_count != len(mutable_bindings):
        raise ValueError(
            f"ABC gripper manifest declares {declared_mapped_count} mapped episodes, found {len(mutable_bindings)}."
        )
    unresolved_by_subset = Counter(subset for subset, _, _ in unresolved_source_metadata_episodes)
    for subset in ABC_GRIPPER_SUBSETS:
        classified_count = subset_episode_counts[subset] + unresolved_by_subset[subset]
        source_metadata_count = source_metadata_subset_episode_counts[subset]
        if classified_count != source_metadata_count:
            raise ValueError(
                f"ABC gripper manifest subset {subset!r} maps or skips {classified_count} source-metadata episodes, "
                f"expected {source_metadata_count}."
            )
    return ABCGripperManifest(
        path=resolved_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_metadata_sha256=source_metadata_sha256,
        source_metadata_subset_episode_counts=source_metadata_subset_episode_counts,
        unresolved_source_metadata_episodes=tuple(unresolved_source_metadata_episodes),
        bindings=mutable_bindings,
        source_episode_counts=source_episode_counts,
        unmapped_episode_counts=unmapped_episode_counts,
        subset_episode_counts=subset_episode_counts,
    )
