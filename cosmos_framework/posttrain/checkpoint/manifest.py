# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small control manifests for direct-primary DCP checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

POSTTRAIN_MANIFEST = ".posttrain-manifest.json"
MANIFEST_PROTOCOL = "posttrain.checkpoint.v1"
VALIDATION_PUBLICATION_PROTOCOL = "posttrain.validation-publication.v1"
CONTROL_ROOT = ".posttrain/generations"


@dataclass(frozen=True)
class CommittedManifest:
    """One validated control pointer and the exact bytes identifying it."""

    value: dict[str, Any]
    encoded: bytes
    digest: str


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode one canonical JSON object."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def is_generation_id(value: Any) -> bool:
    """Return whether a value is one canonical lowercase ``uuid4().hex`` spelling."""
    return isinstance(value, str) and len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def iteration_name(path: str) -> str:
    """Extract and validate one iteration-directory name from a local path or S3 URI."""
    normalized = path.rstrip("/")
    name = (
        posixpath.basename(urlsplit(normalized).path)
        if normalized.startswith("s3://")
        else os.path.basename(normalized)
    )
    if not name.startswith("iter_") or not name.removeprefix("iter_").isdigit():
        raise ValueError(f"checkpoint path must end in one iter_* segment, got {path!r}")
    return name


def iteration_number(name: str) -> int:
    """Parse a non-negative iteration number from a canonical iteration name."""
    suffix = name.removeprefix("iter_")
    if not suffix.isdigit():
        raise ValueError(f"checkpoint iteration name must end in decimal digits, got {name!r}")
    return int(suffix)


def is_canonical_relative_path(value: Any) -> bool:
    """Return whether a checkpoint-relative object path has one unambiguous spelling."""
    if not isinstance(value, str) or not value or value.startswith("/"):
        return False
    if "\\" in value or "%" in value or "?" in value or "#" in value:
        return False
    return all(segment not in ("", ".", "..") for segment in value.split("/"))


def parse_json_object(payload: bytes, *, context: str) -> dict[str, Any]:
    """Decode one JSON object with a contextual checkpoint error."""
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"malformed {context}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def validate_publication_record(value: Any, iteration: int) -> dict[str, Any]:
    """Validate one v1 validation-publication record."""
    if not isinstance(value, dict) or set(value) != {
        "step",
        "id",
        "archive_required",
        "destination",
        "manifest",
    }:
        raise RuntimeError("checkpoint validation_publication record has an unsupported schema")
    step = value.get("step")
    publication_id = value.get("id")
    destination = value.get("destination")
    if isinstance(step, bool) or not isinstance(step, int) or step != iteration:
        raise RuntimeError("checkpoint validation_publication step does not match its iteration")
    if not is_sha256(publication_id):
        raise RuntimeError("checkpoint validation_publication id is not a canonical SHA-256")
    if not isinstance(value.get("archive_required"), bool):
        raise RuntimeError("checkpoint validation_publication archive_required must be a bool")
    if (
        not isinstance(destination, dict)
        or not destination
        or any(not isinstance(key, str) or not isinstance(item, str) for key, item in destination.items())
    ):
        raise RuntimeError("checkpoint validation_publication destination must contain string fields")
    if value.get("manifest") != f".validation-publication-v1-{publication_id}.json":
        raise RuntimeError("checkpoint validation_publication manifest does not match its id")
    return dict(value)


def dcp_tensor_inventory(metadata: Any, metadata_path: str) -> dict[str, tuple[int, ...] | None]:
    """Return logical DCP state keys and tensor shapes from parsed metadata."""
    state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(state_dict_metadata, dict):
        raise RuntimeError(f"DCP metadata at {metadata_path!r} has no state_dict_metadata mapping")
    inventory: dict[str, tuple[int, ...] | None] = {}
    for key, descriptor in state_dict_metadata.items():
        if not isinstance(key, str):
            raise RuntimeError(f"DCP metadata at {metadata_path!r} has a non-string logical key")
        size = getattr(descriptor, "size", None)
        if size is None:
            inventory[key] = None
            continue
        try:
            shape = tuple(int(dimension) for dimension in size)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"DCP metadata at {metadata_path!r} has an invalid tensor shape for {key!r}") from error
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"DCP metadata at {metadata_path!r} has an invalid tensor shape for {key!r}")
        inventory[key] = shape
    return inventory


def _is_component_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and posixpath.basename(value) == value


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate_manifest(manifest: Any, *, iter_location: str) -> dict[str, Any]:
    """Validate one direct-DCP control pointer without binding it to a storage URI."""
    if not isinstance(manifest, dict) or manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise RuntimeError(f"unsupported posttrain checkpoint manifest at {iter_location!r}")
    name = iteration_name(iter_location)
    iteration = manifest.get("iteration")
    if (
        not _is_nonnegative_int(iteration)
        or manifest.get("iteration_name") != name
        or iteration != iteration_number(name)
    ):
        raise RuntimeError(f"posttrain checkpoint manifest iteration mismatch at {iter_location!r}")
    if manifest.get("checkpoint_format") != "dcp":
        raise RuntimeError(f"posttrain object-store checkpoint is not DCP: {iter_location!r}")
    generation = manifest.get("control_generation")
    if not is_generation_id(generation):
        raise RuntimeError(f"posttrain checkpoint manifest has an invalid control generation: {iter_location!r}")

    bulk = manifest.get("bulk_components")
    if (
        not isinstance(bulk, list)
        or not {"model", "optim"}.issubset(bulk)
        or len(set(bulk)) != len(bulk)
        or any(not _is_component_name(component) for component in bulk)
    ):
        raise RuntimeError(f"posttrain checkpoint manifest has invalid DCP components: {iter_location!r}")

    controls = manifest.get("control_components")
    if not isinstance(controls, dict) or "trainer" not in controls:
        raise RuntimeError(f"posttrain checkpoint manifest requires a trainer component: {iter_location!r}")
    for component, descriptor in controls.items():
        expected_path = f"{CONTROL_ROOT}/{generation}/{component}.pt"
        if (
            not _is_component_name(component)
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "size", "sha256"}
            or descriptor.get("path") != expected_path
            or not _is_nonnegative_int(descriptor.get("size"))
            or not is_sha256(descriptor.get("sha256"))
        ):
            raise RuntimeError(f"invalid control component descriptor for {component!r} in {iter_location!r}")

    publication = manifest.get("validation_publication")
    if publication is not None:
        validate_publication_record(publication, iteration)
    return dict(manifest)


def validate_visibility_commit(
    value: Any,
    *,
    record: Mapping[str, Any],
    active_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one validation visibility commit and its optional active-manifest binding."""
    if (
        not isinstance(value, dict)
        or set(value) != {"protocol", "record", "base_manifest_sha256", "successor_manifest"}
        or value.get("protocol") != VALIDATION_PUBLICATION_PROTOCOL
        or value.get("record") != dict(record)
        or not is_sha256(value.get("base_manifest_sha256"))
        or not isinstance(value.get("successor_manifest"), dict)
    ):
        raise RuntimeError("unsupported validation publication commit")
    if active_manifest is not None and json_bytes(value["successor_manifest"]) != json_bytes(active_manifest):
        raise RuntimeError("checkpoint validation publication commit does not bind the active manifest")
    return dict(value)
