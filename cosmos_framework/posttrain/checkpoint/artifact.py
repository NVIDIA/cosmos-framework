# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Commit-marked, non-resumable checkpoint artifacts."""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, release_page_cache
from cosmos_framework.posttrain.checkpoint.manifest import (
    is_canonical_relative_path,
    is_sha256,
    iteration_name,
    iteration_number,
    parse_json_object,
)
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.utils.object_store import file_sha256

CHECKPOINT_ARTIFACT_MANIFEST = ".posttrain-checkpoint-artifact.json"
CHECKPOINT_ARTIFACT_PROTOCOL = "posttrain.checkpoint-artifact.v1"
ARTIFACT_CONTROL_ROOT = ".posttrain-artifact/generations"
CHECKPOINT_ARTIFACTS_DIR = "checkpoint_artifacts"


def _is_component_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and posixpath.basename(value) == value


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate_artifact_manifest(manifest: Any, *, artifact_location: str) -> dict[str, Any]:
    """Validate one commit-last artifact manifest."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "protocol",
        "iteration",
        "iteration_name",
        "checkpoint_format",
        "checkpoint_components",
        "components",
    }:
        raise RuntimeError(f"unsupported posttrain checkpoint artifact manifest at {artifact_location!r}")
    if manifest.get("protocol") != CHECKPOINT_ARTIFACT_PROTOCOL:
        raise RuntimeError(f"unsupported posttrain checkpoint artifact manifest at {artifact_location!r}")
    name = iteration_name(artifact_location)
    iteration = manifest.get("iteration")
    if (
        not _is_nonnegative_int(iteration)
        or manifest.get("iteration_name") != name
        or iteration != iteration_number(name)
    ):
        raise RuntimeError(f"posttrain checkpoint artifact iteration mismatch at {artifact_location!r}")
    if manifest.get("checkpoint_format") not in ("consolidated", "dcp"):
        raise RuntimeError(f"posttrain checkpoint artifact has an invalid format at {artifact_location!r}")
    try:
        plan = CheckpointComponentPlan.from_value(manifest.get("checkpoint_components"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"posttrain checkpoint artifact has invalid logical components at {artifact_location!r}"
        ) from error
    if plan.is_full or list(plan.names) != manifest.get("checkpoint_components"):
        raise RuntimeError(
            f"posttrain checkpoint artifact must contain a canonical partial plan at {artifact_location!r}"
        )

    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise RuntimeError(f"posttrain checkpoint artifact has no physical components at {artifact_location!r}")
    for component, descriptor in components.items():
        if not _is_component_name(component) or not isinstance(descriptor, dict):
            raise RuntimeError(f"invalid checkpoint artifact component {component!r} at {artifact_location!r}")
        kind = descriptor.get("kind")
        path = descriptor.get("path")
        if kind == "dcp":
            valid = set(descriptor) == {"kind", "path"} and path == component
        else:
            valid = (
                kind == "file"
                and set(descriptor) in ({"kind", "path", "size"}, {"kind", "path", "size", "sha256"})
                and is_canonical_relative_path(path)
                and _is_nonnegative_int(descriptor.get("size"))
                and ("sha256" not in descriptor or is_sha256(descriptor.get("sha256")))
            )
        if not valid:
            raise RuntimeError(f"invalid checkpoint artifact descriptor for {component!r} at {artifact_location!r}")
    return dict(manifest)


def build_artifact_manifest(
    artifact_location: str,
    checkpoint_format: str,
    checkpoint_components: tuple[str, ...],
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate one canonical artifact manifest."""
    name = iteration_name(artifact_location)
    manifest: dict[str, Any] = {
        "protocol": CHECKPOINT_ARTIFACT_PROTOCOL,
        "iteration": iteration_number(name),
        "iteration_name": name,
        "checkpoint_format": checkpoint_format,
        "checkpoint_components": list(checkpoint_components),
        "components": {component: dict(descriptor) for component, descriptor in components.items()},
    }
    return validate_artifact_manifest(manifest, artifact_location=artifact_location)


def local_artifact_location(iter_hint: str) -> str:
    """Derive the sibling local artifact location for one checkpoint iteration hint."""
    name = iteration_name(iter_hint)
    checkpoint_root = os.path.dirname(os.path.abspath(iter_hint))
    return os.path.join(os.path.dirname(checkpoint_root), CHECKPOINT_ARTIFACTS_DIR, name)


def committed_artifact_iterations(run_dir: str) -> list[str]:
    """Names of the committed ``checkpoint_artifacts/iter_*`` entries under one run-attempt directory.

    Presence of the commit-last manifest is the test, not the payload: a pruned artifact whose ``model.pt`` left
    for the archive is still committed, and a restart into its directory would still collide at that iteration.
    A directory without the manifest is an interrupted save and does not count.
    """
    root = os.path.join(run_dir, CHECKPOINT_ARTIFACTS_DIR)
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if name.startswith("iter_") and os.path.isfile(os.path.join(root, name, CHECKPOINT_ARTIFACT_MANIFEST))
    )


def read_local_artifact_manifest(artifact_location: str) -> dict[str, Any]:
    """Read and validate one committed local artifact manifest."""
    manifest_path = os.path.join(artifact_location, CHECKPOINT_ARTIFACT_MANIFEST)
    with open(manifest_path, "rb") as handle:
        return validate_artifact_manifest(
            parse_json_object(handle.read(), context=f"checkpoint artifact manifest at {artifact_location!r}"),
            artifact_location=artifact_location,
        )


def local_artifact_component_descriptor(
    artifact_location: str,
    name: str,
) -> tuple[bool, Mapping[str, Any] | None]:
    """Return whether a local path is an artifact and its optional component descriptor."""
    manifest_path = os.path.join(artifact_location, CHECKPOINT_ARTIFACT_MANIFEST)
    if not os.path.isfile(manifest_path):
        return False, None
    manifest = read_local_artifact_manifest(artifact_location)
    return True, manifest["components"].get(name)


def commit_local_artifact(
    artifact_location: str,
    checkpoint_format: str,
    checkpoint_components: tuple[str, ...],
    dcp_component_names: list[str],
    file_components: Mapping[str, dict[str, Any] | None],
) -> None:
    """Write local file components and expose the artifact manifest last."""
    manifest_path = os.path.join(artifact_location, CHECKPOINT_ARTIFACT_MANIFEST)
    if os.path.exists(manifest_path):
        raise FileExistsError(f"refusing to replace committed checkpoint artifact: {artifact_location!r}")
    os.makedirs(artifact_location, exist_ok=True)
    components: dict[str, dict[str, Any]] = {}
    for name in sorted(dcp_component_names):
        if not _is_component_name(name):
            raise ValueError(f"DCP component name must be one path segment, got {name!r}")
        component_path = os.path.join(artifact_location, name)
        from torch.distributed.checkpoint import FileSystemReader

        FileSystemReader(component_path).read_metadata()
        components[name] = {"kind": "dcp", "path": name}
    import torch

    for name, state in file_components.items():
        if state is None:
            continue
        if not _is_component_name(name):
            raise ValueError(f"checkpoint artifact component name must be one path segment, got {name!r}")
        relative = f"{name}.pt"
        path = os.path.join(artifact_location, relative)
        torch.save(state, path)
        # Bind the file just written into the native manifest. The shared archive publisher re-hashes
        # it against this descriptor before reserving the destination, rejecting changed local bytes.
        components[name] = {
            "kind": "file",
            "path": relative,
            "size": os.path.getsize(path),
            "sha256": file_sha256(path),
        }
        # The write left the whole payload dirty in the page cache and the hash re-read it; flush and
        # drop it so a model-sized artifact does not sit in the job cgroup next to the live trainers.
        release_page_cache(path, sync=True)
    if not components:
        raise RuntimeError("checkpoint artifact contains no physical components")
    manifest = build_artifact_manifest(
        artifact_location,
        checkpoint_format,
        checkpoint_components,
        components,
    )
    atomic_write_json(manifest_path, manifest)
