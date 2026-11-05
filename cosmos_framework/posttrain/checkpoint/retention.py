# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint retention selection and verified local payload release."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cosmos_framework.posttrain.checkpoint.archive.marker import marker_covers
from cosmos_framework.posttrain.checkpoint.archive.reader import (
    ArchiveNotVerifiedError,
    bound_marker,
    remote_inventory_status,
    unverified_objects,
)
from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST, read_local_artifact_manifest
from cosmos_framework.posttrain.checkpoint.io import latest_dir
from cosmos_framework.posttrain.checkpoint.manifest import iteration_name
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.log import logger

ARTIFACT_ROOT_NAME = "checkpoint_artifacts"


def _manifest_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, CHECKPOINT_ARTIFACT_MANIFEST)


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def prune_local_iterations(ckpt_root: str, keep_last: int, *, require_uploaded: bool = False) -> None:
    """Delete old iteration directories without removing latest or the only archive copy."""
    import shutil

    if _rank() != 0 or keep_last <= 0 or not os.path.isdir(ckpt_root):
        return
    iterations = sorted(name for name in os.listdir(ckpt_root) if name.startswith("iter_"))
    latest = latest_dir(ckpt_root)
    latest_name = os.path.basename(latest) if latest else None
    for name in iterations[:-keep_last]:
        if name == latest_name:
            continue
        path = os.path.join(ckpt_root, name)
        if require_uploaded and not os.path.isfile(os.path.join(path, ".uploaded")):
            continue
        shutil.rmtree(path)


def keep_newest_iterations(keep_last: int) -> Callable[[list[str]], set[str]]:
    """Return a retention policy keeping the newest optimizer components."""

    def _keep(iter_names: list[str]) -> set[str]:
        return set(iter_names[-keep_last:]) if keep_last > 0 else set(iter_names)

    return _keep


def prune_local_optimizer_state(
    ckpt_root: str,
    keep_last: int,
    keep_fn: Callable[[list[str]], set[str]] | None = None,
    *,
    require_uploaded: bool = False,
    optimizer_subdirs: tuple[str, ...] = ("",),
) -> None:
    """Remove root-level or named-role optimizer components outside the retained set."""
    import shutil

    if _rank() != 0 or not os.path.isdir(ckpt_root):
        return
    if keep_fn is None:
        if keep_last <= 0:
            return
        keep_fn = keep_newest_iterations(keep_last)
    iterations = sorted(name for name in os.listdir(ckpt_root) if name.startswith("iter_"))
    keep = set(keep_fn(list(iterations)))
    latest = latest_dir(ckpt_root)
    latest_name = os.path.basename(latest) if latest else None
    for name in iterations:
        # The authoritative resume target is protected even from a custom policy and from a newer
        # DCP directory appearing while older async archive/retention work is finishing.
        if name in keep or name == latest_name:
            continue
        if require_uploaded and not os.path.isfile(os.path.join(ckpt_root, name, ".uploaded")):
            continue
        for subdir in optimizer_subdirs:
            component_root = os.path.join(ckpt_root, name, subdir) if subdir else os.path.join(ckpt_root, name)
            consolidated = os.path.join(component_root, "optim.pt")
            if os.path.isfile(consolidated):
                os.remove(consolidated)
            dcp_dir = os.path.join(component_root, "optim")
            if os.path.isdir(dcp_dir):
                shutil.rmtree(dcp_dir)


def _payload_paths(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(str(descriptor["path"]) for descriptor in manifest["components"].values())


def _remove_payloads(artifact_dir: str, manifest: Mapping[str, Any]) -> list[str]:
    """Unlink the manifest's physical components; the manifest and marker stay as the local record."""
    removed: list[str] = []
    for relative in _payload_paths(manifest):
        path = os.path.join(artifact_dir, *relative.split("/"))
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(relative)
        elif os.path.isfile(path):
            os.remove(path)
            removed.append(relative)
    return removed


# ---------------------------------------------------------------- prune -------------------------------------------------
def _prunable_artifact_dir(path: str) -> str:
    path = os.path.abspath(path)
    artifact_dir = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isfile(_manifest_path(artifact_dir)):
        raise ArchiveNotVerifiedError("not a committed local checkpoint artifact")
    if os.path.basename(os.path.dirname(artifact_dir)) != ARTIFACT_ROOT_NAME:
        raise ArchiveNotVerifiedError(
            f"outside a {ARTIFACT_ROOT_NAME}/ root; the trainer owns full-checkpoint retention"
        )
    iteration_name(artifact_dir)
    return artifact_dir


def _local_payload_files(artifact_dir: str, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return ``{relative: absolute}`` for every payload file currently on disk."""
    files: dict[str, str] = {}
    for relative in _payload_paths(manifest):
        path = os.path.join(artifact_dir, *relative.split("/"))
        if os.path.isfile(path):
            files[relative] = path
        elif os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for filename in names:
                    shard = os.path.join(root, filename)
                    files[os.path.relpath(shard, artifact_dir).replace(os.sep, "/")] = shard
    return files


def prune_verified_artifacts(paths: Sequence[str], *, credentials: str | None = None) -> dict[str, Any]:
    """Release local artifact payloads whose archived copies verify live; keep everything else.

    Only ``checkpoint_artifacts/<iter_*>`` directories (or payload files inside them) are
    considered, so this never competes with the trainer's full-checkpoint retention. Payloads are
    unlinked only when the marker binds the destination these credentials address and covers the
    directory, records a digest for every local payload file that the local bytes still match, and
    every recorded object is live with that size and digest. Absent or different objects,
    digest-less entries, legacy markers, and invalid manifests keep the copy with a reason. A store
    that reports this artifact's digest for an object of another size is corrupt: such artifacts
    are kept and ``RuntimeError`` is raised after every path has been examined. Returns
    ``{"removed": {path: [relative...]}, "kept": {path: reason}, "freed_bytes": int}``.
    """
    credentials = credentials or object_store.DEFAULT_CREDENTIALS
    client = object_store.build_s3_client(credentials)
    report: dict[str, Any] = {"removed": {}, "kept": {}, "freed_bytes": 0}
    corrupt: list[str] = []
    for path in paths:
        try:
            artifact_dir = _prunable_artifact_dir(path)
            manifest = read_local_artifact_manifest(artifact_dir)
            marker, bucket, prefix = bound_marker(artifact_dir, credentials)
            if not marker_covers(artifact_dir, marker.destination):
                raise ArchiveNotVerifiedError("archive marker does not cover the directory's current contents")
            local_files = _local_payload_files(artifact_dir, manifest)
            if not local_files:
                raise ArchiveNotVerifiedError("no local payload to release")
            for relative, local in sorted(local_files.items()):
                entry = marker.files.get(relative)
                if entry is None or entry["sha256"] is None:
                    raise ArchiveNotVerifiedError(f"{relative}: no archived digest")
                if os.path.getsize(local) != entry["size"] or object_store.file_sha256(local) != entry["sha256"]:
                    raise ArchiveNotVerifiedError(f"{relative}: local bytes differ from the archived copy")
            statuses = remote_inventory_status(client, bucket, prefix, marker.files)
            if "corrupt" in statuses.values():
                corrupt.append(f"s3://{bucket}/{prefix}")
                raise ArchiveNotVerifiedError(
                    "store reports this artifact's digest for an object of another size (corrupt store)"
                )
            problems = unverified_objects(statuses)
            if problems:
                raise ArchiveNotVerifiedError("; ".join(problems))
        except (ArchiveNotVerifiedError, OSError, RuntimeError, ValueError) as reason:
            report["kept"][path] = str(reason)
            logger.info(f"prune keep {path}: {reason}")
            continue
        freed = sum(os.path.getsize(local) for local in local_files.values())
        report["removed"][path] = _remove_payloads(artifact_dir, manifest)
        report["freed_bytes"] += freed
        logger.info(
            f"prune removed {report['removed'][path]} under {artifact_dir} ({freed / 2**30:.1f} GiB, verified live)"
        )
    if corrupt:
        raise RuntimeError(
            f"corrupt object store: {', '.join(corrupt)} report this artifact's digest for other-sized objects"
        )
    return report
