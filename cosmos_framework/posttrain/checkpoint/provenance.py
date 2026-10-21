# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint provenance and atomic control-journal updates."""

from __future__ import annotations

import os
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import (
    atomic_write_json,
    component_kind,
    fsync_directory,
    is_complete_iter_dir,
)

EXTERNAL_COPY_PROVENANCE_FILE = ".external_copy_provenance.json"
CheckpointProvenance = dict[str, int | str]


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def checkpoint_inventory_signature(
    checkpoint_dir: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Hash checkpoint structure plus small-file contents without reading giant tensors."""
    normalized_overrides = {relative.replace(os.sep, "/"): path for relative, path in (overrides or {}).items()}
    digest = sha256()
    for directory, dirnames, filenames in os.walk(checkpoint_dir):
        dirnames.sort()
        filenames.sort()
        relative_directory = os.path.relpath(directory, checkpoint_dir)
        digest.update(f"dir\0{relative_directory}\0".encode("utf-8", errors="surrogateescape"))
        for filename in filenames:
            relative = filename if relative_directory == "." else os.path.join(relative_directory, filename)
            if (
                relative
                in (
                    ".uploaded",
                    ".uploaded.tmp",
                    EXTERNAL_COPY_PROVENANCE_FILE,
                    EXTERNAL_COPY_PROVENANCE_FILE + ".tmp",
                )
                or relative.startswith(".")
                and relative.endswith(".pt.tmp")
            ):
                continue
            path = normalized_overrides.get(relative.replace(os.sep, "/"), os.path.join(directory, filename))
            stat = os.stat(path, follow_symlinks=False)
            kind = "symlink" if os.path.islink(path) else "file"
            if kind == "file" and stat.st_size <= 4 * 1024 * 1024:
                # Small files are content-bound below. Ignoring inode/timestamps makes an atomically
                # renamed temp describe the same logical successor after promotion.
                record = f"{kind}\0{relative}\0{stat.st_size}\0"
            else:
                record = (
                    f"{kind}\0{relative}\0{stat.st_dev}\0{stat.st_ino}\0{stat.st_size}\0"
                    f"{stat.st_mtime_ns}\0{stat.st_ctime_ns}\0"
                )
            digest.update(record.encode("utf-8", errors="surrogateescape"))
            if kind == "symlink":
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif stat.st_size <= 4 * 1024 * 1024:
                with open(path, "rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(source_dir: str) -> CheckpointProvenance:
    """Build a stable, cheap identity for one immutable training checkpoint."""
    source = os.path.realpath(os.path.expanduser(source_dir))
    if not os.path.isdir(source):
        raise FileNotFoundError(f"external pending checkpoint source is not a directory: {source!r}")
    if not is_complete_iter_dir(source) or component_kind(source, "optim") is None:
        raise RuntimeError(
            f"external pending checkpoint source is incomplete: {source!r} "
            "(model, optimizer, and trainer components are required)"
        )
    signature = checkpoint_inventory_signature(source)
    return {"version": 1, "source": source, "signature": signature}


def load_with_stable_provenance(
    source_dir: str,
    load: Callable[[], Any],
    *,
    provenance: Callable[[str], CheckpointProvenance] = checkpoint_provenance,
) -> tuple[Any, CheckpointProvenance]:
    """Load only when the checkpoint identity stays stable across the entire read."""
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

    before = run_rank_converged(
        lambda: provenance(source_dir),
        "checkpoint pre-load provenance capture",
    )
    before = rank_identical(before, "checkpoint pre-load provenance")
    value = load()
    after = run_rank_converged(
        lambda: provenance(source_dir),
        "checkpoint post-load provenance capture",
    )
    after = rank_identical(after, "checkpoint post-load provenance")
    if after != before:
        raise RuntimeError(f"checkpoint changed while it was being loaded: {source_dir!r}")
    return value, before


def read_external_copy_provenance(target_dir: str) -> dict[str, Any] | None:
    """Read a valid external-copy marker, failing closed on malformed state."""
    import json

    marker = os.path.join(target_dir, EXTERNAL_COPY_PROVENANCE_FILE)
    if not os.path.isfile(marker):
        return None
    try:
        with open(marker, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) not in (
        {"version", "source", "signature", "copy_signature"},
        {"version", "source", "signature", "copy_signature", "successor_signatures"},
    ):
        return None
    if payload.get("version") != 1 or not isinstance(payload.get("source"), str):
        return None
    if not isinstance(payload.get("signature"), str) or not isinstance(payload.get("copy_signature"), str):
        return None
    successors = payload.get("successor_signatures", [])
    if not isinstance(successors, list) or any(not isinstance(signature, str) for signature in successors):
        return None
    return payload


def target_matches_external_copy(target_dir: str, provenance: CheckpointProvenance) -> bool:
    """Return whether target is the exact copied source or an authorized journal successor."""
    if not is_complete_iter_dir(target_dir) or component_kind(target_dir, "optim") is None:
        return False
    marker = read_external_copy_provenance(target_dir)
    if marker is None:
        return False
    marker_source = {key: marker[key] for key in ("version", "source", "signature")}
    accepted_signatures = {str(marker["copy_signature"]), *marker.get("successor_signatures", [])}
    return marker_source == provenance and checkpoint_inventory_signature(target_dir) in accepted_signatures


def write_external_copy_provenance(staging_dir: str, provenance: CheckpointProvenance) -> None:
    """Durably record the source and initial copy signature inside staging."""
    marker_payload = {
        **provenance,
        "copy_signature": checkpoint_inventory_signature(staging_dir),
        "successor_signatures": [],
    }
    atomic_write_json(os.path.join(staging_dir, EXTERNAL_COPY_PROVENANCE_FILE), marker_payload)


def replace_component_atomic(iter_dir: str, name: str, state: dict[str, Any]) -> None:
    """Atomically replace one small component, extending external-copy provenance first."""
    import torch

    if not name or os.path.basename(name) != name:
        raise ValueError(f"checkpoint component name must be one path segment, got {name!r}")
    replacement_error: BaseException | None = None
    if _rank() == 0:
        temporary = os.path.join(iter_dir, f".{name}.pt.tmp")
        marker_temporary = os.path.join(iter_dir, EXTERNAL_COPY_PROVENANCE_FILE + ".tmp")
        try:
            from cosmos_framework.posttrain.checkpoint.worker import join_pending_rank0

            join_pending_rank0()
            if not os.path.isdir(iter_dir):
                raise FileNotFoundError(f"cannot replace component in missing checkpoint {iter_dir!r}")
            destination = os.path.join(iter_dir, f"{name}.pt")
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            marker_payload = read_external_copy_provenance(iter_dir)
            if marker_payload is not None:
                current_signature = checkpoint_inventory_signature(iter_dir)
                accepted_signatures = {
                    str(marker_payload["copy_signature"]),
                    *marker_payload.get("successor_signatures", []),
                }
                if current_signature not in accepted_signatures:
                    raise RuntimeError("external-copy provenance does not authorize the checkpoint's current inventory")
                successor_signature = checkpoint_inventory_signature(
                    iter_dir,
                    overrides={f"{name}.pt": temporary},
                )
                if successor_signature not in accepted_signatures:
                    # Marker first is deliberate: a crash before component rename leaves the old
                    # inventory authorized; a crash after it also authorizes the new inventory.
                    successors = [*marker_payload.get("successor_signatures", []), successor_signature]
                    atomic_write_json(
                        os.path.join(iter_dir, EXTERNAL_COPY_PROVENANCE_FILE),
                        {**marker_payload, "successor_signatures": successors},
                    )
            os.replace(temporary, destination)
            fsync_directory(iter_dir)
        except BaseException as error:  # noqa: BLE001 -- every distributed rank must exit together
            replacement_error = error
        finally:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
                if os.path.exists(marker_temporary):
                    os.remove(marker_temporary)
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve cleanup failure too
                if replacement_error is None:
                    replacement_error = cleanup_error
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    raise_if_any_rank_failed(replacement_error, f"atomic checkpoint component replacement ({name})")
