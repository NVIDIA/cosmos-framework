# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Digest-bound publication contract for served Hugging Face export directories."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.checkpoint.conversion.hf import INDEX_FILE, TRAINER_BACKUP_SUFFIX
from cosmos_framework.posttrain.utils import object_store

PUBLICATION_COMMIT = ".posttrain-hf-export-publication.json"


@dataclass(frozen=True)
class HfPublication:
    """Result of one complete or already-complete HF prefix publication."""

    files: tuple[str, ...]
    total_bytes: int
    written_objects: int


def export_files(local_dir: str, commit_name: str = PUBLICATION_COMMIT) -> list[str]:
    """Regular served files, excluding local trainer backups and an earlier publication commit."""
    return sorted(
        name
        for name in os.listdir(local_dir)
        if os.path.isfile(os.path.join(local_dir, name))
        and not name.endswith(TRAINER_BACKUP_SUFFIX)
        and name != commit_name
    )


def publication_id(name: str, local_dir: str, files: Sequence[str]) -> str:
    """Stable request identity; the shared uploader binds each inventory entry to its content digest."""
    digest = hashlib.sha256(f"hf-export\n{name}\n".encode())
    for relative in files:
        digest.update(f"{relative}\t{os.path.getsize(os.path.join(local_dir, relative))}\n".encode())
    return digest.hexdigest()


def publish_hf_export(
    local_dir: str,
    name: str,
    *,
    bucket: str,
    prefix: str,
    credentials: str,
    workers: int = 8,
    commit_name: str = PUBLICATION_COMMIT,
    overrides: dict[str, Any] | None = None,
) -> HfPublication:
    """Publish one mutable served prefix with digest identity and an explicit uncommitted replacement window."""
    files = export_files(local_dir, commit_name)
    if not files:
        raise ValueError(f"{local_dir}: no HF export files to publish")
    total = sum(os.path.getsize(os.path.join(local_dir, relative)) for relative in files)
    manifest = {"id": publication_id(name, local_dir, files), "kind": "hf-export", "name": name, "prefix": prefix}
    written = object_store.upload_prefix_transactional(
        local_dir,
        bucket,
        prefix,
        credentials=credentials,
        overrides=overrides or {},
        commit_manifest=manifest,
        commit_path=commit_name,
        workers=workers,
        relative_paths=files,
        transfer="single_put",
        digest_all=True,
        publish_last=[INDEX_FILE] if INDEX_FILE in files else [],
        invalidate_before_publish=True,
    )
    return HfPublication(tuple(files), total, written)
