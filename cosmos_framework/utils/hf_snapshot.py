# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Content identity shared by training artifacts and unadapted deployment.

The snapshot bytes and canonical digest encoding retain the adapter-v1 contract.
This module has no trainer, Torch or optional model dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
)
SIDECAR_DIRS = ("additional_chat_templates",)


def file_sha256(path: str) -> str:
    """Stream a snapshot file without materializing its weight bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def snapshot_identity(directory: str, *, weights: bool) -> dict[str, Any]:
    """Hash actual bytes, including every weight shard; same-size substitutions fail.

    Local immutable HF snapshots are the first supported identity contract. A Hub
    branch name or object-store URI must be staged before training/export.
    """
    root = Path(directory)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(
            f"adapter identity requires a complete local HF snapshot: {directory!r}; stage a pinned revision first"
        )
    names = {"config.json"} | {n for n in SIDECAR_FILES if (root / n).is_file()}
    for directory_name in SIDECAR_DIRS:
        if (root / directory_name).is_dir():
            names.update(str(p.relative_to(root)) for p in (root / directory_name).rglob("*") if p.is_file())
    if weights:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            names.add(index.name)
            shards = set(json.loads(index.read_text())["weight_map"].values())
        else:
            shards = {"model.safetensors"}
        for name in shards:
            if Path(name).is_absolute() or ".." in Path(name).parts or not (root / name).is_file():
                raise ValueError(f"base snapshot has a missing/invalid weight shard: {name!r}")
        names.update(shards)
    files = {name: file_sha256(str(root / name)) for name in sorted(names)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}
