# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the cosmos-video-reasoning-v1.0 annotation format."""

from pathlib import Path
from typing import List, Optional

from cosmos_framework.data.reasoner.formats.utils import FormatError, read_json_object

FORMAT = "cosmos-video-reasoning-v1.0"


def find_datasets(root: Path) -> List[Path]:
    """Recursively find all cosmos-video-reasoning-v1.0 dataset roots.

    A directory qualifies if it contains at least one ``*.json`` whose
    top-level ``format`` is ``"cosmos-video-reasoning-v1.0"``. The validator reads
    each annotation file again later for schema validation — this discovery
    pass is short-circuited (returns on the first matching file) and only
    looks at the top-level ``format`` field, so per-file work here is
    minimal even when the directory contains unrelated JSON.
    """
    if not root.is_dir():
        return []
    for p in root.glob("*.json"):
        try:
            data = read_json_object(p)
        except FormatError:
            continue  # discovery-pass: ignore unparseable / non-object JSON
        if data.get("format") == FORMAT:
            return [root]
    return sorted(ds for child in sorted(root.iterdir()) if child.is_dir() for ds in find_datasets(child))


def resolve_media_path(dataset_path: Path, media_root: Optional[str], item_path: str) -> Path:
    """Resolve ``item_path`` against ``media_root`` per the cosmos-video-reasoning-v1.0 rules.

    - ``media_root`` ``None`` → ``dataset_path / item_path``.
    - Relative ``media_root`` → ``dataset_path / media_root / item_path``.
    - Absolute ``media_root`` → ``media_root / item_path``.
    """
    if media_root is None:
        return dataset_path / item_path
    base = Path(media_root)
    if base.is_absolute():
        return base / item_path
    return dataset_path / base / item_path
