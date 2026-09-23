# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the cosmos-reason-v1.0 annotation format."""

from pathlib import Path
from typing import List


def find_datasets(root: Path) -> List[Path]:
    """Recursively find all cosmos-reason dataset roots (those containing meta.json)."""
    if (root / "meta.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(ds for child in sorted(root.iterdir()) if child.is_dir() for ds in find_datasets(child))
