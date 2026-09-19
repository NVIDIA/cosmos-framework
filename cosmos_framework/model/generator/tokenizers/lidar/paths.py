# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical S3 / local asset paths for the LiDAR tokenizer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# =============================================================================
# Published asset locations
# =============================================================================


DEFAULT_CREDENTIAL_PATH = "credentials/gcp_checkpoint.secret"

# imaginaire4 repo root: .../tokenizers/lidar/paths.py -> parents[5]
_REPO_ROOT = Path(__file__).resolve().parents[5]


# =============================================================================
# Path / credential helpers
# =============================================================================


def is_remote_uri(path: str | Path) -> bool:
    """Return True when ``path`` is an object-store URI (e.g. ``s3://...``)."""
    return "://" in str(path)


def resolve_credential_path(credential_path: str | None = None) -> str | None:
    """Resolve the object-store credential file, honoring env overrides.

    Lookup order:
      1. ``LIDAR_TOKENIZER_CREDENTIAL_PATH``
      2. explicit ``credential_path`` argument
      3. ``DEFAULT_CREDENTIAL_PATH`` (repo-relative or cwd-relative)
    """
    path = os.environ.get("LIDAR_TOKENIZER_CREDENTIAL_PATH") or credential_path or DEFAULT_CREDENTIAL_PATH
    resolved = Path(path).expanduser()
    if resolved.is_file():
        return str(resolved)

    repo_relative = _REPO_ROOT / resolved
    return str(repo_relative) if repo_relative.is_file() else None


def s3_backend_args(credential_path: str | None = None) -> dict[str, Any] | None:
    """Build ``easy_io`` backend args for GCS/S3 LiDAR assets."""
    resolved = resolve_credential_path(credential_path)
    if resolved is None:
        return None

    return {
        "backend": "s3",
        "path_mapping": None,
        "s3_credential_path": resolved,
    }
