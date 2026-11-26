# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Runtime gate for the upstream source checkouts of docker/posttrain/native_action.

The installer clones each public repository at its pinned commit and checks that the
checkout is clean. At runtime only that commit is compared, read from ``.git`` without
running git; installed source bytes are never hashed.
"""

from __future__ import annotations

from pathlib import Path


def checkout_revision(path: str | Path) -> str | None:
    """Commit checked out in the git work tree that contains ``path``, or None outside one."""
    for parent in Path(path).resolve().parents:
        git = parent / ".git"
        if git.is_dir():
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref: "):
                return head  # The installer's detached checkout.
            ref = git / head.removeprefix("ref: ")
            return ref.read_text().strip() if ref.is_file() else None
    return None


def require_source_revision(module_file: str, revision: str, name: str) -> None:
    """Fail before model loading unless ``module_file`` comes from the pinned source checkout."""
    actual = checkout_revision(module_file)
    if actual != revision:
        raise ValueError(
            f"{name} must be the public source checkout at {revision}, found "
            f"{actual or 'no git checkout'} at {module_file}; install it with "
            "docker/posttrain/native_action/install.py (see docker/posttrain/native_action/README.md)"
        )
