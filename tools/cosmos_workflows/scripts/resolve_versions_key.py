#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a dotted key in versions.yaml to a string value.

Single-purpose companion to ``resolve_cosmos_image.py``: that tool resolves an
image via ``models/<name>/references/skill_info.yaml``, this one resolves a
direct key path in ``versions.yaml``. Use it in shell scripts and skill
docs so the YAML schema is known in exactly one place.

Examples
--------
    resolve_versions_key.py images.containers.cosmos_framework
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Resolve resources relative to this checkout, with an optional explicit root override.
_BUNDLED_WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_ROOT = Path(os.environ.get("COSMOS_WORKFLOWS_ROOT") or _BUNDLED_WORKFLOW_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "key",
        help="Dotted key path, e.g. 'images.containers.cosmos_framework'.",
    )
    parser.add_argument(
        "--workflow-root",
        type=Path,
        default=DEFAULT_WORKFLOW_ROOT,
        help="Path to the packaged Cosmos workflow bundle (defaults to $COSMOS_WORKFLOWS_ROOT).",
    )
    return parser.parse_args()


def resolve(versions_path: Path, dotted_key: str) -> str:
    """Walk ``dotted_key`` through the YAML and return the leaf string."""
    with versions_path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{versions_path} must contain a YAML object at the top level")

    cursor: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"key '{dotted_key}' not found in {versions_path}")
        cursor = cursor[part]

    if not isinstance(cursor, str) or not cursor.strip():
        raise ValueError(f"key '{dotted_key}' did not resolve to a non-empty string in {versions_path}")
    return cursor.strip()


def main() -> int:
    """Print the resolved value or exit non-zero with a diagnostic."""
    args = parse_args()
    versions_path = args.workflow_root.expanduser() / "versions.yaml"
    if not versions_path.exists():
        print(f"versions.yaml not found at {versions_path}", file=sys.stderr)
        return 2
    try:
        print(resolve(versions_path, args.key))
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
