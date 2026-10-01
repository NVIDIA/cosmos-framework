# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Write deterministic source and runtime provenance during an image build."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib
import json
import os
import platform
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    workspace = Path(os.environ.get("PROVENANCE_WORKSPACE", "/workspace")).resolve(strict=True)
    output_dir = Path(os.environ.get("PROVENANCE_OUTPUT_DIR", "/opt/tao")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [workspace / "cosmos_framework", workspace / "pyproject.toml", workspace / "uv.lock", workspace / "Dockerfile"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    entries = [(str(path.relative_to(workspace)), _sha256(path)) for path in sorted(files)]
    manifest_text = "".join(f"{digest}  {name}\n" for name, digest in entries)
    (output_dir / "source-manifest.sha256").write_text(manifest_text, encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "repository": "cosmos-framework",
        "repository_commit": os.environ.get("SOURCE_COMMIT"),
        "repository_tree": os.environ.get("SOURCE_TREE"),
        "source_dirty": os.environ.get("SOURCE_DIRTY") == "1",
        "repositories": {
            "cosmos-framework": {
                "commit": os.environ.get("SOURCE_COMMIT"),
                "tree": os.environ.get("SOURCE_TREE"),
                "dirty": os.environ.get("SOURCE_DIRTY") == "1",
            }
        },
        "source_manifest_sha256": manifest_sha256,
        "dependency_lock_sha256": _sha256(workspace / "uv.lock"),
        "dockerfile_sha256": _sha256(workspace / "Dockerfile"),
        "build_timestamp": os.environ.get("BUILD_TIMESTAMP"),
        "base_image": os.environ.get("PROVENANCE_BASE_IMAGE"),
        "cuda_build_version": os.environ.get("CUDA_VERSION"),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "packages": {
            name: _version(name)
            for name in ("cosmos-framework", "torch", "transformers", "torchcodec", "PyNvVideoCodec", "peft")
        },
        "package_locations": {
            name: str(Path(importlib.import_module(name).__file__).resolve())
            for name in ("cosmos_framework", "torch", "transformers")
        },
    }
    (output_dir / "image-provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if payload["source_dirty"]:
        raise RuntimeError("A reproducibility image cannot be built from a dirty source tree")
    if not payload["repository_commit"] or not payload["repository_tree"] or not payload["build_timestamp"]:
        raise RuntimeError("SOURCE_COMMIT, SOURCE_TREE, and BUILD_TIMESTAMP build arguments are required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
