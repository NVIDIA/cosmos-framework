# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Content identity of a model artifact directory, for exact-resume identities.

A frozen KL reference built from ``rl.ref.model_dir`` is part of the training objective: two runs
anchored to different weights optimise different losses. The directory PATH does not identify those
weights (a mount can move, the same path can be re-populated with another checkpoint), so the
objective identity binds what is inside the directory instead:

* every metadata file — ``config.json``, ``generation_config.json``, any ``*.index.json``, and a
  DCP ``.metadata`` — by its full content;
* every weight file (``*.safetensors``, ``*.distcp``, ``*.bin``, ``*.pt``, ``*.pth``) by its
  relative path, its size, and its full-content SHA-256 digest.

The full digest is intentional: this mapping participates in the objective's exact-resume identity,
so sampling file windows would permit an unsampled byte change to keep the same identity. A
re-sharded copy of the same weights differs too (by design: exact resume binds the artifact, not the
function it computes). Tokenizer and processor files are not part of the identity: they do not
change the reference logprobs of fixed token ids. Stdlib only, so the login-node config path can
import it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

ARTIFACT_IDENTITY_VERSION = 2
ARTIFACT_IDENTITY_KIND = "model_dir_full_content"

METADATA_FILENAMES = frozenset({"config.json", "generation_config.json", ".metadata"})
METADATA_SUFFIXES = (".index.json",)
WEIGHT_SUFFIXES = (".safetensors", ".distcp", ".bin", ".pt", ".pth")

HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _is_metadata_file(name: str) -> bool:
    return name in METADATA_FILENAMES or name.endswith(METADATA_SUFFIXES)


def _is_weight_file(name: str) -> bool:
    return name.endswith(WEIGHT_SUFFIXES)


def _sha256(path: Path) -> tuple[int, str]:
    """Size and SHA-256 of one stable complete file, read in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_after != identity_before:
        raise RuntimeError(f"model artifact file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def model_artifact_identity(model_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Content identity of the model artifact stored under ``model_dir``.

    Returns a finite-JSON mapping: ``version``, ``kind``, ``sha256`` (over the sorted relative paths,
    sizes, and digests described in the module docstring), ``weight_files`` and ``weight_bytes``
    (informative counts). Hidden directories are skipped and symlinked directories are not followed;
    the result depends only on the directory's contents, never on its location.

    Raises:
        FileNotFoundError: ``model_dir`` is not a directory.
        ValueError: the directory holds no weight file (an identity of nothing would let an empty or
            half-copied reference directory pass as an anchor).
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model artifact directory does not exist: {root}")
    metadata: list[tuple[str, str]] = []
    weights: list[tuple[str, int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in filenames:
            path = Path(dirpath, filename)
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if _is_metadata_file(filename):
                _size, file_sha256 = _sha256(path)
                metadata.append((relative, file_sha256))
            elif _is_weight_file(filename):
                size, file_sha256 = _sha256(path)
                weights.append((relative, size, file_sha256))
    if not weights:
        raise ValueError(
            f"no model weight file ({', '.join(WEIGHT_SUFFIXES)}) under {root}; a frozen reference cannot be "
            "anchored to an empty or partially copied directory"
        )
    digest = hashlib.sha256()
    digest.update(f"{ARTIFACT_IDENTITY_KIND}.v{ARTIFACT_IDENTITY_VERSION}\0".encode())
    for relative, file_sha256 in sorted(metadata):
        digest.update(f"meta\0{relative}\0{file_sha256}\n".encode())
    for relative, size, file_sha256 in sorted(weights):
        digest.update(f"weight\0{relative}\0{size}\0{file_sha256}\n".encode())
    return {
        "version": ARTIFACT_IDENTITY_VERSION,
        "kind": ARTIFACT_IDENTITY_KIND,
        "sha256": digest.hexdigest(),
        "weight_files": len(weights),
        "weight_bytes": sum(size for _, size, _ in weights),
    }
