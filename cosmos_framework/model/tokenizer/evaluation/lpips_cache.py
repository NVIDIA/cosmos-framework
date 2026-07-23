# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Process-shared LPIPS weight-cache warmup without serialized model construction."""

from __future__ import annotations

import fcntl
import os
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch

_ModelT = TypeVar("_ModelT")
_READY_MARKER_PAYLOAD = b"cosmos3-tokenizer-lpips-cache-ready-v1\n"
_READY_MARKER_READ_ATTEMPTS = 10
_READY_MARKER_READ_RETRY_SECONDS = 0.01


def _lpips_cache_paths(cache_key: str) -> tuple[Path, Path]:
    """Return validated lock and completion-marker paths in the shared torch cache."""
    if not cache_key or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in cache_key):
        raise ValueError(f"LPIPS cache key contains unsupported characters: {cache_key!r}.")
    checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir / f"{cache_key}.lock", checkpoints_dir / f"{cache_key}.ready"


def _required_cache_artifacts_exist(checkpoints_dir: Path, filenames: tuple[str, ...]) -> bool:
    """Return whether every named cache artifact is a nonempty regular file."""
    for filename in filenames:
        artifact_name = Path(filename)
        if not filename or artifact_name.name != filename or filename in {".", ".."}:
            raise ValueError(f"LPIPS cache artifact must be one plain filename, got {filename!r}.")
        artifact_path = checkpoints_dir / filename
        if artifact_path.is_symlink():
            raise RuntimeError(f"LPIPS cache artifact must not be a symlink: {artifact_path}")
        if not artifact_path.exists():
            return False
        if not artifact_path.is_file():
            raise RuntimeError(f"LPIPS cache artifact must be a regular file: {artifact_path}")
        if artifact_path.stat().st_size == 0:
            return False
    return True


def _ready_marker_exists(ready_path: Path, required_cache_filenames: tuple[str, ...] = ()) -> bool:
    """Return whether the marker and its underlying weight artifacts are ready."""
    if ready_path.is_symlink():
        raise RuntimeError(f"LPIPS cache completion marker must not be a symlink: {ready_path}")
    if not ready_path.exists():
        return False
    if not ready_path.is_file():
        raise RuntimeError(f"LPIPS cache completion marker must be a regular file: {ready_path}")
    observed_payload = b""
    for attempt in range(_READY_MARKER_READ_ATTEMPTS):
        observed_payload = ready_path.read_bytes()
        if observed_payload == _READY_MARKER_PAYLOAD:
            return _required_cache_artifacts_exist(ready_path.parent, required_cache_filenames)
        if attempt + 1 < _READY_MARKER_READ_ATTEMPTS:
            # Lustre can briefly expose a newly hard-linked marker before a
            # different client observes all bytes of its fsynced source inode.
            time.sleep(_READY_MARKER_READ_RETRY_SECONDS * (attempt + 1))
    raise RuntimeError(
        f"LPIPS cache completion marker has invalid content ({len(observed_payload)} bytes): {ready_path}"
    )


def _publish_ready_marker(ready_path: Path, required_cache_filenames: tuple[str, ...]) -> None:
    """Publish a durable marker after the shared model weights are fully available."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".lpips-ready-", dir=ready_path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(_READY_MARKER_PAYLOAD)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary_path, ready_path)
        except FileExistsError:
            if not _ready_marker_exists(ready_path, required_cache_filenames):
                raise RuntimeError(f"LPIPS cache completion marker could not be published: {ready_path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def construct_with_shared_cache_warmup(
    cache_key: str,
    constructor: Callable[[], _ModelT],
    *,
    required_cache_filenames: tuple[str, ...] = (),
) -> _ModelT:
    """Warm shared LPIPS weights once, then construct each process's model in parallel."""
    lock_path, ready_path = _lpips_cache_paths(cache_key)
    if _ready_marker_exists(ready_path, required_cache_filenames):
        return constructor()

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"Could not open the shared LPIPS cache warmup lock: {lock_path}") from error
    lock_acquired = False
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RuntimeError(f"LPIPS cache warmup lock must be a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        lock_acquired = True
        if not _ready_marker_exists(ready_path, required_cache_filenames):
            model = constructor()
            if not _required_cache_artifacts_exist(ready_path.parent, required_cache_filenames):
                raise RuntimeError("LPIPS model construction did not populate every required cache artifact.")
            _publish_ready_marker(ready_path, required_cache_filenames)
            return model
    finally:
        try:
            if lock_acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    return constructor()
