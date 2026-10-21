# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Materialize an immutable S3 Transfer length index for local SQLite reads."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


def _build_index_s3_client(creds_path: str | Path | None, *, max_pool_connections: int = 10) -> Any:
    """Use index-specific credentials, or boto3's normal credential chain."""
    credentials: dict[str, Any] = {}
    if creds_path is not None:
        with Path(creds_path).open() as handle:
            supplied = json.load(handle)
        credentials = {
            key: supplied[key]
            for key in (
                "endpoint_url",
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "region_name",
            )
            if supplied.get(key)
        }
    return boto3.client(
        "s3",
        **credentials,
        config=Config(
            signature_version="s3v4",
            max_pool_connections=max_pool_connections,
            s3={"addressing_style": "path"},
            connect_timeout=15,
            read_timeout=60,
            retries={"mode": "standard", "max_attempts": 8},
        ),
    )


def _sha256_file(path: Path) -> str:
    """Hash without holding the index in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> list[int]:
    """Identify replacements and writes, even when size and mtime are preserved."""
    metadata = path.stat()
    return [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]


def _verified_index_state(path: Path, expected_hash: str, marker: Path | None = None) -> dict[str, Any] | None:
    """Reuse a checksum only for the same unchanged file under the caller's lock."""
    if path.is_symlink() or not path.is_file():
        return None
    # Repair legacy permissions before recording the identity. Avoid an
    # unconditional chmod: even an unchanged mode would invalidate ctime.
    if stat.S_IMODE(path.stat().st_mode) != 0o400:
        path.chmod(0o400)
    state = {"sha256": expected_hash, "file_identity": _file_identity(path)}
    if marker is not None and not marker.is_symlink() and marker.is_file():
        try:
            if json.loads(marker.read_text()) == state:
                return state
        except (ValueError, UnicodeError):
            # An interrupted or malformed marker requires verification again.
            pass
    actual_hash = _sha256_file(path)
    if _file_identity(path) != state["file_identity"]:
        raise ValueError("Transfer length index changed while its SHA-256 was being verified.")
    return state if actual_hash == expected_hash else None


def _ensure_private_cache_directory(path: Path) -> None:
    """Create or tighten a cache directory without following a pre-existing symlink."""
    path.mkdir(mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise PermissionError(
            f"Transfer index cache directory must be owned by the current user, not a symlink: {path}"
        )
    path.chmod(0o700)


def _ensure_safe_cache_parent(path: Path) -> Path:
    """Require a cache path that other users cannot replace, including its ancestors."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    # Container user namespaces may expose the filesystem root under a mapped UID.
    # Its owner already controls the namespace containing every cache path.
    trusted_owners = {0, os.getuid(), current.lstat().st_uid}
    for component in (None, *absolute.parts[1:]):
        if component is not None:
            current = current / component
            current.mkdir(mode=0o700, exist_ok=True)
        metadata = current.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid not in trusted_owners
            or (metadata.st_mode & 0o022 and not metadata.st_mode & stat.S_ISVTX)
        ):
            raise PermissionError(
                f"Unsafe Transfer index cache directory: {current}. Cache directories and their ancestors "
                "must be real directories owned by the current user, root, or the filesystem root's owner; "
                "shared writable directories must have the sticky bit set. "
                "Choose a private path or a cache beneath root-owned /tmp."
            )
    return current


def resolve_transfer_length_manifest(
    manifest_path: str | Path,
    *,
    creds_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a local manifest, downloading and verifying its S3 index if needed.

    Local paths retain their existing behavior. S3 manifests are read afresh so
    updates cannot silently reuse an old index. The cache identity includes the
    URI and manifest bytes. A file lock shares one download and checksum
    verification across ranks while the cached file identity is unchanged,
    without a distributed barrier.
    """
    uri = str(manifest_path)
    parsed = urlsplit(uri)
    if not parsed.scheme:
        return Path(manifest_path).expanduser().resolve()
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError(f"Expected a local path or an s3:// manifest URI, got {uri!r}.")
    client = _build_index_s3_client(creds_path)
    key = parsed.path.lstrip("/")
    response = client.get_object(Bucket=parsed.netloc, Key=key)
    try:
        manifest_bytes = response["Body"].read()
    finally:
        response["Body"].close()
    manifest = json.loads(manifest_bytes)
    index_file = manifest.get("index_file")
    if (
        not isinstance(index_file, str)
        or index_file in {"", ".", "..", "manifest.json", ".download.lock", ".index-verified.json"}
        or Path(index_file).name != index_file
        or "\\" in index_file
    ):
        raise ValueError("The S3 Transfer manifest index_file must name a sibling SQLite file.")
    expected_hash = manifest.get("index_sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("The S3 Transfer manifest requires a lowercase SHA-256 index_sha256.")
    cache_root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / f"cosmos3-transfer-length-index-{os.getuid()}"
    )
    if cache_dir is None:
        cache_root = _ensure_safe_cache_parent(cache_root.parent) / cache_root.name
        _ensure_private_cache_directory(cache_root)
    else:
        # An explicit cache parent may be shared with the sticky bit set; keep its
        # permissions and protect our entries below it. Validate ancestors too:
        # a replaceable parent would invalidate every later path-based operation.
        cache_root = _ensure_safe_cache_parent(cache_root)
    identity = hashlib.sha256(uri.encode() + b"\0" + manifest_bytes).hexdigest()
    output = cache_root / identity
    _ensure_private_cache_directory(output)
    output = output.resolve()
    index_path = output / index_file
    local_manifest = output / "manifest.json"
    verification_marker = output / ".index-verified.json"
    # Keep synchronization inside the verified private entry, including when its
    # parent is shared. Legacy locks in the parent must not affect initialization.
    lock_fd = os.open(output / ".download.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a+") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        verified_state = _verified_index_state(index_path, expected_hash, verification_marker)
        if verified_state is None:
            index_key = key.rsplit("/", 1)[0] + "/" + index_file if "/" in key else index_file
            with tempfile.NamedTemporaryFile(dir=output, prefix=".index-", delete=False) as handle:
                temporary_index = Path(handle.name)
            try:
                # Boto3 streams multipart downloads to disk and retries failed parts.
                # Avoid extra thread pools in every rank during distributed startup.
                client.download_file(
                    parsed.netloc, index_key, str(temporary_index), Config=TransferConfig(use_threads=False)
                )
                verified_state = _verified_index_state(temporary_index, expected_hash)
                if verified_state is None:
                    raise ValueError(f"Transfer length-index SHA-256 mismatch for s3://{parsed.netloc}/{index_key}.")
                os.replace(temporary_index, index_path)
                # Rename can change ctime; bind the verified bytes to their final
                # published identity before any other rank acquires the lock.
                verified_state["file_identity"] = _file_identity(index_path)
            finally:
                temporary_index.unlink(missing_ok=True)
        # Repair permissions on indexes published by earlier cache versions in
        # _verified_index_state, then share that verification across local ranks.
        with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=".verification-", delete=False) as handle:
            temporary_verification = Path(handle.name)
            json.dump(verified_state, handle)
        try:
            temporary_verification.chmod(0o400)
            os.replace(temporary_verification, verification_marker)
        finally:
            temporary_verification.unlink(missing_ok=True)
        # Publish the manifest only after the complete, verified index is present.
        with tempfile.NamedTemporaryFile(dir=output, prefix=".manifest-", delete=False) as handle:
            temporary_manifest = Path(handle.name)
            handle.write(manifest_bytes)
        try:
            temporary_manifest.chmod(0o400)
            os.replace(temporary_manifest, local_manifest)
        finally:
            temporary_manifest.unlink(missing_ok=True)
    return local_manifest
