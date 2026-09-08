# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolve stable identities for local and S3 tokenizer checkpoints."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import boto3

_S3_CHECKPOINT_OBJECT_SUFFIXES = frozenset({".ckpt", ".pt", ".pth", ".safetensors"})
_S3_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def extract_checkpoint_provenance(
    payload: Mapping[str, object],
    *,
    source: str,
) -> tuple[str | None, dict[str, object] | None]:
    """Extract normalized checkpoint provenance from a latent-statistics payload."""
    signature = payload.get("signature")
    source_checkpoint = payload.get("source_checkpoint")
    source_checkpoint_identity = payload.get("source_checkpoint_identity")
    if isinstance(signature, Mapping):
        if source_checkpoint is None:
            source_checkpoint = signature.get("vae_path")
        if source_checkpoint_identity is None:
            source_checkpoint_identity = signature.get("vae_identity")
    if source_checkpoint is not None and not isinstance(source_checkpoint, str):
        raise ValueError(
            f"Latent-normalization source_checkpoint in {source} must be a string, "
            f"got {type(source_checkpoint).__name__}."
        )
    if source_checkpoint_identity is not None and not isinstance(source_checkpoint_identity, Mapping):
        raise ValueError(
            f"Latent-normalization source_checkpoint_identity in {source} must be a mapping, "
            f"got {type(source_checkpoint_identity).__name__}."
        )
    normalized_checkpoint_identity = (
        dict(source_checkpoint_identity) if isinstance(source_checkpoint_identity, Mapping) else None
    )
    return source_checkpoint, normalized_checkpoint_identity


def stable_object_metadata(response: dict[str, object]) -> dict[str, object]:
    """Extract stable identity fields from an S3 response."""
    last_modified = response.get("LastModified")
    return {
        "etag": str(response.get("ETag", "")).strip('"'),
        "version_id": response.get("VersionId"),
        "content_length": response.get("ContentLength", response.get("Size")),
        "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified),
    }


def make_s3_client(credentials_path: str | None) -> Any:
    """Construct an S3 client from the repository credential-file format."""
    if credentials_path is None:
        return boto3.client("s3")
    with open(credentials_path) as credential_file:
        credentials = json.load(credential_file)
    return boto3.client(
        "s3",
        aws_access_key_id=credentials.get("aws_access_key_id"),
        aws_secret_access_key=credentials.get("aws_secret_access_key"),
        endpoint_url=credentials.get("endpoint_url"),
        region_name=credentials.get("region_name", "us-east-1"),
    )


def _sha256_file(path: Path) -> str:
    """Hash one local file without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        while chunk := checkpoint_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _head_s3_object(client: Any, *, bucket: str, key: str) -> dict[str, object] | None:
    """Return exact-object metadata, or ``None`` when the key does not exist."""
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        error = getattr(exc, "response", {}).get("Error", {})
        if str(error.get("Code", "")) in _S3_NOT_FOUND_CODES:
            return None
        raise
    if not isinstance(response, dict):
        raise TypeError(f"S3 HEAD for s3://{bucket}/{key} returned {type(response).__name__}, expected dict.")
    return response


def _list_s3_prefix_objects(client: Any, *, bucket: str, prefix: str) -> list[dict[str, object]]:
    """Return stable identities for checkpoint children, excluding a folder marker."""
    objects: list[dict[str, object]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if item["Key"] == prefix:
                continue
            objects.append(
                {
                    "key": item["Key"],
                    **stable_object_metadata(item),
                }
            )
    objects.sort(key=lambda item: str(item["key"]))
    return objects


def resolve_checkpoint_identity(checkpoint_path: str, credentials_path: str | None) -> dict[str, object]:
    """Resolve a stable model identity for local files/directories or S3 objects."""
    parsed = urlparse(checkpoint_path)
    if parsed.scheme == "s3":
        client = make_s3_client(credentials_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        object_suffix = PurePosixPath(key).suffix.lower()
        if not key.endswith("/") and object_suffix in _S3_CHECKPOINT_OBJECT_SUFFIXES:
            response = _head_s3_object(client, bucket=bucket, key=key)
            if response is None:
                raise FileNotFoundError(f"Checkpoint object does not exist: {checkpoint_path}")
            return {
                "kind": "s3_object",
                "bucket": bucket,
                "key": key,
                **stable_object_metadata(response),
            }

        prefix = key.rstrip("/") + "/"
        objects = _list_s3_prefix_objects(client, bucket=bucket, prefix=prefix)
        exact_response = None if key.endswith("/") else _head_s3_object(client, bucket=bucket, key=key)
        if objects:
            if exact_response is not None:
                raise ValueError(
                    f"Ambiguous S3 checkpoint path {checkpoint_path}: an exact non-checkpoint object and "
                    "checkpoint children both exist."
                )
            return {"kind": "s3_prefix", "bucket": bucket, "prefix": prefix, "objects": objects}
        if exact_response is not None:
            raise ValueError(
                f"S3 checkpoint object {checkpoint_path} has unsupported suffix {object_suffix!r}; "
                f"expected one of {sorted(_S3_CHECKPOINT_OBJECT_SUFFIXES)} or a DCP prefix."
            )
        raise FileNotFoundError(f"Checkpoint object or prefix does not exist: {checkpoint_path}")

    path = Path(checkpoint_path)
    if path.is_file():
        return {
            "kind": "local_file",
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        files = [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]
        return {
            "kind": "local_directory",
            "path": str(path.resolve()),
            "files": [
                {
                    "path": str(candidate.relative_to(path)),
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
                for candidate in files
            ],
        }
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
