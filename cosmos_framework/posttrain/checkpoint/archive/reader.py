# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify archived artifacts and stage only the payloads bound by their committed inventory."""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Mapping
from contextlib import closing
from typing import Any

from cosmos_framework.posttrain.checkpoint.archive.marker import (
    ARCHIVE_RECORD,
    Marker,
    parse_archive_record,
    read_marker,
)
from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST, validate_artifact_manifest
from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes, atomic_write_json
from cosmos_framework.posttrain.checkpoint.manifest import (
    is_canonical_relative_path,
    is_sha256,
    iteration_name,
    parse_json_object,
    sha256,
)
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.log import logger

STAGED_RECORD = ".staged.json"
STAGED_PROTOCOL = "posttrain.artifact-staging.v1"
_PART_SUFFIX = ".part"
_STAGED_KEYS = frozenset({"protocol", "source", "manifest_sha256", "files", "complete"})


class ArchiveNotVerifiedError(Exception):
    """The archive cannot be proven to cover this local artifact."""


def _lower_metadata(response: Mapping[str, Any]) -> dict[str, str]:
    return object_store.response_metadata(response)


# ---------------------------------------------------------------- remote verification -----------------------------------
def remote_object_status(client: Any, bucket: str, key: str, size: int, digest: str | None) -> str:
    """Classify one object against its recorded size and digest from what the store reports.

    ``verified``: size matches and, when a digest is recorded, the object's ``sha256`` metadata
    equals it. ``absent``: no usable HEAD. ``unverifiable``: same size but no digest metadata for a
    recorded digest (nothing vouches for the bytes). ``different``: another size or digest.
    ``corrupt``: the store claims this exact digest for an object of another size, which no honest
    store can do. This is an identity check, not a re-read of the bytes.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 -- absent or unreachable: nothing is verified either way
        return "absent"
    remote_digest = _lower_metadata(head).get("sha256")
    if head.get("ContentLength") != size:
        return "corrupt" if digest is not None and remote_digest == digest else "different"
    if digest is None:
        return "verified"
    if remote_digest is None:
        return "unverifiable"
    return "verified" if remote_digest == digest else "different"


def remote_inventory_status(
    client: Any,
    bucket: str,
    prefix: str,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    return {
        relative: remote_object_status(client, bucket, f"{prefix}/{relative}", int(entry["size"]), entry["sha256"])
        for relative, entry in sorted(files.items())
    }


def unverified_objects(statuses: Mapping[str, str]) -> list[str]:
    return [f"{relative}: {status}" for relative, status in statuses.items() if status != "verified"]


def bound_marker(artifact_dir: str, credentials: str) -> tuple[Marker, str, str]:
    """Return the protocol marker bound to the credentials' endpoint, with its bucket and prefix.

    The marker's destination identity is recomputed from the credentials file's endpoint authority
    and must equal the recorded one, so a marker is never "verified" against a different store.
    """
    marker = read_marker(artifact_dir)
    if marker is None:
        raise ArchiveNotVerifiedError("no usable archive marker")
    if marker.legacy:
        raise ArchiveNotVerifiedError("legacy archive marker (existence-only, never digest-verified)")
    destination = marker.destination
    bucket, prefix = destination.get("bucket", ""), destination.get("prefix", "")
    if destination.get("kind") != "object_store" or not bucket or not prefix:
        raise ArchiveNotVerifiedError("archive marker does not name an object-store destination")
    name = os.path.basename(artifact_dir)
    if posixpath.basename(prefix) != name:
        raise ArchiveNotVerifiedError(f"archive marker destination {prefix!r} does not name this artifact")
    expected = object_store.upload_destination_identity(
        {"bucket": bucket, "prefix": posixpath.dirname(prefix), "credentials": credentials},
        name,
    )
    if expected != destination:
        raise ArchiveNotVerifiedError("archive marker binds another destination than these credentials address")
    return marker, bucket, prefix


def verify_archived_artifact(artifact_dir: str, *, credentials: str | None = None) -> bool:
    """Return whether every object the artifact's marker records is live and identical at its destination.

    False for a missing, legacy, or foreign-destination marker, for any absent or different object,
    and for a hashed entry whose object carries no digest metadata. The check is over what the
    store reports (size and ``sha256`` metadata), not a re-read of the bytes.
    """
    artifact_dir = os.path.abspath(artifact_dir)
    credentials = credentials or object_store.DEFAULT_CREDENTIALS
    try:
        marker, bucket, prefix = bound_marker(artifact_dir, credentials)
    except ArchiveNotVerifiedError as reason:
        logger.warning(f"{artifact_dir}: not verifiable ({reason})")
        return False
    client = object_store.build_s3_client(credentials)
    problems = unverified_objects(remote_inventory_status(client, bucket, prefix, marker.files))
    if problems:
        logger.warning(f"{artifact_dir}: archive s3://{bucket}/{prefix} does not verify ({'; '.join(problems)})")
        return False
    return True


# ---------------------------------------------------------------- fetch -------------------------------------------------
def _parse_artifact_uri(artifact_uri: str) -> tuple[str, str, str]:
    if not artifact_uri.startswith("s3://"):
        raise ValueError(f"expected s3://bucket/.../checkpoint_artifacts/iter_*, got {artifact_uri!r}")
    bucket, _, prefix = artifact_uri[len("s3://") :].partition("/")
    prefix = prefix.strip("/")
    name = iteration_name(artifact_uri)
    if not bucket or not prefix:
        raise ValueError(f"expected s3://bucket/.../checkpoint_artifacts/iter_*, got {artifact_uri!r}")
    return bucket, prefix, name


def _read_staged_record(local_dir: str) -> dict[str, Any] | None:
    """Parse this tool's ownership record, or None when there is none that proves ownership."""
    try:
        with open(os.path.join(local_dir, STAGED_RECORD), "rb") as handle:
            value: Any = json.loads(handle.read())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != _STAGED_KEYS or value["protocol"] != STAGED_PROTOCOL:
        return None
    source, files = value["source"], value["files"]
    if (
        not isinstance(source, dict)
        or not source
        or any(not isinstance(key, str) or not isinstance(item, str) for key, item in source.items())
        or not is_sha256(value["manifest_sha256"])
        or not isinstance(files, dict)
        or not isinstance(value["complete"], bool)
    ):
        return None
    for relative, entry in files.items():
        if (
            not is_canonical_relative_path(relative)
            or not isinstance(entry, dict)
            or set(entry) != {"size", "sha256"}
            or type(entry["size"]) is not int
            or (entry["sha256"] is not None and not is_sha256(entry["sha256"]))
        ):
            return None
    return value


def _discard_recorded(local_dir: str, record: Mapping[str, Any]) -> list[str]:
    removed: list[str] = []
    for relative in sorted(record["files"]):
        path = os.path.join(local_dir, *str(relative).split("/"))
        for candidate in (path, path + _PART_SUFFIX):
            if os.path.lexists(candidate):
                os.remove(candidate)
                removed.append(candidate)
    record_path = os.path.join(local_dir, STAGED_RECORD)
    if os.path.lexists(record_path):
        os.remove(record_path)
        removed.append(record_path)
    if os.path.isdir(local_dir) and not os.listdir(local_dir):
        os.rmdir(local_dir)
        removed.append(local_dir + os.sep)
    return removed


def _require_staged_ownership(local_dir: str, expected: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Refuse occupied destination paths before changing a prior staging claim or its files."""
    owned = {str(relative) for relative in record["files"]}
    owned.update(relative + _PART_SUFFIX for relative in tuple(owned))
    for relative in (*expected, CHECKPOINT_ARTIFACT_MANIFEST):
        for candidate in (relative, relative + _PART_SUFFIX):
            path = os.path.join(local_dir, *candidate.split("/"))
            if os.path.lexists(path) and candidate not in owned:
                raise RuntimeError(f"refusing to replace {path!r}: this path is not owned by the prior staging record")


def discard_staged_artifact(local_dir: str) -> list[str]:
    """Remove only what this tool staged into ``local_dir`` and return the removed paths.

    The files named by the ownership record, their ``.part`` temporaries, and the record itself go;
    the directory goes only once it is empty. Without a record proving ownership nothing is removed
    and ``RuntimeError`` is raised: a caller's files beside a staged copy are never this tool's.
    """
    local_dir = os.path.abspath(local_dir)
    record = _read_staged_record(local_dir)
    if record is None:
        raise RuntimeError(
            f"refusing to discard {local_dir!r}: no {STAGED_RECORD} ownership record written by this tool"
        )
    return _discard_recorded(local_dir, record)


def _required_object_digest(client: Any, bucket: str, key: str, location: str) -> str:
    """Return the ``sha256`` metadata of one object that the manifest records no digest for.

    Raises ``RuntimeError`` naming the key when the object cannot be read or carries no well-formed
    digest: nothing vouches for its bytes, so the archive is unverifiable and must not be staged.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 -- absent or unreadable: nothing to verify against
        raise RuntimeError(f"archived artifact {location}: s3://{bucket}/{key} is not readable ({error})") from error
    digest = _lower_metadata(head).get("sha256")
    if digest is None or not is_sha256(digest):
        raise RuntimeError(
            f"archived artifact {location}: s3://{bucket}/{key} carries no sha256 metadata and the manifest records "
            "no digest for it; the archive is unverifiable (published before digests were attached?) and is not staged"
        )
    return digest


def _reusable(path: str, expected: Mapping[str, Any], prior: Mapping[str, Any] | None) -> bool:
    """A staged file is reused only when this tool's record proves its digest and the bytes still match."""
    if prior is None or prior["sha256"] is None or not os.path.isfile(path):
        return False
    if os.path.getsize(path) != expected["size"] or prior["size"] != expected["size"]:
        return False
    if expected["sha256"] is not None and expected["sha256"] != prior["sha256"]:
        return False
    return object_store.file_sha256(path) == prior["sha256"]


def _archive_inventory(
    client: Any, bucket: str, prefix: str, encoded_manifest: bytes
) -> dict[str, dict[str, Any]] | None:
    """Read an optional committed inventory; only a missing object permits legacy fallback."""
    from botocore.exceptions import ClientError

    key = f"{prefix}/{ARCHIVE_RECORD}"
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        with closing(response["Body"]) as body:
            payload = bytes(body.read())
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return None
        raise RuntimeError(f"cannot read archived artifact inventory s3://{bucket}/{key}") from error
    digest = sha256(payload)
    if _lower_metadata(response).get("sha256") not in (None, digest):
        raise RuntimeError(f"archived inventory s3://{bucket}/{key} carries digest metadata for other bytes")
    manifest_digest = sha256(encoded_manifest)
    files = parse_archive_record(payload, manifest_digest)
    if files.get(CHECKPOINT_ARTIFACT_MANIFEST) != {"size": len(encoded_manifest), "sha256": manifest_digest}:
        raise RuntimeError(f"archived inventory s3://{bucket}/{key} does not describe the committed manifest")
    return files


def _physical_inventory(
    client: Any,
    bucket: str,
    prefix: str,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Select physical components from the committed inventory, never from a live DCP listing."""
    location = f"s3://{bucket}/{prefix}"
    expected: dict[str, dict[str, Any]] = {}
    for descriptor in manifest["components"].values():
        relative = str(descriptor["path"])
        if descriptor["kind"] == "dcp":
            if inventory is None:
                raise RuntimeError(
                    f"archived artifact {location}: DCP component {relative!r} has no committed shard inventory; "
                    "publish a local copy to a new archive destination before fetching it"
                )
            shards = {path: dict(entry) for path, entry in inventory.items() if path.startswith(relative + "/")}
            if f"{relative}/.metadata" not in shards or not any(path.endswith(".distcp") for path in shards):
                raise RuntimeError(f"archived artifact {location}: inventory has incomplete DCP component {relative!r}")
            expected.update(shards)
            continue
        entry = {"size": int(descriptor["size"]), "sha256": descriptor.get("sha256")}
        if inventory is not None:
            archived = inventory.get(relative)
            if (
                archived is None
                or archived["size"] != entry["size"]
                or entry["sha256"] is not None
                and archived["sha256"] != entry["sha256"]
            ):
                raise RuntimeError(f"archived artifact {location}: inventory disagrees with component {relative!r}")
            entry = dict(archived)
        elif entry["sha256"] is None:
            entry["sha256"] = _required_object_digest(client, bucket, f"{prefix}/{relative}", location)
        expected[relative] = entry
    if not expected:
        raise RuntimeError(f"archived artifact {location} names no physical payload")
    return expected


def fetch_archived_artifact(artifact_uri: str, dst_dir: str, *, credentials: str | None = None) -> str:
    """Stage one archived artifact into ``<dst_dir>/<iter_name>`` and return that directory.

    The native manifest commits the immutable archive inventory, which names every required file
    and its digest. DCP requires that inventory: a live listing cannot reveal a deleted shard.
    Older file-only artifacts may instead use their descriptor digest or object metadata. A
    missing inventory for DCP, or a file with neither digest source, is refused before staging.
    Each object is then downloaded through a ``.part`` temporary and
    proven by size and SHA-256; the manifest is written after the payloads and the
    ``STAGED_RECORD`` ownership record last. A directory that
    holds files without this tool's record is foreign and is never reused, replaced, or deleted.
    This tool's own copy of the same source and manifest is reused file by file where the recorded
    digest still matches (an interrupted fetch resumes); a copy of another source is replaced,
    removing only the files its record lists.
    """
    from botocore.exceptions import ClientError

    bucket, prefix, name = _parse_artifact_uri(artifact_uri)
    credentials = credentials or object_store.DEFAULT_CREDENTIALS
    source = object_store.upload_destination_identity(
        {"bucket": bucket, "prefix": posixpath.dirname(prefix), "credentials": credentials},
        name,
    )
    client = object_store.build_s3_client(credentials)
    location = f"s3://{bucket}/{prefix}"
    manifest_key = f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}"
    try:
        response = client.get_object(Bucket=bucket, Key=manifest_key)
        with closing(response["Body"]) as body:
            encoded = bytes(body.read())
    except Exception as error:  # noqa: BLE001 -- absent or unreadable commit: nothing to stage
        raise RuntimeError(f"archived artifact {location} has no readable manifest ({error})") from error
    manifest = validate_artifact_manifest(
        parse_json_object(encoded, context=f"checkpoint artifact manifest at {location!r}"),
        artifact_location=location,
    )
    manifest_sha256 = sha256(encoded)
    if _lower_metadata(response).get("sha256") not in (None, manifest_sha256):
        raise RuntimeError(f"archived manifest {location} carries digest metadata for other bytes (corrupt store)")
    # Remote paths are store-controlled input: the complete declaration is validated before any
    # local path is created, including diagnostic files that this fetch does not need to download.
    inventory = _archive_inventory(client, bucket, prefix, encoded)
    # Every selected object is bound to a digest before any local path exists; legacy file-only
    # artifacts may obtain missing descriptor digests from object metadata.
    expected = _physical_inventory(client, bucket, prefix, manifest, inventory)

    local_dir = os.path.join(os.path.abspath(dst_dir), name)
    prior_files: Mapping[str, Mapping[str, Any]] = {}
    if os.path.isdir(local_dir):
        record = _read_staged_record(local_dir)
        if record is None:
            if os.listdir(local_dir):
                raise RuntimeError(
                    f"refusing to reuse or replace {local_dir!r}: it holds files this tool did not stage "
                    f"(no {STAGED_RECORD} ownership record)"
                )
        else:
            _require_staged_ownership(local_dir, expected, record)
            if record["source"] == source and record["manifest_sha256"] == manifest_sha256:
                prior_files = record["files"]
            else:
                _discard_recorded(local_dir, record)
    os.makedirs(local_dir, exist_ok=True)
    record_path = os.path.join(local_dir, STAGED_RECORD)
    staged = {relative: dict(entry) for relative, entry in expected.items()}
    staged[CHECKPOINT_ARTIFACT_MANIFEST] = {"size": len(encoded), "sha256": manifest_sha256}
    claim = {"protocol": STAGED_PROTOCOL, "source": source, "manifest_sha256": manifest_sha256}
    atomic_write_json(record_path, {**claim, "files": staged, "complete": False})
    transfer = object_store._transfer_config()
    for relative, entry in sorted(expected.items()):
        path = os.path.join(local_dir, *relative.split("/"))
        prior = prior_files.get(relative)
        if _reusable(path, entry, prior):
            staged[relative]["sha256"] = prior["sha256"] if prior is not None else None
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        key = f"{prefix}/{relative}"
        part = path + _PART_SUFFIX
        try:
            client.download_file(bucket, key, part, Config=transfer)
            size = os.path.getsize(part)
            if size != entry["size"]:
                raise RuntimeError(
                    f"s3://{bucket}/{key}: downloaded {size} bytes, the manifest records {entry['size']}"
                )
            digest = object_store.file_sha256(part)
            want = str(entry["sha256"])
            if digest != want:
                raise RuntimeError(f"s3://{bucket}/{key}: downloaded bytes have sha256 {digest}, expected {want}")
            os.replace(part, path)
        except BaseException as error:
            if os.path.lexists(part):
                os.remove(part)
            if isinstance(error, ClientError):
                raise RuntimeError(f"archived artifact {location}: cannot download required object {key!r}") from error
            raise
        staged[relative]["sha256"] = digest
    atomic_write_bytes(os.path.join(local_dir, CHECKPOINT_ARTIFACT_MANIFEST), encoded)
    atomic_write_json(record_path, {**claim, "files": staged, "complete": True})
    logger.info(f"staged {location} -> {local_dir} ({len(expected)} payload objects)")
    return local_dir
