# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Destination- and digest-bound ``.uploaded`` archive markers.

The marker records where one local checkpoint directory was archived (the non-secret destination
identity from ``object_store.upload_destination_identity``) and which files that archive covers,
with sizes for every file and SHA-256 digests where the archive path hashed them. Local retention
gates (``retention``) keep testing the marker's existence only; ``marker_covers`` is
the stricter local gate for callers that must know the marker describes THIS destination and the
directory's current contents, and the file inventory is what remote verification HEADs against.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.manifest import is_canonical_relative_path, is_sha256

MARKER_NAME = ".uploaded"
PROTOCOL = "posttrain.archive-marker.v1"
ARCHIVE_RECORD = ".posttrain-archive.json"
ARCHIVE_PROTOCOL = "posttrain.artifact-archive.v1"
_MARKER_KEYS = frozenset({"protocol", "destination", "files", "manifest_sha256"})
_FILE_KEYS = frozenset({"size", "sha256"})


def parse_archive_record(payload: bytes, manifest_sha256: str) -> dict[str, dict[str, Any]]:
    """Validate a remote inventory declaration against its committed native manifest.

    The declaration is immutable and precedes all payload writes. It is authoritative only once
    the native manifest with this digest exists; a declaration alone is an unfinished publication.
    """
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("unreadable artifact archive inventory") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"protocol", "manifest_sha256", "files"}
        or value["protocol"] != ARCHIVE_PROTOCOL
        or not is_sha256(value["manifest_sha256"])
        or value["manifest_sha256"] != manifest_sha256
        or not isinstance(value["files"], dict)
        or not value["files"]
    ):
        raise RuntimeError("artifact archive inventory does not bind this manifest")
    try:
        files = {relative: _validated_file_entry(relative, entry) for relative, entry in value["files"].items()}
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid artifact archive inventory: {error}") from error
    if ARCHIVE_RECORD in files or MARKER_NAME in files or any(entry["sha256"] is None for entry in files.values()):
        raise RuntimeError("artifact archive inventory must contain digested payloads only")
    return files


@dataclass(frozen=True)
class Marker:
    """One parsed archive marker.

    ``legacy`` is True for markers written before this protocol: the one-line bucket text of older
    framework releases, or another writer's JSON. A legacy marker proves only that an archive
    attempt was marked. It binds no destination and no digests, so ``marker_covers`` never accepts
    it and nothing can be digest-verified against the remote copy through it.
    """

    protocol: str
    destination: dict[str, str]
    files: dict[str, dict[str, Any]]
    manifest_sha256: str | None
    legacy: bool


def _validated_destination(destination: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(destination, Mapping) or not destination:
        raise ValueError("archive marker destination must be a non-empty mapping")
    if any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in destination.items()):
        raise ValueError("archive marker destination must map non-empty strings to strings")
    return dict(destination)


def _validated_file_entry(relative: str, entry: Mapping[str, object]) -> dict[str, Any]:
    if not is_canonical_relative_path(relative):
        raise ValueError(f"archive marker file path must be canonical and relative, got {relative!r}")
    if not isinstance(entry, Mapping) or set(entry) != _FILE_KEYS:
        raise ValueError(f"archive marker entry for {relative!r} must contain exactly size and sha256")
    size = entry["size"]
    digest = entry["sha256"]
    if type(size) is not int or size < 0:
        raise ValueError(f"archive marker size for {relative!r} must be a non-negative int")
    if digest is not None and not is_sha256(digest):
        raise ValueError(f"archive marker sha256 for {relative!r} must be a lowercase SHA-256 or None")
    return {"size": size, "sha256": digest}


def payload_inventory(iter_dir: str) -> dict[str, int]:
    """Return the size of every file below ``iter_dir`` that an archive marker must cover.

    The marker itself and the temporary of an interrupted marker write are not payload.
    """
    inventory: dict[str, int] = {}
    for root, _dirs, names in os.walk(iter_dir):
        for name in names:
            path = os.path.join(root, name)
            relative = os.path.relpath(path, iter_dir).replace(os.sep, "/")
            if relative in (MARKER_NAME, MARKER_NAME + ".tmp"):
                continue
            inventory[relative] = os.path.getsize(path)
    return inventory


def write_marker(
    iter_dir: str,
    *,
    destination: Mapping[str, str],
    files: Mapping[str, Mapping[str, object]],
    manifest_sha256: str | None = None,
) -> str:
    """Durably write the marker for one archived directory and return the marker path.

    ``files`` maps archive-relative paths to ``{"size": int, "sha256": str | None}``; ``sha256`` is
    None for payloads the archive path did not hash (bulk model/optimizer bytes), so the marker is
    destination-bound for full checkpoints and destination- and digest-bound for hashed payloads.
    The write is atomic (same-directory temporary, fsync, rename): a crash leaves either no marker or
    a complete one, and an existence-only retention gate never honours a torn file.
    """
    if manifest_sha256 is not None and not is_sha256(manifest_sha256):
        raise ValueError("archive marker manifest_sha256 must be a lowercase SHA-256 or None")
    payload = {
        "protocol": PROTOCOL,
        "destination": _validated_destination(destination),
        "files": {relative: _validated_file_entry(relative, entry) for relative, entry in sorted(files.items())},
        "manifest_sha256": manifest_sha256,
    }
    path = os.path.join(iter_dir, MARKER_NAME)
    atomic_write_json(path, payload)
    return path


def _parse_protocol_marker(value: dict[str, Any]) -> Marker | None:
    if set(value) != _MARKER_KEYS:
        return None
    destination = value["destination"]
    files = value["files"]
    manifest_sha256 = value["manifest_sha256"]
    if not isinstance(destination, dict) or not isinstance(files, dict):
        return None
    if manifest_sha256 is not None and not is_sha256(manifest_sha256):
        return None
    try:
        return Marker(
            protocol=PROTOCOL,
            destination=_validated_destination(destination),
            files={relative: _validated_file_entry(relative, entry) for relative, entry in files.items()},
            manifest_sha256=manifest_sha256,
            legacy=False,
        )
    except (TypeError, ValueError):
        return None


def _legacy_marker(protocol: str = "") -> Marker:
    return Marker(protocol=protocol, destination={}, files={}, manifest_sha256=None, legacy=True)


def _is_one_line_text(text: str) -> bool:
    body = text.strip("\r\n")
    return bool(body.strip()) and "\n" not in body and "\r" not in body and not body.lstrip().startswith(("{", "["))


def read_marker(iter_dir: str) -> Marker | None:
    """Read the marker of one directory, or return None when there is no usable marker.

    A protocol marker parses into its destination and inventory. A one-line text marker or another
    writer's JSON object reads as ``legacy`` (existence-only, never digest-verified). An empty,
    torn, or otherwise malformed marker reads as None: it proves nothing, and a caller that gates on
    existence alone still sees the file through ``os.path.isfile``.
    """
    path = os.path.join(iter_dir, MARKER_NAME)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    try:
        value: Any = json.loads(text)
    except ValueError:
        return _legacy_marker() if _is_one_line_text(text) else None
    if isinstance(value, dict):
        if value.get("protocol") == PROTOCOL:
            return _parse_protocol_marker(value)
        protocol = value.get("protocol")
        return _legacy_marker(protocol if isinstance(protocol, str) else "")
    return _legacy_marker() if _is_one_line_text(text) else None


def marker_covers(iter_dir: str, destination: Mapping[str, str]) -> bool:
    """Return whether a protocol marker binds exactly ``destination`` and covers the directory.

    Every payload file currently on disk must appear in the marker inventory at its current size. A
    file the archive did not cover (a partial archive), a resized file, another destination's
    marker, a legacy marker, or an unusable marker all fail. Inventory entries without a local file
    (optimizer state pruned after the archive) do not.
    """
    marker = read_marker(iter_dir)
    if marker is None or marker.legacy or marker.destination != dict(destination):
        return False
    for relative, size in payload_inventory(iter_dir).items():
        entry = marker.files.get(relative)
        if entry is None or entry["size"] != size:
            return False
    return True
