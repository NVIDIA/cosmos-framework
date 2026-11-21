# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Content-checked random access to committed, uncompressed physical image shards.

The logical dataset still hashes each original PNG, not its storage container.
No persistent file descriptors cross data-loader forks or share seek positions.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlsplit

from cosmos_framework.posttrain.utils.object_store import file_sha256

FORMAT = "cosmos-flow-media-shard-v1"
SCHEME = "flowpack-v1"
MAGIC = b"COSMOS_FLOW_MEDIA_V1\0"
MAX_IMAGE_BYTES = 64 * 1024**2


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate flow media metadata key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite flow media metadata: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _stat(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def media_reference(directory, index_sha256, member):
    return f"{SCHEME}:{quote(str(directory), safe='/')}?index_sha256={index_sha256}#{quote(member, safe='/')}"


def _parse(value):
    parts = urlsplit(value)
    query = parse_qs(parts.query, strict_parsing=True)
    if parts.scheme != SCHEME or parts.netloc or set(query) != {"index_sha256"} or len(query["index_sha256"]) != 1:
        raise ValueError("invalid flowpack-v1 media reference")
    sha = query["index_sha256"][0]
    member = unquote(parts.fragment)
    directory = unquote(parts.path)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", sha)
        or not directory
        or not member
        or PurePosixPath(member).is_absolute()
        or ".." in PurePosixPath(member).parts
    ):
        raise ValueError("invalid flowpack-v1 member or index identity")
    return Path(directory), sha, member


def resolve_media(value, root):
    if value.startswith(SCHEME + ":"):
        directory, sha, member = _parse(value)
        return media_reference((Path(root) / directory).resolve(), sha, member)
    return str((Path(root) / value).resolve())


@lru_cache(maxsize=8)
def _index(directory, expected_sha, marker_stat, index_stat, payload_stat):
    root = Path(directory)
    if marker_stat[2] > 1024**2 or index_stat[2] > 64 * 1024**2:
        raise ValueError("flow media index/commit record exceeds its bounded metadata size")
    marker = _json((root / "complete.json").read_text())
    raw = (root / "index.json").read_bytes()
    if (
        not isinstance(marker, dict)
        or marker.get("format") != FORMAT
        or marker.get("index_sha256") != expected_sha
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        raise ValueError(f"incomplete or changed flow media index: {root}")
    index = _json(raw)
    if not isinstance(index, dict) or index.get("format") != FORMAT or marker.get("payload_bytes") != payload_stat[2]:
        raise ValueError(f"flow media payload identity/size mismatch: {root}")
    with (root / "media.bin").open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"wrong flow media shard format: {root}")
    expected_offset = len(MAGIC)
    entries = index.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("flow media shard has no entries")
    for name, entry in entries.items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("unsafe flow media member")
        if (
            not isinstance(entry, dict)
            or set(entry) != {"offset", "bytes", "sha256"}
            or type(entry["offset"]) is not int
            or entry["offset"] != expected_offset
            or type(entry["bytes"]) is not int
            or not 0 < entry["bytes"] <= MAX_IMAGE_BYTES
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("invalid, overlapping or unbounded flow media entry")
        expected_offset += entry["bytes"]
    if expected_offset != payload_stat[2]:
        raise ValueError("unindexed or truncated flow media payload")
    if (_stat(root / "complete.json"), _stat(root / "index.json"), _stat(root / "media.bin")) != (
        marker_stat,
        index_stat,
        payload_stat,
    ):
        raise ValueError("flow media shard changed while validating its index")
    return index


def _packed_index(directory, expected_sha):
    root = Path(directory)
    try:
        states = tuple(_stat(root / name) for name in ("complete.json", "index.json", "media.bin"))
    except FileNotFoundError as error:
        raise ValueError(f"incomplete flow media shard: {root}") from error
    return _index(str(root), expected_sha, *states), states


def packed_index(directory, expected_sha):
    """Return detached metadata; callers cannot change the validated read cache."""
    index, states = _packed_index(directory, expected_sha)
    return copy.deepcopy(index), states


def read_packed_media(value):
    root, sha, member = _parse(value)
    index, states = _packed_index(root, sha)
    entry = index["entries"].get(member)
    if entry is None:
        raise ValueError(f"flow media member absent: {member}")
    with (root / "media.bin").open("rb") as handle:
        info = os.fstat(handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != states[2]:
            raise ValueError("flow media payload replaced before reading")
        handle.seek(entry["offset"])
        data = handle.read(entry["bytes"])
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"flow media content changed: {member}")
    if tuple(_stat(root / name) for name in ("complete.json", "index.json", "media.bin")) != states:
        raise ValueError("flow media shard changed during read")
    return data


def media_sha256(value):
    if value.startswith(SCHEME + ":"):
        return hashlib.sha256(read_packed_media(value)).hexdigest()
    return file_sha256(value)


@contextmanager
def open_flow_image(value):
    from PIL import Image

    if isinstance(value, str) and value.startswith(SCHEME + ":"):
        with io.BytesIO(read_packed_media(value)) as stream, Image.open(stream) as image:
            yield image
    else:
        with Image.open(value) as image:
            yield image
