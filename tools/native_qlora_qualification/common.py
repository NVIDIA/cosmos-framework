# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Standard-library identities shared by qualification tools; never imports a model."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CAMERA = r"(?:Camera[0-9]+|[0-9]+(?:\.[0-9]+){3})"
_ROW_ID = re.compile(
    rf"(?P<date>[0-9]{{8}})_(?P<record>[0-9]+)_"
    rf"(?:(?:CN|CY|SN|SY)[0-9]+_T[0-9]+_{_CAMERA}_[0-9]+_[0-9]+"
    rf"|normal_{_CAMERA}_[0-9]+_event_[0-9]+_[0-9]+)"
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank row at line {number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at line {number}")
            rows.append(row)
    if not rows:
        raise ValueError("empty input")
    return rows


def recording_key(identifier: str) -> str:
    """Join all trial/camera/event clips from the same dated recording.

    WTS scenario folders include date, recording, scenario and trial. Removing
    the latter two deliberately groups related trials; normal clips also share
    a recording timestamp. This is conservative provenance grouping, not proof
    that different recordings contain independent actors or events.
    """
    match = _ROW_ID.fullmatch(identifier) if isinstance(identifier, str) else None
    if match is None:
        raise ValueError("unsupported WTS row ID; supply reviewed grouping before partitioning")
    datetime.datetime.strptime(match["date"], "%Y%m%d")
    return match["date"] + "_" + match["record"]


def inventory_groups(inventory: list[dict[str, Any]]) -> dict[str, str]:
    """Connected components of recording families AND exact video bytes."""
    parent: dict[str, str] = {}
    row_keys: dict[str, str] = {}
    media_owner: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for item in inventory:
        identifier = item["id"]
        if identifier in row_keys:
            raise ValueError("duplicate inventory row ID")
        media = item.get("sha256")
        if not isinstance(media, str) or re.fullmatch("[0-9a-f]{64}", media) is None:
            raise ValueError("inventory requires exact media SHA256")
        key = recording_key(identifier)
        row_keys[identifier] = key
        find(key)
        if media in media_owner:
            a, b = sorted((find(key), find(media_owner[media])))
            parent[b] = a
        else:
            media_owner[media] = key
    members: dict[str, list[str]] = {}
    for key in parent:
        members.setdefault(find(key), []).append(key)
    hashes = {root: digest_json(sorted(keys)) for root, keys in members.items()}
    return {identifier: hashes[find(key)] for identifier, key in row_keys.items()}


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
