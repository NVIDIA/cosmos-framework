# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read captured bundle configs and assets with deterministic augmentation seeds."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


def sample_seed(seed: int, epoch: int, index: int) -> int:
    digest = hashlib.sha256(f"local-simulation-v1:{seed}:{epoch}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes bundle: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _constructor(name: Any, constructors: dict[str, type[Any]]) -> type[Any]:
    if not isinstance(name, str) or name not in constructors:
        raise ValueError(f"Unsupported local transform constructor: {name!r}")
    return constructors[name]


def decode_config(value: Any, *, constructors: dict[str, type[Any]]) -> Any:
    """Bind captured constructor names only to objects supplied by trusted code."""
    if isinstance(value, list):
        return [decode_config(item, constructors=constructors) for item in value]
    if not isinstance(value, dict):
        return value
    if "__dataclass__" in value:
        target = _constructor(value["__dataclass__"], constructors)
        if not dataclasses.is_dataclass(target):
            raise ValueError("Captured dataclass constructor is not a dataclass.")
        return target(**decode_config(value["fields"], constructors=constructors))
    if "__enum__" in value:
        target = _constructor(value["__enum__"], constructors)
        if not issubclass(target, enum.Enum):
            raise ValueError("Captured enum constructor is not an enum.")
        return target(decode_config(value["value"], constructors=constructors))
    if "__tuple__" in value:
        return tuple(decode_config(item, constructors=constructors) for item in value["__tuple__"])
    if "__mapping__" in value:
        decoded = {
            decode_config(key, constructors=constructors): decode_config(item, constructors=constructors)
            for key, item in value["__mapping__"]
        }
    else:
        decoded = {key: decode_config(item, constructors=constructors) for key, item in value.items()}
    if "_target_" in decoded:
        decoded["_target_"] = _constructor(decoded["_target_"], constructors)
    return decoded


class LocalByteRangeBackend:
    """Original URI identities resolve exclusively to verified bundle files."""

    root: Path
    assets: dict[str, str]

    def __init__(self, root: Path, assets: dict[str, str]) -> None:
        self.root = root
        self.assets = assets

    def get(self, filepath: str, offset: int = 0, size: int | None = None) -> bytes:
        path = local_path(self.root, self.assets[filepath])
        with path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read() if size is None else stream.read(size)
        if size is not None and len(payload) != size:
            raise ValueError(f"Incomplete local range: {path}, offset={offset}, size={size}")
        return payload
