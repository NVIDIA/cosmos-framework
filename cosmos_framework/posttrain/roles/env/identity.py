# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stable content identity for finite environment sources."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from collections.abc import Mapping, Set
from enum import Enum
from typing import Any


def _tagged_digest(tag: str, parts: list[bytes]) -> bytes:
    """Hash one typed, length-delimited value without process-local ``repr``."""
    digest = hashlib.sha256()
    encoded_tag = tag.encode()
    digest.update(len(encoded_tag).to_bytes(8, byteorder="big"))
    digest.update(encoded_tag)
    digest.update(len(parts).to_bytes(8, byteorder="big"))
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return digest.digest()


def stable_value_digest(value: Any, active: set[int] | None = None) -> bytes:
    """Hash known content-bearing values and reject opaque or recursive objects."""
    if value is None:
        return _tagged_digest("none", [])
    if isinstance(value, bool):
        return _tagged_digest("bool", [b"1" if value else b"0"])
    if isinstance(value, int):
        return _tagged_digest("int", [str(value).encode()])
    if isinstance(value, float):
        spelling = value.hex() if math.isfinite(value) else str(value)
        return _tagged_digest("float", [spelling.encode()])
    if isinstance(value, str):
        return _tagged_digest("str", [value.encode()])
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _tagged_digest("bytes", [bytes(value)])
    if isinstance(value, os.PathLike):
        path = os.path.realpath(os.path.expanduser(os.fsdecode(os.fspath(value))))
        if not os.path.isfile(path):
            raise ValueError(f"validation media path must name a readable file, got {path!r}")
        digest = hashlib.sha256()
        with open(path, "rb") as media_file:
            for chunk in iter(lambda: media_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return _tagged_digest("path-content", [digest.digest()])
    if isinstance(value, Enum):
        type_name = f"{type(value).__module__}.{type(value).__qualname__}".encode()
        return _tagged_digest("enum", [type_name, stable_value_digest(value.value, active)])

    identity = id(value)
    active = set() if active is None else active
    if identity in active:
        raise ValueError(f"validation row contains a recursive {type(value).__name__} value")

    module_name = type(value).__module__
    if module_name == "torch" or module_name.startswith("torch."):
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"unsupported torch validation value {type(value).__qualname__}")
        tensor = value.detach().resolve_conj().resolve_neg()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()  # [*shape]
        flat = tensor.cpu().contiguous().reshape(-1)  # [N]
        raw = flat.view(torch.uint8).numpy().tobytes()  # [N * bytes_per_element]
        return _tagged_digest(
            "torch.Tensor",
            [str(value.dtype).encode(), str(tuple(value.shape)).encode(), raw],
        )

    if module_name == "numpy" or module_name.startswith("numpy."):
        import numpy as np

        array = np.asarray(value)
        if array.dtype.hasobject:
            return _tagged_digest("numpy.object", [stable_value_digest(array.tolist(), active)])
        contiguous = np.ascontiguousarray(array)  # [*shape]
        return _tagged_digest(
            "numpy.ndarray",
            [str(contiguous.dtype).encode(), str(tuple(contiguous.shape)).encode(), contiguous.tobytes()],
        )

    if module_name == "PIL" or module_name.startswith("PIL."):
        mode = getattr(value, "mode", None)
        size = getattr(value, "size", None)
        tobytes = getattr(value, "tobytes", None)
        if not isinstance(mode, str) or not isinstance(size, tuple) or not callable(tobytes):
            raise TypeError(f"unsupported PIL validation value {type(value).__qualname__}")
        # Palette indices are not visual content; canonical RGBA includes palette and transparency.
        canonical = value.convert("RGBA")
        return _tagged_digest("PIL.Image", [b"RGBA", str(canonical.size).encode(), bytes(canonical.tobytes())])

    active.add(identity)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            type_name = f"{type(value).__module__}.{type(value).__qualname__}".encode()
            parts = [type_name]
            for field in dataclasses.fields(value):
                parts.extend((field.name.encode(), stable_value_digest(getattr(value, field.name), active)))
            return _tagged_digest("dataclass", parts)
        if isinstance(value, Mapping):
            pairs = sorted(
                (stable_value_digest(key, active), stable_value_digest(item, active)) for key, item in value.items()
            )
            return _tagged_digest("mapping", [part for pair in pairs for part in pair])
        if isinstance(value, tuple):
            return _tagged_digest("tuple", [stable_value_digest(item, active) for item in value])
        if isinstance(value, list):
            return _tagged_digest("list", [stable_value_digest(item, active) for item in value])
        if isinstance(value, Set):
            return _tagged_digest("set", sorted(stable_value_digest(item, active) for item in value))
    finally:
        active.remove(identity)
    raise TypeError(
        "validation rows must contain stable primitives, paths, PIL images, tensors, NumPy arrays, "
        f"dataclasses, or containers; got opaque {type(value).__module__}.{type(value).__qualname__}"
    )


def dataset_fingerprint(dataset: Any) -> str:
    """Return one stable content identity for a finite map-style source."""
    fingerprint = getattr(dataset, "cursor_fingerprint", None)
    if callable(fingerprint):
        return str(fingerprint())
    digest = hashlib.sha256()
    digest.update(f"type={type(dataset).__module__}.{type(dataset).__qualname__}|len={len(dataset)}|".encode())
    for index in range(len(dataset)):
        encoded = stable_value_digest(dataset[index])
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return f"sha256={digest.hexdigest()}|len={len(dataset)}"
