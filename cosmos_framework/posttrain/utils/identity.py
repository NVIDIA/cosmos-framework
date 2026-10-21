# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical finite-JSON identities used by validation and exact resume."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
from collections.abc import Mapping
from typing import Any

import attrs
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _qualified_symbol(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError(f"cannot identify callable config value {value!r}")
    return f"{module}.{qualname}"


def canonical_config_value(value: Any) -> Any:
    """Project typed construction config into deterministic JSON-compatible data."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"config contains non-finite float {value!r}")
        return value
    if isinstance(value, enum.Enum):
        return {
            "__enum__": _qualified_symbol(type(value)),
            "value": canonical_config_value(value.value),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return {"__torch_value__": str(value)}
    if isinstance(value, os.PathLike):
        return {"__path__": os.fspath(value)}
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, (DictConfig, ListConfig)):
        return canonical_config_value(OmegaConf.to_container(value, resolve=True))
    if attrs.has(type(value)):
        return {
            "__attrs_type__": _qualified_symbol(type(value)),
            "fields": {
                field.name: canonical_config_value(getattr(value, field.name)) for field in attrs.fields(type(value))
            },
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass_type__": _qualified_symbol(type(value)),
            "fields": {
                field.name: canonical_config_value(getattr(value, field.name)) for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("config mapping keys must all be strings")
        return {key: canonical_config_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_config_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_config_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, type) or callable(value):
        return {"__callable__": _qualified_symbol(value)}
    raise TypeError(f"config contains unsupported value of type {type(value).__module__}.{type(value).__qualname__}")


def qualified_type(value: Any) -> str:
    """Return the import-qualified concrete type of ``value``."""
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _canonical_mapping_key(key: Any, *, context: str) -> str:
    """Encode supported mapping keys without cross-type aliases."""
    if isinstance(key, str):
        tagged_key: tuple[str, Any] = ("str", key)
    elif isinstance(key, bool):
        tagged_key = ("bool", key)
    elif isinstance(key, int):
        tagged_key = ("int", key)
    elif isinstance(key, float):
        if not math.isfinite(key):
            raise TypeError(f"{context} contains non-finite float mapping key {key!r}")
        tagged_key = ("float", key.hex())
    elif key is None:
        tagged_key = ("none", None)
    else:
        raise TypeError(f"{context} contains unsupported mapping key type {qualified_type(key)}")
    return json.dumps(tagged_key, separators=(",", ":"), allow_nan=False)


def project_typed_json_value(value: Any, *, context: str, active: set[int] | None = None) -> Any:
    """Project supported typed runtime metadata into finite JSON."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{context} contains non-finite float {value!r}")
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}

    active = set() if active is None else active
    object_id = id(value)
    if object_id in active:
        raise TypeError(f"{context} contains a recursive value")
    active.add(object_id)
    try:
        if isinstance(value, dict):
            projected: dict[str, Any] = {}
            for key, item in value.items():
                canonical_key = _canonical_mapping_key(key, context=context)
                if canonical_key in projected:
                    raise TypeError(f"{context} contains a canonical mapping-key collision at {canonical_key!r}")
                projected[canonical_key] = project_typed_json_value(
                    item,
                    context=f"{context}[{canonical_key}]",
                    active=active,
                )
            return {key: projected[key] for key in sorted(projected)}
        if isinstance(value, (list, tuple)):
            return [
                project_typed_json_value(item, context=f"{context}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        # Hugging Face and tokenizers AddedToken objects share this stable surface.
        if hasattr(value, "content") and all(
            hasattr(value, field) for field in ("single_word", "lstrip", "rstrip", "normalized", "special")
        ):
            return {
                "type": qualified_type(value),
                "content": str(value.content),
                "single_word": bool(value.single_word),
                "lstrip": bool(value.lstrip),
                "rstrip": bool(value.rstrip),
                "normalized": bool(value.normalized),
                "special": bool(value.special),
            }
    finally:
        active.remove(object_id)
    raise TypeError(f"{context} contains unsupported value type {qualified_type(value)}")


def normalize_typed_json_dict(value: Any, *, context: str) -> dict[str, Any]:
    """Project a typed runtime mapping into a canonical finite-JSON dictionary."""
    projected = project_typed_json_value(value, context=context)
    if not isinstance(projected, dict):
        raise TypeError(f"{context} must project to a dictionary, got {type(projected).__name__}")
    return normalize_json_dict(projected, context=context)


def canonical_json_bytes(value: Any, *, context: str) -> bytes:
    """Encode finite JSON without Python's bool/number equality coercions."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must contain only finite JSON-compatible values") from error


def canonical_json_sha256(value: Any, *, context: str) -> str:
    """Hash one finite JSON value using its canonical encoding."""
    return hashlib.sha256(canonical_json_bytes(value, context=context)).hexdigest()


def normalize_json_dict(identity: Any, *, context: str) -> dict[str, Any]:
    """Validate and canonicalize a finite JSON dictionary."""
    if not isinstance(identity, dict):
        raise TypeError(f"{context} must return a JSON-compatible dict, got {type(identity).__name__}")

    def require_string_keys(value: Any, value_context: str, active: set[int]) -> None:
        if not isinstance(value, (dict, list, tuple)):
            return
        object_id = id(value)
        if object_id in active:
            raise TypeError(f"{value_context} contains a recursive value")
        active.add(object_id)
        try:
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                if isinstance(value, dict) and not isinstance(key, str):
                    raise TypeError(
                        f"{value_context} contains non-string mapping key {key!r} of type {qualified_type(key)}"
                    )
                require_string_keys(item, f"{value_context}.{key}", active)
        finally:
            active.remove(object_id)

    require_string_keys(identity, context, set())
    normalized = json.loads(canonical_json_bytes(identity, context=context))
    if not isinstance(normalized, dict):
        raise RuntimeError(f"JSON normalization changed {context} away from a dict")
    return normalized


def json_identity_equal(first: Any, second: Any, *, context: str) -> bool:
    """Compare finite JSON values by canonical bytes."""
    return canonical_json_bytes(first, context=context) == canonical_json_bytes(second, context=context)
