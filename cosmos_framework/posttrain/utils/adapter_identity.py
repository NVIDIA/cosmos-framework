# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Portable processor identity for the verified native adapter path only.

The ordinary packer identity remains unchanged: exact training continuation still
binds its original paths. Portable adapter reload replaces only known tokenizer
artifact references with verified source content IDs, preserving all numerical
processor, template, tokenizer and masking settings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.utils.adapter_source import verify_adapter_snapshot
from cosmos_framework.posttrain.utils.identity import (
    canonical_config_value,
    canonical_json_sha256,
    json_identity_equal,
    normalize_json_dict,
    project_typed_json_value,
)

_SOURCE_FIELDS = frozenset({"name_or_path", "_name_or_path"})
# Deliberately explicit: a future tokenizer path field needs its own review.
_ARTIFACT_FIELDS = {
    "vocab_file": frozenset({"vocab.json"}),
    "merges_file": frozenset({"merges.txt"}),
    "tokenizer_file": frozenset({"tokenizer.json"}),
    "tokenizer_config_file": frozenset({"tokenizer_config.json"}),
    "special_tokens_map_file": frozenset({"special_tokens_map.json"}),
    "added_tokens_file": frozenset({"added_tokens.json"}),
    "chat_template_file": frozenset({"chat_template.json", "chat_template.jinja"}),
}


def _path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValueError(f"tokenizer {field} must be a nonempty local artifact path")
    text = os.fspath(value)
    if not isinstance(text, str) or "://" in text:
        raise ValueError(f"tokenizer {field} must identify a local verified artifact")
    return Path(text).expanduser()


def _reject_unknown_references(value: Any, *, context: str) -> None:
    """Reject unsupported path-bearing keys without rewriting semantic strings."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                isinstance(key, str)
                and item is not None
                and (
                    key in _SOURCE_FIELDS or key.endswith(("_file", "_files", "_path", "_paths", "_dir", "_directory"))
                )
            ):
                raise ValueError(f"unsupported tokenizer artifact reference {context}.{key}")
            _reject_unknown_references(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unknown_references(item, context=f"{context}[{index}]")


def adapter_processor_identity(
    packer: Any,
    *,
    source_path: str | os.PathLike[str],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize known tokenizer paths only after proving the supplied snapshot.

    Source verification is content-based and cached by inode/size/mtime/ctime by
    ``verify_adapter_snapshot``. Call after the existing source preflight and
    packer construction; neither this helper nor verification downloads anything.
    External paths, symlink escapes, absent files and unmanifested references fail.
    ``AddedToken`` and typed mapping-key semantics use the ordinary identity
    projector, exactly as the native packer does.
    """
    root = Path(source_path).expanduser().resolve()
    verified = verify_adapter_snapshot(root)
    supplied = normalize_json_dict(canonical_config_value(source_identity), context="adapter verified source identity")
    if not json_identity_equal(verified, supplied, context="adapter verified source identity"):
        raise ValueError("adapter processor source identity differs from the verified snapshot")
    source_ref = {
        "repository": verified["repository"],
        "revision": verified["revision"],
        "source_identity_sha256": canonical_json_sha256(verified, context="adapter verified source identity"),
    }

    def source_reference(value: Any, field: str) -> dict[str, Any]:
        if _path(value, field=field).resolve() != root:
            raise ValueError(f"tokenizer {field} differs from its verified source directory")
        return dict(source_ref)

    def artifact_reference(value: Any, field: str) -> dict[str, Any] | None:
        if value is None:
            return None
        path = _path(value, field=field)
        # HF normally supplies absolute cached paths; a bare artifact filename
        # is interpreted against the explicitly supplied snapshot, never cwd.
        if not path.is_absolute():
            path = root / path if len(path.parts) == 1 else path.resolve()
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(root))
        except ValueError as error:
            raise ValueError(f"tokenizer {field} references an external artifact") from error
        record = verified["files"].get(name)
        if record is None or not resolved.is_file():
            raise ValueError(f"tokenizer {field} references an absent or unmanifested artifact: {name}")
        if name not in _ARTIFACT_FIELDS[field]:
            raise ValueError(f"tokenizer {field} references an incompatible artifact: {name}")
        return {"source": dict(source_ref), "file": name, **record}

    identity = normalize_json_dict(packer.validation_semantics_identity(), context="native packer identity")
    tokenizer = packer.tokenizer
    token_identity = identity.get("tokenizer")
    if not isinstance(token_identity, dict) or not isinstance(token_identity.get("settings"), dict):
        raise ValueError("portable adapter processor requires the native tokenizer identity contract")
    if "name_or_path" not in token_identity["settings"]:
        raise ValueError("portable adapter processor requires tokenizer.name_or_path")
    token_identity["settings"]["name_or_path"] = source_reference(tokenizer.name_or_path, "name_or_path")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if init_kwargs is None:
        normalized = None
    elif not isinstance(init_kwargs, dict):
        raise TypeError("portable adapter processor requires tokenizer.init_kwargs to be a dict or None")
    else:
        normalized = dict(init_kwargs)
        for field in normalized:
            if field in _SOURCE_FIELDS:
                normalized[field] = source_reference(normalized[field], field)
            elif field in _ARTIFACT_FIELDS:
                normalized[field] = artifact_reference(normalized[field], field)
            else:
                _reject_unknown_references({field: normalized[field]}, context="tokenizer.init_kwargs")
        normalized = project_typed_json_value(normalized, context="portable tokenizer.init_kwargs")
    token_identity["init_kwargs_sha256"] = (
        None if normalized is None else canonical_json_sha256(normalized, context="portable tokenizer.init_kwargs")
    )
    identity["adapter_processor_identity"] = {
        "protocol": "cosmos.adapter-processor.v1",
        "source": source_ref,
    }
    return normalize_json_dict(identity, context="portable adapter processor identity")
