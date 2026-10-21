# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Inspectable RL topology Manifest — the declared roles and edges a recipe wires.

Pure data (torch/ray-free): `bind_check` reads a Manifest to validate wiring and per-edge fields
before any GPU or cluster work. Distinct from `bind_device_groups`, which places a live topology
on a Ray cluster.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

# Version of the persisted Manifest document. bind_check is preflight-only; the version lets a
# persisted manifest be read back by later wiring/solver work. Bump on any breaking schema change.
MANIFEST_SCHEMA_VERSION = 1


class ManifestSchemaError(ValueError):
    """A persisted Manifest document does not satisfy a supported schema."""


class Plane(str, Enum):
    DATA = "data"
    WEIGHT = "weight"
    CONTROL = "control"
    CREDIT = "credit"


@dataclass(frozen=True)
class RoleDecl:
    """A declared role and the implementation (`impl_key`) that fills it."""

    name: str
    impl_key: str


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    plane: Plane = Plane.DATA
    # Transport tags for the AOT transport check (paired). `relationship` is the placement relationship
    # (same_process | same_gpu | host | cross_device); `transport_capability` is the payload capability
    # ('object' | 'tensor'). Either left None ⇒ transport is checked only at runtime mover construction.
    relationship: Optional[str] = None
    transport_capability: Optional[str] = None


@dataclass(frozen=True)
class Manifest:
    roles: tuple[RoleDecl, ...]
    edges: tuple[Edge, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION


def role_by_name(manifest: Manifest, name: str) -> Optional[RoleDecl]:
    for role in manifest.roles:
        if role.name == name:
            return role
    return None


def _edge_document(edge: Edge) -> dict[str, object]:
    doc: dict[str, object] = {"src": edge.src, "dst": edge.dst, "plane": edge.plane.value}
    if edge.relationship is not None:
        doc["relationship"] = edge.relationship
    if edge.transport_capability is not None:
        doc["transport_capability"] = edge.transport_capability
    return doc


def manifest_document(manifest: Manifest) -> dict[str, object]:
    """Serialize a Manifest to a plain JSON-able dict (schema_version + roles + edges) for the
    pre-flight run log — the persisted form later wiring/solver work reads back."""
    return {
        "schema_version": manifest.schema_version,
        "roles": [{"name": r.name, "impl_key": r.impl_key} for r in manifest.roles],
        "edges": [_edge_document(e) for e in manifest.edges],
    }


def _require_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestSchemaError(f"{where} must be an object, got {type(value).__name__}")
    if any(not isinstance(key, str) for key in value):
        raise ManifestSchemaError(f"{where} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_exact_keys(document: Mapping[str, object], expected: set[str], where: str) -> None:
    actual = set(document)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    raise ManifestSchemaError(f"{where} fields differ from schema: missing={missing}, unknown={unknown}")


def _require_nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestSchemaError(f"{where} must be a non-empty string")
    return value


def _require_optional_string(document: Mapping[str, object], key: str, where: str) -> str | None:
    if key not in document:
        return None
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestSchemaError(f"{where}.{key} must be a non-empty string when present")
    return value


def _validate_loaded_manifest(manifest: Manifest) -> None:
    """Enforce cross-entry invariants after individual v1 fields have parsed."""

    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for role in manifest.roles:
        if role.name in seen_names:
            duplicate_names.add(role.name)
        seen_names.add(role.name)
    if duplicate_names:
        raise ManifestSchemaError(f"manifest.roles contains duplicate names: {sorted(duplicate_names)}")

    seen_edges: set[tuple[str, str, Plane]] = set()
    for index, edge in enumerate(manifest.edges):
        edge_key = (edge.src, edge.dst, edge.plane)
        if edge_key in seen_edges:
            raise ManifestSchemaError(
                f"manifest.edges[{index}] duplicates edge ({edge.src!r}, {edge.dst!r}, {edge.plane.value!r})"
            )
        seen_edges.add(edge_key)


def _role_from_document(value: object, index: int) -> RoleDecl:
    where = f"manifest.roles[{index}]"
    document = _require_mapping(value, where)
    _require_exact_keys(document, {"name", "impl_key"}, where)
    return RoleDecl(
        name=_require_nonempty_string(document["name"], f"{where}.name"),
        impl_key=_require_nonempty_string(document["impl_key"], f"{where}.impl_key"),
    )


def _edge_from_document(value: object, index: int) -> Edge:
    where = f"manifest.edges[{index}]"
    document = _require_mapping(value, where)
    required = {"src", "dst", "plane"}
    optional = {"relationship", "transport_capability"}
    actual = set(document)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        raise ManifestSchemaError(f"{where} fields differ from schema: missing={missing}, unknown={unknown}")

    plane_value = _require_nonempty_string(document["plane"], f"{where}.plane")
    try:
        plane = Plane(plane_value)
    except ValueError as error:
        raise ManifestSchemaError(
            f"{where}.plane must be one of {[candidate.value for candidate in Plane]}, got {plane_value!r}"
        ) from error
    return Edge(
        src=_require_nonempty_string(document["src"], f"{where}.src"),
        dst=_require_nonempty_string(document["dst"], f"{where}.dst"),
        plane=plane,
        relationship=_require_optional_string(document, "relationship", where),
        transport_capability=_require_optional_string(document, "transport_capability", where),
    )


def manifest_from_document(value: object) -> Manifest:
    """Load a version-1 Manifest document, rejecting ambiguous or unsupported input.

    The loader is intentionally strict: unknown fields are rejected rather than silently dropped,
    and future schema versions fail until an explicit migration is implemented and tested.
    """

    document = _require_mapping(value, "manifest")
    _require_exact_keys(document, {"schema_version", "roles", "edges"}, "manifest")

    schema_version = document["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ManifestSchemaError("manifest.schema_version must be an integer")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"unsupported manifest schema_version {schema_version}; supported={MANIFEST_SCHEMA_VERSION}"
        )

    role_values = document["roles"]
    edge_values = document["edges"]
    if not isinstance(role_values, list):
        raise ManifestSchemaError(f"manifest.roles must be an array, got {type(role_values).__name__}")
    if not isinstance(edge_values, list):
        raise ManifestSchemaError(f"manifest.edges must be an array, got {type(edge_values).__name__}")

    roles = tuple(_role_from_document(role, index) for index, role in enumerate(role_values))
    edges = tuple(_edge_from_document(edge, index) for index, edge in enumerate(edge_values))
    manifest = Manifest(roles=roles, edges=edges, schema_version=schema_version)
    _validate_loaded_manifest(manifest)
    return manifest
