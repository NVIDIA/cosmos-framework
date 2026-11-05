# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU tests for strict, schema-versioned Manifest document handling."""

from __future__ import annotations

import json

import pytest

from cosmos_framework.posttrain.control.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Edge,
    Manifest,
    ManifestSchemaError,
    Plane,
    RoleDecl,
    manifest_document,
    manifest_from_document,
    role_by_name,
)
from cosmos_framework.posttrain.utils.entry import render_manifest_validation


@pytest.mark.L0
@pytest.mark.CPU
def test_role_by_name() -> None:
    m = Manifest(roles=(RoleDecl("data", "dpo_env"),), edges=())
    found = role_by_name(m, "data")
    assert found is not None and found.impl_key == "dpo_env"
    assert role_by_name(m, "missing") is None


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_document_is_schema_versioned_and_jsonable() -> None:
    m = Manifest(
        roles=(
            RoleDecl("data", "dpo_env"),
            RoleDecl("policy", "dpo_trainer"),
        ),
        edges=(Edge("data", "policy"),),
    )
    doc = manifest_document(m)
    assert doc["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert doc["roles"] == [{"name": "data", "impl_key": "dpo_env"}, {"name": "policy", "impl_key": "dpo_trainer"}]
    assert doc["edges"] == [{"src": "data", "dst": "policy", "plane": "data"}]
    json.dumps(doc)  # must be JSON-serializable for run-log persistence


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_document_round_trip_preserves_optional_edge_fields() -> None:
    manifest = Manifest(
        roles=(RoleDecl("policy", "grpo_trainer"), RoleDecl("rollout", "grpo_rollout")),
        edges=(
            Edge(
                "policy",
                "rollout",
                Plane.WEIGHT,
                relationship="same_gpu",
                transport_capability="tensor",
            ),
            Edge("rollout", "policy", Plane.DATA, transport_capability="object"),
        ),
    )

    rendered, _ = render_manifest_validation(manifest)
    persisted_record = json.loads(rendered.splitlines()[0])

    assert manifest_from_document(persisted_record["manifest"]) == manifest


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_loader_rejects_invalid_or_unsupported_versions() -> None:
    for schema_version in (True, 1.0, "1", 0, MANIFEST_SCHEMA_VERSION + 1):
        document = {"schema_version": schema_version, "roles": [], "edges": []}

        with pytest.raises(ManifestSchemaError, match="schema_version"):
            manifest_from_document(document)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_loader_rejects_unknown_or_malformed_fields() -> None:
    cases: list[tuple[object, str]] = [
        ([], "manifest must be an object"),
        ({"schema_version": 1, "roles": []}, "missing=\\['edges'\\]"),
        ({"schema_version": 1, "roles": [], "edges": [], "typo": 1}, "unknown=\\['typo'\\]"),
        (
            {"schema_version": 1, "roles": [{"name": "data", "impl_key": "env", "typo": 1}], "edges": []},
            "manifest.roles\\[0\\] fields differ",
        ),
        (
            {"schema_version": 1, "roles": [{"name": "data", "impl_key": "env"}], "edges": "data-policy"},
            "manifest.edges must be an array",
        ),
        (
            {"schema_version": 1, "roles": [{"name": " ", "impl_key": "env"}], "edges": []},
            "manifest.roles\\[0\\].name must be a non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "roles": [{"name": "data", "impl_key": "env"}, {"name": "policy", "impl_key": "trainer"}],
                "edges": [{"src": 1, "dst": "policy", "plane": "data"}],
            },
            "manifest.edges\\[0\\].src must be a non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "roles": [{"name": "data", "impl_key": "env"}, {"name": "policy", "impl_key": "trainer"}],
                "edges": [{"src": "data", "dst": "policy", "plane": "data", "relationship": None}],
            },
            "relationship must be a non-empty string when present",
        ),
    ]
    for document, message in cases:
        with pytest.raises(ManifestSchemaError, match=message):
            manifest_from_document(document)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_loader_rejects_duplicate_roles_and_edges() -> None:
    duplicate_roles = {
        "schema_version": 1,
        "roles": [{"name": "policy", "impl_key": "trainer"}, {"name": "policy", "impl_key": "rollout"}],
        "edges": [],
    }
    duplicate_edges = {
        "schema_version": 1,
        "roles": [{"name": "policy", "impl_key": "trainer"}, {"name": "rollout", "impl_key": "rollout"}],
        "edges": [
            {"src": "policy", "dst": "rollout", "plane": "weight"},
            {"src": "policy", "dst": "rollout", "plane": "weight", "relationship": "same_gpu"},
        ],
    }

    with pytest.raises(ManifestSchemaError, match="duplicate names"):
        manifest_from_document(duplicate_roles)
    with pytest.raises(ManifestSchemaError, match="duplicates edge"):
        manifest_from_document(duplicate_edges)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_loader_rejects_unknown_plane_discriminator() -> None:
    document = {
        "schema_version": 1,
        "roles": [{"name": "data", "impl_key": "env"}, {"name": "policy", "impl_key": "trainer"}],
        "edges": [{"src": "data", "dst": "policy", "plane": "telepathy"}],
    }

    with pytest.raises(ManifestSchemaError, match="must be one of"):
        manifest_from_document(document)
