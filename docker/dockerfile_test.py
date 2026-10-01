# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import runpy
from pathlib import Path

import pytest


def test_runtime_tree_is_accessible_to_arbitrary_container_users() -> None:
    """Pyxis commonly starts the container with the submitting host UID."""

    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()
    assert "chmod a+rx /workspace /workspace/docker /workspace/docker/entrypoint.sh" in dockerfile
    assert "chmod -R a+rX /opt/cosmos /workspace" in dockerfile
    assert "test -x /workspace/docker/entrypoint.sh" in dockerfile
    assert "test -x /workspace/.venv/bin/python" in dockerfile


def test_entrypoint_never_mutates_the_packaged_python_environment() -> None:
    entrypoint = (Path(__file__).parent / "entrypoint.sh").read_text()
    assert "pip install" not in entrypoint
    assert 'exec "$@"' in entrypoint


@pytest.fixture
def source_provenance(monkeypatch):
    for key in ("SOURCE_COMMIT", "SOURCE_TREE", "SOURCE_DIRTY", "BUILD_TIMESTAMP", "REQUIRE_SOURCE_PROVENANCE"):
        monkeypatch.delenv(key, raising=False)
    return runpy.run_path(str(Path(__file__).with_name("write_image_provenance.py")))["_source_provenance"]


def test_default_build_does_not_claim_verified_source(source_provenance):
    source = source_provenance()
    assert source["commit"] is None
    assert source["tree"] is None
    assert source["dirty"] is True
    assert source["verified"] is False


@pytest.mark.parametrize("key", ["SOURCE_COMMIT", "SOURCE_TREE", "BUILD_TIMESTAMP", "REQUIRE_SOURCE_PROVENANCE"])
def test_partial_or_required_provenance_fails_closed(source_provenance, monkeypatch, key):
    monkeypatch.setenv(key, "1")
    with pytest.raises(RuntimeError, match="build arguments are required"):
        source_provenance()


@pytest.mark.parametrize("dirty", ["1", "unknown", ""])
def test_release_provenance_rejects_unverified_source(source_provenance, monkeypatch, dirty):
    for key in ("SOURCE_COMMIT", "SOURCE_TREE", "BUILD_TIMESTAMP"):
        monkeypatch.setenv(key, "supplied")
    monkeypatch.setenv("SOURCE_DIRTY", dirty)
    with pytest.raises(RuntimeError, match="verified clean source tree"):
        source_provenance()


def test_complete_clean_provenance_is_verified(source_provenance, monkeypatch):
    for key in ("SOURCE_COMMIT", "SOURCE_TREE", "BUILD_TIMESTAMP"):
        monkeypatch.setenv(key, "supplied")
    monkeypatch.setenv("SOURCE_DIRTY", "0")
    assert source_provenance()["verified"] is True
