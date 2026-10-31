# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Host GPU results bind an environment manifest, never a fictitious image hash."""

import hashlib
import json

import pytest

from cosmos_framework.posttrain.tests.qualify_adapters import environment_identity, installed_environment


@pytest.mark.parametrize("mutation", [None, "wheel", "distribution", "python", "cuda"])
def test_environment_identity_rejects_runtime_drift(tmp_path, monkeypatch, mutation):
    wheel = tmp_path / "framework.whl"
    wheel.write_bytes(b"source-attested wheel")
    manifest = installed_environment(str(wheel))
    if mutation == "distribution":
        manifest["distributions"]["torch"] = "0.invalid"
    elif mutation == "python":
        manifest["python"] = "different interpreter"
    elif mutation == "cuda":
        manifest["cuda_runtime"] = "different CUDA"
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("ADAPTER_ENVIRONMENT_MANIFEST", str(path))
    if mutation == "wheel":
        wheel.write_bytes(b"different build")
    if mutation is not None:
        with pytest.raises(ValueError, match="differs from its sealed manifest"):
            environment_identity(str(wheel))
    else:
        result = environment_identity(str(wheel))
        assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert result["manifest"] == manifest


def test_container_workflow_does_not_require_host_manifest(monkeypatch):
    monkeypatch.delenv("ADAPTER_ENVIRONMENT_MANIFEST", raising=False)
    assert environment_identity("unused.whl") is None
