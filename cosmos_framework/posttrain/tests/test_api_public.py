# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU checks for the lightweight experimental public import boundary."""

from __future__ import annotations

import subprocess
import sys

import pytest

import cosmos_framework.posttrain.api as public_api
from cosmos_framework.posttrain import schema as schema_implementation
from cosmos_framework.posttrain.control import manifest as manifest_implementation


@pytest.mark.L0
@pytest.mark.CPU
def test_public_api_reexports_existing_contracts_without_wrapping() -> None:
    expected = {
        "Edge": manifest_implementation.Edge,
        "FINAL_REWARD_KEY": schema_implementation.FINAL_REWARD_KEY,
        "MANIFEST_SCHEMA_VERSION": manifest_implementation.MANIFEST_SCHEMA_VERSION,
        "Manifest": manifest_implementation.Manifest,
        "ManifestSchemaError": manifest_implementation.ManifestSchemaError,
        "ObsBundle": schema_implementation.ObsBundle,
        "Plane": manifest_implementation.Plane,
        "RoleDecl": manifest_implementation.RoleDecl,
        "Status": schema_implementation.Status,
        "Trajectory": schema_implementation.Trajectory,
        "Transition": schema_implementation.Transition,
        "manifest_document": manifest_implementation.manifest_document,
        "manifest_from_document": manifest_implementation.manifest_from_document,
    }

    assert set(public_api.__all__) == set(expected)
    assert all(getattr(public_api, name) is implementation for name, implementation in expected.items())


@pytest.mark.L0
@pytest.mark.CPU
def test_public_api_import_stays_free_of_training_dependencies() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import cosmos_framework.posttrain.api; "
        "assert 'torch' not in sys.modules; assert 'ray' not in sys.modules",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
