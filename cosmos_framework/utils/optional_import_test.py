# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.utils.optional_import import MissingModule, optional_import

pytestmark = pytest.mark.L0

_HAS_TRAINING_DEPS = importlib.util.find_spec("transformers") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MISSING = "cosmos_framework_no_such_module_for_test"
_CONTRACT = "cosmos_framework.data.generator.multiview.camera_geometry_contract"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the stand-in exists on Windows only")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="checks the import error kept off Windows")


def test_optional_import_returns_the_installed_module() -> None:
    assert optional_import("json") is json


def test_a_failed_import_raises_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ModuleNotFoundError, match=_MISSING):
        optional_import(_MISSING)


def test_missing_module_raises_on_first_use_and_names_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    missing = optional_import(_MISSING)
    assert isinstance(missing, MissingModule)
    assert not missing
    with pytest.raises(AttributeError, match=_MISSING) as raised:
        missing.anything
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_missing_module_follows_the_attribute_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    missing = optional_import(_MISSING)
    assert not hasattr(missing, "anything")
    assert getattr(missing, "anything", None) is None


def _import_without_pyarrow(module: str, *extra: str) -> subprocess.CompletedProcess:
    """Import ``module`` in a fresh interpreter whose pyarrow is unimportable."""
    code = "; ".join(
        [
            "import sys",
            "sys.modules['pyarrow'] = None",
            "sys.modules['pyarrow.parquet'] = None",
            f"import {module}",
            *extra,
        ]
    )
    return subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=900)


@posix_only
def test_contract_module_still_fails_at_import_without_pyarrow_off_windows() -> None:
    result = _import_without_pyarrow(_CONTRACT)
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr and "pyarrow" in result.stderr


@windows_only
def test_contract_module_imports_without_pyarrow_and_fails_loudly_on_use() -> None:
    result = _import_without_pyarrow(
        _CONTRACT,
        f"m = sys.modules['{_CONTRACT}']",
        "assert m.CAMERA_SIDECAR_SCHEMA_VERSION_V1_1 == b'1.1.0'",
        "raised = False",
        "exec('try:\\n    m.POSE_SCHEMA\\nexcept AttributeError as error:\\n    raised = \"pyarrow\" in str(error)')",
        "assert raised, 'schema access must raise an AttributeError that names pyarrow'",
    )
    assert result.returncode == 0, result.stderr


@windows_only
@pytest.mark.skipif(not _HAS_TRAINING_DEPS, reason="needs the training dependencies")
@pytest.mark.parametrize(
    "module",
    [
        "cosmos_framework.data.generator.multiview.camera_geometry",
        "cosmos_framework.data.generator.multiview.camera_pose_actions",
        "cosmos_framework.callbacks.data_stats_writer",
        "cosmos_framework.configs.base.reasoner.config",
    ],
)
def test_reasoner_config_graph_imports_without_pyarrow(module: str) -> None:
    """pyarrow has no win_arm64 wheel; the reasoner config graph must import without it."""
    result = _import_without_pyarrow(module)
    assert result.returncode == 0, result.stderr


def test_schemas_build_when_pyarrow_is_present() -> None:
    pytest.importorskip("pyarrow")
    from cosmos_framework.data.generator.multiview import camera_geometry_contract as contract

    assert contract.POSE_SCHEMA.names[:2] == ["stream_id", "sample_index"]
    assert contract.CAMERA_CALIBRATION_SCHEMA_V1_1.metadata[b"schema_version"] == b"1.1.0"
    assert contract.POSE_SCHEMA is contract.POSE_SCHEMA  # cached, identity comparisons keep working
