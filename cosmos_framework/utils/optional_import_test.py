# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
import json
import subprocess
import sys

import pytest

from cosmos_framework.utils.optional_import import MissingModule, optional_import

_HAS_TRAINING_DEPS = importlib.util.find_spec("transformers") is not None


def test_optional_import_returns_the_installed_module() -> None:
    assert optional_import("json") is json


def test_missing_module_raises_on_first_use_and_names_the_module() -> None:
    missing = optional_import("cosmos_framework_no_such_module_for_test")
    assert isinstance(missing, MissingModule)
    assert not missing
    with pytest.raises(ModuleNotFoundError, match="cosmos_framework_no_such_module_for_test"):
        missing.anything


def _import_without_pyarrow(module: str, *extra: str) -> None:
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
    subprocess.run([sys.executable, "-c", code], check=True, timeout=900)


def test_contract_module_imports_without_pyarrow_and_fails_loudly_on_use() -> None:
    module = "cosmos_framework.data.generator.multiview.camera_geometry_contract"
    _import_without_pyarrow(
        module,
        f"m = sys.modules['{module}']",
        "assert m.CAMERA_SIDECAR_SCHEMA_VERSION_V1_1 == b'1.1.0'",
        "raised = False",
        "exec('try:\\n    m.POSE_SCHEMA\\nexcept ModuleNotFoundError:\\n    raised = True')",
        "assert raised, 'schema access must raise ModuleNotFoundError without pyarrow'",
    )


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
    _import_without_pyarrow(module)


def test_schemas_build_when_pyarrow_is_present() -> None:
    pytest.importorskip("pyarrow")
    from cosmos_framework.data.generator.multiview import camera_geometry_contract as contract

    assert contract.POSE_SCHEMA.names[:2] == ["stream_id", "sample_index"]
    assert contract.CAMERA_CALIBRATION_SCHEMA_V1_1.metadata[b"schema_version"] == b"1.1.0"
    assert contract.POSE_SCHEMA is contract.POSE_SCHEMA  # cached, identity comparisons keep working
