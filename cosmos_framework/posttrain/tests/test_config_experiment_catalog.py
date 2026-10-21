# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The inherited experiment catalog must survive an absent optional third-party dependency.

`build_root` imports the whole cosmos3 experiment catalog so posttrain experiments can inherit i4
experiments through `defaults`. One action experiment imports a robotics dataset library that is
not part of the posttrain runtime, and a module-scope `ModuleNotFoundError` there used to abort the
config load of every recipe - including text DPO and OPD, which have nothing to do with it.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import pytest

from cosmos_framework.posttrain.config.config import (
    _import_experiment_catalog,
    _missing_optional_dependency,
)


def _write_package(root: Path, name: str, modules: dict[str, str]) -> None:
    """Create an importable package whose modules have the given bodies."""
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for module_name, body in modules.items():
        (package / f"{module_name}.py").write_text(textwrap.dedent(body))


@pytest.mark.L0
@pytest.mark.CPU
def test_absent_third_party_dependency_is_skippable() -> None:
    assert _missing_optional_dependency(ModuleNotFoundError("x", name="lerobot")) == "lerobot"
    assert _missing_optional_dependency(ModuleNotFoundError("x", name="lerobot.datasets")) == "lerobot"


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_in_repo_module_is_not_skippable() -> None:
    """A broken first-party import is our bug and must not be swallowed."""
    assert _missing_optional_dependency(ModuleNotFoundError("x", name="cosmos_framework.nope")) is None
    assert _missing_optional_dependency(ModuleNotFoundError("x", name=None)) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_catalog_import_skips_only_the_module_with_the_absent_dependency(tmp_path: Path) -> None:
    _write_package(
        tmp_path,
        "catalog_ok_and_bad",
        {
            "good": "REGISTERED = True\n",
            "needs_absent_dep": "import a_library_that_is_not_installed  # noqa: F401\n",
            "also_good": "REGISTERED = True\n",
        },
    )
    sys.path.insert(0, str(tmp_path))
    try:
        skipped = _import_experiment_catalog("catalog_ok_and_bad")
        assert skipped == {"catalog_ok_and_bad.needs_absent_dep": "a_library_that_is_not_installed"}
        assert sys.modules["catalog_ok_and_bad.good"].REGISTERED is True
        assert sys.modules["catalog_ok_and_bad.also_good"].REGISTERED is True
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith("catalog_ok_and_bad")]:
            del sys.modules[name]


@pytest.mark.L0
@pytest.mark.CPU
def test_catalog_import_reraises_a_first_party_import_error(tmp_path: Path) -> None:
    _write_package(
        tmp_path,
        "catalog_first_party_bug",
        {"broken": "import cosmos_framework.posttrain.no_such_module  # noqa: F401\n"},
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ModuleNotFoundError):
            _import_experiment_catalog("catalog_first_party_bug")
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith("catalog_first_party_bug")]:
            del sys.modules[name]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("module_name", ["config_dpo", "config_opd", "config_nft"])
def test_every_root_builder_can_disable_experiment_registration(module_name: str) -> None:
    """`config_grpo` already had this; a concrete recipe needs it on every algorithm root."""
    import importlib

    module = importlib.import_module(f"cosmos_framework.posttrain.config.{module_name}")
    parameter = inspect.signature(module.make_config).parameters.get("register_experiment_configs")
    assert parameter is not None, f"{module_name}.make_config must accept register_experiment_configs"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


@pytest.mark.L0
@pytest.mark.CPU
def test_catalog_import_reraises_a_plain_import_error(tmp_path: Path) -> None:
    """Only an absent *module* is skippable. A missing name in a present module is a real bug."""
    _write_package(
        tmp_path,
        "catalog_bad_name",
        {"broken": "from cosmos_framework.posttrain import does_not_exist  # noqa: F401\n"},
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ImportError):
            _import_experiment_catalog("catalog_bad_name")
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith("catalog_bad_name")]:
            del sys.modules[name]
