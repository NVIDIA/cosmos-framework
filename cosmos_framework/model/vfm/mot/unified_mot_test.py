# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression tests for package-relative model-config resolution.

The shipped config defaults (cosmos_framework/configs/base/defaults/vlm.py)
reference the per-architecture JSONs by a repo-root-relative path, e.g.
``"cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"``.
``_MoTConfigBase.from_json_file`` used to ``open()`` that string verbatim, so it
only resolved when the process CWD was the framework repo root. Launching
``cosmos_framework.scripts.train`` from any other directory (e.g. a cookbook
folder) raised ``FileNotFoundError`` during model construction. These tests pin
the package-root fallback that fixes it.
"""

from pathlib import Path

import cosmos_framework
from cosmos_framework.model.vfm.mot.unified_mot import (
    Qwen3VLMoTConfig,
    _PACKAGE_ROOT,
    _resolve_packaged_config_path,
)

# A config JSON that ships inside the package, named exactly as the config
# defaults reference it (relative to the package root).
_SHIPPED_REL = "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"


def test_package_root_contains_cosmos_framework():
    # _PACKAGE_ROOT must be the directory that *contains* the cosmos_framework
    # package, so that "<root>/cosmos_framework/..." resolves in both editable
    # and wheel installs.
    assert (_PACKAGE_ROOT / "cosmos_framework").is_dir()
    assert Path(cosmos_framework.__file__).resolve().parent == _PACKAGE_ROOT / "cosmos_framework"


def test_resolve_absolute_path_passes_through(tmp_path):
    abs_path = tmp_path / "cfg.json"
    abs_path.write_text("{}")
    assert _resolve_packaged_config_path(str(abs_path)) == str(abs_path)


def test_resolve_existing_relative_path_passes_through(tmp_path, monkeypatch):
    # A relative path that exists against the CWD is returned unchanged — the
    # package-root fallback must not shadow a real working-directory file.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.json").write_text("{}")
    assert _resolve_packaged_config_path("local.json") == "local.json"


def test_resolve_shipped_config_from_foreign_cwd(tmp_path, monkeypatch):
    # The actual regression: from a directory that is not the repo root, the
    # shipped repo-root-relative path still resolves to the packaged file.
    monkeypatch.chdir(tmp_path)
    resolved = Path(_resolve_packaged_config_path(_SHIPPED_REL))
    assert resolved.is_absolute()
    assert resolved.is_file()
    assert resolved == _PACKAGE_ROOT / _SHIPPED_REL


def test_from_json_file_loads_shipped_config_from_foreign_cwd(tmp_path, monkeypatch):
    # End-to-end: from_json_file (the call made during model construction) loads
    # the shipped config from a foreign CWD instead of raising FileNotFoundError.
    monkeypatch.chdir(tmp_path)
    config = Qwen3VLMoTConfig.from_json_file(_SHIPPED_REL)
    assert isinstance(config.config_dict, dict)
    assert config.config_dict.get("model_type")  # sanity: a real Qwen3-VL config
