# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline regression coverage for the separately locked HF runtime helper."""

import importlib.resources
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import tomllib


def test_packaged_hf_cli_project_has_its_matching_lock() -> None:
    # Loading a tokenizer needs this package resource even when weights are cached.
    helper = importlib.resources.files("cosmos_framework").joinpath("utils", "hf_cli")
    project = tomllib.loads(helper.joinpath("pyproject.toml").read_text())
    lock = tomllib.loads(helper.joinpath("uv.lock").read_text())
    pin = next(dep for dep in project["project"]["dependencies"] if dep.startswith("hf=="))
    locked = {item["name"]: item["version"] for item in lock["package"]}

    assert locked["hf"] == pin.partition("==")[2]
    assert project["tool"]["uv"]["package"] is False


@pytest.mark.skipif(
    not os.environ.get("IMAGINAIRE_HF_CLI_ENVIRONMENT") or shutil.which("uv") is None,
    reason="Requires the dedicated installed HF CLI environment for an offline subprocess check",
)
def test_real_hf_subprocess_resolves_a_cached_file_offline(tmp_path: Path) -> None:
    from cosmos_framework.utils.checkpoint_db import _hf_download

    # Exercise the actual packaged helper against a fabricated cached snapshot.
    # No model weights or network access are needed.
    revision = "0" * 40
    cache = tmp_path / "hub"
    snapshot = cache / "models--cosmos-test--cached-resource" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    target = snapshot / "config.json"
    target.write_text(json.dumps({"test": "offline packaged hf helper"}))
    environment = {
        "HF_HOME": str(tmp_path),
        "HF_HUB_CACHE": str(cache),
        "HUGGINGFACE_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1",
        "UV_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # Do not inject the test runner's extra site-packages into the helper.
        "PYTHONPATH": "",
    }

    with patch.dict(os.environ, environment):
        resolved = _hf_download(
            ["cosmos-test/cached-resource", "--repo-type", "model", "--revision", revision, "config.json"]
        )

    assert Path(resolved).resolve() == target.resolve()
    assert json.loads(Path(resolved).read_text())["test"] == "offline packaged hf helper"
