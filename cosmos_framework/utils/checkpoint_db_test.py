# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from cosmos_framework.utils import checkpoint_db
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf


def _hf_download_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, **fields) -> list[str]:
    """Return the arguments ``CheckpointDirHf`` hands to the hf CLI on ``platform``."""
    captured: list[list[str]] = []

    def fake_hf_download(cmd_args: list[str]) -> str:
        captured.append(cmd_args)
        return str(tmp_path)

    monkeypatch.setattr(checkpoint_db, "_hf_download", fake_hf_download)
    monkeypatch.setattr(checkpoint_db.sys, "platform", platform)
    CheckpointDirHf(repository="org/repo", revision="main", **fields).download()
    return captured[0]


def test_default_include_is_passed_as_written_off_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _hf_download_args(monkeypatch, tmp_path, "linux")
    assert args == ["org/repo", "--repo-type", "model", "--revision", "main", "--include", "*"]


def test_default_include_is_dropped_on_windows_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _hf_download_args(monkeypatch, tmp_path, "win32")
    assert args == ["org/repo", "--repo-type", "model", "--revision", "main"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_subdirectory_patterns_use_forward_slashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
) -> None:
    (tmp_path / "sub").mkdir()
    args = _hf_download_args(monkeypatch, tmp_path, platform, subdirectory="sub", exclude=("*.bin",))
    assert args[-4:] == ["--include", "sub/*", "--exclude", "sub/*.bin"]
