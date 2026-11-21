# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build the wheel without runtime dependencies and check its tracked package inventory."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def wheel_project() -> Iterator[Path]:
    # Hatch 1.29 disables VCS patterns when the absolute project path matches an
    # ignore rule. The repository ignores /tmp, so stage beside this checkout.
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix=".cosmos-wheel-inventory-", dir=root.parent) as directory:
        yield Path(directory)


def test_wheel_contains_tracked_runtime_modules_without_ignored_files(tmp_path: Path, wheel_project: Path) -> None:
    pytest.importorskip("hatchling", reason="wheel inventory requires the declared Hatchling build dependency")
    root = Path(__file__).resolve().parents[1]
    tracked = {
        path
        for path in subprocess.check_output(["git", "ls-files", "-z", "cosmos_framework"], cwd=root)
        .decode()
        .split("\0")
        if path
    }
    assert "cosmos_framework/posttrain/roles/env/sft.py" in tracked
    # Copy tracked files only, so local data or an active developer environment
    # cannot leak into the test artifact. Read current bytes to test pending edits.
    project = wheel_project
    for relative in sorted(tracked | {"pyproject.toml", ".gitignore"}):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    ignored = {
        "cosmos_framework/posttrain/roles/env/private.json",
        "cosmos_framework/posttrain/roles/env/.env",
        "cosmos_framework/posttrain/roles/env/__pycache__/sft.cpython-312.pyc",
        "cosmos_framework/posttrain/roles/env/nested/credentials.py",
        "cosmos_framework/posttrain/roles/env/venv/bin/activate",
        "cosmos_framework/another_scope/env/private.py",
        "cosmos_framework/__pycache__/private.cpython-312.pyc",
        "cosmos_framework/posttrain/roles/env/inventory_probe_test.py",
    }
    for relative in ignored:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("packaging exclusion sentinel\n", encoding="utf-8")
    wheels = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(wheels)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    built = list(wheels.glob("*.whl"))
    assert len(built) == 1
    # The existing wheel contract intentionally excludes *_test.py everywhere.
    expected = {path for path in tracked if not path.endswith("_test.py")}
    with zipfile.ZipFile(built[0]) as wheel:
        names = set(wheel.namelist())
        actual = {path for path in names if path.startswith("cosmos_framework/")}
        assert expected - actual == set(), f"tracked runtime files omitted from wheel: {sorted(expected - actual)}"
        assert actual - expected == set(), f"untracked/ignored package files included: {sorted(actual - expected)}"
        assert not names.intersection(ignored)
        for relative in expected:
            assert wheel.read(relative) == (root / relative).read_bytes(), f"wheel changed source bytes: {relative}"
