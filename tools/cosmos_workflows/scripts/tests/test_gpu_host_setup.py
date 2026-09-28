# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for layered GPU host runtime requirements."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "execution/gpu-host/scripts/setup-nvidia-gpu-host.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def fake_runtime(tmp_path: Path, *, driver: str, cuda: str, toolkit: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    cuda_bin = tmp_path / "cuda" / "bin"
    bin_dir.mkdir()
    cuda_bin.mkdir(parents=True)

    write_executable(
        bin_dir / "nvidia-smi",
        """#!/bin/sh
case "$*" in
  *--query-gpu=driver_version*) printf '%s\n' "$FAKE_DRIVER_VERSION" ;;
  *-L*) printf '%s\n' 'GPU 0: Fake NVIDIA GPU' ;;
  *) printf '%s\n' 'Fake NVIDIA GPU' ;;
esac
""",
    )
    write_executable(
        cuda_bin / "nvcc",
        """#!/bin/sh
printf 'Cuda compilation tools, release %s, V%s\n' "$FAKE_CUDA_VERSION" "$FAKE_CUDA_VERSION"
""",
    )
    write_executable(
        bin_dir / "dpkg-query",
        """#!/bin/sh
printf '%s-1\n' "$FAKE_TOOLKIT_VERSION"
""",
    )
    write_executable(
        bin_dir / "docker",
        """#!/bin/sh
if [ "$1" = info ]; then
  case "$*" in *--format*) printf '%s\n' '{"nvidia":{}}' ;; esac
  exit 0
fi
exit 0
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "NVIDIA_CUDA_PATH": str(cuda_bin.parent),
            "FAKE_DRIVER_VERSION": driver,
            "FAKE_CUDA_VERSION": cuda,
            "FAKE_TOOLKIT_VERSION": toolkit,
        }
    )
    return env


def run_check(tmp_path: Path, *, driver: str, cuda: str, toolkit: str, args=()):
    return subprocess.run(
        [str(SCRIPT), "--backend", "docker", "--check-only", *args],
        check=False,
        capture_output=True,
        text=True,
        env=fake_runtime(tmp_path, driver=driver, cuda=cuda, toolkit=toolkit),
    )


def test_default_runtime_minimums_accept_newer_versions(tmp_path):
    result = run_check(tmp_path, driver="595.58.03", cuda="13.2", toolkit="1.19.1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NVIDIA driver 595.58.03 >= 580" in result.stdout
    assert "CUDA Toolkit 13.2 >= 13.0" in result.stdout
    assert "NVIDIA Container Toolkit 1.19.1 >= 1.19.0" in result.stdout


def test_model_runtime_overrides_accept_versions_at_or_above_bounds(tmp_path):
    result = run_check(
        tmp_path,
        driver="595.58.03",
        cuda="13.2",
        toolkit="1.19.1",
        args=(
            "--min-driver-version",
            "580",
            "--min-cuda-version",
            "13.0",
            "--min-container-toolkit-version",
            "1.19.1",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("driver", "cuda", "toolkit", "expected"),
    [
        ("520.61.05", "13.2", "1.19.1", "NVIDIA driver >= 580"),
        ("595.58.03", "12.9", "1.19.1", "CUDA Toolkit >= 13.0"),
        (
            "595.58.03",
            "13.2",
            "1.18.1",
            "NVIDIA Container Toolkit >= 1.19.1",
        ),
    ],
)
def test_model_runtime_overrides_reject_versions_below_bounds(tmp_path, driver, cuda, toolkit, expected):
    result = run_check(
        tmp_path,
        driver=driver,
        cuda=cuda,
        toolkit=toolkit,
        args=(
            "--min-driver-version",
            "580",
            "--min-cuda-version",
            "13.0",
            "--min-container-toolkit-version",
            "1.19.1",
        ),
    )
    assert result.returncode == 1
    assert expected in result.stdout


def test_runtime_override_rejects_non_numeric_version(tmp_path):
    result = run_check(
        tmp_path,
        driver="595.58.03",
        cuda="13.2",
        toolkit="1.19.1",
        args=("--min-driver-version", "latest"),
    )
    assert result.returncode == 2
    assert "numeric dot-separated" in result.stderr
