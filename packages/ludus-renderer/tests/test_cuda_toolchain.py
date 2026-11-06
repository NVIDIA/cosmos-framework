# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest
from ludus_renderer import _cuda_toolchain


@pytest.mark.L0
def test_cuda_major():
    assert _cuda_toolchain._cuda_major("13.0") == 13
    assert _cuda_toolchain._cuda_major("12.8.1") == 12
    assert _cuda_toolchain._cuda_major(None) is None


@pytest.mark.L0
def test_from_root(tmp_path: Path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "bin" / "nvcc").touch()

    toolchain = _cuda_toolchain._from_root(tmp_path, "test", 13)

    assert toolchain is not None
    assert toolchain.root == tmp_path
    assert toolchain.nvcc == tmp_path / "bin" / "nvcc"
    assert toolchain.lib_dir == tmp_path / "lib"


@pytest.mark.L0
def test_validate_major_rejects_mismatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_cuda_toolchain.torch.version, "cuda", "13.0")

    with pytest.raises(RuntimeError, match="does not match PyTorch CUDA"):
        _cuda_toolchain._validate_major(12)


@pytest.mark.L0
def test_distribution_root_accepts_shared_nvidia_layout():
    root = _cuda_toolchain._distribution_root("nvidia-nvjpeg")

    if root is not None:
        assert root.name.startswith("cu")
        assert (root / "include" / "nvjpeg.h").is_file()


@pytest.mark.L0
def test_find_nvidia_library_falls_back_to_toolkit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    include_dir = tmp_path / "include"
    lib_dir = tmp_path / "lib"
    bin_dir = tmp_path / "bin"
    include_dir.mkdir()
    lib_dir.mkdir()
    bin_dir.mkdir()
    (include_dir / "nvjpeg.h").touch()
    (lib_dir / "libnvjpeg.so.13").touch()
    nvcc = bin_dir / "nvcc"
    nvcc.touch()
    toolchain = _cuda_toolchain.CudaToolchain(tmp_path, include_dir, lib_dir, nvcc, "test", 13)
    monkeypatch.setattr(_cuda_toolchain, "_distribution_root", lambda _: None)

    actual_include, actual_lib, soname = _cuda_toolchain.find_nvidia_library(
        "missing-package",
        "nvjpeg.h",
        ("libnvjpeg.so.13",),
        toolchain,
    )

    assert actual_include == include_dir
    assert actual_lib == lib_dir
    assert soname == "libnvjpeg.so.13"
