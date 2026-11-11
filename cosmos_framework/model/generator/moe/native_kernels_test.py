# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).with_name("native_kernels.py")


def _load_without_triton(monkeypatch: pytest.MonkeyPatch, platform: str):
    """Execute native_kernels.py as a standalone module with triton unimportable."""
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(sys, "platform", platform)
    spec = importlib.util.spec_from_file_location("_native_kernels_without_triton", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_triton_is_an_import_error_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ImportError):
        _load_without_triton(monkeypatch, "linux")


def test_windows_stub_imports_and_fails_loudly_at_kernel_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_without_triton(monkeypatch, "win32")
    assert module.TOKEN_GROUP_ALIGN_SIZE_M == 16
    # Kernels are launched as kernel[grid](...): the subscript must raise the explanation.
    with pytest.raises(RuntimeError, match="Triton is unavailable"):
        module._fill_indices_kernel[(1,)]
    with pytest.raises(RuntimeError, match="Triton is unavailable"):
        module._fill_indices_kernel()
