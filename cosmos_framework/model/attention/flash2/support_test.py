# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
from importlib.metadata import PackageNotFoundError
from types import ModuleType
from unittest.mock import patch

import pytest

from cosmos_framework.model.attention.flash2 import flash2_supported


@pytest.fixture
def flash_attn_namespace(monkeypatch):
    """Model the namespace shared by FA2 and FA4 without requiring CUDA kernels."""
    namespace = ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", namespace)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    return namespace


def test_missing_distribution(flash_attn_namespace):
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("flash_attn")):
        assert not flash2_supported()


@pytest.mark.parametrize("version, supported", [("2.7.4.post1", True), ("2.6.0", False)])
def test_distribution_version(flash_attn_namespace, version, supported):
    with patch("importlib.metadata.version", return_value=version):
        assert flash2_supported() is supported


@pytest.mark.parametrize("version, supported", [("2.7.4.post1", True), ("2.6.0", False)])
def test_module_version(flash_attn_namespace, version, supported):
    flash_attn_namespace.__version__ = version
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("flash_attn")):
        assert flash2_supported() is supported


def test_no_cuda(flash_attn_namespace, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    flash_attn_namespace.__version__ = "2.7.4.post1"
    assert not flash2_supported()
