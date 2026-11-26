# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixtures shared by the native QLoRA tests."""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture
def restore_determinism_state(monkeypatch):
    """Restore global determinism flags and the related environment after a test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        previous = os.environ.get(name)
        # Register the original environment with monkeypatch while preserving
        # absence for enable_determinism's existing late-CUDA-init guard.
        monkeypatch.setenv(name, previous if previous is not None else "")
        if previous is None:
            del os.environ[name]
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
