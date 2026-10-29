# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for `utils/determinism.py`.

`enable_determinism` is the single source of truth for the determinism recipe (hoisted from
`tests/parity_cli._enable_determinism`, which `test_parity_cosmos3.py` already relies on for a
bit-identical 8B loss). These tests cover: every knob gets set, and the fail-loud guard on
`CUBLAS_WORKSPACE_CONFIG` (set-after-CUDA-init silently does nothing, so it must raise, not warn).

Mutates process-global state (`os.environ`, `torch.backends.cudnn`, `torch.use_deterministic_algorithms`)
— every test restores it via `monkeypatch`/explicit cleanup so nothing leaks into the rest of the
pytest process (a stray `use_deterministic_algorithms(True)` would make unrelated GPU tests slower or
fail in confusing ways).
"""

from __future__ import annotations

import os

import pytest
import torch

from cosmos_framework.posttrain.utils.determinism import enable_determinism


@pytest.fixture(autouse=True)
def _restore_determinism_state(monkeypatch):
    """Snapshot + restore the global knobs `enable_determinism` mutates, regardless of outcome."""
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.backends.cudnn.deterministic = prev_cudnn_deterministic
    torch.backends.cudnn.benchmark = prev_cudnn_benchmark
    torch.use_deterministic_algorithms(prev_deterministic_algorithms, warn_only=prev_warn_only)


@pytest.mark.L0
@pytest.mark.CPU
def test_enable_determinism_sets_every_knob(monkeypatch):
    # Pin CUDA as UNINITIALIZED. This test asserts the happy path, and the guard's whole job is to
    # refuse that path once CUDA is up (see the next test). On a GPU-visible box the repo conftest
    # initializes CUDA before every test — including CPU-marked ones — so without this the guard
    # correctly fires and this test fails for a reason that has nothing to do with what it asserts.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    enable_determinism()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


@pytest.mark.parametrize("strict", [False, True])
def test_explicit_strict_mode_controls_pytorch_enforcement(monkeypatch, strict):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    enable_determinism(strict=strict)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is not strict
    assert os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "1"


def test_unsupported_deterministic_operation_is_not_silently_downgraded(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    calls = []

    def unsupported(mode, *, warn_only):
        calls.append((mode, warn_only))
        raise RuntimeError("unsupported deterministic implementation")

    # Use a nested patch scope so the autouse fixture can restore Torch's real setter.
    with monkeypatch.context() as scoped:
        scoped.setattr(torch, "use_deterministic_algorithms", unsupported)
        with pytest.raises(RuntimeError, match="unsupported deterministic implementation"):
            enable_determinism(strict=True)
    assert calls == [(True, False)]


@pytest.mark.L0
@pytest.mark.CPU
def test_it_fails_loud_when_cublas_config_cannot_take_effect(monkeypatch):
    """CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is created. Setting it after CUDA is
    already initialized silently does nothing — and a silently non-deterministic run is the exact
    failure this task exists to remove. It must raise, not warn."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        enable_determinism()


@pytest.mark.L0
@pytest.mark.CPU
def test_deterministic_defaults_off():
    from cosmos_framework.posttrain.config.rl import RunConfig

    assert RunConfig().deterministic is False
