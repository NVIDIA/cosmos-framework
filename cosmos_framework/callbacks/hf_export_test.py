# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hermetic tests for HF export worker error propagation."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.fixture
def hf_export_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "cosmos_framework.callbacks.hf_export"

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float64 = "float64"
    fake_torch.dtype = object
    fake_torch.Tensor = object
    fake_torch.distributed = types.SimpleNamespace(tensor=types.SimpleNamespace(DTensor=type("DTensor", (), {})))

    fake_log = types.ModuleType("cosmos_framework.utils.log")
    fake_log.error = lambda *args, **kwargs: None
    fake_log.info = lambda *args, **kwargs: None
    fake_log.warning = lambda *args, **kwargs: None

    fake_callback = types.ModuleType("cosmos_framework.utils.callback")
    fake_callback.Callback = type("Callback", (), {})

    fake_distributed = types.ModuleType("cosmos_framework.utils.distributed")
    fake_distributed.is_rank0 = lambda: True

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.log", fake_log)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.callback", fake_callback)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.distributed", fake_distributed)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)


def test_save_and_upload_stores_worker_exception_when_export_fails(hf_export_module) -> None:
    callback = hf_export_module.HFExportCallback()

    def _raise(*args, **kwargs) -> None:
        raise RuntimeError("worker failed")

    callback._do_save_and_upload = _raise

    callback._save_and_upload([], {}, 0, None, "model", "/tmp/export", 12)

    assert isinstance(callback._worker_exception, RuntimeError)
    assert str(callback._worker_exception) == "worker failed"
