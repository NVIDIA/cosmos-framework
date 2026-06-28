# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hermetic tests for the project logging wrappers."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _CaptureLogger:
    def __init__(self, *args, events: list[dict] | None = None, **kwargs) -> None:
        self.events = [] if events is None else events
        self._options = (None, None, [], {})
        self._exception = None
        self._rank0_only = True

    def remove(self, *args, **kwargs) -> None:
        return None

    def add(self, *args, **kwargs) -> int:
        return 0

    def opt(self, *, depth=None, exception=None):
        clone = _CaptureLogger(events=self.events)
        clone._exception = exception
        clone._rank0_only = self._rank0_only
        return clone

    def bind(self, **kwargs):
        clone = _CaptureLogger(events=self.events)
        clone._exception = self._exception
        clone._rank0_only = kwargs.get("rank0_only", self._rank0_only)
        return clone

    def _record(self, level: str, message: str, *args, **kwargs) -> None:
        self.events.append(
            {
                "level": level,
                "message": message,
                "args": args,
                "kwargs": kwargs,
                "exception": self._exception,
                "rank0_only": self._rank0_only,
            }
        )

    def trace(self, message: str, *args, **kwargs) -> None:
        self._record("trace", message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        self._record("debug", message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self._record("info", message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        self._record("success", message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._record("warning", message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._record("error", message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        self._record("critical", message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        self._record("exception", message, *args, **kwargs)


@pytest.fixture
def log_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "cosmos_framework.utils.log"

    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_available = lambda: False
    fake_dist.is_initialized = lambda: False
    fake_torch.distributed = fake_dist

    fake_loguru = types.ModuleType("loguru")
    fake_loguru_logger = types.ModuleType("loguru._logger")
    fake_loguru_logger.Core = type("Core", (), {})
    fake_loguru_logger.Logger = _CaptureLogger

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "loguru", fake_loguru)
    monkeypatch.setitem(sys.modules, "loguru._logger", fake_loguru_logger)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    module.logger = _CaptureLogger()
    yield module

    sys.modules.pop(module_name, None)


def test_info_supports_percent_style_formatting(log_module) -> None:
    log_module.info("Wrote %d upsampled prompts to %s", 3, "output.json")

    assert log_module.logger.events == [
        {
            "level": "info",
            "message": "Wrote 3 upsampled prompts to output.json",
            "args": (),
            "kwargs": {},
            "exception": None,
            "rank0_only": True,
        }
    ]


def test_error_supports_exc_info_and_rank0_override(log_module) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        log_module.error(
            "[HFExportCallback] Export worker for iter %d raised an exception: %s",
            7,
            "boom",
            exc_info=True,
            rank0_only=False,
        )

    assert log_module.logger.events == [
        {
            "level": "error",
            "message": "[HFExportCallback] Export worker for iter 7 raised an exception: boom",
            "args": (),
            "kwargs": {},
            "exception": True,
            "rank0_only": False,
        }
    ]
