# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for Qwen3Guard safety-fallback behavior."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.fixture
def qwen3guard_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "cosmos_framework.auxiliary.guardrail.qwen3guard.qwen3guard"
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    fake_core = types.ModuleType("cosmos_framework.auxiliary.guardrail.common.core")
    fake_core.ContentSafetyGuardrail = type("ContentSafetyGuardrail", (), {})
    fake_core.GuardrailRunner = type("GuardrailRunner", (), {})
    fake_log = types.ModuleType("cosmos_framework.utils.log")
    fake_log.debug = lambda *args, **kwargs: None
    fake_log.warning = lambda *args, **kwargs: None
    fake_log.error = lambda *args, **kwargs: None
    fake_misc = types.ModuleType("cosmos_framework.utils.misc")

    class _Color:
        @staticmethod
        def green(value: str) -> str:
            return value

        @staticmethod
        def red(value: str) -> str:
            return value

    fake_misc.Color = _Color

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "cosmos_framework.auxiliary.guardrail.common.core", fake_core)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.log", fake_log)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.misc", fake_misc)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)


class _DummySequence:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _DummySequence(self.values[item])
        return self.values[item]

    def tolist(self) -> list[int]:
        return list(self.values)


class _DummyInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2, 3]])
        self.input_ids = [[1, 2, 3]]

    def to(self, device: str):
        return self


class _DummyTokenizer:
    def __init__(self, moderation_output: str) -> None:
        self.moderation_output = moderation_output

    def apply_chat_template(self, messages, tokenize: bool = False) -> str:
        return "prompt"

    def __call__(self, texts, return_tensors: str = "pt") -> _DummyInputs:
        return _DummyInputs()

    def decode(self, output_ids, skip_special_tokens: bool = True) -> str:
        return self.moderation_output


class _DummyModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [_DummySequence([1, 2, 3, 4, 5])]


def test_extract_label_and_categories_blocks_unparseable_output(qwen3guard_module) -> None:
    guard = object.__new__(qwen3guard_module.Qwen3Guard)
    guard.tokenizer = _DummyTokenizer("moderation output without a recognized safety label")
    guard.model = _DummyModel()

    safe, message = guard.extract_label_and_categories("hello")

    assert safe is False
    assert message == "Prompt blocked by Qwen3Guard. Unable to determine safety label from moderation output."


def test_is_safe_blocks_when_guardrail_raises(qwen3guard_module) -> None:
    guard = object.__new__(qwen3guard_module.Qwen3Guard)

    def _raise(prompt: str) -> tuple[bool, str]:
        raise RuntimeError("boom")

    guard.extract_label_and_categories = _raise

    safe, message = guard.is_safe("hello")

    assert safe is False
    assert message == "Prompt blocked by Qwen3Guard because the guardrail could not complete safety evaluation."
