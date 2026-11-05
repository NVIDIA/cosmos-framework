# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from typing import Any

import pytest

from cosmos_framework.posttrain.roles.rollout import vllm_weights as weight_compat
from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Engine:
    def __init__(self, participants: int = 1) -> None:
        self._participants = participants

    def apply_model(self, fn: Any) -> list[str]:
        return [fn(object()) for _index in range(self._participants)]


class _Body:
    model: _Engine
    _weight_broadcast_spec: object

    def __init__(self, participants: int = 1) -> None:
        self.model = _Engine(participants)
        self._weight_broadcast_spec = object()
        self.ready: list[int] = []

    def mark_weight_version_ready(self, version: int) -> None:
        self.ready.append(version)


def test_vllm_r2r_enters_the_single_underlying_model_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, object, bool]] = []

    def _broadcast(model: object, spec: object, *, verify_values: bool = False) -> str:
        seen.append((model, spec, verify_values))
        return "digest"

    monkeypatch.setattr(weight_compat, "_vllm_broadcast_live_weights", _broadcast)
    body = _Body()

    digest = VLLMRollout.broadcast_live_weights(body, 5, verify_values=True)  # type: ignore[arg-type]

    assert digest == "digest"
    assert len(seen) == 1 and seen[0][1] is body._weight_broadcast_spec
    assert seen[0][2] is True
    assert body.ready == [5]


def test_vllm_r2r_initializes_the_collective_in_the_underlying_model_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, object]] = []

    def _initialize(model: object, spec: object) -> None:
        seen.append((model, spec))

    monkeypatch.setattr(weight_compat, "_vllm_initialize_weight_broadcast", _initialize)
    body = _Body()

    VLLMRollout.initialize_weight_broadcast(body)  # type: ignore[arg-type]

    assert len(seen) == 1 and seen[0][1] is body._weight_broadcast_spec


def test_vllm_r2r_rejects_multiple_model_participants_per_rollout_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broadcast(model: object, spec: object, *, verify_values: bool = False) -> str:
        del model, spec, verify_values
        return "digest"

    monkeypatch.setattr(weight_compat, "_vllm_broadcast_live_weights", _broadcast)
    body = _Body(participants=2)

    with pytest.raises(RuntimeError, match="exactly one live vLLM model participant"):
        VLLMRollout.broadcast_live_weights(body, 5)  # type: ignore[arg-type]

    assert body.ready == []
