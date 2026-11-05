# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The rollout decides the engine's routed-expert layout from the live model, not from a version.

vLLM 0.23 lets the ``experts`` module own ``w13_weight``/``w2_weight`` itself; vLLM 0.24 nests them
in a ``routed_experts`` child. The mapper emits checkpoint-container names for both; the rollout
resolves them onto the live owner and brackets the stream with the reload. No vLLM import: the owner
query is the one seam faked here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.roles.rollout import vllm_weights as weight_compat
from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

_EXPERTS = "language_model.model.layers.0.mlp.experts"
_QKV = "language_model.model.layers.0.self_attn.qkv_proj.weight"
_CANONICAL = (f"{_EXPERTS}.w13_weight", f"{_EXPERTS}.w2_weight", _QKV)


class _FakeEngineModel:
    """Engine-side model: one routed-expert owner at ``owner`` plus one packed dense parameter."""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self._params = {
            f"{owner}.w13_weight": torch.zeros(2),
            f"{owner}.w2_weight": torch.zeros(2),
            _QKV: torch.zeros(2),
        }

    def named_parameters(self) -> list[tuple[str, torch.Tensor]]:
        return list(self._params.items())

    def load_weights(self, named: list[tuple[str, torch.Tensor]]) -> set[str]:
        return {name for name, _tensor in named if name in self._params}


class _FakeLLM:
    def __init__(self, model: _FakeEngineModel) -> None:
        self._m = model
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    def apply_model(self, fn):  # noqa: ANN001, ANN201 - mirrors vllm.LLM.apply_model
        return [fn(self._m)]


class _FakeBackend:
    """One-bucket streaming backend that records the destinations the rollout installs."""

    def __init__(self, bucket: dict[str, torch.Tensor]) -> None:
        self._bucket = bucket
        self._destination_names: tuple[str, ...] | None = None

    def set_destination_names(self, names: tuple[str, ...] | None) -> None:
        self._destination_names = names

    def destination_names(self) -> tuple[str, ...] | None:
        return self._destination_names

    def bucket_count(self) -> int:
        return 1

    def apply_bucket(self, model, i, payload, sink=None) -> None:  # noqa: ANN001
        sink(self._bucket)


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_resolves_moe_destinations_onto_the_live_layout_flat_or_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Container names land on the live owner (0.23 flat, 0.24 nested), pass coverage and bracket the reload; dense has no layout."""
    assert weight_compat.resolve_vllm_moe_layout([], _CANONICAL) is None

    events: list[str] = []
    monkeypatch.setattr(weight_compat, "_vllm_routed_expert_owner_names", lambda model: [model.owner])
    monkeypatch.setattr(weight_compat, "_vllm_begin_moe_reload", lambda _model: events.append("begin") or 1)

    def _finish(_model, *, model_dtype: torch.dtype) -> int:  # noqa: ANN001
        assert model_dtype is torch.bfloat16
        events.append("finish")
        return 1

    monkeypatch.setattr(weight_compat, "_vllm_finish_moe_reload", _finish)

    for owner in (_EXPERTS, f"{_EXPERTS}.routed_experts"):
        events.clear()
        engine = _FakeEngineModel(owner)
        live_names = {name for name, _tensor in engine.named_parameters()}
        rollout = VLLMRollout.__new__(VLLMRollout)  # bypass __init__ (needs a real vllm.LLM)
        rollout.model = _FakeLLM(engine)
        rollout._weight_backend = _FakeBackend({name: torch.ones(2) for name in live_names})

        rollout.set_weight_destination_names(_CANONICAL)
        assert set(rollout._weight_backend.destination_names()) == live_names
        rollout.check_plan_coverage()  # exact live match, before any tensor moves
        rollout.apply_bucket(None, 0)

        assert events == ["begin", "finish"]
        assert rollout._moe_reload_active is False
        assert rollout._loaded_names == live_names
