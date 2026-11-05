# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The per-bucket receive skeleton on `WeightReceiverBase`: fetch-through-sink, coverage reset and check.

Lifted from `VLLMRollout` so a second engine sink (vLLM-Omni) inherits it instead of copying it. A fake backend
hands the sink the bucket dict directly; what is pinned is the base's bookkeeping around the two hooks.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.roles.rollout.base import WeightReceiverBase


class _Backend:
    """Stands in for `WeightSync`: `apply_bucket` feeds the payload (already a state dict) to the sink."""

    def __init__(self, buckets: int) -> None:
        self._buckets = buckets
        self.calls: list[tuple[str, int]] = []

    def bucket_count(self) -> int:
        return self._buckets

    def apply_bucket(self, model: Any, i: int, payload: Any, sink: Any = None) -> None:
        self.calls.append(("apply", i))
        sink(payload)


class _Receiver(WeightReceiverBase):
    def __init__(self, engine_names: set[str], backend: _Backend | None) -> None:
        self.model = object()
        self._weight_backend = backend
        self._loaded_names = set()
        self._engine_names = set(engine_names)
        self.sunk: list[dict] = []

    def _engine_sink(self, state: dict) -> set[str]:
        self.sunk.append(dict(state))
        return set(state)

    def _engine_param_names(self) -> set[str]:
        return set(self._engine_names)


def _bucket(*names: str) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(1) for name in names}


@pytest.mark.L0
@pytest.mark.CPU
def test_no_backend_is_a_noop() -> None:
    receiver = _Receiver({"a"}, None)
    receiver.apply_bucket(_bucket("a"), 0)
    assert receiver.sunk == []


@pytest.mark.L0
@pytest.mark.CPU
def test_a_covering_stream_reaches_the_sink_bucket_by_bucket() -> None:
    backend = _Backend(2)
    receiver = _Receiver({"a", "b", "c"}, backend)
    receiver.apply_bucket(_bucket("a", "b"), 0)
    receiver.apply_bucket(_bucket("c"), 1)
    assert [sorted(s) for s in receiver.sunk] == [["a", "b"], ["c"]]
    assert backend.calls == [("apply", 0), ("apply", 1)]
    assert receiver._loaded_names == {"a", "b", "c"}


@pytest.mark.L0
@pytest.mark.CPU
def test_the_last_bucket_fails_when_an_engine_param_was_never_loaded() -> None:
    receiver = _Receiver({"a", "b"}, _Backend(2))
    receiver.apply_bucket(_bucket("a"), 0)
    with pytest.raises(ValueError, match="did not cover the engine.*'b'"):
        receiver.apply_bucket(_bucket("a"), 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_zero_resets_the_accumulator_so_a_previous_stream_cannot_mask_a_gap() -> None:
    receiver = _Receiver({"a", "b"}, _Backend(2))
    receiver.apply_bucket(_bucket("a"), 0)
    receiver.apply_bucket(_bucket("b"), 1)  # stream 1 covers everything
    receiver.apply_bucket(_bucket("a"), 0)  # stream 2 forgets `b`
    with pytest.raises(ValueError, match="did not cover the engine"):
        receiver.apply_bucket(_bucket("a"), 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_unverifiable_coverage_is_not_a_pass_and_not_a_check() -> None:
    """An engine that cannot report names sets the accumulator to None; the base then neither raises nor claims."""
    receiver = _Receiver({"a", "b"}, _Backend(1))

    def _blind_sink(state: dict) -> None:
        return None

    receiver._engine_sink = _blind_sink  # type: ignore[method-assign]
    receiver.apply_bucket(_bucket("a"), 0)
    assert receiver._loaded_names is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("hook, args", [("_engine_sink", ({},)), ("_engine_param_names", ())])
def test_the_base_refuses_to_stand_in_for_an_engine(hook: str, args: tuple) -> None:
    receiver = object.__new__(WeightReceiverBase)
    with pytest.raises(NotImplementedError, match=hook):
        getattr(receiver, hook)(*args)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_inherits_the_skeleton_and_keeps_its_own_bracket() -> None:
    """vLLM keeps `apply_bucket` (the MoE reload bracket) and `_engine_sink`; the coverage steps are the base's.

    Only the decision on an unverifiable stream is vLLM's own: a dummy-loaded engine refuses one."""
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    assert VLLMRollout._coverage_reset is WeightReceiverBase._coverage_reset
    assert VLLMRollout._coverage_check is WeightReceiverBase._coverage_check
    assert VLLMRollout._check_unverifiable_coverage is not WeightReceiverBase._check_unverifiable_coverage
    assert VLLMRollout.apply_bucket is not WeightReceiverBase.apply_bucket
    assert VLLMRollout._engine_sink is not WeightReceiverBase._engine_sink


@pytest.mark.L0
@pytest.mark.CPU
def test_unverifiable_bucket_stays_unverifiable_until_the_next_stream(monkeypatch) -> None:
    receiver = _Receiver({"a", "b"}, _Backend(2))
    original = receiver._engine_sink
    answers = iter((None, {"b"}))
    with monkeypatch.context() as patch:
        patch.setattr(receiver, "_engine_sink", lambda state: next(answers))
        receiver.apply_bucket(_bucket("a"), 0)
        receiver.apply_bucket(_bucket("b"), 1)
    assert receiver._loaded_names is None
    assert receiver._engine_sink == original
    receiver.apply_bucket(_bucket("a"), 0)
    assert receiver._loaded_names == {"a"}
    receiver.apply_bucket(_bucket("b"), 1)
    assert receiver._loaded_names == {"a", "b"}
