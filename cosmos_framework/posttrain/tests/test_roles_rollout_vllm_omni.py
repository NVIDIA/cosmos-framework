# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The model-agnostic vLLM-Omni rollout base: dispatch order, the weight gate, and the hook contract.

Bodies are built with ``object.__new__`` and given only the attributes each seam reads -- no engine.
The recording subclass stands in for a model; what is asserted is what the BASE hands the hooks, so a
future Omni model inherits the same guarantees. Stays L0 because neither the base nor these cases
import vLLM-Omni (the sampling-params object is the subclass's business).
"""

from __future__ import annotations

from typing import Any

import pytest

from cosmos_framework.posttrain.roles.rollout.vllm_omni import OmniPolicyBody

_PARAMS = object()  # the subclass's sampling-params object; the base must pass it through untouched


class _RecordingBody(OmniPolicyBody):
    """A minimal Omni model whose hooks record exactly what the base gave them."""

    def _engine_kwargs(self) -> dict[str, Any]:
        self.calls.append(("engine_kwargs",))
        return {"model": "unused"}

    def _build_request(self, obs_list, seed):
        self.calls.append(("build_request", list(obs_list), seed))
        return ["prompt"], _PARAMS

    def _parse_output(self, output, n):
        self.calls.append(("parse_output", output, n))
        return [{"action": output}] * n


class _EngineStub:
    """Stands in for ``Omni``: records the generate call and returns a canned output list."""

    def __init__(self, calls: list, outputs: list) -> None:
        self._calls, self._outputs = calls, outputs

    def generate(self, prompts, params):
        self._calls.append(("generate", list(prompts), params))
        return self._outputs


def _body(outputs: list | None = None, weight_version: int = 0) -> _RecordingBody:
    body = object.__new__(_RecordingBody)
    body.calls = []
    body._weight_version = weight_version
    body._omni = _EngineStub(body.calls, ["out"] if outputs is None else outputs)
    return body


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("seed", [1234, None])
def test_infer_batch_dispatches_through_the_hooks_in_order(seed: int | None) -> None:
    """The base owns the order; a model only fills the three holes. `seed` reaches the hook verbatim."""
    body = _body()
    obs_list = [{"a": 1}, {"a": 2}]

    assert body.infer_batch(obs_list, seed=seed, weight_version=0) == [{"action": "out"}] * 2

    assert body.calls == [
        ("build_request", obs_list, seed),
        ("generate", ["prompt"], _PARAMS),
        ("parse_output", "out", 2),
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_no_observations_touches_neither_the_gate_nor_the_engine() -> None:
    """An empty batch is not a request: it must not consult the weight gate or wake the engine."""
    body = _body(weight_version=0)
    # A mismatching version would raise if the gate ran; the short-circuit is what is being pinned.
    assert body.infer_batch([], seed=1, weight_version=99) == []
    assert body.calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_version_mismatch_raises_before_any_hook_runs() -> None:
    """Silently serving a stale version is how off-policy data enters a training batch."""
    body = _body(weight_version=2)
    with pytest.raises(RuntimeError, match="weight_version=3"):
        body.infer_batch([{"a": 1}], weight_version=3)
    assert body.calls == []
    # Unspecified means the caller opts out; the matching version passes through.
    assert body.infer_batch([{"a": 1}], weight_version=None) == [{"action": "out"}]
    assert body.infer_batch([{"a": 1}], weight_version=2) == [{"action": "out"}]


@pytest.mark.L0
@pytest.mark.CPU
def test_an_empty_engine_reply_raises_instead_of_reaching_the_parser() -> None:
    body = _body(outputs=[])
    with pytest.raises(RuntimeError, match="no output"):
        body.infer_batch([{"a": 1}])
    assert [c[0] for c in body.calls] == ["build_request", "generate"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "hook, args",
    [
        ("_engine_kwargs", ()),
        ("_build_request", ([{"a": 1}], None)),
        ("_parse_output", ("out", 1)),
    ],
)
def test_the_base_refuses_to_stand_in_for_a_model(hook: str, args: tuple) -> None:
    """The hooks are the model contract; a subclass that forgets one must fail loudly, not serve."""
    body = object.__new__(OmniPolicyBody)
    with pytest.raises(NotImplementedError, match=hook):
        getattr(body, hook)(*args)
