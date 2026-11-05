# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""VLLMRollout sleep/wake bookkeeping against a recording fake engine (no vLLM needed)."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sleep(self, level: int = 1) -> None:
        self.calls.append(("sleep", level))

    def wake_up(self, tags=None) -> None:
        self.calls.append(("wake_up", tuple(tags) if tags is not None else None))


class _PinnedExecutorEngine(_Engine):
    """Reproduce pinned vLLM's early return when any allocator tag remains asleep."""

    def __init__(self) -> None:
        super().__init__()
        self.sleeping_tags: set[str] = set()

    def sleep(self, level: int = 1) -> None:
        if self.sleeping_tags:
            return
        super().sleep(level)
        self.sleeping_tags = {"weights", "kv_cache"}

    def wake_up(self, tags=None) -> None:
        requested = tuple(tags) if tags is not None else tuple(self.sleeping_tags)
        super().wake_up(requested)
        self.sleeping_tags -= set(requested)


def _rollout(level: int = 1) -> tuple[VLLMRollout, _Engine]:
    engine = _Engine()
    body = VLLMRollout.__new__(VLLMRollout)  # bypass __init__ (which needs a real vllm.LLM)
    body.model = engine
    body._sleep_level = level
    body._asleep_tags = set()
    return body, engine


@pytest.mark.L0
@pytest.mark.CPU
def test_sleep_is_idempotent_and_uses_the_configured_level():
    body, engine = _rollout(level=2)
    body.sleep()
    body.sleep()
    assert engine.calls == [("sleep", 2)]
    assert body._asleep_tags == {"weights", "kv_cache"}


@pytest.mark.L0
@pytest.mark.CPU
def test_sleep_releases_weights_again_after_a_partial_wake() -> None:
    body, engine = _rollout(level=2)
    body.sleep()
    body.wake_up(("weights",))

    body.sleep()

    assert engine.calls == [
        ("sleep", 2),
        ("wake_up", ("weights",)),
        ("wake_up", ("kv_cache",)),
        ("sleep", 2),
    ]
    assert body._asleep_tags == {"weights", "kv_cache"}


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_cleanup_clears_the_pinned_executor_guard_before_sleeping() -> None:
    body, _ = _rollout(level=2)
    engine = _PinnedExecutorEngine()
    body.model = engine
    body.sleep()
    body.wake_up(("weights",))
    assert body._asleep_tags == engine.sleeping_tags == {"kv_cache"}

    body.sleep()

    assert engine.calls == [
        ("sleep", 2),
        ("wake_up", ("weights",)),
        ("wake_up", ("kv_cache",)),
        ("sleep", 2),
    ]
    assert body._asleep_tags == engine.sleeping_tags == {"weights", "kv_cache"}


@pytest.mark.L0
@pytest.mark.CPU
def test_wake_weights_then_kv_cache_issues_two_partial_wakeups_and_no_repeats():
    body, engine = _rollout()
    body.sleep()
    body.wake_up(("weights",))
    body.wake_up(("weights",))  # already awake: skipped
    assert body._asleep_tags == {"kv_cache"}
    body.wake_up()  # only the still-released region is requested
    assert engine.calls == [("sleep", 1), ("wake_up", ("weights",)), ("wake_up", ("kv_cache",))]
    assert body._asleep_tags == set()
    body.wake_up()  # fully awake: no engine call
    assert len(engine.calls) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_wake_up_rejects_unknown_tags():
    body, _ = _rollout()
    body.sleep()
    with pytest.raises(ValueError, match="unknown vLLM wake_up tags"):
        body.wake_up(("activations",))


@pytest.mark.L0
@pytest.mark.CPU
def test_generate_and_weight_loading_refuse_a_sleeping_engine():
    body, _ = _rollout()
    body.sleep()
    with pytest.raises(RuntimeError, match="released; call wake_up"):
        body.generate([], group=1)
    with pytest.raises(RuntimeError, match="weights"):
        body._engine_sink({})
    body.wake_up(("weights",))
    # Weights resident is enough for loading; generation still needs the KV cache.
    with pytest.raises(RuntimeError, match="kv_cache"):
        body.generate([], group=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_init_rejects_an_invalid_sleep_level():
    class _Packer:
        def adds_special_tokens(self) -> bool:
            return False

    with pytest.raises(ValueError, match="sleep_level"):
        VLLMRollout(object(), packer=_Packer(), sleep_level=3)
