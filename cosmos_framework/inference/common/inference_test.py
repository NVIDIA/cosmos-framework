# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_framework.inference.common.args import GuardrailArgs, SampleArgs, SampleOutputs, SetupArgs
from cosmos_framework.inference.common.inference import GuardrailRunners, Inference, _download_on_rank0


class _DummyInference(Inference):
    calls: list[bool]
    events: list[str] | None

    @property
    def model_config(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _create(cls, setup_args: SetupArgs, /, **kwargs: Any) -> Self:
        instance = cls(setup_args=setup_args, model=Mock(), **kwargs)
        instance.calls = []
        instance.events = None
        return instance

    def create_batches(
        self, sample_args_list: Sequence[SampleArgs]
    ) -> Iterator[tuple[list[SampleArgs], dict[str, Any]]]:
        yield list(sample_args_list), {"payload": [1]}

    def generate_batch(
        self,
        sample_args_list: Sequence[SampleArgs],
        data_batch: dict[str, Any],
        *,
        save_outputs: bool = True,
    ) -> list[SampleOutputs]:
        self.calls.append(save_outputs)
        if self.events is not None:
            self.events.append(f"generate-save-{save_outputs}")
        return [SampleOutputs(args={"name": "sample"})] if save_outputs else []


def _setup_args(**overrides: Any) -> SetupArgs:
    values = {
        "benchmark": True,
        "warmup": 0,
        "num_iterations": 1,
        "profile": False,
        "guardrails": False,
        "keep_going": False,
    }
    values.update(overrides)
    return cast(SetupArgs, SimpleNamespace(**values))


def _sample_args() -> list[SampleArgs]:
    return cast(list[SampleArgs], [SimpleNamespace(name="sample")])


def test_benchmark_runs_warmups_and_requested_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference.common import inference as inference_module

    monkeypatch.setattr(inference_module, "sync_distributed_errors", contextlib.nullcontext)
    pipe = _DummyInference.create(_setup_args(warmup=2, num_iterations=3))

    outputs = pipe.generate(_sample_args())

    assert len(outputs) == 1
    assert pipe.calls == [True, False, False, False, False]
    timer_results = pipe.get_timer_results()
    assert timer_results is not None
    assert len(timer_results["all"]["[warmup] _DummyInference.generate_batch"]) == 2
    assert len(timer_results["all"]["_DummyInference.generate_batch"]) == 3


def test_benchmark_saves_first_warmup_and_profiles_last(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference.common import inference as inference_module

    events: list[str] = []

    class FakeProfiler:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Self:
            events.append("profile-enter")
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            events.append("profile-exit")

    monkeypatch.setattr(inference_module, "sync_distributed_errors", contextlib.nullcontext)
    monkeypatch.setattr(inference_module, "is_rank0", lambda: False)
    monkeypatch.setattr(inference_module.torch.profiler, "profile", FakeProfiler)
    pipe = _DummyInference.create(_setup_args(warmup=3, num_iterations=2, profile=True))
    pipe.events = events

    outputs = pipe.generate(_sample_args())

    assert len(outputs) == 1
    assert events == [
        "generate-save-True",
        "generate-save-False",
        "profile-enter",
        "generate-save-False",
        "profile-exit",
        "generate-save-False",
        "generate-save-False",
    ]


def test_benchmark_without_warmup_profiles_final_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference.common import inference as inference_module

    events: list[str] = []

    class FakeProfiler:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Self:
            events.append("profile-enter")
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            events.append("profile-exit")

    monkeypatch.setattr(inference_module, "sync_distributed_errors", contextlib.nullcontext)
    monkeypatch.setattr(inference_module, "is_rank0", lambda: False)
    monkeypatch.setattr(inference_module.torch.profiler, "profile", FakeProfiler)
    pipe = _DummyInference.create(_setup_args(num_iterations=3, profile=True))
    pipe.events = events

    outputs = pipe.generate(_sample_args())

    assert len(outputs) == 1
    assert events == [
        "generate-save-True",
        "generate-save-False",
        "profile-enter",
        "generate-save-False",
        "profile-exit",
    ]


def test_non_benchmark_run_saves_during_first_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference.common import inference as inference_module

    monkeypatch.setattr(inference_module, "sync_distributed_errors", contextlib.nullcontext)
    pipe = _DummyInference.create(_setup_args(benchmark=False, warmup=2))

    outputs = pipe.generate(_sample_args())

    assert len(outputs) == 1
    assert pipe.calls == [True, False, False]


def test_benchmark_average_drops_cold_first_pass_only_without_warmup() -> None:
    for warmup, expected_average in [(0, 2.0), (1, 4.0)]:
        pipe = _DummyInference.create(_setup_args(warmup=warmup, num_iterations=4))
        assert pipe._timer is not None
        pipe._timer.results = {
            "_DummyInference.generate_batch": [10.0, 2.0, 2.0, 2.0],
            "[warmup] _DummyInference.generate_batch": [12.0, 8.0],
        }

        results = pipe.get_timer_results()

        assert results is not None
        assert results["average"]["_DummyInference.generate_batch"] == expected_average
        assert results["average"]["[warmup] _DummyInference.generate_batch"] == 10.0
        assert results["all"]["_DummyInference.generate_batch"] == [10.0, 2.0, 2.0, 2.0]


def test_download_on_rank0_broadcasts_shared_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    download = Mock(return_value="/shared/cache/model")
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    assert _download_on_rank0(download) == Path("/shared/cache/model")
    download.assert_called_once_with()
    broadcast.assert_called_once_with(["/shared/cache/model", None], src=0)


def test_download_on_nonzero_rank_reuses_broadcast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    download = Mock()

    def broadcast(payload: list[str | None], *, src: int) -> None:
        assert src == 0
        payload[:] = ["/shared/cache/model", None]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    assert _download_on_rank0(download) == Path("/shared/cache/model")
    download.assert_not_called()


def test_guardrail_runners() -> None:
    from cosmos_framework.auxiliary.guardrail.common import presets

    guardrail_args = GuardrailArgs(guardrails=True, offload_guardrail_models=False)
    runners = GuardrailRunners.create(guardrail_args)
    assert runners.text is not None
    assert runners.video is not None

    assert presets.run_text_guardrail("test", runners.text)
    assert not presets.run_text_guardrail("Tesla Cybertruck", runners.text)

    frames_thwc = np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8)
    clean_frames_thwc = presets.run_video_guardrail(frames_thwc, runners.video)
    assert clean_frames_thwc is not None
    np.testing.assert_allclose(frames_thwc, clean_frames_thwc)
