# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU traces verify bounded lifecycle; CUDA profiling is an integration check."""

import json

import pytest
import torch

from cosmos_framework.posttrain.utils.training_profile import NativeStepProfiler, profile_range


def test_disabled_profile_does_not_construct_profiler(monkeypatch):
    monkeypatch.delenv("COSMOS_SFT_PROFILE_DIR", raising=False)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: pytest.fail("disabled profiler constructed"))
    profiler = NativeStepProfiler()
    profiler.start()
    with profile_range("disabled"):
        profiler.step()
    profiler.close()


def test_bounded_trace_contains_forward_backward_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("COSMOS_SFT_PROFILE_WAIT", "0")
    monkeypatch.setenv("COSMOS_SFT_PROFILE_WARMUP", "1")
    monkeypatch.setenv("COSMOS_SFT_PROFILE_ACTIVE", "1")
    monkeypatch.setenv("RANK", "0")
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    profiler = NativeStepProfiler()
    profiler.start()
    try:
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            with profile_range("test.forward"):
                loss = model(torch.ones(2, 8)).square().sum()
            with profile_range("test.backward"):
                loss.backward()
            optimizer.step()
            profiler.step()
    finally:
        profiler.close()
    assert profiler.cycles == 1
    summaries = list((tmp_path / "rank_0").glob("*.summary.json"))
    assert len(summaries) == 1
    rows = json.loads(summaries[0].read_text())
    assert {"test.forward", "test.backward"} <= {row["name"] for row in rows}
    assert (tmp_path / "rank_0/window_0.json.gz").stat().st_size > 0
    assert json.loads((tmp_path / "rank_0/runtime.json").read_text())["schedule"]["active"] == 1
    with pytest.raises(FileExistsError):
        NativeStepProfiler()


@pytest.mark.parametrize("name,value", [("WAIT", "-1"), ("WARMUP", "0"), ("ACTIVE", "0"), ("ACTIVE", "bad")])
def test_invalid_profile_window_fails_before_creating_directory(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path / "trace"))
    monkeypatch.setenv(f"COSMOS_SFT_PROFILE_{name}", value)
    with pytest.raises(ValueError, match=f"COSMOS_SFT_PROFILE_{name}"):
        NativeStepProfiler()
    assert not (tmp_path / "trace").exists()


def test_disabled_helpers_leave_model_processor_and_files_untouched(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.utils.training_profile import (
        ensure_profile_model,
        profile_cpu_range,
        profile_processor,
    )

    monkeypatch.delenv("COSMOS_SFT_PROFILE_DIR", raising=False)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    ensure_profile_model(model)
    with profile_cpu_range("disabled"), profile_processor(object()):
        actual = model(torch.ones(1, 2))
    assert actual.shape == (1, 2)
    assert not list(tmp_path.iterdir())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_real_hooks_preserve_checkpoint_gradients_and_mark_recompute(tmp_path, monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from cosmos_framework.posttrain.utils.training_profile import ensure_profile_model, profile_backward

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(4, 2, bias=False)
            self.lora_B = torch.nn.Linear(2, 4, bias=False)

        def forward(self, x):
            return torch.sin(self.lora_B(self.lora_A(x)))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TinyAttention()
            self.lm_head = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.lm_head(checkpoint(self.attn, x, use_reentrant=False))

    torch.manual_seed(81)
    model = Model()
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = model(inputs)
    expected.square().sum().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("COSMOS_SFT_PROFILE_WAIT", "0")
    monkeypatch.setenv("COSMOS_SFT_PROFILE_WARMUP", "1")
    monkeypatch.setenv("COSMOS_SFT_PROFILE_ACTIVE", "1")
    profiler = NativeStepProfiler()
    ensure_profile_model(model)
    ensure_profile_model(model)  # no duplicate hooks or second inventory
    profiler.start()
    try:
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            actual = model(inputs)
            with profile_backward():
                actual.square().sum().backward()
            assert torch.equal(actual, expected)
            assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
            profiler.step()
    finally:
        profiler.close()
    names = {r["name"] for r in json.loads((tmp_path / "rank_0/window_0.summary.json").read_text())}
    assert {
        "sft.module.adapter_a.forward",
        "sft.module.adapter_b.backward_recompute",
        "sft.module.attention.backward_recompute",
        "sft.module.output_projection.forward",
    } <= names
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_processor_wrapper_restores_exact_method_on_success_and_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.utils.training_profile import profile_processor

    class Video:
        def _decode_and_sample_videos(self, values, *, fail=False):
            if fail:
                raise ValueError("actual decoder failure")
            return [value + 1 for value in values]

    video = Video()
    processor = SimpleNamespace(video_processor=SimpleNamespace(_processor=video))
    before = dict(vars(video))
    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path))
    with profile_processor(processor):
        assert video._decode_and_sample_videos([1, 3]) == [2, 4]
    assert vars(video) == before
    with pytest.raises(ValueError, match="actual decoder failure"), profile_processor(processor):
        video._decode_and_sample_videos([], fail=True)
    assert vars(video) == before
    rows = [json.loads(line) for line in next(tmp_path.glob("cpu-spans_*.jsonl")).read_text().splitlines()]
    assert len(rows) == 4
    assert {r["name"] for r in rows} == {"sft.video_decode_and_sample", "sft.processor"}
    for inner, outer in zip(rows[::2], rows[1::2], strict=True):
        assert (
            outer["begin_monotonic_ns"]
            <= inner["begin_monotonic_ns"]
            < inner["end_monotonic_ns"]
            <= outer["end_monotonic_ns"]
        )


def test_spawned_worker_cpu_spans_are_separate_and_bounded(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    from cosmos_framework.posttrain.utils import training_profile as module

    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path))
    with module.profile_cpu_range("parent"):
        pass
    code = 'from cosmos_framework.posttrain.utils.training_profile import profile_cpu_range\nwith profile_cpu_range("worker"):\n pass\n'
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=30)
    paths = list(tmp_path.glob("cpu-spans_*.jsonl"))
    assert len(paths) == 2
    records = [json.loads(p.read_text()) for p in paths]
    assert {r["name"] for r in records} == {"parent", "worker"}
    assert len({r["pid"] for r in records}) == 2
    key = (str(tmp_path), os.getpid())
    path, _ = module._CPU_EVENT_STATE[key]
    module._CPU_EVENT_STATE[key] = path, 4096
    with pytest.raises(RuntimeError, match="4096 spans"):
        with module.profile_cpu_range("overflow"):
            pass


def test_stop_failure_still_removes_all_model_hooks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.utils.training_profile import ensure_profile_model

    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", str(tmp_path))
    model = torch.nn.ModuleDict({"lora_A": torch.nn.Linear(2, 1)})
    ensure_profile_model(model)
    assert model["lora_A"]._forward_hooks
    profiler = NativeStepProfiler.__new__(NativeStepProfiler)
    profiler.started = True

    def fail():
        raise OSError("trace export failed")

    profiler.profiler = SimpleNamespace(stop=fail)
    with pytest.raises(OSError, match="trace export failed"):
        profiler.close()
    assert not profiler.started
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
