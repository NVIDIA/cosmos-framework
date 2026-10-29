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
