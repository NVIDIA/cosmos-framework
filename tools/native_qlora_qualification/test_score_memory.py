# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual scorer control flow and event consumer; allocator counters are CPU fakes."""

import json
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.utils import qlora_metrics
from tools.native_qlora_qualification import performance, score
from tools.native_qlora_qualification.test_score_preprocessing import execution as execution


def test_scoring_reset_retains_integrity_peak_and_preserves_outputs(execution, monkeypatch):
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.data.packer.sft_preprocess import SFTPreprocessingPipeline

    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    baseline, expected = execution.run("baseline", 100_000)
    memory = execution.root / "phases.jsonl"
    monkeypatch.setenv("COSMOS_QLORA_METRICS_FILE", str(memory))
    allocator = SimpleNamespace(current=100, peak=999, resets=0, synchronizations=0)

    def allocate(value):
        allocator.current = value
        allocator.peak = max(allocator.peak, value)

    def synchronize(_device):
        allocator.synchronizations += 1

    def reset():
        allocator.resets += 1
        allocator.peak = allocator.current

    cuda = qlora_metrics.torch.cuda
    monkeypatch.setattr(cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "synchronize", synchronize)
    monkeypatch.setattr(cuda, "memory_allocated", lambda _device: allocator.current)
    monkeypatch.setattr(cuda, "memory_reserved", lambda _device: allocator.current + 64)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda _device: allocator.peak)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda _device: allocator.peak + 64)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", reset)
    native_evaluate = score.evaluate_row
    native_build = build.build_backbone
    native_close = SFTPreprocessingPipeline.close

    def evaluate(*args):
        allocate(500)
        return native_evaluate(*args)

    def backbone(cfg):
        model = native_build(cfg)
        # Frozen-base validation has a larger temporary allocation than rows.
        model.net._cosmos_adapter_checkpoint_codec.validate_frozen_base = lambda: allocate(700)
        return model

    def close(value):
        native_close(value)
        allocate(100)

    monkeypatch.setattr(score, "evaluate_row", evaluate)
    monkeypatch.setattr(build, "build_backbone", backbone)
    monkeypatch.setattr(SFTPreprocessingPipeline, "close", close)
    measured, actual = execution.run("measured", 100_000)
    records = [json.loads(line) for line in memory.read_text().splitlines()]
    assert [r["phase"] for r in records] == ["scoring_start", "scoring_complete"]
    assert [r["reset_peak_after_record"] for r in records] == [True, False]
    assert allocator.resets == 1 and allocator.synchronizations == 2
    assert records[0]["cuda"]["peak_allocated_bytes"] == 999
    assert records[1]["cuda"]["peak_allocated_bytes"] == 700
    assert records[1]["cuda"]["allocated_bytes"] == 100
    report = performance.memory_summary(records, [], {"device_bytes": 4096, "uuid": "CPU-fake"})
    assert report["phases"]["scoring"]["allocator_peak_allocated_bytes"] == 700
    assert report["phases"]["scoring"]["allocator_peak_reserved_bytes"] == 764
    assert len(report["intervals"]) == 1
    assert report["sampled"]["owned_device_process_peak_bytes"] is None
    assert report["sampled"]["host_pss_complete"] is False
    for name in ("selection.json", "selected_rows.jsonl", "model_identity.json", "scores.jsonl"):
        assert (baseline / name).read_bytes() == (measured / name).read_bytes()
    assert expected["metrics"] == actual["metrics"]
    assert expected["preprocessing_cache"]["stats"]["samples"] == actual["preprocessing_cache"]["stats"]["samples"]


@pytest.mark.parametrize("failure", ["forward", "source", "media"])
def test_failed_scoring_records_after_cache_close_without_success(execution, monkeypatch, failure):
    def record(phase, *, reset_peak=False):
        execution.events.append((phase, reset_peak))

    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", record)
    execution.failure = failure
    with pytest.raises(RuntimeError):
        execution.run("failed", 100_000)
    assert execution.events == ["init", ("scoring_start", True), "close", ("scoring_failed", False), "destroy"]
    assert not (execution.root / "failed" / "summary.json").exists()
    assert execution.pipelines[0]._closed


def test_failed_measurement_does_not_replace_original_scoring_error(execution, monkeypatch):
    def record(phase, *, reset_peak=False):
        if phase == "scoring_failed":
            raise OSError("measurement disk failure")

    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", record)
    execution.failure = "forward"
    with pytest.raises(RuntimeError, match="fixture forward failure") as caught:
        execution.run("failed", 100_000)
    if hasattr(caught.value, "add_note"):
        assert any("measurement disk failure" in note for note in caught.value.__notes__)
    assert execution.events == ["init", "close", "destroy"]
    assert not (execution.root / "failed" / "summary.json").exists()


@pytest.mark.parametrize("primary", ["forward", "measurement"])
def test_cleanup_errors_preserve_primary_and_attempt_all_cleanup(execution, monkeypatch, primary):
    from cosmos_framework.posttrain.data.packer.sft_preprocess import SFTPreprocessingPipeline
    from cosmos_framework.posttrain.utils import dist

    native_close = SFTPreprocessingPipeline.close

    def close(value):
        native_close(value)
        raise RuntimeError("cache cleanup failure")

    def destroy():
        execution.events.append("destroy")
        raise RuntimeError("teardown failure")

    def record(phase, *, reset_peak=False):
        execution.events.append(phase)
        if primary == "measurement":
            raise OSError("primary measurement failure")

    monkeypatch.setattr(SFTPreprocessingPipeline, "close", close)
    monkeypatch.setattr(dist, "destroy_process_group_if_initialized", destroy)
    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", record)
    execution.failure = "forward" if primary == "forward" else None
    kind, message = (
        (RuntimeError, "fixture forward failure") if primary == "forward" else (OSError, "primary measurement failure")
    )
    with pytest.raises(kind, match=message) as caught:
        execution.run("failed", 100_000)
    assert execution.events == [
        "init",
        "scoring_start",
        "close",
        *(["scoring_failed"] if primary == "forward" else []),
        "destroy",
    ]
    assert execution.pipelines[0]._closed
    if hasattr(caught.value, "add_note"):
        assert any("cache cleanup failure" in note for note in caught.value.__notes__)
        assert any("teardown failure" in note for note in caught.value.__notes__)
    assert not (execution.root / "failed" / "summary.json").exists()


def test_teardown_failure_after_successful_scoring_remains_an_error(execution, monkeypatch):
    from cosmos_framework.posttrain.utils import dist

    def destroy():
        execution.events.append("destroy")
        raise RuntimeError("teardown failure")

    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    monkeypatch.setattr(dist, "destroy_process_group_if_initialized", destroy)
    with pytest.raises(RuntimeError, match="teardown failure"):
        execution.run("failed", 100_000)
    assert execution.events == ["init", "close", "destroy"]


@pytest.mark.parametrize("phase", ["scoring_start", "scoring_complete"])
@pytest.mark.parametrize("cache_bytes", [0, 100_000])
def test_measurement_write_failure_still_cleans_up(execution, monkeypatch, phase, cache_bytes):
    def record(actual, *, reset_peak=False):
        if actual == phase:
            raise OSError("measurement disk failure")

    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", record)
    with pytest.raises(OSError, match="measurement disk failure"):
        execution.run("failed", cache_bytes)
    assert execution.events == ["init", *(["close"] if cache_bytes else []), "destroy"]
    assert all(p._closed for p in execution.pipelines)
    assert not (execution.root / "failed" / "summary.json").exists()


def test_disabled_measurement_does_not_query_cuda(execution, monkeypatch):
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)

    def forbidden():
        raise AssertionError("disabled memory measurement queried CUDA")

    monkeypatch.setattr(qlora_metrics.torch.cuda, "is_initialized", forbidden)
    execution.run("disabled", 100_000)
    assert execution.events == ["init", "close", "destroy"]
