# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.comm.weight_transfer import WeightPushTiming
from cosmos_framework.posttrain.control.loop.weight_sync import (
    WEIGHT_SYNC_TIMING_ENV,
    _emit_weight_push_timings,
    summarize_weight_push_timings,
)
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.utils.log import logger

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _record(
    *,
    rank: int,
    total: float,
    materialize: tuple[float, ...],
    gather: tuple[float | None, ...],
    deliver: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "version": 7,
        "plan_s": 1.0 + rank,
        "coverage_s": 2.0 + rank,
        "materialize_s": materialize,
        "gather_gpu_s": gather,
        "deliver_s": deliver,
        "ready_s": 3.0 + rank,
        "total_s": total,
        "rank": rank,
        "host": f"node-{rank}",
        "bucket_meta": (
            {"staged_bytes": 1024, "names": 2, "first": "a", "last": "b"},
            {"staged_bytes": 2048, "names": 1, "first": "c", "last": "c"},
        ),
    }


def _records() -> list[dict[str, Any]]:
    return [
        _record(rank=0, total=20.0, materialize=(4.0, 8.0), gather=(2.0, None), deliver=(5.0, 9.0)),
        _record(rank=1, total=30.0, materialize=(6.0, 10.0), gather=(3.0, 4.0), deliver=(7.0, 11.0)),
    ]


def test_weight_push_timing_summary_preserves_buckets_and_slowest_cell() -> None:
    summary = summarize_weight_push_timings(_records())

    assert summary["version"] == 7 and summary["cells"] == 2 and summary["buckets"] == 2
    assert summary["total"] == {
        "n": 2,
        "min_s": 20.0,
        "p50_s": 25.0,
        "p90_s": 30.0,
        "p99_s": 30.0,
        "max_s": 30.0,
    }
    assert summary["slowest"] == {"rank": 1, "host": "node-1", "total_s": 30.0}
    assert summary["bucket"][0]["materialize"]["p50_s"] == 5.0
    assert summary["bucket"][0]["meta"] == {"staged_bytes": 1024, "names": 2, "first": "a", "last": "b"}
    assert summary["bucket"][1]["gather_gpu"] == {
        "n": 1,
        "min_s": 4.0,
        "p50_s": 4.0,
        "p90_s": 4.0,
        "p99_s": 4.0,
        "max_s": 4.0,
    }


def test_trainer_exposes_its_last_weight_push_timing() -> None:
    trainer = LLMTrainer.__new__(LLMTrainer)
    trainer._weight_backend = SimpleNamespace(
        last_push_timing=WeightPushTiming(
            version=9,
            plan_s=1.0,
            coverage_s=2.0,
            materialize_s=(3.0,),
            gather_gpu_s=(2.5,),
            deliver_s=(4.0,),
            ready_s=5.0,
            total_s=15.0,
        ),
        plan=lambda: [[("a", (4,), "torch.bfloat16")]],
    )

    timing = trainer.weight_push_timing()
    assert timing is not None
    assert timing["version"] == 9 and timing["total_s"] == 15.0
    assert timing["materialize_s"] == (3.0,) and timing["gather_gpu_s"] == (2.5,)
    assert timing["bucket_meta"] == ({"staged_bytes": 16, "names": 1, "first": "a", "last": "a"},)
    assert isinstance(timing["rank"], int) and isinstance(timing["host"], str)


def test_weight_push_timing_summary_rejects_inconsistent_streams() -> None:
    records = _records()
    records[1] = {**records[1], "version": 8}
    with pytest.raises(ValueError, match="disagree on version"):
        summarize_weight_push_timings(records)

    records = _records()
    records[1] = {**records[1], "deliver_s": (7.0,)}
    with pytest.raises(ValueError, match="bucket count"):
        summarize_weight_push_timings(records)

    records = _records()
    records[1] = {
        **records[1],
        "bucket_meta": (
            {"staged_bytes": 1024, "names": 2, "first": "a", "last": "b"},
            {"staged_bytes": 4096, "names": 1, "first": "c", "last": "c"},
        ),
    }
    with pytest.raises(ValueError, match="bucket metadata"):
        summarize_weight_push_timings(records)


def test_weight_push_timing_summary_rejects_cells_without_a_record() -> None:
    with pytest.raises(ValueError, match="1 of 3 policy cells recorded no weight-push timing"):
        summarize_weight_push_timings([*_records(), None])


def test_weight_push_timing_summary_rejects_non_finite_durations() -> None:
    records = _records()
    records[1] = {**records[1], "total_s": float("nan")}
    with pytest.raises(ValueError, match="1 non-finite duration"):
        summarize_weight_push_timings(records)


class _Handle:
    def __init__(self, records: list[dict[str, Any] | None]) -> None:
        self._records = records

    def all(self) -> dict[tuple[int, int], dict[str, Any] | None]:
        return {(0, index): record for index, record in enumerate(self._records)}


class _Policy:
    def weight_push_timing(self) -> _Handle:
        return _Handle(_records())


class _ByteFreePolicy:
    def weight_push_timing(self) -> _Handle:
        return _Handle([None, None])


class _InconsistentPolicy:
    def weight_push_timing(self) -> _Handle:
        first, second = _records()
        return _Handle([first, {**second, "version": 8}])


class _PartialPolicy:
    def weight_push_timing(self) -> _Handle:
        return _Handle([_records()[0], None])


class _FailedCellHandle:
    def as_completed(self) -> Iterator[tuple[tuple[int, int], object]]:
        yield (0, 0), _records()[0]
        yield (0, 1), RuntimeError("policy actor died")


class _FailedCellPolicy:
    def weight_push_timing(self) -> _FailedCellHandle:
        return _FailedCellHandle()


def test_weight_push_timing_emission_is_opt_in_json(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(logger, "info", lambda *args: calls.append(args))

    _emit_weight_push_timings(_Policy())
    assert calls == []

    monkeypatch.setenv(WEIGHT_SYNC_TIMING_ENV, "1")
    _emit_weight_push_timings(_Policy())
    assert calls[0][0] == "benchmark weight_sync_detail %s"
    payload = json.loads(calls[0][1])
    assert payload["cells"] == 2 and payload["bucket"][1]["deliver"]["max_s"] == 11.0


def test_weight_push_timing_emission_skips_byte_free_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(logger, "info", lambda *args: calls.append(args))
    monkeypatch.setenv(WEIGHT_SYNC_TIMING_ENV, "1")

    _emit_weight_push_timings(_ByteFreePolicy())
    assert calls == []


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (_InconsistentPolicy(), "disagree on version"),
        (_PartialPolicy(), "1 of 2 policy cells recorded no weight-push timing"),
        (_FailedCellPolicy(), "mesh operation failed on 1 cell(s)"),
        (object(), "object exposes no weight_push_timing()"),
    ],
    ids=["inconsistent-records", "partial-records", "failed-cell", "no-probe"],
)
def test_weight_push_timing_emission_warns_instead_of_ending_the_step(
    monkeypatch: pytest.MonkeyPatch,
    policy: object,
    reason: str,
) -> None:
    infos: list[tuple[Any, ...]] = []
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(logger, "info", lambda *args: infos.append(args))
    monkeypatch.setattr(logger, "warning", lambda *args: warnings.append(args))
    monkeypatch.setenv(WEIGHT_SYNC_TIMING_ENV, "1")

    _emit_weight_push_timings(policy)

    assert infos == []
    assert len(warnings) == 1 and warnings[0][1] == WEIGHT_SYNC_TIMING_ENV
    assert reason in str(warnings[0][2])


@pytest.mark.parametrize("trainer_cls", [LLMTrainer, DiffusionTrainer], ids=["llm", "diffusion"])
def test_trainers_without_a_recorded_weight_stream_report_no_push_timing(
    trainer_cls: type[LLMTrainer] | type[DiffusionTrainer],
) -> None:
    assert trainer_cls.__new__(trainer_cls).weight_push_timing() is None
