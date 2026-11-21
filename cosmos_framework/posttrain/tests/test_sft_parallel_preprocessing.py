# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ordered two-worker preparation, total cache/CPU bounds and fail-closed cleanup."""

import copy
import hashlib
import os
import random
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.data.packer import sft_preprocess as prep
from cosmos_framework.posttrain.tests.test_sft_preprocessing import _local_pipeline, _text_row, _video_row

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _affinities():
    return [sorted(os.sched_getaffinity(int(task.name))) for task in Path("/proc/self/task").iterdir()]


def test_real_two_workers_match_serial_inputs_rng_cache_bound_and_cold_resume(tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("two inherited CPUs required")
    packer, pipeline = _local_pipeline(tmp_path, prefetch=True, workers=2, cache_bytes=100_001)
    rows = [_text_row("A"), _text_row("BB")]
    expected = [packer.sft_process_sample(row) for row in rows]
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    parent_affinity = os.sched_getaffinity(0)
    try:
        for _ in range(2):
            pipeline.prepare_window(rows)
            assert [pipeline.sft_process_sample(row) for row in rows] == expected
            pipeline.finish_window()
        stats = pipeline.preprocessing_stats
        assert stats["timing"]["processing_calls"] == stats["timing"]["wait_calls"] == 4
        assert stats["samples"]["hits"] == stats["samples"]["misses"] == 2
        assert stats["samples"]["limit"] + stats["videos"]["limit"] == 100_001
        assert stats["samples"]["bytes"] + stats["videos"]["bytes"] <= 100_001
        partitions = [set(value) for value in pipeline.worker_cpu_affinities]
        assert partitions[0].isdisjoint(partitions[1]) and set.union(*partitions) == parent_affinity
        for pool, cpus in zip(pipeline._executors, partitions, strict=True):
            assert all(set(value) == cpus for value in pool.submit(_affinities).result(timeout=30))
        assert pipeline.validation_semantics_identity() == packer.validation_semantics_identity()
        assert random.getstate() == before_python and np.array_equal(np.random.get_state()[1], before_numpy[1])
        assert torch.equal(torch.get_rng_state(), before_torch) and os.sched_getaffinity(0) == parent_affinity
        processes = [p for pool in pipeline._executors for p in pool._processes.values()]
    finally:
        pipeline.close()
    assert all(not process.is_alive() for process in processes)
    _, resumed = _local_pipeline(tmp_path, prefetch=True, workers=2, cache_bytes=100_001)
    try:
        resumed.prepare_window(rows)
        assert [resumed.sft_process_sample(row) for row in rows] == expected
        resumed.finish_window()
        assert resumed.preprocessing_stats["samples"]["hits"] == 0
        assert resumed.preprocessing_stats["timing"]["processing_calls"] == 2
    finally:
        resumed.close()


class _Pool:
    def __init__(self, **kwargs):
        self.spec = kwargs["initargs"][0]
        self.calls = []
        self._processes = {}

    def submit(self, function, target):
        future = Future()
        self.calls.append((target, future))
        return future

    def shutdown(self, **kwargs):
        pass


def _reply(pipeline, worker, calls, sample):
    stats = copy.deepcopy(pipeline._worker_stats[worker])
    stats["timing"] = {"processing_calls": calls, "processing_seconds": calls * 0.25}
    return sample, stats


def test_out_of_order_completions_still_consume_order_and_sum_latest_worker_counters(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(prep.os, "sched_getaffinity", lambda _pid: {2, 3, 4, 5})
    packer, pipeline = _local_pipeline(tmp_path, prefetch=True, workers=2)
    rows = [_text_row("A"), _text_row("BB")]
    expected = [packer.sft_process_sample(row) for row in rows]
    try:
        for calls in (1, 2):
            pipeline.prepare_window(rows)
            # Second completion arrives first. It must not advance the cursor.
            pipeline._executors[1].calls[-1][1].set_result(_reply(pipeline, 1, calls, expected[1]))
            with pytest.raises(ValueError, match="row order"):
                pipeline.sft_process_sample(rows[1])
            with pytest.raises(RuntimeError, match="completely consumed"):
                pipeline.finish_window()
            pipeline._executors[0].calls[-1][1].set_result(_reply(pipeline, 0, calls, expected[0]))
            assert [pipeline.sft_process_sample(row) for row in rows] == expected
            pipeline.finish_window()
        assert pipeline.preprocessing_stats["timing"]["processing_calls"] == 4
        assert pipeline.preprocessing_stats["timing"]["processing_seconds"] == 1.0
    finally:
        pipeline.close()


def test_distinct_questions_on_identical_media_share_worker_and_total_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(prep.os, "sched_getaffinity", lambda _pid: {2, 3})
    _, pipeline = _local_pipeline(tmp_path, prefetch=True, workers=2, cache_bytes=99)
    first, hashes = _video_row(tmp_path, "A")
    second, _ = _video_row(tmp_path, "BB")
    pipeline._cache.media = prep._MediaIdentity(hashes)
    try:
        pipeline.prepare_window([first, second])
        assert sorted(len(pool.calls) for pool in pipeline._executors) == [0, 2]
        assert sum(pool.spec.cache_bytes for pool in pipeline._executors) == 99
    finally:
        pending = [future for _, _, future in pipeline._pending]
        pipeline.close()
    assert all(future.cancelled() for future in pending)


def _routing_pipeline(tmp_path, monkeypatch, rows=8):
    monkeypatch.setattr(prep, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(prep.os, "sched_getaffinity", lambda _pid: {2, 3, 4, 5})
    _, pipeline = _local_pipeline(tmp_path, prefetch=True, workers=2)
    pipeline._max_pending = rows
    targets, hashes = [], {}
    for index in range(rows):
        path = tmp_path / f"media-{index}.mp4"
        path.write_bytes(f"distinct fixture media {index}".encode())
        target = _text_row(str(index))
        target.messages[0]["content"] = [{"type": "video", "video": str(path)}]
        targets.append(target)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    pipeline._cache.media = prep._MediaIdentity(hashes)
    return pipeline, targets


def _complete_routing_window(pipeline, rows):
    for index, (_, worker, future) in enumerate(pipeline._pending):
        future.set_result(_reply(pipeline, worker, index + 1, {"ordinal": index}))
    assert [pipeline.sft_process_sample(row) for row in rows] == [{"ordinal": i} for i in range(len(rows))]
    pipeline.finish_window()
    assert not pipeline._window_owners and pipeline._window_assignments == [0, 0]


def test_unseen_media_balance_each_window_without_changing_order_or_existing_owners(tmp_path, monkeypatch):
    pipeline, rows = _routing_pipeline(tmp_path, monkeypatch)
    try:
        pipeline.prepare_window(rows)
        assert [worker for _, worker, _ in pipeline._pending] == [0, 1] * 4
        assert pipeline._window_assignments == [4, 4]
        owners = dict(pipeline._media_owners)
        _complete_routing_window(pipeline, rows)
        # Reverse consumption order cannot migrate established media owners.
        pipeline.prepare_window(rows[::-1])
        assert [worker for _, worker, _ in pipeline._pending] == [1, 0] * 4
        assert dict(pipeline._media_owners) == owners
        _complete_routing_window(pipeline, rows[::-1])
    finally:
        pipeline.close()
    assert not pipeline._media_owners and not pipeline._window_owners


def test_same_window_owner_survives_lru_eviction_and_later_reassignment_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "MEDIA_OWNER_CAPACITY", 2)
    pipeline, rows = _routing_pipeline(tmp_path, monkeypatch, rows=4)
    window = [rows[0], rows[1], rows[2], rows[0]]
    try:
        pipeline.prepare_window(window)
        assert [worker for _, worker, _ in pipeline._pending] == [0, 1, 0, 0]
        assert len(pipeline._media_owners) == 2 and len(pipeline._window_owners) == 3
        _complete_routing_window(pipeline, window)
        # B was evicted. A retains its owner; B may be reassigned at this new
        # update boundary, with unchanged ordered samples and cache byte limits.
        pipeline.prepare_window([rows[1], rows[0]])
        assert [worker for _, worker, _ in pipeline._pending] == [0, 0]
        assert len(pipeline._media_owners) == 2
        _complete_routing_window(pipeline, [rows[1], rows[0]])
    finally:
        pipeline.close()


def test_no_media_balances_current_window_and_failed_submission_clears_transient_owners(tmp_path, monkeypatch):
    pipeline, rows = _routing_pipeline(tmp_path, monkeypatch)
    try:
        text = [_text_row(str(i)) for i in range(8)]
        pipeline.prepare_window(text)
        assert [worker for _, worker, _ in pipeline._pending] == [0, 1] * 4
        assert not pipeline._media_owners
        _complete_routing_window(pipeline, text)

        def fail(*args):
            raise BrokenProcessPool("owned worker failed")

        pipeline._executors[1].submit = fail
        with pytest.raises(BrokenProcessPool):
            pipeline.prepare_window(rows)
        assert pipeline._failed and not pipeline._window_owners and pipeline._window_assignments == [0, 0]
        with pytest.raises(RuntimeError, match="failed"):
            pipeline.sft_process_sample(rows[0])
    finally:
        pipeline.close()


def test_real_dead_worker_poisoned_window_cannot_complete_or_start_another(tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("two inherited CPUs required")
    _, pipeline = _local_pipeline(tmp_path, prefetch=True, workers=2)
    try:
        with pytest.raises(BrokenProcessPool):
            pipeline._executors[1].submit(os._exit, 17).result(timeout=30)
        with pytest.raises(BrokenProcessPool):
            pipeline.prepare_window([_text_row("A"), _text_row("B")])
        with pytest.raises(RuntimeError, match="completely consumed"):
            pipeline.finish_window()
        with pytest.raises(RuntimeError):
            pipeline.prepare_window([_text_row("C")])
        with pytest.raises(RuntimeError, match="failed"):
            pipeline.sft_process_sample(_text_row("A"))
    finally:
        pipeline.close()


@pytest.mark.parametrize("workers", [True, 0, 3, 1.5, "2"])
def test_worker_count_rejects_unsupported_values(workers):
    with pytest.raises(ValueError, match="preprocessing_workers"):
        SFTConfig(preprocessing_prefetch=True, preprocessing_workers=workers)


def test_two_workers_require_prefetch_and_enough_inherited_cpus(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requires preprocessing_prefetch"):
        SFTConfig(preprocessing_workers=2)
    monkeypatch.setattr(prep.os, "sched_getaffinity", lambda _pid: {0})
    with pytest.raises(ValueError, match="at least one allowed CPU per worker"):
        _local_pipeline(tmp_path, prefetch=True, workers=2)
