# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Final metric publication through the real OneLogger callback and timer lifecycle."""

from typing import Any

import pytest

from cosmos_framework.utils.one_logger.one_logger_utils import OneLoggerUtils


class _MetricSink:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    def store_has_key(self, key: str) -> bool:
        return key in self.store

    def store_set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def store_get(self, key: str) -> Any:
        return self.store[key]

    def log_metrics(self, values: dict[str, Any]) -> None:
        self.summary.update(values)

    def log_app_tag(self, value: str) -> None:
        del value


def _logger(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _MetricSink]:
    # The legacy callback decorators erase callable signatures; exercise their real runtime behavior.
    sink = _MetricSink()

    def connect(instance: OneLoggerUtils) -> None:
        setattr(instance, "one_logger", sink)

    monkeypatch.setattr(OneLoggerUtils, "_set_one_logger", connect)
    instance = OneLoggerUtils(
        {
            "enable_for_current_rank": True,
            "one_logger_project": "unit-test",
            "one_logger_run_name": "final-counters",
            "one_logger_async": False,
            "log_every_n_train_iterations": 50,
            "app_tag_run_name": "final-counters",
            "app_tag_run_version": "1",
            "world_size": 1,
            "global_batch_size": 8,  # Across ranks per batch; accumulation repeats the batch hooks.
            "batch_size": 8,
            "micro_batch_size": 8,
            "app_tag": "unit-test",
            "train_iterations_target": 1000,
            "train_samples_target": 8000,
            "is_baseline_run": False,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": False,
            "is_test_iterations_enabled": False,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": True,
            "flops_per_sample": 100,
            "save_checkpoint_strategy": "sync",
        }
    )
    return instance, sink


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("start", "steps", "microbatches", "checkpoint"),
    [(0, 32, 1, True), (100, 2, 3, True), (0, 32, 1, False), (0, 50, 1, False), (17, 0, 1, False)],
)
def test_train_end_flushes_completed_counters(
    monkeypatch: pytest.MonkeyPatch, start: int, steps: int, microbatches: int, checkpoint: bool
) -> None:
    logger, sink = _logger(monkeypatch)
    initial_samples = start * microbatches * 8
    logger.on_train_start(train_iterations_start=start, train_samples_start=initial_samples)
    for step in range(steps):
        for _ in range(microbatches):
            logger.on_train_step_start()
            logger.on_train_batch_start()
            logger.on_train_batch_end()
        if checkpoint and step == steps - 1:
            # The native trainer saves before ending the final step timer.
            logger.on_save_checkpoint_start(global_step=start + steps)
            logger.on_save_checkpoint_end(global_step=start + steps)
        logger.on_train_step_end()
    stored_steps = sink.store["train_iterations"]
    logger.on_train_end()
    assert sink.summary["train_iterations_end"] == start + steps
    assert sink.summary["train_iterations"] == stored_steps == steps
    assert sink.store["train_iterations"] == stored_steps, "Publishing must not count another update"
    assert sink.summary["train_samples_end"] == initial_samples + steps * microbatches * 8
    assert sink.summary["train_iterations_time_total"] == sink.store["train_iterations_time_total"]
    assert sink.summary["train_batch_time_total"] == sink.store["train_batch_time_total"]
    assert sink.summary["app_train_loop_finish_time"] > 0
