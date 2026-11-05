# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Control barriers for simulator-driven decision windows and fixed-case evaluation."""

from __future__ import annotations

from cosmos_framework.posttrain.utils.entry import resolve_all
from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle, Handle, Scatter, WorkerGroup


def collect_decision_window(collector: WorkerGroup, simulator: WorkerGroup, *, version: int, decisions: int) -> Handle:
    """Keep callback targets free while the simulator drives their synchronous peers."""
    try:
        resolve_all(collector.begin_window(version))
        report = simulator.run_window(policy=collector, version=version, decisions=decisions)
        resolve_all(report)
        resolve_all(collector.validate_window(report))
    except BaseException:
        resolve_all(collector.abort_window())
        raise
    # Only metadata was awaited above; sampled records flow directly to reward/replay.
    return collector.finish_window(report)


def evaluate_decision_cases(
    collector: WorkerGroup,
    simulator: WorkerGroup,
    *,
    version: int,
    cases: list[dict[str, int | str]] | Scatter,
    seed: int | Scatter,
    decisions: int,
) -> Handle:
    """Drain simulator callbacks before finishing on collector actors; preserve per-cell failures."""
    try:
        resolve_all(collector.begin_evaluation(version))
        report = simulator.run_evaluation(
            policy=collector, cases=cases, version=version, seed=seed, decisions=decisions
        )
        if isinstance(report, CollectiveHandle):
            for _ in report.as_completed():
                pass
        else:
            resolve_all(report)
        return collector.finish_evaluation(report, seed)
    finally:
        # Ordered after finish_evaluation, including cells whose simulator result raised
        # during argument resolution before the collector method could run.
        resolve_all(collector.abort_evaluation())
