# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Chunk timing shared by the streaming tokenizers' cost predictors.

The VAEs encode and decode a long clip as a loop over a few distinct chunk shapes, so a call's
time is the sum of its chunks' times. Timing each distinct chunk once, through whatever the
tokenizer dispatches it to (an AOT-compiled package, a ``torch.compile`` graph, or eager), predicts
a clip of any length without running it.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TypeVar

import torch
import torch.distributed as dist

_Key = TypeVar("_Key", bound=Hashable)


def median_cuda_seconds(
    call: Callable[[], object],
    *,
    warmup: int = 3,
    iters: int = 8,
) -> float:
    """Median seconds per ``call()`` on the current CUDA stream, timed with CUDA events.

    The warmup calls absorb one-time costs a steady-state call does not pay (cuDNN autotuning,
    ``torch.compile`` tracing, lazy allocations). Events are recorded without synchronizing and
    read once at the end, so the timed calls queue back to back as they do in a real clip.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) / 1000.0 for start, end in events)
    return samples[len(samples) // 2]


def time_across_ranks(
    measure: Callable[[], Mapping[_Key, float]],
    group: dist.ProcessGroup | None = None,
) -> dict[_Key, float]:
    """Run ``measure`` on every rank of ``group`` (the default group if ``None``); return per-key maxima.

    The slowest rank's time is the one a plan has to respect, and taking it on every rank gives
    every rank the same table, which is what lets them plan identically without communicating.
    A rank whose ``measure`` raises (e.g. out of memory) still joins the exchange, so every rank
    raises rather than waiting on it. Every rank of ``group`` must call this. Outside a
    distributed run, returns what ``measure`` returns.
    """
    local_error: Exception | None = None
    try:
        seconds = dict(measure())
    except Exception as error:  # Re-raised below, once every rank knows.
        local_error, seconds = error, {}
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size(group) == 1:
        if local_error is not None:
            raise local_error
        return seconds
    report = (None if local_error is None else f"{type(local_error).__name__}: {local_error}", seconds)
    gathered: list[tuple[str | None, dict[_Key, float]] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, report, group=group)
    if local_error is not None:
        raise local_error
    result: dict[_Key, float] = {}
    for rank, rank_report in enumerate(gathered):
        assert rank_report is not None
        message, rank_seconds = rank_report
        if message is not None:
            raise RuntimeError(f"Rank {rank} of the group failed while timing chunks: {message}")
        for key, value in rank_seconds.items():
            result[key] = max(result.get(key, 0.0), value)
    return result
