# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in SampleBatcher implementations."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Callable, Iterator, Optional

from cosmos_framework.data.vfm.dataflow.base import SampleBatcher


class SimpleBatcher(SampleBatcher):
    """Fixed-size batching — stock DataLoader behavior. Never needs sample_size."""

    def __init__(self, batch_size: int, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.drop_last = drop_last

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        buf: list[dict] = []
        for s in samples:
            buf.append(s)
            if len(buf) == self.batch_size:
                yield buf
                buf = []
        if buf and not self.drop_last:
            yield buf


class _Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PoolPackingBatcher(SampleBatcher):
    """Pool-based greedy bin-packing batcher (re-homed from PackingIterableDataset).

    Buffers ``pool_size`` samples and assembles each batch by greedily selecting
    candidates that fit within the padded token budget, never mixing modalities
    within a batch. ``sample_size`` defaults to ``len(sample["input_ids"])``;
    pass ``size_fn`` to override, or subclass and override the method.
    """

    def __init__(
        self,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        batching_strategy: str = "prefer_closest",
        apply_long_sample_halving: bool = True,
        size_fn: Optional[Callable[[dict], int]] = None,
    ):
        assert batching_strategy in ("prefer_first", "prefer_closest"), (
            f"batching_strategy must be 'prefer_first' or 'prefer_closest', got {batching_strategy!r}"
        )
        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.max_batch_size = max_batch_size
        self.long_threshold = long_threshold
        self.batching_strategy = batching_strategy
        self.apply_long_sample_halving = apply_long_sample_halving
        self._size_fn = size_fn

    def sample_size(self, sample: dict) -> int:
        if self._size_fn is not None:
            return self._size_fn(sample)
        return int(sample["input_ids"].shape[0])

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        pool: deque[dict] = deque()
        src = iter(samples)
        exhausted = False
        while True:
            while not exhausted and len(pool) < self.pool_size:
                try:
                    pool.append(next(src))
                except StopIteration:
                    exhausted = True
            if not pool:
                return
            yield self._best_fit_batch(pool)

    def _max_tokens(self, cur_max: int) -> int:
        if not self.apply_long_sample_halving:
            return self.max_tokens
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    @staticmethod
    def _get_modality(sample: dict) -> "_Modality":
        if "pixel_values" in sample:
            return _Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return _Modality.VIDEO
        return _Modality.TEXT

    @staticmethod
    def _padded_cost(cur_max: int, k: int) -> int:
        return cur_max * k

    def _best_fit_batch(self, pool: deque) -> list[dict]:
        seed = pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.sample_size(seed)
        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]
        chosen = [seed]
        cur_max = L0
        while pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            best_idx = self._find_best_candidate(pool, cur_max, len(chosen), seed_modality)
            if best_idx is None:
                break
            cand = self._remove_from_pool(pool, best_idx)
            chosen.append(cand)
            cur_max = max(cur_max, self.sample_size(cand))
        return chosen

    def _find_best_candidate(self, pool, cur_max, num_chosen, seed_modality):
        if self.batching_strategy == "prefer_first":
            return self._find_best_candidate_prefer_first(pool, cur_max, num_chosen, seed_modality)
        return self._find_best_candidate_prefer_closest(pool, cur_max, num_chosen, seed_modality)

    def _find_best_candidate_prefer_first(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                if best_new_tokens is None or new_tokens < best_new_tokens:
                    best_new_tokens = new_tokens
                    best_idx = idx
        return best_idx

    def _find_best_candidate_prefer_closest(self, pool, cur_max, num_chosen, seed_modality):
        best_idx = None
        best_new_tokens = None
        smallest_length_diff = None
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_max = max(cur_max, L)
            new_tokens = self._padded_cost(new_max, num_chosen + 1)
            if new_tokens <= self._max_tokens(cur_max):
                length_diff = abs(L - cur_max)
                if (
                    best_new_tokens is None
                    or new_tokens < best_new_tokens
                    or (new_tokens == best_new_tokens and length_diff < smallest_length_diff)
                ):
                    best_new_tokens = new_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff
        return best_idx

    @staticmethod
    def _remove_from_pool(pool: deque, idx: int) -> dict:
        if idx == 0:
            return pool.popleft()
        elif idx == len(pool) - 1:
            return pool.pop()
        else:
            pool.rotate(-idx)
            item = pool.popleft()
            pool.rotate(idx)
            return item
