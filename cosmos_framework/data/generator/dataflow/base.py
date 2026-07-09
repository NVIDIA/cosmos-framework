# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The four dataflow role ABCs.

A raw item flows through four independently-swappable roles in a fixed order
enforced by the loader:

    DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator

See docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class DataDistributor(ABC):
    """Owns the raw dataset, shards it disjointly across DP ranks x workers,
    shuffles, and (later) carries checkpoint/resume state."""

    @abstractmethod
    def stream(
        self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int
    ) -> Iterator[Any]:
        """Yield this (rank, worker)'s disjoint slice of raw items, indefinitely."""

    def state_dict(self) -> dict:
        """Resume state. No-op default; resumable distributors override."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore resume state. No-op default; resumable distributors override."""
        return None


class RawItemProcessor(ABC):
    """Transforms one raw dataset item into one training-ready sample dict."""

    @abstractmethod
    def process(self, item: Any) -> dict:
        ...


class SampleBatcher(ABC):
    """Consumes a stream of samples and yields groups (the selection strategy)."""

    @abstractmethod
    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        """Pull from ``samples``; yield one ``list[dict]`` per batch."""

    def sample_size(self, sample: dict) -> int:
        """Per-sample token cost for packing batchers. Non-packing batchers
        never call this; packing batchers override it (or inject a size_fn)."""
        raise NotImplementedError


class BatchCollator(ABC):
    """Collates one group of samples into one batch dict for ``model.forward()``."""

    @abstractmethod
    def collate(self, samples: list[dict]) -> dict:
        ...
