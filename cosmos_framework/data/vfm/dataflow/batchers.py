# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in SampleBatcher implementations."""

from __future__ import annotations

from typing import Iterator

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
