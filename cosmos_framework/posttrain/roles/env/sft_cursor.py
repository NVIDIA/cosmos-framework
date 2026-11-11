# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Finite SFT epochs with explicit, zero-weight update-tail slots.

The existing IndexedCursor owns ordering, dataset identity and dataset-local RNG
state. Every rank advances that same global index stream, but reads only its own
row. Padding never consumes a row or wraps into the following epoch.
"""

from __future__ import annotations

import math
from typing import Any

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor


class FiniteSFTCursor:
    def __init__(self, dataset, *, rank, world_size, shuffle, seed, epochs, samples_per_update):
        for name, value in (("epochs", epochs), ("samples_per_update", samples_per_update)):
            if type(value) is not int or value < 1:
                raise ValueError(f"finite SFT {name} must be a positive integer")
        self.source = IndexedCursor(dataset, shuffle=shuffle, seed=seed, cursor_resume="strict", label="finite SFT")
        self.rows = len(dataset)
        self.rank = rank
        self.world_size = world_size
        self.epochs = epochs
        self.samples_per_update = samples_per_update
        self.updates_per_epoch = math.ceil(self.rows / (world_size * samples_per_update))
        self.slots_per_epoch = self.updates_per_epoch * samples_per_update
        self.position = 0

    @property
    def identity(self):
        return {
            "version": 1,
            "kind": "finite_sft_zero_weight_update_tail",
            "epochs": self.epochs,
            "samples_per_update_per_rank": self.samples_per_update,
            "dataset_rows": self.rows,
            "world_size": self.world_size,
        }

    def next_index(self) -> int | None:
        if self.position >= self.epochs * self.slots_per_epoch:
            raise RuntimeError("finite SFT training coverage is exhausted; no next-epoch rows may be replayed")
        offset = (self.position % self.slots_per_epoch) * self.world_size
        count = min(self.world_size, max(0, self.rows - offset))
        indices = [self.source.next_index() for _ in range(count)]
        self.position += 1
        return indices[self.rank] if self.rank < count else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "finite_sft": self.identity,
            "rank": self.rank,
            "position": self.position,
            "source": self.source.state_dict(),
        }

    def load_state_dict(self, state) -> bool:
        if state.get("finite_sft") != self.identity or state.get("rank") != self.rank:
            raise ValueError("finite SFT cursor coverage/topology identity differs from checkpoint")
        position = state.get("position")
        if type(position) is not int or not 0 <= position <= self.epochs * self.slots_per_epoch:
            raise ValueError("finite SFT cursor position is invalid")
        epoch, slot = divmod(position, self.slots_per_epoch)
        consumed = epoch * self.rows + min(slot * self.world_size, self.rows)
        source = state.get("source")
        expected_epoch, expected_position = (0, 0) if consumed == 0 else divmod(consumed - 1, self.rows)
        if consumed:
            expected_epoch += 1
            expected_position += 1
        if not isinstance(source, dict) or (source.get("epoch"), source.get("position")) != (
            expected_epoch,
            expected_position,
        ):
            raise ValueError("finite SFT source position disagrees with the coverage cursor")
        self.source.load_state_dict(source)
        self.position = position
        return True
