# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline supervised trajectories and finite, zero-weight-padded held-out batches."""

from __future__ import annotations

import math
from typing import Any

from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


class SFTDatasetEnv:
    def __init__(
        self,
        dataset: Any,
        *,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        training: bool = True,
    ) -> None:
        if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("invalid SFT data-parallel coordinates")
        if not len(dataset):
            raise ValueError("SFT dataset is empty")
        if training and len(dataset) % world_size:
            raise ValueError("SFT training dataset length must be divisible by the DP world size; no rows are dropped")
        self.dataset: Any = dataset
        self.rank: int = rank
        self.world_size: int = world_size
        self.training: bool = training
        self.source_fingerprint: str = dataset.cursor_fingerprint()
        self.cursor: IndexedCursor = IndexedCursor(
            dataset,
            shard_index=rank,
            shard_count=world_size,
            shuffle=shuffle,
            seed=seed,
            cursor_resume="strict",
            label="SFT source",
        )
        self._loaded_checkpoint_provenance: dict[str, int | str] | None = None

    def _trajectory(self, index: int, *, weight: float | None = None) -> Trajectory:
        from cosmos_framework.posttrain.schema import SFT_OPENER_SELF, SFT_SUPERVISE_ALL, SFTTarget

        row = self.dataset[index]
        transition = Transition(
            obs=ObsBundle(),
            # The row's own supervision and opener rules (``normalize_row`` refused any other value); absent = every
            # assistant turn, the opener the student's.
            targets=SFTTarget(
                messages=row["messages"],
                tools=row.get("tools"),
                supervise=row.get("supervise", SFT_SUPERVISE_ALL),
                opener=row.get("opener", SFT_OPENER_SELF),
            ),
        )
        if weight is not None:
            transition.algo_extra["validation_weight"] = weight
        return Trajectory(id=f"{self.source_fingerprint}:{index}", transitions=[transition], status=Status.COMPLETED)

    def get_samples(self, count: int) -> list[Trajectory]:
        if not self.training:
            raise RuntimeError("held-out SFT sources must use finite validation batches")
        if type(count) is not int or count < 1:
            raise ValueError("SFT sample count must be a positive integer")
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        return run_rank_converged(
            lambda: [self._trajectory(self.cursor.next_index()) for _ in range(count)], "SFT training batch"
        )

    def validation_batch_count(self, batch_size: int, max_samples: int | None = None) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("validation batch size must be positive")
        if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
            raise ValueError("validation max_samples must be positive or None")
        count = min(len(self.dataset), max_samples) if max_samples is not None else len(self.dataset)
        return math.ceil(count / (batch_size * self.world_size))

    def get_validation_batch(
        self, batch_index: int, batch_size: int, max_samples: int | None = None
    ) -> list[Trajectory]:
        batches = self.validation_batch_count(batch_size, max_samples)
        if type(batch_index) is not int or not 0 <= batch_index < batches:
            raise ValueError(f"validation batch index must be in [0, {batches})")
        count = min(len(self.dataset), max_samples) if max_samples is not None else len(self.dataset)
        start = (batch_index * self.world_size + self.rank) * batch_size
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        return run_rank_converged(
            lambda: [
                self._trajectory(index if index < count else 0, weight=float(index < count))
                for index in range(start, start + batch_size)
            ],
            "SFT validation batch",
        )

    def source_identity(self) -> str:
        return self.source_fingerprint

    def state_dict(self) -> dict[str, Any]:
        """Collect rank cursors: the LLM engine persists control-owned data components verbatim."""
        from cosmos_framework.posttrain.utils.dist import run_rank_converged
        from cosmos_framework.posttrain.utils.rank_state import collect_rank_state

        local = run_rank_converged(self.cursor.state_dict, "SFT cursor snapshot")
        state = collect_rank_state(local)
        if state is None:
            raise RuntimeError("SFT requires an exact data cursor")
        return state

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank

        local = state_for_current_rank(state)
        if local is None:
            raise ValueError("SFT checkpoint is missing its data cursor")
        return self.cursor.load_state_dict(local)

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        self._loaded_checkpoint_provenance = dict(provenance)

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        return self._loaded_checkpoint_provenance


def build_sft_env(
    config: SFTDatasetConfig, seed: int, *, training: bool = True, rank: int = 0, world_size: int = 1
) -> SFTDatasetEnv:
    return SFTDatasetEnv(
        build_sft_dataset(config),
        rank=rank,
        world_size=world_size,
        shuffle=config.shuffle,
        seed=seed,
        training=training,
    )
