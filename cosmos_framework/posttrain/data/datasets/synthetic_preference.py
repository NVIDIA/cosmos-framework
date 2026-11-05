# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""In-memory synthetic preference pairs — the zero-dependency data source behind the
quick-start toy experiments (`rl.data.synthetic_pairs > 0`).

Replaces the old entry-level fallback that WROTE synthetic `.pt` files + a jsonl to the
shared out_dir: that required rank-0-writes-others-wait coordination (RANK/WORLD_SIZE
env plumbing and an atomic-rename handshake in `entrypoints/dpo.py::main`). Generating
frames deterministically from per-pair seeds needs NO files and NO cross-rank
coordination — every rank constructs the identical dataset independently, and the
distributed sampler shards it exactly like a real one.

Contract: identical to `DPOPreferenceDataset` (it IS one — only the entry source and the
frame source are overridden), so the env/trainer/objective are untouched. Frames are
uint8 noise at 256x256; the parent applies the same bucket-snap and `_side` formatting
as for real data, so the toy path exercises the full preprocessing stack.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.posttrain.data.datasets.dpo_preference import DPOPreferenceDataset


class SyntheticPreferenceDataset(DPOPreferenceDataset):
    """`n_pairs` deterministic synthetic pairs. Entries carry integer SEEDS in the
    winner/loser slots; `_read_frames` turns a seed into uint8 noise frames. Same seed
    ⇒ bit-identical data on every rank/process — no files, no coordination."""

    def __init__(
        self,
        n_pairs: int = 64,
        resolution: str | None = "256",
        num_frames: int = 9,
        media: str = "video",
        cfg_dropout_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        if n_pairs < 1:
            raise ValueError(f"SyntheticPreferenceDataset: n_pairs must be >= 1, got {n_pairs}")
        self._n_pairs = int(n_pairs)
        self._data_seed = int(seed)
        super().__init__(
            jsonl_path="",  # unused: _read_entries is overridden
            resolution=resolution,
            num_frames=num_frames,
            media=media,
            cfg_dropout_rate=cfg_dropout_rate,
            seed=seed,
        )

    def _read_entries(self, jsonl_path: str, **_filters: float) -> list[dict[str, Any]]:
        # Winner/loser slots hold per-side generator seeds (consumed by _read_frames).
        return [
            {
                "prompt": f"a synthetic sample {i}",
                self.winner_video_key: self._data_seed * 100_003 + 2 * i + 1,
                self.loser_video_key: self._data_seed * 100_003 + 2 * i + 2,
                "score_gap": 1.0,
                "fps": 24,
            }
            for i in range(self._n_pairs)
        ]

    def _read_frames(self, spec: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(spec))
        shape = (3, 256, 256) if self.media == "image" else (3, self.num_frames, 256, 256)
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)

    def _membership_media_identities(
        self,
        entry: dict[str, Any],
        side: str,
        media_key: str,
    ) -> tuple[str, ...]:
        """Synthetic media is a pure function of its generator seed and shape contract."""
        del side
        return (f"synthetic-preference-v1:{self.media}:{self.num_frames}:{int(entry[media_key])}",)
