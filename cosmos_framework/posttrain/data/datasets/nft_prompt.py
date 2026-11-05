# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic prompt-only image source for online DiffusionNFT."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


class NFTPromptImageDataLoader:
    """Replay a frozen prompt JSONL as batch-size-one T2I shape templates.

    The generated image, not the placeholder pixels, is scored and optimized. The
    zero image exists only to carry the native Cosmos3 image batch schema and the
    requested generation geometry. ``set_start_iteration`` addresses source rows,
    so the WFM source can restore an exact optimizer-window cursor.
    """

    def __init__(
        self,
        path: str,
        *,
        prompt_key: str = "prompt",
        height: int = 720,
        width: int = 720,
        fps: float = 24.0,
        expected_sha256: str = "",
        expected_rows: int = 0,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        num_replicas: int = 1,
        source_rank: int = 0,
        drop_last: bool = True,
    ) -> None:
        if not path.strip():
            raise ValueError("NFT T2I prompt path must be non-empty")
        if not prompt_key.strip():
            raise ValueError("NFT T2I prompt_key must be non-empty")
        if height <= 0 or width <= 0:
            raise ValueError(f"NFT T2I dimensions must be positive, got {height}x{width}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"NFT T2I fps must be finite and positive, got {fps}")
        if expected_rows < 0:
            raise ValueError(f"NFT T2I expected_rows must be non-negative, got {expected_rows}")
        if num_replicas < 1:
            raise ValueError(f"NFT T2I num_replicas must be positive, got {num_replicas}")
        if source_rank < 0 or source_rank >= num_replicas:
            raise ValueError(f"NFT T2I source_rank={source_rank} is outside [0, {num_replicas})")
        self.path = str(Path(path).expanduser().resolve())
        self.prompt_key = prompt_key
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.num_replicas = int(num_replicas)
        self.source_rank = int(source_rank)
        self.drop_last = bool(drop_last)
        self._rows, self._file_sha256 = self._read_rows()
        if expected_sha256 and self._file_sha256 != expected_sha256:
            raise ValueError(f"NFT T2I prompt SHA-256 differs: expected {expected_sha256}, got {self._file_sha256}")
        if expected_rows and len(self._rows) != expected_rows:
            raise ValueError(f"NFT T2I prompt row count differs: expected {expected_rows}, got {len(self._rows)}")
        self._samples_per_epoch = len(
            torch.utils.data.DistributedSampler(
                self._rows,
                num_replicas=self.num_replicas,
                rank=self.source_rank,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
                drop_last=self.drop_last,
            )
        )
        if self._samples_per_epoch == 0:
            raise ValueError(
                "NFT T2I source rank receives no prompt rows; "
                f"rows={len(self._rows)}, num_replicas={self.num_replicas}, drop_last={self.drop_last}"
            )
        self._start_row = 0

    def _read_rows(self) -> tuple[list[str], str]:
        source = Path(self.path)
        if not source.is_file():
            raise FileNotFoundError(f"NFT T2I prompt JSONL does not exist: {source}")
        payload = source.read_bytes()
        prompts: list[str] = []
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid NFT T2I JSON on line {line_number} of {source}") from error
            if not isinstance(row, dict):
                raise ValueError(f"NFT T2I line {line_number} must be a JSON object")
            prompt = row.get(self.prompt_key)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"NFT T2I line {line_number} needs a non-empty string field {self.prompt_key!r}")
            prompts.append(prompt)
        if not prompts:
            raise ValueError(f"NFT T2I prompt JSONL contains no usable rows: {source}")
        return prompts, hashlib.sha256(payload).hexdigest()

    def set_start_iteration(self, iteration: int) -> None:
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError(f"NFT T2I start row must be a non-negative integer, got {iteration!r}")
        self._start_row = iteration

    def cursor_fingerprint(self) -> str:
        """Bind resume to the exact ordered prompt artifact and image geometry."""
        contract = {
            "file_sha256": self._file_sha256,
            "prompt_key": self.prompt_key,
            "rows": len(self._rows),
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "num_replicas": self.num_replicas,
            "source_rank": self.source_rank,
            "drop_last": self.drop_last,
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _epoch_indices(self, epoch: int) -> list[int]:
        sampler = torch.utils.data.DistributedSampler(
            self._rows,
            num_replicas=self.num_replicas,
            rank=self.source_rank,
            shuffle=self.shuffle,
            seed=self.shuffle_seed,
            drop_last=self.drop_last,
        )
        sampler.set_epoch(epoch)
        return list(sampler)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        row = self._start_row
        epoch = -1
        epoch_indices: list[int] = []
        while True:
            next_epoch, offset = divmod(row, self._samples_per_epoch)
            if next_epoch != epoch:
                epoch = next_epoch
                epoch_indices = self._epoch_indices(epoch)
            prompt = self._rows[epoch_indices[offset]]
            image = torch.zeros((3, self.height, self.width), dtype=torch.uint8)  # [C,H,W]
            yield {
                # Native OmniMoT image batches are batch -> image list -> [C,H,W].
                "images": [[image]],
                "ai_caption": prompt,
                "image_size": [
                    torch.tensor(
                        [self.height, self.width, self.height, self.width],
                        dtype=torch.float32,
                    )
                ],  # one sample [4]
                "conditioning_fps": torch.tensor([self.fps], dtype=torch.bfloat16),  # [1]
            }
            row += 1
