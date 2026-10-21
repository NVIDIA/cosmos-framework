# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic map-style arithmetic source for text and multimodal examples."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np

from cosmos_framework.posttrain.config.rl import normalize_video_fps


class DeterministicArithmeticDataset:
    """Deterministic map-style arithmetic rows for the shared ``DatasetEnv``."""

    def __init__(
        self,
        *,
        seed: int = 0,
        length: int = 1,
        max_operand: int = 50,
        modality: str = "text",
        image_size: int = 64,
        video_frames: int = 4,
        video_fps: float = 24.0,
        fingerprint_prefix: str = "deterministic-arithmetic",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        if modality not in ("text", "image", "video"):
            raise ValueError(f"modality must be 'text', 'image', or 'video', got {modality!r}")
        if not isinstance(fingerprint_prefix, str) or not fingerprint_prefix:
            raise ValueError("fingerprint_prefix must be a non-empty string")
        for name, value in (
            ("length", length),
            ("max_operand", max_operand),
            ("image_size", image_size),
            ("video_frames", video_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.seed: int = seed
        self.length: int = length
        self.max_operand: int = max_operand
        self.modality: str = modality
        self.image_size: int = image_size
        self.video_frames: int = video_frames
        self.video_fps: float = normalize_video_fps(video_fps)
        self.fingerprint_prefix: str = fingerprint_prefix

    def __len__(self) -> int:
        return self.length

    def _image(self, color_seed: int) -> Any:
        from PIL import Image

        color = tuple((color_seed + offset) % 256 for offset in (17, 83, 149))
        pixels = np.empty((self.image_size, self.image_size, 3), dtype=np.uint8)  # [H,W,3]
        pixels[:, :] = color  # [H,W,3]
        return Image.fromarray(pixels, mode="RGB")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < self.length:
            raise IndexError(index)
        digest = hashlib.blake2b(
            f"{self.seed}:{index}".encode("ascii"),
            digest_size=16,
            person=b"ppo-arithmetic",
        ).digest()
        rng = random.Random(int.from_bytes(digest, byteorder="big"))
        a = rng.randint(0, self.max_operand)
        b = rng.randint(0, self.max_operand)
        color_seed = rng.randint(0, 255)
        sample_id = f"ppo-{self.seed:08d}-{index:012d}"
        representations: dict[str, Any] = {
            "prompt": f"Return only the integer answer. {a} + {b} =",
            "reference": a + b,
            "source_id": sample_id,
        }
        if self.modality == "image":
            representations["image"] = self._image(color_seed)
            representations["image_id"] = f"{sample_id}-image"
        elif self.modality == "video":
            representations["video"] = [
                self._image(color_seed + frame_index * 19) for frame_index in range(self.video_frames)
            ]
            representations["video_id"] = f"{sample_id}-video"
            representations["video_fps"] = self.video_fps
        return representations

    def cursor_fingerprint(self) -> str:
        """Return the stable synthetic-source identity used by ``IndexedCursor``."""
        payload = {
            "seed": self.seed,
            "length": self.length,
            "max_operand": self.max_operand,
            "modality": self.modality,
            "image_size": self.image_size,
            "video_frames": self.video_frames,
            "video_fps": self.video_fps,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return f"{self.fingerprint_prefix}-sha256=" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["DeterministicArithmeticDataset"]
