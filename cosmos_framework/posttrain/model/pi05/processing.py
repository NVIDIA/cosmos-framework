# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Physical Intelligence
# SPDX-License-Identifier: OpenMDW-1.1
"""Stateless pi05_libero transforms, adapted from the pinned RLinf OpenPI path."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from PIL import Image
from torch.nn import functional as F

from cosmos_framework.posttrain.model.pi05.config import PALIGEMMA_VOCAB_SIZE, Pi05ModelConfig


@dataclass(frozen=True)
class Pi05Inputs:
    images: tuple[torch.Tensor, ...]  # three [B,3,224,224] float32 views
    image_masks: torch.Tensor  # [B,3], last view is padding
    tokens: torch.Tensor  # [B,200]
    token_mask: torch.Tensor  # [B,200]
    state: torch.Tensor  # [B,32], normalized; stateless policy does not consume it


@dataclass(frozen=True)
class Pi05ReplayInputs:
    """Compact original-invocation inputs; GPU materialization owns image normalization."""

    images: tuple[torch.Tensor, ...]  # two contiguous [B,224,224,3] uint8 views
    image_masks: torch.Tensor  # [B,3] bool, missing third view is always masked
    tokens: torch.Tensor  # [B,200] int64
    token_mask: torch.Tensor  # [B,200] bool
    state: torch.Tensor  # [B,32] float64, native normalized state


def validate_replay_inputs(inputs: Pi05ReplayInputs) -> int:
    """Validate all invocation rows before selection or GPU preprocessing."""
    if inputs.tokens.ndim != 2 or inputs.tokens.shape[1] != 200 or inputs.tokens.dtype != torch.int64:
        raise ValueError("pi05 replay inputs.tokens must be int64 [B,200]")
    batch = inputs.tokens.shape[0]
    if batch < 1:
        raise ValueError("pi05 replay inputs require a nonempty invocation")
    if len(inputs.images) != 2:
        raise ValueError("pi05 replay inputs.images must contain two views")
    for index, image in enumerate(inputs.images):
        if image.shape != (batch, 224, 224, 3) or image.dtype != torch.uint8:
            raise ValueError(f"pi05 replay inputs.images[{index}] must be uint8 [B,224,224,3]")
    if inputs.image_masks.shape != (batch, 3) or inputs.image_masks.dtype != torch.bool:
        raise ValueError("pi05 replay inputs.image_masks must be bool [B,3]")
    if not bool(inputs.image_masks[:, :2].all()) or bool(inputs.image_masks[:, 2].any()):
        raise ValueError("pi05 replay inputs.image_masks must retain two real views and a missing third view")
    if inputs.token_mask.shape != (batch, 200) or inputs.token_mask.dtype != torch.bool:
        raise ValueError("pi05 replay inputs.token_mask must be bool [B,200]")
    if bool(((inputs.tokens < 0) | (inputs.tokens >= PALIGEMMA_VOCAB_SIZE)).any()):
        raise ValueError(f"pi05 replay inputs.tokens must be within [0, {PALIGEMMA_VOCAB_SIZE})")
    if (
        inputs.state.shape != (batch, 32)
        or inputs.state.dtype != torch.float64
        or not bool(torch.isfinite(inputs.state).all())
    ):
        raise ValueError("pi05 replay inputs.state must be finite float64 [B,32]")
    return batch


def _resize_image(image: np.ndarray) -> torch.Tensor:
    """Apply the OpenPI client's PIL resize to validated HWC uint8 RGB input."""
    height, width = image.shape[:2]
    scale = max(height / 224, width / 224)
    resized = (int(height / scale), int(width / scale))
    pixels = Image.fromarray(image).resize((resized[1], resized[0]), resample=Image.Resampling.BILINEAR)
    pad_h, pad_w = 224 - resized[0], 224 - resized[1]
    padded = Image.new(pixels.mode, (224, 224), 0)
    padded.paste(pixels, (pad_w // 2, pad_h // 2))
    return torch.from_numpy(np.array(padded))


class Pi05Processor:
    def __init__(self, config: Pi05ModelConfig) -> None:
        self.config = config
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=Path(config.tokenizer_path).read_bytes())
        data = json.loads(Path(config.norm_stats_path).read_text())["norm_stats"]
        self.quantiles = {}
        for key, width in (("state", 8), ("actions", 7)):
            low, high = (np.asarray(data[key][name], dtype=np.float64) for name in ("q01", "q99"))
            if low.shape != (32,) or high.shape != low.shape or not np.isfinite([low, high]).all():
                raise ValueError(f"pi05 norm_stats.{key} requires finite q01/q99 arrays of length 32")
            if (high < low).any():
                raise ValueError(f"pi05 norm_stats.{key}.q99 must be >= q01")
            self.quantiles[key] = (low[:width], high[:width])

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        # pi05_libero explicitly disables discrete_state_input.
        ids = self.tokenizer.encode(cleaned, add_bos=True) + self.tokenizer.encode("\n")
        count = min(len(ids), 200)
        return (
            torch.tensor(ids[:200] + [0] * (200 - count), dtype=torch.long),
            torch.arange(200) < count,
        )

    def encode(self, observations: Sequence[Mapping[str, object]]) -> Pi05ReplayInputs:
        if not observations:
            raise ValueError("pi05 observations must be nonempty")
        # Validate the entire request before preprocessing any row.
        for index, observation in enumerate(observations):
            for key in ("image", "wrist_image"):
                image = np.asarray(observation[key])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"pi05 observations[{index}].{key} must be HWC uint8 RGB")
            state = np.asarray(observation["state"])
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError(f"pi05 observations[{index}].state must contain 8 finite values")
            if not isinstance(observation["prompt"], str):
                raise ValueError(f"pi05 observations[{index}].prompt must be a string")
        views = tuple(
            torch.stack([_resize_image(np.asarray(row[key])) for row in observations])
            for key in ("image", "wrist_image")
        )
        tokens, masks = zip(*(self.tokenize(row["prompt"]) for row in observations), strict=True)
        low, high = self.quantiles["state"]
        states = np.stack([(np.asarray(row["state"]) - low) / (high - low + 1e-6) * 2 - 1 for row in observations])
        return Pi05ReplayInputs(
            images=views,
            image_masks=torch.tensor([[True, True, False]]).expand(len(observations), -1),
            tokens=torch.stack(tokens),
            token_mask=torch.stack(masks),
            state=F.pad(torch.as_tensor(states), (0, 24)),
        )

    def materialize(self, inputs: Pi05ReplayInputs, *, device: torch.device | str) -> Pi05Inputs:
        validate_replay_inputs(inputs)
        # Normalize only after transfer; NHWC storage followed by permute keeps
        # the original channels-last convolution layout and FP32 operation order.
        views = tuple(
            image.to(device=device, dtype=torch.float32).contiguous().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            for image in inputs.images
        )
        return Pi05Inputs(
            images=(*views, -torch.ones_like(views[0])),
            image_masks=inputs.image_masks.to(device),
            tokens=inputs.tokens.to(device),
            token_mask=inputs.token_mask.to(device),
            state=inputs.state.to(device),
        )

    def prepare(self, observations: Sequence[Mapping[str, object]], *, device: torch.device | str) -> Pi05Inputs:
        return self.materialize(self.encode(observations), device=device)

    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[1:] != (10, 32):
            raise ValueError("pi05 normalized actions must have shape [B,10,32]")
        low, high = self.quantiles["actions"]
        # NumPy computes the affine shift in native FP32 before the FP64 stats
        # promote its product. Promoting the actions first changes decoded values.
        values = (actions[:, : self.config.action_chunk, :7] + 1.0) / 2.0
        low_tensor = torch.as_tensor(low, device=values.device)
        high_tensor = torch.as_tensor(high, device=values.device)
        return values.to(torch.float64) * (high_tensor - low_tensor + 1e-6) + low_tensor
