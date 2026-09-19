# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L1: Gaussian noise on the low-resolution conditioning latent of super-resolution samples.

Cascaded generators that condition on their own (or a degraded) low-resolution input add noise to
that conditioning at training time so the model does not copy its artifacts (noise-conditioning
augmentation in Imagen / CDM, FlashVideo's latent blend, SeedVR's LR-latent noise). Here the noise
is applied after VAE encoding to the conditioning vision items of samples that came from the SR
data streams, and only during training.

The mixing follows the rectified-flow interpolation used by the model, ``z' = (1 - t) z + t eps``
with ``t ~ U[t_min, t_max]`` drawn once per sample. With ``persist_over_time`` the same ``eps`` is
shared by every latent frame of the clip (temporal-persist, as in the Transfer1 decoder), otherwise
``eps`` is i.i.d. per latent frame. The applied ``t`` is not fed to the network; a level embedding
is a follow-up.
"""

from __future__ import annotations

from typing import Sequence

import attrs
import torch


@attrs.define(slots=False)
class SRLatentConditionNoiseConfig:
    """Config for L1 latent conditioning noise. ``None`` on the model config disables it."""

    prob: float = 0.8
    """Probability of noising the conditioning latents of an eligible sample."""

    t_min: float = 0.05
    t_max: float = 0.40
    """Range of the interpolation weight ``t`` toward pure noise; calibrated by experiment E3."""

    persist_over_time: bool = True
    """Share one noise realisation across latent frames (True) or draw i.i.d. per frame (False)."""

    dataset_names: tuple[str, ...] = ("video_sr", "image_sr")
    """Only samples whose ``dataset_name`` is in this set are eligible (SR streams only)."""


def sr_sample_mask(dataset_names: Sequence[str] | str | None, batch_size: int, eligible: Sequence[str]) -> list[bool]:
    """Per-sample eligibility from ``data_batch["dataset_name"]`` (a list per sample, or one string)."""
    if dataset_names is None:
        return [False] * batch_size
    if isinstance(dataset_names, str):
        return [dataset_names in eligible] * batch_size
    names = list(dataset_names)
    if len(names) != batch_size:
        raise ValueError(f"dataset_name has {len(names)} entries for batch_size {batch_size}")
    return [str(n) in eligible for n in names]


def apply_sr_latent_condition_noise(
    x0_tokens_vision: list[torch.Tensor],  # flattened items, each [1,C,T,H,W]
    num_vision_items_per_sample: list[int] | None,
    eligible_samples: Sequence[bool],
    cfg: SRLatentConditionNoiseConfig,
    generator: torch.Generator | None = None,
) -> tuple[list[torch.Tensor], list[float | None]]:
    """Noise the conditioning (non-last) vision items of eligible multi-item samples.

    Returns the new latent list (non-eligible items are the same tensor objects) and the ``t``
    used per sample (``None`` when the sample was skipped), for logging.
    """
    batch_size = len(eligible_samples)
    if num_vision_items_per_sample is None:
        # Single-item samples have no conditioning item to noise.
        return list(x0_tokens_vision), [None] * batch_size
    if len(num_vision_items_per_sample) != batch_size:
        raise ValueError(
            f"num_vision_items_per_sample has {len(num_vision_items_per_sample)} entries for batch_size {batch_size}"
        )
    if sum(num_vision_items_per_sample) != len(x0_tokens_vision):
        raise ValueError(f"{len(x0_tokens_vision)} vision items but samples declare {sum(num_vision_items_per_sample)}")
    out = list(x0_tokens_vision)
    applied_t: list[float | None] = []
    offset = 0
    for eligible, num_items in zip(eligible_samples, num_vision_items_per_sample):
        start, end = offset, offset + num_items
        offset = end
        if not eligible or num_items < 2:
            applied_t.append(None)
            continue
        ref = x0_tokens_vision[start]
        draw = torch.rand(2, generator=generator, device=ref.device if generator is None else generator.device)  # [2]
        if float(draw[0]) >= cfg.prob:
            applied_t.append(None)
            continue
        t = cfg.t_min + float(draw[1]) * (cfg.t_max - cfg.t_min)
        applied_t.append(t)
        for item_idx in range(start, end - 1):  # every item but the generated (last) one
            z = x0_tokens_vision[item_idx]  # [1,C,T,H,W]
            if cfg.persist_over_time:
                eps = torch.randn(
                    (z.shape[0], z.shape[1], 1, *z.shape[3:]), generator=generator, device=z.device, dtype=z.dtype
                ).expand_as(z)  # [1,C,T,H,W]
            else:
                eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)  # [1,C,T,H,W]
            out[item_idx] = (1.0 - t) * z + t * eps  # [1,C,T,H,W]
    return out, applied_t
