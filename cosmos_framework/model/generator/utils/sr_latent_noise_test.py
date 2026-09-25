# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.utils.sr_latent_noise import (
    SRLatentConditionNoiseConfig,
    apply_sr_latent_condition_noise,
    sr_sample_mask,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _latents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    # Sample 0: SR pair (LR small grid, HR big grid). Sample 1: single item. Sample 2: transfer pair.
    return [
        torch.randn(1, 16, 3, 8, 13, generator=g),
        torch.randn(1, 16, 3, 15, 26, generator=g),
        torch.randn(1, 16, 3, 15, 26, generator=g),
        torch.randn(1, 16, 3, 15, 26, generator=g),
        torch.randn(1, 16, 3, 15, 26, generator=g),
    ]


def test_mask_from_dataset_names() -> None:
    eligible = ("video_sr", "image_sr")
    assert sr_sample_mask(["video_sr", "video_data", "video_transfer"], 3, eligible) == [True, False, False]
    assert sr_sample_mask("image_sr", 2, eligible) == [True, True]
    assert sr_sample_mask(None, 2, eligible) == [False, False]
    with pytest.raises(ValueError):
        sr_sample_mask(["video_sr"], 2, eligible)


def test_only_sr_conditioning_items_are_noised_and_generated_items_untouched() -> None:
    latents = _latents()
    cfg = SRLatentConditionNoiseConfig(prob=1.0, t_min=0.3, t_max=0.3)
    gen = torch.Generator().manual_seed(1)
    out, applied = apply_sr_latent_condition_noise(latents, [2, 1, 2], [True, False, False], cfg, generator=gen)
    assert applied[0] == pytest.approx(0.3) and applied[1] is None and applied[2] is None
    assert not torch.equal(out[0], latents[0])  # LR of the SR sample is noised
    for i in (1, 2, 3, 4):
        assert out[i] is latents[i]  # HR of SR sample, single item, transfer pair: untouched
    # z' = 0.7 z + 0.3 eps -> variance 0.49 + 0.09 for unit-variance inputs
    assert out[0].var().item() == pytest.approx(0.58, abs=0.05)


def test_persist_over_time_shares_noise_across_latent_frames() -> None:
    z = torch.zeros(1, 16, 4, 6, 6)
    cfg_persist = SRLatentConditionNoiseConfig(prob=1.0, t_min=1.0, t_max=1.0, persist_over_time=True)
    out, _ = apply_sr_latent_condition_noise([z, z.clone()], [2], [True], cfg_persist, torch.Generator().manual_seed(0))
    assert torch.equal(out[0][:, :, 0], out[0][:, :, 3])  # pure noise, identical over time
    cfg_iid = SRLatentConditionNoiseConfig(prob=1.0, t_min=1.0, t_max=1.0, persist_over_time=False)
    out, _ = apply_sr_latent_condition_noise([z, z.clone()], [2], [True], cfg_iid, torch.Generator().manual_seed(0))
    assert not torch.equal(out[0][:, :, 0], out[0][:, :, 3])


def test_probability_and_range_are_respected() -> None:
    latents = _latents()
    cfg = SRLatentConditionNoiseConfig(prob=0.5, t_min=0.1, t_max=0.2)
    gen = torch.Generator().manual_seed(3)
    ts = []
    for _ in range(200):
        _, applied = apply_sr_latent_condition_noise(latents[:2], [2], [True], cfg, generator=gen)
        ts.append(applied[0])
    skipped = sum(t is None for t in ts)
    assert 60 < skipped < 140  # prob 0.5
    assert all(0.1 <= t <= 0.2 for t in ts if t is not None)


def test_single_item_batches_and_bad_shapes() -> None:
    latents = _latents()
    out, applied = apply_sr_latent_condition_noise(latents, None, [True] * 5, SRLatentConditionNoiseConfig())
    assert all(a is None for a in applied) and all(o is z for o, z in zip(out, latents))
    with pytest.raises(ValueError):
        apply_sr_latent_condition_noise(latents, [2, 2], [True, True], SRLatentConditionNoiseConfig())
