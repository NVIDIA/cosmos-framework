# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.transfer import build_control_cfg_postprocess
from cosmos_framework.model.generator.mot.diffusion_cache import DiffusionCache, _velocity_pathways
from cosmos_framework.model.generator.omni_mot_model import VelocityPostprocess
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("interval", [None, [0.25, 0.75]])
def test_control_cfg_declares_actual_cfg_branches(cache_enabled: bool, interval: list[float] | None) -> None:
    """Declare every executed branch, including when caching is off, for FSDP alignment."""
    control = torch.zeros(2, 3)
    target = torch.ones(4, 3)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    no_control = torch.zeros(target.numel())
    model = SimpleNamespace(_get_velocity=Mock(return_value=[no_control]))
    if cache_enabled:
        model._diffusion_cache = Mock()
    builder = build_control_cfg_postprocess(control_guidance=2.0, control_guidance_interval=interval)
    assert builder is not None
    postprocess = builder(
        model=model,
        cond_tokens=[[7]],
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        gen_data_clean=data,
    )
    assert isinstance(postprocess, VelocityPostprocess)
    noise = torch.zeros(control.numel() + target.numel())
    full_velocity = torch.ones_like(noise)
    for value in (0.9, 0.75, 0.5, 0.25, 0.1):
        timestep = torch.tensor([[value]])
        before = model._get_velocity.call_count
        pathways = postprocess.cfg_branches(timestep)
        assert model._get_velocity.call_count == before
        postprocess([full_velocity], [noise], timestep, 1.0)
        assert model._get_velocity.call_count - before == len(pathways)
        active = interval is None or 0.25 < value < 0.75
        assert pathways == (("cond_no_control",) if active else ())
    if cache_enabled:
        assert model._diffusion_cache.mock_calls == []


def test_unit_control_guidance_uses_single_branch() -> None:
    assert build_control_cfg_postprocess(control_guidance=1.0) is None


@pytest.mark.parametrize("text_guidance", [1.0, 3.0])
@pytest.mark.parametrize("control_guidance", [0.5, 1.5, 2.0])
def test_control_cfg_increment_is_independent_of_text_guidance(text_guidance: float, control_guidance: float) -> None:
    control = torch.zeros(2, 3)
    target = torch.ones(4, 3)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    full = torch.arange(control.numel() + target.numel(), dtype=torch.float32) + 10.0
    no_control = torch.arange(target.numel(), dtype=torch.float32) + 1.0
    unconditional = torch.full_like(full, -2.0)
    noise = torch.zeros_like(full)
    timestep = torch.tensor([[0.5]])
    model = SimpleNamespace(_get_velocity=Mock(return_value=[no_control]))
    builder = build_control_cfg_postprocess(control_guidance=control_guidance)
    assert builder is not None
    postprocess = builder(
        model=model,
        cond_tokens=[[7]],
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        gen_data_clean=data,
    )
    assert postprocess is not None

    adjusted = postprocess([full], [noise], timestep, text_guidance)[0]
    actual = unconditional + text_guidance * (adjusted - unconditional)
    expected = unconditional + text_guidance * (full - unconditional)
    expected[control.numel() :] += (control_guidance - 1.0) * (full[control.numel() :] - no_control)
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="text guidance scale 0.0"):
        postprocess([full], [noise], timestep, 0.0)


class _CacheLanguageModel(torch.nn.Module):
    """Distinct, constant residuals make cross-branch cache reuse observable."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, pack: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls += 1
        return {**pack, "full_only_seq": pack["full_only_seq"] + pack["delta"]}, {}


class _CacheModel:
    def __init__(self) -> None:
        self.net = SimpleNamespace(language_model=_CacheLanguageModel())

    def generate_samples_from_batch(self, **kwargs: Any) -> None:
        pass

    def denoise(self, *, data_batch_packed: Any) -> torch.Tensor:
        output, _ = self.net.language_model(data_batch_packed.pack)
        return output["full_only_seq"].flatten()

    def _get_velocity(
        self,
        *,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        gen_data_clean: GenerationDataClean,
        text_tokens: list[list[int]],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        assert gen_data_clean.x0_tokens_vision is not None
        latents = gen_data_clean.x0_tokens_vision
        conditioned = len(latents) > 1
        delta = 9.0 if text_tokens == [[0]] else (2.0 if conditioned else 5.0)
        batch = SimpleNamespace(
            vision=SimpleNamespace(
                tokens=latents,
                token_shapes=[tuple(latent.shape[-3:]) for latent in latents],
                timesteps=timestep.flatten(),
                condition_mask=[
                    torch.full((1, 1, 1), conditioned and index == 0, dtype=torch.bool) for index in range(len(latents))
                ],
            ),
            pack={
                "causal_seq": torch.zeros(1, 1),
                "full_only_seq": noise_x[0].reshape(-1, 1),
                "delta": delta,
            },
        )
        return [self.denoise(data_batch_packed=batch)]


@pytest.mark.parametrize("text_cfg", [False, True])
def test_control_cfg_cache_reuses_separate_branches_and_refreshes_at_interval_boundaries(text_cfg: bool) -> None:
    """Exercise the Transfer callback through real SeaCache hooks and sampler branch registration."""
    control = torch.ones(2, 1, 2, 2)
    target = torch.zeros_like(control)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    plans = [SequencePlan(has_text=True, has_vision=True)]
    model = _CacheModel()
    cache = DiffusionCache(
        num_steps=6,
        config={"ret_steps": 0, "cutoff_from_end": 0, "max_consecutive_cached": 0},
    )
    cache.install(SimpleNamespace(model=model))
    builder = build_control_cfg_postprocess(control_guidance=2.0, control_guidance_interval=[0.25, 0.75])
    assert builder is not None
    postprocess = builder(model=model, cond_tokens=[[7]], sequence_plans=plans, gen_data_clean=data)
    assert isinstance(postprocess, VelocityPostprocess)

    for step, value in enumerate((0.9, 0.8, 0.6, 0.4, 0.2, 0.1)):
        timestep = torch.tensor([[value]])
        # Fully conditioned control changes must not influence SEA's target indicator.
        control.fill_(float(100 * (step + 1)))
        noise = [torch.cat([control.flatten(), target.flatten()])]
        branches = postprocess.cfg_branches(timestep)
        pathways = _velocity_pathways(text_cfg, cfg_branches=branches)
        cache.begin_step(step, pathways)
        before = model.net.language_model.calls
        full = model._get_velocity(
            noise_x=noise,
            timestep=timestep,
            gen_data_clean=data,
            text_tokens=[[7]],
        )
        text_guidance = 3.0 if text_cfg else 1.0
        mixed = postprocess(full, noise, timestep, text_guidance)
        expected = 2.0 + (2.0 - 5.0) / text_guidance if branches else 2.0
        torch.testing.assert_close(mixed[0][control.numel() :], torch.full((target.numel(),), expected))
        if text_cfg:
            negative = model._get_velocity(
                noise_x=noise,
                timestep=timestep,
                gen_data_clean=data,
                text_tokens=[[0]],
            )
            torch.testing.assert_close(negative[0][control.numel() :], torch.full((target.numel(),), 9.0))
            prediction = negative[0] + text_guidance * (mixed[0] - negative[0])
            # Text-CFG baseline is 9 + 3 * (2 - 9) = -12; control adds 2 - 5 = -3.
            torch.testing.assert_close(
                prediction[control.numel() :], torch.full((target.numel(),), -15.0 if branches else -12.0)
            )
        # First step and both interval boundaries refresh every branch; the next step reuses each one.
        assert model.net.language_model.calls - before == (len(pathways) if step % 2 == 0 else 0)
        assert set(cache._pathways) == set(pathways)
        assert not cache._pending_pathways
