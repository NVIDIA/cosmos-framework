# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.transfer import build_control_cfg_postprocess
from cosmos_framework.model.generator.omni_mot_model import VelocityPostprocess
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _control_cfg_fixture(
    *, control_guidance_interval: list[float] | None = None
) -> tuple[VelocityPostprocess, SimpleNamespace, torch.Tensor]:
    control = torch.zeros(2, 3)
    target = torch.ones(4, 3)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    cache = Mock()
    model = SimpleNamespace(
        _diffusion_cache=cache,
        _get_velocity=Mock(return_value=[torch.zeros(target.numel())]),
        tensor_kwargs={},
    )
    builder = build_control_cfg_postprocess(
        control_guidance=2.0,
        control_guidance_interval=control_guidance_interval,
    )
    assert builder is not None
    postprocess = builder(
        model=model,
        cond_tokens=[[7]],
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        gen_data_clean=data,
    )
    assert isinstance(postprocess, VelocityPostprocess)
    return postprocess, model, torch.arange(control.numel() + target.numel(), dtype=torch.float32)


@pytest.mark.L0
@pytest.mark.CPU
def test_control_cfg_prepares_all_cache_pathways_before_forward() -> None:
    postprocess, model, noise = _control_cfg_fixture()
    timestep = torch.tensor([0.5])

    postprocess.prepare([noise], timestep)

    model._diffusion_cache.prepare_control_cfg_step.assert_called_once()
    branches, timestep_key = model._diffusion_cache.prepare_control_cfg_step.call_args.args
    assert list(branches) == ["cfg0", "cfg1", "cfg2"]
    assert len(branches["cfg0"]) == 2
    assert len(branches["cfg1"]) == 1
    assert len(branches["cfg2"]) == 2
    torch.testing.assert_close(branches["cfg0"][-1], branches["cfg1"][0])
    assert timestep_key == pytest.approx(0.5)


@pytest.mark.L0
@pytest.mark.CPU
def test_control_cfg_interval_keeps_stable_negative_pathway_key() -> None:
    postprocess, model, noise = _control_cfg_fixture(control_guidance_interval=[0.0, 0.4])
    timestep = torch.tensor([0.5])

    postprocess.prepare([noise], timestep)
    result = postprocess([noise], [noise], timestep)

    branches = model._diffusion_cache.prepare_control_cfg_step.call_args.args[0]
    assert list(branches) == ["cfg0", "cfg2"]
    assert result[0] is noise
    model._get_velocity.assert_not_called()
