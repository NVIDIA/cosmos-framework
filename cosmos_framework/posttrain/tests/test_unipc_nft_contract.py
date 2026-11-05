# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-free UniPC schedule/final-latent contract for online NFT."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytest.importorskip("diffusers")

from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.posttrain.model.unipc import build_unipc_sampler
from cosmos_framework.posttrain.roles.rollout.wfm import WFMOnlineRollout
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel


def _cosmos_rl_unipc_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "packages/cosmos-rl/cosmos_rl/policy/model/wfm/samplers/fm_solvers_unipc.py"
    )
    spec = importlib.util.spec_from_file_location("cosmos_rl_nft_unipc_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Cosmos-RL UniPC reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.L0
@pytest.mark.CPU
def test_online_policy_sampler_pins_base_shift_before_runtime_shift() -> None:
    class _Inference:
        num_train_timesteps: int = 1000
        scheduler_type: str = "unipc"

    class _Config:
        rectified_flow_inference_config: _Inference = _Inference()

    class _Omni:
        config: _Config = _Config()
        tensor_kwargs: dict[str, object] = {"device": torch.device("cpu")}

    class _Policy:
        omnimot: _Omni = _Omni()

    sampler = build_unipc_sampler(_Policy())

    assert sampler.cfg.num_train_timesteps == 1000
    assert sampler.cfg.shift == 1.0
    assert sampler.cfg.use_dynamic_shifting is False

    _Policy.omnimot.config.rectified_flow_inference_config.scheduler_type = "edm"
    with pytest.raises(ValueError, match="needs scheduler_type='unipc'"):
        build_unipc_sampler(_Policy())


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_identity_binds_native_unipc_schedule_config() -> None:
    policy = ToyFlowModel(dim=2, seed=0)
    first = UniPCSampler(
        cfg=UniPCSamplerConfig(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False),
        tensor_kwargs={"device": torch.device("cpu")},
    )
    second = UniPCSampler(
        cfg=UniPCSamplerConfig(num_train_timesteps=999, shift=1.0, use_dynamic_shifting=False),
        tensor_kwargs={"device": torch.device("cpu")},
    )
    first_identity = WFMOnlineRollout(
        policy,
        sampler=first,
        num_steps=2,
        shift=5.0,
        guidance=3.0,
    ).validation_semantics_identity()
    second_identity = WFMOnlineRollout(
        policy,
        sampler=second,
        num_steps=2,
        shift=5.0,
        guidance=3.0,
    ).validation_semantics_identity()
    assert first_identity["sampler"]["config"]["num_train_timesteps"] == 1000
    assert first_identity != second_identity


@pytest.mark.L0
@pytest.mark.CPU
def test_unipc_applies_runtime_shift_once_and_matches_direct_scheduler() -> None:
    steps = 4
    runtime_shift = 5.0
    config = UniPCSamplerConfig(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    sampler = UniPCSampler(cfg=config, tensor_kwargs={"device": torch.device("cpu")})
    initial = torch.linspace(-1.0, 1.0, 16).reshape(1, 2, 2, 4)  # [C,T,H,W]

    def velocity(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        scale = timestep.to(latent.dtype).reshape(1, 1, 1, 1) / 1000.0  # [1,1,1,1]
        return 0.25 * latent + scale  # [C,T,H,W]

    actual = sampler(
        velocity,
        initial.clone(),
        num_steps=steps,
        shift=runtime_shift,
        seed=123,
    )  # [C,T,H,W]
    assert isinstance(actual, torch.Tensor)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=torch.device("cpu"), shift=runtime_shift)
    expected = initial.clone()  # [C,T,H,W]
    for timestep in scheduler.timesteps:
        prediction = velocity(expected, timestep.reshape(1, 1))  # [C,T,H,W]
        expected = scheduler.step(
            model_output=prediction,
            timestep=timestep,
            sample=expected.unsqueeze(0),  # [B,C,T,H,W]
            return_dict=False,
            generator=torch.Generator(device="cpu").manual_seed(123),
        )[0].squeeze(0)  # [C,T,H,W]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert scheduler.config.shift == 1.0
    assert scheduler.timesteps.numel() == steps


@pytest.mark.L0
@pytest.mark.CPU
def test_native_unipc_schedule_and_steps_match_same_revision_cosmos_rl() -> None:
    reference_class = _cosmos_rl_unipc_module().FlowUniPCMultistepScheduler
    native = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    reference = reference_class(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    native.set_timesteps(4, device=torch.device("cpu"), shift=5.0)
    reference.set_timesteps(4, device=torch.device("cpu"), shift=5.0)
    torch.testing.assert_close(native.sigmas, reference.sigmas, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native.timesteps, reference.timesteps, rtol=0.0, atol=0.0)

    native_sample = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 2, 2, 4)  # [B,C,T,H,W]
    reference_sample = native_sample.clone()  # [B,C,T,H,W]
    for native_timestep, reference_timestep in zip(native.timesteps, reference.timesteps, strict=True):
        native_velocity = 0.25 * native_sample + native_timestep / 1000.0  # [B,C,T,H,W]
        reference_velocity = 0.25 * reference_sample + reference_timestep / 1000.0  # [B,C,T,H,W]
        native_sample = native.step(
            native_velocity,
            native_timestep,
            native_sample,
            return_dict=False,
        )[0]  # [B,C,T,H,W]
        reference_sample = reference.step(
            reference_velocity,
            reference_timestep,
            reference_sample,
            return_dict=False,
        )[0]  # [B,C,T,H,W]
    torch.testing.assert_close(native_sample, reference_sample, rtol=0.0, atol=0.0)
