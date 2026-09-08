# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for `uses_single_timestep`, the gate on the timestep-embedding fast path.

`Cosmos3VFMNetwork._embed_packed_timesteps` uses this flag to decide whether to
evaluate the fp32 `time_embedder` on one row and `expand`, or on every noisy
token. The flag is therefore only valid when all noised tokens really do share a
single timestep scalar — a statement about timestep *values*, not about batch
size.

These tests pin that contract from both sides: uniform values must enable the
fast path regardless of batch size, and differing values must disable it.
"""

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import (
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from cosmos_framework.data.generator.sequence_packing.packers import uses_single_timestep
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

SPECIAL_TOKENS = {
    "eos_token_id": 1,
    "start_of_generation": 2,
    "end_of_generation": 3,
    "start_of_video": 4,
    "end_of_video": 5,
}

# Small enough to keep the test fast; the flag does not depend on these.
_LATENT_C = 16
_VAE_SPATIAL_DOWNSAMPLE = 4


def _pack(input_timesteps: torch.Tensor, batch_size: int = 1, num_frames: int = 4):
    """Pack a minimal vision-only batch and return the resulting PackedSequence."""
    videos = [torch.randn(3, num_frames, 32, 32) for _ in range(batch_size)]
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch={"video": videos}, input_video_key="video", input_image_key="images"
    )
    stacked = torch.stack(videos, dim=0)
    _, _, frames, height, width = stacked.shape
    x0_tokens_vision = [
        torch.randn(
            1,
            _LATENT_C,
            frames,
            height // _VAE_SPATIAL_DOWNSAMPLE,
            width // _VAE_SPATIAL_DOWNSAMPLE,
        ).to(stacked.dtype)
        for _ in range(batch_size)
    ]
    gen_data_clean = GenerationDataClean(
        batch_size=batch_size,
        is_image_batch=False,
        raw_state_vision=stacked,
        x0_tokens_vision=x0_tokens_vision,
        raw_state_action=None,
    )
    return pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=[[30] * 8 for _ in range(batch_size)],
        gen_data_clean=gen_data_clean,
        input_timesteps=input_timesteps,
        special_tokens=SPECIAL_TOKENS,
        max_num_tokens=36864,
        latent_patch_size=1,
    )


def test_single_sample_uses_fast_path():
    """The pre-existing behaviour: one sample, one timestep."""
    packed = _pack(torch.tensor([0.5]), batch_size=1)
    assert packed.uses_single_timestep is True


def test_uniform_batch_uses_fast_path():
    """A batch sharing one timestep must take the fast path.

    This is the case the inference template path always produces:
    `_copy_timestep_to_template` broadcasts a single scalar over every noisy
    token, so the values are uniform by construction at every sampler step.
    Gating on `numel() == 1` rejected it for any batch above one.
    """
    packed = _pack(torch.tensor([0.5, 0.5]), batch_size=2)
    assert packed.uses_single_timestep is True


def test_batched_cfg_shape_uses_fast_path():
    """The `use_batched_cfg` doubling must not disable the fast path.

    The cond+uncond fast path packs a `[2B]` timestep tensor of zeros, so a
    `numel()`-based gate turns the fast path off at batch size 1 — one
    performance feature disabling another.
    """
    packed = _pack(torch.zeros(2), batch_size=2)
    assert packed.uses_single_timestep is True


def test_differing_timesteps_disables_fast_path():
    """Per-sample sigmas must fall back to the general path."""
    packed = _pack(torch.tensor([0.25, 0.75]), batch_size=2)
    assert packed.uses_single_timestep is False


def test_diffusion_forcing_uniform_uses_fast_path():
    """Per-frame sigmas that happen to be identical are still a single scalar."""
    packed = _pack(torch.full((1, 4), 0.5), batch_size=1)
    assert packed.uses_single_timestep is True


def test_diffusion_forcing_per_frame_disables_fast_path():
    """Genuine per-frame sigmas must fall back to the general path."""
    packed = _pack(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), batch_size=1)
    assert packed.uses_single_timestep is False


@pytest.mark.parametrize("timesteps", [torch.tensor([0.5]), torch.tensor([0.5, 0.5])])
def test_fast_path_matches_general_path(timesteps):
    """The fast path must agree with the general path on the packed values.

    `_embed_packed_timesteps` takes `timesteps[:1]` and broadcasts it, so the
    flag is only sound if every packed timestep equals the first one. Asserting
    that here means the flag cannot be set true for a tensor the fast path would
    misrepresent, independently of how the flag is derived.
    """
    batch_size = timesteps.numel()
    packed = _pack(timesteps, batch_size=batch_size)
    assert packed.vision is not None
    packed_timesteps = packed.vision.timesteps
    assert packed_timesteps.numel() > 1, "expected multiple noisy tokens to make this meaningful"
    if packed.uses_single_timestep:
        assert torch.equal(packed_timesteps, packed_timesteps[:1].expand_as(packed_timesteps)), (
            "fast path would broadcast a value that does not match every packed timestep"
        )


@pytest.mark.parametrize(
    ("timesteps", "expected"),
    [
        (torch.tensor([0.5]), True),
        (torch.tensor([0.5, 0.5]), True),
        (torch.zeros(4), True),
        (torch.full((2, 3), 0.25), True),
        (torch.tensor([0.5, 0.5, 0.6]), False),
        (torch.tensor([[0.1, 0.2]]), False),
        # No entries means no noised tokens to share a scalar.
        (torch.empty(0), False),
        # NaN never compares equal, so the fast path is declined. Conservative is correct.
        (torch.tensor([float("nan"), float("nan")]), False),
    ],
)
def test_uses_single_timestep_helper(timesteps, expected):
    assert uses_single_timestep(timesteps) is expected


def test_independent_modality_schedule_invalidates_fast_path():
    """Per-sample sigmas written after packing must clear the flag.

    Under ``independent_action_schedule`` / ``independent_sound_schedule``,
    ``training_step`` replaces the packed action/sound timesteps with one sigma
    per sample after the packer has already set the flag from the *vision*
    timesteps. A uniform vision batch with differing action sigmas would
    otherwise leave the flag true and let ``_embed_packed_timesteps`` broadcast
    sample 0's sigma across the whole batch. This reproduces that composition.
    """
    packed = _pack(torch.tensor([0.5, 0.5]), batch_size=2)
    assert packed.uses_single_timestep is True, "uniform vision timesteps should start on the fast path"

    per_sample_action_sigmas = torch.tensor([0.25, 0.75])
    packed.uses_single_timestep &= uses_single_timestep(per_sample_action_sigmas)
    assert packed.uses_single_timestep is False

    shared_action_sigmas = torch.tensor([0.5, 0.5])
    packed_shared = _pack(torch.tensor([0.5, 0.5]), batch_size=2)
    packed_shared.uses_single_timestep &= uses_single_timestep(shared_action_sigmas)
    assert packed_shared.uses_single_timestep is True
