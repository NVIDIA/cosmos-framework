# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Keep the native denoiser's item batch axis out of the sampler's latent shape."""

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.quantization.calibration import _run_framework_denoiser


class GridDenoiser:
    """Exercise native packing and grid decoding without allocating a reasoner."""

    latent_patch_size = 1
    latent_channel = 2
    unpatchify_and_unpack_latents = Cosmos3VFMNetwork.unpatchify_and_unpack_latents
    _decode_grid_stream = Cosmos3VFMNetwork._decode_grid_stream

    def _derive_include_end_of_generation_token(self):
        return False

    def _pack_input_sequence(self, *args, **kwargs):
        return pack_input_sequence(
            *args,
            special_tokens={"eos_token_id": 0, "start_of_generation": 1, "end_of_generation": 2},
            latent_patch_size=self.latent_patch_size,
            **kwargs,
        )

    def denoise(self, *, data_batch_packed):
        # Real grid decoding returns one [1,C,T,H,W] tensor per vision item.
        hidden = torch.ones(sum(data_batch_packed.sample_lens), self.latent_channel)
        prediction = self._decode_grid_stream(
            data_batch_packed.vision,
            hidden,
            vae2llm=nn.Identity(),
            llm2vae=nn.Identity(),
            latent_channel=self.latent_channel,
            patch_latent_dim=self.latent_channel,
            original_latent_shapes=None,
        )
        return {"preds_vision": prediction}


@pytest.mark.parametrize("frames,i2v", [(1, False), (3, True)])
def test_fixed_step_calibration_preserves_latent_shape_and_clean_frame(frames, i2v):
    model = GridDenoiser()
    noise = torch.zeros(2, frames, 2, 2)
    reference = torch.full_like(noise, 0.25) if i2v else None
    mask = torch.zeros(1, frames, 1, 1) if i2v else None
    if i2v:
        mask[:, 0] = 1
        noise[:, 0] = reference[:, 0]
    calls = []

    def velocity(latent, timestep):
        assert latent.shape == noise.shape
        prediction = _run_framework_denoiser(
            model,
            hidden=latent.unsqueeze(0),
            timestep=timestep,
            text_indexes=[3, 4],
            frame_rate=24.0,
            i2v=i2v,
        )
        assert prediction.shape == (1, *noise.shape)
        calls.append(prediction)
        return prediction[0]

    result = FixedStepSampler([1.0, 0.9375, 0.8333333333333334, 0.625])(
        velocity, noise, seed=7, condition_reference=reference, condition_mask=mask
    )
    assert len(calls) == 4
    assert result.shape == noise.shape
    assert torch.isfinite(result).all()
    if i2v:
        torch.testing.assert_close(result[:, 0], reference[:, 0])
