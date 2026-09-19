# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU dry run of sequence packing for (LR, HR) samples whose two vision items have different latent grids."""

import pytest
import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_SPECIAL_TOKENS = {"eos_token_id": 151645, "start_of_generation": 151652, "end_of_generation": 151653}
_LATENT_C, _SPATIAL, _TEMPORAL, _PATCH = 16, 16, 4, 2


def _latent(frames: int, h: int, w: int) -> torch.Tensor:  # returns [1,C,T',H',W']
    return torch.randn(1, _LATENT_C, 1 + (frames - 1) // _TEMPORAL, h // _SPATIAL, w // _SPATIAL)


def _pack(share: bool, frames: int = 9, hr: tuple[int, int] = (480, 832), lr: tuple[int, int] = (240, 416)):
    plan = SequencePlan(
        has_text=True, has_vision=True, condition_frame_indexes_vision=[0], share_vision_temporal_positions=share
    )
    gen = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[_latent(frames, *lr), _latent(frames, *hr)],
        num_vision_items_per_sample=[2],
        fps_vision=torch.tensor([24.0]),
    )
    return pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[[5, 6, 7, 8]],
        gen_data_clean=gen,
        input_timesteps=torch.rand(1),
        special_tokens=_SPECIAL_TOKENS,
        latent_patch_size=_PATCH,
        temporal_compression_factor=_TEMPORAL,
    )


def test_two_item_sr_sample_packs_without_shared_temporal_grid() -> None:
    packed = _pack(share=False)

    def _grid(h: int, w: int) -> tuple[int, int]:
        # Latent dims that are not a multiple of the patch size are padded up by the packer (15 -> 8 tokens).
        return -(-(h // _SPATIAL) // _PATCH), -(-(w // _SPATIAL) // _PATCH)

    t_latent = 1 + 8 // _TEMPORAL
    lr_grid, hr_grid = _grid(240, 416), _grid(480, 832)  # (8,13), (15,26)
    assert packed.vision is not None
    # Both items are in the packed vision stream with their own grids; total vision tokens = LR + HR.
    assert len(packed.vision.token_shapes) == 2
    lr_shape, hr_shape = packed.vision.token_shapes
    assert tuple(lr_shape[-2:]) == lr_grid and tuple(hr_shape[-2:]) == hr_grid
    total = sum(int(torch.tensor(s[-3:]).prod()) for s in packed.vision.token_shapes)
    assert total == t_latent * (lr_grid[0] * lr_grid[1] + hr_grid[0] * hr_grid[1])
    # Only the HR (last) item is generated; the LR item is pure conditioning.
    assert bool(packed.vision.condition_mask[0].all()) and not bool(packed.vision.condition_mask[1].all())


def test_two_item_sr_sample_with_shared_grid_is_rejected() -> None:
    with pytest.raises(AssertionError, match="equal spatial grid"):
        _pack(share=True)
