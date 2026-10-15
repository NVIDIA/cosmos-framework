# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression coverage for joint sensor clean replay."""

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.data.generator.sequence_packing.packers import make_teacher_forcing_clean_pack


@pytest.mark.L0
@pytest.mark.CPU
def test_clean_replay_marks_lidar_clean_without_mutating_the_training_pack() -> None:
    lidar = ModalityData(
        tokens=[torch.randn(3, 2)],  # list of [T,C]
        condition_mask=[torch.tensor([1.0, 0.0, 0.0])],  # list of [T]
        mse_loss_indexes=torch.tensor([1, 2]),  # [N_noisy]
        timesteps=torch.tensor([0.4, 0.4]),  # [N_noisy]
        noisy_frame_indexes=[torch.tensor([1, 2])],  # list of [N_noisy]
    )
    packed = PackedSequence(text_ids=torch.tensor([1]), sample_lens=[3], lidar=lidar)  # [S_text]
    clean = make_teacher_forcing_clean_pack(packed)
    assert clean.lidar is not None
    assert clean.lidar.condition_mask[0].all()
    assert clean.lidar.mse_loss_indexes.numel() == 0
    assert clean.lidar.timesteps.numel() == 0
    assert clean.lidar.noisy_frame_indexes[0].numel() == 0
    torch.testing.assert_close(clean.lidar.tokens[0], lidar.tokens[0])
    assert clean.lidar.tokens[0].data_ptr() != lidar.tokens[0].data_ptr()
    assert lidar.mse_loss_indexes.tolist() == [1, 2]
    assert lidar.condition_mask[0].tolist() == [1, 0, 0]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("modality_name", ["vision", "lidar", "action", "sound"])
def test_clean_replay_preserves_each_modality_and_its_metadata_types(modality_name: str) -> None:
    modality = ModalityData(
        tokens=[torch.arange(6, dtype=torch.float64).reshape(3, 2)],  # list[[3,2]]
        condition_mask=[torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)],  # list[[3]]
        mse_loss_indexes=torch.tensor([1, 2], dtype=torch.int32),  # [2]
        timesteps=torch.tensor([0.4, 0.4], dtype=torch.float64),  # [2]
        noisy_frame_indexes=[torch.tensor([1, 2], dtype=torch.int32)],  # list[[2]]
    )
    packed = PackedSequence(text_ids=torch.tensor([1]), sample_lens=[3])  # [1]
    setattr(packed, modality_name, modality)
    clean = getattr(make_teacher_forcing_clean_pack(packed), modality_name)
    assert isinstance(clean, ModalityData)
    assert torch.equal(clean.tokens[0], modality.tokens[0])
    assert clean.tokens[0].data_ptr() != modality.tokens[0].data_ptr()
    assert clean.condition_mask[0].tolist() == [1.0, 1.0, 1.0]
    for field in ("mse_loss_indexes", "timesteps"):
        original = getattr(modality, field)
        emptied = getattr(clean, field)
        assert emptied.numel() == 0
        assert (emptied.dtype, emptied.device) == (original.dtype, original.device)
        assert original.numel() == 2
    assert clean.noisy_frame_indexes[0].numel() == 0
    assert clean.noisy_frame_indexes[0].dtype == torch.int32
    assert modality.noisy_frame_indexes[0].tolist() == [1, 2]
