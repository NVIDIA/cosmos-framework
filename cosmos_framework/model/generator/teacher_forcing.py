# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy

import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence


def make_teacher_forcing_clean_pack(
    packed_seq: PackedSequence,
) -> PackedSequence:
    """Return a clean replay pack with generated modalities treated as conditions."""
    clean_pack = copy.deepcopy(packed_seq)
    for modality in (clean_pack.vision, clean_pack.action, clean_pack.sound):
        if modality is not None:
            mark_modality_as_clean_condition(modality)
    return clean_pack


def mark_modality_as_clean_condition(modality: ModalityData) -> None:
    """Clear noised-token metadata for a finalized modality in the TF clean pass."""
    modality.condition_mask = [torch.ones_like(mask) for mask in modality.condition_mask]
    modality.mse_loss_indexes = torch.empty(
        0, dtype=modality.mse_loss_indexes.dtype, device=modality.mse_loss_indexes.device
    )  # [0]
    modality.timesteps = torch.empty(0, dtype=modality.timesteps.dtype, device=modality.timesteps.device)  # [0]
    modality.noisy_frame_indexes = [
        torch.empty(0, dtype=indexes.dtype, device=indexes.device) for indexes in modality.noisy_frame_indexes
    ]  # list of [0]
