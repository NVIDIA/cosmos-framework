# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    prepare_sequence_pack_metadata,
    to_device_nonblocking,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_to_device_nonblocking_is_identity_on_the_same_device() -> None:
    tensor = torch.arange(6, dtype=torch.int32)
    assert to_device_nonblocking(tensor, "cpu") is tensor
    assert to_device_nonblocking(tensor, torch.device("cpu")) is tensor
    empty = torch.empty(0, dtype=torch.int64)
    assert to_device_nonblocking(empty, "cpu").numel() == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_metadata_from_host_indexes_matches_metadata_from_device_indexes() -> None:
    """``PackedSequence.to_cuda`` validates the layout with the host copy of ``text_indexes``;
    the result must not depend on which copy is handed in."""
    sample_lens = [7, 5]
    split_lens = [3, 4, 2, 3]
    attn_modes = ["causal", "full", "causal", "full"]
    text_indexes = torch.tensor([0, 1, 2, 7, 8], dtype=torch.int64)
    reference = prepare_sequence_pack_metadata(
        sample_lens=sample_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        packed_und_token_indexes=text_indexes,
        device=torch.device("cpu"),
    )
    other = prepare_sequence_pack_metadata(
        sample_lens=sample_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        packed_und_token_indexes=text_indexes.clone(),
        device=torch.device("cpu"),
    )
    for name in vars(reference):
        a, b = getattr(reference, name), getattr(other, name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), name
        else:
            assert a == b, name
