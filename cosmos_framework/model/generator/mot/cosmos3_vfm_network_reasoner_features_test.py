# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_encode_text_uses_zero_layout_for_generator_only_model() -> None:
    hidden_size = 8
    generator_norm = torch.nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
    network = SimpleNamespace(
        hidden_size=hidden_size,
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                include_und_pathway=False,
                norm_moe_gen=generator_norm,
            )
        ),
    )
    packed_seq = SimpleNamespace(
        sequence_length=11,
        text_ids=torch.tensor([3, 4, 5], dtype=torch.long),
    )

    packed, dtype = Cosmos3VFMNetwork._encode_text(network, packed_seq)

    assert packed.shape == (11, hidden_size)
    assert packed.dtype == torch.bfloat16
    assert dtype == torch.bfloat16
    assert torch.count_nonzero(packed) == 0


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_encode_text_requires_generator_norm_for_generator_only_model() -> None:
    network = SimpleNamespace(
        hidden_size=8,
        language_model=SimpleNamespace(model=SimpleNamespace(include_und_pathway=False)),
    )
    packed_seq = SimpleNamespace(
        sequence_length=4,
        text_ids=torch.tensor([1], dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="must retain norm_moe_gen"):
        Cosmos3VFMNetwork._encode_text(network, packed_seq)
