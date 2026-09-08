# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for scalar and framewise action-domain routing in Cosmos3 VFM."""

import pytest
import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork


@pytest.mark.L0
@pytest.mark.CPU
def test_select_action_domain_ids_expands_scalar_metadata() -> None:
    domain_ids = Cosmos3VFMNetwork._select_action_domain_ids(
        torch.tensor(22, dtype=torch.long),
        token_count=4,
    )

    torch.testing.assert_close(domain_ids, torch.tensor([22, 22, 22, 22], dtype=torch.long))


@pytest.mark.L0
@pytest.mark.CPU
def test_select_action_domain_ids_indexes_framewise_metadata() -> None:
    domain_ids = Cosmos3VFMNetwork._select_action_domain_ids(
        torch.tensor([2, 2, 22, 22], dtype=torch.long),
        token_count=4,
        token_indexes=torch.tensor([1, 3], dtype=torch.long),
    )

    torch.testing.assert_close(domain_ids, torch.tensor([2, 22], dtype=torch.long))


@pytest.mark.L0
@pytest.mark.CPU
def test_select_action_domain_ids_rejects_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="one ID per action token"):
        Cosmos3VFMNetwork._select_action_domain_ids(
            torch.tensor([2, 22], dtype=torch.long),
            token_count=4,
        )
