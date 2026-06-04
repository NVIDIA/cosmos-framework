# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for DefaultBatchCollator."""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.dataflow.collators import DefaultBatchCollator


def test_default_collator_stacks_like_torch():
    samples = [
        {"x": torch.tensor([1.0, 2.0]), "y": 0},
        {"x": torch.tensor([3.0, 4.0]), "y": 1},
    ]
    batch = DefaultBatchCollator().collate(samples)
    assert batch["x"].shape == (2, 2)
    assert torch.equal(batch["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(batch["y"], torch.tensor([0, 1]))
