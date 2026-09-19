# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.utils.data_and_condition import select_target_image_sizes

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _size(h: int, w: int) -> torch.Tensor:  # returns [4]
    return torch.tensor([float(h), float(w), float(h), float(w)])


def test_single_item_batches_pass_through() -> None:
    sizes = [_size(480, 832), _size(720, 1280)]
    assert select_target_image_sizes(sizes, None, 2) == sizes
    assert select_target_image_sizes(sizes, [1, 1], 2) == sizes


def test_multi_item_samples_return_last_item_per_sample() -> None:
    # Sample 0: SR pair (LR 240p, HR 480p). Sample 1: SR pair at 720p. Flattened in item order.
    sizes = [_size(240, 416), _size(480, 832), _size(360, 640), _size(720, 1280)]
    selected = select_target_image_sizes(sizes, [2, 2], 2)
    assert [s.tolist() for s in selected] == [sizes[1].tolist(), sizes[3].tolist()]


def test_mixed_item_counts_are_indexed_correctly() -> None:
    sizes = [_size(256, 256), _size(240, 416), _size(480, 832)]
    selected = select_target_image_sizes(sizes, [1, 2], 2)
    assert [s.tolist() for s in selected] == [sizes[0].tolist(), sizes[2].tolist()]


def test_inconsistent_counts_raise() -> None:
    sizes = [_size(240, 416), _size(480, 832), _size(720, 1280)]
    with pytest.raises(ValueError, match="declare"):
        select_target_image_sizes(sizes, [2, 2], 2)
    with pytest.raises(ValueError, match="batch_size"):
        select_target_image_sizes(sizes, [2, 1, 1], 2)
