# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.model.generator.hf_model import _ValidationVideoFeatureCache


def _visual_encoder(calls: list[tuple[int, int]]):
    def encode(pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        calls.append((int(pixel_values.shape[0]), int(grid_thw.shape[0])))
        raw_sizes = [int(row.prod().item()) for row in grid_thw]
        chunks = torch.split(pixel_values, raw_sizes, dim=0)
        # spatial_merge_size=2 makes every [1, 2, 2] test grid one output row.
        main = torch.stack([chunk.sum(dim=0) for chunk in chunks])
        return main, [main + 10, main + 20]

    return encode


def test_validation_feature_cache_preserves_visual_forward_contract() -> None:
    cache = _ValidationVideoFeatureCache(capacity=3)
    calls: list[tuple[int, int]] = []
    encode = _visual_encoder(calls)
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    first_main, first_deep = cache.get_or_encode(
        ["video-a", "video-b"], pixels, grid, encode, spatial_merge_size=2
    )
    second_main, second_deep = cache.get_or_encode(
        ["video-a", "video-b"], pixels + 1000, grid, encode, spatial_merge_size=2
    )

    assert calls == [(8, 2)]
    assert isinstance(second_main, torch.Tensor)
    assert torch.equal(second_main, first_main)
    assert all(torch.equal(actual, expected) for actual, expected in zip(second_deep, first_deep))
    assert cache.hits == 2
    assert cache.misses == 2
    assert cache.global_all_hit_calls == 1

    third_main, _ = cache.get_or_encode(
        ["video-a", "video-c"], pixels + 1000, grid, encode, spatial_merge_size=2
    )
    assert calls == [(8, 2), (4, 1)]
    assert torch.equal(third_main[0], first_main[0])
    assert not torch.equal(third_main[1], first_main[1])
    assert cache.hits == 3
    assert cache.misses == 3


def test_validation_feature_cache_bypasses_unkeyed_visual_input() -> None:
    cache = _ValidationVideoFeatureCache(capacity=1)
    calls: list[tuple[int, int]] = []
    encode = _visual_encoder(calls)
    grid = torch.tensor([[1, 2, 2]])
    pixels = torch.ones(4, 3)

    main, deep = cache.get_or_encode(None, pixels, grid, encode, spatial_merge_size=2)

    assert calls == [(4, 1)]
    assert main.shape == (1, 3)
    assert len(deep) == 2
    assert cache.bypassed_calls == 1
    assert cache.entries == {}
