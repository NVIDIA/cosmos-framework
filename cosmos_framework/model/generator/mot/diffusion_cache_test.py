# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.mot.diffusion_cache import DiffusionCache


def _const_indicator(value: float = 1.0) -> list[torch.Tensor]:
    return [torch.full((1, 2, 2, 1), value)]


def _dummy_residual() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 2), torch.zeros(1, 2)


@pytest.mark.L0
@pytest.mark.CPU
def test_max_consecutive_cached_forces_full_per_pathway() -> None:
    cache = DiffusionCache(
        num_steps=20,
        config={
            "ret_steps": 0,
            "cutoff_from_end": 0,
            "diffusion_cache_thresh": 1.0,
            "max_consecutive_cached": 3,
        },
    )
    cache.state.step = 0
    assert cache._should_compute("cfg0", _const_indicator()) is True
    cache._pathways["cfg0"].history = [(0, _dummy_residual())]

    for step in range(1, 4):
        cache.state.step = step
        assert cache._should_compute("cfg0", _const_indicator()) is False
        cache._pathways["cfg0"].consecutive_cached += 1

    cache.state.step = 4
    assert cache._should_compute("cfg0", _const_indicator()) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_diffusion_cache_uses_tuned_defaults() -> None:
    cache = DiffusionCache(num_steps=10)

    assert cache.config.diffusion_cache_thresh == pytest.approx(0.25)
    assert cache.config.max_consecutive_cached == 2


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_max_consecutive_cached_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="max_consecutive_cached"):
        DiffusionCache(num_steps=10, config={"max_consecutive_cached": value})
