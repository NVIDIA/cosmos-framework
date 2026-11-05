# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.inference.sample_selection import select_limited_indices


@pytest.mark.L0
@pytest.mark.CPU
def test_select_limited_indices_first_preserves_first_n() -> None:
    assert select_limited_indices(range(8), limit=3, strategy="first") == [0, 1, 2]


@pytest.mark.L0
@pytest.mark.CPU
def test_select_limited_indices_deterministic_random_is_stable_random_set() -> None:
    first = select_limited_indices(range(100), limit=10, seed=123, namespace="robotics")
    second = select_limited_indices(range(100), limit=10, seed=123, namespace="robotics")
    different_namespace = select_limited_indices(range(100), limit=10, seed=123, namespace="manifest")

    assert first == second
    assert first != list(range(10))
    assert first != different_namespace
    assert first == sorted(first)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("strategy", ["first", "deterministic_random"])
def test_select_limited_indices_limit_bounds(strategy: str) -> None:
    candidates = list(range(0, 20, 2))

    assert select_limited_indices(candidates, limit=None, strategy=strategy) == candidates
    assert select_limited_indices(candidates, limit=len(candidates), strategy=strategy) == candidates
    assert select_limited_indices(candidates, limit=0, strategy=strategy) == []
    subset = select_limited_indices(candidates, limit=4, strategy=strategy)
    assert len(subset) == 4
    assert set(subset) <= set(candidates)


@pytest.mark.L0
@pytest.mark.CPU
def test_select_limited_indices_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported sample selection strategy"):
        select_limited_indices(range(8), limit=3, strategy="shuffle")  # type: ignore[arg-type]
