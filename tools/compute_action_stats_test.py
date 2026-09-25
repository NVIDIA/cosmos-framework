# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest
from compute_action_stats import compute_stats


def test_compute_stats_matches_small_dataset() -> None:
    stats = compute_stats(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [4.0, 5.0],
        ],
        reservoir_size=10,
    )

    np.testing.assert_allclose(stats["mean"], [2.0, 3.0])
    np.testing.assert_allclose(stats["std"], np.sqrt([8.0 / 3.0, 8.0 / 3.0]))
    np.testing.assert_allclose(stats["min"], [0.0, 1.0])
    np.testing.assert_allclose(stats["max"], [4.0, 5.0])
    np.testing.assert_allclose(stats["q01"], [0.04, 1.04])
    np.testing.assert_allclose(stats["q99"], [3.96, 4.96])


def test_compute_stats_reservoir_is_deterministic() -> None:
    values = [np.arange(100, dtype=np.float64).reshape(50, 2)]
    assert compute_stats(values, reservoir_size=5, seed=7) == compute_stats(
        values, reservoir_size=5, seed=7
    )


@pytest.mark.parametrize(
    "values, message",
    [
        ([], "no action values"),
        ([[[1.0, 2.0]], [[3.0]]], "dimension"),
        ([[[float("nan")]]], "NaN"),
    ],
)
def test_compute_stats_rejects_invalid_values(values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_stats(values)


def test_compute_stats_rejects_invalid_reservoir_size() -> None:
    with pytest.raises(ValueError, match="reservoir_size"):
        compute_stats([[1.0]], reservoir_size=0)
