# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_normalization import (
    denormalize_action,
    load_action_stats,
    normalize_action,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RAW_STATS = {
    "mean": [0.0, 1.0, -1.0],
    "std": [1.0, 2.0, 0.5],
    "min": [-2.0, -1.0, -3.0],
    "max": [2.0, 3.0, 1.0],
    "q01": [-1.0, 0.0, -2.0],
    "q99": [1.0, 2.0, 0.0],
}


def _tensor_stats(raw=_RAW_STATS) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}


def _action() -> torch.Tensor:
    return torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, 0.0]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# load_action_stats
# ---------------------------------------------------------------------------


def test_load_action_stats_flat(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(_RAW_STATS))
    result = load_action_stats(str(p))
    assert set(result) == set(_RAW_STATS)
    for key, value in result.items():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, np.array(_RAW_STATS[key], dtype=np.float32))


def test_load_action_stats_filters_unknown_keys(tmp_path):
    raw = {**_RAW_STATS, "extra_field": [1.0, 2.0]}
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(raw))
    result = load_action_stats(str(p))
    assert "extra_field" not in result


def test_load_action_stats_missing_file():
    with pytest.raises(FileNotFoundError):
        load_action_stats("/nonexistent/path/stats.json")


# ---------------------------------------------------------------------------
# normalize_action / denormalize_action — round-trip identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quantile", "meanstd", "minmax"])
def test_round_trip(method):
    action = _action()
    stats = _tensor_stats()
    normalized = normalize_action(action, method, stats)
    recovered = denormalize_action(normalized, method, stats)
    torch.testing.assert_close(recovered, action, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# normalize_action — endpoint correctness
# ---------------------------------------------------------------------------


def test_normalize_quantile_endpoints():
    stats = _tensor_stats()
    q01, q99 = stats["q01"], stats["q99"]
    assert torch.allclose(normalize_action(q01.unsqueeze(0), "quantile", stats), torch.full((1, 3), -1.0))
    assert torch.allclose(normalize_action(q99.unsqueeze(0), "quantile", stats), torch.full((1, 3), 1.0))


def test_normalize_minmax_endpoints():
    stats = _tensor_stats()
    lo, hi = stats["min"], stats["max"]
    assert torch.allclose(normalize_action(lo.unsqueeze(0), "minmax", stats), torch.full((1, 3), -1.0))
    assert torch.allclose(normalize_action(hi.unsqueeze(0), "minmax", stats), torch.full((1, 3), 1.0))


def test_normalize_meanstd_zero_mean():
    stats = _tensor_stats()
    result = normalize_action(stats["mean"].unsqueeze(0), "meanstd", stats)
    assert torch.allclose(result, torch.zeros(1, 3))


# ---------------------------------------------------------------------------
# denormalize_action — endpoint correctness
# ---------------------------------------------------------------------------


def test_denormalize_quantile_endpoints():
    stats = _tensor_stats()
    q01, q99 = stats["q01"], stats["q99"]
    assert torch.allclose(denormalize_action(torch.full((1, 3), -1.0), "quantile", stats), q01.unsqueeze(0))
    assert torch.allclose(denormalize_action(torch.full((1, 3), 1.0), "quantile", stats), q99.unsqueeze(0))


def test_denormalize_minmax_endpoints():
    stats = _tensor_stats()
    lo, hi = stats["min"], stats["max"]
    assert torch.allclose(denormalize_action(torch.full((1, 3), -1.0), "minmax", stats), lo.unsqueeze(0))
    assert torch.allclose(denormalize_action(torch.full((1, 3), 1.0), "minmax", stats), hi.unsqueeze(0))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_normalize_zero_range_no_nan():
    stats = {k: torch.zeros(3) for k in ("q01", "q99", "mean", "std", "min", "max")}
    action = torch.ones(1, 3)
    for method in ("quantile", "meanstd", "minmax"):
        result = normalize_action(action, method, stats)
        assert torch.isfinite(result).all(), f"{method} produced non-finite output with zero range"


def test_normalize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_action(_action(), "unknown_method", _tensor_stats())


def test_denormalize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        denormalize_action(_action(), "unknown_method", _tensor_stats())
