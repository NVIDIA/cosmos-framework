# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for the instruments both GR00T parity gates share: `evaluate`'s rule that an unlisted name must
be bitwise equal, `flatten_tensors` descending through `.data`, and the draw captures restoring the
process-global symbols they patch."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from cosmos_framework.posttrain.tests.helpers.parity_diff import (
    capture_draws,
    capture_initial_noise,
    compare,
    evaluate,
    flatten_tensors,
)


class _Batch:
    """Stand-in for `transformers.BatchFeature`: a `.data` mapping."""

    def __init__(self, data: dict) -> None:
        self.data = data


@pytest.mark.L0
@pytest.mark.CPU
def test_compare_reports_without_judging() -> None:
    same = compare("same", torch.zeros(2, 3), torch.zeros(2, 3))
    assert same.exact and same.max_abs == 0.0 and same.max_rel == 0.0 and same.shape == (2, 3)
    differs = compare("differs", torch.zeros(2, 3), torch.full((2, 3), 0.5))
    assert not differs.exact and differs.max_abs == pytest.approx(0.5) and differs.as_json()["bitwise_equal"] is False
    # A shape mismatch is a different quantity, not a large difference.
    with pytest.raises(ValueError, match="not a tolerance question"):
        compare("x", torch.zeros(2, 3), torch.zeros(2, 4))


@pytest.mark.L0
@pytest.mark.CPU
def test_evaluate_requires_bitwise_equality_for_unlisted_names() -> None:
    exact = compare("exact", torch.zeros(2), torch.zeros(2)).as_json()
    small = compare("small", torch.zeros(2), torch.full((2,), 0.5)).as_json()

    over, not_bitwise = evaluate([exact, small], {})
    assert over == [] and [d["name"] for d in not_bitwise] == ["small"]
    over, not_bitwise = evaluate([exact, small], {"small": 1.0})
    assert over == [] and not_bitwise == []
    over, not_bitwise = evaluate([exact, small], {"small": 0.1})
    assert [d["name"] for d in over] == ["small"] and not_bitwise == []


@pytest.mark.L0
@pytest.mark.CPU
def test_flatten_tensors_descends_through_data_and_nested_mappings_but_never_into_a_tensor() -> None:
    batch = _Batch({"inputs": {"state": torch.zeros(1, 4), "nested": {"action": np.zeros((2, 2))}}})
    assert sorted(flatten_tensors(batch)) == ["inputs.nested.action", "inputs.state"]
    assert list(flatten_tensors({"x": torch.zeros(3, 3)})) == ["x"]
    assert flatten_tensors(torch.zeros(3)) == {}  # no prefix -> nothing to name


@pytest.mark.L0
@pytest.mark.CPU
def test_capture_draws_records_values_and_counts_dropout() -> None:
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    with capture_draws() as log:
        noise = torch.randn(2, 3, 4)
        uniform = torch.rand(2)
        beta = Beta(torch.tensor(1.5), torch.tensor(1.0)).sample([2])
        dropout(torch.ones(8))
        F.dropout(torch.ones(8), p=0.5, training=True)

    assert len(log.randn) == 1 and torch.equal(log.randn[0], noise)
    assert len(log.rand) == 1 and torch.equal(log.rand[0], uniform)
    assert len(log.beta) == 1 and torch.equal(log.beta[0], beta)
    assert log.dropout_calls == 2  # nn.Dropout resolves F.dropout at call time, so both routes count


@pytest.mark.L0
@pytest.mark.CPU
def test_capture_draws_restores_every_patched_symbol_even_when_the_block_raises() -> None:
    before = (torch.randn, torch.rand, F.dropout, "sample" in vars(Beta))
    with pytest.raises(RuntimeError, match="boom"):
        with capture_draws():
            raise RuntimeError("boom")
    assert (torch.randn, torch.rand, F.dropout, "sample" in vars(Beta)) == before


@pytest.mark.L0
@pytest.mark.CPU
def test_capture_initial_noise_filters_by_rank_and_appends_live() -> None:
    store: list[torch.Tensor] = []
    real_randn = torch.randn
    with capture_initial_noise(store):
        torch.randn(5)  # 1-D: ignored
        wanted = torch.randn(1, 2, 3)
        assert len(store) == 1  # appended as it happens, not at exit
    assert torch.randn is real_randn
    assert len(store) == 1 and torch.equal(store[0], wanted)
