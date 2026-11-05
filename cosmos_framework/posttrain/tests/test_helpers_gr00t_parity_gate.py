# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The serving gate's process conditions and its rule, without models: `stage1_conditions` and `evaluate_gate`."""

from __future__ import annotations

import os

import pytest

from cosmos_framework.posttrain.tests.helpers.gr00t_parity_gate import (
    DIT_SDPA_MODE_ENV,
    TOLERANCES,
    evaluate_gate,
    stage1_conditions,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_stage1_conditions_remove_the_attention_override_and_hand_it_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isaac reads the override at call time; the port has no reader, so the comparison must run without it."""
    monkeypatch.setenv(DIT_SDPA_MODE_ENV, "math")
    with stage1_conditions():
        assert DIT_SDPA_MODE_ENV not in os.environ
    assert os.environ[DIT_SDPA_MODE_ENV] == "math", "the trainer's setting must survive the gate"
    monkeypatch.delenv(DIT_SDPA_MODE_ENV)
    with stage1_conditions():
        assert DIT_SDPA_MODE_ENV not in os.environ
    assert DIT_SDPA_MODE_ENV not in os.environ


def _row(name: str, max_abs: float) -> dict:
    return {"name": name, "shape": [1], "max_abs": max_abs, "max_rel": max_abs, "bitwise_equal": max_abs == 0.0}


@pytest.mark.L0
@pytest.mark.CPU
def test_evaluate_gate_requires_bitwise_rows_identical_processors_and_two_implementations() -> None:
    good = {
        "diffs": [_row("backbone_features", 0.0), _row("serving_batch_invariance_features", 0.75)],
        "processor_identical": True,
        "class_provenance_ok": True,
    }
    assert evaluate_gate(good)[0] is True
    assert TOLERANCES == {"serving_batch_invariance_features": 3.0}, "the only non-parity row"
    assert evaluate_gate({**good, "diffs": [_row("backbone_features", 1e-6)]})[0] is False
    assert evaluate_gate({**good, "diffs": [_row("serving_batch_invariance_features", 3.5)]})[0] is False
    assert evaluate_gate({**good, "processor_identical": False})[0] is False
    assert evaluate_gate({**good, "class_provenance_ok": False})[0] is False, "the port against itself is not parity"
    assert evaluate_gate({k: v for k, v in good.items() if k != "class_provenance_ok"})[0] is False
