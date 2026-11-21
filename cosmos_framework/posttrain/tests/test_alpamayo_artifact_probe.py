# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe comparison guards; synthetic records do not qualify real artifact conversion."""

import copy

import numpy as np
import pytest

from cosmos_framework.posttrain.model.alpamayo.qualification import compare_outputs


def record():
    return {
        "format": "cosmos-alpamayo-artifact-probe-v1",
        "sample_id": "unit",
        "seed": 17,
        "input_tensors": {"inputs": "same-fixture"},
        "flow_loss": 1.0,
        "flow_prediction": np.ones((1, 64, 2)).tolist(),
        "trajectory_xyz_meters": np.zeros((1, 64, 3)).tolist(),
        "task_metrics": {
            f"open_loop_{kind}_{horizon}_meters": 1.0 for kind in ("ade", "fde") for horizon in ("1s", "3s", "6p4s")
        },
    }


def test_equal_outputs_pass_without_claiming_task_quality():
    result = compare_outputs(record(), record())
    assert result["passed"]
    assert result["claim"] == "artifact_numerical_conversion_only_not_recipe_quality_or_closed_loop"


@pytest.mark.parametrize("failure", ["velocity", "loss", "trajectory", "task_metric"])
def test_each_declared_numerical_guard_can_fail(failure):
    reference = record()
    candidate = copy.deepcopy(reference)
    if failure == "velocity":
        candidate["flow_prediction"] = (np.ones((1, 64, 2)) * 1.02).tolist()
    elif failure == "loss":
        candidate["flow_loss"] = 1.011
    elif failure == "trajectory":
        candidate["trajectory_xyz_meters"][0][-1][0] = 0.26
    else:
        candidate["task_metrics"]["open_loop_ade_6p4s_meters"] = 1.021
    assert not compare_outputs(reference, candidate)["passed"]


def test_mismatched_preprocessing_is_an_error_not_a_tolerance():
    candidate = record()
    candidate["input_tensors"] = {"inputs": "different"}
    with pytest.raises(ValueError, match="input_tensors"):
        compare_outputs(record(), candidate)
