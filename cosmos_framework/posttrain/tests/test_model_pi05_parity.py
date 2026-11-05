# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Parity instruments reject mismatched sources and controlled numerical perturbations."""

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.pi05_parity import (
    NoiseTape,
    compare_records,
    prepare_request,
    python_seed,
    read_request,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_request_noise_and_temporary_injection_restore_rng(tmp_path: Path) -> None:
    output = tmp_path / "request.npz"
    prepare_request(output, batch=2, seed=1234, observations=None)
    request = read_request(output)
    assert request["image"].shape == (2, 256, 256, 3)
    state = random.getstate()
    with python_seed(int(request["seed"])):
        assert random.randint(0, 2) == int(request["selected_step"])
    assert random.getstate() == state
    original = torch.normal
    tape = NoiseTape(request["noise"])
    with patch.object(torch, "normal", tape.normal):
        for expected in request["noise"]:
            result = torch.normal(0.0, 1.0, size=expected.shape, device="cpu", dtype=torch.float32)
            np.testing.assert_array_equal(result.numpy(), expected)
    tape.verify()
    assert torch.normal is original
    with pytest.raises(ValueError, match="unexpected parity noise draw"):
        tape.draw(request["noise"].shape[1:], "cpu")


def test_comparison_rejects_perturbed_values_dtypes_and_sources() -> None:
    identity = {key: "same" for key in ("request_sha256", "checkpoint_sha256", "tokenizer_sha256", "norm_stats_sha256")}
    identity["selected_step"] = 1
    identity.update(parameter_dtypes={}, matmul_precision="high", cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True)
    reference = {"metadata": identity, "tensors": {"velocity.0": torch.tensor([1.0], dtype=torch.float64)}}
    candidate = {"metadata": dict(identity), "tensors": {"velocity.0": reference["tensors"]["velocity.0"].clone()}}
    assert compare_records(reference, candidate)["passed"]
    candidate["tensors"]["velocity.0"][0] += 1e-10
    report = compare_records(reference, candidate)
    assert not report["passed"] and report["rows"][0]["max_abs"] > 0
    candidate["tensors"]["velocity.0"] = reference["tensors"]["velocity.0"].float()
    assert not compare_records(reference, candidate)["passed"]
    candidate["metadata"]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="source mismatch"):
        compare_records(reference, candidate)
