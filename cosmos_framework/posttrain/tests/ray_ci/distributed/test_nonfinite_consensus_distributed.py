# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Non-finite consensus coverage on a two-GPU Ray/NCCL grid."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("ray")

from cosmos_framework.posttrain.tests.helpers.nonfinite_ray import (
    assert_nonfinite_consensus,
    make_nonfinite_grid,
)


@pytest.mark.L0
@pytest.mark.GPU
def test_nonfinite_consensus_and_recovery_on_gpu_ray_grid(ray_two_gpu_hosts: list[Any]) -> None:
    policy = make_nonfinite_grid("nonfinite_gpu", ray_two_gpu_hosts, backend="nccl")
    assert len(set(policy.hosts().values())) == 1
    assert len(set(policy.gpus().values())) == 2
    assert_nonfinite_consensus(policy, expected_backend="nccl")
