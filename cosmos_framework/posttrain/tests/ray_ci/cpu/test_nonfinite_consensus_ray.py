# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Non-finite consensus coverage on two real Ray workers using Gloo."""

from __future__ import annotations

from typing import Any

import pytest

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.tests.helpers.nonfinite_ray import (
    assert_nonfinite_consensus,
    make_nonfinite_grid,
)
from cosmos_framework.posttrain.workers.ray import _Host


@pytest.mark.L0
@pytest.mark.CPU
def test_nonfinite_consensus_and_recovery_on_cpu_ray_grid() -> None:
    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/cosmos3-posttrain-nonfinite-ray-cpu",
    )
    hosts: list[Any] = []
    try:
        hosts = [_Host.options(num_cpus=1).remote() for _ in range(2)]
        policy = make_nonfinite_grid("nonfinite_cpu", hosts, backend="gloo")
        assert_nonfinite_consensus(policy, expected_backend="gloo")
    finally:
        try:
            for host in hosts:
                ray.kill(host, no_restart=True)
        finally:
            ray.shutdown()
