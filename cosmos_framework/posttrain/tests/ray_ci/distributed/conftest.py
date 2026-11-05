# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reusable fixtures for single-node distributed Ray CI tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
import torch

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.workers.ray import _Host

_WORLD_SIZE = 2
_RAY_TEMP_DIR = "/tmp/cosmos3-posttrain-ray-distributed"
_DISTRIBUTED_RAY_CI_ENV = "COSMOS3_POSTTRAIN_DISTRIBUTED_RAY_CI"


@pytest.fixture(scope="session", autouse=True)
def require_distributed_ray_ci() -> None:
    """Keep dedicated distributed Ray tests out of the generic L0 sweep."""
    if os.environ.get(_DISTRIBUTED_RAY_CI_ENV) != "1":
        pytest.skip(f"requires {_DISTRIBUTED_RAY_CI_ENV}=1 from the dedicated distributed Ray CI job")


@pytest.fixture
def ray_two_gpu_hosts() -> Iterator[list[Any]]:
    """Start two one-GPU Ray actors on one node and clean them up after the test."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip("requires at least two visible CUDA devices")

    hosts: list[Any] = []
    try:
        ray.init(
            address="local",
            num_cpus=_WORLD_SIZE,
            num_gpus=_WORLD_SIZE,
            include_dashboard=False,
            log_to_driver=True,
            _temp_dir=_RAY_TEMP_DIR,
        )
        for _ in range(_WORLD_SIZE):
            hosts.append(_Host.options(num_cpus=1, num_gpus=1).remote())
        yield hosts
    finally:
        try:
            for host in hosts:
                ray.kill(host, no_restart=True)
        finally:
            ray.shutdown()
