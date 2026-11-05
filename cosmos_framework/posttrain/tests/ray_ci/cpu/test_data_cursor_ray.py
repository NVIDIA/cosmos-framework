# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage 1 preference-cursor coverage on two real Ray workers using Gloo."""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.tests.helpers.data_cursor_ray import (
    assert_cursor_checkpoint_round_trip,
    make_cursor_grid,
    make_cursor_per_role_grids,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_checkpoint_round_trip_on_cpu_ray_grid() -> None:
    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/cosmos3-stage1-cursor-ray-cpu",
    )
    try:
        data, policy = make_cursor_grid("stage1_cursor_cpu", gpu_request=0, inject_device_mesh=False)
        assert_cursor_checkpoint_round_trip(data, policy)
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_checkpoint_crosses_separate_role_ray_grids() -> None:
    ray.init(
        address="local",
        num_cpus=5,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/c3-stage1-per-role-ray",
    )
    try:
        data, policy = make_cursor_per_role_grids("stage1_cursor_per_role_cpu")
        assert_cursor_checkpoint_round_trip(
            data,
            policy,
            expect_separate_roles=True,
        )
    finally:
        ray.shutdown()
