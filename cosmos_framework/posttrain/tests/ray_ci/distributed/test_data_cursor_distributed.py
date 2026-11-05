# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage 1 preference-cursor coverage on a two-GPU Ray/NCCL grid."""

from __future__ import annotations

import pytest
import torch

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.tests.helpers.data_cursor_ray import (
    assert_cursor_checkpoint_round_trip,
    make_cursor_grid,
)


@pytest.mark.L0
@pytest.mark.GPU
def test_cursor_checkpoint_round_trip_on_gpu_ray_grid() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two visible CUDA devices")

    ray.init(
        address="local",
        num_cpus=3,
        num_gpus=2,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/cosmos3-stage1-cursor-ray-gpu",
    )
    try:
        data, policy = make_cursor_grid("stage1_cursor_gpu", gpu_request=1)
        assert len(set(data.hosts().values())) == 1
        assert len(set(data.gpus().values())) == 2
        assert_cursor_checkpoint_round_trip(data, policy)
    finally:
        ray.shutdown()
