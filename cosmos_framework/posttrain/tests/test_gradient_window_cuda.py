# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reducer input lifetime under real asynchronous CUDA allocator reuse.

This isolates the cross-stream hook/cast boundary. It does not replace native
FSDP/HSDP training and checkpoint qualification on two physical hosts.
"""

import gc

import pytest
import torch

from cosmos_framework.posttrain.roles.trainer.gradient_window import OwnedGradientWindow, Owner

pytestmark = [pytest.mark.L1, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("elements", [262144, 2097152])
def test_capture_keeps_reduce_scatter_input_alive_through_later_fp32_cast(elements):
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    window = OwnedGradientWindow(
        {"fixture": [Owner("weight", parameter, (1,), 1, 1)]}, microbatches=1, world_size=1
    )
    window.begin_window()
    window.begin_micro(1)
    with torch.cuda.stream(producer):
        value = torch.full((elements,), 3.0, dtype=torch.float64, device="cuda")
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        window.hook("fixture")(value)
        window.pending["fixture"][1].synchronize()
        # Delay the consumer without delaying the allocating stream. Torch's
        # reducer casts after the hook returns and drops its FP64 reference.
        torch.cuda._sleep(200_000_000)
        placeholder = value.float()
    del value
    gc.collect()
    with torch.cuda.stream(producer):
        pressure = [torch.full((elements,), 7.0, dtype=torch.float64, device="cuda") for _ in range(4)]
    torch.cuda.synchronize()
    assert len(pressure) == 4  # retain the pressure allocations through synchronization
    assert torch.equal(window.pending["fixture"][0], torch.full_like(window.pending["fixture"][0], 3.0))
    assert torch.count_nonzero(placeholder) == 0
