# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""`utils/profile.py::GpuSpan`: a CUDA-event span that never blocks and reads None until it can answer."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.utils.profile import GpuSpan


@pytest.mark.L0
@pytest.mark.CPU
def test_gpu_span_reads_none_until_both_events_completed() -> None:
    span = GpuSpan()
    assert span.seconds() is None, "never entered: nothing to measure"
    with span:
        pass
    if not torch.cuda.is_available():
        assert span.seconds() is None, "no CUDA: no events, no number"
        return
    torch.cuda.synchronize()  # the caller's own sync point; the span itself never synchronizes
    elapsed = span.seconds()
    assert isinstance(elapsed, float) and elapsed >= 0.0
