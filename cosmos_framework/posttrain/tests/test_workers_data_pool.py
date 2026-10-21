# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-level invariants of the Ray/NIXL data pool: field alignment and byte encoding."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ray")  # workers.data_pool imports `ray` at module load; CI images without ray skip

from cosmos_framework.posttrain.workers import data_pool  # noqa: E402
from cosmos_framework.posttrain.workers.data_pool import (  # noqa: E402
    FIELD_ALIGN_BYTES,
    PageAllocator,
    actor_cuda_accessible,
    plan_page_packing,
    tensor_as_bytes,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

PAGE = 4096


@pytest.mark.parametrize(
    ("cuda_available", "ray_initialized", "gpu_ids", "expected"),
    [
        (False, True, ["0"], False),  # no CUDA at all
        (True, False, [], True),  # outside Ray: torch visibility is the only signal
        (True, True, [], False),  # CPU-placed Ray actor that still sees the node's GPUs
        (True, True, ["0"], True),  # GPU-assigned Ray actor
    ],
)
def test_actor_cuda_accessible_requires_a_ray_gpu_assignment(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    ray_initialized: bool,
    gpu_ids: list[str],
    expected: bool,
) -> None:
    monkeypatch.setattr(data_pool.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(data_pool.ray, "is_initialized", lambda: ray_initialized)
    monkeypatch.setattr(data_pool.ray, "get_gpu_ids", lambda: gpu_ids)
    assert actor_cuda_accessible() is expected


def test_plan_page_packing_aligns_the_next_field() -> None:
    packing = plan_page_packing(4, PAGE, normal_used=1)
    assert (packing.offset, packing.reserve_units, packing.oversize) == (FIELD_ALIGN_BYTES, 0, False)


def test_plan_page_packing_spills_when_only_the_unaligned_placement_fits() -> None:
    used = PAGE - 300  # unaligned tail of 300 bytes; aligned tail is only 256
    assert plan_page_packing(300, PAGE, normal_used=used) == plan_page_packing(300, PAGE, normal_used=None)
    assert plan_page_packing(256, PAGE, normal_used=used).offset == PAGE - 256


def test_page_allocator_places_fields_on_aligned_offsets() -> None:
    allocator = PageAllocator(2 * PAGE, page_bytes=PAGE)
    first = allocator.allocate("lease", 1)
    second = allocator.allocate("lease", 4)
    assert first.page_id == second.page_id
    assert (first.offset, second.offset) == (0, FIELD_ALIGN_BYTES)


def test_page_allocator_rejects_unaligned_page_size() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        PageAllocator(2 * 100, page_bytes=100)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor(3.5, dtype=torch.float32),
        torch.tensor(7, dtype=torch.int64),
        torch.arange(6, dtype=torch.float16).reshape(2, 3),
        torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),  # non-contiguous
    ],
)
def test_tensor_as_bytes_round_trips_scalars_and_shapes(tensor: torch.Tensor) -> None:
    raw = tensor_as_bytes(tensor)
    assert raw.dtype == torch.uint8 and raw.dim() == 1 and raw.numel() == tensor.numel() * tensor.element_size()
    rebuilt = raw.view(tensor.dtype).reshape(tensor.shape)
    assert torch.equal(rebuilt, tensor)
