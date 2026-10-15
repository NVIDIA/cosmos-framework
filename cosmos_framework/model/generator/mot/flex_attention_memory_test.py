# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Periodic GC must reclaim unreachable inputs from fresh Flex/Flash compilations."""

import gc
import time
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.mot import flex_attention as attention_module
from cosmos_framework.model.generator.mot.flex_attention import FlexBackend, resolve_flex_backend
from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_selective_ac


def _mask(length: int, backend: FlexBackend) -> BlockMask:
    q_len, kv_len = 256 * (1 + length % 3), 1152 + 128 * length
    q_view = torch.arange(q_len, device="cuda") // 64 % 7  # [Q]
    kv_view = torch.arange(kv_len, device="cuda") // 128 % 7  # [K]

    def predicate(
        _b: torch.Tensor,  # []
        _h: torch.Tensor,  # []
        q_idx: torch.Tensor,  # []
        kv_idx: torch.Tensor,  # []
    ) -> torch.Tensor:  # []
        return (q_view[q_idx] == kv_view[kv_idx]) | (kv_idx < 128)  # []

    return create_block_mask(
        predicate, B=1, H=1, Q_LEN=q_len, KV_LEN=kv_len, device="cuda", BLOCK_SIZE=backend.block_size
    )


class _MemoryAttention(torch.nn.Module):
    backend: FlexBackend

    def __init__(self, backend: FlexBackend) -> None:
        super().__init__()
        self.backend = backend

    def forward(
        self,
        q: torch.Tensor,  # [1,Q,4,128]
        k: torch.Tensor,  # [1,K,2,128]
        v: torch.Tensor,  # [1,K,2,128]
        block_mask: BlockMask,
    ) -> torch.Tensor:  # [1,Q,4,128]
        # Reproduce non-leaf K/V concatenation, layout conversion and explicit
        # checkpoint mask arguments without allocating a production-size model.
        keys = torch.cat((k[:, :128], k[:, 128:]), dim=1)  # [1,K,2,128]
        values = torch.cat((v[:, :128], v[:, 128:]), dim=1)  # [1,K,2,128]
        return attention_module.flex_attention(q, keys, values, block_mask, self.backend)  # [1,Q,4,128]


def _memory_step(
    module: torch.nn.Module,
    backend: FlexBackend,
    length: int,
) -> list[weakref.ReferenceType[object]]:
    """Return weak references only; no completed-step graph is intentionally kept."""
    mask = _mask(length, backend)
    q_len, kv_len = mask.shape[-2:]
    inputs = [
        torch.randn(1, count, heads, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for count, heads in ((q_len, 4), (kv_len, 2), (kv_len, 2))
    ]  # list[[1,S,H,128]]
    output = module(*inputs, mask)  # [1,Q,4,128]
    loss = output.float().square().mean()  # []
    loss.backward()
    for value in inputs:
        assert value.grad is not None and torch.isfinite(value.grad).all().item()
    tensors = [value for value in vars(mask).values() if isinstance(value, torch.Tensor)]  # list[[...]]
    return [weakref.ref(value) for value in [*inputs, output, loss, mask, *tensors]]


# Cold FA4 compilation and full-generation GC need headroom beyond CI's 60s default.
# Run in the isolated GPU phase so other test workers do not compete for CPU time.
@pytest.mark.serial
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fresh compiled Flash tensor lifetime requires CUDA")
@pytest.mark.parametrize("selective_ac", [False, True], ids=["eager", "selective"])
@pytest.mark.parametrize(
    "fresh_shape_each_step",
    [
        pytest.param(False, marks=[pytest.mark.L0, pytest.mark.timeout(180)], id="smoke"),
        pytest.param(True, marks=[pytest.mark.L1, pytest.mark.timeout(600)], id="stress"),
    ],
)
def test_periodic_gc_releases_fresh_flash_inputs(
    monkeypatch: pytest.MonkeyPatch,
    selective_ac: bool,
    fresh_shape_each_step: bool,
    record_property: Callable[[str, Any], None],
) -> None:
    torch.manual_seed(1729)
    device = torch.device("cuda")
    reason = attention_module.flash_backend_unavailable_reason(device)
    if reason is not None:
        pytest.skip(reason)
    backend = resolve_flex_backend(device, preference="flex_flash")
    # Keep the original regression compile budget so all 30 fresh shapes stay cached,
    # independently of any training recipe.
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 512)
    # Force static shape specializations to exercise GC with the shared wrapper.
    compiled = torch.compile(torch_flex_attention, dynamic=False, fullgraph=True)
    monkeypatch.setattr(attention_module, "_COMPILED_FLEX_ATTENTION", compiled)
    attention = _MemoryAttention(backend)
    module = (
        _apply_selective_ac(attention, ActivationCheckpointingConfig(mode="selective", save_ops_regex=["fmha"]))
        if selective_ac
        else attention
    )
    # Preserve the original regression cadence without importing a Model 2 recipe.
    policy = dict(every_n=10, warm_up=0, gc_level=2)
    callback = ManualGarbageCollection(**policy)
    was_enabled = gc.isenabled()
    references: list[weakref.ReferenceType[object]] = []
    observations: list[dict[str, float | int]] = []
    measured_shapes: set[int] = set()
    try:
        # Isolate the shared code cache without releasing compiled graphs during measurement.
        torch._dynamo.reset()
        for _ in range(3):
            _memory_step(module, backend, 0)
        gc.collect(2)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for iteration in range(1, 31):
            callback.on_training_step_start(None, {}, iteration)
            assert not gc.isenabled()
            # L0 compiles a new shape in each of the first two GC windows, then reuses one.
            # L1 compiles a fresh shape every step and retains the growing compiler cache.
            length = iteration if fresh_shape_each_step else 1 + ((iteration - 1) // policy["every_n"]) % 2
            measured_shapes.add(length)
            references.extend(_memory_step(module, backend, length))
            if iteration % policy["every_n"] == 0:
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                start = time.monotonic()
                callback.every_n_impl(None, None, {}, {}, None, iteration)
                collection_seconds = time.monotonic() - start
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated()
                live = sum(reference() is not None for reference in references)
                observations.append(
                    {
                        "iteration": iteration,
                        "before_allocated_bytes": before,
                        "after_allocated_bytes": after,
                        "live_weak_references": live,
                        "collection_seconds": collection_seconds,
                    }
                )
                assert live == 0, observations
                assert after <= baseline + 1024**2, observations
        record_property("selective_ac", selective_ac)
        record_property("gc_policy", policy)
        record_property("post_collection_memory", observations)
        record_property("fresh_shapes", len(measured_shapes))
        record_property("measured_steps", 30)
    finally:
        torch._dynamo.reset()
        gc.collect(2)
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
