# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks of diagnostic routing/lifetime guards, never CUDA qualification."""

import gc
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

from tools.native_qlora_qualification.tests import test_saved_dequant_cuda as diagnostic


@pytest.mark.parametrize("mode,compiled", [("none", False), ("selective", False), ("full", True)])
def test_unqualified_policy_rejected(mode, compiled):
    with pytest.raises(ValueError, match="only eager full"):
        diagnostic.require_policy(mode, compiled)


def test_actual_wrapper_rejects_reentrant_selective_and_unwrapped():
    module = torch.nn.Linear(2, 2)
    diagnostic.assert_wrappers([checkpoint_wrapper(module, preserve_rng_state=True, determinism_check="default")])
    for wrapper in (
        module,
        checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.REENTRANT),
        checkpoint_wrapper(module, context_fn=lambda: None),
    ):
        with pytest.raises(AssertionError):
            diagnostic.assert_wrappers([wrapper])


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16), requires_grad=False)
        self.weight.quant_state = object()
        self.adapter = torch.nn.Parameter(torch.full((4, 4), 0.125))
        self.logical_shape = (4, 4)
        self.fixture_full_checkpoint = True

    def adapter_delta(self, x, *, output_dtype):
        return torch.nn.functional.linear(x.float(), self.adapter).to(output_dtype)


@pytest.mark.parametrize("input_grad", [True, False])
def test_only_dequant_is_no_grad_and_adapters_still_update(input_grad):
    module = Small()
    x = torch.ones(1, 3, 4, dtype=torch.bfloat16, requires_grad=input_grad)
    seen = []

    def dequant(weight, state):
        assert state is module.weight.quant_state
        seen.append(torch.is_grad_enabled())
        return weight.clone()

    def native(_):
        raise AssertionError("training candidate incorrectly routed through native inference")

    output = diagnostic.saved_forward(module, x, native, dequant)
    output.float().sum().backward()
    assert seen == [False]
    assert (x.grad is not None) == input_grad
    assert torch.count_nonzero(module.adapter.grad) == 16 and module.weight.grad is None
    assert not any(name in vars(module) for name in ("dense_weight", "cached_weight"))


def test_inference_and_unwrapped_training_routes():
    module = Small()
    x = torch.ones(1, 1, 4, dtype=torch.bfloat16)
    calls = []

    def native(value):
        calls.append(value)
        return value

    def dequant(*args):
        raise AssertionError("inference changed its upstream route")

    with torch.no_grad():
        assert diagnostic.saved_forward(module, x, native, dequant) is x
    module.eval()
    assert diagnostic.saved_forward(module, x, native, dequant) is x
    module.train()
    module.fixture_full_checkpoint = False
    with pytest.raises(AssertionError, match="unwrapped"):
        diagnostic.saved_forward(module, x, native, dequant)
    assert len(calls) == 2


def test_storage_handle_survives_python_tensor_and_view_ownership():
    state = object()
    track = diagnostic.DenseTracker({id(state): (0, 0)})

    def dequant(tensor, state):
        return tensor.clone()

    functional = SimpleNamespace(dequantize_4bit=dequant)
    with track.instrument(functional):
        value = functional.dequantize_4bit(torch.ones(4, 4), state)
        view = value.view(-1)
        del value
        gc.collect()
        assert track.live() == 64  # underlying StorageImpl retained by a distinct tensor
        del view
        gc.collect()
        assert track.live() == 0
    assert functional.dequantize_4bit is dequant
    assert track.report()["total_dequantizations"] == 1


def test_tracker_restores_backend_after_exception():
    state = object()

    def original(*args):
        return torch.ones(1)

    functional = SimpleNamespace(dequantize_4bit=original)
    with pytest.raises(RuntimeError, match="intentional"):
        with diagnostic.DenseTracker({id(state): (0, 0)}).instrument(functional):
            raise RuntimeError("intentional")
    assert functional.dequantize_4bit is original


def test_real_shapes_dense_block_bound():
    assert len(diagnostic.SHAPES) == 7 and diagnostic.BLOCK_BYTES == 385875968
    assert all(len(value) == 64 for value in diagnostic.PINNED_SOURCES.values())


def test_cross_block_overlap_is_recorded_even_below_peak_bytes():
    states = [object(), object()]
    tracker = diagnostic.DenseTracker({id(state): (block, 0) for block, state in enumerate(states)})
    functional = SimpleNamespace(dequantize_4bit=lambda tensor, state: tensor.clone())
    with tracker.instrument(functional):
        large = functional.dequantize_4bit(torch.ones(128), states[0])
        del large
        first = functional.dequantize_4bit(torch.ones(4), states[0])
        second = functional.dequantize_4bit(torch.ones(4), states[1])
        assert tracker.peak_bytes == 512 and tracker.peak_blocks == [0]
        assert tracker.max_simultaneous_blocks == 2
        del first, second
        assert tracker.live() == 0


@pytest.mark.parametrize(
    "name,memory,capability,bf16",
    [
        ("NVIDIA A100-SXM4-40GB", 40_000_000_000, (8, 0), True),
        ("NVIDIA H100 80GB HBM3", 80_000_000_000, (9, 0), True),
        ("unqualified 80GB", 80_000_000_000, (8, 0), True),
        ("NVIDIA A100-SXM4-80GB", 80_000_000_000, (8, 6), True),
        ("NVIDIA A100-SXM4-80GB", 80_000_000_000, (8, 0), False),
    ],
)
def test_unqualified_hardware_rejected(name, memory, capability, bf16):
    with pytest.raises(ValueError, match="qualified A100 80GB"):
        diagnostic.require_device(name, memory, capability, bf16)


def test_qualified_a10080_hardware_boundary():
    diagnostic.require_device("NVIDIA A100-SXM4-80GB", 79_000_000_000, (8, 0), True)
    with pytest.raises(ValueError, match="qualified A100 80GB"):
        diagnostic.require_device("NVIDIA A100-SXM4-80GB", 78_999_999_999, (8, 0), True)


def test_absolute_tolerance_cannot_hide_large_relative_gradient_error():
    expected = torch.full((8, 8), 1e-6)
    actual = expected * 2
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=2e-5)
    with pytest.raises(AssertionError):
        diagnostic.difference(actual, expected, rtol=0.03, atol=2e-5)
    assert diagnostic.difference(expected.clone(), expected, rtol=0.03, atol=2e-5)["exact"]


def test_historical_mode_keeps_original_pins(monkeypatch):
    monkeypatch.delenv("COSMOS_SAVED_DEQUANT_IMPLEMENTATION", raising=False)
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_BUILD_RECEIPT_SHA", "ignored")
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_WHEEL_SHA", "ignored")
    assert diagnostic.implementation_pins() == (False, diagnostic.BUILD_RECEIPT_SHA, diagnostic.WHEEL_SHA)


def test_production_mode_requires_explicit_external_pins(monkeypatch):
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_IMPLEMENTATION", "native")
    monkeypatch.delenv("COSMOS_SAVED_DEQUANT_BUILD_RECEIPT_SHA", raising=False)
    monkeypatch.delenv("COSMOS_SAVED_DEQUANT_WHEEL_SHA", raising=False)
    with pytest.raises(KeyError):
        diagnostic.implementation_pins()
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_BUILD_RECEIPT_SHA", "a" * 64)
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_WHEEL_SHA", "b" * 64)
    assert diagnostic.implementation_pins() == (True, "a" * 64, "b" * 64)
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_WHEEL_SHA", "B" * 64)
    with pytest.raises(ValueError, match="externally pinned"):
        diagnostic.implementation_pins()
    monkeypatch.setenv("COSMOS_SAVED_DEQUANT_IMPLEMENTATION", "unknown")
    with pytest.raises(ValueError, match="fixture or native"):
        diagnostic.implementation_pins()
