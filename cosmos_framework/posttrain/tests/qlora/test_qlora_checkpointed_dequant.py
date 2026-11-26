# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""checkpointed_dequant dispatch and checkpoint/autograd contracts (CPU stubs stand in for NF4)."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.multiprocessing.reductions import StorageWeakRef

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def native_backend(monkeypatch):
    """Load the actual native module with a CPU-only dequantization/backend stand-in."""
    observed = SimpleNamespace(native_calls=0, dequant_calls=0, allocations=0, dense=[])

    class Linear(nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            observed.allocations += 1
            self.compute_dtype = kwargs["compute_dtype"]
            self.quant_state = None

        def forward(self, x):
            observed.native_calls += 1
            return F.linear(x, self.weight.detach())

    def params(data, requires_grad=False, **kwargs):
        p = nn.Parameter(data, requires_grad=requires_grad)
        p.bnb_quantized = False
        p.quant_state = None
        return p

    def dequantize(data, state):
        assert not torch.is_grad_enabled(), "only dequantization must be detached"
        observed.dequant_calls += 1
        result = data.clone()
        observed.dense.append(StorageWeakRef(result.untyped_storage()))
        return result

    fake = SimpleNamespace(
        nn=SimpleNamespace(Linear4bit=Linear, Params4bit=params),
        functional=SimpleNamespace(dequantize_4bit=dequantize),
        __version__="0.49.2",
    )
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.49.2" if name == "bitsandbytes" else original_version(name)
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    source = Path(__file__).parents[2] / "model/quantization/bnb.py"
    name = "_native_checkpointed_dequant_cpu_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, observed


def tiny_linear(backend, *, mode="checkpointed_dequant", dtype=torch.float32):
    module = backend.NativeLoraLinear4bit(
        nn.Linear(8, 8, bias=False, device="meta"),
        SimpleNamespace(rank=2, alpha=4, dropout=0),
        adapter_compute_dtype=dtype,
        quantized_linear_execution=mode,
    )
    g = torch.Generator().manual_seed(91)
    module.weight = nn.Parameter(torch.randn(8, 8, generator=g, dtype=torch.bfloat16) * 0.03, requires_grad=False)
    module.weight.quant_state = object()
    module.quant_state = module.weight.quant_state
    module.lora_A.weight = nn.Parameter(torch.randn(2, 8, generator=g) * 0.02)
    module.lora_B.weight = nn.Parameter(torch.zeros(8, 2))
    return module


@pytest.mark.parametrize("bad", ["", "dense", None, True, []])
def test_unknown_execution_refused_before_allocation(native_backend, bad):
    backend, observed = native_backend
    with pytest.raises(ValueError, match="execution"):
        tiny_linear(backend, mode=bad)
    assert observed.allocations == 0


def test_candidate_refuses_bf16_adapters_before_allocation(native_backend):
    backend, observed = native_backend
    with pytest.raises(ValueError, match="FP32"):
        tiny_linear(backend, dtype=torch.bfloat16)
    assert observed.allocations == 0


def test_default_and_inference_keep_original_backend(native_backend):
    backend, observed = native_backend
    x = torch.ones(1, 8, dtype=torch.bfloat16, requires_grad=True)
    default = tiny_linear(backend, mode="bnb")
    candidate = tiny_linear(backend)
    assert torch.equal(default(x), F.linear(x, default.weight))
    with torch.no_grad():
        assert torch.equal(candidate(x), default(x))
    candidate.eval()
    assert torch.equal(candidate(x), default(x))
    assert observed.native_calls == 5 and observed.dequant_calls == 0
    assert list(default.state_dict()) == list(candidate.state_dict()) == ["weight", "lora_A.weight", "lora_B.weight"]


@pytest.mark.parametrize("input_grad", [False, True])
def test_unwrapped_training_rejected_before_dequantization(native_backend, input_grad):
    backend, observed = native_backend
    module = tiny_linear(backend)
    with pytest.raises(RuntimeError, match="block scope"):
        module(torch.ones(2, 8, dtype=torch.bfloat16, requires_grad=input_grad))
    assert observed.dequant_calls == observed.native_calls == 0


def test_scope_exact_membership_nested_thread_and_exception_cleanup(native_backend):
    backend, _ = native_backend
    one = tiny_linear(backend)
    other = tiny_linear(backend)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exact candidate"):
        backend.checkpointed_dequant_contexts(())
    with pytest.raises(ValueError, match="duplicate"):
        backend.checkpointed_dequant_contexts((one, one))
    with pytest.raises(ValueError, match="exact candidate"):
        backend.checkpointed_dequant_contexts((tiny_linear(backend, mode="bnb"),))
    with pytest.raises(LookupError):
        with backend.checkpointed_dequant_contexts((one,))[0]:
            one(x)
            with pytest.raises(RuntimeError, match="block scope"):
                other(x)
            with pytest.raises(RuntimeError, match="nested"):
                with backend.checkpointed_dequant_contexts((other,))[0]:
                    pass
            one(x)  # Nested refusal must not clear the original owner.
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(RuntimeError, match="block scope"):
                    pool.submit(one, x).result()
            raise LookupError("injected")
    with pytest.raises(RuntimeError, match="block scope"):
        one(x)


@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("early_stop", [False, True])
def test_actual_nonreentrant_checkpoint_updates_and_releases_dense_storage(native_backend, input_grad, early_stop):
    from torch.utils.checkpoint import set_checkpoint_early_stop

    backend, observed = native_backend
    baseline = tiny_linear(backend, mode="bnb")
    candidate = tiny_linear(backend)
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    packed_id = id(candidate.weight)
    state_id = id(candidate.weight.quant_state)
    wrapped = checkpoint_wrapper(
        candidate, preserve_rng_state=True, context_fn=partial(backend.checkpointed_dequant_contexts, (candidate,))
    )
    control = checkpoint_wrapper(baseline, preserve_rng_state=True)
    optimizers = [
        torch.optim.AdamW(m.adapter_parameters().values(), lr=0.01, weight_decay=0) for m in (baseline, candidate)
    ]
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(17), dtype=torch.bfloat16)
    for step in range(3):
        actual = []
        for model, optimizer in zip((control, wrapped), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            leaf = x.detach().clone().requires_grad_(input_grad)
            with set_checkpoint_early_stop(early_stop):
                value = model(leaf)
            if model is wrapped:
                gc.collect()
                assert all(ref.expired() for ref in observed.dense)
            value.float().square().mean().backward()
            actual.append((value.detach(), None if leaf.grad is None else leaf.grad.clone()))
            params = list(model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params if p.requires_grad)
            optimizer.step()
        assert torch.equal(actual[0][0], actual[1][0])
        if input_grad:
            assert torch.equal(actual[0][1], actual[1][1])
        gc.collect()
        assert all(ref.expired() for ref in observed.dense)
        assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(), candidate.parameters(), strict=True))
        with pytest.raises(RuntimeError, match="block scope"):
            candidate(x)
    assert observed.dequant_calls >= 3
    assert torch.equal(candidate.weight, before["weight"]) and candidate.weight.grad is None
    assert id(candidate.weight) == packed_id and id(candidate.weight.quant_state) == state_id
    assert not torch.equal(candidate.lora_A.weight, before["lora_A.weight"])
    assert not torch.equal(candidate.lora_B.weight, before["lora_B.weight"])
    assert list(candidate.state_dict()) == list(before)
    assert not any(isinstance(v, torch.Tensor) for v in vars(candidate).values())


@pytest.mark.parametrize("failure", ["dtype", "shape", "requires_grad"])
def test_candidate_checks_actual_dequantized_weight(native_backend, monkeypatch, failure):
    backend, _ = native_backend
    module = tiny_linear(backend)
    weight = torch.ones(8, 8, dtype=torch.bfloat16)
    if failure == "dtype":
        weight = weight.float()
    elif failure == "shape":
        weight = weight[:1]
    else:
        weight.requires_grad_(True)
    monkeypatch.setattr(backend.bnb.functional, "dequantize_4bit", lambda *args: weight)
    with backend.checkpointed_dequant_contexts((module,))[0], pytest.raises(RuntimeError, match="logical frozen BF16"):
        module(torch.ones(2, 8, dtype=torch.bfloat16))


def test_candidate_compile_refused_before_device_use(native_backend, monkeypatch):
    backend, _ = native_backend
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kw: pytest.fail("must reject before RNG/device access"))
    with pytest.raises(ValueError, match="eager full"):
        backend.kernel_preflight(
            torch.device("cpu"),
            object(),
            compile_model=True,
            quantized_linear_execution="checkpointed_dequant",
        )


def test_failed_checkpoint_recomputation_clears_scope_without_retry(native_backend):
    backend, observed = native_backend
    module = tiny_linear(backend)

    class FailingBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = module
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("injected backward failure")
            return self.linear(x)

    wrapped = checkpoint_wrapper(
        FailingBlock(), preserve_rng_state=True, context_fn=partial(backend.checkpointed_dequant_contexts, (module,))
    )
    x = torch.ones(2, 8, dtype=torch.bfloat16, requires_grad=True)
    value = wrapped(x)
    with pytest.raises(RuntimeError, match="injected backward failure"):
        value.float().square().mean().backward()
    with pytest.raises(RuntimeError, match="block scope"):
        module(x)
    del value, wrapped
    gc.collect()
    assert all(ref.expired() for ref in observed.dense)
