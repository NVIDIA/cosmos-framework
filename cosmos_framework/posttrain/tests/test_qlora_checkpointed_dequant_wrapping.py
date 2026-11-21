# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU checkpoint/context contracts; the tiny linears are not NF4 kernels."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import sys
import weakref
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils.checkpoint import checkpoint

from cosmos_framework.model.generator import parallelize_vlm
from cosmos_framework.posttrain.model import qwen3_vl
from cosmos_framework.posttrain.model.quantization.loader import canonical_targets

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def graph(monkeypatch):
    """Use the actual bnb scope implementation, with explicitly CPU-only linears."""
    backend = ModuleType("bitsandbytes")
    backend.nn = SimpleNamespace(Linear4bit=nn.Linear)
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.49.2" if name == "bitsandbytes" else original_version(name)
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", backend)
    source = Path(qwen3_vl.__file__).parent / "quantization" / "bnb.py"
    name = "cosmos_framework.posttrain.model.quantization.bnb"
    spec = importlib.util.spec_from_file_location(name, source)
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, api)
    spec.loader.exec_module(api)
    events, dense_refs = [], []

    class CpuLinear(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.logical_shape = shape
            self.quantized_linear_execution = "checkpointed_dequant"
            self.register_buffer("packed_surrogate", torch.eye(2) * 0.95)
            self.adapter = nn.Parameter(torch.full((2, 2), 0.01))

        def forward(self, x):
            if torch.is_grad_enabled():
                if id(self) not in api._CHECKPOINTED_DEQUANT_OWNERS.get():
                    raise RuntimeError("candidate outside its own checkpoint context")
                events.append(id(self))
            dense = self.packed_surrogate.clone()
            dense_refs.append(weakref.ref(dense))
            return torch.nn.functional.linear(x, dense) + torch.nn.functional.linear(x, self.adapter)

    api.NativeLoraLinear4bit = CpuLinear

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.mlp = nn.Module()

        def forward(self, x):
            for child in (*self.self_attn.children(), *self.mlp.children()):
                x = child(x).sin()
            return x

    net = nn.Module()
    net.model = nn.Module()
    net.model.language_model = nn.Module()
    layers = net.model.language_model.layers = nn.ModuleList([Block() for _ in range(36)])
    targets = {}
    for target, shape in canonical_targets().items():
        parent, _, leaf = target.rpartition(".")
        linear = CpuLinear(shape)
        net.get_submodule(parent).add_module(leaf, linear)
        targets[target] = linear
    ac = SimpleNamespace(mode="full", preserve_rng_state=True, determinism_check="default")
    original_state = {name: value.clone() for name, value in net.state_dict().items()}
    original_parameters = {id(value) for value in net.parameters()}
    net._no_split_modules = [Block.__name__]
    slots = parallelize_vlm.apply_ac(SimpleNamespace(model=net), ac)
    return SimpleNamespace(
        net=net,
        layers=layers,
        slots=slots,
        targets=targets,
        ac=ac,
        api=api,
        events=events,
        dense_refs=dense_refs,
        original_state=original_state,
        original_parameters=original_parameters,
        linear_type=CpuLinear,
    )


def bind(graph):
    qwen3_vl._bind_checkpointed_dequant_contexts(graph.net, graph.slots, graph.ac, graph.targets)


def test_all_blocks_real_checkpoint_recompute_cleanup_and_state_layout(graph):
    bind(graph)
    assert {id(value) for value in graph.net.parameters()} == graph.original_parameters
    actual_state = graph.net.state_dict()
    assert actual_state.keys() == graph.original_state.keys()
    assert all(torch.equal(actual_state[name], value) for name, value in graph.original_state.items())
    graph.net.load_state_dict(graph.original_state, strict=True)
    for requires_grad in (True, False):
        x = torch.full((2, 2), 0.1, requires_grad=requires_grad)
        result = x
        begin = len(graph.events)
        for block in graph.layers:
            result = block(result)
        assert len(graph.events) - begin == 252
        assert not graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()
        gc.collect()
        assert all(ref() is None for ref in graph.dense_refs)
        result.sum().backward()
        assert len(graph.events) - begin == 504
        assert not graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()
        assert all(parameter.grad is not None for parameter in graph.net.parameters())
        if requires_grad:
            assert x.grad is not None
        gc.collect()
        assert all(ref() is None for ref in graph.dense_refs)
        graph.net.zero_grad(set_to_none=True)


def test_raw_block_and_neighbor_cannot_borrow_checkpoint_scope(graph):
    bind(graph)
    x = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="outside its own"):
        graph.layers[0]._checkpoint_wrapped_module(x)
    forward_context, _ = graph.layers[0].checkpoint_fn.keywords["context_fn"]()
    with forward_context:
        with pytest.raises(RuntimeError, match="outside its own"):
            graph.layers[1]._checkpoint_wrapped_module(x)
    assert not graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()
    with torch.no_grad():
        graph.layers[0]._checkpoint_wrapped_module(x)


@pytest.mark.parametrize("backward", [False, True])
def test_fatal_forward_or_recompute_exception_does_not_leave_scope(graph, backward):
    bind(graph)
    block = graph.layers[0]._checkpoint_wrapped_module
    original = block.forward
    calls = 0

    class Fatal(BaseException):
        pass

    def fail(x):
        nonlocal calls
        calls += 1
        if calls == (2 if backward else 1):
            raise Fatal()
        return original(x)

    block.forward = fail
    with pytest.raises(Fatal):
        graph.layers[0](torch.ones(2, 2, requires_grad=True)).sum().backward()
    assert not graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()


@pytest.mark.parametrize(
    "damage",
    [
        "no_ac",
        "selective",
        "rng",
        "determinism",
        "missing_slot",
        "duplicate_slot",
        "unregistered_slot",
        "reentrant",
        "changed_callable",
        "bound_args",
        "context_fn",
        "extra_kwarg",
        "missing_target",
        "duplicate_target",
        "wrong_mode",
        "wrong_shape",
        "wrong_owner",
        "nested_wrapper",
        "extra_target",
        "aliased_target",
    ],
)
def test_coverage_and_checkpoint_contract_fail_before_any_binding(graph, damage):
    last = graph.layers[-1]
    if damage in {"no_ac", "selective"}:
        graph.ac.mode = "none" if damage == "no_ac" else "selective"
    elif damage == "rng":
        graph.ac.preserve_rng_state = False
        last.checkpoint_fn.keywords["preserve_rng_state"] = False
    elif damage == "determinism":
        graph.ac.determinism_check = "none"
        last.checkpoint_fn.keywords["determinism_check"] = "none"
    elif damage == "missing_slot":
        graph.slots.pop()
    elif damage == "duplicate_slot":
        graph.slots[-1] = graph.slots[0]
    elif damage == "unregistered_slot":
        graph.layers[-1] = nn.Identity()
    elif damage == "reentrant":
        last.checkpoint_impl = CheckpointImpl.REENTRANT
    elif damage == "changed_callable":
        last.checkpoint_fn = partial(lambda *a, **k: checkpoint(*a, **k), **last.checkpoint_fn.keywords)
    elif damage == "bound_args":
        last.checkpoint_fn = partial(checkpoint, nn.Identity(), **last.checkpoint_fn.keywords)
    elif damage == "context_fn":
        last.checkpoint_fn.keywords["context_fn"] = lambda: (None, None)
    elif damage == "extra_kwarg":
        last.checkpoint_fn.keywords["debug"] = False
    elif damage == "missing_target":
        graph.targets.pop(next(reversed(graph.targets)))
    elif damage == "duplicate_target":
        graph.targets[next(reversed(graph.targets))] = next(iter(graph.targets.values()))
    elif damage == "wrong_mode":
        next(reversed(graph.targets.values())).quantized_linear_execution = "bnb"
    elif damage == "wrong_shape":
        next(reversed(graph.targets.values())).logical_shape = (2, 2)
    elif damage == "wrong_owner":
        last._checkpoint_wrapped_module.mlp.down_proj = nn.Identity()
    elif damage == "nested_wrapper":
        last._checkpoint_wrapped_module.extra = checkpoint_wrapper(nn.Identity())
    elif damage == "extra_target":
        graph.net.extra = graph.linear_type((2, 2))
    elif damage == "aliased_target":
        last._checkpoint_wrapped_module.extra = next(reversed(graph.targets.values()))
    functions = [block.checkpoint_fn for _, _, block in graph.slots]
    with pytest.raises(ValueError, match="checkpointed dequantization"):
        bind(graph)
    assert all(block.checkpoint_fn is old for (_, _, block), old in zip(graph.slots, functions, strict=True))


def test_second_binding_rejects_existing_context_without_mutation(graph):
    bind(graph)
    functions = [block.checkpoint_fn for block in graph.layers]
    with pytest.raises(ValueError, match="unchanged native"):
        bind(graph)
    assert all(block.checkpoint_fn is old for block, old in zip(graph.layers, functions, strict=True))


def test_nested_block_scope_refused_and_outer_scope_restored(graph):
    bind(graph)
    outer, _ = graph.layers[0].checkpoint_fn.keywords["context_fn"]()
    inner, _ = graph.layers[1].checkpoint_fn.keywords["context_fn"]()
    with outer:
        before = graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()
        with pytest.raises(RuntimeError, match="nested"):
            with inner:
                pytest.fail("nested block context must not run")
        assert graph.api._CHECKPOINTED_DEQUANT_OWNERS.get() == before
    assert not graph.api._CHECKPOINTED_DEQUANT_OWNERS.get()


@pytest.mark.parametrize("execution", ["bnb", "checkpointed_dequant"])
def test_native_builder_routes_actual_apply_ac_only_for_opt_in(graph, monkeypatch, execution):
    from cosmos_framework.posttrain.config.model import (
        NativeLoRAConfig,
        QuantizedBaseTrainingConfig,
        Qwen3VLModelConfig,
    )
    from cosmos_framework.posttrain.model.quantization import loader

    # Let the real shared native apply_ac discover and wrap all36 tiny blocks.
    for index, wrapper in enumerate(graph.layers):
        graph.layers[index] = wrapper._checkpoint_wrapped_module
    graph.net._no_split_modules = [type(graph.layers[0]).__name__]
    for module in graph.targets.values():
        module.quantized_linear_execution = execution
    lifecycle = SimpleNamespace(
        targets=graph.targets,
        materialize=lambda _net: None,
        load=lambda *_args: None,
        finalize=lambda _net: None,
    )
    monkeypatch.setattr(loader, "NativeAdapterLifecycle", lambda *_args: lifecycle)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3_vl", tie_word_embeddings=False),
    )

    def build_with_actual_parallelize(**kwargs):
        kwargs["hooks"].parallelize(graph.net, None)
        return graph.net

    monkeypatch.setattr(qwen3_vl, "build_qwen3_vl_policy", build_with_actual_parallelize)
    if execution == "bnb":
        monkeypatch.setattr(
            qwen3_vl, "_bind_checkpointed_dequant_contexts", lambda *_args: pytest.fail("default changed")
        )
    cfg = Qwen3VLModelConfig(
        name_or_path="test/metadata-only",
        attn_impl="sdpa",
        adapter=NativeLoRAConfig(),
        quantized_base=QuantizedBaseTrainingConfig(),
        quantized_linear_execution=execution,
    )
    assert qwen3_vl.build_qwen3_vl_model(cfg) is graph.net
    assert all(
        ("context_fn" in wrapper.checkpoint_fn.keywords) == (execution == "checkpointed_dequant")
        for wrapper in graph.layers
    )
    assert {id(value) for value in graph.net.parameters()} == graph.original_parameters
    assert graph.net.state_dict().keys() == graph.original_state.keys()
