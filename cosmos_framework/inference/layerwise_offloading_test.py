# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CUDA transfers, compiled decoder execution, and staging-slot lifetimes."""

import copy

import pytest
import torch

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.inference.component_offloading import Cosmos3CPUOffload
from cosmos_framework.inference.layerwise_offloading import Cosmos3LayerwiseOffload
from cosmos_framework.inference.offloading_storage import _LayerWeights
from cosmos_framework.inference.offloading_test import _net
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    InferenceTextKVPrefill,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
    prefill_inference_text_kv,
)
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_compile
from cosmos_framework.model.generator.mot.text_prefill_test import _model, _pack

pytestmark = [pytest.mark.L0, pytest.mark.GPU, pytest.mark.gpus(1)]


def _setup(layer_count=3):
    torch.manual_seed(42)
    model, _ = _model()
    while len(model.layers) < layer_count:
        model.layers.append(copy.deepcopy(model.layers[0]))
    del model.layers[layer_count:]
    net = _net(model)
    install_inference_memory_attention_dispatch(net)
    return model, net


@pytest.mark.parametrize("layer_count", [1, 3])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_layerwise_first_forward_prefill_on_nondefault_stream(layer_count, compiled):
    model, net = _setup(layer_count)
    inputs = [_pack(model, lengths) for lengths in [(6,), (5, 3)]]
    references = []
    for pack, mask, positions, template in inputs:
        caches = make_inference_text_kv_cache(layer_count)
        prefill_inference_text_kv(net, template, caches)
        references.append(
            model(pack, attention_mask=mask, position_ids=positions, memory=InferenceTextKVMemoryState(caches))[0]
        )
    if compiled:
        torch._dynamo.reset()
        apply_compile(
            net.language_model,
            CompileConfig(enabled=True, use_cuda_graphs=False, mark_unbacked=False, compile_dynamic=False),
        )
    net.cpu()
    keys = tuple(net.state_dict())
    offloader = Cosmos3LayerwiseOffload(net, torch.device("cuda", 0))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    before_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    before_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    try:
        with torch.cuda.stream(stream):
            # Repeat phase switches and odd-length passes on a non-default
            # compute stream, including a changed packed CFG batch shape.
            for _ in range(2):
                branch_caches = [make_inference_text_kv_cache(layer_count) for _ in inputs]
                prepare = InferenceTextKVPrefill(
                    net,
                    [(item[3], cache) for item, cache in zip(inputs, branch_caches, strict=True)],
                    offloader.activate,
                )
                slots = None
                for (pack, mask, positions, _), reference, caches in zip(
                    inputs, references, branch_caches, strict=True
                ):
                    for _ in range(3):
                        actual, _ = model(
                            pack,
                            attention_mask=mask,
                            position_ids=positions,
                            memory=InferenceTextKVMemoryState(caches, prepare=prepare),
                        )
                        torch.testing.assert_close(get_gen_seq(actual), get_gen_seq(reference), atol=0.02, rtol=0.02)
                        current = [tuple(t.data_ptr() for t in slot.buffers.values()) for slot in offloader._slots]
                        if slots is None:
                            slots = current
                        else:
                            assert slots == current
                    assert all(p.device.type == "cpu" for layer in model.layers for p in layer.parameters())
        stream.synchronize()
        assert tuple(net.state_dict()) == keys
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == before_breaks
        if compiled:
            assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > before_graphs
    finally:
        offloader.close()
    assert not offloader._slots
    with pytest.raises(RuntimeError, match="closed"):
        offloader.activate("generator")


@pytest.mark.parametrize("failure", ["compute", "prefetch"])
@torch.no_grad()
def test_layerwise_failure_drains_transfers_and_can_retry(monkeypatch, failure):
    model, net = _setup()
    pack, mask, positions, template = _pack(model, (6,))
    reference, _ = model(
        pack,
        attention_mask=mask,
        position_ids=positions,
        memory=InferenceTextKVMemoryState(make_inference_text_kv_cache(3)),
    )
    fail = False
    original_forward = model.layers[1].forward

    def forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if fail and failure == "compute":
            raise RuntimeError("injected compute failure")
        return output

    monkeypatch.setattr(model.layers[1], "forward", forward)
    offloader = Cosmos3LayerwiseOffload(net.cpu(), torch.device("cuda", 0))
    original_prefetch = offloader._prefetch

    def prefetch(weights, slot):
        original_prefetch(weights, slot)
        if fail and failure == "prefetch":
            raise RuntimeError("injected prefetch failure")

    monkeypatch.setattr(offloader, "_prefetch", prefetch)
    try:
        caches = make_inference_text_kv_cache(3)
        prepare = InferenceTextKVPrefill(net, [(template, caches)], offloader.activate)
        fail = True
        with pytest.raises(RuntimeError, match="injected"):
            model(
                pack,
                attention_mask=mask,
                position_ids=positions,
                memory=InferenceTextKVMemoryState(caches, prepare=prepare),
            )
        assert offloader.active_component is None
        assert not offloader._slots
        assert all(p.device.type == "cpu" for layer in model.layers for p in layer.parameters())
        fail = False
        caches = make_inference_text_kv_cache(3)  # A retry starts a fresh request after partial prefill.
        prepare = InferenceTextKVPrefill(net, [(template, caches)], offloader.activate)
        actual, _ = model(
            pack,
            attention_mask=mask,
            position_ids=positions,
            memory=InferenceTextKVMemoryState(caches, prepare=prepare),
        )
        torch.testing.assert_close(get_gen_seq(actual), get_gen_seq(reference), atol=0.02, rtol=0.02)
    finally:
        offloader.close()
    assert model.layers[1].forward is forward


@torch.no_grad()
def test_layerwise_host_storage_preserves_mixed_dtypes_and_strides():
    tensors = [torch.randn(4, 6, dtype=torch.bfloat16).t(), torch.arange(5, dtype=torch.float32)]
    originals = [tensor.clone() for tensor in tensors]
    strides = [tensor.stride() for tensor in tensors]
    weights = _LayerWeights(tensors)
    assert all(host.is_pinned() for host in weights.host.values())
    backing = {dtype: host.to("cuda", non_blocking=True) for dtype, host in weights.host.items()}
    weights.bind(backing)
    for tensor, expected, stride in zip(tensors, originals, strides, strict=True):
        assert tensor.is_cuda and tensor.stride() == stride
        torch.testing.assert_close(tensor.cpu(), expected, atol=0, rtol=0)
    weights.restore()
    for tensor, expected in zip(tensors, originals, strict=True):
        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)


@torch.no_grad()
def test_layerwise_reasoner_forward_and_allocation_retry(monkeypatch):
    model, net = _setup()
    ids = torch.randint(0, 128, (1, 6), device="cuda")
    expected = model.reasoner_forward(ids, cache=None)
    offloader = Cosmos3LayerwiseOffload(net.cpu(), torch.device("cuda", 0))
    original_empty = torch.empty
    allocations = 0

    def empty(*args, **kwargs):
        nonlocal allocations
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            allocations += 1
            if allocations == 2:
                raise torch.OutOfMemoryError("injected allocation failure")
        return original_empty(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", empty)
            with pytest.raises(torch.OutOfMemoryError, match="injected"):
                offloader.activate("reasoner")
        assert not offloader._slots and offloader.active_component is None
        offloader.activate("reasoner")
        actual = model.reasoner_forward(ids, cache=None)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        offloader.close()


@pytest.mark.parametrize("offloader_type", [Cosmos3CPUOffload, Cosmos3LayerwiseOffload])
@torch.no_grad()
def test_offload_rejects_shared_decoder_storage_without_mutation(offloader_type):
    model, net = _setup()
    net.cpu()
    model.layers[1].mlp.gate_proj.weight = model.layers[0].mlp.gate_proj.weight
    before = {name: (tensor.data_ptr(), tensor.shape) for name, tensor in net.state_dict().items()}
    with pytest.raises(ValueError, match="aliased"):
        offloader_type(net, torch.device("cuda", 0))
    assert before == {name: (tensor.data_ptr(), tensor.shape) for name, tensor in net.state_dict().items()}
