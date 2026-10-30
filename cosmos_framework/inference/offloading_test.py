# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise real transfers, resident weights, retry safety and repeated requests."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.inference.component_offloading import Cosmos3CPUOffload
from cosmos_framework.inference.layerwise_offloading import Cosmos3LayerwiseOffload
from cosmos_framework.inference.offloading_storage import _LayerWeights
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    InferenceTextKVPrefill,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
    prefill_inference_text_kv,
)
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_compile
from cosmos_framework.model.generator.mot.text_prefill_test import _model, _pack
from cosmos_framework.model.generator.omni_mot_model import _materialize_inference_net_on_cpu
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding

pytestmark = [pytest.mark.L0, pytest.mark.GPU, pytest.mark.gpus(1)]


@pytest.mark.parametrize("mode", ["none", "component", "layerwise"])
def test_request_cache_release_after_temporary_tensors_are_freed(monkeypatch, mode):
    from cosmos_framework.inference.inference import OmniInference

    references = []
    releases = []
    result = []

    def generate(*args, **kwargs):
        temporary = torch.empty(1024, device="cuda")
        references.append(weakref.ref(temporary))
        return result

    def release():
        assert references and references[0]() is None
        releases.append(True)

    pipe = SimpleNamespace(
        setup_args=SimpleNamespace(
            enable_cpu_offload=mode == "component", enable_layerwise_offload=mode == "layerwise"
        ),
        _generate_batch=generate,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", release)
    assert OmniInference.generate_batch(pipe, [], {}) is result
    assert len(releases) == (mode != "none")


@pytest.mark.parametrize("offload_type", [Cosmos3CPUOffload, Cosmos3LayerwiseOffload])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("reasoner_policy", ["high_precision", "base_precision"])
@torch.no_grad()
def test_fp8_mixed_precision_offload(offload_type, compiled, reasoner_policy):
    from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
    from cosmos_framework.utils.generator.mixed_precision import install_mixed_precision_runtime
    from cosmos_framework.utils.generator.mixed_precision_test import _make_fp8_linear

    class Layer(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.self_attn = nn.Module()
            for suffix in ("", "_moe_gen"):
                for name in ("input_layernorm", "post_attention_layernorm"):
                    setattr(self, name + suffix, nn.Identity())
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                    setattr(self.self_attn, name + suffix, nn.Identity())
                setattr(self, "mlp" + suffix, _make_fp8_linear(16, 16, device=device))

        def forward(self, x, *, gen_only=False):
            return self.mlp_moe_gen(x)

        def reasoner_forward(self, x):
            return self.mlp(x)

    def make_net(device):
        model = nn.Module()
        model.layers = nn.ModuleList([Layer(device) for _ in range(3)])
        return _net(model)

    net, reference = make_net("cpu"), make_net("cuda")
    pairs = list(zip(net.language_model.model.layers, reference.language_model.model.layers, strict=True))
    for index, (layer, resident) in enumerate(pairs):
        for name in ("mlp", "mlp_moe_gen"):
            source, target = getattr(layer, name).weight, getattr(resident, name).weight
            source.scale.fill_(0.1 * (index + 1))
            source.act_quant_scale.fill_(0.2 * (index + 1))
            for part in source.__tensor_flatten__()[0]:
                getattr(target, part).copy_(getattr(source, part))
    config = QuantizationConfig(
        mixed_precision_first_steps=1,
        mixed_precision_last_steps=1,
        mixed_precision_reasoner_policy=reasoner_policy,
    )
    runtime = install_mixed_precision_runtime(net, config)
    reference_runtime = install_mixed_precision_runtime(reference, config)
    if compiled:
        # Compare the same execution mode; compiled FP8 arithmetic can round
        # differently from eager execution even without offloading.
        for pair in pairs:
            for layer in pair:
                layer.forward = torch.compile(layer.forward, fullgraph=True)
                layer.reasoner_forward = torch.compile(layer.reasoner_forward, fullgraph=True)
    offloader = offload_type(net, torch.device("cuda", 0))
    x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                offloader.activate("reasoner")
                for layer, resident in pairs:
                    torch.testing.assert_close(layer.reasoner_forward(x), resident.reasoner_forward(x), rtol=0, atol=0)
                offloader.activate("generator")
                for step in range(3):
                    runtime.set_step(step, 3)
                    reference_runtime.set_step(step, 3)
                    for layer, resident in pairs:
                        torch.testing.assert_close(layer(x, gen_only=True), resident(x, gen_only=True), rtol=0, atol=0)
                runtime.reset()
                reference_runtime.reset()
                assert runtime.last_trace == ("W8A16", "W8A8", "W8A16")
                offloader.offload()
                for parameter in net.parameters():
                    assert parameter.device.type == "cpu"
                    for part in parameter.__tensor_flatten__()[0]:
                        assert getattr(parameter, part).device.type == "cpu"
                        assert getattr(parameter, part).is_pinned()
    finally:
        stream.synchronize()
        if isinstance(offloader, Cosmos3LayerwiseOffload):
            offloader.close()
        else:
            offloader.offload()


def _net(model):
    net = nn.Module()
    net.language_model = nn.Module()
    net.language_model.model = model
    return net


def test_cpu_materialization_preserves_fp8_payload_and_scales():
    from cosmos_framework.utils.generator.quantization import swap_modelopt_fp8_linears_on_meta

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16, bias=False, device="meta", dtype=torch.bfloat16)

        def init_weights(self, buffer_device):
            assert self.linear.weight.is_meta

    net = Net()
    swap_modelopt_fp8_linears_on_meta(net, ["linear"])
    _materialize_inference_net_on_cpu(net, buffer_device=torch.device("cuda"))
    weight = net.linear.weight
    assert weight.device.type == "cpu"
    assert weight.dtype == torch.bfloat16
    assert weight.qdata.dtype == torch.float8_e4m3fn
    for part in weight.__tensor_flatten__()[0]:
        assert getattr(weight, part).device.type == "cpu"


@pytest.mark.parametrize("inference_parameter", [False, True])
def test_weight_binding_preserves_inference_status(inference_parameter):
    with torch.inference_mode(inference_parameter):
        parameter = nn.Parameter(torch.arange(6.0).reshape(2, 3).t())
    expected = parameter.detach().clone()
    weights = _LayerWeights([parameter])
    for inference_mode in (True, False, True):
        with torch.inference_mode(inference_mode):
            backing = {dtype: host.to("cuda", non_blocking=True) for dtype, host in weights.host.items()}
            weights.bind(backing)
            assert parameter.is_inference() == inference_parameter
            torch.testing.assert_close(parameter.cpu(), expected)
            weights.restore()
            assert parameter.is_inference() == inference_parameter
            torch.testing.assert_close(parameter, expected)


def test_cpu_materialization_matches_resident_rotary_frequencies():
    config = Qwen3VLTextConfig()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
            with torch.device("meta"):
                self.rotary = Qwen3VLTextRotaryEmbedding(config)

        def init_weights(self, buffer_device):
            assert self.weight.is_meta
            self.rotary.init_weights(buffer_device=buffer_device)

    resident = Qwen3VLTextRotaryEmbedding(config)
    resident.init_weights(buffer_device=torch.device("cuda"))
    net = Net()
    _materialize_inference_net_on_cpu(net, buffer_device=torch.device("cuda"))
    assert all(t.device.type == "cpu" for t in (*net.parameters(), *net.buffers()))
    torch.testing.assert_close(net.rotary.inv_freq, resident.inv_freq.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("offload_type", [Cosmos3CPUOffload, Cosmos3LayerwiseOffload])
@pytest.mark.parametrize("compile_layers", [False, True])
@torch.no_grad()
def test_offload_matches_resident_across_requests(compile_layers, offload_type):
    model, _ = _model()
    net = _net(model)
    inputs = [_pack(model, lengths) for lengths in [(6,), (5, 3), (0,)]]
    install_inference_memory_attention_dispatch(net)
    references = []
    for pack, mask, positions, template in inputs:
        caches = make_inference_text_kv_cache(len(model.layers))
        prefill_inference_text_kv(net, template, caches)
        references.append(
            model(pack, attention_mask=mask, position_ids=positions, memory=InferenceTextKVMemoryState(caches))[0]
        )
    if compile_layers:
        torch._dynamo.reset()
        apply_compile(
            net.language_model,
            CompileConfig(enabled=True, use_cuda_graphs=False, mark_unbacked=False, compile_dynamic=False),
        )
    keys = tuple(net.state_dict())
    net.cpu()  # Checkpoint loading leaves the complete net on CPU.
    offloader = offload_type(net, torch.device("cuda", 0))
    assert tuple(net.state_dict()) == keys
    assert model.embed_tokens.weight.is_cuda
    assert model.rotary_emb.inv_freq.is_cuda
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    before_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    before_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    phases = []
    forward_calls = []

    def activate(name):
        phases.append(name)
        offloader.activate(name)

    hook = model.register_forward_pre_hook(lambda *_: forward_calls.append(True))
    for request in range(2):
        caches = [make_inference_text_kv_cache(2) for _ in inputs]
        prepare = InferenceTextKVPrefill(
            net, [(item[3], cache) for item, cache in zip(inputs, caches, strict=True)], activate
        )
        branches = (0, 1, 2, 0)
        for step, branch in enumerate(branches):
            pack, mask, positions, _ = inputs[branch]
            output, _ = model(
                pack,
                attention_mask=mask,
                position_ids=positions,
                memory=InferenceTextKVMemoryState(caches[branch], prepare=prepare),
            )
            torch.testing.assert_close(get_gen_seq(output), get_gen_seq(references[branch]), atol=0.02, rtol=0.02)
            assert len(forward_calls) == request * len(branches) + step + 1  # No nested model prefill forward.
            assert phases == ["reasoner", "generator"] * (request + 1)
            assert all(cache.is_initialized for group in caches for cache in group)
            assert model.layers[0].mlp.gate_proj.weight.device.type == "cpu"
            if offload_type is Cosmos3CPUOffload:
                current = model.layers[0].mlp_moe_gen.gate_proj.weight.data_ptr()
            else:
                current = tuple(buffer.data_ptr() for slot in offloader._slots for buffer in slot.buffers.values())
            if step == 0:
                backing = current
            else:
                assert current == backing  # No component reload or layer-slot reallocation.
    hook.remove()
    if compile_layers:
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > before_graphs
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == before_breaks
    offloader.offload()
    assert offloader.active_component is None
    assert model.layers[0].mlp_moe_gen.gate_proj.weight.device.type == "cpu"


@pytest.mark.parametrize("failure", ["allocation", "copy"])
def test_failed_transfer_clears_state_and_can_retry(monkeypatch, failure):
    model, _ = _model()
    offloader = Cosmos3CPUOffload(_net(model).cpu(), torch.device("cuda", 0))
    offloader.activate("reasoner")
    original_copy = torch.Tensor.copy_
    original_empty = torch.empty_like
    allocations = 0
    copies = 0

    def fail_copy(target, source, **kwargs):
        nonlocal copies
        if target.is_cuda and source.device.type == "cpu":
            copies += 1
            if copies == 2:  # The first H2D copy is queued; loading has not finished.
                raise torch.OutOfMemoryError("injected transfer failure")
        return original_copy(target, source, **kwargs)

    def fail_allocation(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise torch.OutOfMemoryError("injected allocation failure")
        return original_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        if failure == "copy":
            patch.setattr(torch.Tensor, "copy_", fail_copy)
        else:
            patch.setattr(torch, "empty_like", fail_allocation)
        with pytest.raises(torch.OutOfMemoryError, match="injected"):
            offloader.activate("generator")
    assert offloader.active_component is None
    assert not offloader._device_weights
    assert all(p.device.type == "cpu" for layer in model.layers for p in layer.parameters())
    offloader.activate("reasoner")
    assert model.layers[0].mlp.gate_proj.weight.is_cuda
    with pytest.raises(ValueError, match="Unknown"):
        offloader.activate("vae")
    offloader.offload()


def test_component_transfers_use_pinned_masters_without_d2h(monkeypatch):
    model, _ = _model()
    offloader = Cosmos3CPUOffload(_net(model).cpu(), torch.device("cuda", 0))
    masters = [host for layers in offloader._groups.values() for weights in layers for host in weights.host.values()]
    pointers = [host.data_ptr() for host in masters]
    assert all(host.is_pinned() for host in masters)
    copies = []
    original_copy = torch.Tensor.copy_

    def copy(target, source, **kwargs):
        copies.append((target.device.type, source.device.type, source.is_pinned(), kwargs.get("non_blocking")))
        return original_copy(target, source, **kwargs)

    def forbid_move(*args, **kwargs):
        raise AssertionError("Phase changes must rebind CPU masters, not move tensors")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", copy)
        patch.setattr(torch.Tensor, "to", forbid_move)
        patch.setattr(torch.Tensor, "cpu", forbid_move)
        with torch.cuda.stream(stream):
            for name in ("reasoner", "generator", "reasoner"):
                offloader.activate(name)
                before = len(copies)
                offloader.activate(name)
                assert len(copies) == before
            offloader.offload()
    assert copies and all(copy == ("cuda", "cpu", True, True) for copy in copies)
    assert [host.data_ptr() for host in masters] == pointers
    assert all(p.device.type == "cpu" for layer in model.layers for p in layer.parameters())


@torch.no_grad()
def test_resident_fp8_moves_payload_and_scales_without_replacing_parameter():
    from cosmos_framework.inference.offloading_storage import _move_resident_tensors
    from cosmos_framework.utils.generator.mixed_precision_test import _make_fp8_linear

    linear = _make_fp8_linear(16, 16, device="cpu")
    reference = _make_fp8_linear(16, 16, device="cuda")
    weight = linear.weight
    names, _ = weight.__tensor_flatten__()
    for name in names:
        getattr(reference.weight, name).copy_(getattr(weight, name))
    _move_resident_tensors(linear, set(), torch.device("cuda", 0))
    assert linear.weight is weight
    assert weight.is_cuda
    assert all(getattr(weight, name).is_cuda for name in names)
    x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(linear(x), reference(x), rtol=0, atol=0)
