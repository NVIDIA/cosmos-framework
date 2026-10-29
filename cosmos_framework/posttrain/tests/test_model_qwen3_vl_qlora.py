# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU lifecycle contracts. These tests do not qualify a quantized GPU kernel."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.posttrain.model.quantization.loader import (
    SOURCE_ELEMENTS,
    NativeAdapterLifecycle,
    TensorSlot,
    _safetensor_inventory,
    canonical_targets,
    guard_resident_movement,
    materialize_slots,
    parameter_slots,
    resolve_targets,
    tensor_digest,
)
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _profile_model() -> nn.Module:
    net = nn.Module()
    for name, (output, input_) in canonical_targets().items():
        owner = net
        for component in name.split(".")[:-1]:
            if component not in owner._modules:
                owner.add_module(component, nn.Module())
            owner = owner._modules[component]
        owner.add_module(name.rsplit(".", 1)[1], nn.Linear(input_, output, bias=False, device="meta"))
    net.add_module("visual", nn.Linear(8, 8, bias=False, device="meta"))
    net.add_module("lm_head", nn.Linear(8, 8, bias=False, device="meta"))
    return net


def test_profile_resolves_exact_language_coverage_without_allocating_dense_weights():
    net = _profile_model()
    result = resolve_targets(net)
    assert len(result) == 252
    assert sum(module.weight.numel() for module in result.values()) == SOURCE_ELEMENTS
    assert all(module.weight.is_meta for module in result.values())
    assert "visual" not in result and "lm_head" not in result


@pytest.mark.parametrize("failure", ["missing", "shape", "alias", "bias"])
def test_profile_rejects_unmatched_or_excluded_alias_targets(failure):
    net = _profile_model()
    owner = net.get_submodule("model.language_model.layers.0.self_attn")
    if failure == "missing":
        del owner.q_proj
    elif failure == "shape":
        owner.q_proj = nn.Linear(8, 8, bias=False, device="meta")
    elif failure == "bias":
        owner.q_proj.bias = nn.Parameter(torch.empty(4096, device="meta"))
    else:
        net.lm_head.weight = owner.q_proj.weight
    with pytest.raises(ValueError, match="target"):
        resolve_targets(net)


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(12, 8, dtype=torch.bfloat16, device="meta")
        self.embed.weight.requires_grad_(False)
        base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16, device="meta")
        base.weight.requires_grad_(False)
        self.block = LoraInjectedLinear(base, 2, 4, adapter_dtype=torch.float32)
        self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32), persistent=False)

    def forward(self, tokens):
        return self.block(self.embed(tokens))


def _materialize_tiny(seed=42, *, checkpoint=True):
    net = _TinyAdapter()
    adapter = net.block
    slots = parameter_slots(net)
    expected_buffer = net.inv_freq.clone()
    if checkpoint:
        net.block = checkpoint_wrapper(net.block, preserve_rng_state=True)
    materialize_slots(slots, [TensorSlot(net, "inv_freq")], device=torch.device("cpu"), skip=set())
    base_generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        net.embed.weight.copy_(torch.randn(12, 8, generator=base_generator, dtype=torch.bfloat16))
        adapter.weight.copy_(torch.randn(8, 8, generator=base_generator, dtype=torch.bfloat16))
    adapter.reset_adapter_parameters(generator=torch.Generator().manual_seed(seed))
    assert torch.equal(expected_buffer, net.inv_freq)
    assert net.inv_freq.dtype == torch.float32
    return net, adapter, slots


def test_slots_survive_ac_and_dedicated_initialization_ignores_global_rng():
    first, a, slots = _materialize_tiny()
    torch.rand(377)
    second, b, _ = _materialize_tiny()
    assert isinstance(a, NativeLoRAMixin)
    assert slots["block.lora_A.weight"].tensor is a.lora_A.weight
    assert a.lora_A.weight.dtype == torch.float32 and a.weight.dtype == torch.bfloat16
    assert torch.equal(a.lora_A.weight, b.lora_A.weight)
    assert torch.equal(a.lora_B.weight, b.lora_B.weight)
    assert first.embed.weight.requires_grad is False and second.embed.weight.requires_grad is False


def test_nonreentrant_ac_trains_adapters_with_frozen_embedding_over_multiple_updates():
    net, adapter, _ = _materialize_tiny()
    before = (tensor_digest(net.embed.weight), tensor_digest(adapter.weight))
    optimizer = torch.optim.AdamW(adapter.adapter_parameters().values(), lr=0.01, weight_decay=0)
    expected = {id(parameter) for parameter in adapter.adapter_parameters().values()}
    assert {id(parameter) for parameter in net.parameters() if parameter.requires_grad} == expected
    initial_a, initial_b = tensor_digest(adapter.lora_A.weight), tensor_digest(adapter.lora_B.weight)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = net(torch.tensor([[1, 2, 3]])).float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert adapter.lora_A.weight.grad is not None and adapter.lora_B.weight.grad is not None
        if step == 0:
            assert not torch.count_nonzero(adapter.lora_A.weight.grad)
        else:
            assert torch.count_nonzero(adapter.lora_A.weight.grad)
        optimizer.step()
    assert initial_a != tensor_digest(adapter.lora_A.weight)
    assert initial_b != tensor_digest(adapter.lora_B.weight)
    assert before == (tensor_digest(net.embed.weight), tensor_digest(adapter.weight))
    assert net.embed.weight.grad is None and adapter.weight.grad is None


def test_materialization_never_fills_quantized_slots_or_missing_buffer_values():
    net = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16, device="meta")
    materialize_slots(parameter_slots(net), [], device=torch.device("cpu"), skip={"weight"})
    assert net.weight.is_meta
    net.register_buffer("inv_freq", torch.empty(4, device="meta"), persistent=False)
    with pytest.raises(ValueError, match="initialized values"):
        materialize_slots({}, [TensorSlot(net, "inv_freq")], device=torch.device("cpu"), skip=set())


def test_source_inventory_checks_duplicate_and_index_completeness(tmp_path):
    from safetensors.torch import save_file

    save_file({"weight": torch.ones(2, 3, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    inventory = _safetensor_inventory(tmp_path)
    assert inventory["weight"][1:] == ((2, 3), "BF16")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"wrong": "model.safetensors"}}))
    with pytest.raises(ValueError, match="inventory disagree"):
        _safetensor_inventory(tmp_path)
    (tmp_path / "model.safetensors.index.json").unlink()
    save_file({"weight": torch.ones(2, 3, dtype=torch.bfloat16)}, tmp_path / "extra.safetensors")
    with pytest.raises(ValueError, match="duplicate source tensor"):
        _safetensor_inventory(tmp_path)


def test_stream_loader_copies_into_final_slots_without_full_state_dict(tmp_path):
    from safetensors.torch import save_file

    net = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    lifecycle = object.__new__(NativeAdapterLifecycle)
    lifecycle.cfg = SimpleNamespace(name_or_path=str(tmp_path))
    lifecycle.loaded = False
    lifecycle.quantized = False
    lifecycle.targets = {}
    lifecycle.source_slots = parameter_slots(net)
    lifecycle.inventory = _safetensor_inventory(tmp_path)
    net.state_dict = lambda *args, **kwargs: pytest.fail("full-model state_dict extraction is prohibited")
    lifecycle.load(net, str(tmp_path), None, None)
    assert torch.equal(net.weight, expected)
    with pytest.raises(ValueError, match="one unsharded load"):
        lifecycle.load(net, str(tmp_path), None, None)


def test_tensor_fingerprint_hashes_bf16_and_packed_bytes_without_dtype_casts():
    packed = torch.tensor([0, 127, 255], dtype=torch.uint8)
    before = packed.clone()
    assert tensor_digest(packed)["dtype"] == "uint8"
    assert torch.equal(before, packed)
    assert tensor_digest(torch.ones(2, dtype=torch.bfloat16))["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="materialized"):
        tensor_digest(torch.empty(4, device="meta"))


def test_postconstruction_placement_preserves_identity_and_rejects_casts_before_mutation():
    net, adapter, _ = _materialize_tiny()
    original = {id(parameter): tensor_digest(parameter) for parameter in net.parameters()}
    guard_resident_movement(net, torch.device("cpu"))
    assert net.to(device="cpu") is net
    assert {id(parameter): tensor_digest(parameter) for parameter in net.parameters()} == original
    for operation in (lambda: net.to(dtype=torch.bfloat16), lambda: net.to_empty(device="cpu"), lambda: net.float()):
        with pytest.raises(ValueError, match="native adapter model"):
            operation()
    assert {id(parameter): tensor_digest(parameter) for parameter in net.parameters()} == original
    assert adapter.lora_A.weight.dtype == torch.float32
