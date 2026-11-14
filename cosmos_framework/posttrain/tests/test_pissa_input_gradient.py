# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""PiSSA input adjoints, legacy identity, and explicit backward-policy ownership."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cosmos_framework.posttrain.checkpoint.adapters import (
    FORMAT,
    PISSA_GRADIENT_FORMAT,
    adapter_configuration,
    make_adapter_identity,
    state_spec,
)
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import (
    PosttrainAdapterLinear,
    assert_trainability,
    install_adapters,
    optimizer_groups,
    resolve_targets,
)
from cosmos_framework.posttrain.tests.test_adapters import FIRST, identity, tiny_net


def component(**kwargs):
    return Component(targets=[FIRST], method="pissa", rank=3, alpha=7, initialization="svd", **kwargs)


def layer(*, policy="shared_fp32_v1", dtype=torch.bfloat16, rs=False):
    torch.manual_seed(31)
    base = nn.Linear(8, 12, dtype=dtype)
    base.requires_grad_(False)
    return PosttrainAdapterLinear(base, component(pissa_input_gradient=policy, use_rslora=rs))


def legacy_forward(module, inputs):
    """Frozen pre-correction expression, including its two independent casts."""
    base = F.linear(inputs, module.weight, module.bias)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        delta = F.linear(F.linear(inputs.float(), module.lora_A.weight), module.lora_B.weight) * module._lora_scale
        initial = F.linear(F.linear(inputs.float(), module.initial_A), module.initial_B) * module._lora_scale
        return (base.float() + (delta - initial)).to(base.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rs", [False, True])
def test_initial_residual_input_adjoint_is_exactly_the_frozen_base(dtype, rs):
    module = layer(dtype=dtype, rs=rs)
    inputs = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    cotangent = torch.randn(2, 5, 12, dtype=dtype)
    output = module(inputs)
    expected = F.linear(inputs, module.weight, module.bias)
    assert torch.equal(output, expected)
    actual_grad = torch.autograd.grad(output, inputs, cotangent)[0]
    expected_grad = torch.autograd.grad(expected, inputs, cotangent)[0]
    # At A=A0/B=B0, the residual branch has zero input derivative. Cancellation
    # occurs before casting that adjoint back to BF16, even for rsPiSSA.
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rs", [False, True])
def test_legacy_graph_and_shared_forward_and_factor_gradients(dtype, rs):
    original = layer(policy="separate_casts_v1", dtype=dtype, rs=rs)
    with torch.no_grad():
        original.lora_A.weight.add_(0.003)
        original.lora_B.weight.sub_(0.007)
    shared = copy.deepcopy(original)
    shared.pissa_input_gradient = "shared_fp32_v1"
    inputs = torch.randn(2, 7, 8, dtype=dtype)
    cotangent = torch.randn(2, 7, 12, dtype=dtype)
    observations = []
    for module, forward in ((original, original), (original, lambda x: legacy_forward(original, x)), (shared, shared)):
        x = inputs.detach().clone().requires_grad_(True)
        y = forward(x)
        grads = torch.autograd.grad(y, (x, module.lora_A.weight, module.lora_B.weight), cotangent)
        observations.append((y.detach(), grads))
    assert torch.equal(observations[0][0], observations[1][0])
    assert all(torch.equal(a, b) for a, b in zip(observations[0][1], observations[1][1], strict=True))
    assert torch.equal(observations[0][0], observations[2][0])
    assert all(torch.equal(a, b) for a, b in zip(observations[0][1][1:], observations[2][1][1:], strict=True))


@pytest.mark.parametrize("rs", [False, True])
def test_multiple_updates_and_activation_replay_preserve_owners_and_residual(rs):
    plain = layer(rs=rs)
    replay = copy.deepcopy(plain)
    frozen = {name: tensor.clone() for name, tensor in plain.state_dict().items() if "lora_" not in name}
    owners = [[id(p) for p in model.parameters()] for model in (plain, replay)]
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01) for model in (plain, replay)
    ]
    initial = plain.lora_A.weight.detach().clone()
    for _ in range(3):
        inputs = torch.randn(2, 5, 8, dtype=torch.bfloat16)
        grads = []
        for index, (model, optimizer) in enumerate(zip((plain, replay), optimizers, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            x = inputs.detach().clone().requires_grad_(True)
            y = checkpoint(model, x, use_reentrant=False) if index else model(x)
            loss = (y.float() - 0.4).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            grads.append([p.grad.clone() for p in model.parameters() if p.requires_grad])
            assert all(torch.isfinite(g).all() for g in grads[-1])
            optimizer.step()
        assert all(torch.equal(a, b) for a, b in zip(*grads, strict=True))
        assert all(torch.equal(value, replay.state_dict()[name]) for name, value in plain.state_dict().items())
    assert not torch.equal(initial, plain.lora_A.weight)
    for index, model in enumerate((plain, replay)):
        assert [id(p) for p in model.parameters()] == owners[index]
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "lora", "initialization": "default"},
        {"method": "dora", "initialization": "default"},
        {"execution": "pissa_dense_v1"},
        {"execution": "merged_weight_bf16_v1"},
        {"dropout": 0.1},
        {"pissa_input_gradient": "unknown"},
    ],
)
def test_invalid_input_gradient_combinations_rejected(changes):
    settings = dict(targets=[FIRST], method="pissa", pissa_input_gradient="shared_fp32_v1")
    settings.update(changes)
    with pytest.raises(ValueError):
        Component(**settings)


def test_default_identities_stay_legacy_and_nondefault_is_versioned():
    net = tiny_net()
    legacy = resolve_targets(net, AdapterConfig(components=[component()]))
    assert all("pissa_input_gradient" not in row["settings"] for row in legacy["targets"])
    assert state_spec(identity(legacy))
    shared = resolve_targets(net, AdapterConfig(components=[component(pissa_input_gradient="shared_fp32_v1")]))
    metadata = make_adapter_identity(
        "unused",
        "unused",
        shared,
        {},
        base_dtype="float32",
        source_identity={"base": {"sha256": "fixture"}, "processor": {"sha256": "fixture"}, "model_contract": {}},
    )
    assert metadata["format"] == PISSA_GRADIENT_FORMAT
    assert state_spec(metadata) == state_spec(identity(legacy))
    assert resolve_targets(net, adapter_configuration(metadata)) == shared
    with pytest.raises(ValueError, match="format/execution mismatch"):
        state_spec({**metadata, "format": FORMAT})
    with pytest.raises(ValueError, match="format/execution mismatch"):
        state_spec({**identity(legacy), "format": PISSA_GRADIENT_FORMAT})


@pytest.mark.parametrize("bad", ["policy", "method", "execution", "dropout", "model_execution"])
def test_manifest_cannot_smuggle_unsupported_backward_policy(bad):
    net = tiny_net()
    manifest = resolve_targets(net, AdapterConfig(components=[component(pissa_input_gradient="shared_fp32_v1")]))
    metadata = {**identity(manifest), "format": PISSA_GRADIENT_FORMAT}
    settings = metadata["resolved"]["targets"][0]["settings"]
    if bad == "policy":
        settings["pissa_input_gradient"] = "unknown"
    elif bad == "method":
        settings["method"] = "dora"
    elif bad == "execution":
        settings["execution"] = "pissa_dense_v1"
    elif bad == "dropout":
        settings["dropout"] = 0.1
    else:
        metadata["resolved"]["model_execution"] = "qwen_vl_hf_ordered_v1"
    with pytest.raises(ValueError, match="PiSSA|shared PiSSA"):
        state_spec(metadata)


def test_tied_aliases_require_identical_input_gradient_policy_before_injection():
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    net = tiny_net()
    block = net.model.language_model.layers[0].self_attn
    block.v_proj.weight = block.q_proj.weight
    second = FIRST.replace("q_proj", "v_proj")
    left = component(pissa_input_gradient="shared_fp32_v1")
    right = component()
    right.targets = [second]
    with pytest.raises(ValueError, match="incompatible settings for tied aliases"):
        install_adapters(net, AdapterConfig(components=[left, right]))
    assert type(block.q_proj) is type(block.v_proj) is nn.Linear
    right.pissa_input_gradient = "shared_fp32_v1"
    install_adapters(net, AdapterConfig(components=[left, right]))
    assert block.q_proj.weight is block.v_proj.weight
    assert block.q_proj.lora_A.weight is block.v_proj.lora_A.weight
    assert block.q_proj.lora_B.weight is block.v_proj.lora_B.weight
    optimizer = build_optimizer(net, OptimizerConfig(lr=0.01), parameter_groups=optimizer_groups(net, 0.01))
    assert_trainability(net, optimizer)
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len({id(p) for p in owned}) == 2


def test_model_profile_conflict_rejected_before_construction():
    with pytest.raises(ValueError, match="default_v1 model execution"):
        AdapterConfig(
            components=[component(pissa_input_gradient="shared_fp32_v1")], model_execution="qwen_vl_hf_ordered_v1"
        )
