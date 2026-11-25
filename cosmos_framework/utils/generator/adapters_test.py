# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cosmos_framework.utils.generator.adapter_config import AdapterComponentConfig, AdapterConfig, adapter_architecture
from cosmos_framework.utils.generator.adapters import (
    AdapterLinear,
    adapter_state_spec,
    assert_trainability,
    identity_differences,
    install_adapters,
    merged_target_weight,
    model_base_fingerprint,
    require_adapter_calls,
    resolve_targets,
    tensor_identity,
    trainable_spec,
)
from cosmos_framework.utils.generator.lora import apply_lora_trainable_scope, merge_lora_adapters_

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

METHODS = [("lora", False), ("lora", True), ("dora", False), ("dora", True), ("pissa", False), ("pissa", True)]


class _Attention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.norm = nn.LayerNorm(hidden)


class _TinyText(nn.Module):
    """Module-path layout of a Hugging Face text decoder (``model.layers.N...``)."""

    def __init__(self, hidden: int = 16, layers: int = 3, *, tied_target: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block(hidden) for _ in range(layers))
        self.lm_head = nn.Linear(hidden, 32, bias=False)
        if tied_target:
            self.model.layers[1].self_attn.v_proj.weight = self.model.layers[0].self_attn.v_proj.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            x = block.norm(x + block.self_attn.v_proj(block.self_attn.q_proj(x)))
        return self.lm_head(x)


def _config(method: str = "lora", **kwargs) -> AdapterConfig:
    targets = kwargs.pop("targets", ["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"])
    return AdapterConfig(
        components=[AdapterComponentConfig(targets=targets, method=method, rank=4, alpha=8.0, **kwargs)]
    )


def _perturb(net: nn.Module, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in net.parameters():
            if parameter.requires_grad:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator).to(parameter.dtype))


def test_component_config_validation() -> None:
    with pytest.raises(TypeError, match="not a string"):
        AdapterComponentConfig(targets="model.layers.0.q_proj")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown adapter component"):
        AdapterComponentConfig(component="action", targets=["x"])
    with pytest.raises(ValueError, match="unsupported adapter method"):
        AdapterComponentConfig(targets=["x"], method="vera")
    with pytest.raises(ValueError, match="PiSSA requires dropout=0"):
        AdapterComponentConfig(targets=["x"], method="pissa", dropout=0.1)
    with pytest.raises(ValueError, match="PiSSA initialization"):
        AdapterComponentConfig(targets=["x"], method="pissa", initialization="default")
    with pytest.raises(TypeError, match="alpha must be a number"):
        AdapterComponentConfig(targets=["x"], alpha=True)
    with pytest.raises(ValueError, match="PiSSA only"):
        AdapterComponentConfig(targets=["x"], method="dora", initialization="svd")
    with pytest.raises(ValueError, match="language/vision"):
        AdapterComponentConfig(component="projector", targets=["x"], layers=[0])
    with pytest.raises(ValueError, match="rank"):
        AdapterComponentConfig(targets=["x"], rank=0)
    with pytest.raises(ValueError, match="at least one"):
        AdapterConfig()
    pissa = AdapterComponentConfig(targets=["x"], method="pissa", alpha=16)
    assert pissa.initialization is None and pissa.pissa_initialization == "svd"
    assert isinstance(pissa.alpha, float) and pissa.tensor_settings() == {
        "method": "pissa",
        "rank": 8,
        "alpha": 16.0,
        "use_rslora": False,
    }
    pissa.method = "lora"  # e.g. a command-line override of a PiSSA recipe: the unset initialization stays valid
    pissa.validate()
    config = AdapterConfig(components=[{"targets": ["x"], "rank": 4, "alpha": 8.0, "use_rslora": True}])
    assert isinstance(config.components[0], AdapterComponentConfig)
    assert config.components[0].scaling == pytest.approx(4.0)
    assert config.components[0].weight_decay is None


def test_architecture_binding_locates_components() -> None:
    vlm = adapter_architecture("qwen3_vl")
    assert vlm.locate("model.language_model.layers.3.self_attn.q_proj") == ("language", 3)
    assert vlm.locate("model.visual.blocks.26.attn.qkv") == ("vision", 26)
    assert vlm.locate("model.visual.deepstack_merger_list.1.linear_fc1") == ("projector", None)
    assert vlm.locate("lm_head") is None
    with pytest.raises(ValueError, match="no adapter binding"):
        adapter_architecture("qwen3_vl_moe")  # MoE routers and fused experts are not supported targets


def test_resolve_targets_validates_selectors() -> None:
    net = _TinyText()
    manifest = resolve_targets(net, _config(layers=[0, -1]))
    assert manifest["architecture"] == "qwen3"
    assert [row["name"] for row in manifest["targets"]] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.v_proj",
    ]
    assert manifest["trainable_parameters"] == 4 * 4 * (16 + 16)
    with pytest.raises(ValueError, match="matched no"):
        resolve_targets(net, _config(targets=["model.layers.*.mlp.up_proj"]))
    with pytest.raises(ValueError, match="outside language"):
        resolve_targets(net, _config(targets=["lm_head"]))
    with pytest.raises(ValueError, match="only plain nn.Linear"):
        resolve_targets(net, _config(targets=["model.layers.0.norm"]))
    with pytest.raises(ValueError, match="layers matched no modules"):
        resolve_targets(net, _config(layers=[0, 7]))
    with pytest.raises(ValueError, match="invalid for 3 language blocks"):
        resolve_targets(net, _config(layers=[-4]))
    with pytest.raises(ValueError, match="shares its weight"):
        resolve_targets(_TinyText(tied_target=True), _config())
    duplicate = AdapterConfig(
        components=[
            AdapterComponentConfig(targets=["model.layers.0.self_attn.q_proj"]),
            AdapterComponentConfig(targets=["model.layers.*.self_attn.q_proj"], rank=2),
        ]
    )
    with pytest.raises(ValueError, match="more than one component group"):
        resolve_targets(net, duplicate)


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_install_preserves_initial_function(method: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    net = _TinyText().to(dtype)
    x = torch.randn(5, 16, dtype=dtype)
    expected = net(x)
    manifest = install_adapters(net, _config(method, use_rslora=method == "dora"))
    actual = net(x)
    tolerance = 0 if dtype == torch.float32 and method == "lora" else (1e-5 if dtype == torch.float32 else 2e-2)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert set(trainable_spec(manifest)) == {n for n, p in net.named_parameters() if p.requires_grad}
    for module in net.modules():
        if isinstance(module, AdapterLinear):
            assert module.weight.dtype == dtype and not module.weight.requires_grad
            assert module.lora_A.weight.dtype == torch.float32
    actual.float().square().mean().backward()
    for name, parameter in net.named_parameters():
        assert (parameter.grad is not None) == parameter.requires_grad, name


@pytest.mark.parametrize("initialization", ["svd", "svd_lowrank"])
def test_pissa_initialization_extracts_principal_components(initialization: str) -> None:
    torch.manual_seed(0)
    net = _TinyText()
    install_adapters(net, _config("pissa", initialization=initialization, pissa_niter=8))
    layer = net.model.layers[0].self_attn.q_proj
    u, s, vh = torch.linalg.svd(layer.weight.float())
    principal = (u[:, :4] * s[:4]) @ vh[:4]
    tolerance = 1e-5 if initialization == "svd" else 1e-3
    torch.testing.assert_close(
        layer.scaling * layer.lora_B.weight @ layer.lora_A.weight, principal, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(layer.initial_A, layer.lora_A.weight)


@pytest.mark.parametrize(("method", "use_rslora"), METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_weight_matches_adapter_forward(method: str, use_rslora: bool, dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    net = _TinyText().to(dtype)
    manifest = install_adapters(net, _config(method, use_rslora=use_rslora))
    _perturb(net)
    state = net.state_dict()
    assert set(adapter_state_spec(tensor_identity(manifest))) <= set(state)
    x = torch.randn(7, 16)
    for row in manifest["targets"]:
        layer = net.get_submodule(row["name"])
        merged = merged_target_weight(row, state[f"{row['name']}.weight"], state)
        bias = layer.bias.float() if layer.bias is not None else None
        # Reference: the adapter forward with a float32 copy of the (already rounded) base.
        reference = AdapterLinear(
            nn.Linear(16, 16, bias=bias is not None),
            AdapterComponentConfig(targets=["x"], method=method, rank=4, alpha=8.0, use_rslora=use_rslora),
            initialize=False,
        )
        reference.load_state_dict(
            {k.removeprefix(row["name"] + "."): v.float() for k, v in state.items() if k.startswith(row["name"] + ".")}
        )
        torch.testing.assert_close(nn.functional.linear(x, merged, bias), reference(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("method", "use_rslora"), METHODS)
def test_merged_weight_matches_real_bfloat16_forward(method: str, use_rslora: bool) -> None:
    torch.manual_seed(4)
    net = _TinyText().to(torch.bfloat16)
    manifest = install_adapters(net, _config(method, use_rslora=use_rslora))
    _perturb(net)
    state = net.state_dict()
    x = torch.randn(7, 16).to(torch.bfloat16)
    for row in manifest["targets"]:
        layer = net.get_submodule(row["name"])
        merged = merged_target_weight(row, state[f"{row['name']}.weight"], state).to(torch.bfloat16)
        expected = nn.functional.linear(x, merged, layer.bias)
        # The adapter forward rounds once at its output; the merged bf16 weight rounds W + delta first.
        torch.testing.assert_close(layer(x).float(), expected.float(), atol=3e-2, rtol=3e-2)


def test_dora_gradient_matches_reference_with_detached_norm() -> None:
    torch.manual_seed(2)
    base = nn.Linear(12, 10, bias=True)
    layer = AdapterLinear(base, AdapterComponentConfig(targets=["x"], method="dora", rank=3, alpha=6.0))
    _perturb(layer)
    x = torch.randn(5, 12)
    a = layer.lora_A.weight.detach().clone().requires_grad_(True)
    b = layer.lora_B.weight.detach().clone().requires_grad_(True)
    m = layer.lora_magnitude.detach().clone().requires_grad_(True)
    direction = layer.weight.detach() + layer.scaling * (b @ a)
    expected = nn.functional.linear(x, (m / direction.norm(dim=1).detach())[:, None] * direction, layer.bias.detach())
    expected.square().sum().backward()
    layer(x).square().sum().backward()
    for actual, reference in ((layer.lora_A.weight, a), (layer.lora_B.weight, b), (layer.lora_magnitude, m)):
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-5, rtol=1e-5)


def test_dora_dropout_matches_reference_and_eval_skips_it() -> None:
    torch.manual_seed(5)
    layer = AdapterLinear(
        nn.Linear(12, 10, bias=True),
        AdapterComponentConfig(targets=["x"], method="dora", rank=3, alpha=6.0, dropout=0.3),
    )
    _perturb(layer)
    x = torch.randn(4, 12)
    weight, bias, a, b, m = (
        layer.weight.detach(),
        layer.bias.detach(),
        layer.lora_A.weight.detach(),
        layer.lora_B.weight.detach(),
        layer.lora_magnitude.detach(),
    )
    direction = weight + layer.scaling * (b @ a)
    ratio = m / direction.norm(dim=1)
    torch.manual_seed(9)
    dropped = nn.functional.dropout(x, 0.3, training=True)
    # PEFT's DoRA with dropout: base(x) + (ratio - 1) * W x_dropped + ratio * s B A x_dropped.
    expected = (
        nn.functional.linear(x, weight, bias)
        + (ratio - 1) * (dropped @ weight.T)
        + ratio * layer.scaling * (dropped @ a.T @ b.T)
    )
    torch.manual_seed(9)
    torch.testing.assert_close(layer.train()(x), expected, atol=1e-5, rtol=1e-5)
    # In eval, dropout is the identity and the layer equals the merged DoRA weight.
    torch.testing.assert_close(
        layer.eval()(x), nn.functional.linear(x, ratio[:, None] * direction, bias), atol=1e-5, rtol=1e-5
    )


def test_base_fingerprint_ignores_storage_dtype_but_not_values() -> None:
    torch.manual_seed(0)
    reference = _TinyText()
    manifest = install_adapters(reference, _config())
    torch.manual_seed(0)
    same = _TinyText().to(torch.bfloat16)
    install_adapters(same, _config())
    torch.manual_seed(1)
    different = _TinyText()
    install_adapters(different, _config())
    assert model_base_fingerprint(reference, manifest) == model_base_fingerprint(same, same.adapter_manifest)
    assert model_base_fingerprint(reference, manifest) != model_base_fingerprint(different, different.adapter_manifest)


def test_dora_with_dropout_trains_magnitude() -> None:
    torch.manual_seed(2)
    net = _TinyText()
    install_adapters(net, _config("dora", dropout=0.2))
    net.train()
    net(torch.randn(3, 16)).sum().backward()
    assert net.model.layers[0].self_attn.q_proj.lora_magnitude.grad is not None


def test_generic_lora_helpers_leave_dora_and_pissa_alone() -> None:
    for method in ("dora", "pissa"):
        net = _TinyText()
        install_adapters(net, _config(method))
        _perturb(net)
        x = torch.randn(3, 16)
        before = net(x)
        assert merge_lora_adapters_(net) == 0  # the generic plain-LoRA merge does not apply to these modules
        torch.testing.assert_close(net(x), before, atol=0, rtol=0)
        apply_lora_trainable_scope(net, lora_target_modules="q_proj,v_proj")
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        assert trainable == set(trainable_spec(net.adapter_manifest))


def test_trainable_spec_weight_decay_and_identity() -> None:
    config = AdapterConfig(
        components=[
            AdapterComponentConfig(targets=["model.layers.*.self_attn.q_proj"], method="dora", weight_decay=0.1),
            AdapterComponentConfig(targets=["model.layers.*.self_attn.v_proj"], lr_multiplier=0.5, dropout=0.1),
        ]
    )
    manifest = install_adapters(_TinyText(), config)
    spec = trainable_spec(manifest)
    assert spec["model.layers.0.self_attn.q_proj.lora_A.weight"] == (1.0, 0.1)
    assert spec["model.layers.0.self_attn.q_proj.lora_magnitude"] == (1.0, 0.0)
    assert spec["model.layers.0.self_attn.v_proj.lora_B.weight"] == (0.5, None)
    identity = tensor_identity(manifest)
    retrained = AdapterConfig(
        components=[
            AdapterComponentConfig(targets=["model.layers.*.self_attn.q_proj"], method="dora", lr_multiplier=3.0),
            AdapterComponentConfig(targets=["model.layers.*.self_attn.v_proj"], weight_decay=0.2),
        ]
    )
    assert identity_differences(identity, tensor_identity(install_adapters(_TinyText(), retrained))) == []
    rescaled = AdapterConfig(
        components=[
            AdapterComponentConfig(targets=["model.layers.*.self_attn.q_proj"], method="dora", alpha=4.0),
            AdapterComponentConfig(targets=["model.layers.0.self_attn.v_proj"]),
        ]
    )
    differences = identity_differences(identity, tensor_identity(install_adapters(_TinyText(), rescaled)))
    assert "model.layers.0.self_attn.q_proj.alpha: 16.0 (saved) != 4.0 (configured)" in differences
    assert "targets: unexpected model.layers.1.self_attn.v_proj" in differences


def test_trainability_detects_extra_trainable_and_optimizer_mismatch() -> None:
    net = _TinyText()
    manifest = install_adapters(net, _config())
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad])
    assert_trainability(net, manifest, optimizer)
    with pytest.raises(ValueError, match="optimizer parameters differ"):
        assert_trainability(net, manifest, torch.optim.AdamW([next(p for p in net.parameters() if p.requires_grad)]))
    net.lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainability mismatch"):
        assert_trainability(net, manifest)


class _ReadsWeightDirectly(_TinyText):
    """A parent that consumes a target's weight without calling the module (like a fused router)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = self.model.layers[0]
        return nn.functional.linear(x, block.self_attn.q_proj.weight) + block.self_attn.v_proj(x)


def test_require_adapter_calls_rejects_targets_bypassed_by_the_forward() -> None:
    net = _TinyText()
    install_adapters(net, _config())
    require_adapter_calls(net)
    net(torch.randn(2, 16))
    net(torch.randn(2, 16))  # the check is one-shot and removes its hooks
    bypassed = _ReadsWeightDirectly()
    install_adapters(bypassed, _config(layers=[0]))
    require_adapter_calls(bypassed)
    with pytest.raises(RuntimeError, match=r"not called .*q_proj"):
        bypassed(torch.randn(2, 16))


def test_require_adapter_calls_under_activation_checkpointing() -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    for model, expect_error in ((_TinyText(), False), (_ReadsWeightDirectly(), True)):
        install_adapters(model, _config(layers=[0]))
        for index, block in enumerate(model.model.layers):
            model.model.layers[index] = checkpoint_wrapper(block)
        require_adapter_calls(model)
        x = torch.randn(2, 16, requires_grad=True)
        if expect_error:
            with pytest.raises(RuntimeError, match="not called"):
                model(x)
        else:
            model(x).sum().backward()


def test_install_requires_materialized_base() -> None:
    with torch.device("meta"):
        net = _TinyText()
    with pytest.raises(ValueError, match="materializing"):
        install_adapters(net, _config())
