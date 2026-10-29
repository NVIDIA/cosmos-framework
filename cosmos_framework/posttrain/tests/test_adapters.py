# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small correctness tests; GPU/model qualification is a separate gate."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapters import (
    FORMAT,
    prepare_merged_state,
    read_adapter_artifact,
    snapshot_identity,
    state_spec,
    write_adapter_artifact,
)
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.model.adapters import (
    PosttrainAdapterLinear,
    assert_trainability,
    install_adapters,
    optimizer_groups,
    resolve_targets,
)


def tiny_net() -> nn.Module:
    net = nn.Module()
    net.model = nn.Module()
    net.model.language_model = nn.Module()
    net.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for block in net.model.language_model.layers:
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Linear(8, 8)
        block.self_attn.v_proj = nn.Linear(8, 8)
    net.model.language_model.norm = nn.LayerNorm(8)
    net.model.visual = nn.Module()
    net.model.visual.merger = nn.Module()
    net.model.visual.merger.linear_fc1 = nn.Linear(8, 8)
    net.model.visual.blocks = nn.ModuleList([nn.Module()])
    net.model.visual.blocks[0].attn = nn.Module()
    net.model.visual.blocks[0].attn.qkv = nn.Linear(8, 24)
    return net


LANG = "model.language_model.layers.*.self_attn.q_proj"
FIRST = "model.language_model.layers.0.self_attn.q_proj"


def identity(manifest: dict) -> dict:
    return dict(
        format=FORMAT,
        architecture="qwen3_vl",
        base_storage_dtype="float32",
        dependencies={},
        base={"sha256": "fixture"},
        processor={"sha256": "fixture"},
        resolved=manifest,
        media={},
        pissa_representation="original_base_plus_trained_minus_initial_factors",
    )


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_method_initialization_updates_and_nonmutating_merge(method, rslora):
    torch.manual_seed(13)
    base = nn.Linear(8, 12)
    original = copy.deepcopy(base)
    cfg = Component(
        targets=[FIRST],
        method=method,
        use_rslora=rslora,
        rank=3,
        alpha=7,
        initialization="svd" if method == "pissa" else "default",
    )
    layer = PosttrainAdapterLinear(base, cfg)
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)
    x = torch.randn(2, 7, 8)
    torch.testing.assert_close(layer(x), original(x), atol=2e-6, rtol=2e-6)
    assert layer._lora_scale == pytest.approx(7 / (3**0.5 if rslora else 3))
    frozen = layer.weight.detach().clone()
    before = {n: p.detach().clone() for n, p in layer.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = (layer(x) - 0.3).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)
        if method != "pissa" and step == 0:
            assert layer.lora_A.weight.grad.count_nonzero() == 0
        opt.step()
    assert all(not torch.equal(before[n], p) for n, p in layer.named_parameters() if p.requires_grad)
    assert torch.equal(frozen, layer.weight)
    layer.eval()
    state = {"layer." + n: t.clone() for n, t in layer.state_dict().items()}
    merged = layer.merged_state_weight(state, "layer")
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, merged, layer.bias), rtol=3e-6, atol=3e-6)
    assert all(torch.equal(t, state["layer." + n]) for n, t in layer.state_dict().items())
    with pytest.raises(RuntimeError, match="non-mutating"):
        layer.merge_adapter_()


def test_independent_components_and_optimizer_membership():
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    net = tiny_net()
    cfg = AdapterConfig(
        components=[
            Component(targets=[LANG], rank=2, dropout=0.2, layers=[1]),
            Component(
                component="projector",
                targets=["model.visual.merger.linear_fc1"],
                rank=3,
                method="dora",
                lr_multiplier=0.3,
            ),
            Component(
                component="vision", targets=["model.visual.blocks.*.attn.qkv"], layers=[0], rank=4, method="pissa"
            ),
        ],
        extra_modules=[AdapterExtraModuleConfig(targets=["model.language_model.norm"], weight_decay=0.01)],
    )
    manifest = install_adapters(net, cfg)
    assert {n for n, m in net.named_modules() if isinstance(m, PosttrainAdapterLinear)} == {
        r["name"] for r in manifest["targets"]
    }
    assert net.get_submodule("model.language_model.layers.1.self_attn.q_proj").lora_dropout.p == 0.2
    optimizer = build_optimizer(net, OptimizerConfig(lr=0.01), parameter_groups=optimizer_groups(net, 0.01))
    assert_trainability(net, optimizer)
    assert sorted({g["lr"] for g in optimizer.param_groups}) == [0.003, 0.01]
    assert sorted({g["weight_decay"] for g in optimizer.param_groups}) == [0, 0.01]
    assert (
        sum(p.numel() for p in net.parameters() if p.requires_grad)
        == sum(r["trainable_count"] for r in manifest["targets"]) + 16
    )
    optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
    with pytest.raises(ValueError, match="duplicates"):
        assert_trainability(net, optimizer)


def test_merge_probability_gate_reports_near_ties_and_rejects_changed_confident_decisions():
    from cosmos_framework.posttrain.tests.adapter_numerics import assert_bf16_deployment_drift, compare_logits

    reference = [torch.tensor([[[10.0, 10.005, -30.0]]])]
    actual = [reference[0][..., [1, 0, 2]]]
    measured = compare_logits(actual, reference)
    assert measured["argmax_equal"] == 0 and measured["positions"] == 1
    assert 0 < measured["maximum_argmax_probability_regret"] < 0.02
    assert_bf16_deployment_drift(measured)
    confident = [torch.tensor([[[20.0, 0.0, -30.0]]])]
    with pytest.raises(AssertionError, match="drift"):
        assert_bf16_deployment_drift(compare_logits([confident[0][..., [1, 0, 2]]], confident))
    with pytest.raises(ValueError, match="finite"):
        compare_logits([torch.full_like(confident[0], float("nan"))], confident)


@pytest.mark.parametrize(
    "components,match",
    [
        ([Component(targets=[LANG, "model.language_model.layers.*.typo"])], "matched no"),
        ([Component(targets=["model.visual.blocks.*.attn.qkv"])], "outside language"),
        ([Component(targets=[LANG], rank=2), Component(targets=[LANG], rank=3)], "conflicting"),
        ([Component(targets=["model.language_model.norm"])], "plain nn.Linear"),
        ([Component(targets=["model.language_model.norm"], layers=[0])], "non-layer modules"),
        ([Component(targets=[LANG], layers=[0, 99])], "indices matched no"),
    ],
)
def test_resolution_refuses_before_mutation(components, match):
    net = tiny_net()
    ids = {n: id(p) for n, p in net.named_parameters()}
    with pytest.raises(ValueError, match=match):
        install_adapters(net, AdapterConfig(components=components))
    assert ids == {n: id(p) for n, p in net.named_parameters()}
    assert not any(isinstance(m, PosttrainAdapterLinear) for m in net.modules())


@pytest.mark.parametrize("same_module", [False, True])
def test_shared_parameters_and_module_aliases(same_module, tmp_path):
    net = tiny_net()
    attention = net.model.language_model.layers[0].self_attn
    if same_module:
        attention.v_proj = attention.q_proj
    else:
        attention.v_proj.weight = attention.q_proj.weight
    old_weight = attention.q_proj.weight
    with pytest.raises(ValueError, match="unselected"):
        resolve_targets(net, AdapterConfig(components=[Component(targets=[FIRST])]))
    second = FIRST.replace("q_proj", "v_proj")
    with pytest.raises(ValueError, match="incompatible"):
        resolve_targets(
            net, AdapterConfig(components=[Component(targets=[FIRST], rank=2), Component(targets=[second], rank=3)])
        )
    manifest = install_adapters(
        net, AdapterConfig(components=[Component(targets=[FIRST, second], rank=2, method="dora")])
    )
    assert attention.q_proj.weight is attention.v_proj.weight is old_weight
    assert attention.q_proj.lora_A is attention.v_proj.lora_A
    assert attention.q_proj.magnitude is attention.v_proj.magnitude
    merged = prepare_merged_state(net.state_dict(), identity(manifest))
    assert merged[FIRST + ".weight"] is merged[second + ".weight"]
    from cosmos_framework.posttrain.checkpoint.adapters import load_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    ident = identity(manifest)
    portable = tmp_path / "portable"
    write_adapter_artifact(str(portable), ident, {n: net.state_dict()[n] for n in state_spec(ident)})
    restored = tiny_net()
    other = restored.model.language_model.layers[0].self_attn
    if same_module:
        other.v_proj = other.q_proj
    else:
        other.v_proj.weight = other.q_proj.weight
    install_adapters(restored, AdapterConfig(components=[Component(targets=[FIRST, second], rank=2, method="dora")]))
    restored._posttrain_adapter_identity = ident
    load_adapter_artifact(restored, str(portable))
    assert other.q_proj.weight is other.v_proj.weight
    assert other.q_proj.lora_A is other.v_proj.lora_A
    assert other.q_proj.magnitude is other.v_proj.magnitude
    torch.testing.assert_close(other.q_proj.lora_A.weight, attention.q_proj.lora_A.weight, atol=0, rtol=0)
    with pytest.raises(ValueError, match="cannot reconstruct custom adapter weight aliases"):
        export_merged_adapter(str(portable), str(tmp_path / "base"), str(tmp_path / "merged"))


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_portable_fresh_process_reload_and_errors(tmp_path, method):
    torch.manual_seed(6)
    net = tiny_net()
    base = copy.deepcopy(net.state_dict())
    cfg = AdapterConfig(components=[Component(targets=[LANG], rank=2, method=method)])
    manifest = install_adapters(net, cfg)
    ident = identity(manifest)
    net._posttrain_adapter_identity = ident
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.01)
    state = net.state_dict()
    path = tmp_path / "portable"
    write_adapter_artifact(str(path), ident, {n: state[n] for n in state_spec(ident)})
    torch.save({"base": base, "expected": state, "identity": ident, "config": cfg}, tmp_path / "fixture.pt")
    script = """
import sys, torch
from cosmos_framework.posttrain.tests.test_adapters import tiny_net
from cosmos_framework.posttrain.model.adapters import install_adapters
from cosmos_framework.posttrain.checkpoint.adapters import load_adapter_artifact, prepare_merged_state
f = torch.load(sys.argv[1], weights_only=False)
net = tiny_net(); net.load_state_dict(f['base'])
# Exact restore/warm start must never run decomposition.
torch.svd_lowrank = lambda *a, **k: (_ for _ in ()).throw(AssertionError('repeated SVD'))
install_adapters(net, f['config'], initialize=False)
net._posttrain_adapter_identity = f['identity']
load_adapter_artifact(net, sys.argv[2])
assert all(torch.equal(v, f['expected'][n]) for n,v in net.state_dict().items())
merged = tiny_net(); merged.load_state_dict(prepare_merged_state(net.state_dict(), f['identity']))
x = torch.randn(4,8)
for name in ['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.1.self_attn.q_proj']:
    torch.testing.assert_close(net.get_submodule(name)(x), merged.get_submodule(name)(x), atol=2e-6, rtol=2e-6)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "fixture.pt"), str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    wrong = copy.deepcopy(ident)
    wrong["base"] = {"sha256": "wrong"}
    with pytest.raises(ValueError, match="base"):
        read_adapter_artifact(str(path), expected_identity=wrong)
    with pytest.raises(ValueError, match="incomplete/extra"):
        write_adapter_artifact(str(tmp_path / "missing"), ident, {})
    (path / "adapter.safetensors").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="digest"):
        read_adapter_artifact(str(path))


def test_base_identity_hashes_weights_not_only_sizes(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (tmp_path / "model.safetensors").write_bytes(b"original")
    before = snapshot_identity(str(tmp_path), weights=True)
    (tmp_path / "model.safetensors").write_bytes(b"modified")
    assert before != snapshot_identity(str(tmp_path), weights=True)


@pytest.mark.parametrize(
    "kwargs", [{"method": "pissa", "dropout": 0.1}, {"method": "dora", "initialization": "svd"}, {"dtype": "bfloat16"}]
)
def test_unsupported_combinations_fail(kwargs):
    with pytest.raises(ValueError):
        Component(targets=[FIRST], **kwargs)


@pytest.mark.parametrize("interrupt_commit", [False, True])
def test_portable_commit_last_when_noreplace_rename_is_unsupported(tmp_path, monkeypatch, interrupt_commit):
    import errno

    from cosmos_framework.posttrain.checkpoint import adapters as artifacts

    net = tiny_net()
    ident = identity(install_adapters(net, AdapterConfig(components=[Component(targets=[FIRST], rank=2)])))
    state = {n: net.state_dict()[n] for n in state_spec(ident)}

    def unsupported(*_):
        raise OSError(errno.EOPNOTSUPP, "fixture: filesystem lacks renameat2")

    monkeypatch.setattr(artifacts, "rename_directory_noreplace", unsupported)
    target = tmp_path / "portable"
    rename = artifacts.os.rename

    def checked_move(source, destination):
        if str(destination).endswith(artifacts.MANIFEST):
            with pytest.raises(ValueError, match="incomplete"):
                read_adapter_artifact(str(target))
            if interrupt_commit:
                raise OSError(errno.EIO, "fixture: interrupted before manifest commit")
        return rename(source, destination)

    monkeypatch.setattr(artifacts.os, "rename", checked_move)
    if interrupt_commit:
        with pytest.raises(OSError, match="interrupted"):
            write_adapter_artifact(str(target), ident, state)
        with pytest.raises(ValueError, match="incomplete"):
            read_adapter_artifact(str(target))
    else:
        write_adapter_artifact(str(target), ident, state)
        _, actual = read_adapter_artifact(str(target), expected_identity=ident)
        assert all(torch.equal(actual[n], state[n]) for n in state)
    # Successful and interrupted reservations both remain immutable to retries.
    with pytest.raises(FileExistsError):
        write_adapter_artifact(str(target), ident, state)


def test_portable_fallback_competing_writers_do_not_replace(tmp_path, monkeypatch):
    import errno
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cosmos_framework.posttrain.checkpoint import adapters as artifacts

    net = tiny_net()
    ident = identity(install_adapters(net, AdapterConfig(components=[Component(targets=[FIRST], rank=2)])))
    state = {n: net.state_dict()[n] for n in state_spec(ident)}
    barrier = threading.Barrier(2)

    def unsupported(*_):
        barrier.wait(timeout=10)
        raise OSError(errno.EOPNOTSUPP, "fixture: filesystem lacks renameat2")

    monkeypatch.setattr(artifacts, "rename_directory_noreplace", unsupported)
    target = str(tmp_path / "portable")

    def publish():
        try:
            write_adapter_artifact(target, ident, state)
            return "committed"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: publish(), range(2))) == ["committed", "exists"]
    _, actual = read_adapter_artifact(target, expected_identity=ident)
    assert all(torch.equal(actual[n], state[n]) for n in state)


def test_profile_rejects_unqualified_attention():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.adapters import validate_adapter_profile

    with pytest.raises(ValueError, match="attn_impl=eager"):
        validate_adapter_profile(
            Qwen3VLModelConfig(attn_impl="flash_attention_2"),
            SimpleNamespace(model_type="qwen3_vl"),
            None,
            AdapterConfig(components=[Component(targets=[FIRST])]),
            False,
        )


def test_dora_forward_and_gradients_match_peft_reference():
    pytest.importorskip("peft")
    from peft.tuners.lora.dora import DoraLinearLayer

    torch.manual_seed(3)
    base = nn.Linear(8, 12)
    base.requires_grad_(False)
    layer = PosttrainAdapterLinear(base, Component(targets=[FIRST], method="dora", rank=3, dropout=0.2))
    with torch.no_grad():
        layer.lora_B.weight.normal_(std=0.1)
        layer.magnitude.mul_(1.1)
    a, b = copy.deepcopy(layer.lora_A), copy.deepcopy(layer.lora_B)
    reference = DoraLinearLayer(fan_in_fan_out=False)
    reference.weight = nn.Parameter(layer.magnitude.detach().clone())
    x = torch.randn(3, 8, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    torch.manual_seed(7)
    actual = layer(x)
    torch.manual_seed(7)
    dropped = torch.nn.functional.dropout(other, p=0.2, training=True)
    expected = base(other) + reference(dropped, lora_A=a, lora_B=b, scaling=layer._lora_scale, base_layer=base)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for actual_grad, expected_grad in [
        (x.grad, other.grad),
        (layer.lora_A.weight.grad, a.weight.grad),
        (layer.lora_B.weight.grad, b.weight.grad),
        (layer.magnitude.grad, reference.weight.grad),
    ]:
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)


def test_pissa_exact_principal_components_and_residual():
    base = nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.diag(torch.tensor([8.0, 6.0, 4.0, 3.0, 2.0, 1.0])))
    layer = PosttrainAdapterLinear(
        base, Component(targets=[FIRST], rank=2, alpha=7, method="pissa", initialization="svd")
    )
    principal = layer._lora_scale * layer.initial_B @ layer.initial_A
    torch.testing.assert_close(
        principal, torch.diag(torch.tensor([8.0, 6.0, 0.0, 0.0, 0.0, 0.0])), atol=1e-6, rtol=1e-6
    )
    residual = base.weight - principal
    torch.testing.assert_close(residual.square().sum(), torch.tensor(30.0), atol=1e-6, rtol=1e-6)
