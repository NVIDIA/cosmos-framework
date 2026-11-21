# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small real-tensor FP32-head contracts; no NF4 or task-quality qualification."""

import copy
import json
import shutil
import subprocess
import sys

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.dense_adapter import _digest_json
from cosmos_framework.posttrain.checkpoint.family_dense import (
    FAMILY_UNMERGED_MANIFEST,
    export_family_dense,
    export_family_unmerged,
    load_family_unmerged,
    read_family_unmerged,
)
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone
from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
from cosmos_framework.posttrain.tests.test_checkpoint_family_dense import BUILD
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
LANGUAGE = {"backbone.0": (4, 3), "backbone.2": (4, 4)}
HEADS = {"action_head.q": (4, 4), "action_head.out": (2, 4)}


class HeadTaskNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(39)
            self.backbone = nn.Sequential(
                nn.Linear(3, 4, bias=False, device=device, dtype=torch.bfloat16),
                nn.Tanh(),
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
            )
            self.action_head = nn.Module()
            self.action_head.q = nn.Linear(4, 4, device=device, dtype=torch.float32)
            self.action_head.out = nn.Linear(4, 2, device=device, dtype=torch.float32)
            # A custom category-shaped parameter remains completely outside the allowlist.
            self.action_head.category = nn.Parameter(torch.randn(2, 3, 4, device=device, dtype=torch.float32))
            self.embedding = nn.Embedding(5, 3, device=device, dtype=torch.bfloat16)
            self.output = nn.Linear(3, 5, bias=False, device=device, dtype=torch.bfloat16)
            self.output.weight = self.embedding.weight
        self.register_buffer("scale", torch.tensor([0.2, 0.3], dtype=torch.float32), persistent=False)

    def forward(self, inputs):
        x = self.backbone(inputs.to(torch.bfloat16))
        return self.action_head.out(torch.tanh(self.action_head.q(x.float()))) * self.scale


def install_head(net, *, enabled=True, targets=HEADS, **kwargs):
    return install_family_backbone(
        net,
        config=kwargs.pop("config", FamilyBackboneConfig(mode="bf16_lora", rank=32, alpha=64, seed=17)),
        targets=LANGUAGE,
        auxiliary_parameters=kwargs.pop("auxiliary_parameters", ()),
        head_targets=targets if enabled else None,
        source_identity={"repository": "fixture/native-head", "revision": "a" * 40},
        task_identity={"family": "fixture", "head_profile": enabled},
        device=torch.device("cpu"),
        **kwargs,
    )


def batch():
    return torch.tensor([[0.1, 0.2, -0.3], [-0.3, 0.1, 0.2]]), torch.tensor([[0.01, -0.02], [-0.02, 0.01]])


def update(net, state, optimizer):
    state.assert_optimizer(optimizer)
    optimizer.zero_grad(set_to_none=True)
    inputs, target = batch()
    loss = (net(inputs) - target).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state.codec.parameters.values())
    optimizer.step()
    state.validate_frozen_base()
    return loss.detach()


def prepared(tmp_path):
    net = HeadTaskNet()
    state = install_head(net)
    state.codec.bind_processor_identity({"fixture": "two_axes"})
    state.codec.bind_training_identity({"steps": 4, "seed": 17})
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=0.01)
    for _ in range(3):
        update(net, state, optimizer)
    source = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000003"),
        state.codec.parameters,
        state.codec.metadata(optimizer),
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    return net, state, optimizer, source


def exported(tmp_path):
    net, state, optimizer, source = prepared(tmp_path)
    output = tmp_path / "unmerged"
    manifest = export_family_unmerged(
        net, checkpoint=source, output=str(output), exporter_build=BUILD, max_shard_size_bytes=1024
    )
    return net, state, optimizer, source, output, manifest


def test_meta_stream_preserves_language_initialization_fp32_head_global_rng_modes_and_initial_arithmetic():
    source = HeadTaskNet()
    original = {name: value.detach().clone() for name, value in source.named_parameters()}
    plain = HeadTaskNet()
    plain_state = install_head(plain, enabled=False)
    streamed = HeadTaskNet(device="meta")
    streamed.action_head.q.eval()
    reads = []

    def read(name):
        reads.append(name)
        return original[name]

    rng = torch.get_rng_state().clone()
    state = install_head(streamed, read_tensor=read)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(reads) == len(set(reads)) == len(original)
    assert not streamed.action_head.q.training and streamed.action_head.out.training
    assert streamed.embedding.weight is streamed.output.weight
    assert "head_adapters" not in plain_state.identity
    assert state.identity["targets"] == plain_state.identity["targets"]
    assert state.identity["auxiliary_parameters"] == []
    assert state.identity["head_adapters"]["targets"] == {name: list(shape) for name, shape in HEADS.items()}
    for name in plain_state.codec.parameters:
        torch.testing.assert_close(state.codec.parameters[name], plain_state.codec.parameters[name], atol=0, rtol=0)
    for name, value in original.items():
        torch.testing.assert_close(streamed.get_parameter(name), value, atol=0, rtol=0)
        assert not streamed.get_parameter(name).requires_grad
    for net in (source, plain, streamed):
        inputs = batch()[0].clone().requires_grad_()
        output = net(inputs)
        (gradient,) = torch.autograd.grad(output.sum(), inputs)
        if net is source:
            expected_output, expected_gradient = output.detach(), gradient
        else:
            torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
            torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
    state.validate_frozen_base()


def test_only_complete_fp32_language_and_head_factors_update_and_portable_next_update_matches(tmp_path):
    net, state, optimizer, source = prepared(tmp_path)
    expected = {f"{name}.lora_{factor}.weight" for name in (*LANGUAGE, *HEADS) for factor in ("A", "B")}
    assert set(state.codec.parameters) == expected
    assert {name for name, p in net.named_parameters() if p.requires_grad} == expected
    assert all(p.dtype == torch.float32 for p in state.codec.parameters.values())
    assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in optimizer.state.values())
    assert all(torch.count_nonzero(net.get_parameter(f"{name}.lora_B.weight")) for name in HEADS)
    assert all(net.get_parameter(f"{name}.{leaf}").grad is None for name in HEADS for leaf in ("weight", "bias"))
    fresh = HeadTaskNet()
    restored = install_head(fresh)
    restored.codec.bind_processor_identity(state.codec.identity["processor_contract"])
    restored.codec.bind_training_identity(state.codec.identity["training_contract"])
    load_adapter_weights(source, restored.codec, manager=CheckpointManager(checkpoint_format="dcp"))
    other_optimizer = torch.optim.AdamW(restored.codec.parameters.values(), lr=0.01)
    # Portable artifacts contain no optimizer: supply it explicitly for this next-update check.
    other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored.codec.validate_metadata(state.codec.metadata(optimizer), other_optimizer, purpose="resume")
    torch.testing.assert_close(update(net, state, optimizer), update(fresh, restored, other_optimizer), atol=0, rtol=0)
    for name, value in state.codec.parameters.items():
        torch.testing.assert_close(restored.codec.parameters[name], value, atol=0, rtol=0)
    with torch.no_grad():
        fresh.action_head.q.bias.add_(1)
    with pytest.raises(RuntimeError, match="frozen family tensor"):
        restored.validate_frozen_base()


@pytest.mark.parametrize("autocast", [False, True])
def test_unmerged_restores_every_factor_fp32_bias_frozen_custom_tensor_alias_and_output(tmp_path, autocast):
    net, state, _, _, output, manifest = exported(tmp_path)
    fresh = HeadTaskNet(device="meta")
    rng = torch.get_rng_state().clone()
    load_family_unmerged(fresh, output, device="cpu", expected_identity=state.codec.identity)
    assert torch.equal(rng, torch.get_rng_state())
    assert fresh.embedding.weight is fresh.output.weight
    assert all(not p.requires_grad and not p.is_meta for p in fresh.parameters())
    assert {name: tensor_digest(p) for name, p in fresh.named_parameters()} == {
        name: tensor_digest(p) for name, p in net.named_parameters()
    }
    for name in HEADS:
        module = fresh.get_submodule(name)
        assert type(module) is LoraInjectedLinear
        assert module.weight.dtype == module.bias.dtype == torch.float32
        assert module._adapter_compute_dtype is None
    assert manifest["numerical_contract"]["head_adapters"]["base_dtype"] == "float32"
    assert set(manifest["trained_tensors"]) == set(state.codec.parameters)
    net.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        torch.testing.assert_close(fresh(batch()[0]), net(batch()[0]), atol=0, rtol=0)


def test_fresh_process_head_reload_has_no_original_weights_quantizer_or_initialization(tmp_path):
    net, _, _, source, output, _ = exported(tmp_path)
    torch.save({"inputs": batch()[0], "output": net(batch()[0]).detach()}, tmp_path / "expected.pt")
    shutil.rmtree(source)
    script = """
import sys, torch
from cosmos_framework.posttrain.tests.test_family_head_adapters import HeadTaskNet
from cosmos_framework.posttrain.checkpoint import family_dense
from cosmos_framework.utils.generator.lora import NativeLoRAMixin
def forbidden(*args, **kwargs): raise AssertionError('original source/adapter initialization used')
family_dense.load_adapter_weights = forbidden
family_dense._effective_weight = forbidden
NativeLoRAMixin.reset_adapter_parameters = forbidden
sys.modules['cosmos_framework.posttrain.model.quantization.bnb'] = None
sys.modules['bitsandbytes'] = None
net = HeadTaskNet(device='meta')
rng = torch.get_rng_state().clone()
family_dense.load_family_unmerged(net,sys.argv[1],device='cpu',expected_identity={'format':'cosmos.family-backbone.v1'})
assert torch.equal(rng,torch.get_rng_state())
fixture=torch.load(sys.argv[2],weights_only=True)
torch.testing.assert_close(net(fixture['inputs']),fixture['output'],atol=0,rtol=0)
assert net.embedding.weight is net.output.weight
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), str(tmp_path / "expected.pt")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_head_mode_merged_export_refuses_before_creating_output(tmp_path, dtype):
    net, _, _, source = prepared(tmp_path)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="unmerged deployment only"):
        export_family_dense(
            net, checkpoint=source, output=str(output), exporter_build=BUILD, merged_backbone_dtype=dtype
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "fault", ["biasless", "bf16", "subclass", "tied_bias", "module_alias", "shape", "auxiliary", "rank"]
)
def test_invalid_head_target_refused_before_any_model_mutation(fault):
    net = HeadTaskNet()
    targets = dict(HEADS)
    kwargs = {}
    if fault == "biasless":
        net.action_head.q.bias = None
    elif fault == "bf16":
        net.action_head.q.to(torch.bfloat16)
    elif fault == "subclass":

        class DifferentLinear(nn.Linear):
            pass

        net.action_head.q = DifferentLinear(4, 4)
    elif fault == "tied_bias":
        net.tied = net.action_head.q.bias
    elif fault == "module_alias":
        net.alias = net.action_head.q
    elif fault == "shape":
        targets["action_head.q"] = (3, 4)
    elif fault == "auxiliary":
        kwargs["auxiliary_parameters"] = ("action_head.q.weight",)
    else:
        kwargs["config"] = FamilyBackboneConfig(mode="bf16_lora", rank=16, alpha=64)
    original = [(name, id(p), p.requires_grad, tensor_digest(p)) for name, p in net.named_parameters()]
    with pytest.raises(ValueError):
        install_head(net, targets=targets, **kwargs)
    assert original == [(name, id(p), p.requires_grad, tensor_digest(p)) for name, p in net.named_parameters()]


@pytest.mark.parametrize(
    "fault",
    [
        "rank",
        "base_dtype",
        "bias",
        "compute",
        "missing_bias",
        "missing_factor",
        "factor_dtype",
        "factor_value",
        "extra_target",
    ],
)
def test_unmerged_rejects_wrong_head_policy_and_incomplete_frozen_or_trained_state(tmp_path, fault):
    _, _, _, _, output, manifest = exported(tmp_path)
    identity = manifest["source_metadata"]["identity"]
    head = identity["head_adapters"]
    if fault in ("rank", "base_dtype", "bias", "compute"):
        head[fault] = 16 if fault == "rank" else "wrong"
    elif fault == "missing_bias":
        del identity["frozen_tensors"]["action_head.q.bias"]
    elif fault == "extra_target":
        head["targets"]["action_head.category"] = [3, 4]
    elif fault == "missing_factor":
        del manifest["trained_tensors"]["action_head.q.lora_A.weight"]
    elif fault == "factor_dtype":
        manifest["tensors"]["action_head.q.lora_A.weight"]["dtype"] = "bfloat16"
    else:
        manifest["tensors"]["action_head.q.lora_B.weight"]["sha256"] = "0" * 64
    manifest["source_metadata_sha256"] = _digest_json(manifest["source_metadata"])
    (output / FAMILY_UNMERGED_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_family_unmerged(output)
