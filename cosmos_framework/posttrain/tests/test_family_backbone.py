# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU tensor/gradient/artifact contracts; not NF4 accelerator qualification."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import (
    load_adapter_weights,
    read_adapter_metadata,
    save_adapter_artifact,
)
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.quantization.family import (
    FamilyBackboneConfig,
    install_family_backbone,
    resolve_family_targets,
)
from cosmos_framework.utils.generator.lora import NativeLoRAMixin

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
TARGETS = {"backbone.0": (4, 4), "backbone.2": (4, 4)}
AUXILIARY = ["projector.weight", "projector.bias", "head.weight", "head.bias"]
SOURCE = {"repository": "fixture/native-task", "revision": "a" * 40}
TASK = {"objective": "fixture_action_mse", "action_horizon": 2, "units": "normalized", "embodiment": "fixture"}


class TaskPolicy(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314)
            self.vision = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
            self.projector = nn.Linear(4, 4, device=device, dtype=torch.bfloat16)
            self.backbone = nn.Sequential(
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
                nn.Tanh(),
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
            )
            self.head = nn.Linear(4, 2, device=device, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.tensor(0.5, dtype=torch.float32), persistent=False)

    def forward(self, images):
        visual = self.vision(images)
        projected = nn.functional.linear(visual.float(), self.projector.weight.float(), self.projector.bias.float())
        hidden = self.backbone(projected.to(torch.bfloat16))
        return nn.functional.linear(hidden.float(), self.head.weight.float(), self.head.bias.float()) * self.scale


def install(net, *, mode="bf16_lora", targets=TARGETS, auxiliary=AUXILIARY, **kwargs):
    return install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode=mode, rank=2, alpha=4.0, seed=17, dropout=kwargs.pop("dropout", 0.0)),
        targets=targets,
        auxiliary_parameters=auxiliary,
        source_identity=kwargs.pop("source_identity", SOURCE),
        task_identity=kwargs.pop("task_identity", TASK),
        device=torch.device("cpu"),
        **kwargs,
    )


def batch():
    generator = torch.Generator().manual_seed(91)
    return torch.randn(6, 4, dtype=torch.bfloat16, generator=generator), torch.randn(6, 2, generator=generator)


def update(net, state, optimizer):
    state.assert_optimizer(optimizer)
    images, actions = batch()
    optimizer.zero_grad(set_to_none=True)
    loss = (net(images) - actions).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state.codec.parameters.values())
    optimizer.step()
    state.validate_frozen_base()
    return loss.detach()


@pytest.mark.parametrize("mode", ["bf16_lora", "bf16_frozen"])
def test_actual_task_loss_updates_exact_adapters_and_auxiliaries_but_never_frozen_base(mode):
    net = TaskPolicy()
    images, _ = batch()
    initial_outputs = net(images).detach()
    initial = {name: p.detach().clone() for name, p in net.named_parameters()}
    rng = torch.get_rng_state().clone()
    state = install(net, mode=mode)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(net(images), initial_outputs, atol=0, rtol=0)
    adapter_names = (
        {f"{name}.lora_{factor}.weight" for name in TARGETS for factor in ("A", "B")} if mode == "bf16_lora" else set()
    )
    expected = adapter_names | set(AUXILIARY)
    assert set(state.codec.parameters) == expected
    assert {name for name, p in net.named_parameters() if p.requires_grad} == expected
    assert all(p.dtype == torch.float32 for p in state.codec.parameters.values())
    assert bool(adapter_names) == any(isinstance(module, NativeLoRAMixin) for module in net.modules())
    before = {name: p.detach().clone() for name, p in state.codec.parameters.items()}
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=0.02, weight_decay=0.0)
    update(net, state, optimizer)
    if adapter_names:
        # Zero B legitimately gives A zero task gradients on the first update.
        assert all(torch.count_nonzero(net.get_parameter(f"{name}.lora_A.weight").grad) == 0 for name in TARGETS)
        assert all(torch.count_nonzero(net.get_parameter(f"{name}.lora_B.weight").grad) > 0 for name in TARGETS)
    for _ in range(3):
        update(net, state, optimizer)
    assert all(not torch.equal(before[name], parameter) for name, parameter in state.codec.parameters.items())
    for name, original in initial.items():
        if name not in AUXILIARY:
            torch.testing.assert_close(net.get_parameter(name), original, atol=0, rtol=0)
            assert net.get_parameter(name).grad is None
    assert all(value["exp_avg"].dtype == torch.float32 for value in optimizer.state.values())


def test_meta_stream_reads_each_source_once_and_preserves_rng_initialized_buffers_and_mixed_modes():
    source = TaskPolicy()
    values = {name: p.detach().clone() for name, p in source.named_parameters()}
    streamed = TaskPolicy(device="meta")
    streamed.backbone[0].eval()
    modes = {name: module.training for name, module in streamed.named_modules()}
    reads = []

    def read(name):
        reads.append(name)
        return values[name]

    rng = torch.get_rng_state().clone()
    state = install(streamed, read_tensor=read, dropout=0.1)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(reads) == len(set(reads)) == len(values) and set(reads) == set(values)
    assert not any(p.is_meta for p in streamed.parameters())
    assert streamed.scale.item() == 0.5 and "scale" in state.identity["base_buffers"]
    assert all(streamed.get_submodule(name).training == mode for name, mode in modes.items())
    assert not streamed.backbone[0].lora_dropout.training
    assert streamed.backbone[2].lora_dropout.training
    state.validate_frozen_base()


def test_adapter_dtype_is_explicit_under_nonstandard_default_dtype():
    net = TaskPolicy()
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        state = install(net)
    finally:
        torch.set_default_dtype(old)
    assert all(p.dtype == torch.float32 for p in state.codec.parameters.values())


@pytest.mark.parametrize("mode", ["bf16_lora", "bf16_frozen"])
def test_portable_roundtrip_retains_trained_heads_and_exact_next_update(tmp_path, mode):
    net = TaskPolicy()
    state = install(net, mode=mode)
    state.codec.bind_processor_identity({"images": "fixture_4vector", "statistics_sha256": "c" * 64})
    state.codec.bind_training_identity({"optimizer": "AdamW", "objective": TASK, "seed": 91})
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=0.02, weight_decay=0.0)
    for _ in range(3):
        update(net, state, optimizer)
    metadata = state.codec.metadata(optimizer)
    checkpoint = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000003"),
        state.codec.parameters,
        metadata,
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    assert read_adapter_metadata(checkpoint) == {**metadata, "optimizer": None}
    assert set(metadata["parameters"]) == set(state.codec.parameters)
    assert set(AUXILIARY) <= set(metadata["parameters"])
    # A separate interpreter reconstructs the same task, including the trained
    # projector and head; no live model or parent-process tensor crosses the boundary.
    fresh_output = tmp_path / "fresh-process.pt"
    script = """
import sys, torch
from cosmos_framework.posttrain.tests.test_family_backbone import TaskPolicy, install, batch
from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata, load_adapter_weights
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
net = TaskPolicy()
state = install(net, mode=sys.argv[3])
metadata = read_adapter_metadata(sys.argv[1])
state.codec.bind_processor_identity(metadata["identity"]["processor_contract"])
load_adapter_weights(sys.argv[1], state.codec, manager=CheckpointManager(checkpoint_format="dcp"))
state.validate_frozen_base()
torch.save({"output": net(batch()[0]).detach(), "parameters": sorted(state.codec.parameters)}, sys.argv[2])
"""
    subprocess.run(
        [sys.executable, "-c", script, checkpoint, str(fresh_output), mode],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    fresh_result = torch.load(fresh_output, weights_only=True)
    assert fresh_result["parameters"] == sorted(state.codec.parameters)
    torch.testing.assert_close(fresh_result["output"], net(batch()[0]), atol=0, rtol=0)
    fresh = TaskPolicy()
    restored = install(fresh, mode=mode)
    restored.codec.bind_processor_identity(metadata["identity"]["processor_contract"])
    restored.codec.bind_training_identity(metadata["identity"]["training_contract"])
    load_adapter_weights(checkpoint, restored.codec, manager=CheckpointManager(checkpoint_format="dcp"))
    torch.testing.assert_close(fresh(batch()[0]), net(batch()[0]), atol=0, rtol=0)
    # Optimizer state is supplied explicitly: a portable artifact is not itself a resume checkpoint.
    resumed_optimizer = torch.optim.AdamW(restored.codec.parameters.values(), lr=0.02, weight_decay=0.0)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    state.codec.validate_metadata(metadata, optimizer, purpose="resume")
    restored.codec.validate_metadata(metadata, resumed_optimizer, purpose="resume")
    torch.testing.assert_close(
        update(fresh, restored, resumed_optimizer), update(net, state, optimizer), atol=0, rtol=0
    )
    for name, parameter in state.codec.parameters.items():
        torch.testing.assert_close(restored.codec.parameters[name], parameter, atol=0, rtol=0)
    with pytest.raises(ValueError, match="cannot resume"):
        read_adapter_metadata(checkpoint, require_resume=True)


@pytest.mark.parametrize("fault", ["omit_aux", "base", "task", "mode", "processor", "rank"])
def test_reload_refuses_incompatible_complete_task_identity(tmp_path, fault):
    net = TaskPolicy()
    state = install(net)
    state.codec.bind_processor_identity({"normalization": "original"})
    checkpoint = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000001"),
        state.codec.parameters,
        state.codec.metadata(),
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    kwargs = {}
    if fault == "omit_aux":
        kwargs["auxiliary"] = AUXILIARY[:-1]
    elif fault == "base":
        kwargs["source_identity"] = {**SOURCE, "revision": "b" * 40}
    elif fault == "task":
        kwargs["task_identity"] = {**TASK, "action_horizon": 3}
    elif fault == "mode":
        kwargs["mode"] = "bf16_frozen"
    fresh_net = TaskPolicy()
    fresh = install(fresh_net, **kwargs)
    fresh.codec.bind_processor_identity({"normalization": "other" if fault == "processor" else "original"})
    if fault == "rank":
        fresh.codec.identity["adapter"]["rank"] += 1
    with pytest.raises(ValueError, match="differs"):
        load_adapter_weights(checkpoint, fresh.codec, manager=CheckpointManager(checkpoint_format="dcp"))


@pytest.mark.parametrize("fault", ["unmatched", "shape", "biased", "subclass", "alias", "tied", "dtype"])
def test_target_resolution_rejects_invalid_architecture_before_mutation(fault):
    net = TaskPolicy()
    targets = dict(TARGETS)
    if fault == "unmatched":
        targets["missing"] = (4, 4)
    elif fault == "shape":
        targets["backbone.0"] = (4, 5)
    elif fault == "biased":
        net.backbone[0] = nn.Linear(4, 4, dtype=torch.bfloat16)
    elif fault == "subclass":

        class CustomLinear(nn.Linear):
            pass

        net.backbone[0] = CustomLinear(4, 4, bias=False, dtype=torch.bfloat16)
    elif fault == "alias":
        net.other = net.backbone[0]
    elif fault == "tied":
        net.vision.weight = net.backbone[0].weight
    else:
        net.backbone[0].weight = nn.Parameter(net.backbone[0].weight.to(torch.float16))
    parameters = dict(net.named_parameters())
    with pytest.raises(ValueError):
        resolve_family_targets(net, targets)
    assert dict(net.named_parameters()) == parameters


@pytest.mark.parametrize(
    "auxiliary", ["head.weight", ["head.weight", "head.weight"], ["missing"], ["backbone.0.weight"]]
)
def test_invalid_auxiliary_membership_refuses_before_mutation(auxiliary):
    net = TaskPolicy()
    with pytest.raises(ValueError, match="auxiliary|trainables"):
        install(net, auxiliary=auxiliary)
    assert all(p.requires_grad for p in net.parameters())


@pytest.mark.parametrize("fault", ["missing_reader", "shape", "dtype", "nonfinite", "meta_buffer"])
def test_meta_source_requires_exact_cpu_values_and_initialized_buffers(fault):
    source = TaskPolicy()
    values = {name: p.detach().clone() for name, p in source.named_parameters()}
    net = TaskPolicy(device="meta")
    if fault == "shape":
        values["backbone.0.weight"] = torch.zeros(4, 5, dtype=torch.bfloat16)
    elif fault == "dtype":
        values["backbone.0.weight"] = values["backbone.0.weight"].float()
    elif fault == "nonfinite":
        values["backbone.0.weight"][0, 0] = float("nan")
    elif fault == "meta_buffer":
        net.scale = torch.empty((), device="meta")
    with pytest.raises(ValueError):
        install(net, read_tensor=None if fault == "missing_reader" else values.__getitem__)


@pytest.mark.parametrize("name", ["backbone.0.weight", "head.weight"])
def test_nonfinite_cpu_source_refused(name):
    net = TaskPolicy()
    with torch.no_grad():
        net.get_parameter(name).flatten()[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        install(net)


@pytest.mark.parametrize("target", [True, False])
def test_finite_source_must_remain_finite_after_storage_cast(target):
    net = TaskPolicy()
    module = net.backbone[0] if target else net.head
    source_dtype = torch.float32 if target else torch.float64
    module.weight = nn.Parameter(torch.full(module.weight.shape, torch.finfo(source_dtype).max, dtype=source_dtype))
    assert torch.isfinite(module.weight).all()
    with pytest.raises(ValueError, match="overflows"):
        install(net)


@pytest.mark.parametrize("fault", ["frozen_weight", "buffer", "membership", "optimizer", "cast", "reinstall"])
def test_integrity_and_lifecycle_fail_closed(fault):
    net = TaskPolicy()
    state = install(net)
    if fault == "frozen_weight":
        with torch.no_grad():
            net.vision.weight.add_(1)
        with pytest.raises(RuntimeError, match="frozen family tensor"):
            state.validate_frozen_base()
    elif fault == "buffer":
        net.scale.add_(1)
        with pytest.raises(RuntimeError, match="frozen family buffer"):
            state.validate_frozen_base()
    elif fault == "membership":
        net.vision.weight.requires_grad_(True)
        with pytest.raises(ValueError, match="trainable model parameters"):
            state.codec.assert_live()
    elif fault == "optimizer":
        optimizer = torch.optim.AdamW([*state.codec.parameters.values(), net.vision.weight])
        with pytest.raises(ValueError, match="optimizer membership"):
            state.assert_optimizer(optimizer)
    elif fault == "cast":
        with pytest.raises(ValueError, match="blanket|dtype"):
            net.bfloat16()
        assert net.to("cpu") is net
    else:
        with pytest.raises(ValueError):
            install(net)


def test_nf4_cpu_never_silently_falls_back_to_dense():
    with pytest.raises(ValueError, match="CUDA.*fallback"):
        install(TaskPolicy(), mode="nf4_lora")


def test_explicit_single_device_boundary(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="FSDP/DTensor"):
        install(TaskPolicy())


def test_native_channels_last_vision_layout_is_preserved_and_bound():
    net = TaskPolicy()
    net.video = nn.Conv3d(3, 4, (2, 2, 2), bias=False, dtype=torch.bfloat16)
    net.video.to(memory_format=torch.channels_last_3d)
    before = net.video.weight.detach().clone()
    stride = net.video.weight.stride()
    assert not net.video.weight.is_contiguous()
    state = install(net)
    assert net.video.weight.stride() == stride
    torch.testing.assert_close(net.video.weight, before, atol=0, rtol=0)
    state.validate_frozen_base()
    # Equal values with a different execution layout still violate identity.
    net.video.weight.data = net.video.weight.data.contiguous()
    with pytest.raises(RuntimeError, match="frozen family tensor changed: video.weight"):
        state.validate_frozen_base()
