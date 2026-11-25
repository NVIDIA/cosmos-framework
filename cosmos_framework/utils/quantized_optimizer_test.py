# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for the bitsandbytes eight-bit backend helpers."""

import gc

import pytest
import torch

from cosmos_framework.utils.quantized_optimizer_names import validate_quantized_options

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
VALID = dict(lr=1e-4, eps=1e-8, weight_decay=0.0, betas=(0.9, 0.999), bnb_min_8bit_size=4096, bnb_fp32_state_names=[])


@pytest.mark.parametrize(
    "invalid",
    [
        {"bnb_min_8bit_size": 0},
        {"bnb_min_8bit_size": 1.5},
        {"bnb_fp32_state_names": [""]},
        {"bnb_fp32_state_names": ["w", "w"]},
        {"lr": float("nan")},
        {"eps": -1.0},
        {"weight_decay": True},
        {"betas": (0.9,)},
        {"betas": (0.9, 1.0)},
    ],
)
def test_shared_validation_refuses_what_the_backend_cannot_run(invalid):
    with pytest.raises(ValueError):
        validate_quantized_options(**{**VALID, **invalid})
    validate_quantized_options(**VALID)


def _paged_optimizer():
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    net = torch.nn.Linear(1024, 1024).cuda()
    optimizer = build_optimizer(net, OptimizerConfig(name="paged_adamw_8bit", bnb_min_8bit_size=1))
    net(torch.randn(2, 1024, device="cuda")).sum().backward()
    optimizer.step()
    return net, optimizer


@CUDA
def test_reset_state_releases_the_managed_paged_buffers():
    net, optimizer = _paged_optimizer()
    owners = [buffer._cosmos_managed_owner for buffer in optimizer.page_mng.paged_tensors]
    assert owners and all(owner() is not None for owner in owners)
    optimizer.reset_state()
    gc.collect()
    assert not optimizer.state and not optimizer.page_mng.paged_tensors
    assert all(owner() is None for owner in owners)  # cudaFree ran: nothing keeps them alive


@CUDA
def test_device_view_addresses_paged_state_on_the_gpu_without_a_copy():
    from cosmos_framework.utils.quantized_optimizer import device_view

    net, optimizer = _paged_optimizer()
    state = optimizer.state[net.weight]["state1"]
    assert state.is_paged and state.device.type == "cpu"
    view = device_view(state)
    assert view.is_cuda and view.data_ptr() == state.data_ptr()
    snapshot = view.clone()
    assert snapshot.is_cuda and torch.equal(snapshot.cpu(), state)
    view.zero_()
    torch.cuda.synchronize()
    assert not state.any()  # writes through the view land in the managed buffer
    plain = optimizer.state[net.weight]["qmap1"]
    assert device_view(plain) is plain


@CUDA
def test_noted_finite_gradients_skip_the_backend_scan_once(monkeypatch):
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    net = torch.nn.Linear(64, 64).cuda()
    optimizer = build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit", bnb_min_8bit_size=1))
    scans = []
    real = torch._foreach_norm
    monkeypatch.setattr(torch, "_foreach_norm", lambda *a, **k: scans.append(1) or real(*a, **k))

    def step(noted):
        for parameter in net.parameters():
            parameter.grad = torch.ones_like(parameter)
        if noted:
            optimizer.note_gradients_checked_finite()
        optimizer.step()

    step(noted=True)
    assert scans == []  # the trainer's collective check already decided
    step(noted=False)
    assert scans == [1]  # a note covers one step only; unnoted steps still scan
    for parameter in net.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError):
        optimizer.step()


@CUDA
def test_paged_state_snapshots_are_staged_on_the_gpu_into_pinned_host_memory():
    net, optimizer = _paged_optimizer()
    live = optimizer.state[net.weight]["state1"]
    saved = optimizer.state_dict()["state"]
    weight = saved[0]  # parameter order: weight, bias
    assert weight["state1"].is_pinned() and torch.equal(weight["state1"], live)  # complete on return
    assert not saved[1]["state1"].is_pinned()  # small, non-paged CUDA state: an ordinary pageable copy
    assert torch.equal(saved[1]["state1"], optimizer.state[net.bias]["state1"].cpu())


@CUDA
def test_non_paged_snapshots_are_pageable_independent_copies_without_extra_syncs(monkeypatch):
    from cosmos_framework.utils.quantized_optimizer import _snapshot

    state = {"a": torch.randn(64, device="cuda"), "b": [torch.arange(8, device="cuda"), 3], "c": torch.ones(2)}
    syncs = []
    real = torch.cuda.synchronize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: syncs.append(device) or real(device))
    snapshot = _snapshot(state)
    # No page-locked host memory for ordinary state (it would stay cached for the whole run).
    assert syncs == []
    assert not snapshot["a"].is_pinned() and not snapshot["b"][0].is_pinned() and snapshot["b"][1] == 3
    assert torch.equal(snapshot["a"], state["a"].cpu()) and torch.equal(snapshot["b"][0], state["b"][0].cpu())
    state["a"].add_(1)
    state["c"].add_(1)
    assert not torch.equal(snapshot["a"], state["a"].cpu()) and torch.equal(snapshot["c"], torch.ones(2))  # independent


@pytest.mark.parametrize(
    "installed, accepted",
    [("0.49.2", True), ("0.49.2+cu130", True), ("0.49.2.post1", True), ("0.49.1", False), ("0.49.2rc1", False)],
)
def test_installed_release_is_compared_without_its_local_build_tag(installed, accepted, monkeypatch):
    import importlib.metadata

    from cosmos_framework.utils.quantized_optimizer import require_installed

    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    if accepted:
        require_installed()
    else:
        with pytest.raises(RuntimeError, match="bitsandbytes==0.49.2 required"):
            require_installed()


@CUDA
def test_restore_never_writes_the_shared_quantization_maps():
    net, optimizer = _paged_optimizer()
    saved = optimizer.state_dict()
    qmap = optimizer.state[net.weight]["qmap1"]
    version = qmap._version
    optimizer.load_state_dict(saved)
    assert optimizer.state[net.weight]["qmap1"] is qmap and qmap._version == version
    assert torch.equal(optimizer.state[net.weight]["state1"], saved["state"][0]["state1"])


@CUDA
@pytest.mark.parametrize(
    "damage",
    ["state1_dtype", "absmax_shape", "qmap_value", "extra_key", "negative_step"],
)
def test_state_payload_is_fully_validated_before_any_allocation(damage):
    net, optimizer = _paged_optimizer()
    saved = optimizer.state_dict()
    optimizer.validate_state_dict(saved)  # a payload this optimizer wrote is accepted
    entry = saved["state"][0]  # the eight-bit weight
    if damage == "state1_dtype":
        entry["state1"] = entry["state1"].float()
    elif damage == "absmax_shape":
        entry["absmax1"] = entry["absmax1"][:-1]
    elif damage == "qmap_value":
        entry["qmap2"] = entry["qmap2"].flip(0)
    elif damage == "extra_key":
        entry["unorm_vec"] = torch.zeros(1)
    else:
        entry["step"] = -1
    state, pages = optimizer.state, list(optimizer.page_mng.paged_tensors)
    with pytest.raises(ValueError, match="quantized optimizer"):
        optimizer.validate_state_dict(saved)
    with pytest.raises(ValueError, match="quantized optimizer"):
        optimizer.load_state_dict(saved)
    assert optimizer.state is state and optimizer.page_mng.paged_tensors == pages  # nothing replaced
