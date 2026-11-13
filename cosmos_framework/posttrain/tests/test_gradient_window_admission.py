# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU admission fixtures around real public FSDP setters, with no collectives."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Replicate, Shard

from cosmos_framework.posttrain.roles.trainer import gradient_window as policy

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class FixtureModule(torch.nn.Module, FSDPModule):
    def __new__(cls):
        # Real FSDP dynamically replaces a module's class; construct just that
        # surface here without pretending to initialize real CPU FSDP shards.
        return object.__new__(cls)

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_group=self.group)


def model(monkeypatch, layout="hsdp"):
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils import dist as framework_dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(framework_dist, "run_rank_converged", lambda fn, context: fn())
    monkeypatch.setattr(framework_dist, "collective_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op: None)
    net = FixtureModule()
    net.weight = torch.nn.Parameter(torch.ones(4))
    net.alias = net.weight
    shards = 4 if layout == "fsdp" else 2
    mesh = torch.arange(4) if layout == "fsdp" else torch.arange(4).reshape(2, 2)
    net.weight.device_mesh = SimpleNamespace(mesh=mesh, ndim=mesh.ndim, size=lambda axis: mesh.size(axis))
    net.weight.placements = (Shard(0),) if layout == "fsdp" else (Replicate(), Shard(0))
    net.weight.to_local = lambda: net.weight[: 4 // shards]
    slot = SimpleNamespace(
        sharded_param=net.weight,
        mp_policy=SimpleNamespace(reduce_dtype=torch.float64, param_dtype=None),
        offload_to_cpu=False,
        unsharded_accumulated_grad=None,
    )
    net.group = SimpleNamespace(
        fsdp_params=[slot],
        _all_reduce_hook=None,
        _all_reduce_hook_stream=None,
        gradient_divide_factor=None,
        reduce_grads=True,
        all_reduce_grads=True,
        _partial_reduce_output=None,
        _is_hsdp=layout == "hsdp",
    )
    return net, torch.optim.AdamW([net.weight], lr=1e-3)


@pytest.mark.parametrize("layout", ["fsdp", "hsdp"])
def test_real_public_setters_install_only_the_unique_owner(monkeypatch, layout):
    net, optimizer = model(monkeypatch, layout)
    window = policy.install_fsdp(net, optimizer, microbatches=2)
    assert len(window.owners) == 1 and window.owners[0].aliases == ("weight", "alias")
    assert net.group.gradient_divide_factor == 1.0
    assert net.group._all_reduce_hook is not None
    window.prepare_update(0)
    assert window.phase == "open" and window.forward_prepared
    with pytest.raises(RuntimeError, match="conflict"):
        policy.install_fsdp(net, optimizer, microbatches=2)


@pytest.mark.parametrize(
    "failure",
    [
        "dtype",
        "compute",
        "offload",
        "no_sync",
        "no_all_reduce",
        "hook",
        "stream",
        "divisor",
        "pending",
        "unsharded",
        "missing_owner",
        "duplicate_optimizer",
        "unknown_optimizer",
        "wrong_world",
        "wrong_mesh",
        "wrong_axis",
        "source",
        "optimizer_type",
        "differentiable",
    ],
)
def test_admission_rejects_incompatible_owners_before_hook_registration(monkeypatch, failure):
    net, optimizer = model(monkeypatch)
    slot = net.group.fsdp_params[0]
    if failure == "dtype":
        slot.mp_policy.reduce_dtype = torch.float32
    elif failure == "compute":
        slot.mp_policy.param_dtype = torch.bfloat16
    elif failure == "offload":
        slot.offload_to_cpu = True
    elif failure == "no_sync":
        net.group.reduce_grads = False
    elif failure == "no_all_reduce":
        net.group.all_reduce_grads = False
    elif failure == "hook":
        net.group._all_reduce_hook = object()
    elif failure == "stream":
        net.group._all_reduce_hook_stream = object()
    elif failure == "divisor":
        net.group.gradient_divide_factor = 4
    elif failure == "pending":
        net.group._partial_reduce_output = torch.ones(1)
    elif failure == "unsharded":
        slot.unsharded_accumulated_grad = torch.ones(1)
    elif failure == "missing_owner":
        net.group.fsdp_params = []
    elif failure == "duplicate_optimizer":
        optimizer.param_groups[0]["params"].append(net.weight)
    elif failure == "unknown_optimizer":
        optimizer.param_groups[0]["params"] = [torch.nn.Parameter(torch.ones(4))]
    elif failure == "wrong_world":
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    elif failure == "wrong_mesh":
        net.weight.device_mesh.mesh = torch.tensor([[0, 1], [2, 5]])
    elif failure == "wrong_axis":
        net.weight.placements = (Shard(0), Replicate())
    elif failure == "optimizer_type":
        optimizer = torch.optim.SGD([net.weight], lr=1e-3)
    elif failure == "differentiable":
        optimizer.defaults["differentiable"] = True
    else:
        monkeypatch.setitem(policy.TORCH_SOURCE_SHA256, "_fully_shard.py", "0" * 64)
    before = net.group._all_reduce_hook
    with pytest.raises((ValueError, RuntimeError)):
        policy.install_fsdp(net, optimizer, microbatches=2)
    assert net.group._all_reduce_hook is before


@pytest.mark.parametrize(
    "failure",
    [
        "optimizer",
        "registry",
        "alias",
        "dtype",
        "no_sync",
        "no_all_reduce",
        "hook",
        "divisor",
        "offload",
        "compute",
        "slot",
        "pending",
        "unsharded",
    ],
)
def test_late_policy_change_fails_before_the_next_native_forward(monkeypatch, failure):
    net, optimizer = model(monkeypatch)
    window = policy.install_fsdp(net, optimizer, microbatches=2)
    slot = net.group.fsdp_params[0]
    if failure == "optimizer":
        optimizer.param_groups[0]["params"].clear()
    elif failure == "registry":
        net.extra = torch.nn.Parameter(torch.ones(1))
    elif failure == "alias":
        net.new_alias = net.weight
    elif failure == "dtype":
        net.weight.data = net.weight.data.to(torch.float64)
    elif failure == "no_sync":
        net.group.reduce_grads = False
    elif failure == "no_all_reduce":
        net.group.all_reduce_grads = False
    elif failure == "hook":
        net.group._all_reduce_hook = lambda value: None
    elif failure == "divisor":
        net.group.gradient_divide_factor = 4
    elif failure == "offload":
        slot.offload_to_cpu = True
    elif failure == "compute":
        slot.mp_policy.param_dtype = torch.bfloat16
    elif failure == "pending":
        net.group._partial_reduce_output = torch.zeros(1)
    elif failure == "unsharded":
        slot.unsharded_accumulated_grad = torch.zeros(1)
    else:
        slot.sharded_param = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(RuntimeError):
        window.prepare_update(0)


def test_no_sync_changed_during_forward_fails_before_backward(monkeypatch):
    net, optimizer = model(monkeypatch)
    window = policy.install_fsdp(net, optimizer, microbatches=2)
    window.prepare_update(0)
    net.group.all_reduce_grads = False
    with pytest.raises(RuntimeError, match="no-sync"):
        window.begin_native_micro(0)
    assert window.phase == "open" and not window.pending


def test_optimizer_changed_during_forward_fails_before_backward(monkeypatch):
    net, optimizer = model(monkeypatch)
    window = policy.install_fsdp(net, optimizer, microbatches=2)
    window.prepare_update(0)
    optimizer.param_groups[0]["params"].clear()
    with pytest.raises(RuntimeError, match="optimizer ownership"):
        window.begin_native_micro(0)
    assert window.phase == "open" and not window.pending
