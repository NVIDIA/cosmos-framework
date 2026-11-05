# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HostOffload: host/device round trips keep values, parameter identity and optimizer keying."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.utils.residency import (
    MEMORY_SAVER_ENABLE_ENV,
    HostOffload,
    make_offload,
    memory_saver_active,
    untracked_allocation,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _model_and_opt(device: str) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16), torch.nn.Linear(16, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, 8, device=device)  # [4,8]
    model(x).sum().backward()
    opt.step()
    return model, opt


@pytest.mark.L0
@pytest.mark.CPU
def test_cpu_state_is_never_moved_and_onload_is_a_noop():
    model, opt = _model_and_opt("cpu")
    offload = HostOffload()
    assert offload.offload([model], optimizer=opt, grads_of=model.parameters()) == 0.0
    assert not offload.offloaded
    assert offload.onload() == 0.0


@cuda
@pytest.mark.L1
@pytest.mark.GPU
def test_offload_frees_device_memory_and_onload_restores_values_and_identity():
    model, opt = _model_and_opt("cuda")
    params = list(model.parameters())
    before = {id(p): p.detach().clone() for p in params}
    state_before = {k: {n: v.clone() for n, v in s.items() if torch.is_tensor(v)} for k, s in opt.state.items()}
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()

    offload = HostOffload()
    moved = offload.offload([model], optimizer=opt)
    assert moved > 0
    assert torch.cuda.memory_allocated() < allocated
    assert all(p.device.type == "cpu" and p.grad.device.type == "cpu" for p in params)
    # Optimizer state is keyed by parameter object; identity must survive the move.
    assert set(opt.state) == set(params)
    assert offload.offload([model], optimizer=opt) == 0.0  # already offloaded

    assert offload.onload() == moved
    assert all(p.device.type == "cuda" for p in params)
    for p in params:
        torch.testing.assert_close(p.detach(), before[id(p)])
    for k, saved in state_before.items():
        for n, v in saved.items():
            assert opt.state[k][n].device == v.device  # moments back on the GPU, `step` stays on CPU
            torch.testing.assert_close(opt.state[k][n], v)
    # A second cycle still round-trips, and training continues on the restored state.
    offload.offload([model], optimizer=opt)
    offload.onload()
    x = torch.randn(4, 8, device="cuda")  # [4,8]
    model(x).sum().backward()
    opt.step()


@cuda
@pytest.mark.L1
@pytest.mark.GPU
def test_fsdp2_model_round_trip_trains_identically_to_an_untouched_twin():
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    if not dist.is_initialized():
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    def build() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 2)).cuda()
        for layer in model:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-2)

    torch.manual_seed(7)
    x = torch.randn(4, 8, device="cuda")  # [4,8]
    ref_model, ref_opt = build()
    model, opt = build()
    for m, o in ((ref_model, ref_opt), (model, opt)):
        m(x).sum().backward()
        o.step()
        o.zero_grad()
    assert all(isinstance(p, DTensor) for p in model.parameters())

    offload = HostOffload()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    assert offload.offload([model], optimizer=opt) > 0
    assert torch.cuda.memory_allocated() < allocated
    assert all(isinstance(p, DTensor) and p.to_local().device.type == "cpu" for p in model.parameters())
    offload.onload()
    assert all(p.to_local().device.type == "cuda" for p in model.parameters())

    for m, o in ((ref_model, ref_opt), (model, opt)):
        m(x).sum().backward()
        o.step()
    for p_ref, p in zip(ref_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(p.to_local(), p_ref.to_local(), rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_without_the_memory_saver_hook_the_host_mover_is_selected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MEMORY_SAVER_ENABLE_ENV, raising=False)
    assert not memory_saver_active()
    assert isinstance(make_offload(), HostOffload)
    with untracked_allocation():  # plain pass-through when nothing is tracked
        pass


@pytest.mark.L0
@pytest.mark.CPU
def test_hook_env_alone_does_not_select_the_memory_saver(monkeypatch: pytest.MonkeyPatch):
    """The env flag says the process was started under the hook; the package must also be present."""
    import importlib.util

    monkeypatch.setenv(MEMORY_SAVER_ENABLE_ENV, "1")
    if importlib.util.find_spec("torch_memory_saver") is None:
        assert not memory_saver_active()
        assert isinstance(make_offload(), HostOffload)


@cuda
@pytest.mark.L1
@pytest.mark.GPU
def test_offload_moves_optimizer_tensors_outside_state_such_as_master_weights():
    """FusedAdam-style optimizers keep fp32 masters in ``param_groups_master`` and capturable ``lr``
    tensors in the groups; both must leave the device with the rest of the optimizer."""
    model = torch.nn.Linear(8, 4).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.param_groups[0]["lr"] = torch.tensor(0.1, device="cuda")
    opt.param_groups_master = [{"params": [p.detach().clone().float() for p in model.parameters()]}]
    master_before = [m.clone() for m in opt.param_groups_master[0]["params"]]

    offload = HostOffload()
    moved = offload.offload([model], optimizer=opt)
    assert opt.param_groups[0]["lr"].device.type == "cpu"
    assert all(m.device.type == "cpu" for m in opt.param_groups_master[0]["params"])
    assert moved > sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20  # more than the params alone

    offload.onload()
    assert opt.param_groups[0]["lr"].device.type == "cuda"
    for m, saved in zip(opt.param_groups_master[0]["params"], master_before, strict=True):
        assert m.device.type == "cuda"
        torch.testing.assert_close(m, saved)
    assert next(model.parameters()).device.type == "cuda"
