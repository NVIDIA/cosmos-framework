# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Two-rank FSDP2 gate for the generator/VFM eight-bit optimizer path.

Uses real ``fully_shard`` with the VFM mixed-precision policy (FP32 shards, BF16
compute, FP32 reduction), the native global-norm clip, the scheduler container and
the production ``DistributedCheckpointer`` save/load in separate process groups,
so resume is checked across fresh processes against an uninterrupted reference.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

# Spawned ranks pick their own devices; conftest pins unmarked tests to one GPU.
DEVICES = os.environ.get("COSMOS_TEST_DEVICES", "0,1")
pytestmark = pytest.mark.skipif(
    len(DEVICES.split(",")) < 2 or not torch.cuda.is_available(),
    reason="FSDP2 eight-bit gate requires two CUDA devices",
)


@pytest.fixture(autouse=True)
def _two_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", DEVICES)


KWARGS = dict(
    lr=5e-4,
    betas=[0.9, 0.99],
    eps=1e-8,
    weight_decay=0.05,
    keys_to_select=["gen", "llm2action"],
    lr_multipliers={"llm2action": 5.0},
)
SAVE_AT, TOTAL = 3, 6


def _setup(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, device_id=torch.device("cuda", rank)
    )


def _model(mesh_shape=(2,), names=("dp_shard",)):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from cosmos_framework.utils.generator.quantized_optimizer_container_test import _Model

    model = _Model("cuda")
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=names)
    if len(mesh_shape) > 1:
        mesh = mesh[names]
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for child in (model.net.gen, model.net.llm2action):
        fully_shard(child, mesh=mesh, mp_policy=policy)
    fully_shard(model.net, mesh=mesh, mp_policy=policy)
    return model


def _build(model, name):
    from cosmos_framework.utils.generator.optimizer import build_lr_scheduler, build_optimizer

    # The legacy optimizers require fused kernels; the eight-bit backend refuses a fused request.
    container = build_optimizer(model, name, **dict(KWARGS, **({"fused": True} if name == "adamw" else {})))
    scheduler = build_lr_scheduler(
        container,
        lr_scheduler_type="LambdaLinear",
        cycle_lengths=[100],
        f_max=[1.0],
        f_min=[0.0],
        f_start=[1e-6],
        warm_up_steps=[2],
        verbosity_interval=0,
    )
    return container, scheduler


def _train_step(model, container, scheduler, step):
    torch.manual_seed(1000 * step + dist.get_rank())
    x = torch.randn(8, 256, device="cuda")
    model.net(x).float().square().mean().backward()
    # Native global-norm clip over DTensor gradients, before the optimizer sees them.
    torch.nn.utils.clip_grad_norm_([p for p in model.net.parameters() if p.grad is not None], 0.05)
    container.step()
    scheduler.step()
    container.zero_grad()


def _full(model):
    return {n: p.full_tensor().detach().cpu() for n, p in model.net.named_parameters()}


def _checkpointer(root, name):
    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.utils.config import CheckpointConfig, JobConfig

    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(root)
    return DistributedCheckpointer(
        CheckpointConfig(dcp_load_dedup=True, strict_resume=True), JobConfig(project="q", group="g", name=name)
    )


def _reference_worker(rank, rendezvous, root, name):
    _setup(rank, rendezvous)
    try:
        model = _model()
        container, scheduler = _build(model, name)
        optimizer = container.optimizers[0]
        assert optimizer.identity["world_size"] == 2 and optimizer.identity["parameters"][0]["placements"] == [
            "Shard(0)"
        ]
        assert all(m.data_ptr() == p.to_local().data_ptr() for m, p in zip(optimizer.masters, optimizer.live))
        frozen = model.net.frozen.weight.detach().clone()
        initial = _full(model)
        checkpointer = _checkpointer(root, name)
        scaler = torch.amp.GradScaler(enabled=False)
        for step in range(1, TOTAL + 1):
            _train_step(model, container, scheduler, step)
            if step == SAVE_AT:
                checkpointer.save(model, container, scheduler, scaler, iteration=step)
        final = _full(model)
        assert torch.equal(frozen, model.net.frozen.weight)
        assert all(not torch.equal(initial[n], final[n]) for n in initial if not n.startswith("frozen"))
        report = container.state_report()
        assert report["master_bytes"] == 0 and report["uninitialized_elements"] == 0
        if rank == 0:
            torch.save(final, Path(root) / "final.pt")
            (Path(root) / "report.json").write_text(json.dumps({k: v for k, v in report.items() if k != "backend"}))
        checkpointer.finalize()
    finally:
        dist.destroy_process_group()


def _legacy_report_worker(rank, rendezvous):
    _setup(rank, rendezvous)
    try:
        model = _model()
        container, scheduler = _build(model, "adamw")
        _train_step(model, container, scheduler, 1)
        expected = sum(2 * p.to_local().numel() * 4 for g in container.optimizers[0].param_groups for p in g["params"])
        # DTensor moments must be counted per local shard, not deduplicated by wrapper.
        assert container.state_report()["moment_bytes"] == expected
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, rendezvous, root, name):
    _setup(rank, rendezvous)
    try:
        model = _model()
        container, scheduler = _build(model, name)
        checkpointer = _checkpointer(root, name)
        scaler = torch.amp.GradScaler(enabled=False)
        iteration = checkpointer.load(model, container, scheduler, scaler)
        assert iteration == SAVE_AT
        for step in range(iteration + 1, TOTAL + 1):
            _train_step(model, container, scheduler, step)
        expected = torch.load(Path(root) / "final.pt")
        for n, value in _full(model).items():
            assert torch.equal(value, expected[n]), n
        checkpointer.finalize()
    finally:
        dist.destroy_process_group()


def _hsdp_worker(rank, rendezvous, name):
    _setup(rank, rendezvous)
    try:
        model = _model((1, 2), ("dp_replicate", "dp_shard"))
        try:
            _build(model, name)
        except ValueError as error:
            assert "one-dimensional FSDP2" in str(error)
        else:
            raise AssertionError("HSDP layout was accepted")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_fsdp2_fresh_process_exact_resume(tmp_path, name):
    torch.multiprocessing.spawn(
        _reference_worker, args=((tmp_path / "r1").as_uri(), str(tmp_path), name), nprocs=2, join=True
    )
    torch.multiprocessing.spawn(
        _resume_worker, args=((tmp_path / "r2").as_uri(), str(tmp_path), name), nprocs=2, join=True
    )


def test_hsdp_two_dimensional_mesh_is_rejected_before_state(tmp_path):
    torch.multiprocessing.spawn(_hsdp_worker, args=((tmp_path / "r").as_uri(), "adamw_bnb_8bit"), nprocs=2, join=True)


def test_legacy_report_counts_local_dtensor_moments(tmp_path):
    torch.multiprocessing.spawn(_legacy_report_worker, args=((tmp_path / "r").as_uri(),), nprocs=2, join=True)


def _logical_eligibility_worker(rank, rendezvous):
    from cosmos_framework.utils.generator.optimizer import build_optimizer

    _setup(rank, rendezvous)
    try:
        model = _model()
        # llm2action.weight is 8192 elements, 4096 per rank. Against a 6000-element threshold the
        # local shard alone would keep FP32 moments; the logical parameter is eligible for 8 bits.
        container = build_optimizer(model, "adamw_bnb_8bit", **dict(KWARGS, bnb_min_8bit_size=6000))
        optimizer = container.optimizers[0]
        torch.manual_seed(7 + rank)
        model.net(torch.randn(8, 256, device="cuda")).float().square().mean().backward()
        container.step()
        dtypes = {
            name: optimizer.backend.state[master]["state1"].dtype
            for name, master in zip(optimizer.names, optimizer.masters)
            if master in optimizer.backend.state
        }
        assert dtypes["llm2action.weight"] == torch.uint8, dtypes
        assert dtypes["gen.bias"] == torch.float32  # 512 logical elements: below threshold everywhere
        assert optimizer.backend.identity["min_8bit_size_basis"] == "logical_parameter"
        gathered = [None, None]
        dist.all_gather_object(gathered, {k: str(v) for k, v in dtypes.items()})
        assert gathered[0] == gathered[1]  # one moment precision per parameter on every rank
    finally:
        dist.destroy_process_group()


def test_eight_bit_eligibility_uses_logical_parameter_size(tmp_path):
    torch.multiprocessing.spawn(_logical_eligibility_worker, args=((tmp_path / "r").as_uri(),), nprocs=2, join=True)
