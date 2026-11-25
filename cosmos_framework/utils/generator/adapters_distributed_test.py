# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FSDP2 / HSDP with float32 adapters on a frozen bfloat16 base, on CPU processes (gloo)."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cosmos_framework.utils.generator.adapter_config import AdapterComponentConfig, AdapterConfig
from cosmos_framework.utils.generator.adapter_training import initialize_adapters
from cosmos_framework.utils.generator.adapters import AdapterLinear, adapter_mixed_precision_policy, install_adapters

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_BATCH = 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.norm = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.v_proj(torch.tanh(self.q_proj(x))))


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block() for _ in range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            x = block(x)
        return x


def _model(adapter_seed: int, *, single_target: bool = False) -> nn.Module:
    torch.manual_seed(0)
    net = _Tiny().to(torch.bfloat16)
    torch.manual_seed(adapter_seed)
    pissa = AdapterComponentConfig(
        targets=["model.layers.*.v_proj"], method="pissa", initialization="svd_lowrank", rank=4, alpha=8.0
    )
    if single_target:
        pissa.targets = ["model.layers.1.v_proj"]
        pissa.validate()
        config = AdapterConfig(components=[pissa])
    else:
        config = AdapterConfig(
            components=[
                AdapterComponentConfig(
                    targets=["model.layers.*.q_proj"], method="dora", rank=4, alpha=8.0, use_rslora=True
                ),
                pissa,
            ]
        )
    install_adapters(net, config, initialize=False)
    initialize_adapters(net)  # targets spread over ranks by cost, seeded from rank 0
    return net


def _inputs() -> torch.Tensor:
    return torch.randn(_BATCH, 16, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)


def _adapter_state(net: nn.Module) -> dict[str, torch.Tensor]:
    return {n: t.detach().clone() for n, t in net.state_dict().items() if "lora_" in n or "initial_" in n}


def _init(rank: int, world: int, init_file: str) -> None:
    # A shared file rendezvous avoids picking free TCP ports, which races between concurrent tests.
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world, timeout=timedelta(seconds=60)
    )


def _worker(rank: int, world: int, replicate: int, init_file: str, out: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    _init(rank, world, init_file)
    try:
        net = _model(adapter_seed=100 + rank)  # every rank seeds differently; all must end with rank 0's seed
        initial = _adapter_state(net)
        shard = world // replicate
        mesh = (
            init_device_mesh("cpu", (replicate, shard), mesh_dim_names=("replicate", "shard"))
            if replicate > 1
            else init_device_mesh("cpu", (shard,))
        )
        for block in net.model.layers:
            fully_shard(block, mesh=mesh, mp_policy=adapter_mixed_precision_policy())
        fully_shard(net, mesh=mesh, mp_policy=adapter_mixed_precision_policy())
        rows = _BATCH // world
        net(_inputs()[rank * rows : (rank + 1) * rows]).float().square().mean().backward()
        grads = {}
        for name, parameter in net.named_parameters():
            if parameter.requires_grad:
                assert parameter.dtype == torch.float32
                grad = parameter.grad
                grads[name] = grad.full_tensor() if isinstance(grad, DTensor) else grad
        if rank == 0:
            torch.save({"initial": initial, "grads": grads}, out)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "replicate"), [(2, 1), (2, 2), (4, 2)], ids=["fsdp_1x2", "replicate_2x1", "hsdp_2x2"]
)
def test_sharded_adapter_gradients_match_single_process(world: int, replicate: int, tmp_path: Path) -> None:
    out = str(tmp_path / "rank0.pt")
    mp.spawn(_worker, args=(world, replicate, str(tmp_path / "rendezvous"), out), nprocs=world, join=True)
    result = torch.load(out)

    reference = _model(adapter_seed=100)
    assert all(
        isinstance(reference.get_submodule(f"model.layers.{i}.{p}"), AdapterLinear)
        for i in (0, 1)
        for p in ("q_proj", "v_proj")
    )
    for name, tensor in _adapter_state(reference).items():
        torch.testing.assert_close(result["initial"][name], tensor, atol=0, rtol=0)
    reference(_inputs()).float().square().mean().backward()
    expected = {n: p.grad for n, p in reference.named_parameters() if p.requires_grad}
    assert result["grads"].keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(result["grads"][name], grad, atol=1e-4, rtol=1e-3, msg=name)


def _more_ranks_worker(rank: int, world: int, init_file: str, out: str) -> None:
    _init(rank, world, init_file)
    try:
        torch.save(_adapter_state(_model(adapter_seed=100 + rank, single_target=True)), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_more_ranks_than_targets(tmp_path: Path) -> None:
    out = str(tmp_path / "state")
    mp.spawn(_more_ranks_worker, args=(4, str(tmp_path / "rendezvous"), out), nprocs=4, join=True)
    expected = _adapter_state(_model(adapter_seed=100, single_target=True))
    for rank in range(4):
        state = torch.load(f"{out}.{rank}")
        assert state.keys() == expected.keys()
        for name, tensor in expected.items():
            torch.testing.assert_close(state[name], tensor, atol=0, rtol=0)


def _failing_worker(rank: int, world: int, failing_rank: int, init_file: str, out: str) -> None:
    _init(rank, world, init_file)
    try:
        if rank == failing_rank:

            def fail(self: AdapterLinear, generator: torch.Generator | None = None) -> None:
                raise RuntimeError(f"svd failed on rank {rank}")

            AdapterLinear.initialize = fail
        try:
            _model(adapter_seed=100)
            outcome = "no error"
        except RuntimeError as error:
            outcome = str(error)
        Path(f"{out}.{rank}").write_text(outcome)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_initialization_failure_on_one_rank_raises_on_every_rank(failing_rank: int, tmp_path: Path) -> None:
    out = str(tmp_path / "outcome")
    mp.spawn(_failing_worker, args=(2, failing_rank, str(tmp_path / "rendezvous"), out), nprocs=2, join=True)
    assert Path(f"{out}.{failing_rank}").read_text() == f"svd failed on rank {failing_rank}"
    assert Path(f"{out}.{1 - failing_rank}").read_text() == "adapter initialization failed on another rank"


def _setup_failure_worker(rank: int, world: int, stage: str, init_file: str, out: str) -> None:
    from cosmos_framework.utils.generator import adapter_training
    from cosmos_framework.utils.generator.adapter_config import AdapterSetup

    _init(rank, world, init_file)
    try:
        failing = {"install": 0, "fingerprint": 1}[stage]
        if rank == failing:
            name = {"install": "install_adapters", "fingerprint": "model_base_fingerprint"}[stage]

            def fail(*args: object, **kwargs: object) -> None:
                raise RuntimeError(f"{stage} failed on rank {rank}")

            setattr(adapter_training, name, fail)
        torch.manual_seed(0)
        net = _Tiny().to(torch.bfloat16)
        config = AdapterConfig(components=[AdapterComponentConfig(targets=["model.layers.*.q_proj"], rank=2)])
        try:
            adapter_training.prepare_adapter_model(
                net, AdapterSetup(config), base_dtype="bfloat16", architecture="qwen3"
            )
            outcome = "no error"
        except RuntimeError as error:
            outcome = str(error)
        Path(f"{out}.{rank}").write_text(outcome)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", ["install", "fingerprint"])
def test_setup_failure_on_one_rank_raises_on_every_rank(stage: str, tmp_path: Path) -> None:
    out = str(tmp_path / "outcome")
    mp.spawn(_setup_failure_worker, args=(2, stage, str(tmp_path / "rendezvous"), out), nprocs=2, join=True)
    failing = {"install": 0, "fingerprint": 1}[stage]
    assert Path(f"{out}.{failing}").read_text() == f"{stage} failed on rank {failing}"
    assert Path(f"{out}.{1 - failing}").read_text() == "adapter installation failed on another rank"
