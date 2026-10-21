# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real FSDP2 coverage for trainer-owned validation cleanup."""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _local_shape(parameter: Any) -> tuple[int, ...]:
    to_local = getattr(parameter, "to_local", None)
    local = to_local() if callable(to_local) else parameter  # [*P_local]
    return tuple(int(dimension) for dimension in local.shape)


def _worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from cosmos_framework.posttrain.roles.trainer.validation.runtime import reshard_fsdp_modules

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
        net = torch.nn.Linear(8, 8, bias=False, device=f"cuda:{rank}")  # weight: [8,8]
        fully_shard(net, mesh=mesh, reshard_after_forward=False)
        weight = next(net.parameters())  # [8,8]
        sharded_shape = _local_shape(weight)
        assert sharded_shape == (8 // world_size, 8)

        with torch.inference_mode():
            output = net(torch.ones(1, 8, device=f"cuda:{rank}"))  # [1,8]
            assert tuple(output.shape) == (1, 8)
            assert _local_shape(next(net.parameters())) == (8, 8)
        reshard_fsdp_modules({"net": net})
        assert _local_shape(next(net.parameters())) == sharded_shape

        with pytest.raises(RuntimeError, match="synthetic objective failure"):
            try:
                with torch.inference_mode():
                    net(torch.ones(1, 8, device=f"cuda:{rank}"))  # [1,8]
                    assert _local_shape(next(net.parameters())) == (8, 8)
                    raise RuntimeError("synthetic objective failure")
            finally:
                reshard_fsdp_modules({"net": net})
        assert _local_shape(next(net.parameters())) == sharded_shape
    finally:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.GPU
def test_preference_validation_reshards_fsdp_root_after_success_and_failure() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    import torch.multiprocessing as mp

    world_size = 2
    with socket.socket() as rendezvous:
        rendezvous.bind(("", 0))
        port = int(rendezvous.getsockname()[1])
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
