# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.checkpoint import dcp as checkpoint
from cosmos_framework.checkpoint.dcp_distill import DistributedCheckpointer as DistillCheckpointer

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def test_staging_bytes_count_local_allocations_and_page_rounding() -> None:
    tensor = torch.empty(1025, dtype=torch.float32)  # [1025]
    view = tensor[::2]  # [513]
    scalar = torch.tensor(1)  # []
    empty = torch.empty(0)  # [0]
    state = {"model": (tensor, "output"), "optim": [view, view, scalar, empty], "iteration": 3}
    # The two source views alias, but staging creates separate dense tensors.
    assert checkpoint._staging_shared_memory_bytes(state, 4096) == 5 * 4096


@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
def test_insufficient_capacity_stops_before_staging_allocation(checkpointer_type: type) -> None:
    handler = object.__new__(checkpointer_type)
    handler.async_mode = checkpoint.AsyncMode.ASYNC_WITH_PINNED_MEM
    handler.cpu_offload_state_dict = None
    state = {"model": (torch.empty(4), "model")}  # model tensor: [4]
    with (
        patch.object(checkpoint.os, "statvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096)),
        patch("torch.distributed._state_dict_utils._create_cpu_state_dict") as allocate,
        pytest.raises(RuntimeError, match="checkpoint.dcp_async_mode_enabled=false"),
    ):
        handler._checkpoint_async_with_pinned_memory("iter_000000002", state)
    allocate.assert_not_called()


def _distributed_capacity_worker(rank: int, rendezvous: str, capacity_store_path: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2,))
        local_tensor = torch.empty(1024)  # [1024]
        tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])  # global: [2048], local: [1024]
        state = {"model": tensor}
        assert checkpoint._staging_shared_memory_bytes(state, 4096) == 4096

        # Both ranks fit together in two pages; one page per rank independently
        # fits, but the aggregate does not fit in a one-page shared mount.
        for pages in (2, 1):
            with patch.object(checkpoint.os, "statvfs", return_value=SimpleNamespace(f_bavail=pages, f_frsize=4096)):
                if pages == 2:
                    checkpoint._check_staging_shared_memory(state)
                else:
                    with pytest.raises(RuntimeError, match="Insufficient /dev/shm"):
                        checkpoint._check_staging_shared_memory(state)

        # With separate mounts, each rank needs only its own page. A shortage
        # on the other host must still cause every rank to reject the save.
        with patch.object(checkpoint.socket, "gethostname", return_value=f"host-{rank}"):
            for remote_shortage in (False, True):
                pages = 0 if remote_shortage and rank == 1 else 1
                with patch.object(
                    checkpoint.os, "statvfs", return_value=SimpleNamespace(f_bavail=pages, f_frsize=4096)
                ):
                    if remote_shortage:
                        with pytest.raises(RuntimeError, match="on host-1"):
                            checkpoint._check_staging_shared_memory(state)
                    else:
                        checkpoint._check_staging_shared_memory(state)

        # A later distillation save releases the previous staging tree on each
        # rank. Hold rank 1's old page until rank 0 enters the capacity check.
        # Both new pages fit after release; a pre-barrier snapshot falsely fails.
        store = dist.FileStore(capacity_store_path, 2)
        if rank == 1:
            store.set("old_pages", "1")
        dist.barrier()
        barrier = dist.barrier

        def enter_capacity_check() -> None:
            if rank == 0:
                store.set("rank0_entered", "1")
            barrier()

        def available_capacity(path: str) -> SimpleNamespace:
            free_pages = 2 - int(store.get("old_pages"))
            if rank == 0:
                store.set("rank0_entered", "1")
            return SimpleNamespace(f_bavail=free_pages, f_frsize=4096)

        if rank == 1:
            store.wait(["rank0_entered"], timedelta(seconds=10))
            store.set("old_pages", "0")
        with (
            patch.object(checkpoint.dist, "barrier", side_effect=enter_capacity_check),
            patch.object(checkpoint.os, "statvfs", side_effect=available_capacity),
        ):
            checkpoint._check_staging_shared_memory(state)
    finally:
        dist.destroy_process_group()


def test_staging_capacity_aggregates_local_shards_and_fails_on_all_ranks(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _distributed_capacity_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "capacity")),
        nprocs=2,
        join=True,
    )
