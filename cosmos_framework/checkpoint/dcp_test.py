# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import builtins
import pickle
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.checkpoint import dcp as checkpoint
from cosmos_framework.checkpoint.dcp_distill import DistributedCheckpointer as DistillCheckpointer
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, pad_packed_bytes

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


class _CallOnLoad:
    def __init__(self, func: Any, *args: Any) -> None:
        self.func = func
        self.args = args

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return self.func, self.args


def _marker_payload(marker: Path) -> _CallOnLoad:
    return _CallOnLoad(builtins.exec, f"open({str(marker)!r}, 'w').write('loaded')")


@pytest.fixture
def _single_rank_cpu() -> Any:
    """Resume as rank 0 of a single process, with the CUDA generator stubbed out for CPU-only machines."""
    with (
        patch.object(checkpoint.dist, "get_rank", return_value=0),
        patch.object(checkpoint.dist, "barrier"),
        patch("torch.cuda.get_rng_state", return_value=torch.zeros(16, dtype=torch.uint8)),
        patch("torch.cuda.set_rng_state"),
    ):
        yield


def _resume_handler(checkpointer_type: type, root: Path, keys: list[str], callbacks: Any = None) -> Any:
    handler = object.__new__(checkpointer_type)
    handler.callbacks = callbacks
    handler.config_checkpoint = SimpleNamespace(
        strict_resume=True, keys_to_skip_loading=[], dcp_load_dedup=False, load_ema_to_reg=False
    )
    handler.save_s3_backend_key = None
    source = SimpleNamespace(path=str(root), warm_start=False, backend_key=None, uses_object_store=False)
    handler.keys_to_resume_during_load = MagicMock(return_value=(keys, source))
    handler._check_checkpoint_exists = MagicMock()
    return handler


def _save_trainer(root: Path, rng_state: dict[str, torch.Tensor]) -> None:
    trainer_state = {"grad_scaler": {"scale": 1024.0, "_growth_tracker": 2}, "iteration": 7, "rng_state_0": rng_state}
    dcp.save(trainer_state, checkpoint_id=root / "trainer", no_dist=True)


def _resume_trainer(checkpointer_type: type, root: Path) -> tuple[int, MagicMock]:
    grad_scaler = MagicMock()
    grad_scaler.state_dict.return_value = {"scale": 0.0, "_growth_tracker": 0}
    handler = _resume_handler(checkpointer_type, root, ["trainer"])
    return handler.load(MagicMock(), grad_scaler=grad_scaler), grad_scaler


def _draws() -> list[tuple[float, float, float]]:
    return [(torch.randn(1).item(), float(np.random.rand()), random.random()) for _ in range(3)]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("_single_rank_cpu")
@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
def test_trainer_resume_restores_rng_state(checkpointer_type: type, tmp_path: Path) -> None:
    _save_trainer(tmp_path, get_rand_state_dict())
    expected = _draws()

    iteration, grad_scaler = _resume_trainer(checkpointer_type, tmp_path)

    assert iteration == 7
    grad_scaler.load_state_dict.assert_called_once_with({"scale": 1024.0, "_growth_tracker": 2})
    assert _draws() == expected


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("_single_rank_cpu")
@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
@pytest.mark.parametrize("prefix", ["numpy", "random"])
def test_trainer_resume_rejects_rng_payload(checkpointer_type: type, prefix: str, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    payload = pickle.dumps(_marker_payload(marker))
    rng_state = get_rand_state_dict()
    rng_state[f"{prefix}_packed_len"] = torch.tensor(len(payload), dtype=torch.long)
    rng_state[f"{prefix}_packed_bytes"] = torch.frombuffer(pad_packed_bytes(payload), dtype=torch.uint8)
    _save_trainer(tmp_path, rng_state)

    with pytest.raises(ValueError, match=r"Refusing to load checkpoint state 'rng_state_0' from '.*trainer'"):
        _resume_trainer(checkpointer_type, tmp_path)

    assert not marker.exists()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("_single_rank_cpu")
@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
def test_trainer_resume_rejects_value_payload(checkpointer_type: type, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    trainer_state = {"grad_scaler": {"scale": 1024.0, "_growth_tracker": 2}, "iteration": _marker_payload(marker)}
    dcp.save(trainer_state, checkpoint_id=tmp_path / "trainer", no_dist=True)

    with pytest.raises(CheckpointException) as excinfo:
        _resume_trainer(checkpointer_type, tmp_path)

    error = next(iter(excinfo.value.failures.values()))[0]
    assert "Refusing to load checkpoint value 'iteration'" in str(error)

    assert not marker.exists()


def _dataloader_callbacks() -> tuple[MagicMock, MagicMock]:
    state_callback = MagicMock(checkpoint_component="dataloader")
    state_callback.has_checkpoint_state.return_value = True
    callbacks = MagicMock()
    callbacks._callbacks = [state_callback]
    return callbacks, state_callback


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("_single_rank_cpu")
@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
def test_dataloader_resume_round_trips_saved_state(checkpointer_type: type, tmp_path: Path) -> None:
    callbacks, state_callback = _dataloader_callbacks()
    handler = _resume_handler(checkpointer_type, tmp_path, ["dataloader"], callbacks)
    state = {0: {"epoch": 2, "index": 41}, 3: {"epoch": 1, "index": 0}, "val": {"format": "lance", "workers": {}}}
    (tmp_path / "dataloader").mkdir()
    handler._save_as_pkl(state, str(tmp_path / "dataloader"))

    handler.load(MagicMock())

    state_callback.load_state_dict.assert_called_once_with(state)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.usefixtures("_single_rank_cpu")
@pytest.mark.parametrize("checkpointer_type", [checkpoint.DistributedCheckpointer, DistillCheckpointer])
def test_dataloader_resume_rejects_payload(checkpointer_type: type, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    callbacks, state_callback = _dataloader_callbacks()
    handler = _resume_handler(checkpointer_type, tmp_path, ["dataloader"], callbacks)
    (tmp_path / "dataloader").mkdir()
    (tmp_path / "dataloader" / "rank_0.pkl").write_bytes(pickle.dumps({0: _marker_payload(marker)}, protocol=2))

    with pytest.raises(ValueError, match=r"Refusing to load checkpoint state 'dataloader' from '.*rank_0.pkl'"):
        handler.load(MagicMock())

    assert not marker.exists()
    state_callback.load_state_dict.assert_not_called()
