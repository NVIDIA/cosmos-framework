# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for inference replica work partitioning: the planner and the executor.

``TestPlanPartition`` is pure Python. ``TestReplicaPartitionerRun``
drives :meth:`ReplicaPartitioner.run` across a real CPU/gloo process group of 4 ranks, the
same ``mp.spawn`` + file-init pattern as ``vae_load_balance_test.py``, so it needs no GPU.
"""

from __future__ import annotations

import os
import tempfile
import traceback
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.model.generator.mot.replica_partition import (
    ReplicaPartitioner,
    WorkUnit,
    plan_partition,
    run_units_locally,
)
from cosmos_framework.utils.generator.multiview import (
    MultiviewDecode,
    decode_multiview_latent_per_view,
    load_multiview_media_units,
)


@pytest.mark.L0
@pytest.mark.CPU
class TestPlanPartition:
    def test_eleven_camera_views_and_a_lidar_clip_split_three_three_three_two(self) -> None:
        """The joint decode's shape: 11 camera units followed by one LiDAR unit."""
        owners = plan_partition(12, 4)
        assert owners == [0, 1, 2, 3] * 3
        assert [owners[:11].count(rank) for rank in range(4)] == [3, 3, 3, 2]
        # The LiDAR unit lands on the rank that got only two camera views.
        assert owners[:11].count(owners[11]) == 2

    def test_fewer_units_than_ranks_leave_later_ranks_idle(self) -> None:
        assert plan_partition(2, 4) == [0, 1]

    def test_encode_places_two_lidar_units_after_twenty_two_camera_units(self) -> None:
        owners = plan_partition(24, 4)
        assert [owners[:22].count(rank) for rank in range(4)] == [6, 6, 5, 5]
        assert owners[22:] == [2, 3]

    def test_single_rank_owns_everything(self) -> None:
        assert plan_partition(3, 1) == [0, 0, 0]

    def test_no_units(self) -> None:
        assert plan_partition(0, 4) == []

    def test_rejects_an_empty_group(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            plan_partition(1, 0)

    def test_rejects_negative_unit_count(self) -> None:
        with pytest.raises(ValueError, match="num_units"):
            plan_partition(-1, 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_run_units_locally_streams_or_returns() -> None:
    units = [WorkUnit(run=lambda i=i: torch.tensor([i])) for i in range(3)]  # each run returns [1]
    assert [int(t.item()) for t in run_units_locally(units)] == [0, 1, 2]
    seen: list[tuple[int, int]] = []
    results = run_units_locally(units, on_collected=lambda index, tensor: seen.append((index, int(tensor.item()))))
    assert results == [None, None, None]
    assert seen == [(0, 0), (1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# ReplicaPartitioner.run over a real 4-rank gloo group.
# ---------------------------------------------------------------------------

_WORLD_SIZE = 4
# Unit i is a deterministic function of i alone, with a per-unit shape and dtype, so a result
# that arrives on the wrong rank or in the wrong slot, or is sized from the wrong metadata,
# shows up as a mismatch. Uneven unit counts exercise round-robin ownership.
_NUM_UNITS = 9


def _unit_value(index: int) -> torch.Tensor:  # returns [1,3+index]
    length = 3 + index
    dtype = torch.float32 if index % 2 == 0 else torch.int64
    return (torch.arange(length) * (index + 1)).to(dtype).reshape(1, length)  # [1,length]


def _check_multiview_media_load(partitioner: ReplicaPartitioner) -> None:
    ran_here: list[int] = []
    fail = False

    def load(index: int) -> torch.Tensor:  # returns [3,T,2,2]
        ran_here.append(index)
        if fail and index == 5:
            raise ValueError("bad camera media")
        return torch.arange(12 * (index + 1)).reshape(3, index + 1, 2, 2).to(torch.uint8)  # [3,T,2,2]

    for count, keep_on_host in ((7, True), (2, False), (0, True)):
        units = [WorkUnit(run=partial(load, index)) for index in range(count)]
        ran_here.clear()
        expected = load_multiview_media_units(units, keep_on_host=keep_on_host)  # list[[3,T,2,2]]
        assert ran_here == list(range(count))
        ran_here.clear()
        actual = load_multiview_media_units(
            units, partitioner=partitioner, keep_on_host=keep_on_host
        )  # list[[3,T,2,2]]
        assert ran_here == list(range(partitioner.group_rank, count, partitioner.group_size))
        assert len(actual) == count
        for clip, local_clip in zip(actual, expected, strict=True):
            assert clip.dtype == torch.uint8
            assert clip.device.type == "cpu"
            assert torch.equal(clip, local_clip)

    fail = True
    units = [WorkUnit(run=partial(load, index)) for index in range(7)]
    expected_error = ValueError if partitioner.group_rank == 1 else RuntimeError
    with pytest.raises(expected_error, match="bad camera media"):
        load_multiview_media_units(units, partitioner=partitioner)


def _check_multiview_decode(partitioner: ReplicaPartitioner) -> None:
    for n_views, batched, assemble_on_cpu in ((7, True, True), (7, False, False), (2, False, True)):
        latent = torch.arange(3 * 2 * n_views, dtype=torch.float32).reshape(3, 2 * n_views, 1, 1)  # [3,V*T,1,1]
        if batched:
            latent = latent.unsqueeze(0)  # [1,3,V*T,1,1]
        temporal_dim = latent.ndim - 3
        ran_here: list[int] = []

        def decode(view_latent: torch.Tensor) -> torch.Tensor:  # [B,3,T,1,1] or [3,T,1,1]
            ran_here.append(int(view_latent.flatten()[0].item()) // 2)
            return view_latent.repeat_interleave(2, dim=temporal_dim) + 1  # [B,3,4,1,1] or [3,4,1,1]

        expected = decode_multiview_latent_per_view(
            decode, latent, n_views, 4, assemble_on_cpu=assemble_on_cpu
        )  # [B,3,V*4,1,1] or [3,V*4,1,1]
        ran_here.clear()
        decoded = decode_multiview_latent_per_view(
            decode, latent, n_views, 4, assemble_on_cpu=assemble_on_cpu, partitioner=partitioner
        )  # [B,3,V*4,1,1] or [3,V*4,1,1]
        assert ran_here == list(range(partitioner.group_rank, n_views, partitioner.group_size))
        # Every rank needs the full output, including other cameras' tails, for the next caption chunk.
        assert torch.equal(decoded, expected)
        assert decoded.device.type == "cpu"

    latent = torch.arange(14, dtype=torch.float32).reshape(1, 1, 14, 1, 1)  # [1,1,V*T,1,1]

    def failing_decode(view_latent: torch.Tensor) -> torch.Tensor:  # [1,1,T,1,1] -> [1,1,T,1,1]
        if int(view_latent.flatten()[0].item()) // 2 == 5:
            raise ValueError("bad camera decode")
        return view_latent + 1  # [1,1,T,1,1]

    expected_error = ValueError if partitioner.group_rank == 1 else RuntimeError
    with pytest.raises(expected_error, match="bad camera decode"):
        decode_multiview_latent_per_view(failing_decode, latent, 7, 2, assemble_on_cpu=True, partitioner=partitioner)


def _check_joint_multiview_decode(partitioner: ReplicaPartitioner) -> None:
    n_views = 7
    latent = torch.linspace(-2, 2, 2 * n_views).reshape(1, 1, 2 * n_views, 1, 1)  # [1,1,V*T,1,1]
    lidar = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 2, 2)  # [1,1,3,2,2]

    def decode(view_latent: torch.Tensor) -> torch.Tensor:  # [1,1,2,1,1] -> [1,1,3,1,1]
        # Joint inference clamps decoder pixels and drops the padded final frame before collection.
        return view_latent.repeat_interleave(2, dim=2).clamp(-1, 1)[:, :, :3]  # [1,1,3,1,1]

    expected_views = [decode(latent[:, :, 2 * view : 2 * view + 2]) for view in range(n_views)]  # list[[1,1,3,1,1]]
    expected_camera = torch.cat(expected_views, dim=2)  # [1,1,V*3,1,1]
    expected_lidar = lidar + 10  # [1,1,3,2,2]
    camera = MultiviewDecode(decode, latent, n_views, 3, assemble_on_cpu=True)
    local: dict[int, torch.Tensor] = {}
    ran_here: list[int] = []

    def keep(index: int, unit: WorkUnit) -> WorkUnit:
        def run() -> torch.Tensor:  # returns [1,1,3,H,W]
            ran_here.append(index)
            local[index] = unit.run()  # [1,1,3,H,W]
            return local[index]

        return WorkUnit(run=run)

    units = [keep(index, unit) for index, unit in enumerate(camera.units)]
    units.append(keep(n_views, WorkUnit(run=lambda: lidar + 10)))  # LiDAR returns [1,1,3,2,2].
    received: list[int] = []
    output_lidar: torch.Tensor | None = None

    def collect(index: int, pixels: torch.Tensor) -> None:  # pixels: [1,1,3,H,W]
        nonlocal output_lidar
        received.append(index)
        if index == n_views:
            output_lidar = pixels.cpu()  # [1,1,3,2,2]
        else:
            camera.collect(index, pixels)

    results = partitioner.run(units, collect_on=0, on_collected=collect, label="joint decode")
    owned = list(range(partitioner.group_rank, n_views + 1, partitioner.group_size))
    assert ran_here == owned
    assert list(local) == owned
    assert results == [None] * (n_views + 1)
    for index, pixels in local.items():
        assert torch.equal(pixels, expected_lidar if index == n_views else expected_views[index])
    output_camera = camera.result()  # [1,1,V*3,1,1] or None
    if partitioner.group_rank == 0:
        assert received == list(range(n_views + 1))
        assert output_camera is not None and output_camera.device.type == "cpu"
        assert output_lidar is not None and output_lidar.device.type == "cpu"
        assert torch.equal(output_camera, expected_camera)
        assert torch.equal(output_lidar, expected_lidar)
    else:
        assert received == []
        assert output_camera is None
        assert output_lidar is None

    def mismatched_decode(view_latent: torch.Tensor) -> torch.Tensor:  # [1,1,2,1,1] -> [1,1,3,H,1]
        pixels = decode(view_latent)  # [1,1,3,1,1]
        if torch.equal(view_latent, latent[:, :, 10:12]):
            return pixels.expand(-1, -1, -1, 2, -1)  # [1,1,3,2,1]
        return pixels

    camera = MultiviewDecode(mismatched_decode, latent, n_views, 3, assemble_on_cpu=True)
    # A spatial mismatch fails during assembly on the collector; every peer must still exit.
    with pytest.raises(RuntimeError) as error:
        partitioner.run(camera.units, collect_on=0, on_collected=camera.collect, label="joint decode")
    if partitioner.group_rank != 0:
        assert "rank 0 failed while collecting" in str(error.value)


def _partition_worker(rank: int, init_file: str, mode: str, result_queue: Any) -> None:
    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE)
        partitioner = ReplicaPartitioner(group=dist.group.WORLD, group_rank=rank, group_size=_WORLD_SIZE, device="cpu")
        ran_here: list[int] = []

        def make_run(index: int) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:  # returns [1,3+index]
                ran_here.append(index)
                if mode == "error" and index == 5:
                    raise ValueError("corrupt clip")
                return _unit_value(index)  # [1,3+index]

            return run

        units = [WorkUnit(run=make_run(i)) for i in range(_NUM_UNITS)]
        report: dict = {"ran_here": ran_here}
        if mode == "multiview_media_load":
            _check_multiview_media_load(partitioner)
        elif mode == "multiview_decode":
            _check_multiview_decode(partitioner)
        elif mode == "joint_multiview_decode":
            _check_joint_multiview_decode(partitioner)
        elif mode == "broadcast":
            results = partitioner.run(units, label="test")
            report["results"] = [None if t is None else t.tolist() for t in results]
        elif mode == "collect":
            streamed: list[tuple[int, list]] = []
            results = partitioner.run(
                units, collect_on=1, on_collected=lambda i, t: streamed.append((i, t.tolist())), label="test"
            )
            report["results"] = results
            report["streamed"] = streamed
        elif mode == "error":
            try:
                partitioner.run(units, label="test")
                report["raised"] = None
            except Exception as error:  # The test asserts on what every rank raised.
                report["raised"] = (type(error).__name__, str(error))
        elif mode == "collect_callback_error":
            received: list[int] = []

            def fail_on_second(index: int, _tensor: torch.Tensor) -> None:
                received.append(index)
                if len(received) == 2:
                    raise MemoryError("host buffer")

            try:
                partitioner.run(units, collect_on=0, on_collected=fail_on_second, label="test")
                report["raised"] = None
            except Exception as error:  # The test asserts on what every rank raised.
                report["raised"] = (type(error).__name__, str(error))
            report["received"] = received
        result_queue.put((rank, "ok", report))
    except Exception:
        result_queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_partition(mode: str, timeout: float = 120.0) -> dict[int, dict]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [
            ctx.Process(target=_partition_worker, args=(rank, init_file, mode, result_queue))
            for rank in range(_WORLD_SIZE)
        ]
        for proc in procs:
            proc.start()
        # Drain every rank's report before asserting: an unread report can block its sender's exit.
        reports = [result_queue.get(timeout=timeout) for _ in range(_WORLD_SIZE)]
        for proc in procs:
            proc.join(timeout=30)
    for rank, status, payload in reports:
        assert status == "ok", f"rank {rank} failed:\n{payload}"
    return {rank: payload for rank, _, payload in reports}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
@pytest.mark.no_xdist
class TestReplicaPartitionerRun:
    def test_multiview_media_load_shares_ordered_camera_clips_and_propagates_errors(self) -> None:
        _run_partition("multiview_media_load")

    def test_multiview_decode_shares_cameras_and_preserves_outputs_on_every_rank(self) -> None:
        _run_partition("multiview_decode")

    def test_joint_decode_collects_cameras_and_lidar_together_and_keeps_owned_outputs(self) -> None:
        _run_partition("joint_multiview_decode")

    def test_every_unit_runs_once_and_every_rank_receives_every_result(self) -> None:
        reports = _run_partition("broadcast")
        # Each rank runs exactly the units the round-robin plan gives it.
        owners = plan_partition(_NUM_UNITS, _WORLD_SIZE)
        for rank, report in reports.items():
            assert report["ran_here"] == [index for index, owner in enumerate(owners) if owner == rank]
        expected = [_unit_value(i).tolist() for i in range(_NUM_UNITS)]
        for report in reports.values():
            assert report["results"] == expected

    def test_collecting_delivers_to_one_rank_in_unit_order(self) -> None:
        reports = _run_partition("collect")
        expected = [(i, _unit_value(i).tolist()) for i in range(_NUM_UNITS)]
        assert reports[1]["streamed"] == expected
        for rank, report in reports.items():
            assert report["results"] == [None] * _NUM_UNITS
            if rank != 1:
                assert report["streamed"] == []

    def test_a_failing_collect_callback_raises_on_every_rank_instead_of_hanging(self) -> None:
        """The collector's callback fails mid-delivery; peers' sends are still matched, then all raise."""
        reports = _run_partition("collect_callback_error")
        assert reports[0]["raised"] == ("MemoryError", "host buffer")
        # Callbacks stop at the failure, but no transfer is left unmatched.
        assert len(reports[0]["received"]) == 2
        for rank in (1, 2, 3):
            name, message = reports[rank]["raised"]
            assert name == "RuntimeError"
            assert "rank 0" in message and "host buffer" in message

    def test_a_failing_unit_raises_on_every_rank_instead_of_hanging(self) -> None:
        reports = _run_partition("error")
        owner = next(rank for rank, report in reports.items() if 5 in report["ran_here"])
        assert reports[owner]["raised"] == ("ValueError", "corrupt clip")
        for rank, report in reports.items():
            if rank != owner:
                name, message = report["raised"]
                assert name == "RuntimeError"
                assert f"rank {owner}" in message and "corrupt clip" in message


@pytest.mark.L0
@pytest.mark.CPU
class TestFromParallelDims:
    @staticmethod
    def _dims(**overrides) -> types.SimpleNamespace:
        dims = {"enable_inference_mode": True, "lb_enabled": True, "lb_size": 4, "cp_size": 2, "cfgp_size": 2}
        return types.SimpleNamespace(**(dims | overrides))

    def test_no_partitioner_outside_inference_or_without_an_lb_group(self) -> None:
        assert ReplicaPartitioner.from_parallel_dims(None, device="cpu") is None
        assert ReplicaPartitioner.from_parallel_dims(self._dims(enable_inference_mode=False), device="cpu") is None
        assert ReplicaPartitioner.from_parallel_dims(self._dims(lb_enabled=False), device="cpu") is None

    def test_an_lb_group_that_is_not_one_replica_shares_nothing(self) -> None:
        """E.g. an inference loader that kept a training experiment's lb=64 with cp=cfgp=1."""
        dims = self._dims(lb_size=64, cp_size=1, cfgp_size=1)
        assert ReplicaPartitioner.from_parallel_dims(dims, device="cpu") is None
