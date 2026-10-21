# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Per-rank chunk extraction from consolidated checkpoint components (checkpoint/consolidated.py).

The resume of a DTensor-sharded trainer from an all-.pt consolidated checkpoint must cost each rank
its own shards and nothing more: the failure this guards against is a one-node resume that put
world-size copies of a 281 GB checkpoint on one host (855 GB of a 900 GB job cgroup) and was
killed on every attempt.
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.checkpoint.consolidated import fill_sharded_state_from_consolidated


def _tensor_bytes(state: Mapping[str, Any]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, Mapping):
            total += _tensor_bytes(value)
        elif torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


def _save_and_mmap(path: Any, state: dict[str, Any]) -> dict[str, Any]:
    torch.save(state, path)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_fill_copies_whole_tensors_replaces_controls_and_reports_bytes_for_an_unsharded_target(tmp_path: Any) -> None:
    """Plain-tensor targets (an unsharded rank, or a replicated placement) receive the whole saved
    tensor, cast to the live dtype; non-tensor leaves take the saved object; nested dicts recurse;
    the count is exactly the bytes of the live target."""
    torch.manual_seed(0)
    saved = {
        "state": {
            "layer.weight": {"exp_avg": torch.randn(6, 4), "exp_avg_sq": torch.rand(6, 4), "step": torch.tensor(7.0)},
            "layer.bias": {"exp_avg": torch.randn(6), "exp_avg_sq": torch.rand(6), "step": torch.tensor(7.0)},
        },
        "param_groups": [{"lr": 0.25, "betas": (0.9, 0.95), "params": ["layer.weight", "layer.bias"]}],
    }
    consolidated = _save_and_mmap(tmp_path / "optim.pt", saved)
    target = {
        "state": {
            "layer.weight": {"exp_avg": torch.zeros(6, 4), "exp_avg_sq": torch.zeros(6, 4), "step": torch.tensor(0.0)},
            "layer.bias": {
                "exp_avg": torch.zeros(6, dtype=torch.float64),
                "exp_avg_sq": torch.zeros(6),
                "step": torch.tensor(0.0),
            },
        },
        "param_groups": [{"lr": 1.0, "betas": (0.5, 0.5), "params": ["layer.weight", "layer.bias"]}],
    }
    expected_bytes = _tensor_bytes(target)

    copied = fill_sharded_state_from_consolidated(target, consolidated, context="unit", strict=True)

    assert copied == expected_bytes
    for name in ("layer.weight", "layer.bias"):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            live, disk = target["state"][name][key], saved["state"][name][key]
            assert live.dtype == (torch.float64 if (name, key) == ("layer.bias", "exp_avg") else disk.dtype)
            assert torch.equal(live.to(disk.dtype), disk), (name, key)
    assert target["param_groups"] == consolidated["param_groups"]
    assert target["param_groups"][0]["lr"] == 0.25

    # A model-side leaf set: strict=False tolerates saved entries the live side has no slot for,
    # the way `set_optimizer_state_dict` matches saved state by the live parameters.
    lenient_target = {"state": {"layer.weight": {"exp_avg": torch.zeros(6, 4)}}}
    assert fill_sharded_state_from_consolidated(lenient_target, consolidated, context="unit", strict=False) == 6 * 4 * 4
    assert torch.equal(lenient_target["state"]["layer.weight"]["exp_avg"], saved["state"]["layer.weight"]["exp_avg"])


@pytest.mark.L0
@pytest.mark.CPU
def test_fill_refuses_missing_slots_mismatched_shapes_and_extras_under_strict(tmp_path: Any) -> None:
    """A live tensor the file lacks is always an error (it would resume from zeros); a global shape
    mismatch is an error; a saved entry without a live slot is an error only under strict; a nested
    state on one side and a tensor on the other is a type error. Every message names the context."""
    consolidated = _save_and_mmap(
        tmp_path / "model.pt", {"a": torch.ones(2, 3), "b": torch.ones(4), "nested": {"c": torch.ones(1)}}
    )

    with pytest.raises(KeyError, match=r"ctx: consolidated state has no entry for 'missing'"):
        fill_sharded_state_from_consolidated(
            {"a": torch.zeros(2, 3), "missing": torch.zeros(1)}, consolidated, context="ctx", strict=False
        )
    with pytest.raises(ValueError, match=r"ctx: 'a' has global shape \(3, 2\) here but \(2, 3\) on disk"):
        fill_sharded_state_from_consolidated({"a": torch.zeros(3, 2)}, consolidated, context="ctx", strict=False)
    with pytest.raises(
        KeyError, match=r"ctx: consolidated state has entries this rank has no slot for: \['b', 'nested'\]"
    ):
        fill_sharded_state_from_consolidated({"a": torch.zeros(2, 3)}, consolidated, context="ctx", strict=True)
    with pytest.raises(TypeError, match=r"ctx: 'nested' is a tensor on this rank but dict on disk"):
        fill_sharded_state_from_consolidated({"nested": torch.zeros(1)}, consolidated, context="ctx", strict=False)
    with pytest.raises(TypeError, match=r"ctx: 'a' is a nested state on this rank but Tensor on disk"):
        fill_sharded_state_from_consolidated({"a": {"x": torch.zeros(1)}}, consolidated, context="ctx", strict=False)
    # Nothing above touched the target that was valid: strict=False with a complete target is clean.
    target = {"a": torch.zeros(2, 3), "b": torch.zeros(4), "nested": {"c": torch.zeros(1)}}
    assert fill_sharded_state_from_consolidated(target, consolidated, context="ctx", strict=True) == (6 + 4 + 1) * 4


def _two_rank_chunk_worker(rank: int, world_size: int, rendezvous_path: str, output_queue: Any, root: str) -> None:
    """Each rank owns `Shard(0)` chunks of two global tensors (one unevenly split) and one replicated
    tensor; rank 0 writes the consolidated file; both ranks fill their DTensor targets from an mmap
    of it and report what they materialized."""
    import os

    import torch.distributed as dist

    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

    try:
        init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=30.0)
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

        mesh = init_device_mesh("cpu", (world_size,))
        # Global tensors are the same on both ranks (seeded); rank 0 is the one that writes them.
        generator = torch.Generator().manual_seed(1234)
        uneven = torch.randn(5, 3, generator=generator)  # rows 0-2 -> rank 0, rows 3-4 -> rank 1
        even = torch.randn(4, 2, generator=generator)  # rows 0-1 -> rank 0, rows 2-3 -> rank 1
        replicated = torch.randn(3, generator=generator)
        path = os.path.join(root, "model.pt")
        if rank == 0:
            torch.save({"net": {"uneven": uneven, "even": even, "replicated": replicated}}, path)
        dist.barrier()

        def _shard(global_tensor: torch.Tensor) -> DTensor:
            chunk = torch.zeros_like(global_tensor.chunk(world_size, dim=0)[rank])
            return DTensor.from_local(chunk, mesh, [Shard(0)], shape=global_tensor.shape, stride=global_tensor.stride())

        target = {
            "uneven": _shard(uneven),
            "even": _shard(even),
            "replicated": DTensor.from_local(torch.zeros_like(replicated), mesh, [Replicate()]),
        }
        consolidated = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["net"]
        copied = fill_sharded_state_from_consolidated(target, consolidated, context=f"rank {rank}", strict=True)

        # A Partial placement is not stored state and is refused before anything is read.
        partial_target = {"uneven": DTensor.from_local(torch.zeros(5, 3), mesh, [Partial()])}
        try:
            fill_sharded_state_from_consolidated(partial_target, consolidated, context=f"rank {rank}", strict=True)
            partial_error = None
        except ValueError as error:
            partial_error = str(error)

        expected_uneven = uneven.chunk(world_size, dim=0)[rank]
        expected_even = even.chunk(world_size, dim=0)[rank]
        output_queue.put(
            (
                rank,
                {
                    "copied": copied,
                    "local_bytes": sum(
                        value.to_local().numel() * value.to_local().element_size() for value in target.values()
                    ),
                    "global_bytes": sum(value.numel() * value.element_size() for value in (uneven, even, replicated)),
                    "uneven_ok": torch.equal(target["uneven"].to_local(), expected_uneven),
                    "even_ok": torch.equal(target["even"].to_local(), expected_even),
                    "replicated_ok": torch.equal(target["replicated"].to_local(), replicated),
                    "uneven_local_shape": tuple(target["uneven"].to_local().shape),
                    "partial_error": partial_error,
                },
            )
        )
    except BaseException:  # noqa: BLE001 -- the parent must receive a traceback instead of hanging
        output_queue.put((rank, {"worker_error": traceback.format_exc()}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
def test_fill_copies_only_each_ranks_chunk_out_of_the_mmap_backed_file_on_two_gloo_ranks(tmp_path: Any) -> None:
    """Two real Gloo ranks with CPU DTensors: each rank's local shard receives exactly its own rows of
    the consolidated tensor (including the uneven 3+2 split), a replicated DTensor receives the whole
    tensor, the bytes a rank materializes are its local shards and not the global tensors, and a
    Partial placement is refused."""
    from cosmos_framework.posttrain.tests.helpers.distributed import run_spawned_ranks

    results = run_spawned_ranks(
        _two_rank_chunk_worker,
        world_size=2,
        rendezvous_path=tmp_path / "consolidated-fill.rendezvous",
        args=(str(tmp_path),),
        timeout_seconds=120.0,
        context="consolidated chunk fill",
    )
    for rank, result in results.items():
        assert result["uneven_ok"] and result["even_ok"] and result["replicated_ok"], (rank, result)
        assert result["uneven_local_shape"] == ((3, 3) if rank == 0 else (2, 3)), (rank, result)
        assert result["copied"] == result["local_bytes"], (rank, result)
        assert result["copied"] < result["global_bytes"], (rank, result)
        assert result["partial_error"] and "Partial placement" in result["partial_error"], (rank, result)
    # The two ranks' chunks plus one replica each account for the whole file: nothing was read twice
    # as a shard, nothing was skipped.
    replicated_bytes = 3 * 4
    assert sum(result["copied"] - replicated_bytes for result in results.values()) == (5 * 3 + 4 * 2) * 4
