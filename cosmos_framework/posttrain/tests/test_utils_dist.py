# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for torch.distributed utility edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.utils.dist import (
    broadcast_from_rank0,
    collective_device,
    raise_if_any_rank_failed,
    rank_identical,
    rank_max_count,
    rank_mean_count,
    run_rank_converged,
)


def _broadcast_failure_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:

        def fail_on_producer() -> str:
            raise OSError("checkpoint bucket unavailable")

        error_text: str | None = None
        try:
            broadcast_from_rank0(fail_on_producer, context="checkpoint download")
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None and "checkpoint bucket unavailable" in error for error in errors)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("backend", "cuda_index", "expected"),
    [
        ("gloo", None, torch.device("cpu")),
        ("nccl", 3, torch.device("cuda", 3)),
    ],
)
def test_collective_device_matches_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cuda_index: int | None,
    expected: torch.device,
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_backend", lambda: backend)
    if cuda_index is not None:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: cuda_index)

    assert collective_device() == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_raise_if_any_rank_failed_preserves_single_rank_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    with pytest.raises(ValueError, match="cursor mismatch"):
        raise_if_any_rank_failed(ValueError("cursor mismatch"), "preference source restore")

    device = torch.device("cpu")
    assert rank_mean_count(3, device) == 3.0
    assert rank_max_count(3, device) == 3
    assert run_rank_converged(lambda: "ready", "single-rank preparation") == "ready"

    def stop_rank() -> None:
        raise SystemExit(7)

    with pytest.raises(SystemExit) as stopped:
        run_rank_converged(stop_rank, "single-rank termination")
    assert stopped.value.code == 7


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_identical_rejects_divergent_plain_torchrun_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def _divergent(values: list[int | None], value: int) -> None:
        values[:] = [value, value + 1]

    monkeypatch.setattr(dist, "all_gather_object", _divergent)
    with pytest.raises(RuntimeError, match="differs across ranks"):
        rank_identical(7, "policy iteration")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("local", [True, False])
def test_rank_any_is_the_identity_outside_a_process_group(monkeypatch: pytest.MonkeyPatch, local: bool) -> None:
    """Single-rank runs must not touch a collective; the local verdict IS the answer."""
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils.dist import rank_any

    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("rank_any must not all-reduce outside a process group")

    monkeypatch.setattr(dist, "all_reduce", _explode)
    assert rank_any(local) is local


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("local", "peer_saw_it", "expected"),
    [
        (False, True, True),  # the case the helper exists for: a vacuous rank must not decide
        (False, False, False),
        (True, False, True),
    ],
)
def test_rank_any_reduces_with_max_so_one_rank_carries_the_verdict(
    monkeypatch: pytest.MonkeyPatch, local: bool, peer_saw_it: bool, expected: bool
) -> None:
    """MAX, not MIN and not rank-local: one rank's True must win over every other rank's False.

    `select_chunk` round-robins, so a rank legitimately holds an empty chunk and its verdict is
    vacuous. `all` would let that rank veto a real observation; a rank-local decision would let it
    certify one. Only `any` is correct, and it has to be a collective so every rank returns the same
    value -- a divergent answer here desynchronizes the next collective.
    """
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils.dist import rank_any

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    seen_op: list[object] = []

    def _fake_all_reduce(tensor: torch.Tensor, op: object = None) -> None:
        seen_op.append(op)
        peer = torch.tensor([1 if peer_saw_it else 0], dtype=tensor.dtype)
        torch.maximum(tensor, peer, out=tensor)

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

    assert rank_any(local) is expected
    assert seen_op == [dist.ReduceOp.MAX]


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_from_rank0_converges_producer_failure(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "broadcast-failure-rendezvous")
    mp.spawn(
        _broadcast_failure_worker,
        args=(world_size, rendezvous_path),
        nprocs=world_size,
        join=True,
    )
