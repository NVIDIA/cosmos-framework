# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real four-process Gloo count/norm and failure consensus, simulated shard hooks.

This does not exercise CUDA FSDP, native Pi0.5 backward, or physical host topology.
Those remain separate installed-package qualification requirements.
"""

import datetime
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from cosmos_framework.posttrain.roles.trainer.gradient_window import (
    NonFiniteGradientWindowError,
    OwnedGradientWindow,
    Owner,
)
from cosmos_framework.posttrain.utils.dist import run_rank_converged

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        parameter = torch.nn.Parameter(torch.tensor([float(rank)]))
        count_calls = []

        def count_sum(value):
            count_calls.append(value)
            count = torch.tensor(value, dtype=torch.int64)
            dist.all_reduce(count)
            return int(count)

        window = OwnedGradientWindow(
            {"group": [Owner("weight", parameter, (4,), 4, 1)]},
            microbatches=1,
            world_size=4,
            converge=run_rank_converged,
            count_sum=count_sum,
        )
        window.prepare_update(0)
        try:
            window.begin_native_micro(True if rank == 1 else 1)
        except RuntimeError as error:
            assert "native valid-mask count preparation" in str(error)
        else:
            raise AssertionError("one-rank invalid count did not converge")
        assert not count_calls
        window.abort()

        # A real incomplete logical tail: one valid example, three zero-weight
        # ranks. Only the explicit test hook/shard split is simulated.
        window.prepare_update(0)
        assert window.begin_native_micro(int(rank == 0)) == 1
        global_derivative = torch.tensor([7.0, 0.0, -2.0, 5.0], dtype=torch.float64) * int(rank == 0)
        dist.all_reduce(global_derivative)
        window.hook("group")(global_derivative[rank : rank + 1].clone())
        parameter.grad = torch.zeros_like(parameter)
        window.after_backward()
        window.finalize()
        assert parameter.grad.item() == global_derivative[rank].item()
        clipped = window.clip_global(1.0)
        assert clipped["norm"].item() == global_derivative.square().sum().sqrt().item()
        coefficient = (1.0 / (global_derivative.square().sum().sqrt() + 1e-6)).float()
        assert torch.equal(parameter.grad, global_derivative[rank : rank + 1].float() * coefficient)
        optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
        optimizer.step()
        window.finish_step()
        assert window.completed_updates == 1

        # One rank has an owner gradient but missed its capture callback. The
        # local failure must converge after backward before the next collective.
        window.prepare_update(0)
        window.begin_native_micro(1)
        if rank != 2:
            window.hook("group")(torch.ones(1, dtype=torch.float64))
        parameter.grad = torch.zeros_like(parameter)
        try:
            window.after_backward()
        except RuntimeError as error:
            assert "fp64 gradient-window shard ownership" in str(error)
        else:
            raise AssertionError("one-rank absent capture did not converge")
        assert window.phase == "idle" and not window.buffers and not window.pending
        assert parameter.grad is None
        # Numerical rejection must be typed and converged, unlike the preceding
        # owner failure. One bad shard cannot leave peers advancing the update.
        window.prepare_update(0)
        window.begin_native_micro(1)
        window.hook("group")(torch.tensor([float("inf") if rank == 1 else 1.0], dtype=torch.float64))
        parameter.grad = torch.zeros_like(parameter)
        try:
            window.after_backward()
        except NonFiniteGradientWindowError:
            pass
        else:
            raise AssertionError("one-rank nonfinite derivative did not converge")
        assert window.phase == "idle" and window.failure is None and parameter.grad is None
        window.prepare_update(0)
        window.begin_native_micro(1)
        window.hook("group")(torch.ones(1, dtype=torch.float64))
        parameter.grad = torch.zeros_like(parameter)
        window.after_backward()
        if rank == 3:
            window.buffers["weight"].fill_(1e300)
        try:
            window.finalize()
        except NonFiniteGradientWindowError:
            pass
        else:
            raise AssertionError("one-rank final cast overflow did not converge")
        assert window.phase == "idle" and window.failure is None and parameter.grad is None
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "invalid_count_before_sum": True,
                    "zero_count_tail": True,
                    "missing_capture_converged": True,
                    "nonfinite_gradient_converged": True,
                    "nonfinite_normalization_converged": True,
                    "completed_updates": window.completed_updates,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_four_process_count_tail_norm_and_failure_consensus(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4, join=True)
    rows = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("rank-*.json"))]
    assert [row["rank"] for row in rows] == list(range(4))
    assert all(
        row["completed_updates"] == 1
        and all(
            row[k]
            for k in (
                "invalid_count_before_sum",
                "zero_count_tail",
                "missing_capture_converged",
                "nonfinite_gradient_converged",
                "nonfinite_normalization_converged",
            )
        )
        for row in rows
    )
