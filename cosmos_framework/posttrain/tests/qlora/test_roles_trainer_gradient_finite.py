# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public gradient-finiteness edge cases, immutability and recovery semantics."""

from __future__ import annotations

import copy
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.roles.trainer import finite

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA kernels")
DEVICES = [
    pytest.param("cpu", marks=pytest.mark.CPU),
    pytest.param("cuda", marks=[pytest.mark.GPU, pytest.mark.gpus(1), CUDA]),
]


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).cpu().clone()


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_exact(a, b)
    else:
        assert left == right


def _edge_gradients(dtype, device, fault):
    info = torch.finfo(dtype)
    gradients = [
        torch.tensor([0.0, -0.0, info.max, -info.max, info.tiny, info.tiny * info.eps], dtype=dtype, device=device),
        torch.zeros(65537, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    ]
    if fault != "finite":
        gradients[1][-1] = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[fault]
    return gradients


def _check_read_only_gradients(gradients):
    before = [(_bytes(tensor), tensor._version, tensor.stride(), tensor.data_ptr()) for tensor in gradients]
    rng = torch.get_rng_state().clone()
    cuda_rng = {tensor.device: torch.cuda.get_rng_state(tensor.device) for tensor in gradients if tensor.is_cuda}
    result = finite.all_ranks_gradients_finite(
        [SimpleNamespace(grad=gradient) for gradient in gradients], require_gradients=True
    )
    for tensor, (values, version, stride, pointer) in zip(gradients, before, strict=True):
        assert torch.equal(_bytes(tensor), values)
        assert tensor._version == version
        assert tensor.stride() == stride and tensor.data_ptr() == pointer
    assert torch.equal(torch.get_rng_state(), rng)
    for device, state in cuda_rng.items():
        assert torch.equal(torch.cuda.get_rng_state(device), state)
    return result


@pytest.mark.L0
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ("finite", "nan", "positive_inf", "negative_inf"))
def test_gradient_finiteness_edges_and_immutability(device, dtype, fault):
    assert _check_read_only_gradients(_edge_gradients(dtype, device, fault)) is (fault == "finite")


@pytest.mark.L0
@pytest.mark.parametrize("device", DEVICES)
def test_mixed_dtype_view_and_subclass_gradients(device):
    class TensorSubclass(torch.Tensor):
        pass

    gradients = [torch.full((3,), 1.0, dtype=dtype, device=device) for dtype in DTYPES]
    gradients += [torch.empty(0, device=device), torch.ones(3, 5, device=device).t()]
    complex_gradient = torch.tensor([1.0 + 1.0j, 2.0 + 0.0j], device=device)
    gradients += [complex_gradient, torch._neg_view(torch.ones(4, device=device))]
    gradients += [torch.ones(3, device=device).as_subclass(TensorSubclass)]
    assert _check_read_only_gradients(gradients)
    complex_gradient[0] = complex(1.0, float("inf"))
    assert not _check_read_only_gradients(gradients)


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_missing_nonfloating_and_tensor_wrapper_semantics():
    assert finite.all_ranks_gradients_finite([])
    assert not finite.all_ranks_gradients_finite([], require_gradients=True)
    empty = torch.nn.Parameter(torch.empty(0))
    empty.grad = torch.empty_like(empty)
    assert finite.all_ranks_gradients_finite([empty], require_gradients=True)
    assert _check_read_only_gradients([torch.ones(2, dtype=torch.int64)])
    assert _check_read_only_gradients([torch.ones(2, dtype=torch.bool)])

    class LocalGradient:
        def to_local(self):
            return torch.tensor([float("nan")])

    assert not finite.all_ranks_gradients_finite([SimpleNamespace(grad=LocalGradient())])
    assert not finite.all_ranks_gradients_finite([SimpleNamespace(grad=None)], require_gradients=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_unsupported_layouts_delegate_without_conversion(monkeypatch):
    sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, float("nan")]), (3,))
    observed = []

    def existing(tensors):
        observed.extend(tensors)
        return {}

    monkeypatch.setattr(finite, "_finite_flags_by_device", existing)
    assert finite.all_ranks_gradients_finite([SimpleNamespace(grad=sparse)], require_gradients=True)
    assert len(observed) == 1 and observed[0] is sparse


def _consensus_worker(rank, rendezvous, output):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.full_like(parameter, float("nan") if rank else 1.0)
        assert not finite.all_ranks_gradients_finite([parameter], require_gradients=True)
        parameter.grad = torch.ones_like(parameter)
        assert not finite.all_ranks_gradients_finite([parameter] if rank == 0 else [], require_gradients=True)
        assert finite.all_ranks_gradients_finite([parameter], require_gradients=True)
        parameters = [parameter]
        if rank:
            misplaced = torch.nn.Parameter(torch.ones(1, device="meta"))
            misplaced.grad = torch.ones_like(misplaced)
            parameters.append(misplaced)
        with pytest.raises(RuntimeError, match="placement error, not a non-finite update"):
            finite.all_ranks_gradients_finite(parameters)
        Path(output, f"rank-{rank}").write_text("nonfinite/missing/finite/placement-consensus-passed")
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_two_rank_rejection_missing_and_placement_consensus(tmp_path):
    context = torch.multiprocessing.spawn(
        _consensus_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=False
    )
    deadline = time.monotonic() + 120
    while not context.join(timeout=max(deadline - time.monotonic(), 0)):
        if time.monotonic() >= deadline:
            for process in context.processes:
                process.kill()
            pytest.fail("two-rank consensus workers did not finish within 120 s")
    assert (tmp_path / "rank-0").read_text() == (tmp_path / "rank-1").read_text()


@pytest.mark.L0
@pytest.mark.parametrize("device", DEVICES)
def test_native_multiupdate_and_recovery(device, tmp_path):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    def run(resume):
        random.seed(62)
        np.random.seed(62)
        torch.manual_seed(62)
        if device == "cuda":
            torch.cuda.manual_seed_all(62)

        def build():
            model = SimpleNamespace(
                net=torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(0.1), torch.nn.Linear(16, 2)).to(
                    device
                )
            )
            optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=0.02, fused=device == "cuda")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

            def objective(policy, _batch):
                x = torch.randn(3, 8, device=device)
                return policy.net(x).square().mean(), {}

            trainer = LLMTrainer(
                model,
                Objective(objective, Reduction.MEAN),
                optimizer,
                lr_scheduler=scheduler,
                grad_accum=2,
                max_grad_norm=1.0,
                require_gradients=True,
                nonfinite_update="fail_stop",
            )
            return model, optimizer, scheduler, trainer

        model, optimizer, scheduler, trainer = build()
        for microstep in range(10):
            trainer.train([])
            if resume and microstep == 5:
                path = tmp_path / "resume"
                trainer.save(str(path))
                model, optimizer, scheduler, trainer = build()
                assert trainer.load(str(path)) == 3
        return {
            "model": copy.deepcopy(model.net.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "trainer": copy.deepcopy(trainer._trainer_state()),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }

    _assert_exact(run(True), run(False))
