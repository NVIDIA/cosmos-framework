# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check CP output gathering against an unsharded, globally averaged objective."""

import copy
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from cosmos_framework.model.generator.mot import context_parallel_utils
from cosmos_framework.model.generator.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
    get_context_parallel_sharded_sequence,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims


class _GatheredPrediction(nn.Module):
    """Keep trainable projections before sharding, inside CP, and after gathering."""

    parallel_dims: ParallelDims | None
    encoder: nn.Linear
    trunk: nn.Linear
    decoder: nn.Linear
    text_length: int
    correct_cp_gradients: bool | None

    def __init__(self, parallel_dims: ParallelDims | None, text_length: int) -> None:
        super().__init__()
        self.parallel_dims = parallel_dims
        self.text_length = text_length
        self.correct_cp_gradients = None
        self.encoder = nn.Linear(5, 7, dtype=torch.float64)
        self.trunk = nn.Linear(7, 7, dtype=torch.float64)
        self.decoder = nn.Linear(7, 3, dtype=torch.float64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [N,5], returns: [N,3]
        encoded = self.encoder(inputs).tanh()  # [N,7]
        text_indexes = torch.arange(self.text_length)  # [N_und]
        gen_indexes = torch.arange(self.text_length, len(inputs))  # [N_gen]
        packed = sequence_pack_from_packed_sequence(
            packed_sequence=encoded,
            attn_modes=["causal", "full"],
            split_lens=[self.text_length, len(inputs) - self.text_length],
            sample_lens=[len(inputs)],
            packed_und_token_indexes=text_indexes,
            packed_gen_token_indexes=gen_indexes,
            full_seq_alignment=4,
            causal_seq_alignment=4,
        )
        positions = torch.arange(len(inputs))  # [N]
        local_pack, _ = get_context_parallel_sharded_sequence(packed, positions, self.parallel_dims)
        text = self.trunk(local_pack["causal_seq"]).tanh()  # [N_und_local,7]
        gen = self.trunk(local_pack["full_only_seq"]).tanh()  # [N_gen_local,7]
        outputs = from_mode_splits(text, gen, local_pack)
        correction = self.correct_cp_gradients
        gather_kwargs = {} if correction is None else {"correct_cp_gradients": correction}
        gathered = get_context_parallel_last_hidden_state(outputs, self.parallel_dims, **gather_kwargs)  # [N,7]
        return self.decoder(gathered)  # [N,3]


def _compare_gradients(
    rank: int,
    world_size: int,
    cp_size: int,
    text_length: int,
    correct_cp_gradients: bool | None,
    rendezvous: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        dims = ParallelDims(world_size=world_size, dp_shard=world_size, dp_replicate=1, cp=cp_size)
        dims.build_meshes("cpu")
        torch.manual_seed(482)
        reference = _GatheredPrediction(None, text_length)
        actual = copy.deepcopy(reference)
        actual.parallel_dims = dims
        actual.correct_cp_gradients = correct_cp_gradients
        distributed_model = DistributedDataParallel(actual)
        group_count = world_size // cp_size
        inputs = [torch.randn(text_length + 9, 5, dtype=torch.float64) for _ in range(group_count)]  # list[[N,5]]
        expected_outputs = [reference(value) for value in inputs]  # list[[N,3]]
        expected_loss = torch.stack([value.square().mean() for value in expected_outputs]).mean()  # []
        actual_output = distributed_model(inputs[rank // cp_size])  # [N,3]
        actual_loss = actual_output.square().mean()  # []
        torch.testing.assert_close(actual_output, expected_outputs[rank // cp_size], atol=1e-12, rtol=1e-12)
        actual_loss.backward()
        expected_loss.backward()
        for name, parameter in actual.named_parameters():
            expected = dict(reference.named_parameters())[name]
            assert parameter.grad is not None and expected.grad is not None, name
            if not correct_cp_gradients and not name.startswith("decoder."):
                # Legacy gathers divide only upstream gradients; prediction-head gradients stay unchanged.
                expected.grad.div_(cp_size)  # [*parameter_shape]
            torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-12, rtol=1e-10, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("cp_size", "text_length"), [(1, 3), (2, 3), (2, 0)])
@pytest.mark.parametrize("correct_cp_gradients", [None, False, True], ids=["default", "disabled", "enabled"])
def test_context_parallel_output_parameter_gradients(
    cp_size: int, text_length: int, correct_cp_gradients: bool | None, tmp_path: Path
) -> None:
    """Check legacy and corrected CP1/CP2 gradients with real collectives, DP samples, and padding."""
    mp.spawn(
        _compare_gradients,
        args=(4, cp_size, text_length, correct_cp_gradients, (tmp_path / "rendezvous").as_uri()),
        nprocs=4,
        join=True,
    )


def _check_interleaved_output(
    dims: ParallelDims,
    split_lens: list[int],
    *,
    correct_cp_gradients: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    check_memory: bool = False,
    bounded_memory: bool = True,
) -> None:
    """Check a two-sample permutation, independent rank cotangents, and optional allocation bounds."""
    width = 64
    length = sum(split_lens)
    original = (
        torch.arange(length * width, device=device, dtype=torch.float32).reshape(length, width).to(dtype)
    )  # [N,D]
    positions = torch.arange(length, device=device)  # [N]
    und_indices = torch.cat(
        (positions[: split_lens[0]], positions[sum(split_lens[:2]) : sum(split_lens[:3])])
    )  # [N_und]
    gen_indices = torch.cat(
        (positions[split_lens[0] : sum(split_lens[:2])], positions[sum(split_lens[:3]) :])
    )  # [N_gen]
    packed = sequence_pack_from_packed_sequence(
        packed_sequence=original,
        attn_modes=["causal", "full", "causal", "full"],
        split_lens=split_lens,
        sample_lens=[sum(split_lens[:2]), sum(split_lens[2:])],
        packed_und_token_indexes=und_indices,
        packed_gen_token_indexes=gen_indices,
        full_seq_alignment=dims.cp_size,
        causal_seq_alignment=dims.cp_size,
    )
    local_pack, _ = get_context_parallel_sharded_sequence(packed, positions, dims)
    und = local_pack["causal_seq"].detach().clone().requires_grad_(True)  # [N_und_local,D]
    gen = local_pack["full_only_seq"].detach().clone().requires_grad_(True)  # [N_gen_local,D]
    outputs = from_mode_splits(und, gen, local_pack)
    # A noncontiguous, rank-dependent cotangent catches wrong reduction, rank and token order.
    cotangent = torch.arange(length * width, device=device, dtype=torch.float32).remainder_(17)  # [N*D]
    cotangent = cotangent.reshape(width, length).T.to(dtype)  # [N,D]
    cotangent.mul_(dims.cp_rank + 1)  # [N,D]
    if check_memory:
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    actual = get_context_parallel_last_hidden_state(
        outputs, dims, correct_cp_gradients=correct_cp_gradients, bounded_memory=bounded_memory
    )  # [N,D]
    if check_memory:
        torch.cuda.synchronize()
        # One full result plus bounded communication scratch; a second full result violates this.
        forward_extra = torch.cuda.max_memory_allocated() - baseline
        assert (
            forward_extra <= actual.numel() * actual.element_size() + 3 * context_parallel_utils._CP_OUTPUT_GATHER_BYTES
        )
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    if check_memory:
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    gradients = torch.autograd.grad(actual, (und, gen), cotangent)  # tuple[[N_und_local,D],[N_gen_local,D]]
    if check_memory:
        torch.cuda.synchronize()
        backward_extra = torch.cuda.max_memory_allocated() - baseline
        gradient_bytes = sum(value.numel() * value.element_size() for value in gradients)
        assert backward_extra <= gradient_bytes + 3 * context_parallel_utils._CP_OUTPUT_GATHER_BYTES
    factor = dims.cp_size * (dims.cp_size + 1) // 2 if correct_cp_gradients else dims.cp_rank + 1
    expected_full = cotangent / (dims.cp_rank + 1) * factor  # [N,D]
    for gradient, indices in zip(gradients, (und_indices, gen_indices), strict=True):
        local_rows = gradient.shape[0]
        offset = dims.cp_rank * local_rows
        count = max(0, min(local_rows, indices.numel() - offset))
        expected = torch.zeros_like(gradient)  # [N_local,D]
        expected[:count] = expected_full[indices[offset : offset + count]]  # [count,D]
        torch.testing.assert_close(gradient, expected, atol=0, rtol=0)


def _compare_interleaved_gradients(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        dims = ParallelDims(world_size=4, dp_shard=4, dp_replicate=1, cp=2)
        dims.build_meshes("cpu")
        context_parallel_utils._CP_OUTPUT_GATHER_BYTES = 1024
        for bounded_memory in (False, True):
            for correction in (False, True):
                for lengths in ([3, 5, 2, 7], [0, 5, 0, 7], [3, 0, 2, 0]):
                    _check_interleaved_output(
                        dims,
                        lengths,
                        correct_cp_gradients=correction,
                        device=torch.device("cpu"),
                        bounded_memory=bounded_memory,
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_context_parallel_interleaved_output_gradients(tmp_path: Path) -> None:
    mp.spawn(_compare_interleaved_gradients, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize("correct_cp_gradients", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_parallel_output_memory(
    correct_cp_gradients: bool, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require bounded native CP output/reduction memory on a four-rank launch."""
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Requires torchrun with four GPUs.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    dims = ParallelDims(world_size=4, dp_shard=4, dp_replicate=1, cp=4)
    dims.build_meshes("cuda")
    monkeypatch.setattr(context_parallel_utils, "_CP_OUTPUT_GATHER_BYTES", 128 * 1024)
    # Warm collectives before measuring tensor allocations. Uneven streams retain padding coverage.
    _check_interleaved_output(
        dims, [3, 5, 2, 7], correct_cp_gradients=correct_cp_gradients, device=torch.device("cuda"), dtype=dtype
    )
    _check_interleaved_output(
        dims,
        [4095, 8193, 4094, 8195],
        correct_cp_gradients=correct_cp_gradients,
        device=torch.device("cuda"),
        dtype=dtype,
        check_memory=True,
    )
