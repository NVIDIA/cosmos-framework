# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

r"""One- and two-rank FSDP2 decoder-layer CPU-offload integration tests.

Run the singleton and distributed cases separately:

    torchrun --standalone --nproc_per_node=1 -m pytest -v -s -p no:randomly \
        cosmos_framework/model/generator/mot/fsdp_cpu_offload_test.py --L1
    torchrun --standalone --nproc_per_node=2 -m pytest -v -s -p no:randomly \
        cosmos_framework/model/generator/mot/fsdp_cpu_offload_test.py --L1
"""

import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import (
    apply_compile,
    apply_fsdp,
    materialize_non_offloaded_state,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh

pytestmark = [pytest.mark.L1, pytest.mark.GPU]


class _CombinedDecoderBlock(nn.Module):
    """Tiny block with separate reasoner and generator weights."""

    reasoner: nn.Linear
    generator: nn.Linear

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.reasoner = nn.Linear(hidden_size, hidden_size, bias=False)
        self.generator = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B,D], returns: [B,D]
        return self.reasoner(inputs) + self.generator(inputs)  # [B,D]

    def reasoner_forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B,D], returns: [B,D]
        return self.reasoner(inputs)  # [B,D]


class _TinyUnifiedModel(nn.Module):
    model: nn.Module

    def __init__(self, hidden_size: int, layer_count: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_CombinedDecoderBlock(hidden_size) for _ in range(layer_count)])


@pytest.fixture(scope="module")
def distributed_group() -> Iterator[None]:
    """Join a one- or two-rank NCCL torchrun group."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if "RANK" not in os.environ:
        pytest.skip("run with torchrun --standalone --nproc_per_node=1 or 2")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (1, 2):
        pytest.skip("requires a one- or two-rank torchrun launch")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        pytest.skip("requires one CUDA device per local rank")

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    try:
        yield
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _require_world_size(expected: int) -> None:
    """Skip a topology-specific test when torchrun has a different world size."""
    actual = dist.get_world_size()
    if actual != expected:
        pytest.skip(f"requires WORLD_SIZE={expected}, got {actual}")


def _reference_outputs(model: _TinyUnifiedModel, inputs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    joint_outputs: list[torch.Tensor] = []
    reasoner_outputs: list[torch.Tensor] = []
    for block in model.model.layers:
        joint_outputs.append(block(inputs).detach().clone())  # [B,D]
        reasoner_outputs.append(block.reasoner_forward(inputs).detach().clone())  # [B,D]
    return joint_outputs, reasoner_outputs


@pytest.mark.parametrize(("dp_shard", "dp_replicate"), [(1, 2), (2, 1)])
@pytest.mark.parametrize("compile_blocks", [False, True])
@torch.no_grad()
def test_combined_blocks_reshard_to_cpu_after_joint_and_reasoner_forwards(
    distributed_group: None,
    dp_shard: int,
    dp_replicate: int,
    compile_blocks: bool,
) -> None:
    """Exercise replicate-only and sharded offload, with and without compile."""
    del distributed_group
    _require_world_size(2)
    torch.manual_seed(1234)
    model = _TinyUnifiedModel(hidden_size=16, layer_count=3).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)  # [B,D]
    expected_joint, expected_reasoner = _reference_outputs(model, inputs)

    parallel_dims = ParallelDims(
        world_size=2,
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        enable_inference_mode=True,
        fsdp_cpu_offload=True,
    )
    parallel_dims.build_meshes(device_type="cuda")
    if compile_blocks:
        apply_compile(model, CompileConfig(enabled=True, use_cuda_graphs=False, mark_unbacked=False))
    apply_fsdp(model, parallel_dims)

    for layer_index, block in enumerate(model.model.layers):
        actual_joint = block(inputs)  # [B,D]
        torch.testing.assert_close(actual_joint, expected_joint[layer_index])
        for parameter in block.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().device.type == "cpu"

        actual_reasoner = block.reasoner_forward(inputs)  # [B,D]
        torch.testing.assert_close(actual_reasoner, expected_reasoner[layer_index])
        for parameter in block.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().device.type == "cpu"

    # A second pass catches collective-order or stale-unsharded-storage bugs.
    for layer_index, block in enumerate(model.model.layers):
        torch.testing.assert_close(block(inputs), expected_joint[layer_index])


@pytest.mark.parametrize(("dp_shard", "dp_replicate"), [(1, 2), (2, 1)])
@torch.no_grad()
def test_unit_materialization_produces_state_dict_safe_cpu_dtensors(
    distributed_group: None,
    dp_shard: int,
    dp_replicate: int,
) -> None:
    """Materialize one block at a time and verify state-dict-safe CPU shards."""
    del distributed_group
    _require_world_size(2)
    with torch.device("meta"):
        model = _TinyUnifiedModel(hidden_size=16, layer_count=2)

    parallel_dims = ParallelDims(
        world_size=2,
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        enable_inference_mode=True,
        fsdp_cpu_offload=True,
    )
    parallel_dims.build_meshes(device_type="cuda")
    apply_fsdp(model, parallel_dims)
    materialize_non_offloaded_state(model, device="cuda")

    state_dict = model.state_dict()
    assert state_dict
    for block in model.model.layers:
        for parameter in block.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().device.type == "cpu"


@torch.no_grad()
def test_single_rank_offload_uses_1d_mesh_and_reshards_to_cpu(distributed_group: None) -> None:
    """Exercise real singleton FSDP hooks instead of only testing mesh selection."""
    del distributed_group
    _require_world_size(1)
    torch.manual_seed(1234)
    model = _TinyUnifiedModel(hidden_size=16, layer_count=2).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)  # [B,D]
    expected_joint, expected_reasoner = _reference_outputs(model, inputs)

    parallel_dims = ParallelDims(
        world_size=1,
        dp_shard=1,
        dp_replicate=1,
        enable_inference_mode=True,
        fsdp_cpu_offload=True,
    )
    parallel_dims.build_meshes(device_type="cuda")
    mesh = fsdp_mesh(parallel_dims)
    assert mesh is not None
    assert mesh.ndim == 1
    apply_fsdp(model, parallel_dims)

    for layer_index, block in enumerate(model.model.layers):
        actual_joint = block(inputs)  # [B,D]
        torch.testing.assert_close(actual_joint, expected_joint[layer_index])
        for parameter in block.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().device.type == "cpu"

        actual_reasoner = block.reasoner_forward(inputs)  # [B,D]
        torch.testing.assert_close(actual_reasoner, expected_reasoner[layer_index])
        for parameter in block.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().device.type == "cpu"
