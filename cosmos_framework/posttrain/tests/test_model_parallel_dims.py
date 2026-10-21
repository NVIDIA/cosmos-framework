# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""posttrain ParallelDims is a thin adapter over cosmos3's ParallelDims. These tests cover what the
adapter ADDS (the build_device_mesh() name; the non-dp-axis fail-loud guard) + that the reused
cosmos3 dp behaviour (infer / HSDP split / enabled flag) still reaches the RL call sites. The dp
factorization edge cases are cosmos3's own concern and are tested there, not re-tested here."""

import pytest

pytest.importorskip("torch")
from cosmos_framework.utils.generator.parallelism import ParallelDims as _Cosmos3ParallelDims
from cosmos_framework.posttrain.model.parallel_dims import ParallelDims


@pytest.mark.L0
@pytest.mark.CPU
def test_is_cosmos3_parallel_dims():
    """The single mesh spec: posttrain's ParallelDims IS cosmos3's (subclass), so both paths share
    one implementation (no drift)."""
    assert issubclass(ParallelDims, _Cosmos3ParallelDims)
    assert isinstance(ParallelDims(world_size=1), _Cosmos3ParallelDims)


@pytest.mark.L0
@pytest.mark.CPU
def test_infer_dp_shard_and_enabled():
    pd = ParallelDims(world_size=8)  # dp_shard/dp_replicate auto-infer via cosmos3
    assert pd.dp_shard == 8 and pd.dp_replicate == 1 and pd.dp_shard_enabled


@pytest.mark.L0
@pytest.mark.CPU
def test_hsdp_split():
    pd = ParallelDims(world_size=8, dp_replicate=2, dp_shard=4)
    assert pd.dp_shard == 4 and pd.dp_replicate == 2 and pd.dp_shard_enabled


@pytest.mark.L0
@pytest.mark.CPU
def test_single_device_not_enabled():
    pd = ParallelDims(world_size=1)
    assert pd.dp_shard == 1 and not pd.dp_shard_enabled


@pytest.mark.L0
@pytest.mark.CPU
def test_bad_product_raises():
    # cosmos3's dp invariant (dp_replicate * dp_shard == world_size) still fires through the adapter.
    with pytest.raises(ValueError):
        ParallelDims(world_size=8, dp_replicate=3, dp_shard=4)  # 12 != 8


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("axis", ["tp", "ep", "pp", "cp", "cfgp"])
def test_non_dp_axis_fails_loud(axis):
    """The adapter's guard: the RL FSDP path implements only dp_replicate × dp_shard, so any non-dp
    axis > 1 (incl. cosmos3-native cp/cfgp) must fail loud until it's wired in model/parallelize.py."""
    with pytest.raises(ValueError):
        ParallelDims(world_size=8, dp_shard=8, **{axis: 2})


@pytest.mark.L1
@pytest.mark.GPU
def test_build_device_mesh_shards():
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) < 2:
        # torchrun-mode test: needs a real multi-rank env:// world (torchrun --nproc_per_node>=2).
        # WORLD_SIZE-based (not RANK) because other tests leak single-rank env defaults.
        pytest.skip("torchrun-mode test: run under torchrun --nproc_per_node>=2 (WORLD_SIZE>=2)")
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    # torchrun provides the PG; build_device_mesh (-> cosmos3 build_meshes) yields the dp_shard mesh.
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    assert pd.dp_shard_mesh.size() == world
