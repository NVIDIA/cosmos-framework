# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from cosmos_framework.posttrain.model.parallelize import parallelize
from cosmos_framework.posttrain.model.specs import ShardPlan
from cosmos_framework.posttrain.tests.helpers.tiny_decoder import Block, TinyDecoder


@pytest.mark.L0
@pytest.mark.CPU
def test_noop_when_pdims_none():
    m = TinyDecoder()
    plan = ShardPlan((Block,), torch.float32, torch.float32)
    parallelize(m, plan, None)  # no dist, no mesh
    # params stay plain tensors (no DTensor)
    assert all(not hasattr(p, "device_mesh") or p.device_mesh is None for p in m.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_unwrapped_probe_only_fires_when_the_plan_delivers_a_compute_dtype():
    """`param_dtype=None` (autocast mode) means the wrap would deliver nothing, so an unwrapped mesh has nothing it
    could fail to honour; a compute dtype that differs from storage is still refused there."""
    m = TinyDecoder().to(torch.bfloat16)
    parallelize(m, ShardPlan((Block,), None, torch.float32), None)
    with pytest.raises(ValueError, match="cannot be honoured without an FSDP wrap"):
        parallelize(m, ShardPlan((Block,), torch.float32, torch.float32), None)


class _RecordingBlock(torch.nn.Module):
    """Bias-free linear: under autocast its output is pure bf16 (nothing promotes it back to fp32, unlike a residual
    add against fp32 activations), so what the NEXT block receives is a clean observable. Records its input dtype."""

    def __init__(self, n=16):
        super().__init__()
        self.lin = torch.nn.Linear(n, n, bias=False)
        self.seen: list = []

    def forward(self, x):
        self.seen.append(x.dtype)
        return self.lin(x)


class _TwoBlocks(torch.nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.b1 = _RecordingBlock(n)
        self.b2 = _RecordingBlock(n)

    def forward(self, x):
        return self.b2(self.b1(x))


def _forward_input_dtype_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        pd = ParallelDims(world_size=world_size, dp_shard=world_size)
        pd.build_device_mesh("cpu")
        models = {}
        for name, param_dtype in (("plain", ...), ("none", None), ("storage", torch.float32)):
            torch.manual_seed(0)
            model = _TwoBlocks()
            if param_dtype is not ...:
                parallelize(model, ShardPlan((_RecordingBlock,), param_dtype, torch.float32), pd)
            models[name] = model
        x = torch.ones(2, 16)
        outs = {}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for name, model in models.items():
                outs[name] = model(x)
        seen = {name: model.b2.seen for name, model in models.items()}
        # b1 runs under autocast and emits bf16; what b2 RECEIVES is the observable.
        assert seen["plain"] == [torch.bfloat16], seen
        # `param_dtype=None`: the wrap is dtype-transparent, so wrapped == unwrapped.
        assert seen["none"] == seen["plain"], seen
        assert outs["none"].dtype is outs["plain"].dtype and torch.equal(outs["none"], outs["plain"])
        # The storage dtype in the plan (the pre-fix encoding of "no cast"): FSDP2's `cast_forward_inputs` casts
        # b2's bf16 input back to fp32 at the block boundary — a wrapped forward that differs from the unwrapped one.
        assert seen["storage"] == [torch.float32], seen
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_autocast_plan_wraps_without_casting_forward_inputs(tmp_path: Path) -> None:
    """Wrapped (dp_shard=2, CPU gloo mesh) vs unwrapped under `torch.autocast(bfloat16)`: with `param_dtype=None` the
    second block sees the same bf16 input either way; with the storage dtype in the plan it sees fp32."""
    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _forward_input_dtype_worker,
        args=(world_size, str(tmp_path / "parallelize-rendezvous")),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L1
@pytest.mark.GPU
def test_shards_blocks_to_dtensor():
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    torch.cuda.set_device(dist.get_rank())
    m = TinyDecoder().cuda()
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    parallelize(m, ShardPlan((Block,), torch.bfloat16, torch.float32), pd)
    assert any(isinstance(p, DTensor) for p in m.parameters())


class _Blk(torch.nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.lin = torch.nn.Linear(n, n)

    def forward(self, x):
        return self.lin(x)


class _RootBlk(torch.nn.Module):
    """A model whose OWN type is placed in plan.block_types (single-block / custom-model shape).
    It is the first element of model.modules(), so it exercises the root-matches-block_types path."""

    def __init__(self, n=16):
        super().__init__()
        self.lin = torch.nn.Linear(n, n)
        self.child = _Blk(n)

    def forward(self, x):
        return self.child(self.lin(x))


@pytest.mark.L1
@pytest.mark.GPU
def test_root_matching_block_type_not_double_sharded():
    """Regression: when type(model) is in plan.block_types, the root must be sharded EXACTLY once
    (by the explicit root fully_shard), not also inside the block loop — a second fully_shard on an
    already-sharded FSDP2 module corrupts param state. Guard: parallelize succeeds and a forward is
    finite (the old code double-shards the root here)."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    torch.cuda.set_device(dist.get_rank())

    m = _RootBlk(16).cuda()
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    # block_types includes the ROOT's own type (_RootBlk) plus the inner block (_Blk).
    parallelize(m, ShardPlan((_RootBlk, _Blk), torch.float32, torch.float32), pd)

    assert any(isinstance(p, DTensor) for p in m.parameters())
    out = m(torch.ones(4, 16, device="cuda"))  # forward proves params weren't corrupted
    assert out.shape == (4, 16) and torch.isfinite(out.float()).all()
