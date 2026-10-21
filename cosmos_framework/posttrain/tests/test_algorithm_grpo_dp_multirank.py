# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L1 @GPU: rank-level data parallelism under a REAL process group (plan Task 6).

Three properties that only exist across processes, so the CPU tests with stub meshes cannot reach
them:

1. `_dp_coords` derives `(dp_rank, dp_world)` from a LIVE full `dp_mesh`, not a hand-built stub.
2. Ranks receive DISJOINT groups whose union is the batch. Every pre-existing GRPO test passes with
   the data path fully mirrored, so this is the one that fails if the wrapper is not wired.
3. `rank_mean_count` returns the DP-global mean over genuinely DIFFERENT per-rank token
   counts. It has only ever averaged identical values, and the `token-mean` reduction divides by it,
   so a wrong denominator silently rescales every gradient.

Run:
    torchrun --standalone --nproc_per_node=4 -m pytest \\
        cosmos_framework/posttrain/tests/test_algorithm_grpo_dp_multirank.py -q --L1 --GPU
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction  # noqa: E402
from cosmos_framework.posttrain.schema import Status, Trajectory  # noqa: E402
from cosmos_framework.posttrain.utils.dist import rank_mean_count  # noqa: E402
from cosmos_framework.posttrain.workers.dispatch import select_chunk  # noqa: E402


def _batch(n_groups: int, size: int) -> list[Trajectory]:
    return [
        Trajectory(transitions=[], id=f"g{g}_k{k}", group_id=f"g{g}", status=Status.COMPLETED)
        for g in range(n_groups)
        for k in range(size)
    ]


def _dp_shard(items: list, *, dp_rank: int, dp_world: int) -> list:
    """The deleted `dp_group_sharded`'s exact partition, via `select_chunk` at n_mini=n_micro=1
    (Task 9 — see `tests/test_workers_chunk_select.py` for the equivalence proof)."""
    return select_chunk(items, mini=0, micro=0, n_mini=1, n_micro=1, dp_rank=dp_rank, dp_world=dp_world)


@pytest.mark.L1
@pytest.mark.GPU
def test_dp_shard_and_token_count_under_a_real_process_group():
    import os

    # HARDWARE GATE FIRST, before touching CUDA. An earlier revision initialized gloo when CUDA was
    # absent and then called torch.cuda.set_device() unconditionally two lines later — the fallback
    # implied CPU support the body does not have, and `rank % torch.cuda.device_count()` divides by
    # zero on a CPU box. This body needs real GPUs: it builds a device mesh and all-reduces on cuda.
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu < 2:
        pytest.skip(f"requires >= 2 visible GPUs, found {n_gpu}")

    # torchrun exports RANK/WORLD_SIZE/MASTER_* but does NOT call init_process_group — the program
    # does. Gating on `not dist.is_initialized()` therefore skipped under torchrun too, which is how
    # this test first "passed" by never running.
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("requires torchrun (see module docstring)")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("requires >= 2 ranks")
    if not dist.is_initialized():
        dist.init_process_group("nccl")  # GPU-only body; no gloo path to mislead
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % n_gpu)
    device = torch.device("cuda", rank % n_gpu)

    # (1) LIVE mesh -> _dp_coords. The CPU tests stub this; here the real ParallelDims answers.
    from cosmos_framework.posttrain.config.build import _dp_coords
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    pdims = ParallelDims(world_size=world, dp_shard=world)
    pdims.build_device_mesh("cuda")
    assert _dp_coords(pdims) == (rank, world), f"rank {rank}: live mesh gave {_dp_coords(pdims)}"

    # (2) disjoint, exhaustive SAMPLE partition across ranks.
    #
    # Samples, not groups. The wrapper used to round-robin whole groups so each landed on one rank —
    # required only while the objective recomputed the group-relative advantage over whatever it
    # received. The advantage is written at reward time now and rides on the sample, so a group is
    # free to span ranks. What must still hold is the partition itself: every sample scored exactly
    # once across the world, which is what makes the DP gradient the true batch gradient.
    n_groups, size = 4 * world, 4
    batch = _batch(n_groups=n_groups, size=size)
    seen: list = []
    inner = Objective(
        fn=lambda m, b: (seen.append(list(b)), (torch.zeros((), device=device), {}))[1],
        reduction=Reduction.TOKEN_SUM,
    )
    dp_rank, dp_world = _dp_coords(pdims)
    inner.fn(object(), _dp_shard(batch, dp_rank=dp_rank, dp_world=dp_world))
    mine = {str(t.id) for t in seen[0]}
    total = n_groups * size
    # Round-robin over a sorted batch: counts differ by at most one, whatever the divisibility.
    assert len(mine) in (total // world, -(-total // world)), (
        f"rank {rank} got {len(mine)} of {total} samples — a round-robin share is {total}/{world} +-1"
    )

    gathered: list = [None] * world
    dist.all_gather_object(gathered, mine)
    for i in range(world):
        for j in range(i + 1, world):
            assert not (gathered[i] & gathered[j]), f"ranks {i},{j} share samples {gathered[i] & gathered[j]}"
    union = set().union(*gathered)
    assert union == {str(t.id) for t in batch}, (
        f"union over ranks must be the whole batch; {len(union)} of {total} samples were scored"
    )

    # (2b) UNEVEN shards agree on a forward count. This is what per-sample sharding costs and what
    # the padding buys back: with 5 samples over 2 ranks the shards are 3 and 2, and FSDP hangs its
    # all-gathers off the module forward, so a rank that issued only its own count would desync the
    # group. Every rank must issue max(counts) forwards. A batch smaller than the mesh (1 sample,
    # 2 ranks) is the extreme case: one rank holds nothing at all and still has to participate.
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.schema import ObsBundle, Transition

    forwards: list = []

    class _CountingLM(torch.nn.Module):
        def __init__(self, vocab: int = 16, d: int = 8):
            super().__init__()
            self.emb, self.head = torch.nn.Embedding(vocab, d), torch.nn.Linear(d, vocab)

        def forward(self, ids):
            forwards.append(tuple(ids.shape))
            return SimpleNamespace(logits=self.head(self.emb(ids)))

    def _samples(n: int) -> list[Trajectory]:
        trajs = []
        for k in range(n):
            tr = Transition(
                obs=ObsBundle(),
                action=torch.tensor([3 + k]),
                policy_info={"prompt_ids": torch.tensor([1, 2])},
                reward={"final": float(k)},
                done=True,
            )
            trajs.append(Trajectory(transitions=[tr], id=f"u_k{k}", group_id="u", status=Status.COMPLETED))
        group_relative()(trajs)
        return trajs

    lm = _CountingLM().to(device)
    for n_samples in (2 * world + 1, 1):  # does-not-divide, and smaller-than-the-mesh
        forwards.clear()
        shard = make_objective()
        shard.fn(lm, _dp_shard(_samples(n_samples), dp_rank=dp_rank, dp_world=dp_world))
        expected = -(-n_samples // world)  # ceil == max over ranks of a round-robin share
        assert len(forwards) == expected, (
            f"rank {rank}: {n_samples} samples over {world} ranks -> issued {len(forwards)} forwards, "
            f"every rank must issue max(counts)={expected} or the FSDP group desyncs"
        )
        counts: list = [None] * world
        dist.all_gather_object(counts, len(forwards))
        assert len(set(counts)) == 1, f"ranks issued different forward counts: {counts}"

    # (2c) A bad advantage on ONE rank must raise on ALL of them, not hang the group.
    #
    # This is the failure mode that replaced the old one. `select_chunk` hands each rank a
    # different slice, so a trajectory with a missing/non-finite advantage lands on exactly ONE rank.
    # A rank-local raise there aborts that rank while its peers walk into the next collective (the
    # forward-count MAX, the FSDP all-gathers, the token-count AVG) and block forever — the run hangs
    # with no error, which is worse than the silent zero the check exists to prevent.
    #
    # The test PASSING is the proof: if `require_advantages` were rank-local, rank 1 would raise and
    # rank 0 would block in the all-reduce until the job timed out.
    from cosmos_framework.posttrain.algorithm.annotators.advantage import require_advantages

    only_rank1_is_bad = _samples(2)
    if rank == 1:
        only_rank1_is_bad[0].transitions[0].advantage = None
    with pytest.raises(RuntimeError, match="advantage validation failed on one or more ranks"):
        require_advantages(only_rank1_is_bad)

    # A non-finite advantage is the same class -- it poisons the batch gradient rather than zeroing
    # it -- and must converge identically.
    only_rank0_is_bad = _samples(2)
    if rank == 0:
        only_rank0_is_bad[0].transitions[0].advantage = float("nan")
    with pytest.raises(RuntimeError, match="advantage validation failed on one or more ranks"):
        require_advantages(only_rank0_is_bad)

    # A clean batch on every rank still costs one all-reduce and passes.
    require_advantages(_samples(2))

    # (2d) A malformed trajectory on ONE rank must raise on ALL of them, not hang the group.
    #
    # Shaping (`shape_policy_input`) is rank-local data handling that raises on bad input — empty
    # `prompt_ids`, a string completion with no packer, an over-long ignore prefix. Under per-sample
    # sharding that input lands on one rank, and shaping used to happen INLINE with the forwards, so
    # that rank aborted part-way through while its peers blocked in the FSDP all-gathers of forwards
    # it would never issue. All shaping is now hoisted above the first forward and converged.
    #
    # Again, the test PASSING is the proof: a rank-local raise would leave rank 0 blocked in the
    # preparation all-reduce until the job timed out.
    bad_on_rank1 = _samples(2)
    if rank == 1:
        bad_on_rank1[0].transitions[0].policy_info["prompt_ids"] = torch.tensor([], dtype=torch.long)
    with pytest.raises(RuntimeError, match="preparation failed on one or more ranks"):
        make_objective().fn(lm, _dp_shard(bad_on_rank1, dp_rank=0, dp_world=1))

    # (3) DP-global MEAN over genuinely different per-rank counts. Rank r contributes (r+1)*10, so
    # the expected value is one NO single rank holds — a mirrored-data run could never detect this.
    got = rank_mean_count((rank + 1) * 10, device)
    expected = sum((r + 1) * 10 for r in range(world)) / world
    assert abs(got - expected) < 1e-6, f"rank {rank}: token count {got} != DP-global mean {expected}"


@pytest.mark.L1
@pytest.mark.GPU
def test_true_hsdp_2x2_mesh_gives_distinct_dp_coords_and_reduces_over_both_axes() -> None:
    """The real HSDP topology: dp_replicate=2 AND dp_shard=2. Needs exactly 4 ranks.

    Neither sibling test reaches this. One is pure FSDP (dp_shard=W, a 1-D mesh), the other is
    replicate-only (dp_shard=1, also 1-D). Only here is `dp_mesh` genuinely 2-D — and that is the
    one shape the whole HSDP change exists to produce.

    It caught a real defect: `_dp_coords` called `mesh.get_local_rank()` with no argument, which
    RAISES on a multi-dimensional mesh ("Optional kwarg `mesh_dim` needs to be specified when
    device_mesh.ndim > 1"). So building the objective died under 2x2 while every CPU test passed,
    because those stub the mesh with a SimpleNamespace that has a no-arg `get_local_rank`.

    Two properties:

    1. `_dp_coords` reports the full policy DP extent (4). Policy has one execution replica, so
       `dp_replicate` and `dp_shard` jointly partition its one collected batch.
    2. Gradients are identical across all 4 ranks after backward — the reduce-scatter within a
       replica and the all-reduce across replicas together make every rank's gradient the mean over
       all 4 ranks' data.
    """
    import os

    import torch.nn as nn

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu < 4:
        pytest.skip(f"2x2 HSDP needs >= 4 visible GPUs, found {n_gpu}")
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun with exactly 4 ranks (--nproc_per_node=4)")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % n_gpu)

    from cosmos_framework.posttrain.config.build import _dp_coords
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.lin(x)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = _Block()

        def forward(self, x):
            return self.b(x)

    pd = ParallelDims(world_size=4, dp_replicate=2, dp_shard=2)
    pd.build_device_mesh("cuda")
    assert pd.dp_mesh.ndim == 2, "fixture must be the genuinely 2-D mesh this regresses on"

    # (1) distinct dp coordinate per rank over the full model-DP extent.
    dp_rank, dp_world = _dp_coords(pd)
    assert dp_world == 4, f"rank {rank}: full policy dp extent {dp_world} != 4"
    coords: list = [None] * world
    dist.all_gather_object(coords, dp_rank)
    assert sorted(coords) == [0, 1, 2, 3]

    # (2) gradients agree across BOTH axes after backward.
    net = _Net().cuda()
    parallelize(net, ShardPlan(block_types=(_Block,), param_dtype=torch.float32, reduce_dtype=torch.float32), pd)
    net(torch.full((4, 8), float(rank + 1), device="cuda")).sum().backward()
    g = net.b.lin.weight.grad
    g = g.full_tensor() if hasattr(g, "full_tensor") else g
    gathered = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(gathered, g.contiguous())
    for i in range(1, world):
        assert torch.allclose(gathered[0], gathered[i]), (
            f"gradients differ across the 2x2 mesh -> an axis is not reducing: "
            f"{float(gathered[0][0, 0])} vs {float(gathered[i][0, 0])}"
        )
    expected = sum(4.0 * (r + 1) for r in range(world)) / world
    assert abs(float(gathered[0][0, 0]) - expected) < 1e-5, f"{float(gathered[0][0, 0])} != mean {expected}"


@pytest.mark.L1
@pytest.mark.GPU
def test_replicate_only_mesh_all_reduces_gradients_across_dp_replicate_axis() -> None:
    """A replicate-only model mesh `(dp_replicate=K, dp_shard=1)` must still reduce gradients.

    `parallelize` used to gate on `dp_shard_enabled` alone and return early for "replicate-only",
    so `fully_shard` was never called: no FSDP wrapper, no gradient all-reduce, replicas training
    independently. This is a model DeviceMesh inside one policy execution replica; placement R is
    not involved.

    Asserts the PROPERTY, not the wrapping: each rank backprops DIFFERENT data, and post-backward
    gradients must be identical across ranks. A test that only checked `isinstance(..., FSDPModule)`
    would pass on a wrapper that never reduces.
    """
    import os

    import torch.nn as nn

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu < 2:
        pytest.skip(f"requires >= 2 visible GPUs, found {n_gpu}")
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun with >= 2 ranks")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % n_gpu)

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.lin(x)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = _Block()

        def forward(self, x):
            return self.b(x)

    pd = ParallelDims(world_size=world, dp_replicate=world, dp_shard=1)
    pd.build_device_mesh("cuda")
    assert not pd.dp_shard_enabled, "fixture must be the replicate-ONLY case this regresses on"

    net = _Net().cuda()
    parallelize(net, ShardPlan(block_types=(_Block,), param_dtype=torch.float32, reduce_dtype=torch.float32), pd)

    net(torch.full((4, 8), float(rank + 1), device="cuda")).sum().backward()  # different data per rank
    g = net.b.lin.weight.grad
    g = g.full_tensor() if hasattr(g, "full_tensor") else g
    gathered = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(gathered, g.contiguous())
    for i in range(1, world):
        assert torch.allclose(gathered[0], gathered[i]), (
            f"gradients differ across replicas -> no cross-replica all-reduce: "
            f"{float(gathered[0][0, 0])} vs {float(gathered[i][0, 0])}"
        )
    expected = sum(4.0 * (r + 1) for r in range(world)) / world  # DP mean of per-rank sums
    assert abs(float(gathered[0][0, 0]) - expected) < 1e-5, f"{float(gathered[0][0, 0])} != mean {expected}"
