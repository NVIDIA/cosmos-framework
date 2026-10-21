# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: rank-level per-sample batch selection (spec 2026-08-04 §2.2 + §5.1).

Task 9 deleted the objective wrapper `dp_group_sharded`: the ingress predicate `select_chunk`
(`workers/dispatch.py`) is now the sole dp-axis owner, and at `n_mini == n_micro == 1` it selects
exactly the partition the deleted wrapper used to (pinned by
`tests/test_workers_chunk_select.py::test_degenerate_case_matches_the_deleted_dp_group_sharded_partition`).
These tests exercise that same partition via `_dp_shard` below, a thin `select_chunk` call at that
degenerate setting, so the properties `dp_group_sharded` used to guarantee stay covered.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.algorithm.primitives.objective import MetricAdapter, Objective, Reduction
from cosmos_framework.posttrain.schema import Status, Trajectory
from cosmos_framework.posttrain.workers.dispatch import select_chunk


def _dp_shard(items: list, *, dp_rank: int, dp_world: int) -> list:
    """The deleted `dp_group_sharded`'s exact partition, via `select_chunk` at n_mini=n_micro=1."""
    return select_chunk(items, mini=0, micro=0, n_mini=1, n_micro=1, dp_rank=dp_rank, dp_world=dp_world)


def _batch(n_groups: int, size: int = 4) -> list[Trajectory]:
    """`n_groups` complete groups of `size` members each."""
    return [
        Trajectory(transitions=[], id=f"g{g}_k{k}", group_id=f"g{g}", status=Status.COMPLETED)
        for g in range(n_groups)
        for k in range(size)
    ]


def _spy():
    """An Objective that records the batch it was handed."""
    seen: list = []

    def fn(model, batch):
        seen.append(list(batch))
        return "loss-sentinel", {"n": len(batch)}

    adapter = MetricAdapter(
        names={"spy_n": "valid_sequence_count"},
        internal_prefix="spy_",
        phases=frozenset({"train"}),
    )
    return Objective(
        fn=fn,
        reduction=Reduction.TOKEN_SUM,
        metric_adapter=adapter,
        identity={"algorithm": "spy"},
    ), seen


@pytest.mark.L0
@pytest.mark.CPU
def test_partitions_samples_disjointly_and_exhaustively():
    """Every SAMPLE lands on exactly one rank; the union over ranks is the input.

    Sharding is per-sample now, not per-group. Group atomicity was required only while the objective
    computed the group-relative advantage itself; the advantage is baked in at reward time, so a
    group may be split across ranks — each sample already carries the value normalized over its
    whole group. Splitting samples also balances better and drops the divisibility constraint that
    coupled `prompts_per_step` to the rank count.
    """
    batch = _batch(n_groups=3, size=4)  # 12 samples, 3 groups — group count does NOT divide 4 ranks
    seen_ids, all_ids = [], set()
    for rank in range(4):
        inner, seen = _spy()
        inner.fn(object(), _dp_shard(batch, dp_rank=rank, dp_world=4))
        ids = {t.id for t in seen[0]}
        assert len(ids) == 3, f"rank {rank} should get 12/4 = 3 samples, got {len(ids)}"
        assert not (ids & all_ids), f"sample on two ranks: {ids & all_ids}"
        all_ids |= ids
        seen_ids.append(ids)
    assert all_ids == {t.id for t in batch}, "union over ranks must be the whole batch"


@pytest.mark.L0
@pytest.mark.CPU
def test_partition_is_deterministic_regardless_of_input_order():
    """Ranks must agree on the partition; input order must not change who owns a group."""
    batch = _batch(n_groups=4, size=4)
    shuffled = batch[::-1]
    for rank in range(2):
        a, seen_a = _spy()
        b, seen_b = _spy()
        a.fn(object(), _dp_shard(batch, dp_rank=rank, dp_world=2))
        b.fn(object(), _dp_shard(shuffled, dp_rank=rank, dp_world=2))
        assert {t.group_id for t in seen_a[0]} == {t.group_id for t in seen_b[0]}


@pytest.mark.L0
@pytest.mark.CPU
def test_build_objective_never_shards_the_batch_regardless_of_pdims(monkeypatch):
    """Task 9: `build_objective` must hand the objective the WHOLE batch, whatever `pdims` says.

    Before Task 9 this was conditional on a `rank_shared_batch` flag: `True` (the ray grid backend)
    wrapped the objective in `dp_group_sharded`, `False` (`grid.backend=local`, where each rank
    already runs its own in-process env/rollout with its own groups) left it whole. That flag —
    and the wrapper it gated — no longer exists: `build_objective` never partitions, on ANY backend.
    Partitioning the dp axis, when a caller wants it, is `select_chunk`'s job now, applied at the
    WorkerGroup ingress before the objective is ever called — not something `build_objective` can
    still be asked to do via a removed keyword.
    """
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import build_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    def _mesh(rank, size):
        return SimpleNamespace(ndim=1, get_local_rank=lambda dim: rank, size=lambda dim=None: size)

    pdims = SimpleNamespace(
        dp_enabled=True, dp_replicate=1, dp_shard_enabled=True, dp_shard_mesh=_mesh(0, 4), dp_mesh=_mesh(0, 4)
    )
    batch = _batch(n_groups=4, size=4)
    rl = GRPOConfig(group_size=4)
    rl.grpo_surrogate = "legacy_pg"

    seen: list = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.algorithm.grpo.make_objective",
        lambda **kw: Objective(fn=lambda m, b: (seen.append(len(b)), None)[1], reduction=Reduction.TOKEN_SUM),
    )

    # A live 4-way dp mesh must not shrink what the objective sees any more.
    build_objective(rl, "unused/model", pdims=pdims).fn(object(), batch)
    assert seen[-1] == len(batch), (
        f"build_objective sharded the batch to {seen[-1]} of {len(batch)} — partitioning is the "
        f"ingress's job now (select_chunk), not build_objective's"
    )

    # None (single-device) must be byte-identical.
    build_objective(rl, "unused/model", pdims=None).fn(object(), batch)
    assert seen[-1] == len(batch)


@pytest.mark.L0
@pytest.mark.CPU
def test_group_relative_annotations_survive_the_shard_because_the_objective_reads_them():
    """Everything computed OVER a group must be computed before the shard, never inside the objective.

    Per-sample sharding is only sound because every group-relative quantity already rides on the
    sample. The advantage does — `require_advantages` makes a missing one loud. Shared-prefix loss
    filtering is the one that would fail QUIETLY: the maximal shared prefix of half a group is at
    least as long as the whole group's, so an objective that re-derived it from its own shard would
    mask more tokens than intended, differently on each rank, with the loss silently depending on how
    dispatch happened to split the batch. It is a reward-stage annotator for exactly that reason.

    Asserted structurally, on the objective factories: neither may accept the filter's knobs, because
    accepting them is what would let a caller re-enable the per-shard computation.
    """
    import inspect

    from cosmos_framework.posttrain.algorithm import grpo

    for factory in (grpo.make_objective, grpo.make_clipped_objective):
        params = inspect.signature(factory).parameters
        assert "min_filter_prefix_tokens" not in params and "eos_token_id" not in params, (
            f"{factory.__name__} accepts a shared-prefix filter knob; that computation needs a WHOLE "
            f"group and the objective only ever sees this rank's shard of one"
        )

    # And the annotation actually survives a shard: mask the group upstream, then shard, and the
    # sampled trajectories still carry the number derived from all four members.
    from cosmos_framework.posttrain.algorithm.annotators.prefix_filter import shared_prefix_ignore
    from cosmos_framework.posttrain.schema import ObsBundle, Transition

    group = []
    for k, (tail, reward) in enumerate(((1, 1.0), (2, 1.0), (3, 0.0), (4, 0.0))):
        tr = Transition(obs=ObsBundle(), action=[7, 8, 9, tail], policy_info={}, done=True)
        tr.reward = {"final": reward}
        group.append(Trajectory(transitions=[tr], id=f"g_k{k}", group_id="g", status=Status.COMPLETED))
    shared_prefix_ignore(1)(group)
    whole = [t.transitions[0].algo_extra["n_ignore_prefix_tokens"] for t in group]
    assert whole == [3, 3, 3, 3], f"fixture check: all four share the prefix [7,8,9], got {whole}"

    inner, seen = _spy()
    inner.fn(object(), _dp_shard(group, dp_rank=0, dp_world=2))
    assert [t.transitions[0].algo_extra["n_ignore_prefix_tokens"] for t in seen[0]] == [3, 3], (
        "the shard must carry the mask derived from the WHOLE group"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_uneven_and_empty_shards_still_issue_one_forward_per_agreed_count():
    """Per-sample sharding makes shard sizes UNEQUAL; the forward count must not follow them.

    FSDP hangs its all-gathers off the module forward, so the forward count is the collective count
    and every rank must issue the same number. Group-wise sharding gave equal counts by construction
    — whole groups, plus a startup gate that refused topologies where the group count did not divide
    the rank count — and the legacy_pg objective simply assumed that. Round-robining SAMPLES breaks
    the assumption two ways: sizes differ by one when the batch does not divide, and a rank gets ZERO
    when the batch is smaller than the mesh. A rank that raised or short-looped there would abort or
    under-issue while its peers blocked in the forward.

    Single-process here, so `rank_max_count` is the identity and the agreed count IS the local
    one; what this pins is that the loop issues exactly `n_forwards` forwards for any batch size,
    which is what makes the padded path correct once the all-reduce is real.
    """
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.schema import ObsBundle, Transition

    forwards: list = []

    class _CountingLM(nn.Module):
        def __init__(self, vocab: int = 16, d: int = 8):
            super().__init__()
            self.emb, self.head = nn.Embedding(vocab, d), nn.Linear(d, vocab)

        def forward(self, ids):
            forwards.append(tuple(ids.shape))
            return SimpleNamespace(logits=self.head(self.emb(ids)))

    def _shard(n: int) -> list[Trajectory]:
        trajs = []
        for k in range(n):
            tr = Transition(
                obs=ObsBundle(),
                action=torch.tensor([3 + k]),
                policy_info={"prompt_ids": torch.tensor([1, 2])},
                reward={"final": float(k)},
                done=True,
            )
            trajs.append(Trajectory(transitions=[tr], id=f"g_k{k}", group_id="g", status=Status.COMPLETED))
        group_relative()(trajs)
        return trajs

    obj = make_objective()
    for n in (1, 2, 3):
        forwards.clear()
        _, metrics = obj.fn(_CountingLM(), _shard(n))
        assert len(forwards) == n, f"batch of {n} issued {len(forwards)} forwards"
        assert metrics["grpo_n"] == n

    # An empty shard with no peers is the all-empty case: a real error, and symmetric, so it raises.
    with pytest.raises(ValueError, match="empty batch"):
        obj.fn(_CountingLM(), [])
