# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Contracts for model-only checkpoint continuation planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.posttrain.checkpoint import ContinuationPlan as PublicContinuationPlan
from cosmos_framework.posttrain.checkpoint.continuation import ContinuationPlan, slice_pool

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _pool(path: Path, rows: int = 20) -> Path:
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(rows)))
    return path


def _run(tmp_path: Path, pool: Path, minimum: int = 0, fraction: str = "null") -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config_resolved.yaml").write_text(
        f"rl:\n  batch_per_replica: 8\n  group_size: 4\n  seed: 7\n"
        f"  min_informative_groups: {minimum}\n  min_informative_groups_frac: {fraction}\n"
        f"  data:\n    path: {pool}\n    shuffle: true\n"
    )
    return run


def test_continuation_plan_requires_exact_dynamic_draws_and_preserves_manifest(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool.jsonl")
    run = _run(tmp_path, pool, minimum=2)
    assert PublicContinuationPlan is ContinuationPlan
    with pytest.raises(ValueError, match="exact prompt-draw count"):
        ContinuationPlan.from_run(str(run), 3)
    plan = ContinuationPlan.from_run(str(run), 3, consumed=8)
    assert (plan.nominal_consumed, plan.consumed, plan.dynamic_sampling) == (6, 8, True)
    output, manifest = plan.materialize()
    assert manifest == json.loads(Path(output).with_name("pool_from8_manifest.json").read_text())
    assert set(manifest) == {
        "version",
        "pool",
        "pool_rows",
        "pool_sha256",
        "consumed",
        "shuffle",
        "seed",
        "rows",
        "order",
    }


def test_continuation_slice_reuses_identity_and_refuses_collision(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool.jsonl")
    output, manifest = slice_pool(str(pool), 4, shuffle=False, seed=0)
    assert slice_pool(str(pool), 4, shuffle=False, seed=0) == (output, manifest)
    with pytest.raises(FileExistsError):
        slice_pool(str(pool), 4, shuffle=True, seed=0)


def _run_with_seeds(tmp_path: Path, pool: Path, *, rl_lines: str, rollout_lines: str = "") -> Path:
    run = tmp_path / "run_seeds"
    run.mkdir()
    (run / "config_resolved.yaml").write_text(
        f"rl:\n  batch_per_replica: 8\n  group_size: 4\n  min_informative_groups: 0\n{rl_lines}"
        f"  data:\n    path: {pool}\n    shuffle: true\n{rollout_lines}"
    )
    return run


def test_run_data_config_seed_is_the_resolved_data_seed(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.checkpoint.continuation import run_data_config

    pool = _pool(tmp_path / "pool.jsonl")
    # default: rl.seed orders the prompts, whatever rollout.seed says
    run = _run_with_seeds(tmp_path, pool, rl_lines="  seed: 7\n", rollout_lines="rollout:\n  seed: 42\n")
    assert run_data_config(str(run))["seed"] == 7
    assert ContinuationPlan.from_run(str(run), 3).seed == 7
    (run / "config_resolved.yaml").unlink()
    run.rmdir()
    # rl.data_seed_follows_rollout_seed: the cursor permuted with rollout.seed, so the slice must too
    run = _run_with_seeds(
        tmp_path,
        pool,
        rl_lines="  seed: 0\n  data_seed_follows_rollout_seed: true\n",
        rollout_lines="rollout:\n  seed: 42\n",
    )
    assert run_data_config(str(run))["seed"] == 42
    assert ContinuationPlan.from_run(str(run), 3).seed == 42
    (run / "config_resolved.yaml").unlink()
    run.rmdir()
    # the flag without an integer rollout.seed cannot be reproduced; refuse rather than guess rl.seed
    run = _run_with_seeds(tmp_path, pool, rl_lines="  seed: 0\n  data_seed_follows_rollout_seed: true\n")
    with pytest.raises(ValueError, match="data_seed_follows_rollout_seed=true requires an integer rollout.seed"):
        run_data_config(str(run))


# --------------------------------------------------------------------------- exact continuation (cursor offset)
# The arm this guards: rl.batch_per_replica 32, rl.group_size 4, grid.shapes.rollout [8, 4] (8 data shards drawing one
# prompt each per call), rl.seed 0 with rl.data_seed_follows_rollout_seed under rollout.seed 42, rl.data.shuffle true,
# dynamic sampling on (rl.min_informative_groups 8). Its warm restarts from the iter_50 model-only artifact re-served
# the cold run's step-1 prompts (identical `[sample] step=1` gold tails at arm-log lines 1630-1631 and 4356-4357).
_ARM_RL = (
    "rl:\n  batch_per_replica: 32\n  group_size: 4\n  seed: 0\n  data_seed_follows_rollout_seed: true\n"
    "  min_informative_groups: 8\n  min_informative_groups_frac: null\n"
)
_ARM_GRID = (
    "grid:\n  backend: ray\n  num_replicas: 1\n  ranks: 1\n  shapes:\n    policy: [1, 32]\n    rollout: [8, 4]\n"
)


def _arm_run(
    tmp_path: Path, pool: Path, *, grid: str = _ARM_GRID, rollout_seed: int = 42, name: str = "arm_run"
) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "config_resolved.yaml").write_text(
        f"{_ARM_RL}  data:\n    path: {pool}\n    prompt_column: messages\n    shuffle: true\n"
        f"rollout:\n  seed: {rollout_seed}\n{grid}"
    )
    return run


def test_continuation_plan_reads_data_replicas_and_emits_exact_overrides(tmp_path: Path) -> None:
    from cosmos_framework.posttrain.checkpoint.continuation import continuation_overrides, run_data_replicas

    pool = _pool(tmp_path / "pool.jsonl", rows=600)
    run = _arm_run(tmp_path, pool)
    assert run_data_replicas(str(run)) == 8
    # 50 steps at 8 prompts per step is the nominal 400; dynamic sampling drew 40 more (5 extra per shard).
    plan = ContinuationPlan.from_run(str(run), 50, consumed=440)
    assert (plan.seed, plan.data_replicas, plan.consumed, plan.consumed_per_shard) == (42, 8, 440, 55)
    assert plan.overrides() == [
        f"rl.data.path={pool}",
        "rl.data.shuffle=true",
        "rl.seed=42",
        "rl.data_seed_follows_rollout_seed=false",
        "rl.data.cursor_offset=55",
        "rl.data.cursor_shards=8",
    ]
    assert continuation_overrides(str(run), 50, consumed=440) == plan.overrides()
    # Every shard answers every draw, so the total must divide by the shard count.
    with pytest.raises(ValueError, match="not a multiple of the run's 8 data shard"):
        ContinuationPlan.from_run(str(run), 50, consumed=441)
    # Dynamic sampling still requires the exact count (the nominal is a lower bound).
    with pytest.raises(ValueError, match="exact prompt-draw count"):
        ContinuationPlan.from_run(str(run), 50)
    # No grid section: one unsharded cursor; grid.num_replicas without shapes is the rollout fallback R.
    assert run_data_replicas(str(_arm_run(tmp_path, pool, grid="", name="bare"))) == 1
    fallback = _arm_run(tmp_path, pool, grid="grid:\n  num_replicas: 4\n", name="fallback")
    assert run_data_replicas(str(fallback)) == 4
    assert ContinuationPlan.from_run(str(fallback), 50, consumed=440).consumed_per_shard == 110
    broken = _arm_run(tmp_path, pool, grid="grid:\n  shapes:\n    policy: [1, 32]\n", name="broken")
    with pytest.raises(ValueError, match="grid.shapes.rollout must be"):
        run_data_replicas(str(broken))


def test_continuation_refuses_a_source_run_whose_shards_refilled_unequally(tmp_path: Path) -> None:
    """`consumed_per_shard = consumed // data_replicas` assumes every shard answered
    every draw. Under `rl.dynamic_sampling_refill_mode=short_cells` (not on main) the short cells refill alone
    and the idle cells draw 0, so the shards' counts differ; the even split then re-serves trained prompts on the
    shard that drew more and skips unseen prompts on the shard that drew less. Part 1 shows that mechanism with the
    framework's cursor; part 2 is the contract: `from_run` refuses such a source run, and accepts the default mode,
    an absent knob, and the knob with dynamic sampling off (where it is inert)."""
    from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor

    rows = list(range(40))

    def cursor(shard: int) -> IndexedCursor:
        return IndexedCursor(rows, shard_index=shard, shard_count=2, shuffle=True, seed=7)

    # Part 1 -- the mechanism. The cold run's two shards drew [4, 2] (shard 1 idle for two refills); total 6.
    cold = [cursor(0), cursor(1)]
    served = [[cold[0].next_index() for _ in range(4)], [cold[1].next_index() for _ in range(2)]]
    cold_next = [cold[0].next_index(), cold[1].next_index()]
    per_shard_even = (4 + 2) // 2
    warm = [cursor(0), cursor(1)]
    for shard in warm:
        shard.advance(per_shard_even)
    warm_first = [warm[0].next_index(), warm[1].next_index()]
    assert warm_first[0] == served[0][3], "shard 0 (drew 4, offset 3) re-serves its 4th, already-trained prompt"
    assert warm_first[1] != cold_next[1] and warm_first[1] not in served[1], (
        "shard 1 (drew 2, offset 3) skips its next unseen prompt"
    )
    exact = [cursor(0), cursor(1)]
    exact[0].advance(4)
    exact[1].advance(2)
    assert [exact[0].next_index(), exact[1].next_index()] == cold_next, "per-shard counts continue exactly"

    # Part 2 -- the contract on the source run's resolved config (the arm shape: 8 shards, dynamic sampling on).
    pool = _pool(tmp_path / "pool.jsonl", rows=600)

    def run_with(rl_extra: str, *, name: str, dynamic: bool = True) -> Path:
        run = _arm_run(tmp_path, pool, name=name)
        text = (run / "config_resolved.yaml").read_text()
        if not dynamic:
            text = text.replace("min_informative_groups: 8", "min_informative_groups: 0")
        (run / "config_resolved.yaml").write_text(text.replace("rl:\n", f"rl:\n{rl_extra}", 1))
        return run

    short = run_with("  dynamic_sampling_refill_mode: short_cells\n", name="short_cells")
    with pytest.raises(ValueError, match="dynamic_sampling_refill_mode='short_cells'.*drew UNEQUAL counts"):
        ContinuationPlan.from_run(str(short), 50, consumed=440)
    with pytest.raises(ValueError, match="dynamic_sampling_refill_mode='short_cells'"):
        ContinuationPlan.from_run(str(short), 50)  # refused before the exact-count requirement
    assert (
        ContinuationPlan.from_run(
            str(run_with("  dynamic_sampling_refill_mode: all\n", name="all")), 50, consumed=440
        ).consumed_per_shard
        == 55
    )
    assert ContinuationPlan.from_run(str(run_with("", name="absent")), 50, consumed=440).consumed_per_shard == 55
    inert = run_with("  dynamic_sampling_refill_mode: short_cells\n", name="inert", dynamic=False)
    assert ContinuationPlan.from_run(str(inert), 50).dynamic_sampling is False


def test_warm_restart_from_a_model_only_artifact_serves_the_next_batch_not_the_first(tmp_path: Path) -> None:
    """A continuation built from the plan's overrides serves the cold run's batch N+1, not its batch 1.

    Written so that a tree WITHOUT the fix fails on the first assertion below with the review's symptom -- the warm
    first batch EQUALS the cold first batch -- rather than on a missing attribute: the plan's overrides are applied
    when the plan has them, and the data role is built from whatever `rl.data` then carries.
    """
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.rl import DatasetConfig, resolve_data_seed
    from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell
    from cosmos_framework.posttrain.roles.env.dataset import DatasetEnv, jsonl_prompts
    from cosmos_framework.posttrain.utils.dotlist import apply_dotlist

    shards = 8
    per_draw = prompts_per_data_cell(policy_batch_size=32, rollouts_per_prompt=4, data_replicas=shards)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps({"messages": f"prompt {i}", "reward_spec": i}) + "\n" for i in range(600)))
    run = _arm_run(tmp_path, pool)

    def data_role(rl: Any, rollout: Any) -> list[DatasetEnv]:
        seed = resolve_data_seed(rl, rollout)
        offset = int(getattr(rl.data, "cursor_offset", 0) or 0)
        continuation = {"cursor_offset": offset, "cursor_shards": int(rl.data.cursor_shards)} if offset else {}
        items = jsonl_prompts(str(pool), rl.data.prompt_column, rl.data.response_column)
        return [
            DatasetEnv(
                items,
                shuffle=bool(rl.data.shuffle),
                seed=seed,
                repeat=True,
                shard_index=i,
                shard_count=shards,
                **continuation,
            )
            for i in range(shards)
        ]

    def draw(role: list[DatasetEnv]) -> list[str]:
        return [obs.representations["prompt"] for env in role for obs in env.get_samples(per_draw)]

    # The cold run: rl.seed 0 following rollout.seed 42; 3 steps, one dynamic-sampling top-up at step 2 -> 4 draws.
    cold_rl = SimpleNamespace(
        seed=0,
        data_seed_follows_rollout_seed=True,
        data=DatasetConfig(path=str(pool), prompt_column="messages", response_column="reward_spec", shuffle=True),
    )
    cold = data_role(cold_rl, SimpleNamespace(seed=42))
    cold_batches = [draw(cold) for draws in (1, 2, 1) for _ in range(draws)]
    consumed_total = sum(env._cursor.consumed for env in cold)
    assert consumed_total == 4 * shards * per_draw
    cold_next = draw(cold)

    # The warm run as the arm restarted it: the same recipe (rl.seed 0 following rollout.seed 42) and a fresh data
    # role -- plus the plan's overrides, which is what the arm's warm restarts lacked.
    plan = ContinuationPlan.from_run(str(run), 3, consumed=consumed_total)
    overrides = plan.overrides() if hasattr(plan, "overrides") else []
    warm_rl = SimpleNamespace(
        seed=0,
        data_seed_follows_rollout_seed=True,
        data=DatasetConfig(path=str(pool), prompt_column="messages", response_column="reward_spec", shuffle=True),
    )
    apply_dotlist(SimpleNamespace(rl=warm_rl), overrides)
    warm_first = draw(data_role(warm_rl, SimpleNamespace(seed=42)))

    assert warm_first != cold_batches[0], (
        "the warm restart re-served the cold run's first prompts (the arm log's identical step-1 gold tails)"
    )
    assert warm_first == cold_next, "the continuation must serve the cold run's NEXT batch"
    # The overrides pin the prompt order to the source's resolved seed, so a continuation that varies rollout.seed to
    # vary SAMPLING still serves the same next batch.
    assert draw(data_role(warm_rl, SimpleNamespace(seed=7))) == cold_next
    assert (warm_rl.seed, warm_rl.data_seed_follows_rollout_seed) == (42, False)
    assert (warm_rl.data.cursor_offset, warm_rl.data.cursor_shards) == (4 * per_draw, shards)


# --------------------------------------------------------------------------- continuation identity
def test_continuation_plan_binds_the_artifact_iteration_to_the_optimizer_step_clock(tmp_path: Path) -> None:
    """`iter_*` counts OPTIMIZER steps and one rollout batch advances `rl.n_mini` of them. Reading the
    iteration as rollout-batch steps over-advanced the cursor n_mini-fold and skipped unseen prompts. The plan takes
    the artifact's iteration, converts it with the source's n_mini and refuses an iteration off the batch grid."""
    pool = _pool(tmp_path / "pool.jsonl", rows=200)
    (tmp_path / "mini").mkdir()
    mini = _run_with_seeds(tmp_path / "mini", pool, rl_lines="  seed: 7\n  mini_batch_per_replica: 4\n")  # 8 // 4 = 2
    plan = ContinuationPlan.from_run(str(mini), 6)  # iter_000006 = 3 rollout batches of 8 // 4 = 2 prompts each
    assert (plan.model_step, plan.batch_steps, plan.n_mini) == (6, 3, 2)
    assert (plan.nominal_consumed, plan.consumed, plan.cursor_offset) == (6, 6, 6)  # not the 12 of a batch-step reading
    with pytest.raises(ValueError, match="model_step 5 is not a multiple of rl.n_mini=2"):
        ContinuationPlan.from_run(str(mini), 5)
    # Without mini-batching (field absent or 0) one rollout batch is one optimizer step: the iteration is the batch count.
    (tmp_path / "plain").mkdir()
    plain = _run_with_seeds(tmp_path / "plain", pool, rl_lines="  seed: 7\n  mini_batch_per_replica: 0\n")
    assert (
        ContinuationPlan.from_run(str(plain), 6).batch_steps,
        ContinuationPlan.from_run(str(plain), 6).consumed,
    ) == (6, 12)
    assert ContinuationPlan.from_run(str(plain), 5).n_mini == 1


def test_continuation_of_a_continuation_carries_the_source_offset_forward(tmp_path: Path) -> None:
    """Run B was itself an exact continuation of run A, so B's resolved config carries
    `rl.data.cursor_offset` / `cursor_shards`. A plan from B that emitted only B's draws made run C re-serve the
    prompts A and B trained. The plan reads B's base offset, requires it to be bound to B's shard count, emits A's
    offset plus B's per-shard draws, and excludes both runs' draws from the remaining-pool view."""
    from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor

    pool = _pool(tmp_path / "pool.jsonl", rows=600)

    def run_b(*, base: str, name: str) -> Path:
        run = _arm_run(tmp_path, pool, name=name)
        text = (run / "config_resolved.yaml").read_text()
        (run / "config_resolved.yaml").write_text(text.replace("    shuffle: true\n", f"    shuffle: true\n{base}", 1))
        return run

    chained = run_b(base="    cursor_offset: 55\n    cursor_shards: 8\n", name="run_b")
    plan = ContinuationPlan.from_run(str(chained), 50, consumed=440)  # B drew 440 over 8 shards = 55 per shard
    assert (plan.base_offset, plan.consumed_per_shard, plan.cursor_offset, plan.total_consumed) == (55, 55, 110, 880)
    assert "rl.data.cursor_offset=110" in plan.overrides() and "rl.data.cursor_shards=8" in plan.overrides()

    # The mechanism on the framework's cursor (shard 0 of 8, the arm's resolved seed 42): A drew 55, B drew 55 more.
    def cursor() -> IndexedCursor:
        return IndexedCursor(list(range(600)), shard_index=0, shard_count=8, shuffle=True, seed=42)

    cold = cursor()
    drawn = [cold.next_index() for _ in range(110)]
    expected_next = cold.next_index()
    only_b = cursor()
    only_b.advance(55)
    assert only_b.next_index() == drawn[55], "B's draws alone re-serve B's first prompt to run C"
    carried = cursor()
    carried.advance(plan.cursor_offset)
    assert carried.next_index() == expected_next, "A's offset plus B's draws serves the next unseen prompt"
    # The base offset must be bound to the run's own shard count; another count, or none, is refused.
    with pytest.raises(ValueError, match="computed for 4 data shard\\(s\\) but the run had 8"):
        ContinuationPlan.from_run(
            str(run_b(base="    cursor_offset: 55\n    cursor_shards: 4\n", name="b4")), 50, consumed=440
        )
    with pytest.raises(ValueError, match="computed without a shard count"):
        ContinuationPlan.from_run(str(run_b(base="    cursor_offset: 55\n", name="b0")), 50, consumed=440)
    # No base: the first continuation in a chain, exactly as before.
    first = ContinuationPlan.from_run(str(run_b(base="", name="a")), 50, consumed=440)
    assert (first.base_offset, first.cursor_offset, first.total_consumed) == (0, 55, 440)


def test_continuation_plan_refuses_coerced_counts(tmp_path: Path) -> None:
    """`int(consumed)` accepted True, 440.9 and "440" and could move an exact continuation to another
    prompt boundary; the cursor and the config refuse coercion, and so does the plan, for both counts."""
    pool = _pool(tmp_path / "pool.jsonl", rows=600)
    run = _arm_run(tmp_path, pool)
    for bad in (True, 50.0, "50"):
        with pytest.raises(TypeError, match="model_step must be an integer"):
            ContinuationPlan.from_run(str(run), bad, consumed=440)  # type: ignore[arg-type]
    for bad in (True, 440.9, "440"):
        with pytest.raises(TypeError, match="consumed must be an integer"):
            ContinuationPlan.from_run(str(run), 50, consumed=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_step must be >= 0"):
        ContinuationPlan.from_run(str(run), -8, consumed=440)
    with pytest.raises(ValueError, match="consumed must be >= 0"):
        ContinuationPlan.from_run(str(run), 50, consumed=-1)
