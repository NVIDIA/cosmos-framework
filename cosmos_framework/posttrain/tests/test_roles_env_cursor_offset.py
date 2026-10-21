# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact continuation of a model-only artifact: the prompt cursor is ADVANCED, not restarted.

The mechanism this guards (the 16-node Ultra GRPO arm of 2026-09-29/30, five warm restarts from the HF mirror of its
iter_50 artifact): a model-only artifact carries no data cursor, and the warm run rebuilt its data role with the same
pool and the same resolved data seed (`rollout.seed` 42 under `rl.data_seed_follows_rollout_seed`) and nothing else,
so every incarnation's step-1 batch was the cold run's step-1 batch -- identical `[sample] step=1` gold tails in the arm
log (lines 1630-1631 vs 4356-4357) and an identical step-1 loss/reward in four warm incarnations. `rl.data.cursor_offset`
advances each data shard's `IndexedCursor` by the draws the source shard consumed, so the continuation serves the
source run's NEXT prompts; `rl.data.cursor_shards` refuses a run whose shard count would change the per-shard stride.

These tests are torch-free (cursor, DatasetEnv, DatasetConfig). The `build_prompt_env` wiring is covered in
`test_roles_env_dataset.py`, whose helpers import the trainer stack.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.rl import DatasetConfig
from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.env.dataset import DatasetEnv

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

# The arm's shape: rl.batch_per_replica 32, rl.group_size 4, grid.shapes.rollout [8, 4] -> 8 data shards drawing
# 1 prompt each per `get_samples` call; resolved data seed 42; rl.data.shuffle true.
ARM_SEED = 42
ARM_DATA_SHARDS = 8
ARM_PROMPTS_PER_SHARD_PER_DRAW = prompts_per_data_cell(policy_batch_size=32, rollouts_per_prompt=4, data_replicas=8)


def _rows(n: int = 200) -> list[dict[str, str]]:
    return [{"prompt": f"row{i}"} for i in range(n)]


def _shards(rows: list[dict[str, str]], **continuation: int) -> list[DatasetEnv]:
    return [
        DatasetEnv(
            rows,
            shuffle=True,
            seed=ARM_SEED,
            repeat=True,
            shard_index=index,
            shard_count=ARM_DATA_SHARDS,
            **continuation,
        )
        for index in range(ARM_DATA_SHARDS)
    ]


def _draw(shards: list[DatasetEnv]) -> list[str]:
    """One `data.get_samples(prompt_draw_count)` across the data role: every shard answers with its share."""
    batch = []
    for env in shards:
        batch.extend(obs.representations["prompt"] for obs in env.get_samples(ARM_PROMPTS_PER_SHARD_PER_DRAW))
    return batch


def test_cursor_advance_equals_drawing_across_an_epoch_wrap() -> None:
    """`advance(n)` leaves the cursor exactly where n `next_index()` calls would, epoch reshuffle included."""
    drawn = IndexedCursor(list(range(10)), shuffle=True, seed=3)
    sequence = [drawn.next_index() for _ in range(25)]
    advanced = IndexedCursor(list(range(10)), shuffle=True, seed=3)
    advanced.advance(23)
    assert advanced.consumed == 23
    assert [advanced.next_index(), advanced.next_index()] == sequence[23:25]
    untouched = IndexedCursor(list(range(10)), shuffle=True, seed=3)
    untouched.advance(0)
    assert untouched.consumed == 0 and untouched.next_index() == sequence[0]


@pytest.mark.parametrize("count", [-1, True, 2.0, "3"])
def test_cursor_advance_refuses_negative_and_non_integer_counts(count: object) -> None:
    with pytest.raises(ValueError, match="advance count must be a non-negative integer"):
        IndexedCursor(list(range(4)), shuffle=True, seed=0).advance(count)  # type: ignore[arg-type]


def test_dataset_env_cursor_offset_serves_the_next_prompt_and_continues_the_id_clock() -> None:
    """The warm data role's first batch is the cold run's batch N+1, with the ids the cold run would have stamped."""
    rows = _rows()
    cold = _shards(rows)
    # Three steps; step 2 needed one dynamic-sampling top-up draw (`extra_rollouts` 1), as the arm's steps did.
    cold_batches = [_draw(cold) for _draws in (1, 2, 1) for _ in range(_draws)]
    consumed_per_shard = cold[0]._cursor.consumed
    assert consumed_per_shard == 4 and all(env._cursor.consumed == 4 for env in cold)
    cold_next = [env.reset() for env in cold]  # the batch the cold run would have trained on next (its step 4)

    # The restart the arm actually did: same pool, same seed, no cursor -> the cold step-1 batch again.
    assert _draw(_shards(rows)) == cold_batches[0]

    warm = _shards(rows, cursor_offset=consumed_per_shard, cursor_shards=ARM_DATA_SHARDS)
    warm_first = [env.reset() for env in warm]
    assert [obs.representations for obs in warm_first] == [obs.representations for obs in cold_next]
    assert [obs.representations["prompt"] for obs in warm_first] != cold_batches[0]
    assert [obs.id for obs in warm_first] == [obs.id for obs in cold_next], "the prompt-id clock continues too"
    # From here on the two roles are the same role: the next draws agree as well.
    assert _draw(warm) == _draw(cold)
    # The continuation's state is the cold run's state at that draw (what a full checkpoint would have carried).
    assert warm[0]._cursor.consumed == cold[0]._cursor.consumed


def test_dataset_env_cursor_offset_without_shuffle_is_a_plain_skip() -> None:
    env = DatasetEnv(_rows(10), shuffle=False, seed=0, repeat=True, cursor_offset=7)
    assert [env.reset().representations["prompt"] for _ in range(4)] == ["row7", "row8", "row9", "row0"]


def test_dataset_env_refuses_a_cursor_shard_count_mismatch_and_bad_offsets() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="computed for 8 data shard\\(s\\) but this source is sharded 4-way"):
        DatasetEnv(rows, shuffle=True, seed=ARM_SEED, shard_index=0, shard_count=4, cursor_offset=4, cursor_shards=8)
    with pytest.raises(ValueError, match="computed for 8 data shard"):
        DatasetEnv(rows, shuffle=True, seed=ARM_SEED, cursor_offset=4, cursor_shards=8)  # unsharded = 1 shard
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError, match="cursor_offset must be a non-negative integer"):
            DatasetEnv(rows, cursor_offset=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cursor_shards must be a non-negative integer"):
        DatasetEnv(rows, cursor_shards=-2)
    # cursor_shards=0 asserts nothing; cursor_offset=0 is the start of the sequence.
    assert DatasetEnv(rows, shuffle=True, seed=ARM_SEED, cursor_offset=0, cursor_shards=0).reset().id


def test_dataset_config_cursor_fields_validate() -> None:
    config = DatasetConfig(
        path="/pool.jsonl", prompt_column="messages", shuffle=True, cursor_offset=55, cursor_shards=8
    )
    assert (config.cursor_offset, config.cursor_shards) == (55, 8)
    assert (DatasetConfig().cursor_offset, DatasetConfig().cursor_shards) == (0, 0)
    with pytest.raises(ValueError, match="rl.data.cursor_offset must be >= 0"):
        DatasetConfig(path="/pool.jsonl", prompt_column="messages", cursor_offset=-1)
    with pytest.raises(TypeError, match="rl.data.cursor_shards must be an integer"):
        DatasetConfig(path="/pool.jsonl", prompt_column="messages", cursor_shards=True)  # type: ignore[arg-type]
    # Review of 983201ea253 (thread 157e699e): an offset without its shard count left the topology check off, so a
    # continuation that changed its replica count served a different sequence unnoticed; the pair is required together.
    with pytest.raises(ValueError, match="rl.data.cursor_offset=5 requires rl.data.cursor_shards"):
        DatasetConfig(path="/pool.jsonl", prompt_column="messages", cursor_offset=5)
    assert (
        DatasetConfig(path="/pool.jsonl", prompt_column="messages", cursor_shards=8).cursor_offset == 0
    )  # no offset: fine
    with pytest.raises(TypeError, match="rl.data.cursor_offset must be an integer"):
        DatasetConfig(path="/pool.jsonl", prompt_column="messages", cursor_offset="55")  # type: ignore[arg-type]
    # The arithmetic source synthesizes prompts from its seed: there is no cursor to continue.
    with pytest.raises(ValueError, match="require an explicit prompt source"):
        DatasetConfig(cursor_offset=3)
    with pytest.raises(ValueError, match="require an explicit prompt source"):
        DatasetConfig(cursor_shards=8)
