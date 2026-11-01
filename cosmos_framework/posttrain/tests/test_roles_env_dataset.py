# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for ArithmeticEnv (DESIGN §1.3 degenerate 1-step Env), the JSONL prompt source, and the
split_prompts dispatch adaptation (Task C2, DESIGN §2.3)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.roles.env import ArithmeticEnv
from cosmos_framework.posttrain.roles.env.dataset import (
    DatasetEnv,
    TrainValidationEnv,
    jsonl_prompts,
    split_prompts,
)
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration


def _write_jsonl(tmp_path: Path, rows: list[dict[str, object]]) -> str:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.mark.L0
@pytest.mark.CPU
def test_reset_yields_prompt_and_answer() -> None:
    env = ArithmeticEnv(seed=0, max_operand=20)
    obs = env.reset()
    reps = obs.representations
    assert "prompt" in reps and reps["prompt"].endswith("=")
    assert "reference" in reps and isinstance(reps["reference"], int)
    a, b = [int(x) for x in reps["prompt"].replace("=", "").split("+")]
    assert reps["reference"] == a + b


@pytest.mark.L0
@pytest.mark.CPU
def test_get_samples_is_deterministic_per_seed() -> None:
    a = [o.representations["prompt"] for o in ArithmeticEnv(seed=7).get_samples(5)]
    b = [o.representations["prompt"] for o in ArithmeticEnv(seed=7).get_samples(5)]
    assert a == b and len(a) == 5


@pytest.mark.L0
@pytest.mark.CPU
def test_arithmetic_env_stamps_unique_prompt_ids() -> None:
    """The data role stamps a globally-unique id on every emitted prompt (the GRPO group_id
    namespace). Distinct prompts MUST get distinct non-empty ids across replicas run with distinct
    seeds, so fan-in never merges two prompts' completions into one group.

    Two REAL replicas are distinguished by their shard coordinates (`data_shard` in
    roles/env/prompt.py threads `shard_index`/`shard_count` from the worker group's replica count),
    which is exactly what the deterministic id now keys uniqueness off of -- so this test gives each
    "replica" its own `shard_index`, mirroring how they are actually constructed in production.
    """
    ids = [o.id for o in ArithmeticEnv(seed=0, shard_index=0, shard_count=2).get_samples(5)]
    ids += [o.id for o in ArithmeticEnv(seed=1, shard_index=1, shard_count=2).get_samples(5)]
    assert all(i for i in ids)  # every id non-empty
    assert len(set(ids)) == len(ids)  # globally unique across both replicas


@pytest.mark.L0
@pytest.mark.CPU
def test_data_shard_count_follows_rollout_replicas_not_num_replicas() -> None:
    """Asymmetric data roles shard by the rollout replica count."""
    from cosmos_framework.posttrain.config.rl import GridConfig
    from cosmos_framework.posttrain.roles.env.prompt import data_shard

    config = SimpleNamespace(grid=GridConfig(num_replicas=1, shapes={"policy": [1, 8], "rollout": [4, 2]}))
    assert data_shard(config)["shard_count"] == 4
    assert data_shard(SimpleNamespace(grid=GridConfig(num_replicas=1))) == {}


@pytest.mark.L0
@pytest.mark.CPU
def test_dataset_env_stamps_unique_prompt_ids() -> None:
    """DatasetEnv (the VLM prompt path) stamps the same globally-unique id as ArithmeticEnv."""
    env = DatasetEnv(dataset=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}])
    ids = [o.id for o in env.get_samples(3)]
    assert all(i for i in ids) and len(set(ids)) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_dataset_env_shared_cursor_and_prompt_id_clock_round_trip() -> None:
    class _Dataset:
        @staticmethod
        def cursor_fingerprint() -> str:
            return "sha256:dataset-env-round-trip"

        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, str]:
            return {"prompt": f"prompt-{index}"}

    source = DatasetEnv(_Dataset(), shuffle=True, seed=7, repeat=True)
    source.get_samples(4)
    state = source.state_dict()
    expected = source.get_samples(3)
    restored = DatasetEnv(_Dataset(), shuffle=True, seed=7, repeat=True)

    assert restored.load_state_dict(state)
    actual = restored.get_samples(3)

    assert [(item.id, item.representations) for item in actual] == [
        (item.id, item.representations) for item in expected
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_idle_refills_preserve_unequal_replica_cursors_across_restore() -> None:
    class _Dataset:
        @staticmethod
        def cursor_fingerprint() -> str:
            return "sha256:idle-refill-cursors-eight-prompts"

        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> dict[str, str]:
            return {"prompt": f"prompt-{index}"}

    rows = _Dataset()
    sources = [DatasetEnv(rows, shard_index=cell, shard_count=2, repeat=True) for cell in range(2)]
    for source in sources:
        source.get_samples(1)
    idle_state = sources[0].state_dict()
    assert sources[0].get_samples(0) == []
    assert sources[0].state_dict() == idle_state
    sources[1].get_samples(2)
    assert [source.state_dict()["emitted"] for source in sources] == [1, 3]
    for cell, source in enumerate(sources):
        restored = DatasetEnv(rows, shard_index=cell, shard_count=2, repeat=True)
        assert restored.load_state_dict(source.state_dict())
        actual, expected = restored.get_samples(3), source.get_samples(3)
        assert [(item.id, item.representations) for item in actual] == [
            (item.id, item.representations) for item in expected
        ]


def _env(
    seed: int = 0,
    shard_index: int | None = None,
    shard_count: int | None = None,
    repeat: bool = False,
    n: int = 20,
) -> DatasetEnv:
    """A small `DatasetEnv` for the id-determinism tests below.

    `n=20` is deliberate: large enough that `get_samples(8)` never raises `StopIteration` even when
    split across `shard_count=2` (10 rows/shard), and small enough that `get_samples(64)` under
    `repeat=True` (below) actually WRAPS the cursor more than once -- the property the wrap test
    needs to exercise.
    """
    rows = [{"prompt": f"p{i}"} for i in range(n)]
    kwargs: dict[str, int] = {}
    if shard_index is not None:
        kwargs["shard_index"] = shard_index
        kwargs["shard_count"] = shard_count if shard_count is not None else 1
    return DatasetEnv(rows, seed=seed, repeat=repeat, **kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_prompt_ids_are_reproducible_across_envs():
    """Two identically-configured envs must emit the SAME id sequence — this is what makes
    sorted(key=(group_id, id)) a canonical order rather than an arbitrary one."""
    a = [o.id for o in _env(seed=1234, shard_index=0).get_samples(8)]
    b = [o.id for o in _env(seed=1234, shard_index=0).get_samples(8)]
    assert a == b


@pytest.mark.L0
@pytest.mark.CPU
def test_prompt_ids_stay_unique_across_replicas():
    """The property the uuid existed to guarantee (dataset.py:363, base.py:57-62): two replicas
    must never collide, or fan-in merges distinct prompts into one GRPO group and corrupts the
    per-prompt mean/std."""
    a = {o.id for o in _env(seed=1234, shard_index=0, shard_count=2).get_samples(8)}
    b = {o.id for o in _env(seed=1234, shard_index=1, shard_count=2).get_samples(8)}
    assert not (a & b)


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_prompts_maps_the_configured_columns(tmp_path: Path) -> None:
    """Columns are renamed to what the pipeline reads: the rollout takes `prompt`, rewards `reference`."""
    conversation = [{"role": "user", "content": "What is 2 + 2?"}]
    path = _write_jsonl(
        tmp_path,
        [
            {"question": "2 + 2 =", "answer": "4", "extra": "ignored"},
            {"question": conversation, "answer": "4"},
        ],
    )
    assert jsonl_prompts(path, "question", "answer") == [
        {"prompt": "2 + 2 =", "reference": "4"},
        {"prompt": conversation, "reference": "4"},
    ]
    assert jsonl_prompts(path, "question") == [
        {"prompt": "2 + 2 ="},
        {"prompt": conversation},
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_prompts_rejects_bad_rows_with_the_file_position(tmp_path: Path) -> None:
    """Downstream a dropped row is indistinguishable from a bad sample, so it fails and says where."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: not valid JSON"):
        jsonl_prompts(str(path), "question")

    missing = _write_jsonl(tmp_path, [{"question": "ok"}, {"other": "x"}])
    with pytest.raises(ValueError, match=r":2: no 'question' field"):
        jsonl_prompts(missing, "question")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no prompts"):
        jsonl_prompts(str(empty), "question")

    null_answer = _write_jsonl(tmp_path, [{"question": "ok", "answer": None}])
    with pytest.raises(ValueError, match=r":1: 'answer' must not be null"):
        jsonl_prompts(null_answer, "question", "answer")

    null_prompt = _write_jsonl(tmp_path, [{"question": None}])
    with pytest.raises(ValueError, match=r":1: 'question' must be a string or conversation list"):
        jsonl_prompts(null_prompt, "question")

    malformed_conversation = _write_jsonl(tmp_path, [{"question": [{"content": "missing role"}]}])
    with pytest.raises(ValueError, match=r":1: 'question' conversation must be a non-empty list"):
        jsonl_prompts(malformed_conversation, "question")

    empty_conversation = _write_jsonl(tmp_path, [{"question": []}])
    with pytest.raises(ValueError, match=r":1: 'question' conversation must be a non-empty list"):
        jsonl_prompts(empty_conversation, "question")


@pytest.mark.L0
@pytest.mark.CPU
def test_dataset_env_repeat_is_explicit() -> None:
    """Online JSONL can wrap, without changing the generic environment's one-pass default."""
    env = DatasetEnv([{"prompt": "a"}, {"prompt": "b"}], repeat=True)
    assert [o.representations["prompt"] for o in env.get_samples(5)] == ["a", "b", "a", "b", "a"]

    env = DatasetEnv([{"prompt": "a"}])
    assert env.reset().representations["prompt"] == "a"
    with pytest.raises(StopIteration):
        env.reset()

    with pytest.raises(ValueError, match="empty prompt source"):
        DatasetEnv([], repeat=True).reset()


@pytest.mark.L0
@pytest.mark.CPU
def test_dataset_env_shuffle_is_deterministic_per_seed() -> None:
    rows = [{"prompt": str(i)} for i in range(8)]
    seen = [
        [o.representations["prompt"] for o in DatasetEnv(rows, shuffle=True, seed=3, repeat=True).get_samples(8)]
        for _ in range(2)
    ]
    assert seen[0] == seen[1]
    assert sorted(seen[0]) == sorted(r["prompt"] for r in rows)  # a permutation, nothing dropped
    assert seen[0] != [r["prompt"] for r in rows]  # and actually shuffled


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_cursor_is_finite_restartable_and_isolated_from_training() -> None:
    """The reused data role owns two cursors: validation never advances or rewinds training."""
    train = DatasetEnv([{"prompt": f"train-{i}"} for i in range(3)], repeat=True)
    env = TrainValidationEnv(
        train,
        [{"prompt": f"val-{i}"} for i in range(4)],
        seed=7,
        max_samples=3,
    )

    assert env.get_samples(1)[0].representations["prompt"] == "train-0"
    assert env.begin_validation(batch_size=2) == 3
    assert [o.representations["prompt"] for o in env.get_validation_samples(2, expected_count=2)] == [
        "val-0",
        "val-1",
    ]
    assert [o.representations["prompt"] for o in env.get_validation_samples(2, expected_count=1)] == ["val-2"]
    assert env.get_validation_samples(2, expected_count=0) == []  # finite: no implicit wrap
    assert env.get_samples(1)[0].representations["prompt"] == "train-1"

    # The replica checks the inventory-derived count before its payload leaves the data plane.
    assert env.begin_validation(batch_size=2) == 3
    with pytest.raises(ValueError, match="expected_count must be an integer"):
        env.get_validation_samples(2, expected_count=True)
    with pytest.raises(RuntimeError, match="returned 2 samples, expected 1"):
        env.get_validation_samples(2, expected_count=1)

    # A new validation run starts the held-out pass from the beginning without touching training.
    assert env.begin_validation(batch_size=2) == 3
    assert [o.representations["prompt"] for o in env.get_validation_samples(3)] == [
        "val-0",
        "val-1",
        "val-2",
    ]
    assert env.get_samples(1)[0].representations["prompt"] == "train-2"


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_cap_is_applied_before_replica_sharding() -> None:
    """`max_samples=N` means N logical examples globally, not N on every data replica."""
    rows = [{"prompt": f"val-{i}"} for i in range(8)]
    envs = [
        TrainValidationEnv(
            DatasetEnv([{"prompt": f"train-{i}"} for i in range(2)], repeat=True),
            rows,
            max_samples=5,
            shard_index=replica,
            shard_count=2,
        )
        for replica in range(2)
    ]
    assert [env.begin_validation(batch_size=8) for env in envs] == [3, 2]
    seen = [obs.representations["prompt"] for env in envs for obs in env.get_validation_samples(8)]
    assert sorted(seen) == [f"val-{i}" for i in range(5)]


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_allows_empty_physical_shards() -> None:
    """A global sample cap may be smaller than DP; empty replicas contribute neutral moments."""
    rows = [{"prompt": "only-global-sample"}]
    envs = [
        TrainValidationEnv(
            DatasetEnv([{"prompt": "train"}], repeat=True),
            rows,
            max_samples=1,
            shard_index=replica,
            shard_count=2,
        )
        for replica in range(2)
    ]
    assert [env.begin_validation() for env in envs] == [1, 0]
    assert [len(env.get_validation_samples(1)) for env in envs] == [1, 0]


@pytest.mark.L0
@pytest.mark.CPU
def test_split_prompts_partitions_contiguously_without_overlap() -> None:
    prompts = list(range(8))
    shards = [split_prompts(prompts, i, 4) for i in range(4)]
    assert shards == [[0, 1], [2, 3], [4, 5], [6, 7]]  # contiguous, disjoint
    assert [p for s in shards for p in s] == prompts  # union == whole list, order preserved


@pytest.mark.L0
@pytest.mark.CPU
def test_split_prompts_uneven_last_shard_absorbs_remainder() -> None:
    prompts = list(range(7))  # 7 not divisible by 3
    shards = [split_prompts(prompts, i, 3) for i in range(3)]
    # floor-division tiling: i*7//3 boundaries = 0,2,4,7 -> sizes 2,2,3; still covers all 7 once.
    assert shards == [[0, 1], [2, 3], [4, 5, 6]]
    assert sum(len(s) for s in shards) == 7


@pytest.mark.L0
@pytest.mark.CPU
def test_split_prompts_deterministic_per_index_across_calls() -> None:
    # Every SPMD rank of one consumer replica calls with the SAME (i, n) and MUST get the identical
    # slice (spec §2.2: divergent per-rank prompts would desync the replica's collectives).
    prompts = [f"p{k}" for k in range(10)]
    assert split_prompts(prompts, 1, 5) == split_prompts(list(prompts), 1, 5) == ["p2", "p3"]


@pytest.mark.L0
@pytest.mark.CPU
def test_split_prompts_n_one_returns_whole_list() -> None:
    # op == "one" never calls split, but n==1 must still be the identity slice (defensive).
    prompts = [1, 2, 3]
    assert split_prompts(prompts, 0, 1) == prompts


@pytest.mark.L0
@pytest.mark.CPU
def test_dataset_env_replicas_read_disjoint_items():
    """Each replica strides through the corpus; together they cover it exactly once."""
    from cosmos_framework.posttrain.roles.env.dataset import DatasetEnv

    rows = [{"prompt": f"p{i}"} for i in range(12)]
    got = [
        [o.representations["prompt"] for o in DatasetEnv(list(rows), shard_index=i, shard_count=3).get_samples(4)]
        for i in range(3)
    ]
    flat = [t for part in got for t in part]
    assert len(set(flat)) == len(flat) == 12, f"shards overlap or lose items: {got}"
    assert set(flat) == {r["prompt"] for r in rows}, "the shards must COVER the corpus"


@pytest.mark.L0
@pytest.mark.CPU
def test_omitted_shard_coords_do_NOT_read_placement(monkeypatch):
    """Omitting the coordinates must mean NO sharding — not "read the placement coordinates".

    `replica_coords()` publishes the COLOCATION's replica count, i.e. the max over its members. In a
    supported fan-out — data(replicas=1) -> rollout(replicas=2) sharing a colocation — the single
    data actor would read 0/2 and keep only `items[0::2]`. Half the corpus is then never sampled,
    and there is no second data actor to read the rest: silent DATA LOSS, which is worse than the
    silent duplication that defaulting to placement was meant to prevent.
    """
    from cosmos_framework.posttrain.config.build import REPLICA_COUNT_ENV, REPLICA_INDEX_ENV
    from cosmos_framework.posttrain.roles.env.dataset import DatasetEnv, _resolve_shard

    # Placement says "cell 0 of 2" — the colocation's shape, not this role's.
    monkeypatch.setenv(REPLICA_INDEX_ENV, "0")
    monkeypatch.setenv(REPLICA_COUNT_ENV, "2")
    assert _resolve_shard(None, None) == (0, 1), "omitted coords must not inherit the colocation count"

    rows = [{"prompt": f"p{i}"} for i in range(6)]
    env = DatasetEnv(list(rows), shard_index=None, shard_count=None)
    got = [o.representations["prompt"] for o in env.get_samples(6)]
    assert sorted(got) == [f"p{i}" for i in range(6)], f"rows dropped without explicit sharding: {got}"

    # Explicit coordinates still shard.
    a = DatasetEnv(list(rows), shard_index=0, shard_count=2).get_samples(3)
    b = DatasetEnv(list(rows), shard_index=1, shard_count=2).get_samples(3)
    both = [o.representations["prompt"] for o in a] + [o.representations["prompt"] for o in b]
    assert sorted(both) == [f"p{i}" for i in range(6)]


class _CountingDataset:
    """Map-style, counts row access — proves the view never materializes."""

    def __init__(self, rows):
        self.rows, self.reads = rows, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        self.reads += 1
        return self.rows[i]


@pytest.mark.L0
@pytest.mark.CPU
def test_column_mapped_is_lazy_and_renames_without_dropping():
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    ds = _CountingDataset([{"q": "a", "gold": "1", "junk": "x"} for _ in range(50)])
    view = _ColumnMapped(ds, prompt_column="q", response_column="gold")
    assert ds.reads == 0, "constructing the view must not touch a single row"
    assert len(view) == 50
    item = view[3]
    assert ds.reads == 1, "one access reads exactly one row"
    assert item == {"prompt": "a", "reference": "1", "junk": "x"}, (
        "the two wired columns are RENAMED and every other column passes through; projecting them "
        "away would hide fields a reward or callback may legitimately read"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_column_mapped_omits_answer_without_a_response_column():
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    view = _ColumnMapped(_CountingDataset([{"q": "a"}]), prompt_column="q")
    assert view[0] == {"prompt": "a"}


@pytest.mark.L0
@pytest.mark.CPU
def test_column_mapped_row_map_rewrites_the_prompt_column_in_place():
    """`row_map` is how a recipe adds a system prompt WITHOUT a config field for it.

    It is a MAP: row in, row out, same keys. The prompt column is rewritten in place, so
    `prompt_column` still names the prompt afterwards -- the property that broke when a builder
    replaced the prompt wholesale and silently made `prompt_column` unreadable.
    """
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    view = _ColumnMapped(
        _CountingDataset([{"q": "2+2?", "gold": "4"}]),
        prompt_column="q",
        response_column="gold",
        row_map=lambda r: {**r, "q": [{"role": "system", "content": "S"}, {"role": "user", "content": r["q"]}]},
    )
    assert view[0] == {
        "prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "2+2?"}],
        "reference": "4",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_gsm8k_items_fn_emits_a_conversation_with_the_system_prompt():
    """Ported from test_roles_env_gsm8k.py. The env no longer templates — it emits the CONVERSATION,
    and the DataPacker renders it. That is what makes a multimodal prompt expressible at all."""
    from cosmos_framework.posttrain.config.rl import DatasetConfig
    from cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k import (
        GSM8K_SYSTEM_PROMPT,
        gsm8k_items,
    )
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    rows = [{"question": "2+2?", "answer": "four #### 4"}, {"question": "3+5?", "answer": "eight #### 8"}]
    cfg = DatasetConfig(
        name="openai/gsm8k", subset="main", split="train", prompt_column="question", response_column="answer"
    )
    view = _ColumnMapped(rows, cfg.prompt_column, cfg.response_column, row_map=gsm8k_items)
    for i, row in enumerate(rows):
        item = view[i]
        assert item["prompt"] == [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        assert item["reference"] == row["answer"]


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos_rl_items_fn_matches_the_reference_wording_byte_for_byte():
    """The parity-only wording must track cosmos-rl's own dataset, not a paraphrase of it.

    The prompt is worth ~0.20 mean reward on GSM8K, so a cross-stack curve comparison that runs a
    drifted copy of the reference wording measures the drift. The literal is read out of
    `cosmos_rl/tools/dataset/gsm8k_grpo.py` rather than restated here, so an edit on either side
    fails this instead of silently changing what the comparison means.
    """
    import ast
    from pathlib import Path

    from cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k import gsm8k_items_cosmos_rl

    source = Path(__file__).resolve().parents[4] / "packages/cosmos-rl/cosmos_rl/tools/dataset/gsm8k_grpo.py"
    literals = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.JoinedStr) and "think step by step" in ast.unparse(node) and "####" in ast.unparse(node)
    ]
    assert literals, f"cosmos-rl's GSM8K prompt f-string is no longer in {source}"
    reference = eval(literals[-1], {"question": "2+2?"})  # noqa: S307 — literal from a repo file

    mapped = gsm8k_items_cosmos_rl({"question": "2+2?", "answer": "4"})
    assert mapped["question"] == [{"role": "user", "content": reference}]
    assert mapped["answer"] == "4", "columns the map does not touch must pass through unchanged"


@pytest.mark.L0
@pytest.mark.CPU
def test_gsm8k_rows_shard_disjointly_through_dataset_env():
    """Ported from test_roles_env_gsm8k.py::test_data_replicas_read_disjoint_rows."""
    from cosmos_framework.posttrain.roles.env import DatasetEnv

    rows = [{"prompt": f"q{i}", "reference": f"#### {i}"} for i in range(12)]

    def _seen(index, count, n=4):
        env = DatasetEnv(list(rows), shard_index=index, shard_count=count, repeat=True)
        return {o.representations["prompt"] for o in env.get_samples(n)}

    shards = [_seen(i, 3) for i in range(3)]
    assert all(len(s) == 4 for s in shards), f"each of 3 shards holds 4 of 12: {shards}"
    assert not (shards[0] & shards[1] or shards[0] & shards[2] or shards[1] & shards[2]), shards
    assert set().union(*shards) == {r["prompt"] for r in rows}, "shards must COVER the dataset"


@pytest.mark.L0
@pytest.mark.CPU
def test_more_replicas_than_rows_is_refused():
    """Ported from test_roles_env_gsm8k.py::test_shard_smaller_than_replica_count_is_refused."""
    from cosmos_framework.posttrain.roles.env import DatasetEnv

    # AT CONSTRUCTION, not on the first reset(): deferred, every data actor and vLLM engine is
    # built before this fires. No `.reset()` here on purpose -- that is the regression.
    with pytest.raises(ValueError, match="would get an empty shard"):
        DatasetEnv([{"prompt": "q0"}], shard_index=1, shard_count=2, repeat=True)
    # the cursor keeps its own copy of the check, reachable through its other constructors
    from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor

    with pytest.raises(ValueError, match="would get an empty shard"):
        IndexedCursor([{"prompt": "q0"}], shard_index=1, shard_count=2).next_index()
    with pytest.raises(ValueError, match="need 0 <= index < count"):
        DatasetEnv([{"prompt": "q0"}], shard_index=2, shard_count=2, repeat=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_fingerprint_distinguishes_same_length_different_content():
    """The whole point of the fingerprint, and what `len=N` alone could not do.

    The cursor already stores seed/epoch/position/dataset_len, which REGENERATE the permutation.
    The fingerprint answers a different question -- whether those indices still point at the same
    examples. Before this, a HF source fell back to `f"len={N}"`, which duplicates the separately
    stored `dataset_len` field and so validated nothing: a hub revision bump or a regenerated
    corpus of equal length restored onto different data silently.

    The stand-ins below mirror the surface verified on datasets 4.6.1 against openai/gsm8k
    (`info.dataset_name`/`config_name`/`version`, `.split`, `._fingerprint`); the real API is
    exercised separately, since CI cannot reach the hub.
    """
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    class _Info:
        dataset_name, config_name, version = "gsm8k", "main", "0.0.0"

    class _HFLike:
        info, split = _Info(), "train"

        def __init__(self, arrow):
            self._fingerprint = arrow

        def __len__(self):
            return 7473

        def __getitem__(self, i):
            return {"question": "q", "answer": "a"}

    view = lambda ds: _ColumnMapped(ds, "question", "answer")  # noqa: E731
    a = view(_HFLike("92d68ca35e1f0878")).cursor_fingerprint()
    b = view(_HFLike("c82b2850d1bb4bcc")).cursor_fingerprint()

    assert a != b, "same length, different content must NOT share a fingerprint"
    assert "len=7473" in a, "the bare length must still be present -- never identify less than before"
    assert "dataset_name=gsm8k" in a and "split=train" in a


@pytest.mark.L0
@pytest.mark.CPU
def test_fingerprint_degrades_when_the_private_arrow_hash_disappears():
    """`_fingerprint` is underscore-private, so a datasets upgrade may drop or rename it.

    That must degrade to the public identity, never raise -- a library bump should not make every
    checkpoint unsavable.
    """
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    class _Info:
        dataset_name, config_name, version = "gsm8k", "main", "0.0.0"

    class _NoArrow:
        info, split = _Info(), "train"

        def __len__(self):
            return 7473

        def __getitem__(self, i):
            return {"question": "q", "answer": "a"}

    fp = _ColumnMapped(_NoArrow(), "question", "answer").cursor_fingerprint()
    assert "arrow=" not in fp
    assert "dataset_name=gsm8k" in fp and "len=7473" in fp


@pytest.mark.L0
@pytest.mark.CPU
def test_a_source_with_its_own_fingerprint_still_wins():
    """DPO's datasets compute a real content digest; that must not be overridden."""
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    class _Own:
        def cursor_fingerprint(self):
            return "sha256:deadbeef"

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"question": "q", "answer": "a"}

    assert _ColumnMapped(_Own(), "question", "answer").cursor_fingerprint() == "sha256:deadbeef"


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_stream_fingerprint_includes_mapping_and_transform() -> None:
    """Logical identity hashes transformed rows, including globals/closures source hashes miss."""
    import random

    torch = pytest.importorskip("torch")

    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped
    from cosmos_framework.posttrain.roles.env.identity import dataset_fingerprint
    from cosmos_framework.posttrain.roles.env.prompt import materialize_validation_items

    class _Own:
        @staticmethod
        def cursor_fingerprint() -> str:
            return "sha256:raw"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, str]:
            return {"question": "q", "alternate": "q", "answer": "a"}

    transform_state = {"prefix": "first"}

    def mapped(row: dict[str, str]) -> dict[str, str]:
        return {**row, "question": f"{transform_state['prefix']}:{row['question']}:{random.random()}"}

    view = _ColumnMapped(_Own(), "question", "answer", mapped)
    random.seed(123)
    rng_before = random.getstate()
    first_rows = materialize_validation_items(view, max_samples=None, seed=7)
    assert random.getstate() == rng_before
    first_fingerprint = dataset_fingerprint(first_rows)

    # The same validation seed governs a random items_fn independently of ambient RNG state.
    random.seed(999)
    second_rows = materialize_validation_items(view, max_samples=None, seed=7)
    assert second_rows == first_rows

    # A referenced mutable global/closure changes actual logical rows and therefore resume identity,
    # even though callable source and the raw HF source identity are unchanged.
    transform_state["prefix"] = "second"
    changed_rows = materialize_validation_items(view, max_samples=None, seed=7)
    assert dataset_fingerprint(changed_rows) != first_fingerprint
    assert view.raw_source_fingerprint() == "sha256:raw"

    def reseed(row: dict[str, str]) -> dict[str, str]:
        torch.manual_seed(123)
        return row

    reseeding_view = _ColumnMapped([{"question": "q", "answer": "a"}], "question", "answer", reseed)
    with pytest.raises(RuntimeError, match="items_fn must not reseed PyTorch"):
        materialize_validation_items(reseeding_view, max_samples=None, seed=7)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_stream_fingerprint_uses_multimodal_content(tmp_path: Path) -> None:
    """Equal image/tensor/array payloads have one process-independent logical identity."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")

    from cosmos_framework.posttrain.roles.env.identity import dataset_fingerprint

    pixels = bytes([0, 1, 2, 3, 4, 5])
    image_a = image_module.frombytes("RGB", (2, 1), pixels)
    image_b = image_module.frombytes("RGB", (2, 1), bytes(pixels))
    tensor_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # [1, 2]
    tensor_b = tensor_a.clone()  # [1, 2]
    array_a = np.asarray([[3, 4]], dtype=np.int16)  # [1, 2]
    array_b = array_a.copy()  # [1, 2]

    first = [{"image": image_a, "tokens": tensor_a, "features": array_a}]
    equivalent = [{"features": array_b, "tokens": tensor_b, "image": image_b}]
    assert dataset_fingerprint(first) == dataset_fingerprint(equivalent)

    changed = [{"image": image_b, "tokens": tensor_b + 1.0, "features": array_b}]  # [1, 2]
    assert dataset_fingerprint(first) != dataset_fingerprint(changed)

    palette_a = image_module.new("P", (1, 1), color=0)
    palette_b = image_module.new("P", (1, 1), color=0)
    palette_a.putpalette([255, 0, 0] + [0] * (256 * 3 - 3))
    palette_b.putpalette([0, 255, 0] + [0] * (256 * 3 - 3))
    assert dataset_fingerprint([{"image": palette_a}]) != dataset_fingerprint([{"image": palette_b}])

    media_a = tmp_path / "a.bin"
    media_b = tmp_path / "b.bin"
    media_a.write_bytes(b"same-content")
    media_b.write_bytes(b"same-content")
    assert dataset_fingerprint([{"media": media_a}]) == dataset_fingerprint([{"media": media_b}])
    media_b.write_bytes(b"changed-content")
    assert dataset_fingerprint([{"media": media_a}]) != dataset_fingerprint([{"media": media_b}])

    class _Opaque:
        pass

    with pytest.raises(TypeError, match="opaque"):
        dataset_fingerprint([{"payload": _Opaque()}])


@pytest.mark.L0
@pytest.mark.CPU
def test_gsm8k_map_writes_the_column_the_config_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    """The row map and `rl.data.prompt_column` must name the SAME column.

    They were stated independently: the map hardcoded "question" and build_config separately set
    prompt_column="question". If those drift -- or prompt_column is overridden to another column the
    dataset also has -- the map writes the conversation to one column while `_ColumnMapped` reads
    the prompt from another, so training silently sees the RAW question with no system prompt. A
    missing column raises; this is the case that does not.
    """
    from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
    from cosmos_framework.posttrain.configs import grpo_qwen3b_gsm8k as recipe

    skip_recipe_registration(monkeypatch, recipe._make_root_config, PosttrainGRPOConfig)
    declared = recipe.build_config().rl.data.prompt_column
    mapped = recipe.gsm8k_items({"question": "2+2?", "answer": "4"})

    assert isinstance(mapped[declared], list), (
        f"the map must rewrite the column the config declares as the prompt ({declared!r}); "
        f"it wrote {[k for k, v in mapped.items() if isinstance(v, list)]!r} instead"
    )
    assert mapped[declared][0]["role"] == "system", "the system prompt must survive the map"
    assert mapped["answer"] == "4", "columns the map does not touch must pass through unchanged"


class _FingerprintedRows:
    """Map-style rows with the stable fingerprint `project_prompt_items` requires for exact resume."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    @staticmethod
    def cursor_fingerprint() -> str:
        return "sha256:conversation-rows"


def _projected(rows: list[dict[str, object]]):
    from cosmos_framework.posttrain.roles.env.dataset import project_prompt_items

    return project_prompt_items(
        _FingerprintedRows(rows),
        source_name="pool.jsonl",
        split="train",
        prompt_column="messages",
        response_column="reward_spec",
        modality="text",
        seed=42,
        fingerprint_prefix="ppo-prompt",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_project_prompt_items_serves_the_conversation_lists_the_grpo_loader_serves(tmp_path: Path) -> None:
    """The PPO data role (`entrypoints/ppo.py::build_data` -> `project_prompt_items`) and the GRPO loader
    (`jsonl_prompts`) read the same multimodal pool rows: a conversation list in the prompt column with the
    image or video inside the content parts. The projection used to refuse every non-string prompt
    ("dataset prompt values must be non-empty strings"), so no such pool could train through PPO while
    the GRPO role served it; now both serve the list untouched. The string path is byte-for-byte the old
    one."""
    from cosmos_framework.posttrain.roles.env.dataset import is_conversation_prompt
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS, write_conversation_jsonl

    view = _projected(CONVERSATION_ROWS)
    items = [view[index] for index in range(len(CONVERSATION_ROWS))]

    for item, row in zip(items, CONVERSATION_ROWS, strict=True):
        assert item == {"prompt": row["messages"], "reference": row["reward_spec"]}
        assert is_conversation_prompt(item["prompt"])
    grpo_rows = jsonl_prompts(write_conversation_jsonl(tmp_path, CONVERSATION_ROWS), "messages", "reward_spec")
    assert [item["prompt"] for item in items] == [row["prompt"] for row in grpo_rows]
    # The fifth row's turn content is a plain string (3,282 of the pool's 18,270 rows): served as that string.
    assert len(items) == 5 and isinstance(items[4]["prompt"][0]["content"], str)
    assert items[4]["prompt"][0]["content"].startswith("Q: Translate")
    assert _projected([{"messages": "2 + 2 =", "reward_spec": "4"}])[0] == {"prompt": "2 + 2 =", "reference": "4"}


@pytest.mark.L0
@pytest.mark.CPU
def test_project_prompt_items_refuses_malformed_prompts_and_names_both_shapes() -> None:
    """One grammar for the list (`is_conversation_prompt`, shared with `jsonl_prompts`) plus the
    projection's own non-emptiness contract on every turn (the contract it already had for strings);
    the refusal names both accepted shapes and happens when the data role is built (row 0 is projected at
    construction). A `None`, a number and an empty string are refused as before."""
    malformed: list[tuple[str, object]] = [
        ("empty list", []),
        ("a non-dict turn", ["not a dict"]),
        ("a turn without role", [{"content": "no role"}]),
        ("a turn without content", [{"role": "user"}]),
        ("an empty role", [{"role": "", "content": "x"}]),
        ("an empty list content", [{"role": "user", "content": []}]),
        ("an empty string content", [{"role": "user", "content": ""}]),
        ("a non-string non-list content", [{"role": "user", "content": 5}]),
        ("an empty string prompt", ""),
        ("a null prompt", None),
        ("a numeric prompt", 5),
    ]
    for label, prompt in malformed:
        with pytest.raises(ValueError, match=r"non-empty strings or conversation lists .* for single-turn training"):
            _projected([{"messages": prompt, "reward_spec": "x"}])  # row 0 is projected at construction
        print(f"refused: {label}")  # noqa: T201 -- which shape failed, in the pytest -rA output


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_rows_serves_the_file_as_written_with_a_content_fingerprint(tmp_path: Path) -> None:
    """`JsonlRows` is the PPO data role's JSONL source. It exists because Hugging Face
    ``load_dataset("json")`` infers one struct type over every content block of a conversation column
    and null-fills the keys a block lacks -- a text block comes back as ``{"type": "text", "text": ...,
    "image": None, "video": None, ...}`` and a chat template testing ``'image' in content`` renders it as
    an image placeholder with the text gone. This source parses each line with `json.loads` (the
    `jsonl_prompts` reader, same errors with the file position) and serves the objects untouched, with the
    content fingerprint `project_prompt_items` needs for exact resume."""
    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS, write_conversation_jsonl

    path = write_conversation_jsonl(tmp_path, CONVERSATION_ROWS)
    rows = JsonlRows(path, fingerprint_prefix="ppo-jsonl")

    assert len(rows) == len(CONVERSATION_ROWS)
    assert [rows[index] for index in range(len(rows))] == CONVERSATION_ROWS
    assert [sorted(block) for turn in rows[3]["messages"] for block in turn["content"]] == [
        ["text", "type"],
        ["text", "type"],
        ["fps", "frames_indices", "total_num_frames", "type", "video"],
    ]
    # The string-content row is served as the string it wrote, with the column the projection drops.
    assert isinstance(rows[4]["messages"][0]["content"], str) and rows[4]["mix_source"] == "textif"
    fingerprint = rows.cursor_fingerprint()
    assert fingerprint.startswith("ppo-jsonl-sha256=") and fingerprint.endswith("|len=5")
    assert JsonlRows(path, fingerprint_prefix="ppo-jsonl").cursor_fingerprint() == fingerprint
    shorter = write_conversation_jsonl(tmp_path, CONVERSATION_ROWS[:3], name="shorter.jsonl")
    assert JsonlRows(shorter, fingerprint_prefix="ppo-jsonl").cursor_fingerprint() != fingerprint

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: not valid JSON"):
        JsonlRows(str(bad))
    not_an_object = tmp_path / "list.jsonl"
    not_an_object.write_text('{"messages": "ok"}\n5\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"list\.jsonl:2: expected a JSON object, got int"):
        JsonlRows(str(not_an_object))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        JsonlRows(str(empty))
    with pytest.raises(ValueError, match="not readable"):
        JsonlRows(str(tmp_path / "missing.jsonl"))


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_rows_reads_the_three_local_file_shapes_the_grpo_loader_shares(tmp_path: Path) -> None:
    """The local-rows contract of `rl.data.path` (and of the GRPO loader, which shares the reader): one JSON
    object per line, the same gzip-compressed (detected by the magic bytes, whatever the suffix), or one
    top-level JSON array of objects (one line or pretty-printed); a leading UTF-8 BOM is dropped. Every shape
    serves the rows as written. Hugging Face ``load_dataset("json")`` -- the reader PPO used before -- accepted
    the same shapes, so no file a PPO configuration pointed at is refused now. An array element that is not an
    object, an array the file does not finish, and a gzip stream that is not one are refused with the position."""
    import gzip

    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS, write_conversation_jsonl

    plain = Path(write_conversation_jsonl(tmp_path, CONVERSATION_ROWS))
    shapes = {
        "pool.jsonl.gz": gzip.compress(plain.read_bytes()),
        "pool_gz_without_suffix.jsonl": gzip.compress(plain.read_bytes()),
        "pool.json": json.dumps(CONVERSATION_ROWS, ensure_ascii=False).encode("utf-8"),
        "pool_pretty.json": json.dumps(CONVERSATION_ROWS, ensure_ascii=False, indent=2).encode("utf-8"),
        "pool_bom.jsonl": b"\xef\xbb\xbf" + plain.read_bytes(),
    }
    reference_rows = jsonl_prompts(str(plain), "messages", "reward_spec")
    for name, payload in shapes.items():
        path = tmp_path / name
        path.write_bytes(payload)
        rows = JsonlRows(str(path), fingerprint_prefix="ppo-jsonl")
        assert [rows[index] for index in range(len(rows))] == CONVERSATION_ROWS, name
        assert rows.cursor_fingerprint().startswith("ppo-jsonl-sha256=") and rows.cursor_fingerprint().endswith(
            "|len=5"
        )
        assert jsonl_prompts(str(path), "messages", "reward_spec") == reference_rows, name
    # The fingerprint is the identity of the file's bytes: equal bytes under another name are the same source,
    # the same rows in another shape are another source.
    fingerprints = {name: JsonlRows(str(tmp_path / name)).cursor_fingerprint() for name in shapes}
    assert fingerprints["pool.jsonl.gz"] == fingerprints["pool_gz_without_suffix.jsonl"]
    assert len(set(fingerprints.values()) | {JsonlRows(str(plain)).cursor_fingerprint()}) == 5

    not_objects = tmp_path / "ints.json"
    not_objects.write_text('[{"messages": "ok"}, 5]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"ints\.json:2: expected a JSON object, got int"):
        JsonlRows(str(not_objects))
    unfinished = tmp_path / "unfinished.json"
    unfinished.write_text('[\n  {"messages": "ok"},\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"unfinished\.json: not a valid JSON array"):
        JsonlRows(str(unfinished))
    corrupt = tmp_path / "corrupt.jsonl.gz"
    corrupt.write_bytes(b"\x1f\x8b" + b"not a gzip stream")
    with pytest.raises(ValueError, match=r"corrupt\.jsonl\.gz' is not readable"):
        JsonlRows(str(corrupt))


@pytest.mark.L0
@pytest.mark.CPU
def test_build_prompt_env_threads_cursor_offset_to_every_data_shard(tmp_path: Path, monkeypatch) -> None:
    """`rl.data.cursor_offset` / `cursor_shards` reach each data actor's DatasetEnv through `build_prompt_env`.

    The arm's shape: grid.shapes.rollout [8, 4] -> 8 data shards; each shard draws one prompt per call. A warm
    data role built with the offset serves, per shard, the cold shard's draw offset + 1 (its NEXT prompt), not its
    first one; a run whose rollout replica count differs from `cursor_shards` is refused by the data role.
    """
    from cosmos_framework.posttrain.config.build import REPLICA_COUNT_ENV, REPLICA_INDEX_ENV
    from cosmos_framework.posttrain.config.rl import DatasetConfig
    from cosmos_framework.posttrain.roles.env.prompt import build_prompt_env

    path = _write_jsonl(tmp_path, [{"messages": f"prompt {i}", "reward_spec": i} for i in range(300)])
    shards, offset = 8, 4
    config = SimpleNamespace(grid=SimpleNamespace(rollout_shape=lambda: (shards, 4)), rollout=SimpleNamespace(seed=42))

    def role(data: DatasetConfig) -> list[Any]:
        envs = []
        for index in range(shards):
            monkeypatch.setenv(REPLICA_INDEX_ENV, str(index))
            monkeypatch.setenv(REPLICA_COUNT_ENV, str(shards))
            rl = SimpleNamespace(
                seed=0, data_seed_follows_rollout_seed=True, data=data, validation=None, max_operand=20
            )
            envs.append(build_prompt_env(rl, config))
        return envs

    cold = role(DatasetConfig(path=path, prompt_column="messages", response_column="reward_spec", shuffle=True))
    for _ in range(offset):
        for env in cold:
            env.reset()
    cold_next = [env.reset().representations["prompt"] for env in cold]
    warm = role(
        DatasetConfig(
            path=path,
            prompt_column="messages",
            response_column="reward_spec",
            shuffle=True,
            cursor_offset=offset,
            cursor_shards=shards,
        )
    )
    assert [env.reset().representations["prompt"] for env in warm] == cold_next
    assert all(env._cursor.consumed == offset + 1 for env in warm)

    config_four = SimpleNamespace(grid=SimpleNamespace(rollout_shape=lambda: (4, 8)), rollout=SimpleNamespace(seed=42))
    monkeypatch.setenv(REPLICA_INDEX_ENV, "0")
    monkeypatch.setenv(REPLICA_COUNT_ENV, "4")
    mismatched = SimpleNamespace(
        seed=0,
        data_seed_follows_rollout_seed=True,
        data=DatasetConfig(
            path=path,
            prompt_column="messages",
            response_column="reward_spec",
            shuffle=True,
            cursor_offset=offset,
            cursor_shards=shards,
        ),
        validation=None,
        max_operand=20,
    )
    with pytest.raises(ValueError, match="computed for 8 data shard"):
        build_prompt_env(mismatched, config_four)


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reader_names_the_first_bad_row_in_file_order_and_non_utf8_bytes(tmp_path: Path) -> None:
    """The shared reader yields rows one at a time, so the loader's per-row checks run in file order and the
    refusal names the FIRST bad row: with line 1 lacking the column and line 2 not JSON, `jsonl_prompts` says
    ``:1: no 'messages' field`` (as it did before the reader was shared) while `JsonlRows`, which has no column
    contract, reaches line 2. Bytes outside UTF-8 are a `ValueError` with the position, not a bare
    `UnicodeDecodeError` escaping from the text layer. A ``[`` line after a row is a non-object row refused
    with its line number, not the start of an array."""
    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows, _jsonl_objects

    order = tmp_path / "order.jsonl"
    order.write_text('{"nope": 1}\nnot json\n', encoding="utf-8")
    rows = _jsonl_objects(str(order))
    assert next(rows) == (1, {"nope": 1})  # line 1 is served before line 2 is parsed
    with pytest.raises(ValueError, match=r"order\.jsonl:2: not valid JSON"):
        next(rows)
    with pytest.raises(ValueError, match=r"order\.jsonl:1: no 'messages' field \(has \['nope'\]\)"):
        jsonl_prompts(str(order), "messages")
    with pytest.raises(ValueError, match=r"order\.jsonl:2: not valid JSON"):
        JsonlRows(str(order))

    not_utf8 = tmp_path / "bytes.jsonl"
    not_utf8.write_bytes(b'{"messages": "ok"}\n{"messages": "\xff"}\n')
    for read in (lambda: jsonl_prompts(str(not_utf8), "messages"), lambda: JsonlRows(str(not_utf8))):
        with pytest.raises(ValueError, match=r"bytes\.jsonl:2: not UTF-8 text \('utf-8' codec can't decode byte 0xff"):
            read()

    # The array shape is decided on the file's FIRST non-blank line only. A `[` line after a row is a non-object
    # row of a JSONL file -- refused with ITS line number, as the loader did before the reader was shared -- not
    # the start of an array that swallows the rest of the file (which accepted a shape no reader had accepted and
    # pointed a refusal at a valid line).
    array_tail = tmp_path / "tail.jsonl"
    array_tail.write_bytes(b'{"messages": "ok"}\n[{"messages": "tail"}]\n')
    ints_tail = tmp_path / "ints_tail.jsonl"
    ints_tail.write_bytes(b'{"messages": "ok"}\n[1, 2]\n{"messages": "after"}\n')
    for path, pattern in (
        (array_tail, r"tail\.jsonl:2: expected a JSON object, got list"),
        (ints_tail, r"ints_tail\.jsonl:2: expected a JSON object, got list"),
    ):
        with pytest.raises(ValueError, match=pattern):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=pattern):
            JsonlRows(str(path))


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reader_reads_a_pretty_printed_single_object_as_one_row(tmp_path: Path) -> None:
    """One JSON OBJECT pretty-printed over several lines is one row of the local-rows contract -- the Hugging Face
    reader PPO used before read such a file as ONE row (flat values as written, nested values null-filled) --
    through both readers, plain or gzip-compressed, after a blank first line, equal to the same object on one
    line or as a one-element array. A JSONL file whose first row is malformed keeps its own message
    (``:1: not valid JSON``). Several pretty-printed objects, and an object opened after a one-line row, are the
    object stream of `test_jsonl_reader_reads_a_whitespace_separated_object_stream`, read as one row each."""
    import gzip

    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS

    for label, row in (("flat", {"messages": "q1", "reward_spec": "a1"}), ("conversation", CONVERSATION_ROWS[0])):
        expected = [{"prompt": row["messages"], "reference": row["reward_spec"]}]
        pretty = json.dumps(row, indent=2).encode("utf-8")
        shapes = {
            "pretty.json": pretty + b"\n",
            "blank_then_pretty.json": b"\n" + pretty,
            "one_line.json": json.dumps(row).encode("utf-8") + b"\n",
            "wrapped.json": json.dumps([row], indent=2).encode("utf-8") + b"\n",
            "pretty.json.gz": gzip.compress(pretty),
        }
        for name, payload in shapes.items():
            path = tmp_path / f"{label}_{name}"
            path.write_bytes(payload)
            rows = JsonlRows(str(path))
            assert len(rows) == 1 and rows[0] == row, path.name
            assert jsonl_prompts(str(path), "messages", "reward_spec") == expected, path.name
    broken_first = tmp_path / "broken_first.jsonl"
    broken_first.write_text('{"messages": "ok",\n{"messages": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken_first\.jsonl:1: not valid JSON"):
        jsonl_prompts(str(broken_first), "messages")
    with pytest.raises(ValueError, match=r"broken_first\.jsonl:1: not valid JSON"):
        JsonlRows(str(broken_first))


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reader_reads_a_whitespace_separated_object_stream(tmp_path: Path) -> None:
    """The general shape of the local-rows contract is a whitespace-separated stream of JSON objects -- the shape
    pyarrow's JSON reader (behind the Hugging Face ``load_dataset("json")`` PPO used before) read: several objects
    each pretty-printed over several lines (what ``json.dump(obj, f, indent=2)`` in a loop writes), a one-line row
    followed by a pretty-printed object, two objects on one line -- every object as written, positioned by the
    line it opens on, through both readers, plain or gzip-compressed. Refusals keep their line: a value that is not
    an object, text that is not JSON, an object with a trailing comma, a byte outside UTF-8 on an inner line. The
    array rule is unchanged (a ``[`` line after a row is a non-object row), and only gzip is decompressed: a bz2-
    or xz-compressed file is refused as non-UTF-8 text, whatever its suffix says."""
    import bz2
    import gzip
    import lzma

    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows, _jsonl_objects
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS

    def positions(path: Path) -> list[int]:
        return [position for position, _obj in _jsonl_objects(str(path))]

    def read_both(path: Path, rows: list[dict[str, Any]]) -> None:
        got = JsonlRows(str(path))
        assert [got[i] for i in range(len(got))] == rows, path.name
        expected = [{"prompt": r["messages"], "reference": r["reward_spec"]} for r in rows]
        assert jsonl_prompts(str(path), "messages", "reward_spec") == expected, path.name

    # (1) the stream `json.dump(row, f, indent=2)` in a loop writes, over the fixture's five rows: five rows as
    # written, each positioned by the line it opens on; the same gzip-compressed.
    stream = b"".join(json.dumps(row, indent=2).encode("utf-8") + b"\n" for row in CONVERSATION_ROWS)
    pretty_stream = tmp_path / "pretty_stream.json"
    pretty_stream.write_bytes(stream)
    read_both(pretty_stream, list(CONVERSATION_ROWS))
    opens_on = [1]
    for row in CONVERSATION_ROWS[:-1]:
        opens_on.append(opens_on[-1] + json.dumps(row, indent=2).count("\n") + 1)
    assert positions(pretty_stream) == opens_on
    compressed = tmp_path / "pretty_stream.jsonl.gz"
    compressed.write_bytes(gzip.compress(stream))
    read_both(compressed, list(CONVERSATION_ROWS))
    assert positions(compressed) == opens_on

    # (2) the two shapes the reader refused before this change while the Hugging Face reader read them as 2 rows,
    # and two objects on one line, which share the line they open on.
    rows_ab = [{"messages": "a", "reward_spec": "x"}, {"messages": "b", "reward_spec": "y"}]
    two_documents = tmp_path / "two_documents.json"
    two_documents.write_text(
        '{\n  "messages": "a", "reward_spec": "x"\n}\n{\n  "messages": "b", "reward_spec": "y"\n}\n'
    )
    read_both(two_documents, rows_ab)
    assert positions(two_documents) == [1, 4]
    opened_later = tmp_path / "opened_later.jsonl"
    opened_later.write_text('{"messages": "a", "reward_spec": "x"}\n{\n  "messages": "b", "reward_spec": "y"\n}\n')
    read_both(opened_later, rows_ab)
    assert positions(opened_later) == [1, 2]
    same_line = tmp_path / "same_line.jsonl"
    same_line.write_text(
        '{"messages": "a", "reward_spec": "x"} {"messages": "b",\n "reward_spec": "y"}\n{"messages": "c", "reward_spec": "z"}\n'
    )
    read_both(same_line, rows_ab + [{"messages": "c", "reward_spec": "z"}])
    assert positions(same_line) == [1, 1, 3]

    # (3) refusals name the line the bad value starts on (a byte outside UTF-8: the line that holds it)
    not_object = tmp_path / "not_object.json"
    not_object.write_text('{\n  "messages": "a"\n}\n[1, 2]\n', encoding="utf-8")
    not_json = tmp_path / "not_json.json"
    not_json.write_text('{\n  "messages": "a"\n}\nnot json\n', encoding="utf-8")
    trailing_comma = tmp_path / "trailing_comma.jsonl"
    trailing_comma.write_text(
        '{"messages": "a", "reward_spec": "x",}\n{"messages": "b", "reward_spec": "y"}\n', encoding="utf-8"
    )
    inner_bytes = tmp_path / "inner_bytes.json"
    inner_bytes.write_bytes(b'{\n  "messages": "a",\n  "reward_spec": "\xff"\n}\n')
    array_tail = tmp_path / "array_tail.jsonl"
    array_tail.write_text('{"messages": "a"}\n[{"messages": "b"}]\n', encoding="utf-8")
    for path, pattern in (
        (not_object, r"not_object\.json:4: expected a JSON object, got list"),
        (not_json, r"not_json\.json:4: not valid JSON"),
        (trailing_comma, r"trailing_comma\.jsonl:1: not valid JSON"),
        (inner_bytes, r"inner_bytes\.json:3: not UTF-8 text \('utf-8' codec can't decode byte 0xff"),
        (array_tail, r"array_tail\.jsonl:2: expected a JSON object, got list"),
    ):
        with pytest.raises(ValueError, match=pattern):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=pattern):
            JsonlRows(str(path))

    # (4) gzip only: bz2 and xz carry no gzip magic bytes, are not decompressed and are refused as non-UTF-8 text
    jsonl = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows_ab)
    for name, payload in (("rows.jsonl.bz2", bz2.compress(jsonl)), ("rows.jsonl.xz", lzma.compress(jsonl))):
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(ValueError, match=r"rows\.jsonl\.(bz2|xz):1: not UTF-8 text"):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=r"rows\.jsonl\.(bz2|xz):1: not UTF-8 text"):
            JsonlRows(str(path))


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reader_refuses_adjacent_objects_without_whitespace_between_them(tmp_path: Path) -> None:
    """Consecutive objects of the stream are separated by at least one whitespace character -- a newline, or a
    space when two objects share a line (two rows, as `test_jsonl_reader_reads_a_whitespace_separated_object_stream`
    pins). An object glued to the one before it (``}{``) is outside the contract: the one-object-per-line path
    refuses such a line as not valid JSON, and the stream path refuses it too, through both readers, naming the
    line and the 1-based column where the second object starts -- on one line, across pretty-printed lines, and
    after a one-line row that opened the stream -- instead of serving both objects as rows. The refusal comes after
    the object ahead of the boundary was served, so a row the loader refuses (a missing column) is named first, in
    file order; other text glued to an object (``}x``, ``}[1]``) is refused as what it is, not as an adjacent
    object."""
    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows, _jsonl_objects

    glued = tmp_path / "glued.jsonl"
    glued.write_text('{"messages":"a"}{"messages":"b"}\n', encoding="utf-8")
    pretty_glued = tmp_path / "pretty_glued.json"
    pretty_glued.write_text('{\n  "messages": "a"\n}{\n  "messages": "b"\n}\n', encoding="utf-8")
    after_row = tmp_path / "after_row.jsonl"
    after_row.write_text(
        '{"messages": "a"}\n{\n  "messages": "b"\n} {"messages": "c"}{"messages": "d"}\n', encoding="utf-8"
    )
    for path, pattern in (
        (glued, r"glued\.jsonl:1: objects must be separated by whitespace \(adjacent objects at column 17\)"),
        (
            pretty_glued,
            r"pretty_glued\.json:3: objects must be separated by whitespace \(adjacent objects at column 2\)",
        ),
        (after_row, r"after_row\.jsonl:4: objects must be separated by whitespace \(adjacent objects at column 20\)"),
    ):
        with pytest.raises(ValueError, match=pattern):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=pattern):
            JsonlRows(str(path))

    # Separated by one space, the same two objects are two rows on one line: the contract's whitespace separation.
    spaced = tmp_path / "spaced.jsonl"
    spaced.write_text('{"messages":"a"} {"messages":"b"}\n', encoding="utf-8")
    assert jsonl_prompts(str(spaced), "messages") == [{"prompt": "a"}, {"prompt": "b"}]
    rows = JsonlRows(str(spaced))
    assert [rows[i] for i in range(len(rows))] == [{"messages": "a"}, {"messages": "b"}]

    # The boundary is refused AFTER the object ahead of it was served: a row the loader refuses is named first
    # (file order); `JsonlRows`, which has no column contract, reaches the boundary.
    no_column_then_glued = tmp_path / "no_column_then_glued.json"
    no_column_then_glued.write_text('{\n  "prompt": "a"\n}{"messages": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"no_column_then_glued\.json:1: no 'messages' field \(has \['prompt'\]\)"):
        jsonl_prompts(str(no_column_then_glued), "messages")
    boundary = r"no_column_then_glued\.json:3: objects must be separated by whitespace \(adjacent objects at column 2\)"
    with pytest.raises(ValueError, match=boundary):
        JsonlRows(str(no_column_then_glued))
    rows = _jsonl_objects(str(no_column_then_glued))
    assert next(rows) == (1, {"prompt": "a"})  # the row ahead of the boundary is served before the boundary is refused
    with pytest.raises(ValueError, match=boundary):
        next(rows)

    # Other text glued to an object is not an "adjacent object": it is refused as what it is, as before.
    glued_text = tmp_path / "glued_text.json"
    glued_text.write_text('{\n  "messages": "a"\n}x\n', encoding="utf-8")
    glued_list = tmp_path / "glued_list.json"
    glued_list.write_text('{\n  "messages": "a"\n}[1]\n', encoding="utf-8")
    for path, pattern in (
        (glued_text, r"glued_text\.json:3: not valid JSON"),
        (glued_list, r"glued_list\.json:3: expected a JSON object, got list"),
    ):
        with pytest.raises(ValueError, match=pattern):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=pattern):
            JsonlRows(str(path))


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reader_reports_stream_errors_in_file_order_before_a_later_non_utf8_byte(tmp_path: Path) -> None:
    """Refusals come in FILE ORDER through both readers when the object stream holds a byte outside UTF-8 further
    down: the stream is decoded as written up to that byte and the objects before it are served one at a time, so
    an invalid object at line 3 (``:3: not valid JSON``) or a row the loader refuses at line 3 (``:3: no 'messages'
    field``) is named before the bad byte at line 7 -- not the bad byte instead of it. When everything before the
    byte is clean, its own refusal names its line: in a later object, between two objects, or after a one-line row
    that preceded the stream."""
    from cosmos_framework.posttrain.roles.env.dataset import JsonlRows, _jsonl_objects

    head = b'{"messages": "a",\n "reward_spec": "x"}\n'  # lines 1-2: the object that opens the stream
    tail = b'{"messages": "c"}\n{"messages": "d"}\n{"messages": "e"}\n{"messages": "\xff"}\n'  # lines 4-7
    utf8_at_7 = r":7: not UTF-8 text \('utf-8' codec can't decode byte 0xff"

    invalid_third = tmp_path / "invalid_third.json"
    invalid_third.write_bytes(head + b'{"messages": "b",}\n' + tail)
    for read in (lambda: jsonl_prompts(str(invalid_third), "messages"), lambda: JsonlRows(str(invalid_third))):
        with pytest.raises(ValueError, match=r"invalid_third\.json:3: not valid JSON"):
            read()
    rows = _jsonl_objects(str(invalid_third))
    assert next(rows) == (1, {"messages": "a", "reward_spec": "x"})  # the row before the bad one is served first
    with pytest.raises(ValueError, match=r"invalid_third\.json:3: not valid JSON"):
        next(rows)

    no_column_third = tmp_path / "no_column_third.json"
    no_column_third.write_bytes(head + b'{"prompt": "b"}\n' + tail)
    with pytest.raises(ValueError, match=r"no_column_third\.json:3: no 'messages' field \(has \['prompt'\]\)"):
        jsonl_prompts(str(no_column_third), "messages")
    # `JsonlRows` has no column contract: every object before the byte is clean, so it names the byte's line.
    with pytest.raises(ValueError, match=r"no_column_third\.json" + utf8_at_7):
        JsonlRows(str(no_column_third))

    clean_then_bad = tmp_path / "clean_then_bad.json"
    clean_then_bad.write_bytes(head + b'{"messages": "b"}\n' + tail)
    for read in (lambda: jsonl_prompts(str(clean_then_bad), "messages"), lambda: JsonlRows(str(clean_then_bad))):
        with pytest.raises(ValueError, match=r"clean_then_bad\.json" + utf8_at_7):
            read()
    rows = _jsonl_objects(str(clean_then_bad))
    assert [next(rows) for _ in range(5)] == [
        (1, {"messages": "a", "reward_spec": "x"}),
        (3, {"messages": "b"}),
        (4, {"messages": "c"}),
        (5, {"messages": "d"}),
        (6, {"messages": "e"}),
    ]
    with pytest.raises(ValueError, match=r"clean_then_bad\.json" + utf8_at_7):
        next(rows)

    # A one-line row, then the stream, then an invalid object (line 5), then the bad byte (line 6): the earlier
    # error first; with the invalid object fixed, the byte's line.
    row_then_stream = tmp_path / "row_then_stream.jsonl"
    row_then_stream.write_bytes(
        b'{"messages": "a"}\n{\n  "messages": "b"\n}\n{"messages": "c",}\n{"messages": "\xff"}\n'
    )
    row_then_bad = tmp_path / "row_then_bad.jsonl"
    row_then_bad.write_bytes(b'{"messages": "a"}\n{\n  "messages": "b"\n}\n{"messages": "c"}\n{"messages": "\xff"}\n')
    between = tmp_path / "between.json"
    between.write_bytes(b'{\n  "messages": "a"\n}\n\xff\n{"messages": "b"}\n')
    for path, pattern in (
        (row_then_stream, r"row_then_stream\.jsonl:5: not valid JSON"),
        (row_then_bad, r"row_then_bad\.jsonl:6: not UTF-8 text \('utf-8' codec can't decode byte 0xff"),
        (between, r"between\.json:4: not UTF-8 text \('utf-8' codec can't decode byte 0xff"),
    ):
        with pytest.raises(ValueError, match=pattern):
            jsonl_prompts(str(path), "messages")
        with pytest.raises(ValueError, match=pattern):
            JsonlRows(str(path))


@pytest.mark.L0
@pytest.mark.CPU
def test_project_prompt_items_refuses_null_filled_content_parts() -> None:
    """A content part carrying the null-fill SIGNATURE -- ``None`` under a payload key, or a ``None`` type --
    is what a schema-inferring reader (Hugging Face ``datasets`` over Arrow) makes of a
    conversation column: every part gains every key of every other part, ``None`` where it did not carry one,
    and the Qwen chat template (``'image' in content``) renders such a text part as an image placeholder with
    the text gone. The projection refuses it whatever source handed it over (the ``rl.data.name`` branch still
    reads through ``datasets``), at CONSTRUCTION when row 0 carries it (row 0 is projected when the data role
    is built, and null-fill is a property of the whole column), at the fetch otherwise. A ``None`` under a
    part's OWN optional key -- a video part written with ``"fps": null`` -- is the DataPacker's contract
    (`maybe_parse_video_content` reads ``fps`` with ``.get``) and is served as written, as are the same parts
    without the null keys. A null payload key is refused even when it matches ``type`` because it can shadow a
    valid alternate handle in the packer (``image: null`` before ``image_url``)."""
    from cosmos_framework.posttrain.roles.env.dataset import is_null_filled_part

    null_filled = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<image-0.png>", "text": None, "video": None, "fps": None},
                {"type": "text", "text": "Describe this image.", "image": None, "video": None, "fps": None},
            ],
        }
    ]
    signature = r"content parts must not carry null values under a payload key"
    with pytest.raises(ValueError, match=signature):
        _projected([{"messages": null_filled, "reward_spec": "x"}])  # row 0: refused at construction, no fetch
    null_type = [{"role": "user", "content": [{"type": None, "text": "Describe this image."}]}]
    with pytest.raises(ValueError, match=signature):
        _projected([{"messages": null_type, "reward_spec": "x"}])
    assert [is_null_filled_part(part) for part in null_filled[0]["content"]] == [True, True]
    assert is_null_filled_part(null_type[0]["content"][0]) and not is_null_filled_part("not a part")
    shadowed_image = [
        {
            "role": "user",
            "content": [{"type": "image", "image": None, "image_url": "https://example.test/image.png"}],
        }
    ]
    with pytest.raises(ValueError, match=signature):
        _projected([{"messages": shadowed_image, "reward_spec": "x"}])
    assert is_null_filled_part(shadowed_image[0]["content"][0])

    as_written = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "<image-0.png>"}, {"type": "text", "text": "Describe this image."}],
        }
    ]
    assert _projected([{"messages": as_written, "reward_spec": "x"}])[0] == {"prompt": as_written, "reference": "x"}
    # A defect confined to a later row is refused when that row is fetched.
    later = _projected([{"messages": as_written, "reward_spec": "x"}, {"messages": null_filled, "reward_spec": "y"}])
    assert later[0] == {"prompt": as_written, "reference": "x"}
    with pytest.raises(ValueError, match=signature):
        later[1]

    fps_null = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video and return only the JSON result."},
                {"type": "video", "video": [f"<frame-{index:02d}.jpg>" for index in range(4)], "fps": None},
            ],
        }
    ]
    assert not is_null_filled_part(fps_null[0]["content"][1])
    assert _projected([{"messages": fps_null, "reward_spec": "x"}])[0] == {"prompt": fps_null, "reference": "x"}


@pytest.mark.L0
@pytest.mark.CPU
def test_build_prompt_env_refuses_a_cursor_offset_on_the_arithmetic_source() -> None:
    """The arithmetic source synthesizes prompts from its seed and owns no cursor: an offset is refused, not ignored."""
    from cosmos_framework.posttrain.roles.env.prompt import build_prompt_env

    data = SimpleNamespace(name="", path="", factory="", shuffle=False, cursor_offset=3, cursor_shards=0)
    rl = SimpleNamespace(seed=0, data_seed_follows_rollout_seed=False, data=data, validation=None, max_operand=20)
    with pytest.raises(ValueError, match="arithmetic source"):
        build_prompt_env(rl, None)
