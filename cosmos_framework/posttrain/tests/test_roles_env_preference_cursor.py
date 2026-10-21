# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for exact map-style preference-source cursor round trips."""

from __future__ import annotations

import random
from typing import Any, Iterator

import pytest

from cosmos_framework.posttrain.config.rl import PreferenceEnvConfig
from cosmos_framework.posttrain.roles.env.preference import (
    CursorStateUnavailable,
    DatasetPreferenceEnv,
)


class _StatefulDataset:
    def __init__(self, size: int, seed: int) -> None:
        self._items = [f"pair-{index}" for index in range(size)]
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        prompt = f"{self._items[index]}:drop={self._rng.random() < 0.5}"
        return prompt, {"sample": self._items[index]}, {"sample": self._items[index]}

    def cursor_fingerprint(self) -> str:
        return "|".join(self._items)

    def cursor_state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class _IterableDataset:
    def __iter__(self) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
        yield "pair", {"sample": "winner"}, {"sample": "loser"}


class _UnidentifiedDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[str, dict[str, str], dict[str, str]]:
        return str(index), {"sample": "winner"}, {"sample": "loser"}


def _observations(env: DatasetPreferenceEnv, count: int) -> list[tuple[str, str]]:
    trajectories = env.get_samples(count)
    return [
        (trajectory.id, str(trajectory.transitions[0].obs.representations["prompt"])) for trajectory in trajectories
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_round_trip_preserves_shuffle_position_ids_and_dataset_rng() -> None:
    env = DatasetPreferenceEnv(_StatefulDataset(9, seed=13), rank=1, world_size=2, shuffle=True, seed=7)
    _observations(env, 3)
    state = env.state_dict()
    assert state is not None
    assert state["version"] == 1
    assert set(state) == {
        "version",
        "rank",
        "world_size",
        "shuffle",
        "seed",
        "dataset_len",
        "dataset_fingerprint",
        "epoch",
        "position",
        "dataset_state",
    }
    expected = _observations(env, 8)  # crosses multiple shuffled epoch boundaries

    restored = DatasetPreferenceEnv(_StatefulDataset(9, seed=13), rank=1, world_size=2, shuffle=True, seed=7)
    assert restored.load_state_dict(state)
    assert _observations(restored, 8) == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_rejects_sharding_or_source_mismatch() -> None:
    source = DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5)
    _observations(source, 1)
    state = source.state_dict()
    assert state is not None
    with pytest.raises(ValueError, match="world_size"):
        DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=1, seed=5).load_state_dict(state)
    different_source = _StatefulDataset(8, seed=3)
    different_source._items[0] = "different-pair"
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        DatasetPreferenceEnv(different_source, rank=0, world_size=2, seed=5).load_state_dict(state)
    unsupported_version = dict(state)
    unsupported_version["version"] = 2
    with pytest.raises(ValueError, match="unsupported DatasetPreferenceEnv cursor version 2"):
        DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5).load_state_dict(
            unsupported_version
        )
    missing_dataset_state = dict(state)
    missing_dataset_state["dataset_state"] = None
    with pytest.raises(ValueError, match="missing required dataset-local state"):
        DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5).load_state_dict(
            missing_dataset_state
        )
    invalid_position = dict(state)
    invalid_position["position"] = 1_000
    with pytest.raises(ValueError, match="position exceeds"):
        DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5).load_state_dict(
            invalid_position
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 1.0, "version must be an integer"),
        ("rank", False, "rank must be an integer"),
        ("world_size", "2", "world_size must be an integer"),
        ("shuffle", 1, "shuffle must be a boolean"),
        ("seed", 5.0, "seed must be an integer"),
        ("dataset_len", 8.0, "dataset_len must be an integer"),
        ("epoch", 1.9, "epoch must be an integer"),
        ("position", "1", "position must be an integer"),
    ],
)
def test_cursor_rejects_coerced_schema_values(field: str, value: object, message: str) -> None:
    source = DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5)
    _observations(source, 1)
    state = source.state_dict()
    assert state is not None
    state[field] = value
    with pytest.raises(ValueError, match=message):
        DatasetPreferenceEnv(_StatefulDataset(8, seed=3), rank=0, world_size=2, seed=5).load_state_dict(state)


@pytest.mark.L0
@pytest.mark.CPU
def test_iterable_source_cursor_policy() -> None:
    assert DatasetPreferenceEnv(_IterableDataset(), cursor_resume="auto").state_dict() is None
    assert DatasetPreferenceEnv(_IterableDataset(), cursor_resume="off").state_dict() is None
    with pytest.raises(CursorStateUnavailable, match="map-style"):
        DatasetPreferenceEnv(_IterableDataset(), cursor_resume="strict").state_dict()


@pytest.mark.L0
@pytest.mark.CPU
def test_map_source_without_stable_identity_follows_cursor_policy() -> None:
    assert DatasetPreferenceEnv(_UnidentifiedDataset(), cursor_resume="auto").state_dict() is None
    with pytest.raises(CursorStateUnavailable, match="cursor_fingerprint"):
        DatasetPreferenceEnv(_UnidentifiedDataset(), cursor_resume="strict").state_dict()


@pytest.mark.L0
@pytest.mark.CPU
def test_cursor_off_ignores_saved_state() -> None:
    source = DatasetPreferenceEnv(_StatefulDataset(4, seed=3), cursor_resume="strict")
    _observations(source, 2)
    state = source.state_dict()
    assert state is not None

    restored = DatasetPreferenceEnv(_StatefulDataset(4, seed=3), cursor_resume="off")
    assert restored.load_state_dict(state) is False
    assert restored.state_dict() is None
    assert _observations(restored, 1)[0][0] == "pref-0"


@pytest.mark.L0
@pytest.mark.CPU
def test_get_samples_remains_rank_local_noncollective(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.roles.env import preference

    def _unexpected_collective(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("get_samples must not perform a distributed collective")

    monkeypatch.setattr(preference, "raise_if_any_rank_failed", _unexpected_collective)
    env = DatasetPreferenceEnv(_StatefulDataset(4, seed=3), cursor_resume="off")
    assert len(env.get_samples(2)) == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_invalid_cursor_resume_mode_fails_at_construction() -> None:
    assert PreferenceEnvConfig().cursor_resume == "auto"
    assert PreferenceEnvConfig(cursor_resume="strict").cursor_resume == "strict"
    with pytest.raises(ValueError, match="cursor_resume"):
        DatasetPreferenceEnv(_StatefulDataset(4, seed=3), cursor_resume="sometimes")
    with pytest.raises(ValueError, match="cursor_resume"):
        PreferenceEnvConfig(cursor_resume="sometimes")


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_batches_mask_pad_without_mutating_cursor_or_dataset_rng() -> None:
    environments = [
        DatasetPreferenceEnv(
            _StatefulDataset(5, seed=13),
            rank=rank,
            world_size=4,
            shuffle=True,
            seed=7,
            cursor_resume="strict",
        )
        for rank in range(4)
    ]
    before = [environment.state_dict() for environment in environments]
    batches = [list(environment.validation_batches(batch_size=2)) for environment in environments]
    after = [environment.state_dict() for environment in environments]

    assert before == after
    assert [len(per_rank) for per_rank in batches] == [1, 1, 1, 1]
    assert [per_rank[0].logical_count for per_rank in batches] == [2, 1, 1, 1]
    assert all(len(per_rank[0].trajectories) == 2 for per_rank in batches)
    weights = [
        [trajectory.transitions[0].algo_extra["validation_weight"] for trajectory in per_rank[0].trajectories]
        for per_rank in batches
    ]
    assert weights == [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    real_ids = {
        trajectory.id
        for per_rank in batches
        for trajectory in per_rank[0].trajectories
        if trajectory.transitions[0].algo_extra["validation_weight"] == 1.0
    }
    assert real_ids == {f"pref-{index}" for index in range(5)}


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_batches_honor_max_samples_and_equal_schedule() -> None:
    environments = [
        DatasetPreferenceEnv(_StatefulDataset(9, seed=3), rank=rank, world_size=3, shuffle=False) for rank in range(3)
    ]

    assert [environment.validation_batch_count(2, max_samples=4) for environment in environments] == [1, 1, 1]
    batches = [list(environment.validation_batches(2, max_samples=4)) for environment in environments]
    assert sum(per_rank[0].logical_count for per_rank in batches) == 4
    assert all(len(per_rank[0].trajectories) == 2 for per_rank in batches)
