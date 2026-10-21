# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: DAPO-style dynamic sampling in the GRPO collection loop (rl.min_informative_groups).

The fake reward role runs the REAL group-relative estimator and reward report, so the informative
stamp the replay buffer gates on comes from its owner. The replay role is one real `ExperienceBuffer`
per replay cell behind a mesh handle, so the per-cell stop rule and quota split are exercised as the
loop sees them.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
from cosmos_framework.posttrain.control import ExperienceBuffer
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config as _config
from cosmos_framework.posttrain.tests.helpers.entry_loop import mock_loop_env as _mock_env
from cosmos_framework.posttrain.utils.batch_stats import IdleRewardReport, RewardBatchStats, reward_batch_stats
from cosmos_framework.posttrain.utils.entry import resolve_all
from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle, Handle, Scatter

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

GROUP_SIZE = 2
Group = tuple[float, ...]  # the rewards of one GRPO group
Draw = tuple[Group, ...]  # one rollout draw: one group per data cell, in cell order


def _group(group_id: str, rewards: Group) -> list[Trajectory]:
    return [
        Trajectory(
            transitions=[Transition(obs=ObsBundle(id=group_id), reward={FINAL_REWARD_KEY: float(r)}, done=True)],
            id=f"{group_id}_k{k}",
            group_id=group_id,
            status=Status.COMPLETED,
        )
        for k, r in enumerate(rewards)
    ]


def _value(payload: Any) -> Any:
    if isinstance(payload, CollectiveHandle):
        return [trajectory for cell in resolve_all(payload) for trajectory in cell]
    return payload.result() if isinstance(payload, Handle) else payload


class _Policy:
    def __init__(self) -> None:
        self._it = 0
        self.batches: list[list[Trajectory]] = []

    def iteration(self) -> int:
        return self._it

    def needs_old_logprobs(self) -> bool:
        return False

    def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
        del mini, micro, n_mini, n_micro
        self.batches.append(list(_value(batch)))
        self._it += 1
        return {"train/loss_avg": 0.5}

    def save(self, path) -> None:
        pass


class _Replay:
    """One real buffer per replay cell behind a `(cells, 1)` mesh handle, as the loop sees the replay WG.

    A draw lands cell-contiguously: data cell ``r``'s share of every payload goes to replay cell ``r``.
    Payload-bearing results (batch, leftovers) are recorded in ``handles`` when the test tracks them.
    """

    def __init__(self, cells: int, handles: list[Handle] | None) -> None:
        self._cells = [ExperienceBuffer() for _ in range(cells)]
        self._handles = handles
        self.drained = 0  # trajectories that left through the buffer's lease-releasing egress
        self.prioritized_calls = 0

    def _mesh(self, values: list[Any], *, track: bool = False) -> CollectiveHandle:
        handle = CollectiveHandle({(r, 0): value for r, value in enumerate(values)}, (len(values), 1))
        if track and self._handles is not None:
            self._handles.append(handle)
        return handle

    def put_samples(self, scored: Any) -> CollectiveHandle:
        if isinstance(scored, CollectiveHandle):
            return self._mesh(
                [cell.put_samples(items) for cell, items in zip(self._cells, resolve_all(scored), strict=True)]
            )
        items = _value(scored)
        share = len(items) // len(self._cells)
        return self._mesh([cell.put_samples(items[r * share : (r + 1) * share]) for r, cell in enumerate(self._cells)])

    def count(self, gate: Any) -> CollectiveHandle:
        return self._mesh([cell.count(gate) for cell in self._cells])

    def get_samples(self, size: int) -> CollectiveHandle:
        return self._mesh([cell.get_samples(size) for cell in self._cells], track=True)

    def select_prioritized(self, size: int, gate: Any) -> CollectiveHandle:
        self.prioritized_calls += 1
        return self._mesh([cell.select_prioritized(size, gate) for cell in self._cells], track=True)

    def drain(self) -> CollectiveHandle:
        leftovers = [cell.drain() for cell in self._cells]
        self.drained += sum(len(cell) for cell in leftovers)
        return self._mesh(leftovers, track=True)

    def __len__(self) -> int:
        return sum(len(cell) for cell in self._cells)


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    draws: list[Draw],
    *,
    cells: int = 1,
    prompts_per_cell: int = 1,
    handles: list[Handle] | None = None,
    **rl: Any,
) -> tuple[dict[str, Any], SimpleNamespace, dict[str, int], list[dict[str, Any]]]:
    """Fake roles over ``cells`` data/replay cells: each rollout draw yields the next scripted group per
    cell; the reward role stamps them with the real estimator and reports with the real reward report.
    ``prompts_per_cell`` is the per-cell prompt count in one logical round.
    With ``handles`` given, every payload-bearing result crosses a recorded handle boundary.
    Returns (units, config, counters, reports)."""
    _mock_env(monkeypatch)
    reported: list[dict[str, Any]] = []
    reporter = SimpleNamespace(
        start=lambda: None, report=lambda metrics, step: reported.append(metrics), finish=lambda: None
    )
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *_a, **_k: [reporter])
    queue = list(draws)
    pending_reports: list[CollectiveHandle] = []
    counters: dict[str, int] = {"generate": 0, "draws": 0}
    dispatch_widths: list[int] = []
    annotate = group_relative(group_size=GROUP_SIZE)

    def wrap(values: list[Any], *, track: bool = True) -> CollectiveHandle:
        handle = CollectiveHandle({(cell, 0): value for cell, value in enumerate(values)}, (cells, 1))
        if handles is not None and track:
            handles.append(handle)
        return handle

    def get_samples(count: int | Scatter) -> CollectiveHandle:
        counts = count.seq if isinstance(count, Scatter) else [count] * cells
        for cell, size in enumerate(counts):
            counters[f"prompts_cell_{cell}"] = counters.get(f"prompts_cell_{cell}", 0) + size
        return wrap([["prompt"] * size for size in counts], track=False)

    def generate(prompts: CollectiveHandle, group: int) -> CollectiveHandle:
        del group
        counters["generate"] += 1
        prompt_cells = resolve_all(prompts)
        counts = [len(cell_prompts) for cell_prompts in prompt_cells]
        width = max(counts, default=0)
        assert width > 0 and all(count in (0, width) for count in counts), counts
        served: list[tuple[int, Draw]] = []
        for _ in range(width):
            draw = queue.pop(0)
            assert len(draw) == cells, "each draw scripts one group per data cell"
            counters["draws"] += 1
            served.append((counters["draws"], draw))
        dispatch_widths.append(width)
        values: list[list[Trajectory]] = []
        for cell, cell_prompts in enumerate(prompt_cells):
            counters[f"engine_cell_{cell}"] = counters.get(f"engine_cell_{cell}", 0) + bool(cell_prompts)
            values.append(
                [trajectory for index, draw in served for trajectory in _group(f"g{index}c{cell}", draw[cell])]
                if cell_prompts
                else []
            )
        return wrap(values)

    def score_with_report(drawn: CollectiveHandle, *, idle: bool | Scatter = False) -> CollectiveHandle:
        values = resolve_all(drawn)
        idle_flags = idle.seq if isinstance(idle, Scatter) else [idle] * cells
        reports: list[RewardBatchStats | IdleRewardReport] = []
        for trajectories, is_idle in zip(values, idle_flags, strict=True):
            if is_idle:
                assert not trajectories
                reports.append(IdleRewardReport())
            else:
                annotate(trajectories)
                reports.append(reward_batch_stats(trajectories))
        pending_reports.append(wrap(reports, track=False))
        return wrap(values)

    policy = _Policy()
    policy.push_weights = lambda _rollout: None
    policy.shared_weight_identity = lambda: None
    units = {
        "data": SimpleNamespace(get_samples=get_samples),
        "rollout": SimpleNamespace(  # no discard sink: dropped draws must never reach a bulk consumer
            generate=generate,
            dispatch_widths=dispatch_widths,
            prepare_weight_version=lambda _step, _identity: None,
            commit_weight_version=lambda _step: None,
        ),
        "rm": SimpleNamespace(score_with_report=score_with_report, batch_stats=lambda: pending_reports.pop(0)),
        "replay": _Replay(cells, handles),
        "policy": policy,
    }
    rl.setdefault("min_informative_groups", 1)
    config = _config(
        max_steps=1,
        prompts_per_step=prompts_per_cell,
        group_size=GROUP_SIZE,
        batch_per_replica=cells * prompts_per_cell * GROUP_SIZE,
        seed=0,
        save_every=0,
        **rl,
    )
    if cells > 1:
        config.grid.rollout_shape = lambda: (cells, 1)
    return units, config, counters, reported


def test_dynamic_sampling_tops_up_drops_tied_groups_and_honours_the_budget_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top up until the informative quota, train on it, sink every other group; at the budget raise or proceed.

    The three-cell case has a non-divisible global quota (2 groups over 3 cells → 1, 1, 0): it must stop
    once cells 0 and 1 hold one informative group each, even though cell 2 never produces one.
    """
    from cosmos_framework.posttrain.entrypoints import grpo

    cases = [
        # (cells, rl knobs, scripted draws, trained batch rewards, informative groups, tied groups, dropped groups)
        (
            1,
            {"dynamic_sampling_max_extra_rollouts": 4},
            [((1.0, 1.0),), ((0.0, 0.0),), ((0.0, 1.0),)],
            [0.0, 1.0],
            1,
            2,
            2,
        ),
        (
            1,
            {"dynamic_sampling_max_extra_rollouts": 1, "dynamic_sampling_on_budget_exhausted": "proceed"},
            [((1.0, 1.0),), ((0.0, 0.0),)],
            [1.0, 1.0],  # the first tied group is the filler
            0,
            2,
            1,
        ),
        (
            3,
            {"min_informative_groups": 2, "dynamic_sampling_max_extra_rollouts": 1},
            [((1.0, 1.0), (0.0, 1.0), (0.0, 0.0)), ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0))],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # cells 0 and 1 train on their informative group, cell 2 on filler
            2,
            4,
            3,
        ),
    ]
    for cells, rl, draws, trained, informative, tied, dropped_groups in cases:
        units, config, counters, reported = _setup(monkeypatch, draws, cells=cells, **rl)
        grpo.loop(units, config, "/tmp/out")
        assert counters["generate"] == len(draws)
        assert [sorted(t.final_reward for t in batch) for batch in units["policy"].batches] == [trained]
        assert units["replay"].drained == dropped_groups * GROUP_SIZE  # whole groups, via the buffer's egress
        assert len(units["replay"]) == 0  # leftovers never reach the next on-policy step
        frame = reported[-1]
        assert frame["train/grpo_dynamic_sampling_extra_rollouts"] == len(draws) - 1
        assert frame["train/grpo_dynamic_sampling_dropped_trajectories"] == dropped_groups * GROUP_SIZE
        assert (frame["train/grpo_informative_group_count"], frame["train/grpo_tied_group_count"]) == (
            informative,
            tied,
        )

    units, config, counters, _ = _setup(
        monkeypatch, [((1.0, 1.0),), ((0.0, 0.0),)], dynamic_sampling_max_extra_rollouts=1
    )
    with pytest.raises(RuntimeError, match="dynamic_sampling_on_budget_exhausted=proceed"):
        grpo.loop(units, config, "/tmp/out")
    assert counters["generate"] == 2 and units["policy"].batches == []

    # Default off: one draw per step, plain get_samples, and the prioritized selection, the drain and the
    # dynamic-sampling metrics never appear.
    units, config, counters, reported = _setup(monkeypatch, [((1.0, 0.0),)], min_informative_groups=0)
    grpo.loop(units, config, "/tmp/out")
    assert counters["generate"] == 1 and len(units["policy"].batches) == 1
    assert units["replay"].prioritized_calls == 0 and units["replay"].drained == 0
    assert "train/grpo_dynamic_sampling_extra_rollouts" not in reported[-1]


@pytest.mark.parametrize("outcome", ["train", "raise"])
def test_dynamic_sampling_releases_every_draw_handle_on_the_pool_plane(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Every draw handle, the leftovers and the pooled pages are released by the shared per-step cleanup,
    whether the step trains or the budget-exhaustion raise aborts it before a batch exists."""
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    second_draw = ((0.0, 1.0),) if outcome == "train" else ((0.0, 0.0),)  # tied again → budget spent → raise
    units, config, counters, _ = _setup(
        monkeypatch, [((1.0, 1.0),), second_draw], handles=handles, dynamic_sampling_max_extra_rollouts=1
    )
    config.rl.data_pool = SimpleNamespace(data_plane="ray_nixl_pool")
    barriers: list[Any] = []
    monkeypatch.setattr(grpo, "release_pool_step_barrier", lambda payload, owners: barriers.append(payload) or 0)

    if outcome == "train":
        grpo.loop(units, config, "/tmp/out")
        trained, batch_handles = [[0.0, 1.0]], 1
    else:
        with pytest.raises(RuntimeError, match="dynamic_sampling_on_budget_exhausted=proceed"):
            grpo.loop(units, config, "/tmp/out")
        trained, batch_handles = [], 0  # no batch was assembled, so the barrier ran over an empty payload

    assert counters["generate"] == 2 and units["replay"].drained == 4 - 2 * batch_handles and len(units["replay"]) == 0
    assert len(handles) == 5 + batch_handles and all(
        handle._released for handle in handles
    )  # (generate, score) x 2 + drain [+ batch]
    assert [[[t.final_reward for t in cell] for cell in payload] for payload in barriers] == [trained]


def test_dynamic_sampling_reads_the_resolved_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop enforces `rl.effective_min_informative_groups` -- the config's resolution of the literal count /
    fraction pair -- not the literal field: with the literal at 0 and the fraction set, the resolved quota of one
    group tops up a tied first draw and the step trains on the informative group; the shortfall names both knobs."""
    from cosmos_framework.posttrain.entrypoints import grpo

    resolved = {"min_informative_groups": 0, "min_informative_groups_frac": 1.0, "effective_min_informative_groups": 1}
    units, config, counters, reported = _setup(monkeypatch, [((1.0, 1.0),), ((0.0, 1.0),)], **resolved)
    grpo.loop(units, config, "/tmp/out")
    assert counters["generate"] == 2
    assert [sorted(t.final_reward for t in batch) for batch in units["policy"].batches] == [[0.0, 1.0]]
    assert reported[-1]["train/grpo_dynamic_sampling_extra_rollouts"] == 1

    units, config, counters, _ = _setup(
        monkeypatch, [((1.0, 1.0),), ((0.0, 0.0),)], dynamic_sampling_max_extra_rollouts=1, **resolved
    )
    with pytest.raises(RuntimeError, match=r"quota of 1 informative groups .*min_informative_groups_frac=1.0"):
        grpo.loop(units, config, "/tmp/out")
    assert counters["generate"] == 2 and units["policy"].batches == []


def _trained(units: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Group ids and rewards of the one batch the fake policy trained on, in loop order."""
    (batch,) = units["policy"].batches
    return [trajectory.group_id for trajectory in batch], [trajectory.final_reward for trajectory in batch]


def _run_both_dispatches(monkeypatch: pytest.MonkeyPatch, draws: list[Draw], **rl: Any) -> dict[str, dict[str, Any]]:
    """Run the same scripted draws under both refill-dispatch values."""
    from cosmos_framework.posttrain.entrypoints import grpo

    out: dict[str, dict[str, Any]] = {}
    for dispatch in ("per_round", "whole_budget"):
        units, config, counters, reported = _setup(
            monkeypatch, list(draws), dynamic_sampling_refill_dispatch=dispatch, **rl
        )
        grpo.loop(units, config, "/tmp/out")
        assert len(units["replay"]) == 0, dispatch
        assert counters["generate"] == len(units["rollout"].dispatch_widths), dispatch
        out[dispatch] = {
            "trained": _trained(units),
            "widths": list(units["rollout"].dispatch_widths),
            "draws": counters["draws"],
            "drained": units["replay"].drained,
            "frame": reported[-1],
        }
    return out


def test_whole_budget_refill_is_one_wide_dispatch_and_trains_per_round_s_groups_at_full_dapo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At full DAPO both values train the same groups while whole_budget uses one wide refill call."""
    draws: list[Draw] = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 1.0), (0.0, 0.0)),
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
    ]
    runs = _run_both_dispatches(
        monkeypatch, draws, cells=2, min_informative_groups=2, dynamic_sampling_max_extra_rollouts=4
    )
    for dispatch, run in runs.items():
        assert run["trained"] == (["g3c0", "g3c0", "g1c1", "g1c1"], [0.0, 1.0, 1.0, 0.0]), dispatch
    assert {name: run["widths"] for name, run in runs.items()} == {
        "per_round": [1, 1, 1],
        "whole_budget": [1, 4],
    }
    assert {name: run["draws"] for name, run in runs.items()} == {"per_round": 3, "whole_budget": 5}
    for key, per_round, whole_budget in (
        ("train/grpo_dynamic_sampling_refill_dispatches", 2, 1),
        ("train/grpo_dynamic_sampling_extra_rollouts", 2, 4),
        ("train/grpo_group_count", 6, 10),
        ("train/grpo_dynamic_sampling_dropped_trajectories", 8, 16),
    ):
        assert (runs["per_round"]["frame"][key], runs["whole_budget"]["frame"][key]) == (
            per_round,
            whole_budget,
        )


def test_whole_budget_refill_below_full_dapo_replaces_tied_filler_and_advances_the_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below full DAPO a wide refill can replace filler and advances the cursor by the whole budget."""
    from cosmos_framework.posttrain.entrypoints import grpo

    lines: list[str] = []
    monkeypatch.setattr(grpo, "info0", lambda message, *args, **kwargs: lines.append(str(message)))
    tied, informative_a, informative_b = (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)
    layout: dict[str, Any] = {
        "cells": 1,
        "prompts_per_cell": 2,
        "min_informative_groups": 1,
        "dynamic_sampling_max_extra_rollouts": 4,
        "dynamic_sampling_on_budget_exhausted": "proceed",
    }
    early = _run_both_dispatches(
        monkeypatch,
        [
            (tied,),
            (tied,),
            (informative_a,),
            (tied,),
            (informative_b,),
            (tied,),
            (informative_a,),
            (tied,),
            (tied,),
            (tied,),
        ],
        **layout,
    )
    assert early["per_round"]["trained"] == (["g3c0", "g3c0", "g1c0", "g1c0"], [0.0, 1.0, 0.0, 0.0])
    assert early["whole_budget"]["trained"] == (
        ["g3c0", "g3c0", "g5c0", "g5c0"],
        [0.0, 1.0, 1.0, 0.0],
    )
    assert {name: run["widths"] for name, run in early.items()} == {
        "per_round": [2, 2],
        "whole_budget": [2, 8],
    }
    assert {name: run["draws"] for name, run in early.items()} == {"per_round": 4, "whole_budget": 10}
    for key, per_round, whole_budget in (
        ("train/grpo_dynamic_sampling_extra_rollouts", 1, 4),
        ("train/grpo_dynamic_sampling_refill_dispatches", 1, 1),
        ("train/grpo_group_count", 4, 10),
        ("train/grpo_informative_group_count", 1, 3),
        ("train/grpo_dynamic_sampling_dropped_trajectories", 4, 16),
    ):
        assert (early["per_round"]["frame"][key], early["whole_budget"]["frame"][key]) == (
            per_round,
            whole_budget,
        )
    for expected in (
        "refill_dispatch=per_round (budget 4 extra rounds, 1 per dispatch; per-engine refill request width "
        "4 sequences = 1 x prompts_per_data_cell 2 x group_size 2, data cells 1:1 to engines)",
        "refill_dispatch=whole_budget (budget 4 extra rounds, 4 per dispatch; per-engine refill request width "
        "16 sequences = 4 x prompts_per_data_cell 2 x group_size 2, data cells 1:1 to engines)",
    ):
        assert any(expected in line for line in lines), (expected, lines)

    exhausted = _run_both_dispatches(
        monkeypatch,
        [
            (tied,),
            (tied,),
            (tied,),
            (tied,),
            (tied,),
            (tied,),
            (tied,),
            (tied,),
            (informative_a,),
            (tied,),
        ],
        **layout,
    )
    for dispatch, run in exhausted.items():
        assert run["trained"] == (["g9c0", "g9c0", "g1c0", "g1c0"], [0.0, 1.0, 0.0, 0.0]), dispatch
        assert run["draws"] == 10
        assert run["frame"]["train/grpo_dynamic_sampling_extra_rollouts"] == 4
    assert {name: run["widths"] for name, run in exhausted.items()} == {
        "per_round": [2, 2, 2, 2, 2],
        "whole_budget": [2, 8],
    }


def test_whole_budget_composes_with_short_cells_and_names_the_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selective whole-budget refill widens only participating cells and keeps idle metrics round-based."""
    from cosmos_framework.posttrain.entrypoints import grpo

    lines: list[str] = []
    monkeypatch.setattr(grpo, "info0", lambda message, *args, **kwargs: lines.append(str(message)))
    units, config, counters, reported = _setup(
        monkeypatch,
        [
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0)),
        ],
        cells=2,
        min_informative_groups=2,
        dynamic_sampling_max_extra_rollouts=2,
        dynamic_sampling_refill_mode="short_cells",
        dynamic_sampling_refill_dispatch="whole_budget",
    )
    grpo.loop(units, config, "/tmp/out")
    assert units["rollout"].dispatch_widths == [1, 2]
    assert (counters["prompts_cell_0"], counters["prompts_cell_1"]) == (3, 1)
    assert reported[-1]["train/grpo_dynamic_sampling_refill_dispatches"] == 1
    assert reported[-1]["train/grpo_dynamic_sampling_extra_rollouts"] == 2
    assert reported[-1]["train/grpo_dynamic_sampling_idle_cells"] == 2
    assert reported[-1]["train/grpo_dynamic_sampling_avoided_prompts"] == 2
    assert any("refill_dispatch=whole_budget" in line for line in lines)


def test_whole_budget_refill_keeps_exactly_the_batch_in_draw_order_and_names_the_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four emitter contracts of `rl.dynamic_sampling_refill_dispatch`, pinned under both quota scopes (`cell` and
    `pooled`): (a) under `proceed` the shortfall line is, whole, the scope's own text, the dispatch clause ``after N
    extra rollouts (refill_dispatch=<d>, <D> refill dispatches; groups generated <G>, trained <T>)`` and the suffix
    ``; proceeding with tied groups as zero-advantage filler`` -- the exact grammar the step-anatomy tooling parses
    from the run log; the `raise` message is the same scope text and clause followed by the refusal text, never by
    the `proceed` suffix; (b) over-collection
    keeps exactly `batch_per_replica` trajectories, the FIRST informative group in draw order, the surplus drained;
    (c) `raise` refuses the step after ONE wide call; (d) the default (`per_round`) issues no refill call when the
    base draw satisfies the quota and reports `refill_dispatches` 0; the start-up line names each dispatch value
    with its per-engine request width."""
    from cosmos_framework.posttrain.entrypoints import grpo

    lines: list[str] = []
    monkeypatch.setattr(grpo, "info0", lambda message, *args, **kwargs: lines.append(str(message)))
    clause = re.compile(
        r"after (\d+) extra rollouts \(refill_dispatch=(\w+), (\d+) refill dispatches; "
        r"groups generated (\d+), trained (\d+)\)"
    )
    tied, informative_a, informative_b = (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)
    for scope in ("cell", "pooled"):
        # (b) one tied base draw, then one wide call holding three informative groups for a one-group share.
        lines.clear()
        units, config, counters, reported = _setup(
            monkeypatch,
            [((1.0, 1.0),), (informative_a,), (informative_b,), (informative_a,)],
            dynamic_sampling_max_extra_rollouts=3,
            dynamic_sampling_refill_dispatch="whole_budget",
            dynamic_sampling_quota_scope=scope,
        )
        grpo.loop(units, config, "/tmp/out")
        ids, rewards = _trained(units)
        assert ids == ["g2c0", "g2c0"] and rewards == [0.0, 1.0], scope  # the FIRST informative group, whole
        assert len(ids) == config.rl.batch_per_replica, scope
        assert units["rollout"].dispatch_widths == [1, 3] and counters["generate"] == 2, scope
        assert units["replay"].drained == 3 * GROUP_SIZE and len(units["replay"]) == 0, scope  # g1 (tied), g3, g4
        frame = reported[-1]
        assert frame["train/grpo_dynamic_sampling_refill_dispatches"] == 1, scope
        assert frame["train/grpo_dynamic_sampling_extra_rollouts"] == 3, scope
        assert frame["train/grpo_dynamic_sampling_dropped_trajectories"] == 3 * GROUP_SIZE, scope
        assert any(
            "refill_dispatch=whole_budget (budget 3 extra rounds, 3 per dispatch; per-engine refill request width "
            "6 sequences = 3 x prompts_per_data_cell 1 x group_size 2, data cells 1:1 to engines)" in line
            for line in lines
        ), (scope, lines)

        # (a) the budget spent on tied draws under `proceed`: the first tied group trains as filler and the ONE
        # shortfall line is, whole, the scope's text, the dispatch clause and the `proceed` suffix, under both
        # dispatch values (the counts are the fakes': 2 extra rollouts, 3 groups drawn, 1 group trained).
        scope_text = (
            "the assembled batch holds 0 informative groups, below the pooled quota, "
            if scope == "pooled"
            else "replay cells [0] hold fewer informative trajectories than their quotas [2] "
        )
        for dispatch, dispatches, widths in (("whole_budget", 1, [1, 2]), ("per_round", 2, [1, 1, 1])):
            lines.clear()
            units, config, counters, reported = _setup(
                monkeypatch,
                [(tied,), ((1.0, 1.0),), (tied,)],
                dynamic_sampling_max_extra_rollouts=2,
                dynamic_sampling_on_budget_exhausted="proceed",
                dynamic_sampling_refill_dispatch=dispatch,
                dynamic_sampling_quota_scope=scope,
            )
            grpo.loop(units, config, "/tmp/out")
            assert _trained(units) == (["g1c0", "g1c0"], [0.0, 0.0]), (scope, dispatch)
            assert units["rollout"].dispatch_widths == widths, (scope, dispatch)
            assert reported[-1]["train/grpo_dynamic_sampling_refill_dispatches"] == dispatches, (scope, dispatch)
            shortfall = [line for line in lines if line.startswith("dynamic sampling batch_step=0: ")]
            assert shortfall == [
                f"dynamic sampling batch_step=0: {scope_text}after 2 extra rollouts (refill_dispatch={dispatch}, "
                f"{dispatches} refill dispatches; groups generated 3, trained 1)"
                "; proceeding with tied groups as zero-advantage filler"
            ], (scope, dispatch, lines)
            assert clause.search(shortfall[0]).groups() == ("2", dispatch, str(dispatches), "3", "1")

        # (c) ... and `raise` refuses the step after the one wide call, exactly as the per_round budget does; the
        # message is the same scope text and clause followed by the refusal text -- no `proceed` suffix.
        units, config, counters, _ = _setup(
            monkeypatch,
            [(tied,), ((1.0, 1.0),), (tied,)],
            dynamic_sampling_max_extra_rollouts=2,
            dynamic_sampling_refill_dispatch="whole_budget",
            dynamic_sampling_quota_scope=scope,
        )
        with pytest.raises(RuntimeError, match="dynamic_sampling_on_budget_exhausted=proceed") as refused:
            grpo.loop(units, config, "/tmp/out")
        assert units["rollout"].dispatch_widths == [1, 2] and units["policy"].batches == [], scope
        message = str(refused.value)
        assert message == (
            f"dynamic sampling batch_step=0: {scope_text}after 2 extra rollouts (refill_dispatch=whole_budget, "
            "1 refill dispatches; groups generated 3, trained 1); the reward regime cannot supply the quota of "
            f"{config.rl.effective_min_informative_groups} informative groups "
            f"(rl.min_informative_groups={config.rl.min_informative_groups}, "
            f"rl.min_informative_groups_frac={config.rl.min_informative_groups_frac}) within "
            f"rl.dynamic_sampling_max_extra_rollouts={config.rl.dynamic_sampling_max_extra_rollouts} "
            "(set rl.dynamic_sampling_on_budget_exhausted=proceed to train on tied groups as zero-advantage filler)"
        ), message
        assert "; proceeding with tied groups as zero-advantage filler" not in message
        assert clause.search(message).groups() == ("2", "whole_budget", "1", "3", "1"), message

        # (d) the default: per_round, one round per call; a step whose base draw satisfies the quota issues no
        # refill call and reports no dispatch.
        lines.clear()
        units, config, counters, reported = _setup(
            monkeypatch, [(informative_a,)], dynamic_sampling_max_extra_rollouts=4, dynamic_sampling_quota_scope=scope
        )
        grpo.loop(units, config, "/tmp/out")
        assert units["rollout"].dispatch_widths == [1] and _trained(units) == (["g1c0", "g1c0"], [0.0, 1.0]), scope
        assert reported[-1]["train/grpo_dynamic_sampling_refill_dispatches"] == 0, scope
        assert reported[-1]["train/grpo_dynamic_sampling_extra_rollouts"] == 0, scope
        assert any(
            "refill_dispatch=per_round (budget 4 extra rounds, 1 per dispatch; per-engine refill request width "
            "2 sequences = 1 x prompts_per_data_cell 1 x group_size 2, data cells 1:1 to engines)" in line
            for line in lines
        ), (scope, lines)


@pytest.mark.parametrize("mode,draws_per_cell,total_groups", [("all", (2, 2, 2), 6), ("short_cells", (2, 1, 1), 4)])
def test_refills_keep_original_cells_and_full_batches(
    monkeypatch: pytest.MonkeyPatch, mode: str, draws_per_cell: tuple[int, ...], total_groups: int
) -> None:
    """Only the short cell advances in selective mode; zero-quota cells retain their base filler."""
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, counters, reported = _setup(
        monkeypatch,
        [((1.0, 1.0), (0.0, 1.0), (0.0, 0.0)), ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0))],
        cells=3,
        handles=handles,
        min_informative_groups=2,
        dynamic_sampling_refill_mode=mode,
    )
    grpo.loop(units, config, "/tmp/out")
    assert tuple(counters[f"prompts_cell_{cell}"] for cell in range(3)) == draws_per_cell
    assert tuple(counters[f"engine_cell_{cell}"] for cell in range(3)) == draws_per_cell
    assert [[trajectory.group_id for trajectory in batch] for batch in units["policy"].batches] == [
        ["g2c0", "g2c0", "g1c1", "g1c1", "g1c2", "g1c2"]
    ]
    assert reported[-1]["train/grpo_group_count"] == total_groups
    assert reported[-1]["train/grpo_dynamic_sampling_avoided_prompts"] == (2 if mode == "short_cells" else 0)
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


@pytest.mark.parametrize("outcome", ["raise", "proceed"])
def test_selective_refill_exhaustion_keeps_budget_and_cleanup(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, counters, _ = _setup(
        monkeypatch,
        [((0.0, 1.0), (1.0, 1.0)), ((0.0, 1.0), (1.0, 1.0))],
        cells=2,
        handles=handles,
        min_informative_groups=2,
        dynamic_sampling_refill_mode="short_cells",
        dynamic_sampling_max_extra_rollouts=1,
        dynamic_sampling_on_budget_exhausted=outcome,
    )
    if outcome == "raise":
        with pytest.raises(RuntimeError, match="dynamic sampling batch_step=0"):
            grpo.loop(units, config, "/tmp/out")
        assert units["policy"].batches == []
    else:
        grpo.loop(units, config, "/tmp/out")
        assert len(units["policy"].batches[0]) == 4
    assert counters["prompts_cell_0"] == 1 and counters["prompts_cell_1"] == 2
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


def test_failed_refill_report_releases_handles_before_the_report_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, _, _ = _setup(
        monkeypatch, [((1.0, 1.0),), ((0.0, 1.0),)], handles=handles, dynamic_sampling_refill_mode="short_cells"
    )
    batch_stats = units["rm"].batch_stats
    calls = 0

    def fail_second_report() -> CollectiveHandle:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("refill report failed")
        return batch_stats()

    units["rm"].batch_stats = fail_second_report
    with pytest.raises(RuntimeError, match="refill report failed"):
        grpo.loop(units, config, "/tmp/out")
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


@pytest.mark.parametrize(
    "scope,generate_calls,batch_groups,batch_informative",
    [
        pytest.param("cell", 2, ["g2c0", "g2c1", "g1c2", "g1c3"], 4, id="cell"),
        pytest.param("pooled", 1, ["g1c0", "g1c1", "g1c2", "g1c3"], 2, id="pooled"),
    ],
)
def test_pooled_quota_stops_once_the_assembled_batch_holds_it(
    monkeypatch: pytest.MonkeyPatch, scope: str, generate_calls: int, batch_groups: list[str], batch_informative: int
) -> None:
    """The base draw lands both informative groups on cells 2 and 3, whose split quota is zero.

    Cell scope still owes cells 0 and 1 one group each and refills; pooled scope stops after the base
    draw because the informative-first batch already holds the global quota of two groups.
    """
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, counters, reported = _setup(
        monkeypatch,
        [((1.0, 1.0), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)), ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0))],
        cells=4,
        handles=handles,
        min_informative_groups=2,
        dynamic_sampling_quota_scope=scope,
    )
    grpo.loop(units, config, "/tmp/out")
    assert counters["generate"] == generate_calls
    trained = units["policy"].batches[0]
    assert [trajectory.group_id for trajectory in trained[::GROUP_SIZE]] == batch_groups
    frame = reported[-1]
    assert frame["train/grpo_dynamic_sampling_extra_rollouts"] == generate_calls - 1
    assert frame["train/grpo_dynamic_sampling_batch_informative_groups"] == batch_informative
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


def test_pooled_refill_idles_cells_whose_share_is_already_informative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selective pooled refills draw only on cells that can still add an informative group to the batch."""
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, counters, reported = _setup(
        monkeypatch,
        [((1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)), ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0))],
        cells=4,
        handles=handles,
        min_informative_groups=2,
        dynamic_sampling_quota_scope="pooled",
        dynamic_sampling_refill_mode="short_cells",
    )
    grpo.loop(units, config, "/tmp/out")
    assert tuple(counters[f"prompts_cell_{cell}"] for cell in range(4)) == (2, 2, 2, 1)
    assert tuple(counters[f"engine_cell_{cell}"] for cell in range(4)) == (2, 2, 2, 1)
    trained = units["policy"].batches[0]
    assert [trajectory.group_id for trajectory in trained[::GROUP_SIZE]] == ["g1c0", "g2c1", "g1c2", "g1c3"]
    frame = reported[-1]
    assert frame["train/grpo_dynamic_sampling_extra_rollouts"] == 1
    assert frame["train/grpo_dynamic_sampling_avoided_prompts"] == 1
    assert frame["train/grpo_dynamic_sampling_batch_informative_groups"] == 2
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


@pytest.mark.parametrize("outcome", ["raise", "proceed"])
def test_pooled_exhaustion_reports_the_assembled_count(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    handles: list[Handle] = []
    units, config, counters, reported = _setup(
        monkeypatch,
        [((0.0, 1.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 1.0))],
        cells=2,
        handles=handles,
        min_informative_groups=2,
        dynamic_sampling_quota_scope="pooled",
        dynamic_sampling_refill_mode="short_cells",
        dynamic_sampling_max_extra_rollouts=1,
        dynamic_sampling_on_budget_exhausted=outcome,
    )
    if outcome == "raise":
        with pytest.raises(RuntimeError, match="holds 1 informative groups, below the pooled quota"):
            grpo.loop(units, config, "/tmp/out")
        assert units["policy"].batches == []
    else:
        grpo.loop(units, config, "/tmp/out")
        assert len(units["policy"].batches[0]) == 4
        assert reported[-1]["train/grpo_dynamic_sampling_batch_informative_groups"] == 1
    assert counters["prompts_cell_0"] == 1 and counters["prompts_cell_1"] == 2
    assert all(handle._released for handle in handles) and len(units["replay"]) == 0


def test_pooled_quota_no_cell_can_serve_raises_instead_of_skipping_refills() -> None:
    """With every cell full, an unmet pooled quota asks for more than the batch holds. Returning no
    short cells would skip both the refill loop and the budget check, so training would silently
    proceed below the quota even under ``dynamic_sampling_on_budget_exhausted=raise``."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import INFORMATIVE_GROUP_GATE
    from cosmos_framework.posttrain.control.buffer import MetadataGate
    from cosmos_framework.posttrain.entrypoints import grpo

    full = SimpleNamespace(count=lambda _gate: CollectiveHandle({(0, 0): 8, (1, 0): 8}, (2, 1)))
    gate = MetadataGate(*INFORMATIVE_GROUP_GATE)

    assert grpo._short_cells(full, gate, (8, 8), scope="pooled", cell_share=8) == ([], 16)
    with pytest.raises(RuntimeError, match="exceeds the 16 trajectories the assembled batch can hold"):
        grpo._short_cells(full, gate, (12, 12), scope="pooled", cell_share=8)
