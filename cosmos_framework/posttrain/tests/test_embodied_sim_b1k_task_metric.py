# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test goal progress at each evaluation budget using a lightweight task metric."""

from __future__ import annotations

import importlib
import random
import sys
import types
from typing import Any

import pytest

TASK_METRIC_MODULE = "cosmos_framework.posttrain.embodied.sims.b1k_task_metric"


def _status(states: list[bool]) -> dict:
    sat = [k for k, s in enumerate(states) if s]
    return {"satisfied": sat, "unsatisfied": [k for k in range(len(states)) if k not in sat]}


def compute_q_score(success, now_satisfied_options, initial_satisfied_options) -> float:
    """Compute the largest fraction of newly satisfied predicates across goal options."""
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    return max(
        (sum(now and not before for now, before in zip(option, initial)) / len(option) if option else 0.0)
        for option, initial in zip(now_satisfied_options, initial_satisfied_options)
    )


class FakeTaskMetric:
    """The v3.9.3 ``TaskMetric`` surface the backend and ``BudgetedTaskMetric`` use, bound to ``env_accessor``."""

    def __init__(self, human_stats=None, env_idx: int = 0, env_accessor=None):
        self.env_accessor = env_accessor
        self.human_stats = {"steps": human_stats["length"]} if human_stats else None
        self.timesteps = 0

    def reset(self, env=None):
        self.timesteps = 0
        self.initial_predicate_states = env.get_goal_option_satisfaction()

    def step(self, env, action, obs, reward, terminated, truncated, info):
        assert env is self.env_accessor, "stepped with an env other than its bound lane"
        self._compute_step_metrics(env, action, obs, reward, terminated, truncated, info)

    def aggregate(self, env):
        assert env is self.env_accessor, "aggregated with an env other than its bound lane"
        return self._compute_episode_metrics(env, {})

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        self.timesteps += 1
        return {"timesteps": self.timesteps}

    def _compute_episode_metrics(self, env, episode_info):
        q = compute_q_score(env.success, env.get_goal_option_satisfaction(), self.initial_predicate_states)
        return {"q_score": {"final": q}, "time": {"simulator_steps": self.timesteps}}


class FakeAgentMetric(FakeTaskMetric):
    pass


def _module(**attrs: Any) -> types.ModuleType:
    mod = types.ModuleType("fake")
    mod.__dict__.update(attrs)
    return mod


def install_metrics(monkeypatch) -> types.ModuleType:
    """Fake ``omnigibson.metrics`` and a fresh ``b1k_task_metric`` imported against it; both undone at teardown."""
    metrics = _module(TaskMetric=FakeTaskMetric, AgentMetric=FakeAgentMetric)
    monkeypatch.setitem(sys.modules, "omnigibson.metrics", metrics)
    monkeypatch.setitem(sys.modules, "omnigibson.metrics.task_metric", _module(compute_q_score=compute_q_score))
    monkeypatch.setitem(sys.modules, TASK_METRIC_MODULE, None)  # recorded, so teardown restores the original
    del sys.modules[TASK_METRIC_MODULE]
    return importlib.import_module(TASK_METRIC_MODULE)


pytestmark = [pytest.mark.L0, pytest.mark.CPU]

HUMAN = 1000  # official budget 1500 steps, relaxed 3000


@pytest.fixture
def mod(monkeypatch) -> types.ModuleType:
    return install_metrics(monkeypatch)


class _Env:
    """One lane: goal predicate ``(start, end)`` holds for ``start < t <= end``."""

    def __init__(self, options, success_at=None):
        self.options, self.success_at, self.t = options, success_at, 0

    @property
    def success(self) -> bool:
        return self.success_at is not None and self.t >= self.success_at

    def get_goal_option_satisfaction(self):
        return [[w[0] < self.t and (len(w) == 1 or self.t <= w[1]) for w in option] for option in self.options]


def _run(mod, options, total, *, success_at=None, human=HUMAN):
    env = _Env(options, success_at)
    metric = mod.BudgetedTaskMetric({"length": human} if human else None, env_accessor=env)
    metric.reset(env)
    for env.t in range(1, total + 1):
        metric._compute_step_metrics(env, None, None, 0.0, env.t == total, False, {})
    return metric._samples, metric._compute_episode_metrics(env, {})


@pytest.mark.parametrize(
    "options,total,success_at,final,relax,end",
    [
        ([[(200,), (400,)]], 3000, None, 1.0, 1.0, 1.0),
        ([[(200,), (2000,)]], 3000, None, 0.5, 1.0, 1.0),
        ([[(2000,), (2500,)]], 3000, None, 0.0, 1.0, 1.0),
        ([[(200, 800), (400, 900)]], 3000, None, 0.0, 0.0, 0.0),
        ([[(9999,)]], 1200, 600, 1.0, 1.0, 1.0),
        ([[(10,), (20,)]], 30, None, 1.0, 1.0, 1.0),
    ],
    ids=["early", "split", "late", "undone", "success_in_budget", "shorter_than_a_stride"],
)
def test_final_is_the_score_at_the_official_budget(mod, options, total, success_at, final, relax, end):
    _, out = _run(mod, options, total, success_at=success_at)
    assert out["q_score"] == {"final": final, "relax_2x": relax, "end": end, "budget_steps": 1500.0}
    assert out["time"]["simulator_steps"] == total


@pytest.mark.parametrize(
    "human,why", [(None, "no human-demo length"), (10, "no predicate sample inside")], ids=["no_stats", "no_sample"]
)
def test_a_stock_final_is_marked_unbudgeted_and_announced(mod, capsys, human, why):
    _, out = _run(mod, [[(10,)]], 100, human=human)
    q = out["q_score"]
    assert q["final"] == q["end"] == 1.0 and q["budget_steps"] == 0.0
    printed = capsys.readouterr().out
    assert "STOCK end-of-episode value" in printed and why in printed


@pytest.mark.parametrize("seed", range(5), ids=[f"seed{i}" for i in range(5)])
def test_at_v1_the_q_score_equals_the_pre_393_formula(mod, monkeypatch, seed):
    """Only where the goal masks come from changed: same samples, budgets and keys as the inline formula."""
    rng = random.Random(seed)
    human = rng.choice([10, 37, 200, 1000])
    total = rng.randint(1, int(3.5 * human) + 30)
    options = [
        [(s,) if rng.random() < 0.6 else (s, s + rng.randint(0, total)) for s in rng.sample(range(-5, total + 5), 3)]
        for _ in range(rng.randint(1, 3))
    ]
    succeed = rng.choice([None, rng.randint(1, total)])

    def legacy(metric, env):
        return max(
            sum(int(not initially and now) for now, initially in zip(option, previous)) / len(option)
            for option, previous in zip(env.get_goal_option_satisfaction(), metric.initial_predicate_states)
        )

    run = _run(mod, options, total, success_at=succeed, human=human)
    monkeypatch.setattr(mod.BudgetedTaskMetric, "_partial", legacy)
    assert run == _run(mod, options, total, success_at=succeed, human=human)
