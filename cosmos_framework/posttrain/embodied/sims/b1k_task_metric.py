# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Record BEHAVIOR-1K goal progress at the official and extended episode budgets."""

from __future__ import annotations

import os

from omnigibson.metrics import TaskMetric
from omnigibson.metrics.task_metric import compute_q_score

_STRIDE = int(os.environ.get("B1K_PREDICATE_STRIDE", "25"))

_OFFICIAL_MULT = 1.5
_RELAX_MULT = 3.0


class BudgetedTaskMetric(TaskMetric):
    """``TaskMetric`` that also reports the score at the official (1.5x) budget."""

    def reset(self, env=None):
        super().reset(env)
        self._samples: list[tuple[int, float]] = []
        self._warned = False

    def _partial(self, env) -> float:
        """The best option's fraction of goal predicates that are NEWLY true (the partial term alone)."""
        return compute_q_score(False, env.get_goal_option_satisfaction(), self.initial_predicate_states)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        out = super()._compute_step_metrics(env, action, obs, reward, terminated, truncated, info)

        if self.timesteps % _STRIDE == 0 or terminated or truncated:
            score = 1.0 if env.success else self._partial(env)
            self._samples.append((self.timesteps, score))
        return out

    def _score_at(self, budget_steps: float) -> float | None:
        """Last sample at or before ``budget_steps`` -- the score had the episode stopped there."""
        inside = [s for step, s in self._samples if step <= budget_steps]
        return inside[-1] if inside else None

    def _compute_episode_metrics(self, env, episode_info):
        out = super()._compute_episode_metrics(env, episode_info)
        end_score = out["q_score"]["final"]
        human = (self.human_stats or {}).get("steps")
        if not human:
            out["q_score"] = {"end": end_score, "final": end_score, "budget_steps": 0.0}
            self._warn_unbudgeted("this task has no human-demo length")
            return out
        official = self._score_at(_OFFICIAL_MULT * human)
        relaxed = self._score_at(_RELAX_MULT * human)
        out["q_score"] = {
            "final": end_score if official is None else official,
            "relax_2x": end_score if relaxed is None else relaxed,
            "end": end_score,
            "budget_steps": float(_OFFICIAL_MULT * human) if official is not None else 0.0,
        }
        if official is None:
            self._warn_unbudgeted(f"no predicate sample inside {_OFFICIAL_MULT}x{human:.0f} steps")
        return out

    def _warn_unbudgeted(self, why: str) -> None:
        if getattr(self, "_warned", False):
            return
        self._warned = True
        print(
            f"[b1k] q_score.final is the STOCK end-of-episode value, not the official "
            f"budgeted score: {why}. Marked with budget_steps=0.",
            flush=True,
        )
