# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Local test assembly of the production decision-window Role boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmos_framework.posttrain.control.loop.decision_window import collect_decision_window, evaluate_decision_cases
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.utils.entry import resolve_all
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


class LocalWindowControl:
    def __init__(
        self,
        rollout: Any,
        gateway: Any,
        recipe: Any,
        replica: int,
        n_lanes: int,
        *,
        environment: Any,
        evaluation_gateway_factory: Callable | None = None,
    ) -> None:
        self.body = PPOWindowCollector(recipe, replica, n_lanes, environment=environment)
        self.collector = WorkerGroup("collector", "env", body=self.body)
        self.simulator = WorkerGroup("sim", "env", body=gateway)
        self.rollout = WorkerGroup("rollout", "rollout", body=rollout)
        self.gateway = gateway
        self._evaluation_factory = evaluation_gateway_factory
        self._evaluation_simulator: WorkerGroup | None = None
        resolve_all(self.collector.bind_rollout(self.rollout))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.body, name)

    def collect_window(self, version: int) -> list:
        return collect_decision_window(
            self.collector, self.simulator, version=version, decisions=self.body.recipe.window_decisions
        ).result()

    def evaluate_cases(self, version: int, cases: list[dict], seed: int) -> dict:
        if self._evaluation_simulator is None:
            gateway = self._evaluation_factory() if self._evaluation_factory else self.gateway
            self._evaluation_simulator = WorkerGroup("eval_sim", "env", body=gateway)
        return evaluate_decision_cases(
            self.collector,
            self._evaluation_simulator,
            version=version,
            cases=cases,
            seed=seed,
            decisions=self.body.environment.evaluation_decisions,
        ).result()

    def evaluate(self, version: int, replicas: int) -> dict:
        return self.evaluate_cases(
            version,
            self.body.environment.evaluation_cases(self.body.replica, replicas),
            self.body.recipe.validation.seed,
        )

    def close(self) -> None:
        try:
            if self._evaluation_simulator is not None and self._evaluation_simulator.body is not self.gateway:
                resolve_all(self._evaluation_simulator.close())
        finally:
            try:
                resolve_all(self.simulator.close())
            finally:
                resolve_all(self.collector.close())
