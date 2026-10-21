# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO seeds, simulator configuration and fixed-case validation for shared collectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    DEFAULT_EXECUTION_PROFILE,
    LiberoCase,
    LiberoManifest,
    execution_protocol,
    libero_seed,
)
from cosmos_framework.posttrain.embodied.window import EvaluationEpisode, WindowSimulatorSpec
from cosmos_framework.posttrain.schema.metrics import WeightedMean


class LiberoEnvironment:
    action_dim: int = 7

    def __init__(self, config: LiberoConfig) -> None:
        self.config = config
        self.execution_horizon = config.execution_horizon
        self.evaluation_decisions = (
            config.max_episode_steps + config.execution_horizon - 1
        ) // config.execution_horizon

    def identity(self) -> dict:
        return self.config.identity()

    def validate_configuration(self) -> None:
        if not self.config.sim_python:
            raise ValueError("environment.config.sim_python is required")
        self.config.validate_sources()

    def gateway_spec(self, *, replica: int, n_lanes: int, seed: int) -> WindowSimulatorSpec:
        return WindowSimulatorSpec(
            name="libero",
            embodiment="libero_sim",
            python=self.config.sim_python,
            options={
                **self.config.simulator_options,
                "seed": seed if self.config.suite == "libero_10" else 0,
                "n_lanes": n_lanes,
                "lane_offset": replica * n_lanes,
                "train_manifest": self.config.train_manifest,
            },
        )

    def training_seed(
        self, *, seed: int, version: int, replica: int, n_lanes: int, num_envs: int, decision: int
    ) -> int:
        if execution_protocol(self.config.suite, self.config.execution_profile).seed_derivation != "legacy":
            return libero_seed(self.config.suite, "training-policy", seed, replica, version, decision)
        return seed + (version * num_envs + replica * n_lanes) * 16 + decision

    def bootstrap_seed(self, *, seed: int, version: int, replica: int) -> int:
        return (
            libero_seed(self.config.suite, "bootstrap-value", seed, replica, version)
            if execution_protocol(self.config.suite, self.config.execution_profile).seed_derivation != "legacy"
            else seed
        )

    def evaluation_seed(self, context: Mapping[str, Any]) -> int:
        case = context["case"]
        if execution_protocol(self.config.suite, self.config.execution_profile).seed_derivation != "legacy":
            return libero_seed(
                self.config.suite,
                "evaluation-policy",
                context["seed"],
                case["task_id"],
                case["init_state_id"],
                context["decision"],
            )
        return context["seed"] + (case["task_id"] * 50 + case["init_state_id"]) * 16 + context["decision"]

    def evaluation_cases(self, replica: int, replicas: int) -> list[dict[str, int | str]]:
        manifest = LiberoManifest.read(self.config.eval_manifest)
        cases = manifest.cases
        if execution_protocol(self.config.suite, self.config.execution_profile).fixed_episode:
            cases = tuple(sorted(cases, key=lambda case: (case.init_state_id, case.task_id)))
        return [asdict(case) for case in cases[replica::replicas]]

    def evaluation_fingerprint(self) -> str:
        return LiberoManifest.read(self.config.eval_manifest).digest

    def validate_evaluation(self, episodes: Sequence[Mapping[str, Any]]) -> tuple[EvaluationEpisode, ...]:
        return validated_episodes(
            episodes, LiberoManifest.read(self.config.eval_manifest), execution_profile=self.config.execution_profile
        )


def validated_episodes(
    episodes: Sequence[Mapping[str, Any]],
    manifest: LiberoManifest,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
) -> tuple[EvaluationEpisode, ...]:
    """Check raw outcomes against the complete source inventory before generic aggregation."""
    cases = [LiberoCase(row["task_id"], row["init_state_id"], manifest.suite) for row in episodes]
    if len(set(cases)) != len(cases) or set(cases) != set(manifest.cases):
        raise ValueError("LIBERO evaluation inventory differs from its complete source manifest")
    protocol = execution_protocol(manifest.suite, execution_profile)
    results = []
    for case, row in zip(cases, episodes, strict=True):
        minimum = 1 if protocol.stop_on_success else protocol.execution_horizon
        if (
            type(row["success"]) is not bool
            or type(row["steps"]) is not int
            or not minimum <= row["steps"] <= protocol.max_episode_steps
        ):
            raise ValueError(f"invalid evaluation outcome for {case.id}")
        if not protocol.stop_on_success and row["steps"] % protocol.execution_horizon:
            raise ValueError(f"evaluation {case.id} did not complete its action chunk")
        if protocol.fixed_episode and row["steps"] != protocol.max_episode_steps:
            raise ValueError(f"evaluation {case.id} did not complete its fixed episode")
        WeightedMean.from_value(row["return"])
        results.append(
            EvaluationEpisode(case.id, f"task_{case.task_id}", row["success"], row["return"], row["steps"], dict(row))
        )
    return tuple(sorted(results, key=lambda row: row.case_id))
