# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test-owned baseline calibration, paired comparison and explicit-trial analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_PROTOCOLS,
    LIBERO_RUNTIME,
    LiberoCase,
    LiberoManifest,
    LiberoTrialManifest,
)
from cosmos_framework.posttrain.model.gr00t.ppo import NATIVE_ODE_REVISION


def completed_case_counts(
    windows: Sequence[tuple[int, Sequence[LiberoCase]]],
    manifest: LiberoManifest,
    *,
    committed_step: int,
) -> dict[LiberoCase, int]:
    """Count completed training episodes on the retained branch, including unvisited source cases.

    Each item contains a window's optimizer step and its completed episode cases. A repeated or
    decreasing step replaces the abandoned suffix; records beyond the committed checkpoint do
    not contribute to the coverage used to accept or extend an experiment.
    """
    if manifest.split != "train":
        raise ValueError("training coverage requires the training manifest")
    retained: dict[int, Sequence[LiberoCase]] = {}
    previous = -1
    for step, cases in windows:
        if step <= previous:
            retained = {kept: rows for kept, rows in retained.items() if kept < step}
        retained[step] = cases
        previous = step
    counts = dict.fromkeys(manifest.cases, 0)
    for step, cases in retained.items():
        if step > committed_step:
            continue
        for case in cases:
            if case not in counts:
                raise ValueError(f"training episode {case.id} is outside its source manifest")
            counts[case] += 1
    return counts


def evaluation_identity(config, manifest: LiberoTrialManifest) -> dict:
    """Test-tool identity for explicit trials, independent of training validation state."""
    from cosmos_framework.utils.lazy_config import instantiate

    recipe = instantiate(config.environment.config)
    return {
        "suite": recipe.suite,
        "trials": manifest.digest,
        "environment": LIBERO_RUNTIME.identity(),
        "protocol": asdict(LIBERO_PROTOCOLS[recipe.suite]),
        "sampling_mode": config.rollout.evaluation_mode,
        "ode_revision": NATIVE_ODE_REVISION if config.rollout.evaluation_mode == "native_ode" else None,
        "batch_size": 1,
        "backbone_buffer_dtype": config.model.backbone_buffer_dtype,
    }


def episode_metrics(episodes: list[dict], manifest: LiberoManifest) -> dict[str, int | float]:
    from cosmos_framework.posttrain.control.validation.rollout import episode_metrics as summarize
    from cosmos_framework.posttrain.embodied.sims.libero.adapter import validated_episodes

    return summarize(validated_episodes(episodes, manifest))


def paired_comparison(baseline: list[dict], candidate: list[dict]) -> dict:
    """Task-stratified paired bootstrap over the same initial states (descriptive 95% interval)."""
    import numpy as np

    base = {(row["task_id"], row["init_state_id"]): int(row["success"]) for row in baseline}
    current = {(row["task_id"], row["init_state_id"]): int(row["success"]) for row in candidate}
    if set(base) != set(current) or len(base) != len(baseline) or len(current) != len(candidate):
        raise ValueError("paired comparison requires identical, unique cases")
    rng = np.random.default_rng(1234)
    samples = np.zeros(10000)
    for task in range(10):
        delta = np.array([current[key] - base[key] for key in sorted(base) if key[0] == task], dtype=float)
        samples += delta[rng.integers(len(delta), size=(len(samples), len(delta)))].sum(axis=1) / len(base)
    return {
        "success_rate_delta": (sum(current.values()) - sum(base.values())) / len(base),
        "paired_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "episodes": len(base),
        "method": "task-stratified paired bootstrap, 10000 replicates, seed 1234",
    }


def load_baseline_reference(
    recipe: LiberoConfig, reference: str, *, sampling_mode: str = "ode", model_source: dict | None = None
) -> list[dict]:
    """Read an immutable step-zero evaluation report before accepting a training baseline."""
    from cosmos_framework.posttrain.model.gr00t.ppo import NATIVE_ODE_REVISION

    manifest = LiberoManifest.read(recipe.eval_manifest)
    value = json.loads(Path(reference).read_text())
    expected = {
        "optimizer_step": 0,
        "publication_version": 0,
        "manifest": manifest.digest,
        "sampling_mode": sampling_mode,
    }
    if recipe.suite == "libero_10":
        expected.update(
            suite=recipe.suite,
            ode_revision=NATIVE_ODE_REVISION,
            environment=LIBERO_RUNTIME.identity(),
        )
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"baseline reference {key} differs from the configured evaluation")
    if model_source is not None and value.get("model_source") != model_source:
        raise ValueError("baseline reference model_source differs from the original HF checkpoint")
    episode_metrics(value["episodes"], manifest)
    for index, row in enumerate(value["episodes"]):
        if recipe.suite == "libero_spatial":
            # Historical Spatial evaluation persisted episode aggregates only.
            continue
        rewards = row["control_rewards"]
        if (
            len(rewards) != row["steps"]
            or not all(math.isfinite(reward) for reward in rewards)
            or not math.isclose(sum(rewards), row["return"], abs_tol=1e-8)
        ):
            raise ValueError(f"baseline reference episodes[{index}] has inconsistent raw feedback")
    return value["episodes"]


def compare_baseline_episodes(
    reference: list[dict], current: list[dict], *, include_control_rewards: bool = True
) -> dict:
    """Compare case outcomes; legacy Spatial reports omit per-control reward arrays."""

    def keyed(rows: list[dict]) -> dict:
        result = {(row["task_id"], row["init_state_id"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("baseline comparison contains duplicate cases")
        return result

    before, after = keyed(reference), keyed(current)
    if set(before) != set(after):
        raise ValueError("baseline comparison case inventory differs")
    mismatches = []
    compared_fields = ("success", "steps", "return") + (("control_rewards",) if include_control_rewards else ())
    for case in sorted(before):
        fields = [key for key in compared_fields if before[case][key] != after[case][key]]
        if fields:
            mismatches.append({"task_id": case[0], "init_state_id": case[1], "fields": fields})
    return {"exact": not mismatches, "episodes": len(before), "mismatches": mismatches}
