# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small native output probe for fresh-process adapter/dense artifact qualification.

This is not a trainer or a replacement task evaluator. The caller constructs a
policy through the normal native factory/loader and supplies one real driving row.
The conversion thresholds were declared before observing full-model cast results.
"""

from __future__ import annotations

import math

CONVERSION_GATES = {
    "flow_velocity_rmse": 0.01,
    "flow_velocity_max_abs": 0.05,
    "flow_loss_relative": 0.01,
    "flow_loss_absolute_floor": 1e-5,
    "trajectory_ade_6p4s_meters": 0.10,
    "trajectory_fde_6p4s_meters": 0.25,
    "task_metric_relative": 0.01,
    "task_metric_absolute_floor_meters": 0.02,
}


def capture_outputs(policy, batch, *, seed=17):
    """One controlled flow point and native Euler trajectory; preserve all training RNG/modes."""
    import torch

    from cosmos_framework.posttrain.entrypoints.action_sft import preserved_evaluation
    from cosmos_framework.posttrain.model.alpamayo.policy import sample_trajectories
    from cosmos_framework.posttrain.model.alpamayo.training import trajectory_metrics
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest

    rows = [step for trajectory in batch for step in trajectory.transitions]
    if len(rows) != 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Alpamayo artifact probe requires one physical trajectory and an unsigned 32-bit seed")
    with preserved_evaluation(policy, seed):
        inputs = policy.packer(batch)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        controlled = {
            **inputs,
            "time": torch.full((1,), 0.5, dtype=torch.float32),
            "noise": torch.randn(inputs["actions"].shape, generator=generator, dtype=torch.float32),
        }
        flow = policy.flow_matching_loss(controlled)
        loss = float(flow.loss)
        prediction = flow.prediction.detach().float().cpu()
    with preserved_evaluation(policy, (seed + 1) % 2**32):
        trajectory = sample_trajectories(policy, inputs).detach().float().cpu()
    expected = torch.tensor(rows[0].obs.representations["flow_sample"]["actions"], dtype=torch.float32)[None]
    metrics = trajectory_metrics(trajectory, expected)
    if not math.isfinite(loss) or not torch.isfinite(prediction).all():
        raise ValueError("Alpamayo artifact probe produced nonfinite native flow outputs")

    def fingerprints(values, prefix=""):
        result = {}
        for name, value in sorted(values.items()):
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                result.update(fingerprints(value, key))
            elif isinstance(value, torch.Tensor):
                result[key] = tensor_digest(value)
            else:
                raise TypeError(f"unsupported Alpamayo probe input: {key}")
        return result

    return {
        "format": "cosmos-alpamayo-artifact-probe-v1",
        "sample_id": rows[0].obs.representations["flow_sample"]["id"],
        "seed": seed,
        "input_tensors": fingerprints(controlled),
        "flow_loss": loss,
        "flow_prediction": prediction.tolist(),
        "trajectory_xyz_meters": trajectory.tolist(),
        "task_metrics": {name: value["sum"] / value["count"] for name, value in metrics.items()},
    }


def compare_outputs(reference, candidate):
    """Return a transparent all-gates decision; never relax a failed conversion budget."""
    import numpy as np

    for name in ("format", "sample_id", "seed", "input_tensors"):
        if reference.get(name) != candidate.get(name):
            raise ValueError(f"Alpamayo artifact comparison has different {name}")
    if reference.get("format") != "cosmos-alpamayo-artifact-probe-v1":
        raise ValueError("unsupported Alpamayo artifact probe")
    values = {}
    for key, shape in (("flow_prediction", (1, 64, 2)), ("trajectory_xyz_meters", (1, 64, 3))):
        pair = [np.asarray(record[key], dtype=np.float64) for record in (reference, candidate)]
        if any(value.shape != shape or not np.isfinite(value).all() for value in pair):
            raise ValueError(f"nonfinite or malformed Alpamayo artifact output: {key}")
        values[key] = pair
    flow_delta = values["flow_prediction"][1] - values["flow_prediction"][0]
    trajectory_delta = values["trajectory_xyz_meters"][1] - values["trajectory_xyz_meters"][0]
    distance = np.linalg.norm(trajectory_delta[..., :2], axis=-1)
    measured = {
        "flow_velocity_rmse": float(np.sqrt(np.square(flow_delta).mean())),
        "flow_velocity_max_abs": float(np.abs(flow_delta).max()),
        "trajectory_ade_6p4s_meters": float(distance.mean()),
        "trajectory_fde_6p4s_meters": float(distance[:, -1].mean()),
    }
    passed = {name: value <= CONVERSION_GATES[name] for name, value in measured.items()}
    losses = [float(row["flow_loss"]) for row in (reference, candidate)]
    if not all(math.isfinite(value) and value >= 0 for value in losses):
        raise ValueError("nonfinite or negative native flow loss in artifact probe")
    measured["flow_loss_absolute_delta"] = abs(losses[1] - losses[0])
    loss_bound = max(
        CONVERSION_GATES["flow_loss_absolute_floor"], CONVERSION_GATES["flow_loss_relative"] * abs(losses[0])
    )
    passed["flow_loss"] = measured["flow_loss_absolute_delta"] <= loss_bound
    metrics = reference["task_metrics"]
    required_metrics = {
        f"open_loop_{metric}_{horizon}_meters" for metric in ("ade", "fde") for horizon in ("1s", "3s", "6p4s")
    }
    if set(metrics) != required_metrics or set(candidate["task_metrics"]) != required_metrics:
        raise ValueError("Alpamayo task metric inventory differs in artifact probe")
    metric_deltas = {}
    for name, baseline in metrics.items():
        value = candidate["task_metrics"][name]
        if not all(math.isfinite(item) and item >= 0 for item in (baseline, value)):
            raise ValueError("nonfinite or negative open-loop metric in artifact probe")
        bound = max(
            CONVERSION_GATES["task_metric_absolute_floor_meters"], CONVERSION_GATES["task_metric_relative"] * baseline
        )
        metric_deltas[name] = {"absolute_delta": abs(value - baseline), "bound": bound}
        passed[name] = abs(value - baseline) <= bound
    return {
        "passed": all(passed.values()),
        "gates": dict(CONVERSION_GATES),
        "measured": measured,
        "flow_loss_bound": loss_bound,
        "task_metric_deltas": metric_deltas,
        "individual_pass": passed,
        "claim": "artifact_numerical_conversion_only_not_recipe_quality_or_closed_loop",
    }
