# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired three-seed quality uncertainty; never a standalone release decision.

Inputs must first pass the experiment, checkpoint and scoring identity contracts.
This module checks statistical pairing and never selects a candidate or removes
outliers. Its fixed-seed hierarchical analysis and crossed-group sensitivity were
declared before continuation training outcomes. Both must meet unchanged margins.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from tools.native_qlora_qualification.quality import INPUT_KEYS, metrics, paired_metrics, validate_scores

TRAINING_SEEDS = (17, 42, 123)
BOOTSTRAP_SEED = 2026
REPETITIONS = 10000
CRITERIA = {
    "relative_native_token_nll_upper_max": 0.02,
    "accuracy_loss_upper_max": 0.01,
    "gain_over_frozen_nf4_lower_min_exclusive": 0.0,
}


def _validate(trained, frozen, groups):
    if set(trained) != set(TRAINING_SEEDS) or any(type(seed) is not int for seed in trained):
        raise ValueError("confirmation requires exactly paired training seeds 17, 42 and 123")
    if set(frozen) != {"nf4", "bf16"}:
        raise ValueError("both deterministic frozen precision arms are required")
    for seed in TRAINING_SEEDS:
        if set(trained[seed]) != {"nf4", "bf16"}:
            raise ValueError("every training seed requires both precision arms")
    anchor = trained[17]["nf4"]
    for rows in [*frozen.values(), *(trained[seed][arm] for seed in TRAINING_SEEDS for arm in ("nf4", "bf16"))]:
        validate_scores(rows)
        if len(rows) != len(anchor) or any(
            any(a[key] != b[key] for key in INPUT_KEYS) for a, b in zip(anchor, rows, strict=True)
        ):
            raise ValueError("confirmation inputs, masks, media, row order or token counts differ")
    if any(
        row["id"] not in groups or not isinstance(groups[row["id"]], str) or not groups[row["id"]] for row in anchor
    ):
        raise ValueError("every scored row requires a connected recording-group label")


def _totals(trained, frozen, groups):
    by_group = defaultdict(list)
    for index, row in enumerate(trained[17]["nf4"]):
        by_group[groups[row["id"]]].append(index)
    result = {}
    for seed in TRAINING_SEEDS:
        result[seed] = []
        for group in sorted(by_group):
            indices = by_group[group]
            nf4 = [trained[seed]["nf4"][i] for i in indices]
            bf16 = [trained[seed]["bf16"][i] for i in indices]
            result[seed].append(
                (
                    len(indices),
                    sum(row["native_supervised_tokens"] for row in nf4),
                    math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in nf4),
                    math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in bf16),
                    sum(row["restricted_choice_correct"] for row in nf4),
                    sum(row["restricted_choice_correct"] for row in bf16),
                    sum(frozen["nf4"][i]["restricted_choice_correct"] for i in indices),
                )
            )
    return result


def _statistics(totals):
    rows, tokens, nf4_ce, bf16_ce, nf4_correct, bf16_correct, frozen_correct = (
        math.fsum(values) for values in zip(*totals, strict=True)
    )
    if rows <= 0 or tokens <= 0 or bf16_ce <= 0:
        raise ValueError("relative uncertainty requires positive reference NLL and nonempty observations")
    return {
        "relative_native_token_nll": nf4_ce / bf16_ce - 1,
        "accuracy_loss": (bf16_correct - nf4_correct) / rows,
        "gain_over_frozen_nf4": (nf4_correct - frozen_correct) / rows,
    }


def _interval(values):
    ordered = sorted(values)

    def quantile(p):
        return ordered[math.ceil(len(ordered) * p) - 1]

    return {
        "interval_95": [quantile(0.025), quantile(0.975)],
        "lower_one_sided_95": quantile(0.05),
        "upper_one_sided_95": quantile(0.95),
    }


def _resample(totals, *, crossed, repetitions):
    group_count = len(totals[17])
    if group_count < 2:
        return {"estimable": False, "reason": "fewer than two connected recording groups", "groups": group_count}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {key: [] for key in ("relative_native_token_nll", "accuracy_loss", "gain_over_frozen_nf4")}
    for _ in range(repetitions):
        seeds = rng.choices(TRAINING_SEEDS, k=len(TRAINING_SEEDS))
        shared_groups = rng.choices(range(group_count), k=group_count) if crossed else None
        sample = []
        for seed in seeds:
            selected = shared_groups if crossed else rng.choices(range(group_count), k=group_count)
            sample.extend(totals[seed][index] for index in selected)
        for key, value in _statistics(sample).items():
            samples[key].append(value)
    bounds = {key: _interval(values) for key, values in samples.items()}
    passed = {
        "native_token_nll": bounds["relative_native_token_nll"]["upper_one_sided_95"]
        <= CRITERIA["relative_native_token_nll_upper_max"],
        "accuracy_preservation": bounds["accuracy_loss"]["upper_one_sided_95"] <= CRITERIA["accuracy_loss_upper_max"],
        "useful_adaptation": bounds["gain_over_frozen_nf4"]["lower_one_sided_95"]
        > CRITERIA["gain_over_frozen_nf4_lower_min_exclusive"],
    }
    return {
        "estimable": True,
        "groups": group_count,
        "paired_training_seeds": list(TRAINING_SEEDS),
        "repetitions": repetitions,
        "rng_seed": BOOTSTRAP_SEED,
        "method": "paired seeds then shared recording-group draw"
        if crossed
        else "paired seeds then independent within-seed recording-group draws",
        "bounds": bounds,
        "criteria": CRITERIA.copy(),
        "criteria_passed": passed,
        "all_criteria_passed": all(passed.values()),
    }


def analyze_confirmation(trained, frozen, groups, *, split_role, repetitions=REPETITIONS):
    """Summarize validated scores; small repetition counts are tests, never a pass."""
    if split_role not in {"development", "regression", "untouched_diagnostic", "final"}:
        raise ValueError("explicit sealed split role required")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive integer bootstrap repetitions required")
    _validate(trained, frozen, groups)
    totals = _totals(trained, frozen, groups)
    pooled = {arm: [row for seed in TRAINING_SEEDS for row in trained[seed][arm]] for arm in ("nf4", "bf16")}
    primary = _resample(totals, crossed=False, repetitions=repetitions)
    sensitivity = _resample(totals, crossed=True, repetitions=repetitions)
    enough = repetitions == REPETITIONS and all(result.get("estimable") for result in (primary, sensitivity))
    slices = {}
    for name, key in (("choice_count", lambda row: str(len(row["choices"]))), ("gold_letter", lambda row: row["gold"])):
        slices[name] = {}
        for value in sorted({key(row) for row in pooled["nf4"]}):
            a = [row for row in pooled["nf4"] if key(row) == value]
            b = [row for row in pooled["bf16"] if key(row) == value]
            slices[name][value] = paired_metrics(a, b)
            subset = {
                seed: {arm: [row for row in rows if key(row) == value] for arm, rows in trained[seed].items()}
                for seed in TRAINING_SEEDS
            }
            frozen_subset = {arm: [row for row in rows if key(row) == value] for arm, rows in frozen.items()}
            slice_totals = _totals(subset, frozen_subset, groups)
            for label, crossed in (("hierarchical_primary", False), ("crossed_group_sensitivity", True)):
                uncertainty = _resample(slice_totals, crossed=crossed, repetitions=repetitions)
                slices[name][value][label] = {
                    field: item
                    for field, item in uncertainty.items()
                    if field not in {"criteria", "criteria_passed", "all_criteria_passed"}
                }
    influences = []
    for index, anchor in enumerate(trained[17]["nf4"]):
        deltas = [
            trained[seed]["nf4"][index]["native_sft_ce"] - trained[seed]["bf16"][index]["native_sft_ce"]
            for seed in TRAINING_SEEDS
        ]
        influences.append(
            {
                "row_identity": anchor["row_identity"],
                "group_sha256": groups[anchor["id"]],
                "mean_nf4_minus_bf16_ce": math.fsum(deltas) / len(TRAINING_SEEDS),
                "by_seed": {
                    str(seed): {
                        "nf4_minus_bf16_ce": delta,
                        "nf4_correct": trained[seed]["nf4"][index]["restricted_choice_correct"],
                        "bf16_correct": trained[seed]["bf16"][index]["restricted_choice_correct"],
                    }
                    for seed, delta in zip(TRAINING_SEEDS, deltas, strict=True)
                },
                "frozen_nf4_minus_bf16_ce": frozen["nf4"][index]["native_sft_ce"]
                - frozen["bf16"][index]["native_sft_ce"],
            }
        )
    influences.sort(key=lambda row: (-row["mean_nf4_minus_bf16_ce"], row["row_identity"]))
    return {
        "protocol": "cosmos.qlora-quality-confirmation.v1",
        "split_role": split_role,
        "rows_per_seed": len(trained[17]["nf4"]),
        "paired_training_seeds": list(TRAINING_SEEDS),
        "groups": len(totals[17]),
        "pooled_trained": {arm: metrics(rows) for arm, rows in pooled.items()},
        "fixed_frozen": {arm: metrics(rows) for arm, rows in frozen.items()},
        "initial_quantization_effect": paired_metrics(frozen["nf4"], frozen["bf16"]),
        "point_estimates": _statistics([item for seed in TRAINING_SEEDS for item in totals[seed]]),
        "by_training_seed": {
            str(seed): {
                "trained_precision_comparison": paired_metrics(trained[seed]["nf4"], trained[seed]["bf16"]),
                "adaptation": {arm: paired_metrics(trained[seed][arm], frozen[arm]) for arm in ("nf4", "bf16")},
            }
            for seed in TRAINING_SEEDS
        },
        "slices_exploratory": slices,
        "per_row_influence_exploratory": influences,
        "hierarchical_primary": primary,
        "crossed_group_sensitivity": sensitivity,
        "conditional_statistical_criteria_passed": enough
        and all(result["all_criteria_passed"] for result in (primary, sensitivity)),
        "artifact_validation_required_separately": True,
        "release_qualified": False,
        "limitations": [
            "Three training seeds and few recording groups give limited uncertainty resolution.",
            "Whole-group pairing cannot establish independence beyond available recording metadata.",
            "Frozen predictions are a fixed deterministic reference, not additional independent training seeds.",
            "Development and exposed regression outcomes cannot establish untouched final quality.",
            "Artifact identity, recovery, memory and task-scope gates are separate requirements.",
            "Slice intervals are exploratory, unadjusted for multiple comparisons and never used for selection.",
        ],
    }
