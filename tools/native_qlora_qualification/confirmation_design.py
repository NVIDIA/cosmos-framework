# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prospective recipe-level NI design; original three-seed reports stay unchanged.

Only metadata/scores are read. Group labels do not prove independence. A source
review must account for recording/session/event relations, authorization and all
previous exposure. Planning is a normal approximation, not measured bootstrap
power. Neither a passing calculation nor this CLI authorizes acquisition/jobs or
qualifies a release. Frozen scores are fixed references, never extra seed draws.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from tools.native_qlora_qualification.common import digest_json, inventory_groups, write_json_new
from tools.native_qlora_qualification.quality import INPUT_KEYS, metrics, paired_metrics, validate_scores
from tools.native_qlora_qualification.quality_confirmation import BOOTSTRAP_SEED, REPETITIONS, _interval, _statistics

PROTOCOL = "cosmos.qlora-confirmation-design.v1"
MARGINS = {"relative_native_token_nll": 0.02, "accuracy_loss": 0.01}
WEIGHTING = {"native_nll": "supervised_tokens", "accuracy": "examples", "tasks": "natural_frequency_no_macro_reweight"}


def _count(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer count")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("required SHA256 is unbound or malformed")
    return value


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite real required")
    return float(value)


def proposal(task_types, recipe_sha256):
    """Concrete candidate, not a sufficient sample size or an operational approval.

    A separate eight-seed/40-group pilot estimates variability. The 20-seed,
    400-group final candidate must be revised/sealed *before* final outcomes if
    pilot planning rejects it. Neither 180 nor 400 groups guarantees power.
    Disjoint seed sets avoid selecting final trained checkpoints on pilot results.
    """
    return {
        "protocol": PROTOCOL,
        "paired_seeds": {"pilot": list(range(1001, 1009)), "confirmation": list(range(2001, 2021))},
        "groups": {"pilot": 40, "confirmation": 400},
        "minimum_groups_per_task": {"pilot": 10, "confirmation": 40},
        "task_types": list(task_types),
        "recipe_sha256": dict(recipe_sha256),
        "weighting": WEIGHTING.copy(),
        "margins": MARGINS.copy(),
        "repetitions": REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inventory_sha256": None,
        "provenance_audit_sha256": None,
        "task_taxonomy_sha256": None,
        "pilot_analysis_sha256": None,
        "planning": {
            "assumed_relative_nll": 0.0,
            "assumed_accuracy_loss": 0.0,
            "variance_multipliers": [1.0, 2.0],
            "target_joint_power": 0.9,
        },
    }


def validate_design(spec, inventory, *, role):
    """Check a sealed metadata design, including transitive leakage across roles.

    Inventory rows contain id, sha256 (video), role, task, session and event.
    Session/event are reviewed provenance labels; unknown/empty labels refuse.
    These checks establish consistency of supplied evidence, not its truth.
    """
    if role not in {"pilot", "confirmation"} or spec.get("protocol") != PROTOCOL:
        raise ValueError("explicit prospective pilot/confirmation protocol required")
    if spec.get("margins") != MARGINS or spec.get("weighting") != WEIGHTING:
        raise ValueError("unchanged NI margins and token/example weighting required")
    if spec.get("repetitions") != REPETITIONS or spec.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise ValueError("prospective resampling policy changed")
    if set(spec["recipe_sha256"]) != {"nf4", "bf16"}:
        raise ValueError("both fixed recipe identities required")
    for sha in spec["recipe_sha256"].values():
        _sha(sha)
    _sha(spec["provenance_audit_sha256"])
    _sha(spec["task_taxonomy_sha256"])
    if role == "confirmation":
        _sha(spec["pilot_analysis_sha256"])
    if _sha(spec["inventory_sha256"]) != digest_json(inventory):
        raise ValueError("inventory does not match its sealed identity")
    planning = spec["planning"]
    if planning.get("variance_multipliers") != [1.0, 2.0] or not 0.5 < _finite(planning["target_joint_power"]) < 1:
        raise ValueError("predeclared 1x/2x planning scenarios and finite joint-power target required")
    for key in ("assumed_relative_nll", "assumed_accuracy_loss"):
        _finite(planning[key])
    tasks = spec["task_types"]
    if len(tasks) != 22 or len(set(tasks)) != 22 or any(not isinstance(x, str) or not x for x in tasks):
        raise ValueError("exact 22 predeclared task types required")
    seed_sets = []
    for split in ("pilot", "confirmation"):
        seeds = spec["paired_seeds"][split]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("at least two distinct paired recipe seeds required")
        seed_sets.append(set(seeds))
        _count(spec["groups"][split], 2)
        if not 1 <= _count(spec["minimum_groups_per_task"][split]) <= spec["groups"][split]:
            raise ValueError("invalid task coverage requirement")
    if seed_sets[0] & seed_sets[1]:
        raise ValueError("pilot and confirmation training seeds overlap")
    base = inventory_groups(inventory)
    parent = {g: g for g in base.values()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for row in inventory:
        _sha(row["raw_row_sha256"])
        if row["role"] not in {"exposed", "pilot", "confirmation"} or row["task"] not in tasks:
            raise ValueError("unknown inventory role/task")
        for key in ("session", "event"):
            value = row.get(key)
            if not isinstance(value, str) or not value or value.lower() in {"unknown", "none", "null"}:
                raise ValueError("reviewed session/event provenance required")
            tag = (key, value)
            if tag in owners:
                a, b = sorted((find(base[row["id"]]), find(owners[tag])))
                parent[b] = a
            owners[tag] = base[row["id"]]
    groups = {identifier: find(group) for identifier, group in base.items()}
    roles = defaultdict(set)
    for row in inventory:
        roles[groups[row["id"]]].add(row["role"])
    if any(len(values) != 1 for values in roles.values()):
        raise ValueError("connected recording/video/session/event component crosses exposure/pilot/final roles")
    rows = [row for row in inventory if row["role"] == role]
    if len({groups[r["id"]] for r in rows}) != spec["groups"][role]:
        raise ValueError("actual independent-group label count differs from sealed design")
    coverage = {task: len({groups[r["id"]] for r in rows if r["task"] == task}) for task in tasks}
    if min(coverage.values()) < spec["minimum_groups_per_task"][role]:
        raise ValueError("predeclared task coverage is missing")
    return groups, {"rows": len(rows), "groups": spec["groups"][role], "task_group_coverage": coverage}


def paired_totals(trained, frozen, groups, tasks, *, seeds):
    """Generalize the original sufficient statistics without changing weights."""
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or set(trained) != set(seeds):
        raise ValueError("complete predeclared paired seed set required")
    if set(frozen) != {"nf4", "bf16"} or any(set(trained[s]) != {"nf4", "bf16"} for s in seeds):
        raise ValueError("both trained arms per seed and both fixed frozen references required")
    anchor = trained[seeds[0]]["nf4"]
    for rows in [*frozen.values(), *(trained[s][a] for s in seeds for a in ("nf4", "bf16"))]:
        validate_scores(rows)
        if len(rows) != len(anchor) or any(
            any(a[k] != b[k] for k in INPUT_KEYS) for a, b in zip(anchor, rows, strict=True)
        ):
            raise ValueError("paired inputs, row order, masks, media or supervised tokens differ")
    if set(groups) != {r["id"] for r in anchor} or set(tasks) != set(groups):
        raise ValueError("score rows and sealed group/task inventory differ")
    indices = defaultdict(list)
    for i, row in enumerate(anchor):
        if not isinstance(groups[row["id"]], str) or not groups[row["id"]]:
            raise ValueError("nonempty group labels required")
        indices[groups[row["id"]]].append(i)
    totals = {}
    for seed in seeds:
        totals[seed] = []
        for group in sorted(indices):
            ids = indices[group]
            a, b = trained[seed]["nf4"], trained[seed]["bf16"]
            totals[seed].append(
                (
                    len(ids),
                    sum(a[i]["native_supervised_tokens"] for i in ids),
                    math.fsum(a[i]["native_sft_ce"] * a[i]["native_supervised_tokens"] for i in ids),
                    math.fsum(b[i]["native_sft_ce"] * b[i]["native_supervised_tokens"] for i in ids),
                    sum(a[i]["restricted_choice_correct"] for i in ids),
                    sum(b[i]["restricted_choice_correct"] for i in ids),
                    sum(frozen["nf4"][i]["restricted_choice_correct"] for i in ids),
                )
            )
    return totals


def resample(totals, *, crossed, repetitions=REPETITIONS):
    """Same Python RNG, quantiles and paired whole-group draws as original code."""
    _count(repetitions)
    seeds = tuple(totals)
    groups = len(totals[seeds[0]])
    if len(seeds) < 2 or groups < 2 or any(len(totals[s]) != groups for s in seeds):
        raise ValueError("complete crossed seed/group design with at least two of each required")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {k: [] for k in (*MARGINS, "gain_over_frozen_nf4")}
    for _ in range(repetitions):
        selected_seeds = rng.choices(seeds, k=len(seeds))
        shared = rng.choices(range(groups), k=groups) if crossed else None
        cells = []
        for seed in selected_seeds:
            selected = shared if crossed else rng.choices(range(groups), k=groups)
            cells.extend(totals[seed][i] for i in selected)
        for key, value in _statistics(cells).items():
            samples[key].append(value)
    bounds = {k: _interval(v) for k, v in samples.items()}
    passed = {k: bounds[k]["upper_one_sided_95"] <= margin for k, margin in MARGINS.items()}
    return {
        "bounds": bounds,
        "ni_criteria_passed": passed,
        "both_ni_passed": all(passed.values()),
        "useful_adaptation_supported": bounds["gain_over_frozen_nf4"]["lower_one_sided_95"] > 0,
        "all_original_quality_guards_passed": all(passed.values())
        and bounds["gain_over_frozen_nf4"]["lower_one_sided_95"] > 0,
        "accuracy_improvement_lower_one_sided_95": -bounds["accuracy_loss"]["upper_one_sided_95"],
        "accuracy_improvement_supported": bounds["accuracy_loss"]["upper_one_sided_95"] < 0,
        "repetitions": repetitions,
        "crossed_shared_group_draw": crossed,
    }


def variance_diagnostics(totals):
    """Size-weighted ratio influences, crossed moments and deletion sensitivity.

    Negative component estimates remain visible. Unshrunk seed/group-mean
    variances provide a deliberately conservative *scenario*, not a confidence
    bound. More questions cannot remove the recipe training-seed variance term.
    """
    seeds = tuple(totals)
    s, g = len(seeds), len(totals[seeds[0]])
    if s < 2 or g < 2 or any(len(totals[k]) != g for k in seeds):
        raise ValueError("complete crossed seed/group matrix required")
    cells = [r for k in seeds for r in totals[k]]
    point = _statistics(cells)
    rows = math.fsum(r[0] for r in cells)
    bf_ce = math.fsum(r[3] for r in cells)
    result = {}
    for key in MARGINS:
        theta = point[key]
        den = (bf_ce if key == "relative_native_token_nll" else rows) / (s * g)
        matrix = [
            [
                ((r[2] - (1 + theta) * r[3]) if key == "relative_native_token_nll" else (r[5] - r[4] - theta * r[0]))
                / den
                for r in totals[k]
            ]
            for k in seeds
        ]
        means_s = [statistics.fmean(r) for r in matrix]
        means_g = [statistics.fmean(matrix[i][j] for i in range(s)) for j in range(g)]
        mu = statistics.fmean(means_s)
        interaction = math.fsum(
            (matrix[i][j] - means_s[i] - means_g[j] + mu) ** 2 for i in range(s) for j in range(g)
        ) / ((s - 1) * (g - 1))
        a, b = statistics.variance(means_s) - interaction / g, statistics.variance(means_g) - interaction / s
        result[key] = {
            "raw_seed_variance": a,
            "raw_group_variance": b,
            "interaction_variance": interaction,
            "seed_variance": max(0.0, a),
            "group_variance": max(0.0, b),
            "unshrunk_seed_mean_variance": statistics.variance(means_s),
            "unshrunk_group_mean_variance": statistics.variance(means_g),
            "seed_influences": means_s,
            "group_influences": means_g,
        }
    return {
        "point_estimates": point,
        "variance_components": result,
        "leave_one_group_out": [
            _statistics([r for k in seeds for i, r in enumerate(totals[k]) if i != j]) for j in range(g)
        ],
        "leave_one_seed_out": {
            str(k): _statistics([r for other in seeds if other != k for r in totals[other]]) for k in seeds
        },
    }


def joint_power(diagnostics, *, seeds, groups, variance_multiplier=1.0, assumed_effects=None, unshrunk=True):
    """Approximate all-four-test success lower bound, without independence claims.

    Normal plug-in power per metric/method; union bound combines four failures.
    This is NOT a calibrated bound on actual bootstrap power. Pilot transport,
    Gaussianity and variance estimates are assumptions, and must be reported.
    """
    _count(seeds, 2)
    _count(groups, 2)
    scale = _finite(variance_multiplier)
    if scale <= 0:
        raise ValueError("positive variance multiplier required")
    effects = dict(dict.fromkeys(MARGINS, 0.0) if assumed_effects is None else assumed_effects)
    if set(effects) != set(MARGINS):
        raise ValueError("both NI effect assumptions required")
    powers = {}
    for key, margin in MARGINS.items():
        c = diagnostics["variance_components"][key]
        a = c["unshrunk_seed_mean_variance"] if unshrunk else c["seed_variance"]
        b = c["unshrunk_group_mean_variance"] if unshrunk else c["group_variance"]
        e = 0.0 if unshrunk else c["interaction_variance"]
        if any(_finite(v) < 0 for v in (a, b, e)):
            raise ValueError("nonnegative variance scenario required")
        gap = margin - _finite(effects[key])
        for method in ("hierarchical", "crossed"):
            variance = scale * (
                a / seeds + b / (groups if method == "crossed" else groups * seeds) + e / (seeds * groups)
            )
            power = (
                statistics.NormalDist().cdf(gap / math.sqrt(variance) - statistics.NormalDist().inv_cdf(0.95))
                if variance
                else float(gap > 0)
            )
            powers[f"{method}:{key}"] = {
                "approximate_power": power,
                "variance": variance,
                "seed_variance_floor": scale * a / seeds,
            }
    return {
        "paired_seeds": seeds,
        "groups": groups,
        "variance_multiplier": scale,
        "assumed_effects": effects,
        "model": "unshrunk_seed_and_group" if unshrunk else "crossed_moment_components",
        "tests": powers,
        "approximate_joint_lower_bound": max(0.0, 1 - math.fsum(1 - p["approximate_power"] for p in powers.values())),
        "calibrated_bootstrap_power": False,
        "group_sufficiency_established": False,
    }


def design_identity(spec):
    """Pilot and final share the entire prospectively locked design."""
    return digest_json({**spec, "pilot_analysis_sha256": None})


def pilot_planning(spec, diagnostics):
    planning = spec["planning"]
    scenarios = [
        joint_power(
            diagnostics,
            seeds=len(spec["paired_seeds"]["confirmation"]),
            groups=spec["groups"]["confirmation"],
            variance_multiplier=x,
            assumed_effects={
                "relative_native_token_nll": planning["assumed_relative_nll"],
                "accuracy_loss": planning["assumed_accuracy_loss"],
            },
        )
        for x in planning["variance_multipliers"]
    ]
    eligible = all(x["approximate_joint_lower_bound"] >= planning["target_joint_power"] for x in scenarios)
    return scenarios, eligible


def analyze(trained, frozen, inventory, spec, *, role, repetitions=REPETITIONS):
    groups, coverage = validate_design(spec, inventory, role=role)
    selected = {r["id"]: r for r in inventory if r["role"] == role}
    tasks = {k: r["task"] for k, r in selected.items()}
    totals = paired_totals(trained, frozen, {k: groups[k] for k in selected}, tasks, seeds=spec["paired_seeds"][role])
    for row in trained[spec["paired_seeds"][role][0]]["nf4"]:
        if (
            row["media_sha256"] != selected[row["id"]]["sha256"]
            or row["raw_row_sha256"] != selected[row["id"]]["raw_row_sha256"]
        ):
            raise ValueError("scored row/media differs from sealed inventory")
    methods = {
        "hierarchical": resample(totals, crossed=False, repetitions=repetitions),
        "crossed": resample(totals, crossed=True, repetitions=repetitions),
    }
    pooled = {a: [r for s in spec["paired_seeds"][role] for r in trained[s][a]] for a in ("nf4", "bf16")}
    slices = {
        t: paired_metrics(
            [r for r in pooled["nf4"] if tasks[r["id"]] == t], [r for r in pooled["bf16"] if tasks[r["id"]] == t]
        )
        for t in spec["task_types"]
    }
    diagnostics = variance_diagnostics(totals)
    planning, planning_eligible = pilot_planning(spec, diagnostics)
    return {
        "protocol": PROTOCOL,
        "role": role,
        "spec_sha256": digest_json(spec),
        "design_identity_sha256": design_identity(spec),
        "recipe_sha256": spec["recipe_sha256"],
        "inventory_sha256": spec["inventory_sha256"],
        "coverage": coverage,
        "paired_seeds": spec["paired_seeds"][role],
        "pooled": {a: metrics(v) for a, v in pooled.items()},
        "fixed_frozen": {a: metrics(v) for a, v in frozen.items()},
        "uncertainty": methods,
        "diagnostics": diagnostics,
        "task_slices_exploratory": slices,
        "planning": planning,
        "pilot_planning_eligible": planning_eligible,
        "planning_eligibility_is_not_power_or_independence_proof": True,
        "numerical_ni_passed": role == "confirmation"
        and repetitions == REPETITIONS
        and all(v["both_ni_passed"] for v in methods.values()),
        "numerical_original_quality_guards_passed": role == "confirmation"
        and repetitions == REPETITIONS
        and all(v["all_original_quality_guards_passed"] for v in methods.values()),
        "conditional_recipe_ni_passed": False,
        "conditional_original_quality_guards_passed": False,
        "release_qualified": False,
        "provenance_truth_or_population_independence_verified": False,
        "artifact_and_provenance_review_required": True,
        "limitations": [
            "Supplied provenance is consistency-checked, not independent acquisition or population proof.",
            "Pilot planning is approximate; no group count guarantees joint bootstrap power.",
            "Task and deletion slices are exploratory, not new selection or removal rules.",
            "New recipe seeds require actual paired training; fixed frozen scores add no seed replication.",
            "Noninferiority alone establishes neither quality improvement nor lower time to a fixed quality target; both need separate evidence.",
        ],
    }


def _read_bound(item, *, jsonl=False):
    import hashlib

    path = Path(item["path"])
    if path.suffix not in {".json", ".jsonl"} or path.is_symlink():
        raise ValueError("regular metadata paths required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(item["sha256"]):
        raise ValueError("metadata hash mismatch")
    return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)


def analyze_bundle(bundle):
    """Check immutable external review/score bindings before conditional NI."""
    spec, inventory = _read_bound(bundle["spec"]), _read_bound(bundle["inventory"])
    provenance = _read_bound(bundle["provenance_audit"])
    taxonomy = _read_bound(bundle["task_taxonomy"])
    relationships = _read_bound(bundle["exposure_and_relationship_source"])
    if (
        bundle["task_taxonomy"]["sha256"] != spec["task_taxonomy_sha256"]
        or taxonomy.get("protocol") != "cosmos.qlora-confirmation-task-taxonomy.v1"
        or taxonomy.get("task_types") != spec["task_types"]
    ):
        raise ValueError("actual task taxonomy does not bind the declared task inventory")
    if (
        bundle["exposure_and_relationship_source"]["sha256"]
        != provenance.get("exposure_and_relationship_source_sha256")
        or relationships.get("inventory_sha256") != spec["inventory_sha256"]
    ):
        raise ValueError("actual relationship/exposure source does not bind this inventory")
    if (
        bundle["provenance_audit"]["sha256"] != spec["provenance_audit_sha256"]
        or provenance.get("protocol") != "cosmos.qlora-confirmation-provenance.v1"
        or provenance.get("inventory_sha256") != spec["inventory_sha256"]
        or provenance.get("task_taxonomy_sha256") != spec["task_taxonomy_sha256"]
        or provenance.get("complete") is not True
        or provenance.get("unresolved_relationships") != []
    ):
        raise ValueError("complete provenance review is not bound to this inventory")
    _sha(provenance["exposure_and_relationship_source_sha256"])
    exposed = relationships.get("known_exposed_rows")
    if not isinstance(exposed, list) or not exposed or len({r["id"] for r in exposed}) != len(exposed):
        raise ValueError("actual nonempty reviewed known-exposure row inventory required")
    by_id = {r["id"]: r for r in inventory}
    for row in exposed:
        actual = by_id.get(row["id"], {})
        if actual.get("role") != "exposed" or any(
            actual.get(k) != row.get(k) for k in ("id", "sha256", "raw_row_sha256", "task", "session", "event")
        ):
            raise ValueError("known exposed row omitted or changed in sealed inventory")
    if bundle["role"] == "confirmation":
        pilot = _read_bound(bundle["pilot_analysis"])
        if (
            bundle["pilot_analysis"]["sha256"] != spec["pilot_analysis_sha256"]
            or pilot.get("protocol") != PROTOCOL
            or pilot.get("role") != "pilot"
            or pilot.get("design_identity_sha256") != design_identity(spec)
            or pilot.get("paired_seeds") != spec["paired_seeds"]["pilot"]
            or pilot.get("coverage", {}).get("groups") != spec["groups"]["pilot"]
            or pilot.get("recipe_sha256") != spec["recipe_sha256"]
            or pilot.get("inventory_sha256") != spec["inventory_sha256"]
            or pilot.get("external_bindings_validated") is not True
        ):
            raise ValueError("separate pilot does not bind this exact prospective design")
        expected_planning, eligible = pilot_planning(spec, pilot["diagnostics"])
        if (
            not eligible
            or pilot.get("planning") != expected_planning
            or pilot.get("pilot_planning_eligible") is not True
        ):
            raise ValueError("pilot planning rejects this candidate design; revise and reseal before final outcomes")
    score_bundle = {key: bundle[key] for key in ("trained", "frozen")}
    artifact = _read_bound(bundle["artifact_validation"])
    if (
        artifact.get("protocol") != "cosmos.qlora-confirmation-score-artifacts.v1"
        or artifact.get("complete") is not True
        or artifact.get("recipe_sha256") != spec["recipe_sha256"]
        or artifact.get("paired_seeds") != spec["paired_seeds"][bundle["role"]]
        or artifact.get("inventory_sha256") != spec["inventory_sha256"]
        or artifact.get("score_bundle_sha256") != digest_json(score_bundle)
    ):
        raise ValueError("successful artifact proof does not bind these recipes/seeds/score files")
    trained = {
        int(s): {a: _read_bound(v, jsonl=True) for a, v in arms.items()} for s, arms in bundle["trained"].items()
    }
    if len(trained) != len(bundle["trained"]):
        raise ValueError("duplicate canonical seed keys")
    frozen = {a: _read_bound(v, jsonl=True) for a, v in bundle["frozen"].items()}
    result = analyze(trained, frozen, inventory, spec, role=bundle["role"])
    result["inputs_sha256"] = digest_json(bundle)
    result["external_bindings_validated"] = True
    result["external_bindings_scope"] = (
        "Hash and metadata consistency only; reviewed source assertions are not independently established population facts."
    )
    result["conditional_recipe_ni_passed"] = result["numerical_ni_passed"]
    result["conditional_original_quality_guards_passed"] = result["numerical_original_quality_guards_passed"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="hash-bound spec/inventory/paired score bundle")
    parser.add_argument("--output", type=Path, required=True, help="exclusive new result; never overwrites history")
    args = parser.parse_args()
    result = analyze_bundle(json.loads(args.inputs.read_text()))
    write_json_new(args.output, result)


if __name__ == "__main__":
    main()
