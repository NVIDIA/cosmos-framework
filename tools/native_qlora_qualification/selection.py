# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Select settings from the sealed WTS development experiment, never final data.

This emits a development decision, NOT cosmos.qlora-selection.v1 authorization.
Future confirmation/frozen-baseline scoring configurations must be locked in a
separate receipt before untouched/final scoring. No checkpoint or job is changed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.native_qlora_qualification.common import digest_json, write_json_new
from tools.native_qlora_qualification.performance import GIB, summarize_run
from tools.native_qlora_qualification.protocol import Experiment
from tools.native_qlora_qualification.quality import INPUT_KEYS, validate_scores
from tools.native_qlora_qualification.score import aggregate

PROTOCOL = "cosmos.qlora-development-selection.v1"
RULE = "Within each precision, minimize development supervised-token NLL among completed finite candidates with owned device-process sampled peak <=24GiB and development restricted-choice accuracy no more than1percentage point below that precision baseline candidate; tie by smaller peak then elapsed seconds. Selection does not imply quality-gate success."
CANDIDATES = {
    "lr_5e5": (16, 32, 5e-5),
    "baseline": (16, 32, 1e-4),
    "lr_2e4": (16, 32, 2e-4),
    "rank32": (32, 64, 1e-4),
}
GATES = {
    "accuracy_gain_vs_frozen_quantized_one_sided_95_lower_min_exclusive": 0.0,
    "accuracy_loss_one_sided_95_upper_max": 0.01,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 2026,
    "paired_training_seeds": [17, 42, 123],
    "relative_supervised_token_nll_one_sided_95_upper_max": 0.02,
    "unit": "related-source-event-component; paired hierarchical seed and group bootstrap",
}


class InvalidSelection(ValueError):
    """Fixed public-safe contract explanation; never includes private source text."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidSelection(reason)


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_bound(path: Path, expected: str | None = None, *, lines: bool = False) -> tuple[Any, str]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        require(is_sha(expected) and actual == expected, "externally pinned input SHA256 differs")
    if lines:
        require(
            bool(raw.splitlines()) and all(line.strip() for line in raw.splitlines()),
            "score JSONL is empty or contains blank rows",
        )
        return [json.loads(line) for line in raw.splitlines()], actual
    return json.loads(raw), actual


def validate_plan(plan: dict, manifest: dict, manifest_sha256: str) -> dict[tuple[str, str], dict]:
    require(plan.get("protocol") == "cosmos.qlora-development-plan.v1", "unsupported development plan protocol")
    require(
        plan.get("selection_rule") == RULE and plan.get("fixed_gates") == GATES,
        "selection rule or fixed release gates changed",
    )
    require(
        plan.get("training_seeds") == [42] and plan.get("paired_confirmation_seeds") == [17, 42, 123],
        "predeclared seed policy differs",
    )
    require(
        plan.get("fresh_base_required") is True
        and plan.get("failed_candidates_retained") is True
        and plan.get("initialization_seed") == "equal training seed in both arms",
        "fresh-base or failed-candidate policy differs",
    )
    require(plan.get("development_manifest_sha256") == manifest_sha256, "plan refers to another development manifest")
    require(
        manifest.get("protocol") == "cosmos.wts-group-split.v1" and manifest.get("role") == "development",
        "selection requires development data, never regression/final",
    )
    require(
        manifest.get("data_sha256") == plan.get("development_data_sha256")
        and is_sha(plan.get("fresh_training_data_sha256")),
        "sealed development/training data identity differs",
    )
    rows = manifest.get("rows", [])
    require(
        len(rows) == 584 and len({row["id"] for row in rows}) == 584,
        "development manifest requires all 584 unique rows",
    )
    require(
        manifest.get("group_count") == 12 and len({row["group_sha256"] for row in rows}) == 12,
        "development manifest requires exactly 12 connected groups",
    )
    require(
        all(is_sha(row[key]) for row in rows for key in ("raw_row_sha256", "group_sha256", "media_sha256")),
        "development manifest lacks exact row/group/media digests",
    )
    planned = {}
    require(len(plan.get("candidates", [])) == 4, "all four predeclared candidates are required")
    for entry in plan["candidates"]:
        candidate = entry["candidate"]
        require(candidate in CANDIDATES and len(entry["ordered_runs"]) == 2, "unknown or incomplete candidate plan")
        for run in entry["ordered_runs"]:
            name = run["name"]
            arms = [arm for arm in ("nf4", "bf16") if name == f"dev_{candidate}_{arm}_seed42"]
            require(
                len(arms) == 1 and run["file"] == name + ".json" and is_sha(run["sha256"]),
                "invalid planned candidate specification",
            )
            key = candidate, arms[0]
            require(key not in planned, "duplicate candidate/precision plan")
            planned[key] = run
    require(
        set(planned) == {(candidate, arm) for candidate in CANDIDATES for arm in ("nf4", "bf16")},
        "plan does not contain all eight candidates",
    )
    return planned


def candidate_scores(
    root: Path, contract: dict, spec: Experiment, manifest: dict, manifest_sha256: str
) -> tuple[dict, list[dict], dict]:
    evidence = {}
    loaded = {}
    for name in ("summary.json", "selection.json", "model_identity.json", "scores.jsonl"):
        loaded[name], evidence[name] = read_bound(root / "scores" / name, lines=name.endswith("jsonl"))
    final, evidence["final-checkpoint.json"] = read_bound(root / "final-checkpoint.json")
    probe, evidence["dcp-probe.json"] = read_bound(root / "probes/dcp/probe.json")
    command, evidence["scoring-command.json"] = read_bound(root / "stages/scoring/command.json")
    summary, selected, identity, rows = (
        loaded[name] for name in ("summary.json", "selection.json", "model_identity.json", "scores.jsonl")
    )
    expected_identity = {
        key: value for key, value in final["metadata"]["identity"].items() if key != "training_contract"
    }
    require(identity == expected_identity, "scoring model identity differs from the final native checkpoint")
    require(
        isinstance(command, list) and command.count("--checkpoint") == 1,
        "scoring command lacks one explicit native checkpoint",
    )
    checkpoint_index = command.index("--checkpoint")
    require(
        checkpoint_index + 1 < len(command) and command[checkpoint_index + 1] == final.get("checkpoint"),
        "scoring command did not use the verified final native checkpoint",
    )
    require(
        isinstance(summary.get("checkpoint"), str)
        and isinstance(final.get("checkpoint"), str)
        and Path(summary["checkpoint"]).resolve() == Path(final["checkpoint"]).resolve(),
        "scorer checkpoint differs from the verified final checkpoint",
    )
    require(
        probe.get("protocol") == "cosmos.qlora-integration-probe.v1"
        and probe.get("checkpoint") == final["checkpoint"]
        and probe.get("identity_sha256") == digest_json(identity),
        "fresh DCP probe identity differs from the final native model",
    )
    require(
        probe.get("trainable_tensors") == probe.get("optimizer_parameters") == len(probe.get("adapters", {})) == 504
        and set(probe.get("adapter_updates", {})) == {"A", "B"}
        and all(type(count) is int and 1 <= count <= 252 for count in probe["adapter_updates"].values()),
        "fresh DCP probe lacks complete adapter membership/updates",
    )
    require(
        summary.get("adapter_tensors_sha256") == digest_json(probe["adapters"]),
        "scoring adapter tensors differ from the fresh final DCP probe",
    )
    require(
        summary.get("protocol") == "cosmos.wts-heldout.v1" and summary.get("complete") is True,
        "scoring summary is incomplete or unsupported",
    )
    require(
        summary.get("adapter_state") == "reloaded_checkpoint" and summary.get("checkpoint") is not None,
        "candidate requires trained checkpoint scoring",
    )
    require(
        summary.get("recipe") == spec.recipe
        and summary.get("overrides") == spec.overrides()
        and summary.get("model_seed") == 42,
        "scorer configuration differs from the sealed candidate",
    )
    require(
        summary.get("world_size") == 1 and summary.get("max_length") == spec.max_length,
        "scoring strategy or sequence limit differs",
    )
    require(
        summary.get("scores_sha256") == evidence["scores.jsonl"]
        and summary.get("selection_sha256") == digest_json(selected)
        and summary.get("model_identity_sha256") == digest_json(identity),
        "scorer artifact digest differs",
    )
    require(
        summary.get("build_receipt")
        == {
            "COSMOS_TRAINING_CODE_REVISION": contract["build"]["code_revision"],
            "COSMOS_TRAINING_WHEEL_SHA256": contract["build"]["wheel_sha256"],
        },
        "scorer build differs from native run",
    )
    require(
        selected.get("split_role") == "development" and selected.get("split_manifest_sha256") == manifest_sha256,
        "scored split is not the sealed development split",
    )
    require(
        selected.get("selected_rows") == selected.get("source_rows") == 584
        and selected.get("group_count") == 12
        and selected.get("source_sha256") == manifest["data_sha256"],
        "scorer did not consume all sealed development rows",
    )
    require(len(rows) == 584, "development score rows are missing or extra")
    validate_scores(rows)
    for actual, expected in zip(rows, manifest["rows"], strict=True):
        require(
            all(actual[key] == expected[key] for key in ("id", "raw_row_sha256", "media_sha256")),
            "ordered score row/media differs from development manifest",
        )
    require(
        selected.get("ordered_ids") == [row["id"] for row in rows]
        and selected.get("ordered_row_identities") == [row["row_identity"] for row in rows],
        "scorer ordered selection identity differs",
    )
    content = digest_json(
        [{key: row[key] for key in ("id", "row_identity", "gold", "choices", "media_sha256")} for row in rows]
    )
    require(
        summary.get("selected_content_sha256") == selected.get("selected_content_sha256") == content,
        "selected content identity differs",
    )
    require(
        summary.get("processor_contract") == identity.get("processor_contract"), "scorer processor identity differs"
    )
    require(
        summary.get("quantized_matrices") == (252 if spec.arm == "nf4" else 0), "observed quantization coverage differs"
    )
    computed = aggregate(rows)
    require(summary.get("metrics") == computed, "summary metrics do not exactly recompute from all development rows")
    stable_paths = {name: root / "scores" / name for name in loaded}
    stable_paths.update(
        {
            "final-checkpoint.json": root / "final-checkpoint.json",
            "dcp-probe.json": root / "probes/dcp/probe.json",
            "scoring-command.json": root / "stages/scoring/command.json",
        }
    )
    for name, file in stable_paths.items():
        read_bound(file, evidence[name], lines=name.endswith("jsonl"))
    return computed, rows, evidence


def select_development(
    plan_path: Path,
    plan_sha256: str,
    development_manifest_path: Path,
    development_manifest_sha256: str,
    runs: list[dict],
) -> dict:
    plan, _ = read_bound(plan_path, plan_sha256)
    manifest, _ = read_bound(development_manifest_path, development_manifest_sha256)
    planned = validate_plan(plan, manifest, development_manifest_sha256)
    provided = {}
    for item in runs:
        require(set(item) == {"candidate", "arm", "path", "contract_sha256"}, "run input schema differs")
        key = item["candidate"], item["arm"]
        require(
            key in planned and key not in provided and is_sha(item["contract_sha256"]),
            "duplicate, unknown or unpinned run input",
        )
        provided[key] = item
    records, success_rows, performance = [], {}, {}
    for key, entry in planned.items():
        candidate, arm = key
        record = {
            "candidate": candidate,
            "arm": arm,
            "state": "incomplete",
            "eligible": False,
            "reasons": [],
            "evidence_sha256": {},
        }
        records.append(record)
        if key not in provided:
            record["reasons"].append("missing_candidate_input")
            continue
        item = provided[key]
        record["contract_sha256"] = item["contract_sha256"]
        root = Path(item["path"])
        try:
            contract, _ = read_bound(root / "contract.json", item["contract_sha256"])
            rank, alpha, lr = CANDIDATES[candidate]
            spec = Experiment(entry["name"], arm, rank=rank, alpha=alpha, learning_rate=lr)
            require(
                contract.get("experiment") == spec.as_dict() and contract.get("overrides") == spec.overrides(),
                "native experiment differs from the predeclared candidate",
            )
            require(
                contract.get("action") == "train-eval"
                and contract.get("split") == "development"
                and "checkpoint" in contract
                and contract["checkpoint"] is None,
                "candidate is not fresh development train-eval",
            )
            files = contract.get("files", [])
            require(len(files) >= 6, "run contract lacks exact input identities")
            require(
                files[0]["sha256"] == entry["sha256"]
                and [row["sha256"] for row in files[2:6]]
                == [
                    development_manifest_sha256,
                    plan["development_data_sha256"],
                    plan["fresh_training_data_sha256"],
                    plan["development_data_sha256"],
                ],
                "run input/specification digests differ from the sealed plan",
            )
            completion, record["evidence_sha256"]["completion"] = read_bound(root / "completion.json")
            require(completion.get("experiment") == spec.as_dict(), "completion experiment differs")
            if completion.get("completed") is False:
                require(
                    isinstance(completion.get("failure"), str) and bool(completion["failure"]),
                    "failed candidate lacks terminal failure evidence",
                )
                record["state"] = "failed"
                record["reasons"].append("native_experiment_failed")
                record["failure_sha256"] = digest_json(completion["failure"])
                continue
            require(
                completion.get("completed") is True and completion.get("failure") is None,
                "candidate completion is ambiguous",
            )
            measured = summarize_run(root)
            require(measured["timing_eligible"] is True, "native performance evidence is incomplete or invalid")
            computed, rows, score_evidence = candidate_scores(
                root, contract, spec, manifest, development_manifest_sha256
            )
            record["evidence_sha256"].update(score_evidence)
            record["performance_evidence_sha256"] = digest_json(measured)
            record["performance_source_sha256"] = measured["source_sha256"]
            record["metrics"] = computed
            record["hyperparameters"] = {"rank": rank, "alpha": alpha, "learning_rate": lr}
            record["complete_elapsed_seconds"] = measured["complete_elapsed_seconds"]
            resource = measured["memory_by_stage"]["training"]["sampled"]
            peak = resource["owned_device_process_peak_bytes"]
            record["sampled_training_peak_bytes"] = peak
            record["memory_samples_are_lower_bounds"] = True
            if resource["gpu_memory_evidence_valid"] is not True or type(peak) is not int or peak <= 0:
                record["reasons"].append("invalid_owned_device_process_memory")
            elif peak > 24 * GIB:
                record["reasons"].append("over_24gib_best_fit_budget")
            record["state"] = "completed"
            success_rows[key], performance[key] = rows, measured
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["state"] = "incomplete"
            record["reasons"].append(str(error) if isinstance(error, InvalidSelection) else type(error).__name__)
            record["diagnostic_sha256"] = digest_json(str(error))
    report = {
        "protocol": PROTOCOL,
        "development_plan_sha256": plan_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "runs_input_sha256": digest_json(runs),
        "development_rows": 584,
        "development_groups": 12,
        "candidates": records,
        "complete": all(row["state"] != "incomplete" for row in records),
        "selection_rule": RULE,
        "fixed_release_gates": GATES,
        "confirmation_seeds_required": [17, 42, 123],
        "confirmation_scoring_configurations_locked": False,
        "untouched_scoring_authorized": False,
        "release_qualified": False,
        "selected": {"nf4": None, "bf16": None},
        "arm_status": {},
    }
    if success_rows:
        anchor = next(iter(success_rows.values()))
        shared_keys = (
            "base_sha256",
            "processor_sha256",
            "targets_sha256",
            "build_sha256",
            "tool_sha256",
            "data_sha256",
            "runtime_sha256",
        )
        anchor_perf = next(iter(performance.values()))
        for key, rows in success_rows.items():
            if any(any(a[field] != b[field] for field in INPUT_KEYS) for a, b in zip(anchor, rows, strict=True)) or any(
                performance[key]["pairing"][field] != anchor_perf["pairing"][field] for field in shared_keys
            ):
                report["complete"] = False
                next(row for row in records if (row["candidate"], row["arm"]) == key)["reasons"].append(
                    "candidate_inputs_processor_runtime_or_build_differ"
                )
    for arm in ("nf4", "bf16"):
        arm_rows = [row for row in records if row["arm"] == arm]
        baseline = next(row for row in arm_rows if row["candidate"] == "baseline")
        if baseline["state"] != "completed":
            report["arm_status"][arm] = "baseline_unavailable"
            continue
        base_correct = sum(row["restricted_choice_correct"] for row in success_rows[("baseline", arm)])
        for row in arm_rows:
            if row["state"] != "completed":
                continue
            correct = sum(r["restricted_choice_correct"] for r in success_rows[(row["candidate"], arm)])
            if 100 * (base_correct - correct) > 584:
                row["reasons"].append("accuracy_below_baseline_minus_one_percentage_point")
            row["eligible"] = not row["reasons"]
        eligible = sorted(
            (row for row in arm_rows if row["eligible"]),
            key=lambda row: (
                row["metrics"]["token_weighted_ce"],
                row["sampled_training_peak_bytes"],
                row["complete_elapsed_seconds"],
            ),
        )
        report.setdefault("ranking", {})[arm] = [row["candidate"] for row in eligible]
        if not report["complete"]:
            report["arm_status"][arm] = "incomplete_evidence"
        elif not eligible:
            report["arm_status"][arm] = "no_eligible_candidate"
        elif (
            len(eligible) > 1
            and all(
                eligible[0][field] == eligible[1][field]
                for field in ("sampled_training_peak_bytes", "complete_elapsed_seconds")
            )
            and eligible[0]["metrics"]["token_weighted_ce"] == eligible[1]["metrics"]["token_weighted_ce"]
        ):
            report["arm_status"][arm] = "unresolved_exact_tie"
        else:
            winner = eligible[0]
            report["selected"][arm] = {
                "candidate": winner["candidate"],
                "hyperparameters": winner["hyperparameters"],
                "contract_sha256": winner["contract_sha256"],
                "evidence_sha256": digest_json(winner),
            }
            report["arm_status"][arm] = "development_candidate_selected"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "development-manifest", "runs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "development-manifest-sha256", "runs-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    inputs, _ = read_bound(args.runs, args.runs_sha256)
    require(inputs.get("protocol") == "cosmos.qlora-development-runs.v1", "unsupported run-input protocol")
    result = select_development(
        args.plan, args.plan_sha256, args.development_manifest, args.development_manifest_sha256, inputs["runs"]
    )
    result["runs_file_sha256"] = args.runs_sha256
    result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json_new(args.output, result)


if __name__ == "__main__":
    main()
