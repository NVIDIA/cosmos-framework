# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prospective Alpamayo development analysis; never trains or selects favorable checkpoints.

Clips are averaged inside fixed coarse blocks. Bootstrap draws cross training
seeds and whole blocks, sharing each block draw across both arms and every seed.
There is no inner clip resampling. Coarse blocks are not verified independent
recording/event families; these intervals cannot qualify a population claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from tools.native_qlora_qualification.action_quality import read_json
from tools.native_qlora_qualification.common import write_json_new
from tools.native_qlora_qualification.performance import quantile

PROTOCOL_SHA = "c78a4ae86af4a874bfa7411b9b7f5c56c5220ca1850f04d45cf77a8e605e57ef"
MANIFEST_SHA = "2272c9fae7cd50eb13adaa8fd3bdaaa8467b91621c4983441b201b1faa97c810"
VALIDATION_SHA = "1bf0d48e2746a3639e34e2472226ac9ff3c1f015d3a449da6a95e948d27acba0"
CODE = "4cba88f4d84ce69037c194e58670e8491388a36d"
WHEEL = "9113351965b501d293f9a71f476926fc212c51732144a63f90aca144120f0697"
CALIBRATION_EVALUATOR_SHA = "1630f63829ff7c8c22918733e5fb725d2a9304eb3862feec324f06a26ee1d122"
SEEDS = (17, 42, 123)
MODES = ("nf4_lora", "bf16_lora", "bf16_frozen")
STEPS = (0, 128, 256, 384, 512)
ADE, FDE = "open_loop_ade_6p4s_meters", "open_loop_fde_6p4s_meters"
GUARDS = {ADE: {"relative": 0.05, "absolute_meters": 0.10}, FDE: {"relative": 0.05, "absolute_meters": 0.25}}


def admitted_source(path, admitted, split):
    """Reconstruct native media-content identity from already verified admission receipts.

    JSONL byte SHA is not the native cursor fingerprint. No model outcomes or new
    image decoding are involved; the native loader separately verifies those bytes.
    """
    from cosmos_framework.posttrain.utils.identity import canonical_json_bytes

    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != admitted["jsonl_sha256"][split]:
        raise ValueError(f"admitted {split} JSONL bytes differ")
    receipts = {item["row_id"]: item for item in admitted["clip_receipts"] if item["split"] == split}
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    seen = set()
    for row in rows:
        identifier = row["id"]
        if identifier in seen or identifier not in receipts:
            raise ValueError("native source identity requires exactly the admitted clip inventory")
        seen.add(identifier)
        receipt = receipts[identifier]
        if row["episode_id"] != receipt["episode_id"]:
            raise ValueError("native source episode differs from admission")

        def media_digest(path):
            if Path(path).parent.name != receipt["clip_id"]:
                raise ValueError("native media path differs from its admitted clip")
            return receipt["receipt"]["files"][Path(path).name]

        row["media"] = {
            camera: [media_digest(item) for item in paths] if isinstance(paths, list) else media_digest(paths)
            for camera, paths in row["media"].items()
        }
    if seen != set(receipts):
        raise ValueError("native source omitted admitted clips")
    content = canonical_json_bytes({"format": "flow-sft-v1", "rows": rows}, context="admitted Alpamayo source")
    return {"format": "physicalai-alpamayo-flow-v1", "sha256": hashlib.sha256(content).hexdigest()}


def load_groups(validation, manifest):
    admitted, digest = read_json(manifest)
    raw = Path(validation).read_bytes()
    if digest != MANIFEST_SHA or hashlib.sha256(raw).hexdigest() != VALIDATION_SHA:
        raise ValueError("Alpamayo analysis requires the exact admitted manifest and validation bytes")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    receipts = {item["row_id"]: item for item in admitted["clip_receipts"] if item["split"] == "validation"}
    selection = admitted["selection"]
    blocks = {tuple(block) for block in selection["blocks"]["validation"]}
    if blocks & {tuple(block) for block in selection["blocks"]["train"]}:
        raise ValueError("training and development coarse blocks overlap")
    groups, counts = {}, defaultdict(int)
    for row in rows:
        item = receipts.get(row["id"])
        if row["id"] in groups or item is None or item["episode_id"] != row["episode_id"]:
            raise ValueError("validation row/episode differs from its admitted clip")
        block = tuple(item["coarse_block"])
        if block not in blocks:
            raise ValueError("validation clip lies outside its admitted coarse block")
        groups[row["id"]] = block
        counts[block] += 1
    if len(groups) != 256 or set(groups) != set(receipts) or len(counts) != 32 or set(counts.values()) != {8}:
        raise ValueError("require exactly256 distinct clips in32 blocks with8 fixed clips each")
    if admitted["jsonl_sha256"]["validation"] != VALIDATION_SHA:
        raise ValueError("admission does not bind the validation source")
    admitted["native_sources"] = {
        "validation": admitted_source(validation, admitted, "validation"),
        "train": admitted_source(Path(validation).with_name("alpamayo-train.jsonl"), admitted, "train"),
    }
    return groups, admitted


def block_means(rows, groups, metric):
    values, seen = defaultdict(list), set()
    for row in rows:
        if not isinstance(row.get("ids"), list) or len(row["ids"]) != 1:
            raise ValueError("trajectory analysis requires one identified clip per native row")
        identifier = row["ids"][0]
        if identifier in seen or identifier not in groups:
            raise ValueError("duplicated or unadmitted trajectory row")
        seen.add(identifier)
        value = row["metrics"][metric]
        if set(value) != {"sum", "count"} or value["count"] != 1 or type(value["count"]) not in (int, float):
            raise ValueError("native trajectory metric must count exactly one physical trajectory")
        number = value["sum"]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("trajectory error must be finite and nonnegative")
        values[groups[identifier]].append(number)
    if seen != set(groups):
        raise ValueError("trajectory evaluation omitted admitted clips")
    return {block: sum(clips) / len(clips) for block, clips in sorted(values.items())}


def draw_mean(cube, seeds, blocks):
    """One block-index vector is common across every sampled training seed."""
    if not seeds or not blocks:
        raise ValueError("crossed draw must contain seeds and blocks")
    return sum(sum(cube[seed][block] for block in blocks) / len(blocks) for seed in seeds) / len(seeds)


def compare_recipes(candidate, reference, *, draws=10000, seed=70241):
    seed_ids = sorted(reference)
    if sorted(candidate) != seed_ids or len(seed_ids) < 2 or type(draws) is not int or draws < 1:
        raise ValueError("crossed recipe analysis needs paired seeds and positive bootstrap draws")
    blocks = sorted(reference[seed_ids[0]])
    if len(blocks) < 2:
        raise ValueError("crossed recipe analysis needs multiple whole blocks")
    cubes = []
    for arm in (candidate, reference):
        if any(sorted(arm[s]) != blocks for s in seed_ids):
            raise ValueError("every arm and seed must share the same blocks")
        cube = [[arm[s][b] for b in blocks] for s in seed_ids]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for row in cube for x in row):
            raise ValueError("block errors must be finite and nonnegative")
        cubes.append(cube)
    left, right = cubes
    s, b = len(seed_ids), len(blocks)
    point_l, point_r = (draw_mean(cube, list(range(s)), list(range(b))) for cube in cubes)
    if point_r <= 0:
        raise ValueError("relative error requires a positive reference mean")
    rng, absolute, relative = random.Random(seed), [], []
    for _ in range(draws):
        selected_seeds, selected_blocks = rng.choices(range(s), k=s), rng.choices(range(b), k=b)
        after, before = (draw_mean(cube, selected_seeds, selected_blocks) for cube in cubes)
        if before <= 0:
            raise ValueError("bootstrap relative error is undefined for a zero reference draw")
        absolute.append(after - before)
        relative.append(after / before - 1)
    per_seed = []
    for index, identifier in enumerate(seed_ids):
        before, after = sum(right[index]) / b, sum(left[index]) / b
        per_seed.append(
            {
                "seed": identifier,
                "candidate_meters": after,
                "reference_meters": before,
                "absolute_difference_meters": after - before,
                "relative_increase": after / before - 1 if before else None,
            }
        )
    return {
        "candidate_equal_seed_block_clip_meters": point_l,
        "reference_equal_seed_block_clip_meters": point_r,
        "absolute_difference_meters": point_l - point_r,
        "relative_increase": point_l / point_r - 1,
        "absolute_difference_upper_one_sided95_meters": quantile(absolute, 0.95),
        "relative_increase_upper_one_sided95": quantile(relative, 0.95),
        "per_seed": per_seed,
        "draws": draws,
        "bootstrap_seed": seed,
        "training_seeds": seed_ids,
        "coarse_blocks": b,
        "method": "paired crossed seed/block bootstrap; one whole-block draw shared across all sampled seeds and both arms; fixed clip averages, no inner clip resampling",
    }


def comparison(candidate, reference):
    result = {}
    for metric, margins in GUARDS.items():
        value = compare_recipes(candidate[metric], reference[metric])
        value["predeclared_margins"] = margins
        value["development_guard_met"] = (
            value["relative_increase_upper_one_sided95"] <= margins["relative"]
            and value["absolute_difference_upper_one_sided95_meters"] <= margins["absolute_meters"]
        )
        result[metric] = value
    result["joint_development_preservation_guards_met"] = all(result[m]["development_guard_met"] for m in GUARDS)
    result["development_ADE_improvement_upper_bound_below_zero"] = (
        result[ADE]["absolute_difference_upper_one_sided95_meters"] < 0
    )
    return result


def publications(run):
    done, timing = run["completion"], run["timing"]
    if (
        done.get("step") != 512
        or done.get("horizon_reached") is not True
        or done.get("resumed") is not False
        or done.get("timing_scope") != "uninterrupted"
    ):
        raise ValueError("analysis requires an uninterrupted completed512-update run")
    if [entry["step"] for entry in done["history"]] != list(STEPS):
        raise ValueError("native full-development cadence differs")
    rows = timing["checkpoint_publications"]
    expected = ([0] if done["target"] is not None and done["target"]["step"] == 0 else []) + list(STEPS[1:])
    if [row["step"] for row in rows] != expected:
        raise ValueError("verified publication steps are missing, duplicated or unplanned")
    seconds = [row["seconds"] for row in rows]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in seconds) or seconds != sorted(
        set(seconds)
    ):
        raise ValueError("publication times must be finite, positive and strictly increasing")
    for row in rows:
        if Path(row["checkpoint"]).name != f"iter_{row['step']:06d}":
            raise ValueError("publication checkpoint does not match its update")
    return rows


def equal_compute_endpoints(candidate, reference):
    left, right = publications(candidate), publications(reference)
    budget = min(left[-1]["seconds"], right[-1]["seconds"])
    result = {
        "budget_seconds": budget,
        "rule": "earlier final NF4/BF16 checkpoint publication; latest published endpoint at or before budget, never quality-selected",
    }
    for name, rows in (("nf4_lora", left), ("bf16_lora", right)):
        eligible = [row for row in rows if row["seconds"] <= budget]
        if not eligible:
            raise ValueError("an arm has no published evaluation within the fixed equal-compute budget")
        chosen = eligible[-1]
        result[name] = {**chosen, "publication_lag_seconds": budget - chosen["seconds"]}
    return result


def validate_training_contract(identity, *, mode, seed, target, admitted):
    """Validate training before separating it from frozen-base calibration identity."""
    contract = identity.get("training_contract", {})
    expected = {
        "format": "cosmos-native-action-sft-v1",
        "seed": seed,
        "max_steps": 512,
        "evaluation_every": 128,
        "evaluation_seed": 123,
        "evaluation_max_samples": 0,
        "target": [ADE, target["threshold"], "min", False],
        "validation": target["source"],
        "lr_schedule": {"kind": "cosine", "f_start": 0.01, "f_min": 0.01, "cycle_steps": 512},
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint training contract differs from the frozen512-update protocol")
    optimizer = contract.get("optimizer", {})
    controls = {
        "name": "adamw",
        "impl": "fused",
        "decay_groups": "",
        "lr": 1e-4,
        "warmup_steps": 100,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "train_keys": [],
        "lr_multipliers": {},
    }
    if any(optimizer.get(key) != value for key, value in controls.items()):
        raise ValueError("checkpoint optimizer differs from the frozen matched control")
    objective = contract.get("objective", {})
    if any(
        objective.get(key) != value
        for key, value in {
            "algorithm": "flow_bc",
            "action_dim": 2,
            "training_prefix": 64,
            "micro_batch_size": 1,
            "training_source": admitted["native_sources"]["train"],
        }.items()
    ):
        raise ValueError("checkpoint native objective or training source differs")
    if objective.get("task") != identity.get("processor_contract"):
        raise ValueError("checkpoint objective differs from its physical trajectory/processor contract")
    if identity.get("format") != "cosmos.family-backbone.v1" or identity.get("adapter") != {
        "mode": mode,
        "rank": 32,
        "alpha": 64.0,
        "dropout": 0.0,
        "seed": seed,
    }:
        raise ValueError("checkpoint adaptation settings differ")
    expected_quantization = (
        {
            "backend": "bitsandbytes",
            "backend_version": "0.49.2",
            "format": "nf4",
            "block_size": 64,
            "nested": True,
            "storage_dtype": "uint8",
            "compute_dtype": "bfloat16",
        }
        if mode == "nf4_lora"
        else None
    )
    if identity.get("quantization") != expected_quantization:
        raise ValueError("checkpoint quantization recipe differs from the frozen comparison")
    for key in (
        "base",
        "processor_contract",
        "task_contract",
        "precision",
        "dependencies",
        "torch_version",
        "implementation",
        "targets",
        "auxiliary_parameters",
    ):
        if identity.get(key) != target["identity"].get(key):
            raise ValueError(f"trained policy differs from frozen calibration contract: {key}")
    factories = contract.get("factories", {})
    if factories != target["scorer"]:
        raise ValueError("training/scoring factory source identity differs from frozen calibration")
    runtime = contract.get("runtime", {})
    if runtime != {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
        "optimizer_state_dtype": "float32",
    }:
        raise ValueError("training numerical runtime contract differs")
    return contract


def load_target(path, groups):
    target, digest = read_json(path)
    expected = {
        "format": "native-family-frozen-base-target-v1",
        "family": "alpamayo",
        "metric": ADE,
        "factor": 0.95,
        "direction": "min",
        "seed": 17,
        "evaluation_seed": 123,
        "evaluation_entrypoint_sha256": CALIBRATION_EVALUATOR_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "cohort_manifest_sha256": MANIFEST_SHA,
        "build": {"implementation_sha": CODE, "wheel_sha256": WHEEL},
    }
    if any(target.get(key) != value for key, value in expected.items()):
        raise ValueError("target is not the frozen Alpamayo baseline protocol")
    baseline = target["baseline"]
    if (
        type(baseline) not in (int, float)
        or not math.isfinite(baseline)
        or baseline <= 0
        or target["threshold"] != 0.95 * baseline
    ):
        raise ValueError("target must remain exactly95 percent of positive frozen-base ADE")
    reference, reference_sha = read_json(target["reference_evaluation"])
    if (
        reference_sha != target["reference_evaluation_sha256"]
        or reference["metrics"][ADE] != baseline
        or reference.get("artifact_kind") != "fresh-frozen-base"
        or reference["identity"]["adapter"]["mode"] != "bf16_frozen"
    ):
        raise ValueError("target calibration evidence changed")
    if any(reference[key] != target[key] for key in ("identity", "source", "scorer")):
        raise ValueError("target calibration identities differ")
    for metric in GUARDS:
        values = block_means(reference["rows"], groups, metric)
        if not math.isclose(
            sum(values.values()) / len(values), reference["metrics"][metric], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("frozen baseline aggregate differs from admitted trajectory rows")
    calibration, _ = read_json(Path(path).parent / "result.json")
    if (
        calibration.get("complete") is not True
        or calibration.get("mode") != "bf16_frozen"
        or calibration.get("family") != "alpamayo"
        or calibration.get("target_sha256") != digest
        or calibration.get("scored_rows") != len(groups)
    ):
        raise ValueError("frozen target requires completed calibration acceptance")
    workflow, _ = read_json(Path(path).parent / "workflow.json")
    if (
        workflow.get("monitor_was_alive_at_end") is not True
        or workflow.get("monitor_exit") != 0
        or type(workflow.get("elapsed_seconds")) not in (int, float)
        or not math.isfinite(workflow["elapsed_seconds"])
        or workflow["elapsed_seconds"] <= 0
    ):
        raise ValueError("frozen target requires completed healthy calibration workflow")
    return target, digest, reference


def load_run(path, *, target, target_sha, admitted):
    run, digest = read_json(path)
    key = run["mode"], run["seed"]
    if (
        key not in {(m, s) for m in MODES for s in SEEDS}
        or run.get("complete") is not True
        or run.get("target_sha256") != target_sha
    ):
        raise ValueError("development result is incomplete or not a planned arm/seed/target")
    protocol, protocol_sha = read_json(Path(path).parent / "protocol.json")
    if (
        protocol.get("protocol_sha256") != PROTOCOL_SHA
        or protocol.get("mode") != key[0]
        or protocol.get("seed") != key[1]
        or protocol.get("implementation") != CODE
        or protocol.get("wheel_sha256") != WHEEL
    ):
        raise ValueError("development execution receipt differs from the sealed protocol/build")
    evidence = run.get("memory_evidence", {})
    if any(
        type(evidence.get(k)) is not int or evidence[k] <= 0
        for k in ("samples", "complete_host_samples", "positive_owned_gpu_observations")
    ):
        raise ValueError("combined comparison requires valid attributed GPU/host memory evidence")
    workflow, workflow_sha = read_json(Path(path).parent / "workflow.json")
    if workflow.get("monitor_alive_at_end") is not True or workflow.get("monitor_exit") != 0:
        raise ValueError("development monitor did not stay healthy through completion")
    if (
        type(workflow.get("elapsed_seconds")) not in (int, float)
        or not math.isfinite(workflow["elapsed_seconds"])
        or workflow["elapsed_seconds"] <= 0
    ):
        raise ValueError("complete workflow clock is missing or invalid")
    rows = publications(run)
    if (
        run["completion"].get("validation_source") != target["source"]
        or run["completion"].get("training_source") != admitted["native_sources"]["train"]
    ):
        raise ValueError("native completion source identities differ from admitted splits")
    history = run["completion"]["history"]
    matches = [r for r in history if r["metrics"][ADE] <= target["threshold"]]
    expected = matches[0] if matches else None
    if run["completion"]["target"] is not None:
        # Native target adds checkpoint/timing to the selected history item.
        actual = run["completion"]["target"]
        if expected is None or actual.get("step") != expected["step"] or actual.get("timing_scope") != "uninterrupted":
            raise ValueError("recorded target is not the first fixed-cadence crossing")
    elif expected is not None:
        raise ValueError("a reached target was incorrectly recorded as unreached")
    seconds = None if expected is None else next(row["seconds"] for row in rows if row["step"] == expected["step"])
    if (
        run["timing"]["unreached"] is not (expected is None)
        or run["timing"]["first_target_publication_seconds"] != seconds
    ):
        raise ValueError("target publication clock or censoring differs")
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata

    metadata = read_adapter_metadata(rows[-1]["checkpoint"], require_resume=True)
    contract = validate_training_contract(
        metadata["identity"], mode=key[0], seed=key[1], target=target, admitted=admitted
    )
    return run, {
        "result_sha256": digest,
        "execution_protocol_sha256": protocol_sha,
        "workflow_sha256": workflow_sha,
        "complete_workflow_seconds": workflow["elapsed_seconds"],
        "contract": contract,
        "metadata": metadata,
    }


def analyze(paths, *, validation, manifest, target_path):
    groups, admitted = load_groups(validation, manifest)
    target, target_sha, baseline = load_target(target_path, groups)
    if target["source"] != admitted["native_sources"]["validation"]:
        raise ValueError("frozen calibration source does not match admitted media-content identity")
    runs, receipts = {}, {}
    for path in paths:
        run, receipt = load_run(path, target=target, target_sha=target_sha, admitted=admitted)
        key = run["mode"], run["seed"]
        if key in runs:
            raise ValueError("duplicate development arm/seed")
        runs[key], receipts[key] = run, receipt
    if set(runs) != {(m, s) for m in MODES for s in SEEDS}:
        raise ValueError("analysis requires all nine planned arm/seed runs; unreached targets stay included")
    common = []
    for receipt in receipts.values():
        contract = copy.deepcopy(receipt["contract"])
        contract.pop("seed")
        common.append(contract)
    if any(contract != common[0] for contract in common[1:]):
        raise ValueError("paired seeds/arms differ beyond their declared training seed")
    score_receipts = []

    def scores(key, step):
        run = runs[key]
        publication = next(item for item in publications(run) if item["step"] == step)
        checkpoint = Path(publication["checkpoint"])
        from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata

        metadata = read_adapter_metadata(str(checkpoint), require_resume=True)
        if metadata != receipts[key]["metadata"]:
            raise ValueError("selected checkpoint identity differs from its validated final recipe")
        path = checkpoint.parent.parent / f"evaluation-{step:06d}.json"
        value, digest = read_json(path)
        history = next(item for item in run["completion"]["history"] if item["step"] == step)
        if value["step"] != step or value["metrics"] != history["metrics"]:
            raise ValueError("selected native evaluation differs from recorded history")
        result = {metric: block_means(value["rows"], groups, metric) for metric in GUARDS}
        for metric in GUARDS:
            if not math.isclose(
                sum(result[metric].values()) / len(result[metric]),
                value["metrics"][metric],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("native aggregate differs from complete balanced clip/block reductions")
        score_receipts.append(
            {"mode": key[0], "seed": key[1], "step": step, "score_sha256": digest, "checkpoint": str(checkpoint)}
        )
        return result

    fixed = {m: {metric: {} for metric in GUARDS} for m in MODES}
    elapsed = {m: {metric: {} for metric in GUARDS} for m in MODES[:2]}
    clocks = []
    for seed in SEEDS:
        for mode in MODES:
            for metric, values in scores((mode, seed), 512).items():
                fixed[mode][metric][seed] = values
        selected = equal_compute_endpoints(runs["nf4_lora", seed], runs["bf16_lora", seed])
        clocks.append({"seed": seed, **selected})
        for mode in MODES[:2]:
            for metric, values in scores((mode, seed), selected[mode]["step"]).items():
                elapsed[mode][metric][seed] = values
    calibration_cost, calibration_cost_sha = read_json(Path(target_path).parent / "workflow.json")
    paired_calibration_path = Path(target_path).parent.parent / "paired-workflow.json"
    paired_calibration = read_json(paired_calibration_path)[0] if paired_calibration_path.is_file() else None
    fixed_result = comparison(fixed["nf4_lora"], fixed["bf16_lora"])
    elapsed_result = comparison(elapsed["nf4_lora"], elapsed["bf16_lora"])
    return {
        "format": "cosmos-native-alpamayo-development-analysis-v1",
        "scope": "exposed development with unverified coarse recording blocks",
        "protocol_sha256": PROTOCOL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "target_sha256": target_sha,
        "input_receipts": [
            {"mode": m, "seed": s, **{k: v for k, v in receipt.items() if k not in ("contract", "metadata")}}
            for (m, s), receipt in receipts.items()
        ],
        "score_receipts": score_receipts,
        "fixed512_nf4_vs_bf16_lora": fixed_result,
        "equal_elapsed_nf4_vs_bf16_lora": elapsed_result,
        "both_development_endpoints_pass_guards": fixed_result["joint_development_preservation_guards_met"]
        and elapsed_result["joint_development_preservation_guards_met"],
        "both_development_endpoints_improve_ADE": fixed_result["development_ADE_improvement_upper_bound_below_zero"]
        and elapsed_result["development_ADE_improvement_upper_bound_below_zero"],
        "fixed512_nf4_vs_matched_expert_only": {
            "scope": "descriptive paired comparison; primary preservation margins apply to matched BF16 LoRA",
            **{metric: compare_recipes(fixed["nf4_lora"][metric], fixed["bf16_frozen"][metric]) for metric in GUARDS},
        },
        "equal_compute_publications_and_lags": clocks,
        "paired_time_to_target": [
            {
                "seed": seed,
                "nf4_seconds": runs["nf4_lora", seed]["timing"]["first_target_publication_seconds"],
                "bf16_seconds": runs["bf16_lora", seed]["timing"]["first_target_publication_seconds"],
                "nf4_unreached": runs["nf4_lora", seed]["timing"]["unreached"],
                "bf16_unreached": runs["bf16_lora", seed]["timing"]["unreached"],
                "nf4_censor_seconds": publications(runs["nf4_lora", seed])[-1]["seconds"],
                "bf16_censor_seconds": publications(runs["bf16_lora", seed])[-1]["seconds"],
                "scope": "per-seed observed/censored times; no attainer-only pooled mean",
            }
            for seed in SEEDS
        ],
        "individual_target_runtime_memory": [
            {
                "mode": m,
                "seed": s,
                "target": r["completion"]["target"],
                "timing": r["timing"],
                "clocks": r["completion"]["clocks"],
                "native_workflow_seconds": r["completion"]["workflow_seconds"],
                "complete_workflow_seconds": receipts[m, s]["complete_workflow_seconds"],
                "memory_evidence": r["memory_evidence"],
            }
            for (m, s), r in sorted(runs.items())
        ],
        "time_to_target_rule": "retain all nine runs, including censored/unreached; do not average only attainers or interpolate; calibration cost is separate",
        "calibration": {
            "target": target,
            "useful_adaptation_target": "0.95 times frozen BF16 ADE; crossing alone is not preservation",
            "reference_metrics": baseline["metrics"],
            "BF16_calibration_cost": calibration_cost,
            "calibration_cost_sha256": calibration_cost_sha,
            "paired_baseline_diagnostic_cost": paired_calibration,
            "search_cost": "no hyperparameter search in this fixed protocol",
        },
        "independent_confirmation": {
            "established": False,
            "reason": "blocks are platform/country/month without recording/session/event provenance; three seeds and exploratory coarse-block intervals cannot establish a population recipe claim",
        },
        "closed_loop_driving_and_inference_latency": {
            "established": False,
            "reason": "offline planar trajectory errors and training clocks do not measure either capability",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="All nine native Alpamayo result.json paths")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json_new(
        Path(args.output),
        analyze(args.run, validation=args.validation, manifest=args.manifest, target_path=args.target),
    )


if __name__ == "__main__":
    main()
