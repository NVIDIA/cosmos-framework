# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate native experiment timing and memory evidence without importing Torch.

Ratios are NF4 / BF16. Timing windows use sums of actual examples/tokens divided
by elapsed update seconds. Stage wall times and nested native phase intervals
are separate views, never additive. External memory peaks are sampled lower
bounds, including when their observed difference exceeds the memory criterion.

CLI: --run RUN (repeatable), --pair-plan PLAN.json (optional), --output NEW.json.
Controlled pair plans retain cosmos.qlora-performance-pairs.v1. Best-fit plans
require v2, --pair-plan-sha256 and selection bindings for confirmation_plan,
development_plan, development_selection and execution_selection, each {path,sha256}.
The execution-selection receipt pins all six confirmation_specs by {path,sha256}.
Pairs are [{repeat: 1, nf4: {path, contract_sha256}, bf16: {...}}, ...].
Paths are private inputs and are never copied into the output. Exactly three
complete pairs are required for inferential summaries; other runs stay visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.native_qlora_qualification.common import digest_json, write_json_new
from tools.native_qlora_qualification.monitor_host import validate_sample
from tools.native_qlora_qualification.protocol import Experiment

GIB = 1024**3
PROTOCOL = "cosmos.native-qlora-performance.v1"
STAGES = ("preflight", "training", "portable_publication", "probe_dcp", "probe_portable", "reload_exact", "scoring")
WINDOWS = (("updates_1_300", 1, 300), ("updates_6_30", 6, 30), ("updates_31_300", 31, 300))
BEST_FIT_FIELDS = frozenset({"learning_rate", "rank", "alpha", "microbatch", "activation_checkpointing"})
CONFIRMATION_SEEDS = (17, 42, 123)


class InvalidEvidence(ValueError):
    """Fixed, public-safe explanation of an evidence-contract violation."""


@dataclass(frozen=True)
class BestFitSelection:
    """Externally pinned decisions; this analyzer does not rerun candidate ranking."""

    specifications: dict[tuple[str, int], dict]
    specification_sha256: dict[tuple[str, int], str]
    source_sha256: dict[str, str]
    data_sha256: str
    bound_files: tuple[tuple[str, str], ...]

    def assert_stable(self) -> None:
        for path, sha in self.bound_files:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise InvalidEvidence("sealed selection evidence changed during analysis")


def read_selection_binding(binding: dict) -> tuple[dict, str]:
    if (
        set(binding) != {"path", "sha256"}
        or not isinstance(binding["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])
    ):
        raise InvalidEvidence("selection inputs require exact paths and SHA256 digests")
    raw = Path(binding["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise InvalidEvidence("sealed selection input digest differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise InvalidEvidence("selection input must be a JSON object")
    return value, binding["sha256"]


def load_best_fit_selection(bindings: dict) -> BestFitSelection:
    """Bind selected five-field changes, unchanged data and all six exact specs.

    execution_selection uses cosmos.qlora-execution-selection.v1, complete=true,
    confirmation_plan_sha256, development_selection_sha256, runtime_matrix_plan_sha256s,
    selected[arm]={activation_checkpointing,microbatch,runtime_completion_sha256,
    lifecycle_completion_sha256}, evidence_sha256 and confirmation_specs=[{path,sha256}].
    Its producer must audit the complete predeclared calibration grid and recovery
    proofs. Here their pinned decision and evidence hashes bind the measured runs;
    they do not authorize untouched scoring or replace model/checkpoint validation.
    """
    expected = {"confirmation_plan", "development_plan", "development_selection", "execution_selection"}
    if set(bindings) != expected:
        raise InvalidEvidence("best-fit comparison requires both selection receipts and their sealed plans")
    loaded = {key: read_selection_binding(value) for key, value in bindings.items()}
    sources = {key: value[1] for key, value in loaded.items()}
    confirmation, development, decision, execution = (
        loaded[key][0]
        for key in ("confirmation_plan", "development_plan", "development_selection", "execution_selection")
    )
    if (
        confirmation.get("protocol") != "cosmos.qlora-confirmation-and-best-fit-plan.v2"
        or confirmation.get("development_plan_sha256") != sources["development_plan"]
        or confirmation.get("paired_seeds") != list(CONFIRMATION_SEEDS)
        or development.get("protocol") != "cosmos.qlora-development-plan.v1"
        or decision.get("protocol") != "cosmos.qlora-development-selection.v1"
        or decision.get("complete") is not True
        or decision.get("development_plan_sha256") != sources["development_plan"]
        or decision.get("development_manifest_sha256") != development.get("development_manifest_sha256")
        or decision.get("confirmation_seeds_required") != list(CONFIRMATION_SEEDS)
        or decision.get("fixed_release_gates") != development.get("fixed_gates")
        or development.get("fixed_gates", {}).get("relative_supervised_token_nll_one_sided_95_upper_max") != 0.02
        or development.get("fixed_gates", {}).get("accuracy_loss_one_sided_95_upper_max") != 0.01
        or development.get("fixed_gates", {}).get("paired_training_seeds") != list(CONFIRMATION_SEEDS)
        or decision.get("development_rows") != 584
        or decision.get("development_groups") != 12
        or execution.get("protocol") != "cosmos.qlora-execution-selection.v1"
        or execution.get("complete") is not True
        or execution.get("confirmation_plan_sha256") != sources["confirmation_plan"]
        or execution.get("development_selection_sha256") != sources["development_selection"]
    ):
        raise InvalidEvidence("best-fit decisions are incomplete or differ from their sealed confirmation scope")
    matrix_hashes = [row["sha256"] for row in confirmation["runtime_matrix_plans"]]
    rule = confirmation["best_fit"]["execution_selection_contract"]
    if (
        len(matrix_hashes) != 2
        or execution.get("runtime_matrix_plan_sha256s") != matrix_hashes
        or rule.get("activation_checkpointing") != ["full", "none"]
        or rule.get("microbatches") != [1, 2, 4]
        or rule.get("fixed_model_execution")
        != {"adapter_compute_dtype": "float32", "logits_mode": "full", "compile_language_model": False}
        or rule.get("timed_updates_inclusive") != [3, 20]
        or confirmation["best_fit"].get("steps") != 300
        or confirmation["best_fit"].get("warmup_steps") != 30
        or confirmation["best_fit"].get("effective_batch") != 8
    ):
        raise InvalidEvidence("best-fit execution selection scope differs")
    selected = {}
    evidence_hashes = execution.get("evidence_sha256", [])
    if (
        not isinstance(evidence_hashes, list)
        or not evidence_hashes
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in evidence_hashes)
    ):
        raise InvalidEvidence("execution selection lacks pinned calibration and recovery evidence")
    for arm in ("nf4", "bf16"):
        winner = decision["selected"][arm]
        records = [
            row for row in decision["candidates"] if row["arm"] == arm and row["candidate"] == winner["candidate"]
        ]
        if (
            decision["arm_status"].get(arm) != "development_candidate_selected"
            or len(records) != 1
            or records[0].get("state") != "completed"
            or records[0].get("eligible") is not True
            or winner.get("evidence_sha256") != digest_json(records[0])
            or winner.get("contract_sha256") != records[0].get("contract_sha256")
            or winner.get("hyperparameters") != records[0].get("hyperparameters")
            or set(winner["hyperparameters"]) != {"learning_rate", "rank", "alpha"}
        ):
            raise InvalidEvidence("development winner is not bound to its eligible candidate evidence")
        choice = execution["selected"][arm]
        if (
            choice.get("activation_checkpointing") not in ("full", "none")
            or type(choice.get("microbatch")) is not int
            or choice["microbatch"] not in (1, 2, 4)
            or any(
                choice.get(key) not in evidence_hashes
                for key in ("runtime_completion_sha256", "lifecycle_completion_sha256")
            )
        ):
            raise InvalidEvidence("execution winner lacks an eligible configuration and recovery identity")
        selected[arm] = {
            **winner["hyperparameters"],
            **{key: choice[key] for key in ("activation_checkpointing", "microbatch")},
        }
    specifications, spec_hashes = {}, {}
    for binding in execution["confirmation_specs"]:
        spec, sha = read_selection_binding(binding)
        actual = Experiment(**spec)
        key = actual.arm, actual.seed
        if key in specifications or actual.seed not in CONFIRMATION_SEEDS:
            raise InvalidEvidence("confirmation specifications must contain each arm and seed exactly once")
        expected_spec = Experiment(actual.name, actual.arm, seed=actual.seed, **selected[actual.arm]).as_dict()
        if spec != expected_spec:
            raise InvalidEvidence("confirmation specification changed an unselected or fixed training setting")
        specifications[key] = {field: value for field, value in spec.items() if field != "name"}
        spec_hashes[key] = sha
    if set(specifications) != {(arm, seed) for arm in ("nf4", "bf16") for seed in CONFIRMATION_SEEDS}:
        raise InvalidEvidence("best-fit comparison requires all six locked confirmation specifications")
    data_hash = digest_json(
        [
            development[key]
            for key in (
                "development_manifest_sha256",
                "development_data_sha256",
                "fresh_training_data_sha256",
                "development_data_sha256",
            )
        ]
    )
    bound_files = tuple(
        (binding["path"], binding["sha256"]) for binding in [*bindings.values(), *execution["confirmation_specs"]]
    )
    selection = BestFitSelection(specifications, spec_hashes, sources, data_hash, bound_files)
    selection.assert_stable()
    return selection


def number(value: Any, *, positive: bool = False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "sum": math.fsum(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": quantile(values, 0.95) if values else None,
        "maximum": max(values) if values else None,
    }


def metric_summary(records: list[dict], *, steps: int, effective_batch: int) -> dict:
    """Reject missing, reordered, skipped, nonfinite, or inconsistent updates."""
    updates = [r for r in records if "timer/e2e_step_s" in r or "train/loss_avg" in r]
    if any(type(r.get("step")) is not int for r in updates) or [r["step"] for r in updates] != list(
        range(1, steps + 1)
    ):
        raise InvalidEvidence("native updates must be exact consecutive integer steps")
    projected = []
    for row in updates:
        for key in (
            "timer/e2e_step_s",
            "train/loss_avg",
            "train/grad_norm",
            "train/valid_token_count",
            "train/valid_sequence_count",
        ):
            if not number(
                row.get(key),
                positive=key in ("timer/e2e_step_s", "train/valid_token_count", "train/valid_sequence_count"),
            ):
                raise InvalidEvidence("missing/nonfinite native update metric")
        if row["train/valid_sequence_count"] != effective_batch:
            raise InvalidEvidence("native effective example count differs from the experiment")
        if row["train/valid_token_count"] != int(row["train/valid_token_count"]):
            raise InvalidEvidence("native supervised-token count must be integral")
        if row.get("optim/update_outcome") != "applied":
            raise InvalidEvidence("native optimizer outcome must be applied")
        for key in ("optim/optimizer_step", "optim/attempt_id"):
            if not number(row.get(key)) or row[key] != row["step"]:
                raise InvalidEvidence("native optimizer and attempt clocks differ")
        for key in (
            "optim/step_skipped",
            "optim/nonfinite_skips_total",
            "optim/nonfinite_loss",
            "optim/nonfinite_gradient",
        ):
            if type(row.get(key)) not in (int, float) or row[key] != 0:
                raise InvalidEvidence("native run contains missing skip controls or rejected updates")
        if row.get("optim/zero_grad_skips_total", 0) != 0:
            raise InvalidEvidence("native run contains zero-gradient skips")
        projected.append(
            {
                "step": row["step"],
                "seconds": row["timer/e2e_step_s"],
                "examples": int(row["train/valid_sequence_count"]),
                "supervised_tokens": int(row["train/valid_token_count"]),
                "supervised_ce": row["train/loss_avg"],
            }
        )
    windows = {}
    for label, first, last in WINDOWS:
        selected = [r for r in projected if first <= r["step"] <= last]
        if last > steps:
            windows[label] = {"complete": False}
            continue
        elapsed = math.fsum(r["seconds"] for r in selected)
        examples = sum(r["examples"] for r in selected)
        tokens = sum(r["supervised_tokens"] for r in selected)
        windows[label] = {
            "complete": True,
            "first_update": first,
            "last_update": last,
            "update_seconds": distribution([r["seconds"] for r in selected]),
            "examples": examples,
            "supervised_tokens": tokens,
            "examples_per_second": examples / elapsed,
            "supervised_tokens_per_second": tokens / elapsed,
        }
    validation = []
    for row in records:
        if "val/loss_avg" in row:
            if type(row.get("step")) is not int or not 0 <= row["step"] <= steps:
                raise InvalidEvidence("invalid validation step")
            for key in ("timer/validation_s", "val/loss_avg", "val/valid_token_count", "val/valid_sequence_count"):
                if not number(row.get(key), positive=key in ("val/valid_token_count", "val/valid_sequence_count")):
                    raise InvalidEvidence("invalid validation metric")
            validation.append(
                {
                    "step": row["step"],
                    "seconds": row["timer/validation_s"],
                    "supervised_ce": row["val/loss_avg"],
                    "supervised_tokens": row["val/valid_token_count"],
                    "examples": row["val/valid_sequence_count"],
                }
            )
    if len({r["step"] for r in validation}) != len(validation):
        raise InvalidEvidence("duplicate validation frame")
    pipeline = {
        k.removeprefix("train/"): v
        for k, v in (updates[-1] if updates else {}).items()
        if k.startswith("train/preprocessing_") and number(v)
    }
    return {
        "updates": projected,
        "windows": windows,
        "validation": validation,
        "initial_validation_seconds": math.fsum(r["seconds"] for r in validation if r["step"] == 0),
        "scheduled_validation_seconds": math.fsum(r["seconds"] for r in validation if r["step"] > 0),
        "preprocessing_last_cumulative_counters": pipeline,
        "preprocessing_counter_scope": "Cumulative counters can include initial/in-loop validation. Processing and waiting overlap execution; never sum them into wall time.",
        "no_skips_evidence": "Explicit per-update applied outcomes, zero rejection counters and matching optimizer/attempt clocks; checkpoint controls audited separately.",
    }


def phase_name(before: str, after: str, *, phase_family: str = "reasoner") -> str:
    if phase_family == "native_action":
        # Adjacent reset-delimited intervals, not enclosing/nested clock totals.
        # A training block can contain many updates, ingestion and logging.
        return {
            ("data_preparation_begin", "data_preparation_complete"): "data_preparation",
            ("data_preparation_complete", "model_loading_begin"): "action_model_setup",
            ("model_loading_begin", "model_loading_complete"): "loading",
            ("model_loading_complete", "evaluation_begin"): "action_trainer_setup_or_resume",
            ("evaluation_complete", "evaluation_begin"): "action_training_block",
            ("checkpoint_complete", "evaluation_begin"): "action_training_block",
            ("checkpoint_complete", "checkpoint_begin"): "action_training_block",
            ("evaluation_begin", "evaluation_complete"): "action_evaluation",
            ("evaluation_complete", "checkpoint_begin"): "action_checkpoint_setup",
            ("checkpoint_begin", "checkpoint_begin"): "action_checkpoint_setup",
            ("checkpoint_begin", "checkpoint_complete"): "checkpoint_write",
            ("checkpoint_complete", "checkpoint_complete"): "action_checkpoint_verification",
            ("checkpoint_complete", "portable_export_begin"): "action_final_integrity",
            ("portable_export_begin", "portable_export_complete"): "portable_export",
            ("family_evaluation_loading_begin", "family_evaluation_loading_complete"): "loading",
            ("family_evaluation_loading_complete", "family_evaluation_begin"): "scoring_setup",
            ("family_evaluation_begin", "family_evaluation_complete"): "action_evaluation",
            ("family_evaluation_complete", "family_evaluation_integrity_complete"): "action_final_integrity",
            (
                "family_evaluation_integrity_complete",
                "family_evaluation_publication_complete",
            ): "evaluation_publication",
            ("family_dense_loading_begin", "family_dense_loading_complete"): "loading",
            ("family_dense_loading_complete", "family_dense_export_complete"): "dense_export",
            ("family_dense_loading_complete", "family_dense_export_failed"): "dense_export_incomplete",
        }.get((before, after), "unclassified")
    if phase_family != "reasoner":
        raise InvalidEvidence("unknown native memory phase family")
    if (before, after) == ("adapter_loading_start", "adapter_loading_complete"):
        return "loading"
    if (before, after) == ("adapter_loading_complete", "trainer_ready"):
        return "trainer_setup"
    if (before, after) == ("adapter_loading_complete", "scoring_start"):
        return "scoring_setup"
    if (before, after) == ("scoring_start", "scoring_complete"):
        return "scoring"
    if (before, after) == ("scoring_start", "scoring_failed"):
        return "scoring_incomplete"
    if (before, after) == ("trainer_ready", "initial_validation"):
        return "initial_validation"
    if (before, after) == ("checkpoint_begin", "checkpoint_complete"):
        return "checkpoint_write"
    step = re.fullmatch(r"optimizer_step_(\d+)", after)
    if step and before == (
        "initial_validation" if int(step[1]) == 1 else f"validation_checkpoint_boundary_{int(step[1]) - 1}"
    ):
        return "first_update" if int(step[1]) == 1 else "updates"
    boundary = re.fullmatch(r"validation_checkpoint_boundary_(\d+)", after)
    if (
        after == "checkpoint_begin"
        and (
            before in ("initial_validation", "checkpoint_complete")
            or re.fullmatch(r"(?:optimizer_step|validation_checkpoint_boundary)_\d+", before)
        )
    ) or (boundary and before == f"optimizer_step_{boundary[1]}"):
        return "validation_or_boundary_overhead"
    if boundary and before == "checkpoint_complete":
        return "post_checkpoint_boundary"
    return "unclassified"


def sampled_memory(samples: list[dict], *, expected_uuid: str, device_bytes: int) -> dict:
    errors, owned, device, rss, pss, stamps = [], [], [], [], [], []
    capacities = set()
    for row in samples:
        host_receipt_valid = True
        if "host_process_memory_method" in row or "host_collection" in row:
            try:
                validate_sample(row)
            except (ValueError, KeyError, TypeError):
                host_receipt_valid = False
                errors.append("invalid_host_collection_receipt")
        if not number(row.get("unix_seconds")):
            errors.append("invalid_timestamp")
        else:
            stamps.append(row["unix_seconds"])
        if row.get("device_uuid") != expected_uuid:
            errors.append("wrong_device_identity")
        capacity = row.get("device_memory_total_bytes")
        if type(capacity) is not int or not device_bytes <= capacity < 2**63:
            errors.append("wrong_device_capacity")
            capacity = device_bytes
        else:
            capacities.add(capacity)
        usage = row.get("device_memory_used_bytes")
        if type(usage) is int and 0 <= usage <= capacity:
            device.append(usage)
        else:
            errors.append("invalid_device_memory")
        processes = row.get("device_compute_processes")
        valid = isinstance(processes, list)
        total = 0
        for process in processes if valid else []:
            if type(process.get("belongs_to_run")) is not bool:
                valid = False
            if process.get("belongs_to_run") is True:
                value = process.get("used_gpu_memory_bytes")
                if type(value) is not int or not 0 <= value <= capacity:
                    valid = False
                else:
                    total += value
        if valid and total <= capacity:
            owned.append(total)
        else:
            errors.append("invalid_owned_process_memory_or_nvml_sentinel")
        for key, values in (("selected_host_rss_sum_bytes", rss), ("selected_host_pss_sum_bytes", pss)):
            value = row.get(key)
            if host_receipt_valid and "host_process_memory_method" in row:
                peak_key = (
                    "host_observed_attempt_rss_peak_bytes"
                    if values is rss
                    else "host_observed_complete_attempt_pss_peak_bytes"
                )
                value = row.get(peak_key)
            if (
                host_receipt_valid
                and type(value) is int
                and 0 <= value < 2**63
                and (values is not pss or row.get("host_pss_complete") is True)
            ):
                values.append(value)
        if not host_receipt_valid or row.get("host_pss_complete") is not True:
            errors.append("incomplete_host_pss")
    if not samples:
        errors.append("no_samples")
    if len(capacities) > 1:
        errors.append("inconsistent_device_capacity")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        errors.append("nonmonotonic_samples")
    gpu_errors = [error for error in errors if error != "incomplete_host_pss"]
    return {
        "samples": len(samples),
        "nvml_capacity_bytes": next(iter(capacities)) if len(capacities) == 1 else None,
        "owned_device_process_peak_bytes": max(owned, default=None),
        "device_peak_bytes": max(device, default=None),
        "host_rss_sum_peak_bytes": max(rss, default=None),
        "host_pss_sum_peak_bytes": max(pss, default=None),
        "host_pss_complete": bool(samples) and len(pss) == len(samples),
        "sampling_gap_seconds": distribution([b - a for a, b in zip(stamps, stamps[1:]) if b > a]),
        "observed_span_seconds": stamps[-1] - stamps[0] if len(stamps) > 1 else 0,
        "errors": sorted(set(errors)),
        "gpu_memory_evidence_valid": not gpu_errors,
        "peaks_are_sampled_lower_bounds": True,
    }


def memory_summary(records: list[dict], samples: list[dict], runtime: dict, *, phase_family: str = "reasoner") -> dict:
    """Summarize adjacent intervals with an explicit entrypoint's phase vocabulary.

    Host RSS/PSS use only external samples inside each interval. VmHWM is a
    cumulative process high-water mark and is not a phase-local memory peak.
    Native action's enclosing training/evaluation/checkpoint clocks remain
    separate evidence; do not add them to these disjoint phase intervals.
    """
    if phase_family not in ("reasoner", "native_action"):
        raise InvalidEvidence("unknown native memory phase family")
    previous, intervals, by_phase = {}, [], defaultdict(list)
    overall_alloc, overall_reserved = [], []
    allocator_valid = True
    cuda_records = 0
    for row in records:
        cuda = row.get("cuda", {})
        cuda_records += bool(cuda)
        for key, values in (("peak_allocated_bytes", overall_alloc), ("peak_reserved_bytes", overall_reserved)):
            value = cuda.get(key)
            if type(value) is int and 0 <= value <= runtime["device_bytes"]:
                values.append(value)
            elif cuda:
                allocator_valid = False
        key = (row.get("pid"), row.get("rank"), cuda.get("device"))
        identified = all(type(v) is int and v >= 0 for v in key)
        prior = previous.get(key) if identified else None
        if prior and prior.get("reset_peak_after_record") is True:
            start, end = prior.get("unix_seconds"), row.get("unix_seconds")
            mono_a, mono_b = prior.get("monotonic_seconds"), row.get("monotonic_seconds")
            if all(number(v) for v in (start, end, mono_a, mono_b)) and end >= start and mono_b >= mono_a:
                phase = phase_name(prior.get("phase", ""), row.get("phase", ""), phase_family=phase_family)
                window = [
                    sample
                    for sample in samples
                    if number(sample.get("unix_seconds")) and start < sample["unix_seconds"] <= end
                ]
                interval = {
                    "phase": phase,
                    "seconds": mono_b - mono_a,
                    "allocator_peak_allocated_bytes": cuda.get("peak_allocated_bytes")
                    if type(cuda.get("peak_allocated_bytes")) is int
                    and 0 <= cuda["peak_allocated_bytes"] <= runtime["device_bytes"]
                    else None,
                    "allocator_peak_reserved_bytes": cuda.get("peak_reserved_bytes")
                    if type(cuda.get("peak_reserved_bytes")) is int
                    and 0 <= cuda["peak_reserved_bytes"] <= runtime["device_bytes"]
                    else None,
                    "sampled": sampled_memory(
                        window, expected_uuid=runtime["uuid"], device_bytes=runtime["device_bytes"]
                    ),
                }
                intervals.append(interval)
                by_phase[phase].append(interval)
        if identified:
            previous[key] = row
    phases = {}
    for name, rows in by_phase.items():
        phases[name] = {
            "interval_seconds": distribution([r["seconds"] for r in rows]),
            "allocator_peak_allocated_bytes": max(
                (r["allocator_peak_allocated_bytes"] for r in rows if r["allocator_peak_allocated_bytes"] is not None),
                default=None,
            ),
            "allocator_peak_reserved_bytes": max(
                (r["allocator_peak_reserved_bytes"] for r in rows if r["allocator_peak_reserved_bytes"] is not None),
                default=None,
            ),
            "sampled_gpu_evidence_complete": all(r["sampled"]["gpu_memory_evidence_valid"] for r in rows),
            "sampled_errors": sorted({error for r in rows for error in r["sampled"]["errors"]}),
            "sampled_owned_device_process_peak_bytes": max(
                (
                    r["sampled"]["owned_device_process_peak_bytes"]
                    for r in rows
                    if r["sampled"]["owned_device_process_peak_bytes"] is not None
                    and r["sampled"]["gpu_memory_evidence_valid"]
                ),
                default=None,
            ),
        }
        for statistic in ("host_rss_sum_peak_bytes", "host_pss_sum_peak_bytes"):
            phases[name][f"sampled_{statistic}"] = max(
                (r["sampled"][statistic] for r in rows if r["sampled"][statistic] is not None), default=None
            )
        phases[name]["sampled_host_pss_complete"] = all(r["sampled"]["host_pss_complete"] for r in rows)
    return {
        "phase_family": phase_family,
        "allocator_evidence_complete": allocator_valid and bool(cuda_records),
        "allocator_peak_allocated_bytes": max(overall_alloc, default=None),
        "allocator_peak_reserved_bytes": max(overall_reserved, default=None),
        "sampled": sampled_memory(samples, expected_uuid=runtime["uuid"], device_bytes=runtime["device_bytes"]),
        "phases": phases,
        "intervals": intervals,
        "phase_semantics": (
            "Adjacent reset-delimited native action intervals; training blocks include multiple updates, ingestion and logging. Nested checkpoint setup/write/verification intervals are disjoint, not additional enclosing checkpoint clocks. Loading includes factory work after model_loading_begin, but excludes earlier process startup. Host RSS/PSS are interval samples, not cumulative VmHWM."
            if phase_family == "native_action"
            else "Peaks end at the current boundary after the preceding reset. checkpoint_begin records the preceding validation/boundary window; only checkpoint_complete after checkpoint_begin is checkpoint writing. Loading excludes earlier source verification and process startup."
        ),
    }


class Evidence:
    def __init__(self, root: Path):
        self.root, self.sources = root, {}

    def read(self, path: Path, *, lines: bool = False) -> Any:
        raw = path.read_bytes()
        self.sources[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        if lines:
            if any(not line.strip() for line in raw.splitlines()):
                raise InvalidEvidence("blank evidence JSONL line")
            return [json.loads(line) for line in raw.splitlines()]
        return json.loads(raw)


def artifact_size(path: Path) -> dict:
    if not path.is_dir() or path.is_symlink():
        raise InvalidEvidence("artifact directory missing or symlinked")
    files = [p for p in path.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in files):
        raise InvalidEvidence("artifact symlink cannot establish owned size")
    return {
        "files": len(files),
        "logical_bytes": sum(p.stat().st_size for p in files),
        "allocated_file_bytes": sum(p.stat().st_blocks * 512 for p in files),
    }


def summarize_run(path: Path) -> dict:
    evidence = Evidence(path)
    report = {
        "protocol": PROTOCOL,
        "completed": False,
        "timing_eligible": False,
        "exclusions": [],
        "release_qualified": False,
    }
    try:
        contract = evidence.read(path / "contract.json")
        completion = evidence.read(path / "completion.json")
        runtime = evidence.read(path / "runtime.json")
        spec = contract["experiment"]
        Experiment(**spec)
        report.update(
            case_id=digest_json(contract),
            experiment={k: v for k, v in spec.items() if k != "name"},
            complete_elapsed_seconds=completion["complete_elapsed_seconds"],
        )
        if (
            completion.get("completed") is not True
            or completion.get("failure") is not None
            or completion["experiment"] != spec
        ):
            raise InvalidEvidence("experiment completion failed or differs")
        if contract.get("action") != "train-eval" or spec["steps"] != 300:
            raise InvalidEvidence("controlled report requires a complete 300-update train-eval experiment")
        if not number(completion.get("complete_elapsed_seconds"), positive=True):
            raise InvalidEvidence("invalid complete elapsed time")
        stages = {}
        for name in STAGES:
            outcome = evidence.read(path / "stages" / name / "outcome.json")
            if (
                outcome.get("name") != name
                or outcome.get("exit_code") != 0
                or not number(outcome.get("elapsed_seconds"), positive=True)
            ):
                raise InvalidEvidence("missing or failed experiment stage")
            stages[name] = outcome["elapsed_seconds"]
        if completion["outcomes"] != [evidence.read(path / "stages" / name / "outcome.json") for name in STAGES]:
            raise InvalidEvidence("completed outcome sequence differs from stages")
        metric_paths = list((path / "native").rglob("metrics.jsonl"))
        if len(metric_paths) != 1:
            raise InvalidEvidence("ambiguous native metrics source")
        records = evidence.read(metric_paths[0], lines=True)
        report["native"] = metric_summary(records, steps=300, effective_batch=spec["effective_batch"])
        if len(contract.get("files", [])) < 6:
            raise InvalidEvidence("run contract lacks exact data and split identities")
        expected_validation = [0, *range(spec["validation_every"], 301, spec["validation_every"])]
        if [row["step"] for row in report["native"]["validation"]] != expected_validation:
            raise InvalidEvidence("native validation schedule differs from the experiment")
        report["stage_seconds"] = stages
        report["stage_seconds_sum"] = math.fsum(stages.values())
        report["orchestrator_and_verification_residual_seconds"] = (
            completion["complete_elapsed_seconds"] - report["stage_seconds_sum"]
        )
        if report["orchestrator_and_verification_residual_seconds"] < 0:
            raise InvalidEvidence("stage costs exceed complete elapsed time")
        samples = evidence.read(path / "nvml-host.jsonl", lines=True)
        report["memory_by_stage"] = {}
        all_phases = []
        for name in STAGES:
            file = path / "stages" / name / "memory.jsonl"
            phases = evidence.read(file, lines=True) if file.exists() else []
            all_phases.extend(phases)
            report["memory_by_stage"][name] = memory_summary(
                phases, [r for r in samples if r.get("stage") == name], runtime
            )
        report["complete_lifecycle_memory"] = memory_summary(all_phases, samples, runtime)
        training_memory = report["memory_by_stage"]["training"]
        observed = training_memory["sampled"]["owned_device_process_peak_bytes"]
        if not number(observed, positive=True):
            training_memory["sampled"]["gpu_memory_evidence_valid"] = False
            training_memory["sampled"]["errors"].append("no_positive_training_gpu_process_samples")
        report["best_fit_24gib_training_budget"] = {
            "observed_peak_within_budget": observed <= 24 * GIB
            if observed is not None and training_memory["sampled"]["gpu_memory_evidence_valid"]
            else None,
            "budget_bytes": 24 * GIB,
            "sampled_peak_lower_bound": True,
            "control_retained_if_over_budget": True,
        }
        final = evidence.read(path / "final-checkpoint.json")
        checkpoints = list((path / "native").rglob("checkpoints/iter_000300"))
        if len(checkpoints) != 1 or final.get("optimizer_step") != 300:
            raise InvalidEvidence("own final checkpoint missing")
        report["artifacts"] = {
            "dcp": artifact_size(checkpoints[0]),
            "portable": artifact_size(path / "portable/checkpoint_artifacts/iter_000300"),
        }
        summary = evidence.read(path / "scores/summary.json")
        if summary.get("complete") is not True:
            raise InvalidEvidence("scoring did not complete")
        if (
            any(path.rglob("window_*.json.gz"))
            or any(path.rglob("*.pt.trace.json*"))
            or contract.get("profile_enabled", False)
        ):
            raise InvalidEvidence("profiled timing excluded")
        controls = {k: v for k, v in spec.items() if k not in ("name", "arm")}
        identity = final["metadata"]["identity"]
        if (identity["quantization"] is not None) != (spec["arm"] == "nf4"):
            raise InvalidEvidence("observed model format differs from experiment arm")
        report["pairing"] = {
            "controls_sha256": digest_json(controls),
            "base_sha256": digest_json(identity["base"]),
            "processor_sha256": digest_json(identity["processor_contract"]),
            "targets_sha256": digest_json(identity["targets"]),
            "adapter_sha256": digest_json(identity["adapter"]),
            "initial_adapters_sha256": digest_json(identity["initial_adapters"]),
            "adapter_policy_sha256": digest_json(
                {k: v for k, v in identity["adapter"].items() if k not in ("rank", "alpha")}
            ),
            "build_sha256": digest_json(contract["build"]),
            "tool_sha256": digest_json(contract["tools"]),
            "data_sha256": digest_json([item["sha256"] for item in contract["files"][2:6]]),
            "allocation_sha256": digest_json(runtime["allocation_id"]) if runtime.get("allocation_id") else None,
            "gpu_uuid_sha256": digest_json(runtime["uuid"]),
            "cpu_affinity_sha256": digest_json(runtime["cpu_affinity"]),
            "cpu_topology_sha256": digest_json(runtime["cpu_topology"]) if runtime.get("cpu_topology") else None,
            "host_sha256": runtime.get("host_id_sha256"),
            "runtime_sha256": digest_json(
                {
                    k: runtime[k]
                    for k in (
                        "name",
                        "device_bytes",
                        "versions",
                        "runtime_precision",
                        "torch_git_version",
                        "torch_cuda",
                        "python",
                        "thread_environment",
                    )
                }
            ),
        }
        specification_hash = contract["files"][0]["sha256"]
        if not isinstance(specification_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", specification_hash):
            raise InvalidEvidence("input specification requires a valid SHA256 identity")
        report["input_specification_sha256"] = specification_hash
        report["adapter_configuration_matches_experiment"] = all(
            identity["adapter"].get(key) == spec[field]
            for key, field in (("rank", "rank"), ("alpha", "alpha"), ("initialization_seed", "seed"))
        )
        report["hardware"] = {
            "accelerator": runtime["name"],
            "device_bytes": runtime["device_bytes"],
            "logical_cpus": len(runtime["cpu_affinity"]),
            "physical_topology_recorded": bool(runtime.get("cpu_topology")),
        }
        report["execution"] = {
            "compilation_requested": spec["compile_language_model"],
            "timing_is_untraced": True,
            "startup_scope": "Training-stage elapsed includes source verification and process setup before adapter_loading_start. The loading interval itself excludes those costs.",
            "preflight_external_memory_sampled": False,
            "memory_scope": "Sampler starts after disposable preflight; complete lifecycle external peaks cover subsequent train/publication/reload/scoring stages. Preflight wall time is included in complete elapsed.",
        }
        report["completed"] = report["timing_eligible"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Exception text can contain private paths/IDs; publish a bounded category only.
        report["exclusions"].append(str(error) if isinstance(error, InvalidEvidence) else type(error).__name__)
        report["diagnostic_sha256"] = digest_json(str(error))
    report["source_sha256"] = {digest_json(name): value for name, value in evidence.sources.items()}
    return report


def paired_bootstrap(nf4: list[float], bf16: list[float], *, repetitions: int = 10000, seed: int = 2026) -> dict:
    if len(nf4) != 3 or len(bf16) != 3 or any(not number(v, positive=True) for v in nf4 + bf16):
        raise InvalidEvidence("inferential comparison requires three complete positive paired observations")
    if type(repetitions) is not int or repetitions < 1 or type(seed) is not int:
        raise InvalidEvidence("bootstrap count and seed must be positive-count/integer inputs")
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        indices = [rng.randrange(3) for _ in range(3)]
        values.append(math.fsum(nf4[i] for i in indices) / math.fsum(bf16[i] for i in indices))
    return {
        "nf4_over_bf16": math.fsum(nf4) / math.fsum(bf16),
        "paired_trial_ratios": [a / b for a, b in zip(nf4, bf16, strict=True)],
        "paired_bootstrap_95_interval": [quantile(values, 0.025), quantile(values, 0.975)],
        "one_sided_95_upper": quantile(values, 0.95),
        "repetitions": repetitions,
        "seed": seed,
        "independent_units": 3,
        "limitations": "Only three paired trials; resampling cannot establish broad hardware or workload generality. Updates within trials are not independent repetitions.",
    }


def compare_pairs(
    pairs: list[tuple[dict, dict]],
    *,
    kind: str = "controlled",
    repetitions: int = 10000,
    selection: BestFitSelection | None = None,
) -> dict:
    if kind not in ("controlled", "best_fit"):
        raise InvalidEvidence("unknown comparison kind")
    if kind == "best_fit" and not isinstance(selection, BestFitSelection):
        raise InvalidEvidence("best-fit comparison requires verified sealed selection receipts")
    if kind == "best_fit":
        selection.assert_stable()
    result = {"kind": kind, "inferential": False, "pairs": len(pairs), "exclusions": [], "release_qualified": False}
    identities = set()
    for a, b in pairs:
        if not a.get("timing_eligible") or not b.get("timing_eligible"):
            result["exclusions"].append("incomplete_or_invalid_pair")
            continue
        if a["experiment"].get("seed") != b["experiment"].get("seed"):
            result["exclusions"].append("paired_training_seeds_differ")
        if a["experiment"]["arm"] != "nf4" or b["experiment"]["arm"] != "bf16":
            raise InvalidEvidence("pair ordering must be NF4 then BF16")
        ids = (a["case_id"], b["case_id"])
        if any(identifier in identities for identifier in ids) or ids[0] == ids[1]:
            raise InvalidEvidence("duplicate run cannot supply another repeat")
        identities.update(ids)
        keys = set(a["pairing"])
        if kind == "best_fit":
            keys -= {"controls_sha256", "adapter_sha256", "initial_adapters_sha256"}
            if a["experiment"]["rank"] == b["experiment"]["rank"]:
                keys.add("initial_adapters_sha256")
        if any(
            a["pairing"][key] is None or b["pairing"][key] is None or a["pairing"][key] != b["pairing"][key]
            for key in keys
        ):
            result["exclusions"].append("unmatched_settings_build_data_allocation_or_hardware")
        if kind == "best_fit":
            differences = {
                key
                for key in set(a["experiment"]) | set(b["experiment"])
                if key != "arm" and a["experiment"].get(key) != b["experiment"].get(key)
            }
            if differences - BEST_FIT_FIELDS:
                result["exclusions"].append("best_fit_pair_changed_unselected_training_controls")
            for row in (a, b):
                key = row["experiment"]["arm"], row["experiment"]["seed"]
                if (
                    row["experiment"] != selection.specifications.get(key)
                    or row.get("input_specification_sha256") != selection.specification_sha256.get(key)
                    or row["pairing"]["data_sha256"] != selection.data_sha256
                    or row.get("adapter_configuration_matches_experiment") is not True
                ):
                    result["exclusions"].append("measured_best_fit_run_differs_from_locked_selection")
                memory = row["memory_by_stage"]["training"]
                sampled = memory["sampled"]
                peaks = [
                    sampled["owned_device_process_peak_bytes"],
                    memory["allocator_peak_allocated_bytes"],
                    memory["allocator_peak_reserved_bytes"],
                ]
                if (
                    sampled["gpu_memory_evidence_valid"] is not True
                    or sampled["host_pss_complete"] is not True
                    or memory["allocator_evidence_complete"] is not True
                    or any(not number(peak, positive=True) for peak in peaks)
                    or max(peaks) > 24 * GIB
                ):
                    result["exclusions"].append("selected_best_fit_run_lacks_valid_under_24gib_training_memory")
            result.setdefault("configuration_differences", []).append(
                {
                    key: [a["experiment"].get(key), b["experiment"].get(key)]
                    for key in set(a["experiment"]) | set(b["experiment"])
                    if key != "arm" and a["experiment"].get(key) != b["experiment"].get(key)
                }
            )
    for arm_index in (0, 1):
        complete = [pair[arm_index] for pair in pairs if pair[arm_index].get("timing_eligible")]
        configurations = {digest_json({k: v for k, v in row["experiment"].items() if k != "seed"}) for row in complete}
        workloads = {row["pairing"]["data_sha256"] for row in complete}
        builds = {row["pairing"]["build_sha256"] for row in complete}
        if len(configurations) > 1 or len(workloads) > 1 or len(builds) > 1:
            result["exclusions"].append("repeats_mix_different_arm_configurations_data_or_builds")
    if len(pairs) != 3:
        result["exclusions"].append("three_complete_pairs_required")
    if kind == "best_fit":
        for arm_index in (0, 1):
            if sorted(
                pair[arm_index]["experiment"]["seed"] for pair in pairs if pair[arm_index].get("timing_eligible")
            ) != list(CONFIRMATION_SEEDS):
                result["exclusions"].append("best_fit_requires_exact_paired_confirmation_seeds")
        result["selection_source_sha256"] = selection.source_sha256
        result["selection_scope"] = (
            "Pinned development/execution decisions and six confirmation specifications. Candidate ranking and recovery proofs are not rerun by this analyzer; untouched scoring is not authorized."
        )
    if result["exclusions"]:
        return result
    result["inferential"] = True
    result["timing"] = {}
    for label, _, _ in WINDOWS:
        result["timing"][label] = paired_bootstrap(
            [a["native"]["windows"][label]["update_seconds"]["mean"] for a, _ in pairs],
            [b["native"]["windows"][label]["update_seconds"]["mean"] for _, b in pairs],
            repetitions=repetitions,
        )
    result["complete_elapsed"] = paired_bootstrap(
        [a["complete_elapsed_seconds"] for a, _ in pairs],
        [b["complete_elapsed_seconds"] for _, b in pairs],
        repetitions=repetitions,
    )
    memory = []
    for a, b in pairs:
        x, y = (r["memory_by_stage"]["training"] for r in (a, b))
        valid = all(r["sampled"]["gpu_memory_evidence_valid"] for r in (x, y))
        benefit = (
            y["sampled"]["owned_device_process_peak_bytes"] - x["sampled"]["owned_device_process_peak_bytes"]
            if valid
            else None
        )
        memory.append(
            {
                "training_seed": a["experiment"]["seed"],
                "sampled_owned_process_benefit_bytes": benefit,
                "sampled_6gib_criterion_met": benefit >= 6 * GIB if benefit is not None else None,
                "allocator_allocated_benefit_bytes": y["allocator_peak_allocated_bytes"]
                - x["allocator_peak_allocated_bytes"]
                if all(r["allocator_peak_allocated_bytes"] is not None for r in (x, y))
                else None,
                "allocator_reserved_benefit_bytes": y["allocator_peak_reserved_bytes"]
                - x["allocator_peak_reserved_bytes"]
                if all(r["allocator_peak_reserved_bytes"] is not None for r in (x, y))
                else None,
            }
        )
    result["training_memory_pairs"] = memory
    result["original_matched_recipe_memory_eligible"] = kind == "controlled"
    result["observed_training_seeds"] = sorted({a["experiment"]["seed"] for a, _ in pairs})
    result["memory_scope"] = (
        "Original criterion concerns >=6GiB lower complete training device-process peaks for every seed. Reported NVML peaks are sampled lower bounds; allocator savings are a separate measure. No universal speed threshold is inferred."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--pair-plan", type=Path)
    parser.add_argument("--pair-plan-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.run)
    plan = None
    plan_sha256 = None
    selection = None
    if args.pair_plan:
        plan_bytes = args.pair_plan.read_bytes()
        plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        plan = json.loads(plan_bytes)
        if plan.get("kind") == "best_fit":
            if (
                plan.get("protocol") != "cosmos.qlora-performance-pairs.v2"
                or not args.pair_plan_sha256
                or plan_sha256 != args.pair_plan_sha256
            ):
                raise InvalidEvidence("best-fit pair plan requires v2 and its externally pinned SHA256")
            selection = load_best_fit_selection(plan["selection"])
        elif plan.get("protocol") != "cosmos.qlora-performance-pairs.v1":
            raise InvalidEvidence("invalid paired plan protocol")
        paths.extend(Path(pair[arm]["path"]) for pair in plan["pairs"] for arm in ("nf4", "bf16"))
    if not paths:
        parser.error("at least one run or pair plan is required")
    reports = {path.resolve(): summarize_run(path.resolve()) for path in paths}
    comparison = None
    if plan:
        pairs = []
        if len({pair["repeat"] for pair in plan["pairs"]}) != len(plan["pairs"]):
            raise InvalidEvidence("repeat IDs must be distinct")
        for pair in plan["pairs"]:
            arms = []
            for arm in ("nf4", "bf16"):
                item = pair[arm]
                run = reports[Path(item["path"]).resolve()]
                actual = hashlib.sha256((Path(item["path"]) / "contract.json").read_bytes()).hexdigest()
                if actual != item["contract_sha256"]:
                    raise InvalidEvidence("pair-plan contract digest differs")
                arms.append(run)
            pairs.append(tuple(arms))
        comparison = compare_pairs(pairs, kind=plan["kind"], selection=selection)
    if args.pair_plan and hashlib.sha256(args.pair_plan.read_bytes()).hexdigest() != plan_sha256:
        raise InvalidEvidence("paired plan changed during analysis")
    if selection is not None:
        selection.assert_stable()
    write_json_new(
        args.output,
        {
            "protocol": PROTOCOL,
            "runs": list(reports.values()),
            "comparison": comparison,
            "pair_plan_sha256": plan_sha256,
            "historical_evidence": "Not pooled: historical timing reports have different scope and remain separately labeled sources.",
            "release_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
