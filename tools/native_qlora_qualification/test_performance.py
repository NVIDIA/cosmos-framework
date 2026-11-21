# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic analyzer contracts; these tests establish no GPU speed or memory result."""

import copy
import json
from pathlib import Path

import pytest

from tools.native_qlora_qualification.performance import (
    GIB,
    STAGES,
    compare_pairs,
    memory_summary,
    metric_summary,
    paired_bootstrap,
    phase_name,
    sampled_memory,
    summarize_run,
)
from tools.native_qlora_qualification.protocol import Experiment


def metrics(steps=300, seconds=2):
    rows = [
        {
            "step": 0,
            "val/loss_avg": 1.0,
            "timer/validation_s": 3.0,
            "val/valid_token_count": 128,
            "val/valid_sequence_count": 64,
        }
    ]
    for step in range(1, steps + 1):
        row = {
            "step": step,
            "timer/e2e_step_s": seconds,
            "train/loss_avg": 0.5,
            "train/grad_norm": 1.0,
            "train/valid_token_count": 16 + step % 2,
            "train/valid_sequence_count": 8,
            "optim/update_outcome": "applied",
            "optim/optimizer_step": float(step),
            "optim/attempt_id": float(step),
            "optim/step_skipped": 0.0,
            "optim/nonfinite_skips_total": 0.0,
            "optim/nonfinite_loss": 0.0,
            "optim/nonfinite_gradient": 0.0,
            "train/preprocessing_processing_seconds": float(step),
            "train/preprocessing_wait_seconds": float(step) / 2,
        }
        if step % 50 == 0:
            row.update(
                {
                    "val/loss_avg": 0.5,
                    "timer/validation_s": 3.0,
                    "val/valid_token_count": 128,
                    "val/valid_sequence_count": 64,
                }
            )
        rows.append(row)
    return rows


def runtime():
    return {
        "name": "NVIDIA A100-SXM4-80GB",
        "device_bytes": 79 * GIB,
        "uuid": "GPU-private-uuid",
        "cpu_affinity": list(range(8)),
        "cpu_topology": [{"cpu": i, "core": i, "package": 0, "numa_nodes": [0]} for i in range(8)],
        "allocation_id": "private-allocation",
        "host_id_sha256": "a" * 64,
        "versions": {"torch": "2.11.0+cu128"},
        "runtime_precision": {"deterministic": True},
        "torch_git_version": "b" * 40,
        "torch_cuda": "12.8",
        "python": "3.12.0",
        "thread_environment": {"OMP_NUM_THREADS": "2"},
    }


def sample(t, *, used=20 * GIB, stage="training"):
    return {
        "unix_seconds": float(t),
        "device_uuid": "GPU-private-uuid",
        "device_memory_total_bytes": 80 * GIB,
        "device_memory_used_bytes": used + GIB,
        "device_compute_processes": [
            {"pid": 123, "used_gpu_memory_bytes": used, "belongs_to_run": True},
            {"pid": 456, "used_gpu_memory_bytes": GIB, "belongs_to_run": False},
        ],
        "selected_host_rss_sum_bytes": 4 * GIB,
        "selected_host_pss_sum_bytes": 3 * GIB,
        "host_pss_complete": True,
        "stage": stage,
    }


def boundary(phase, t, peak, *, pid=1, reset=True):
    return {
        "phase": phase,
        "pid": pid,
        "rank": 0,
        "unix_seconds": float(t),
        "monotonic_seconds": float(t),
        "reset_peak_after_record": reset,
        "cuda": {"device": 0, "peak_allocated_bytes": peak, "peak_reserved_bytes": peak + 1},
    }


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))


def experiment_root(tmp_path: Path, arm="nf4", repeat=1):
    root = tmp_path / f"{arm}_{repeat}"
    spec = Experiment(f"{arm}_{repeat}", arm).as_dict()
    contract = {
        "experiment": spec,
        "action": "train-eval",
        "build": {"code_revision": "c" * 40, "wheel_sha256": "d" * 64},
        "files": [{"path": "/private/source", "sha256": str(i) * 64} for i in range(6)],
        "tools": {"run.py": "e" * 64},
    }
    write(root / "contract.json", contract)
    outcomes = [
        {"name": name, "exit_code": 0, "elapsed_seconds": 700.0 if name == "training" else 10.0} for name in STAGES
    ]
    write(
        root / "completion.json",
        {
            "completed": True,
            "failure": None,
            "experiment": spec,
            "outcomes": outcomes,
            "complete_elapsed_seconds": 770.0,
        },
    )
    write(root / "runtime.json", runtime())
    for row in outcomes:
        write(root / "stages" / row["name"] / "outcome.json", row)
    write(root / "native/run/metrics.jsonl", metrics(), lines=True)
    used = (20 if arm == "nf4" else 27) * GIB
    phases = [
        boundary("adapter_loading_start", 1, 1),
        boundary("adapter_loading_complete", 3, used),
        boundary("trainer_ready", 4, used),
        boundary("initial_validation", 7, used),
        boundary("optimizer_step_1", 9, used),
        boundary("validation_checkpoint_boundary_1", 10, used),
        boundary("checkpoint_begin", 11, used),
        boundary("checkpoint_complete", 13, used),
    ]
    write(root / "stages/training/memory.jsonl", phases, lines=True)
    write(root / "nvml-host.jsonl", [sample(t, used=used) for t in range(1, 14)], lines=True)
    identity = {
        "base": {"revision": "f" * 40},
        "processor_contract": {"identity": "same"},
        "targets": {"language.linear": [2, 2]},
        "adapter": {"rank": 16, "alpha": 32, "seed": 42},
        "initial_adapters": {"tensor": "b" * 64},
        "quantization": {"format": "nf4"} if arm == "nf4" else None,
    }
    write(root / "final-checkpoint.json", {"optimizer_step": 300, "metadata": {"identity": identity}})
    write(root / "native/run/checkpoints/iter_000300/manifest.json", {})
    write(root / "portable/checkpoint_artifacts/iter_000300/manifest.json", {})
    write(root / "scores/summary.json", {"complete": True})
    return root


def test_exact_weighted_throughput_and_nonadditive_pipeline_counters():
    rows = metrics()
    rows[2]["timer/e2e_step_s"] = 20.0
    report = metric_summary(rows, steps=300, effective_batch=8)
    whole = report["windows"]["updates_1_300"]
    assert whole["examples_per_second"] == 2400 / 618
    assert whole["supervised_tokens_per_second"] == 4950 / 618
    assert report["windows"]["updates_6_30"]["update_seconds"]["count"] == 25
    assert report["windows"]["updates_31_300"]["update_seconds"]["count"] == 270
    assert report["initial_validation_seconds"] == 3
    assert report["scheduled_validation_seconds"] == 18
    assert report["preprocessing_last_cumulative_counters"]["preprocessing_processing_seconds"] == 300
    assert whole["update_seconds"]["sum"] != 618 + 300 + 150


@pytest.mark.parametrize(
    "damage", ["float_step", "bool_step", "missing", "duplicate", "nonfinite", "skip", "wrong_batch", "attempt"]
)
def test_bad_native_updates_fail(damage):
    rows = metrics()
    if damage == "float_step":
        rows[1]["step"] = 1.0
    elif damage == "bool_step":
        rows[1]["step"] = True
    elif damage == "missing":
        rows.pop(1)
    elif damage == "duplicate":
        rows[2]["step"] = 1
    elif damage == "nonfinite":
        rows[2]["timer/e2e_step_s"] = float("nan")
    elif damage == "skip":
        rows[2]["optim/step_skipped"] = 1
    elif damage == "wrong_batch":
        rows[2]["train/valid_sequence_count"] = 4
    else:
        rows[2]["optim/attempt_id"] = 3
    with pytest.raises(ValueError):
        metric_summary(rows, steps=300, effective_batch=8)


def test_checkpoint_begin_is_prior_window_and_overall_peak_survives_unknown_boundary():
    rows = [
        boundary("optimizer_step_1", 0, 10),
        boundary("checkpoint_begin", 2, 900),
        boundary("checkpoint_complete", 5, 100),
        boundary("unknown", 6, 1000),
    ]
    report = memory_summary(rows, [sample(i) for i in range(1, 7)], runtime())
    assert report["allocator_peak_allocated_bytes"] == 1000
    assert report["phases"]["checkpoint_write"]["allocator_peak_allocated_bytes"] == 100
    assert report["phases"]["checkpoint_write"]["interval_seconds"]["sum"] == 3
    assert report["phases"]["validation_or_boundary_overhead"]["allocator_peak_allocated_bytes"] == 900
    assert report["phases"]["unclassified"]["allocator_peak_allocated_bytes"] == 1000


def test_native_action_intervals_separate_training_evaluation_and_nested_checkpoint():
    timeline = [
        ("data_preparation_begin", 0),
        ("data_preparation_complete", 2),
        ("model_loading_begin", 3),
        ("model_loading_complete", 5),
        ("evaluation_begin", 6),
        ("evaluation_complete", 10),
        ("evaluation_begin", 20),  # Ten seconds of training, not one update.
        ("evaluation_complete", 24),
        ("checkpoint_begin", 25),
        ("checkpoint_begin", 26),
        ("checkpoint_complete", 29),
        ("checkpoint_complete", 30),
        ("portable_export_begin", 31),
        ("portable_export_complete", 35),
    ]
    rows = [boundary(phase, t, t * 100) for phase, t in timeline]
    for row in rows:
        row["rss_high_water_bytes"] = 70 * GIB  # Cumulative HWM is not a phase-local RSS sample.
    samples = [sample(t) for t in range(1, 36)]
    samples[26]["host_pss_complete"] = False
    report = memory_summary(rows, samples, runtime(), phase_family="native_action")
    phases = report["phases"]
    assert report["phase_family"] == "native_action"
    assert "unclassified" not in phases
    assert phases["action_training_block"]["interval_seconds"]["sum"] == 10
    assert phases["action_evaluation"]["interval_seconds"]["sum"] == 8
    assert phases["action_checkpoint_setup"]["interval_seconds"]["sum"] == 2
    assert phases["checkpoint_write"]["interval_seconds"]["sum"] == 3
    assert phases["action_checkpoint_verification"]["interval_seconds"]["sum"] == 1
    assert phases["portable_export"]["interval_seconds"]["sum"] == 4
    assert sum(phase["interval_seconds"]["sum"] for phase in phases.values()) == 35
    assert phases["checkpoint_write"]["allocator_peak_allocated_bytes"] == 2900
    assert phases["checkpoint_write"]["sampled_host_rss_sum_peak_bytes"] == 4 * GIB
    assert phases["checkpoint_write"]["sampled_host_pss_sum_peak_bytes"] == 3 * GIB
    assert not phases["checkpoint_write"]["sampled_host_pss_complete"]
    assert phases["action_evaluation"]["sampled_host_pss_complete"]


def test_action_and_reasoner_checkpoint_labels_require_explicit_family():
    assert phase_name("checkpoint_complete", "checkpoint_begin") == "validation_or_boundary_overhead"
    assert (
        phase_name("checkpoint_complete", "checkpoint_begin", phase_family="native_action") == "action_training_block"
    )
    assert phase_name("model_loading_complete", "evaluation_begin") == "unclassified"
    assert phase_name("adapter_loading_start", "adapter_loading_complete") == "loading"
    assert phase_name("unknown", "evaluation_begin", phase_family="native_action") == "unclassified"
    with pytest.raises(ValueError, match="phase family"):
        memory_summary([], [], runtime(), phase_family="guessed")
    with pytest.raises(ValueError, match="phase family"):
        phase_name("checkpoint_begin", "checkpoint_complete", phase_family="guessed")


def test_fresh_action_scoring_separates_post_score_integrity_and_publication_peaks():
    rows = [
        boundary("family_evaluation_loading_begin", 0, 0),
        boundary("family_evaluation_loading_complete", 2, 1000),
        boundary("family_evaluation_begin", 3, 2000),
        boundary("family_evaluation_complete", 7, 5000),
        boundary("family_evaluation_integrity_complete", 9, 8000),
        boundary("family_evaluation_publication_complete", 10, 3000),
    ]
    report = memory_summary(rows, [sample(t) for t in range(1, 11)], runtime(), phase_family="native_action")
    phases = report["phases"]
    assert "unclassified" not in phases
    assert report["allocator_peak_allocated_bytes"] == 8000
    assert phases["loading"]["allocator_peak_allocated_bytes"] == 1000
    assert phases["action_evaluation"]["allocator_peak_allocated_bytes"] == 5000
    assert phases["action_evaluation"]["interval_seconds"]["sum"] == 4
    assert phases["action_final_integrity"]["allocator_peak_allocated_bytes"] == 8000
    assert phases["evaluation_publication"]["interval_seconds"]["sum"] == 1
    assert sum(phase["interval_seconds"]["sum"] for phase in phases.values()) == 10


@pytest.mark.parametrize("failed", [False, True])
def test_family_artifact_export_keeps_failure_and_loading_intervals_distinct(failed):
    rows = [
        boundary("family_dense_loading_begin", 0, 0),
        boundary("family_dense_loading_complete", 2, 9000),
        boundary("family_dense_export_failed" if failed else "family_dense_export_complete", 5, 2000),
    ]
    report = memory_summary(rows, [sample(t) for t in range(1, 6)], runtime(), phase_family="native_action")
    phases = report["phases"]
    assert set(phases) == {"loading", "dense_export_incomplete" if failed else "dense_export"}
    export = phases["dense_export_incomplete" if failed else "dense_export"]
    assert export["allocator_peak_allocated_bytes"] == 2000
    assert export["interval_seconds"]["sum"] == 3
    assert report["allocator_peak_allocated_bytes"] == 9000


def test_missing_reset_or_other_process_cannot_establish_loading_interval():
    rows = [
        boundary("adapter_loading_start", 1, 5, reset=False),
        boundary("adapter_loading_complete", 4, 10),
        boundary("adapter_loading_start", 5, 3, pid=2),
        boundary("adapter_loading_complete", 6, 4, pid=1),
    ]
    assert "loading" not in memory_summary(rows, [], runtime())["phases"]


@pytest.mark.parametrize("end,phase", [("scoring_complete", "scoring"), ("scoring_failed", "scoring_incomplete")])
def test_scoring_memory_is_separate_from_setup_and_retains_failed_scope(end, phase):
    rows = [
        boundary("adapter_loading_complete", 1, 900),
        boundary("scoring_start", 3, 500),
        boundary(end, 8, 700, reset=False),
    ]
    result = memory_summary(rows, [sample(i) for i in range(2, 9)], runtime())
    assert set(result["phases"]) == {"scoring_setup", phase}
    assert result["phases"]["scoring_setup"]["allocator_peak_allocated_bytes"] == 500
    assert result["phases"]["scoring_setup"]["interval_seconds"]["sum"] == 2
    assert result["phases"][phase]["allocator_peak_allocated_bytes"] == 700
    assert result["phases"][phase]["interval_seconds"]["sum"] == 5
    assert result["allocator_peak_allocated_bytes"] == 900


def test_scoring_peak_is_not_established_without_its_same_process_reset():
    for start in [boundary("scoring_start", 1, 10, reset=False), boundary("scoring_start", 1, 10, pid=2)]:
        result = memory_summary([start, boundary("scoring_complete", 3, 700, reset=False)], [], runtime())
        assert "scoring" not in result["phases"]


@pytest.mark.parametrize("damage", ["sentinel", "negative", "identity", "capacity", "ownership"])
def test_invalid_nvml_cannot_establish_memory_gate(damage):
    row = sample(1)
    if damage == "sentinel":
        row["device_compute_processes"][0]["used_gpu_memory_bytes"] = 2**64 - 1
    elif damage == "negative":
        row["device_compute_processes"][0]["used_gpu_memory_bytes"] = -1
    elif damage == "identity":
        row["device_uuid"] = "other"
    elif damage == "capacity":
        row["device_memory_total_bytes"] = 1
    else:
        row["device_compute_processes"][0]["belongs_to_run"] = "true"
    assert (
        sampled_memory([row], expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])[
            "gpu_memory_evidence_valid"
        ]
        is False
    )


def test_nvml_capacity_differs_from_torch_usable_memory_and_pss_gaps_are_explicit():
    rows = [sample(1), sample(10)]
    rows[1]["host_pss_complete"] = False
    report = sampled_memory(rows, expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])
    assert report["gpu_memory_evidence_valid"] is True
    assert report["owned_device_process_peak_bytes"] == 20 * GIB
    assert report["nvml_capacity_bytes"] == 80 * GIB
    assert report["host_pss_complete"] is False
    assert report["sampling_gap_seconds"]["maximum"] == 9
    assert report["peaks_are_sampled_lower_bounds"] is True


def test_complete_report_keeps_bf16_over_budget_and_contains_no_private_paths(tmp_path):
    root = experiment_root(tmp_path, "bf16")
    report = summarize_run(root)
    assert report["completed"] and report["timing_eligible"], report
    assert report["best_fit_24gib_training_budget"]["observed_peak_within_budget"] is False
    assert report["best_fit_24gib_training_budget"]["control_retained_if_over_budget"] is True
    assert report["complete_elapsed_seconds"] == 770
    assert report["orchestrator_and_verification_residual_seconds"] == 10
    assert report["artifacts"]["portable"]["logical_bytes"] > 0
    assert "/private/" not in json.dumps(report)
    assert "GPU-private-uuid" not in json.dumps(report)
    assert "private-allocation" not in json.dumps(report)


@pytest.mark.parametrize("damage", ["failed", "missing_metrics", "wrong_validation", "profile", "bad_stage"])
def test_failed_incomplete_profiled_runs_remain_unranked(tmp_path, damage):
    root = experiment_root(tmp_path)
    if damage == "failed":
        file = root / "completion.json"
        row = json.loads(file.read_text())
        row["completed"] = False
        write(file, row)
    elif damage == "missing_metrics":
        (root / "native/run/metrics.jsonl").unlink()
    elif damage == "wrong_validation":
        rows = metrics()
        del rows[50]["val/loss_avg"]
        write(root / "native/run/metrics.jsonl", rows, lines=True)
    elif damage == "profile":
        write(root / "native/profiler/window_0.json.gz", {})
    else:
        write(root / "stages/scoring/outcome.json", {"name": "scoring", "exit_code": 1, "elapsed_seconds": 1})
    report = summarize_run(root)
    assert report["timing_eligible"] is False and report["exclusions"]


def test_three_paired_trials_bootstrap_does_not_pool_updates():
    result = paired_bootstrap([1, 2, 3], [2, 4, 6])
    assert result["nf4_over_bf16"] == 0.5
    assert result["paired_bootstrap_95_interval"] == [0.5, 0.5]
    assert result["independent_units"] == 3 and result["repetitions"] == 10000
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [2, 4])


def test_pair_matching_full_cost_memory_and_three_repeat_requirement(tmp_path):
    pairs = [
        (
            summarize_run(experiment_root(tmp_path, "nf4", repeat)),
            summarize_run(experiment_root(tmp_path, "bf16", repeat)),
        )
        for repeat in range(1, 4)
    ]
    report = compare_pairs(pairs)
    assert report["inferential"], report
    assert all(row["sampled_6gib_criterion_met"] for row in report["training_memory_pairs"])
    assert report["complete_elapsed"]["nf4_over_bf16"] == 1
    assert not compare_pairs(pairs[:2])["inferential"]
    damaged = copy.deepcopy(pairs)
    damaged[1][1]["pairing"]["cpu_affinity_sha256"] = "other"
    assert not compare_pairs(damaged)["inferential"]
    damaged = copy.deepcopy(pairs)
    damaged[1][1]["pairing"]["allocation_sha256"] = None
    assert not compare_pairs(damaged)["inferential"]
    with pytest.raises(ValueError):
        compare_pairs([pairs[0], pairs[0], pairs[2]])


def test_phase_invalid_gpu_identity_has_no_numeric_peak_claim():
    rows = [boundary("checkpoint_begin", 0, 10), boundary("checkpoint_complete", 2, 20)]
    samples = [sample(1)]
    samples[0]["device_uuid"] = "wrong GPU"
    phase = memory_summary(rows, samples, runtime())["phases"]["checkpoint_write"]
    assert phase["sampled_gpu_evidence_complete"] is False
    assert phase["sampled_owned_device_process_peak_bytes"] is None
    assert "wrong_device_identity" in phase["sampled_errors"]


def test_joint_nvml_total_and_process_sentinel_is_rejected():
    row = sample(1)
    row["device_memory_total_bytes"] = 2**64 - 1
    row["device_compute_processes"][0]["used_gpu_memory_bytes"] = 2**64 - 1
    result = sampled_memory([row], expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])
    assert result["gpu_memory_evidence_valid"] is False
    assert result["owned_device_process_peak_bytes"] is None


def test_malformed_experiment_cannot_echo_private_values_in_excluded_report(tmp_path):
    root = experiment_root(tmp_path)
    path = root / "contract.json"
    contract = json.loads(path.read_text())
    contract["experiment"]["unexpected_secret_field"] = "/private/user/sentinel"
    write(path, contract)
    report = summarize_run(root)
    assert report["timing_eligible"] is False
    assert "sentinel" not in json.dumps(report)
    assert "experiment" not in report


def test_best_fit_refuses_unbound_legacy_comparison(tmp_path):
    pairs = [(summarize_run(experiment_root(tmp_path, "nf4")), summarize_run(experiment_root(tmp_path, "bf16")))]
    with pytest.raises(ValueError, match="selection receipts"):
        compare_pairs(pairs, kind="best_fit")


def test_valid_timing_cannot_override_invalid_memory_evidence(tmp_path):
    pairs = [
        (
            summarize_run(experiment_root(tmp_path, "nf4", repeat)),
            summarize_run(experiment_root(tmp_path, "bf16", repeat)),
        )
        for repeat in range(1, 4)
    ]
    pairs[1][0]["memory_by_stage"]["training"]["sampled"]["gpu_memory_evidence_valid"] = False
    report = compare_pairs(pairs)
    assert report["inferential"]
    assert report["training_memory_pairs"][1]["sampled_6gib_criterion_met"] is None


def best_fit_fixture(tmp_path, *, different_rank=True):
    import hashlib

    from tools.native_qlora_qualification.common import digest_json
    from tools.native_qlora_qualification.performance import load_best_fit_selection

    def bind(name, value):
        path = tmp_path / "selection" / (name + ".json")
        write(path, value)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    gates = {
        "relative_supervised_token_nll_one_sided_95_upper_max": 0.02,
        "accuracy_loss_one_sided_95_upper_max": 0.01,
        "paired_training_seeds": [17, 42, 123],
    }
    development = {
        "protocol": "cosmos.qlora-development-plan.v1",
        "development_manifest_sha256": "2" * 64,
        "development_data_sha256": "3" * 64,
        "fresh_training_data_sha256": "4" * 64,
        "fixed_gates": gates,
    }
    bindings = {"development_plan": bind("development", development)}
    confirmation = {
        "protocol": "cosmos.qlora-confirmation-and-best-fit-plan.v2",
        "development_plan_sha256": bindings["development_plan"]["sha256"],
        "paired_seeds": [17, 42, 123],
        "runtime_matrix_plans": [{"sha256": "5" * 64}, {"sha256": "6" * 64}],
        "best_fit": {
            "steps": 300,
            "warmup_steps": 30,
            "effective_batch": 8,
            "execution_selection_contract": {
                "activation_checkpointing": ["full", "none"],
                "microbatches": [1, 2, 4],
                "fixed_model_execution": {
                    "adapter_compute_dtype": "float32",
                    "logits_mode": "full",
                    "compile_language_model": False,
                },
                "timed_updates_inclusive": [3, 20],
            },
        },
    }
    bindings["confirmation_plan"] = bind("confirmation", confirmation)
    hyperparameters = {
        "nf4": {"rank": 32 if different_rank else 16, "alpha": 64.0 if different_rank else 32.0, "learning_rate": 1e-4},
        "bf16": {"rank": 16, "alpha": 32.0, "learning_rate": 5e-5},
    }
    candidates = [
        {
            "candidate": "chosen",
            "arm": arm,
            "state": "completed",
            "eligible": True,
            "hyperparameters": hyperparameters[arm],
            "contract_sha256": "7" * 64,
        }
        for arm in ("nf4", "bf16")
    ]
    decision = {
        "protocol": "cosmos.qlora-development-selection.v1",
        "complete": True,
        "development_plan_sha256": bindings["development_plan"]["sha256"],
        "development_manifest_sha256": development["development_manifest_sha256"],
        "confirmation_seeds_required": [17, 42, 123],
        "fixed_release_gates": gates,
        "development_rows": 584,
        "development_groups": 12,
        "candidates": candidates,
        "arm_status": {arm: "development_candidate_selected" for arm in ("nf4", "bf16")},
        "selected": {
            row["arm"]: {
                "candidate": row["candidate"],
                "hyperparameters": row["hyperparameters"],
                "contract_sha256": row["contract_sha256"],
                "evidence_sha256": digest_json(row),
            }
            for row in candidates
        },
    }
    bindings["development_selection"] = bind("decision", decision)
    selected = {
        "nf4": {
            "activation_checkpointing": "none",
            "microbatch": 2,
            "runtime_completion_sha256": "8" * 64,
            "lifecycle_completion_sha256": "9" * 64,
        },
        "bf16": {
            "activation_checkpointing": "full",
            "microbatch": 1,
            "runtime_completion_sha256": "a" * 64,
            "lifecycle_completion_sha256": "b" * 64,
        },
    }
    specs = []
    pairs = []
    for seed in (17, 42, 123):
        pair = []
        for arm in ("nf4", "bf16"):
            spec = Experiment(
                f"selected_{arm}_{seed}",
                arm,
                seed=seed,
                **hyperparameters[arm],
                **{k: selected[arm][k] for k in ("activation_checkpointing", "microbatch")},
            ).as_dict()
            spec_binding = bind(spec["name"], spec)
            specs.append(spec_binding)
            root = experiment_root(tmp_path, arm, seed)
            contract = json.loads((root / "contract.json").read_text())
            contract["experiment"] = spec
            contract["files"][0]["sha256"] = spec_binding["sha256"]
            contract["files"][5]["sha256"] = development["development_data_sha256"]
            write(root / "contract.json", contract)
            complete = json.loads((root / "completion.json").read_text())
            complete["experiment"] = spec
            write(root / "completion.json", complete)
            final = json.loads((root / "final-checkpoint.json").read_text())
            final["metadata"]["identity"]["adapter"] = {
                "rank": spec["rank"],
                "alpha": spec["alpha"],
                "initialization_seed": seed,
                "parameter_dtype": "float32",
                "method": "lora",
            }
            final["metadata"]["identity"]["initial_adapters"] = {"tensor": digest_json([seed, spec["rank"]])}
            write(root / "final-checkpoint.json", final)
            if arm == "bf16":
                phases = [json.loads(line) for line in (root / "stages/training/memory.jsonl").read_text().splitlines()]
                for phase in phases:
                    phase["cuda"]["peak_allocated_bytes"] = min(phase["cuda"]["peak_allocated_bytes"], 22 * GIB)
                    phase["cuda"]["peak_reserved_bytes"] = min(phase["cuda"]["peak_reserved_bytes"], 22 * GIB + 1)
                write(root / "stages/training/memory.jsonl", phases, lines=True)
                write(root / "nvml-host.jsonl", [sample(t, used=22 * GIB) for t in range(1, 14)], lines=True)
            pair.append(summarize_run(root))
        pairs.append(tuple(pair))
    execution = {
        "protocol": "cosmos.qlora-execution-selection.v1",
        "complete": True,
        "confirmation_plan_sha256": bindings["confirmation_plan"]["sha256"],
        "development_selection_sha256": bindings["development_selection"]["sha256"],
        "runtime_matrix_plan_sha256s": ["5" * 64, "6" * 64],
        "selected": selected,
        "evidence_sha256": ["8" * 64, "9" * 64, "a" * 64, "b" * 64],
        "confirmation_specs": specs,
    }
    bindings["execution_selection"] = bind("execution", execution)
    return pairs, load_best_fit_selection(bindings), bindings


def test_best_fit_requires_selected_five_fields_exact_specs_and_seeds(tmp_path):
    pairs, selection, _ = best_fit_fixture(tmp_path)
    assert all(a["timing_eligible"] and b["timing_eligible"] for a, b in pairs)
    assert not compare_pairs(pairs)["inferential"]
    report = compare_pairs(pairs, kind="best_fit", selection=selection)
    assert report["inferential"], report
    assert report["observed_training_seeds"] == [17, 42, 123]
    assert set(report["configuration_differences"][0]) == {
        "rank",
        "alpha",
        "learning_rate",
        "microbatch",
        "activation_checkpointing",
    }
    assert report["original_matched_recipe_memory_eligible"] is False
    assert "private" not in json.dumps(report) and str(tmp_path) not in json.dumps(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_steps", 10),
        ("cache_bytes", 0),
        ("adapter_compute_dtype", "bfloat16"),
        ("learning_rate", 2e-4),
        ("rank", 64),
        ("microbatch", 4),
        ("activation_checkpointing", "selective"),
    ],
)
def test_best_fit_rejects_unselected_or_additional_changes(tmp_path, field, value):
    pairs, selection, _ = best_fit_fixture(tmp_path)
    pairs[0][0]["experiment"][field] = value
    report = compare_pairs(pairs, kind="best_fit", selection=selection)
    assert not report["inferential"]
    assert "measured_best_fit_run_differs_from_locked_selection" in report["exclusions"]


@pytest.mark.parametrize(
    "damage", ["seed", "spec", "data", "adapter_policy", "adapter_binding", "invalid_memory", "over_budget"]
)
def test_best_fit_rejects_identity_seed_and_budget_failures(tmp_path, damage):
    pairs, selection, _ = best_fit_fixture(tmp_path)
    row = pairs[0][0]
    if damage == "seed":
        row["experiment"]["seed"] = 42
    elif damage == "spec":
        row["input_specification_sha256"] = "f" * 64
    elif damage == "data":
        row["pairing"]["data_sha256"] = "f" * 64
    elif damage == "adapter_policy":
        row["pairing"]["adapter_policy_sha256"] = "f" * 64
    elif damage == "adapter_binding":
        row["adapter_configuration_matches_experiment"] = False
    elif damage == "invalid_memory":
        row["memory_by_stage"]["training"]["sampled"]["gpu_memory_evidence_valid"] = False
    else:
        row["memory_by_stage"]["training"]["allocator_peak_reserved_bytes"] = 25 * GIB
    assert not compare_pairs(pairs, kind="best_fit", selection=selection)["inferential"]


def test_best_fit_same_rank_keeps_initial_adapter_equality(tmp_path):
    pairs, selection, _ = best_fit_fixture(tmp_path, different_rank=False)
    assert compare_pairs(pairs, kind="best_fit", selection=selection)["inferential"]
    pairs[0][0]["pairing"]["initial_adapters_sha256"] = "wrong"
    assert not compare_pairs(pairs, kind="best_fit", selection=selection)["inferential"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "unsealed",
        "incomplete",
        "wrong_plan",
        "unbound_candidate",
        "missing_recovery",
        "spec_control",
        "spec_seed",
    ],
)
def test_best_fit_receipt_negatives(tmp_path, damage):
    import hashlib

    from tools.native_qlora_qualification.performance import load_best_fit_selection

    _, _, bindings = best_fit_fixture(tmp_path)
    if damage == "missing":
        bindings.pop("development_selection")
    elif damage == "unsealed":
        Path(bindings["execution_selection"]["path"]).write_text("{}")
    else:
        key = "development_selection" if damage == "unbound_candidate" else "execution_selection"
        path = Path(bindings[key]["path"])
        value = json.loads(path.read_text())
        if damage == "incomplete":
            value["complete"] = False
        elif damage == "wrong_plan":
            value["confirmation_plan_sha256"] = "f" * 64
        elif damage == "unbound_candidate":
            value["selected"]["nf4"]["hyperparameters"]["rank"] = 64
        elif damage == "missing_recovery":
            value["selected"]["nf4"]["lifecycle_completion_sha256"] = "c" * 64
        else:
            spec_path = Path(value["confirmation_specs"][0]["path"])
            spec = json.loads(spec_path.read_text())
            spec["cache_bytes" if damage == "spec_control" else "seed"] = 0
            write(spec_path, spec)
            value["confirmation_specs"][0]["sha256"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        write(path, value)
        bindings[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if key == "development_selection":
            ep = Path(bindings["execution_selection"]["path"])
            e = json.loads(ep.read_text())
            e["development_selection_sha256"] = bindings[key]["sha256"]
            write(ep, e)
            bindings["execution_selection"]["sha256"] = hashlib.sha256(ep.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_best_fit_selection(bindings)


def test_best_fit_receipts_cannot_change_after_initial_load(tmp_path):
    pairs, selection, bindings = best_fit_fixture(tmp_path)
    Path(bindings["execution_selection"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed during analysis"):
        compare_pairs(pairs, kind="best_fit", selection=selection)


@pytest.mark.parametrize("mode", ["legacy", "missing_digest", "wrong_digest", "valid"])
def test_best_fit_cli_requires_external_plan_digest(tmp_path, monkeypatch, mode):
    import hashlib
    import sys

    from tools.native_qlora_qualification.performance import main

    _, _, bindings = best_fit_fixture(tmp_path)
    pairs = []
    for repeat, seed in enumerate((17, 42, 123), 1):
        pair = {"repeat": repeat}
        for arm in ("nf4", "bf16"):
            root = tmp_path / f"{arm}_{seed}"
            pair[arm] = {
                "path": str(root),
                "contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
            }
        pairs.append(pair)
    plan = {
        "protocol": "cosmos.qlora-performance-pairs.v1" if mode == "legacy" else "cosmos.qlora-performance-pairs.v2",
        "kind": "best_fit",
        "selection": bindings,
        "pairs": pairs,
    }
    path, output = tmp_path / "pairs.json", tmp_path / "public-report.json"
    write(path, plan)
    arguments = ["performance.py", "--pair-plan", str(path), "--output", str(output)]
    if mode != "missing_digest":
        arguments += [
            "--pair-plan-sha256",
            "f" * 64 if mode == "wrong_digest" else hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    monkeypatch.setattr(sys, "argv", arguments)
    if mode == "valid":
        main()
        report = json.loads(output.read_text())
        assert report["comparison"]["inferential"] is True
        assert str(tmp_path) not in output.read_text()
    else:
        with pytest.raises(ValueError, match="externally pinned"):
            main()
        assert not output.exists()


@pytest.mark.parametrize("damage", ["pair_plan", "selection"])
def test_best_fit_publication_refuses_inputs_mutated_during_bootstrap(tmp_path, monkeypatch, damage):
    import hashlib
    import sys

    from tools.native_qlora_qualification import performance

    _, _, bindings = best_fit_fixture(tmp_path)
    pairs = []
    for repeat, seed in enumerate((17, 42, 123), 1):
        pair = {"repeat": repeat}
        for arm in ("nf4", "bf16"):
            root = tmp_path / f"{arm}_{seed}"
            pair[arm] = {
                "path": str(root),
                "contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
            }
        pairs.append(pair)
    path, output = tmp_path / "pairs.json", tmp_path / "public-report.json"
    write(
        path,
        {"protocol": "cosmos.qlora-performance-pairs.v2", "kind": "best_fit", "selection": bindings, "pairs": pairs},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "performance.py",
            "--pair-plan",
            str(path),
            "--pair-plan-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--output",
            str(output),
        ],
    )
    original = performance.paired_bootstrap

    def mutate_then_bootstrap(*args, **kwargs):
        target = path if damage == "pair_plan" else Path(bindings["execution_selection"]["path"])
        target.write_text("{}")
        return original(*args, **kwargs)

    monkeypatch.setattr(performance, "paired_bootstrap", mutate_then_bootstrap)
    with pytest.raises(ValueError, match="changed during analysis"):
        performance.main()
    assert not output.exists()
