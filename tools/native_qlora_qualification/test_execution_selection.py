# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic schema/evidence tests; no runtime speed, memory or quality claim."""

import copy
import hashlib
import json

import pytest

from tools.native_qlora_qualification import execution_selection as select
from tools.native_qlora_qualification.common import digest_json
from tools.native_qlora_qualification.protocol import Experiment
from tools.native_qlora_qualification.test_performance import boundary, runtime, sample


class Fixture:
    def __init__(self, root):
        self.root = root
        self.inputs = {
            "protocol": select.INPUT_PROTOCOL,
            "recorded_root": "/recorded",
            "files": {},
            "plans": {},
            "profiles": [],
        }

    def put(self, name, value, *, lines=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))
        self.inputs["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.inputs["files"][name]

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def evidence(self):
        return select.Evidence(self.root, self.inputs)

    def outcome(self, root, name, interrupted=False, *, lifecycle=False):
        row = {"name": name, "exit_code": 1 if interrupted else 0, "elapsed_seconds": 1.0}
        if lifecycle:
            row.update(timed_out=False, injected_interruption_expected=interrupted, owned_process_cleanup={})
        self.put(f"{root}/stages/{name}/outcome.json", row)
        return row

    def compare(self, root, name, operation):
        if operation == "compare-probes":
            reference, candidate = root + "/probes/dcp", root + "/probes/portable"
        else:
            mode = "accumulation" if name.startswith("accumulation") else "boundary"
            step = 2 if name.endswith("source_compare") else int(name[-1])
            suffix = mode if step == 2 else mode + "_resumed"
            reference = f"{root}/native/qualification/recovery/reference/canary/checkpoints/iter_{step:06d}"
            candidate = f"{root}/native/qualification/recovery/{suffix}/canary/checkpoints/iter_{step:06d}"
        self.put(
            f"{root}/stages/{name}/command.json",
            [
                "python",
                "-m",
                "integration",
                operation,
                "--reference",
                "/recorded/" + reference,
                "--candidate",
                "/recorded/" + candidate,
                "--atol",
                "0",
                "--rtol",
                "0",
            ],
        )

    def probes(self, root, arm, *, identity_sha=None, checkpoints=None):
        probe = {
            "protocol": "cosmos.qlora-integration-probe.v1",
            "quantized_matrices": 252 if arm == "nf4" else 0,
            "trainable_tensors": 504,
            "optimizer_parameters": 504,
            "trainable_elements": 1024,
            "processor_contract": {"fixture": True},
            "adapters": {str(i): "1" * 64 for i in range(504)},
            "adapter_updates": {"A": 252, "B": 252},
            "identity_sha256": identity_sha or "2" * 64,
            "probes": {
                name: {"sequence_length": 300, "supervised_tokens": 2, "loss": 0.3, "zero_media_max_logit_change": 1.0}
                for name in ("text", "image", "video")
            },
        }
        checkpoints = checkpoints or (
            f"/recorded/{root}/native/qualification/recovery/accumulation_resumed/canary/checkpoints/iter_000004",
            f"/recorded/{root}/portable/checkpoint_artifacts/iter_000004",
        )
        for kind, checkpoint in zip(("dcp", "portable"), checkpoints, strict=True):
            self.put(f"{root}/probes/{kind}/probe.json", {**probe, "checkpoint": checkpoint})


def native_rows(seconds):
    rows, expected = [], {"expected_updates": [], "expected_validation_tokens": 16}
    for step in range(21):
        if step:
            rows.append(
                {
                    "step": step,
                    "timer/e2e_step_s": seconds if step > 2 else 100.0,
                    "train/loss_avg": 0.3,
                    "train/grad_norm": 1.0,
                    "train/valid_sequence_count": 8,
                    "train/valid_token_count": 16,
                    "optim/update_outcome": "applied",
                    "optim/optimizer_step": step,
                    "optim/attempt_id": step,
                    "optim/step_skipped": 0,
                    "optim/nonfinite_skips_total": 0,
                    "optim/nonfinite_loss": 0,
                    "optim/nonfinite_gradient": 0,
                    "optim/finite_gradient_check_enabled": 1,
                }
            )
            expected["expected_updates"].append({"step": step, "examples": 8, "supervised_tokens": 16})
        if step in (0, 10, 20):
            rows.append(
                {
                    "step": step,
                    "val/loss_avg": 0.4,
                    "timer/validation_s": 1.0,
                    "val/valid_token_count": 16,
                    "val/valid_sequence_count": 8,
                }
            )
    return rows, expected


def make_cell(f, spec, spec_sha, plan, build, profile, tools, *, seconds, peak):
    root = f"cells/{spec.name}"
    records, expected = native_rows(seconds)
    expected.update(
        completed=True,
        manifest_measured_shapes=profile["measurements"],
        processor_identity_sha256=profile["processor_identity_sha256"],
    )
    f.put(root + "/workload-verification.json", expected)
    f.put(root + "/native/run/metrics.jsonl", records, lines=True)
    native = select.native_metrics(records, expected)
    f.put(root + "/native-metrics-audit.json", native)
    where = {**runtime(), "physical_cpu_cores": 8}
    f.put(root + "/runtime.json", where)
    phases = [
        boundary(name, i + 1, peak)
        for i, name in enumerate(
            (
                "adapter_loading_start",
                "adapter_loading_complete",
                "trainer_ready",
                "initial_validation",
                "optimizer_step_1",
                "validation_checkpoint_boundary_1",
                "optimizer_step_2",
                "checkpoint_begin",
                "checkpoint_complete",
            )
        )
    ]
    samples = [sample(i + 1, used=peak) for i in range(9)]
    f.put(root + "/stages/training/memory.jsonl", phases, lines=True)
    f.put(root + "/nvml-host.jsonl", samples, lines=True)
    memory = select.memory_summary(phases, samples, where)
    measured = select.budget(memory)
    assert measured["evidence_complete"], measured
    tools = {"runtime.py": "1" * 64}
    f.put(
        root + "/contract.json",
        {
            "experiment": spec.as_dict(),
            "spec_sha256": spec_sha,
            "overrides": spec.overrides(),
            "build": build,
            "profile": "short_video8",
            "runtime_stress_only": True,
            "quality_evidence": False,
            "tools": tools,
            **{key: plan[key] for key in ("workload_manifest_sha256", "profile_manifest_sha256")},
        },
    )
    identity = {"training_contract": build, "quantization": {"format": "nf4"} if spec.arm == "nf4" else None}
    checkpoint = f"/recorded/{root}/native/run/checkpoints/iter_000020"
    f.put(
        root + "/final-checkpoint.json",
        {"checkpoint": checkpoint, "optimizer_step": 20, "metadata": {"identity": identity}},
    )
    f.probes(
        root,
        spec.arm,
        identity_sha=digest_json({key: value for key, value in identity.items() if key != "training_contract"}),
        checkpoints=(checkpoint, f"/recorded/{root}/portable/checkpoint_artifacts/iter_000020"),
    )
    f.compare(root, "reload_exact", "compare-probes")
    complete = {
        "protocol": "cosmos.native-qlora-runtime-cell.v1",
        "completed": True,
        "failure": None,
        "experiment": spec.as_dict(),
        "outcomes": [f.outcome(root, name) for name in select.CELL_STAGES],
        "native": native,
        "training_memory": memory,
        "training_budget": measured,
        "eligible_for_higher_microbatch": measured["eligible_for_higher_microbatch"],
        "complete_elapsed_seconds": 100.0,
    }
    sha = f.put(root + "/completion.json", complete)
    return {
        "kind": "runtime",
        "arm": spec.arm,
        "microbatch": spec.microbatch,
        "path": "/recorded/" + root,
        "completion_sha256": sha,
        "eligible_for_higher_microbatch": complete["eligible_for_higher_microbatch"],
    }


def make_lifecycle(f, spec, build, profile):
    root = "lives/" + spec.name
    requested = [x for x in spec.overrides() if x.split("=", 1)[0] not in set(select.FIXED) | select.RESERVED]
    build = {key: build[key] for key in ("code_revision", "wheel_sha256")}
    experiment = {
        "config": spec.recipe,
        "requested_overrides": requested,
        "qualification_overrides": select.controlled_overrides(requested),
        "build_receipt": build,
        "data_sha256": profile["files"]["validation"],
        "exact_atol": 0,
        "exact_rtol": 0,
    }
    f.put(root + "/experiment.json", experiment)
    names, proof = ["reference"], {}
    records = []
    for step in (3, 4):
        row = {key: 0.0 for key in select.NUMERICAL_METRICS}
        row.update(
            step=step,
            **{
                "train/loss_avg": 0.3,
                "optim/update_outcome": "applied",
                "optim/optimizer_step": step,
                "optim/attempt_id": step,
                "optim/finite_gradient_check_enabled": 1,
            },
        )
        records.append(row)
    for run in ("reference", "boundary_resumed", "accumulation_resumed"):
        f.put(f"{root}/native/qualification/recovery/{run}/canary/metrics.jsonl", records, lines=True)
    f.probes(root, spec.arm)
    for mode in ("boundary", "accumulation"):
        f.put(
            root + "/" + mode + "-interruption.json",
            {
                "protocol": "cosmos.qlora-injected-interruption.v1",
                "exit_code": 86,
                "mode": mode,
                "checkpoint": {
                    "optimizer_step": 2,
                    "healthy_completed_boundary": True,
                    "path": f"/recorded/{root}/native/qualification/recovery/{mode}/canary/checkpoints/iter_000002",
                },
                "live_state": {
                    "optimizer_step": 2,
                    "attempt_id": 3 if mode == "accumulation" else 2,
                    "accumulation_step": 1 if mode == "accumulation" else 0,
                    "partial_checkpoint_rejected": mode == "accumulation",
                    "frozen_base_validated": True,
                    "exact_optimizer_membership_validated": True,
                    "adapter_tensors": 504,
                },
            },
        )
        names += [mode, mode + "_source_compare", mode + "_resumed", mode + "_compare_3", mode + "_compare_4"]
        f.compare(root, mode + "_source_compare", "compare-checkpoints")
        proof[mode] = {}
        for step in ("3", "4"):
            stage = mode + "_compare_" + step
            f.compare(root, stage, "compare-checkpoints")
            metrics = {
                "passed": True,
                "optimizer_step": int(step),
                "numerical_metrics": select.continuation_metrics(records, int(step)),
            }
            f.put(f"{root}/stages/{stage}/numerical-update.json", metrics)
            proof[mode][step] = {
                "metrics": metrics,
                "checkpoint": {
                    "optimizer_step": int(step),
                    "healthy_completed_boundary": True,
                    "path": f"/recorded/{root}/native/qualification/recovery/{mode}_resumed/canary/checkpoints/iter_{int(step):06d}",
                },
            }
    names += ["publish_portable", "probe_dcp", "probe_portable", "compare_reload"]
    f.compare(root, "compare_reload", "compare-probes")
    result = {
        "protocol": "cosmos.native-qlora-recovery-result.v1",
        "complete": True,
        "reload_exact": True,
        "build_receipt": build,
        "experiment_sha256": digest_json(experiment),
        "recovery": proof,
        "outcomes": [f.outcome(root, name, name in ("boundary", "accumulation"), lifecycle=True) for name in names],
    }
    sha = f.put(root + "/completion.json", result)
    return {"kind": "lifecycle", "path": "/recorded/" + root, "completion_sha256": sha}


def fixture(tmp_path, monkeypatch, *, overbudget=None):
    f = Fixture(tmp_path)
    build = {"code_revision": select.TRAINING_REVISION, "wheel_sha256": select.WHEEL_SHA}
    workflow = {"files": {"tools/native_qlora_qualification/runtime.py": "1" * 64}}
    workflow_sha = f.put("workflows/941b645/receipt.json", workflow)
    monkeypatch.setattr(select, "WORKFLOW_SHA", workflow_sha)
    policy = {
        "max_length": 4096,
        "processor_kwargs": {
            "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
            "videos_kwargs": {"num_frames": 8, "fps": None, "size": {"shortest_edge": 32768, "longest_edge": 524288}},
        },
        "device_process_budget_bytes": select.BUDGET,
        "microbatches": [{"microbatch": b, "grad_accum": 8 // b} for b in (1, 2, 4)],
    }
    profile = {"policy": policy, "processor_identity_sha256": "a" * 64, "files": {}, "measurements": {}}
    for split, count in (("train", 32), ("validation", 8)):
        rows = [{"id": f"{split}-{i}"} for i in range(count)]
        profile["files"][split] = f.put(f"workload/short_video8/{split}.jsonl", rows, lines=True)
        profile["measurements"][split] = {
            "rows": [
                {"output_row_sha256": digest_json(row), "sequence_length": 300, "supervised_tokens": 2} for row in rows
            ]
        }
    profile_sha = f.put("workload/short_video8/manifest.json", profile)
    top_sha = f.put(
        "workload/manifest.json",
        {
            "protocol": "cosmos.native-qlora-runtime-workloads.v1",
            "seed": 42,
            "runtime_stress_only": True,
            "quality_evidence": False,
            "variants": {"short_video8": {"manifest_sha256": profile_sha, **policy}},
        },
    )
    matrices = []
    for ac in ("full", "none"):
        entries, specs = [], {}
        for micro in (1, 2, 4):
            for arm in select.ARMS:
                spec = Experiment(
                    f"runtime_{ac}_{arm}_b{micro}",
                    arm,
                    microbatch=micro,
                    activation_checkpointing=ac,
                    steps=20,
                    warmup_steps=2,
                    save_every=10,
                    validation_every=10,
                    validation_samples=8,
                )
                path = "specs/" + spec.name + ".json"
                sha = f.put(path, spec.as_dict())
                entries.append(
                    {"arm": arm, "microbatch": micro, "profile": "short_video8", "path": path, "sha256": sha}
                )
                specs[arm, micro] = spec
        matrix_path = f"matrices/{ac}.json"
        matrix_sha = f.put(
            matrix_path,
            {
                "protocol": "cosmos.qlora-runtime-matrix-plan.v1",
                "training_runtime_revision": select.TRAINING_REVISION,
                "stage_iterations": 20,
                "effective_batch": 8,
                "device_process_budget_bytes": select.BUDGET,
                "order": entries,
            },
        )
        matrices.append({"activation_checkpointing": ac, "path": matrix_path, "sha256": matrix_sha})
        plan = {
            "protocol": "cosmos.qlora-gpu-plan.v2",
            "mode": "runtime_profile",
            "profile": "short_video8",
            "matrix_plan": matrix_path,
            "workloads": "workload",
            "workload_manifest_sha256": top_sha,
            "profile_manifest_sha256": profile_sha,
            "inputs": [
                {"path": path, "sha256": sha}
                for path, sha in f.inputs["files"].items()
                if path.startswith(("specs/", "matrices/", "workload/"))
            ],
        }
        plan_sha = f.put(f"plans/runtime_{ac}.json", plan)
        attempt = "attempts/" + ac
        f.put(
            attempt + "/identity.json",
            {
                "plan": plan,
                "plan_sha256": plan_sha,
                "driver_sha256": select.DRIVER_SHA,
                "workflow_receipt_sha256": workflow_sha,
                "build": build,
                "allocation": "private-allocation",
            },
        )
        previous, accepted, outputs, stages = {}, {}, [], []
        for entry in entries:
            arm, micro = entry["arm"], entry["microbatch"]
            spec = specs[arm, micro]
            if micro > 1 and not previous.get(arm):
                outputs.append(
                    {"kind": "runtime_skipped", "arm": arm, "microbatch": micro, "reason": "previous_cell_not_eligible"}
                )
                continue
            peak = 26 * select.GIB if overbudget == (ac, arm, micro) else (10 + micro) * select.GIB
            output = make_cell(
                f,
                spec,
                entry["sha256"],
                plan,
                build,
                profile,
                {},
                seconds=10 - micro + (0 if ac == "none" else 1),
                peak=peak,
            )
            outputs.append(output)
            previous[arm] = output["eligible_for_higher_microbatch"]
            if previous[arm]:
                accepted[arm] = micro
            stages.append(f.outcome(attempt, spec.name))
        for arm in select.ARMS:
            if arm not in accepted:
                outputs.append({"kind": "no_accepted_runtime_cell", "arm": arm, "lifecycle_qualified": False})
            else:
                spec = specs[arm, accepted[arm]]
                outputs.append(make_lifecycle(f, spec, build, profile))
                stages.append(f.outcome(attempt, spec.name + "_largest_accepted_lifecycle"))
        f.put(attempt + "/completion.json", {"completed": True, "outputs": outputs, "outcomes": stages})
        f.inputs["profiles"].append({"activation_checkpointing": ac, "matrix_plan": matrix_path, "attempt": attempt})
    gates = {
        "relative_supervised_token_nll_one_sided_95_upper_max": 0.02,
        "accuracy_loss_one_sided_95_upper_max": 0.01,
        "paired_training_seeds": list(select.SEEDS),
    }
    development = {
        "protocol": "cosmos.qlora-development-plan.v1",
        "fixed_gates": gates,
        "development_manifest_sha256": "a" * 64,
        "development_data_sha256": "b" * 64,
        "fresh_training_data_sha256": "c" * 64,
    }
    dev_sha = f.put("plans/development.json", development)
    candidates = [
        {
            "arm": arm,
            "candidate": "rank32" if arm == "nf4" else "baseline",
            "state": "completed",
            "eligible": True,
            "contract_sha256": "d" * 64,
            "hyperparameters": {
                "rank": 32 if arm == "nf4" else 16,
                "alpha": 64.0 if arm == "nf4" else 32.0,
                "learning_rate": 1e-4,
            },
        }
        for arm in select.ARMS
    ]
    decision = {
        "protocol": "cosmos.qlora-development-selection.v1",
        "complete": True,
        "development_plan_sha256": dev_sha,
        "development_manifest_sha256": development["development_manifest_sha256"],
        "confirmation_seeds_required": list(select.SEEDS),
        "fixed_release_gates": gates,
        "development_rows": 584,
        "development_groups": 12,
        "candidates": candidates,
        "arm_status": {arm: "development_candidate_selected" for arm in select.ARMS},
        "selected": {
            row["arm"]: {key: row[key] for key in ("candidate", "contract_sha256", "hyperparameters")}
            | {"evidence_sha256": digest_json(row)}
            for row in candidates
        },
    }
    for arm in select.ARMS:
        for candidate in ("lr_5e5", "baseline", "lr_2e4", "rank32"):
            if not any(row["arm"] == arm and row["candidate"] == candidate for row in decision["candidates"]):
                decision["candidates"].append(
                    {"arm": arm, "candidate": candidate, "state": "completed", "eligible": False}
                )
    f.put("plans/development-selection.json", decision)
    confirmation = {
        "protocol": "cosmos.qlora-confirmation-and-best-fit-plan.v2",
        "development_plan_sha256": dev_sha,
        "paired_seeds": list(select.SEEDS),
        "runtime_matrix_plans": matrices,
        "best_fit": {
            "steps": 300,
            "warmup_steps": 30,
            "effective_batch": 8,
            "execution_selection_contract": {
                "activation_checkpointing": ["full", "none"],
                "microbatches": [1, 2, 4],
                "profile": "short_video8",
                "timed_updates_inclusive": [3, 20],
                "primary_metric": "completion.native.windows.after_warmup.seconds.mean",
                "fixed_model_execution": {
                    "adapter_compute_dtype": "float32",
                    "logits_mode": "full",
                    "compile_language_model": False,
                },
                "sort_keys": [
                    "unrounded after-warmup mean seconds ascending",
                    "training_budget.observed_process_peak_lower_bound_bytes ascending",
                    "microbatch ascending",
                    "activation_checkpointing full before none",
                ],
            },
        },
    }
    f.put("plans/confirmation.json", confirmation)
    f.inputs["plans"] = {
        "confirmation_plan": "plans/confirmation.json",
        "development_plan": "plans/development.json",
        "development_selection": "plans/development-selection.json",
    }
    return f


def test_complete_raw_evidence_writes_consumer_compatible_six_specs(tmp_path, monkeypatch):
    f = fixture(tmp_path / "evidence", monkeypatch)
    input_path = tmp_path / "inputs.json"
    input_path.write_text(json.dumps(f.inputs))
    sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output = tmp_path / "selection"
    report = select.publish(input_path, sha, f.root, output)
    assert report["complete"], report
    assert len(report["cells"]) == 12
    assert all(
        row["microbatch"] == 4 and row["activation_checkpointing"] == "none" for row in report["selected"].values()
    )
    receipt = json.loads((output / "execution-selection.json").read_text())
    assert len(receipt["confirmation_specs"]) == 6
    specs = [Experiment.read(row["path"]) for row in receipt["confirmation_specs"]]
    assert {(s.arm, s.seed) for s in specs} == {(arm, seed) for arm in select.ARMS for seed in select.SEEDS}
    assert {s.rank for s in specs if s.arm == "nf4"} == {32}
    assert all(s.steps == 300 and s.warmup_steps == 30 and s.effective_batch == 8 for s in specs)
    assert not list(output.glob("*provisional*"))
    assert str(tmp_path) not in json.dumps(report) and "private-allocation" not in json.dumps(report)


def test_overbudget_is_retained_and_only_required_escalation_skips_are_valid(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, overbudget=("none", "bf16", 2))
    report, _ = select.analyze(f.evidence())
    assert report["complete"], report
    rows = [r for r in report["cells"] if r["arm"] == "bf16" and r["activation_checkpointing"] == "none"]
    assert rows[1]["state"] == "completed" and not rows[1]["eligible"]
    assert rows[2]["state"] == "skipped_after_ineligible_predecessor"
    assert report["selected"]["bf16"]["activation_checkpointing"] == "full"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "skipped_feasible",
        "false_budget",
        "timing",
        "uuid",
        "dtype",
        "lifecycle",
        "tolerance",
        "partial",
        "workflow",
    ],
)
def test_negative_evidence_never_selects(tmp_path, monkeypatch, damage):
    f = fixture(tmp_path, monkeypatch)
    cell = "cells/runtime_none_nf4_b1"
    if damage == "missing":
        del f.inputs["files"][cell + "/completion.json"]
    elif damage == "skipped_feasible":
        p = "attempts/none/completion.json"
        v = f.read(p)
        v["outputs"][0] = {
            "kind": "runtime_skipped",
            "arm": "nf4",
            "microbatch": 1,
            "reason": "previous_cell_not_eligible",
        }
        f.put(p, v)
    elif damage in ("false_budget", "timing"):
        p = cell + "/completion.json"
        v = f.read(p)
        if damage == "false_budget":
            v["training_budget"]["observed_process_peak_lower_bound_bytes"] = 1
        else:
            v["native"]["windows"]["after_warmup"]["seconds"]["mean"] = 0.1
        sha = f.put(p, v)
        p = "attempts/none/completion.json"
        v = f.read(p)
        v["outputs"][0]["completion_sha256"] = sha
        f.put(p, v)
    elif damage == "uuid":
        p = cell + "/runtime.json"
        v = f.read(p)
        v["uuid"] = "GPU-different"
        f.put(p, v)
    elif damage == "dtype":
        p = "specs/runtime_none_nf4_b1.json"
        v = f.read(p)
        v["adapter_compute_dtype"] = "bfloat16"
        f.put(p, v)
    elif damage == "lifecycle":
        del f.inputs["files"]["lives/runtime_none_nf4_b4/completion.json"]
    elif damage == "tolerance":
        p = cell + "/stages/reload_exact/command.json"
        v = f.read(p)
        v[-1] = "0.1"
        f.put(p, v)
    elif damage == "partial":
        p = "lives/runtime_none_nf4_b4/accumulation-interruption.json"
        v = f.read(p)
        v["live_state"]["partial_checkpoint_rejected"] = False
        f.put(p, v)
    else:
        p = cell + "/contract.json"
        v = f.read(p)
        v["tools"]["runtime.py"] = "9" * 64
        f.put(p, v)
    report, decision = select.analyze(f.evidence())
    assert not report["complete"] and decision is None
    assert len(report["cells"]) == 12
    assert report["errors"]


def test_exact_tie_order_and_unresolved_duplicate():
    base = {
        "eligible": True,
        "after_warmup_mean_seconds": 2.0,
        "training_budget": {"observed_process_peak_lower_bound_bytes": 10},
        "microbatch": 2,
        "activation_checkpointing": "none",
    }
    full = {**base, "activation_checkpointing": "full"}
    smaller = {**base, "microbatch": 1}
    lower_peak = {**base, "training_budget": {"observed_process_peak_lower_bound_bytes": 9}}
    faster = {**base, "after_warmup_mean_seconds": 1.99999999999999}
    assert select.choose([base, full]) == full
    assert select.choose([full, smaller]) == smaller
    assert select.choose([smaller, lower_peak]) == lower_peak
    assert select.choose([lower_peak, faster]) == faster
    with pytest.raises(ValueError, match="unresolved"):
        select.choose([base, copy.deepcopy(base)])


def test_mutation_after_analysis_refuses_final_receipt(tmp_path, monkeypatch):
    f = fixture(tmp_path / "evidence", monkeypatch)
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(f.inputs))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    original = select.analyze

    def mutate(evidence):
        result = original(evidence)
        p = f.root / "plans/development-selection.json"
        p.write_text(p.read_text() + "\n")
        return result

    monkeypatch.setattr(select, "analyze", mutate)
    with pytest.raises(ValueError, match="changed before publication"):
        select.publish(path, sha, f.root, tmp_path / "output")
    assert not (tmp_path / "output/execution-selection.json").exists()


@pytest.mark.parametrize(
    "damage",
    ["self_comparison", "numerical", "missing_modality", "nf4_count", "nonfinite", "no_conditioning", "wrong_source"],
)
def test_raw_recovery_proofs_are_required(tmp_path, monkeypatch, damage):
    f = fixture(tmp_path, monkeypatch)
    root = "lives/runtime_none_nf4_b4"
    if damage == "self_comparison":
        path = root + "/stages/accumulation_compare_3/command.json"
        value = f.read(path)
        value[value.index("--candidate") + 1] = value[value.index("--reference") + 1]
        f.put(path, value)
    elif damage == "numerical":
        path = root + "/native/qualification/recovery/accumulation_resumed/canary/metrics.jsonl"
        rows = [json.loads(line) for line in (f.root / path).read_text().splitlines()]
        rows[0]["train/loss_avg"] += 0.1
        f.put(path, rows, lines=True)
    elif damage == "wrong_source":
        path = root + "/accumulation-interruption.json"
        value = f.read(path)
        value["checkpoint"]["path"] = "/recorded/other/checkpoints/iter_000002"
        f.put(path, value)
    else:
        path = root + "/probes/dcp/probe.json"
        value = f.read(path)
        if damage == "missing_modality":
            del value["probes"]["image"]
        elif damage == "nf4_count":
            value["quantized_matrices"] = 0
        elif damage == "nonfinite":
            value["probes"]["video"]["loss"] = float("nan")
        else:
            value["probes"]["video"]["zero_media_max_logit_change"] = 0
        f.put(path, value)
    report, _ = select.analyze(f.evidence())
    assert report["complete"] is False


def test_explicit_recorded_alias_only(tmp_path):
    f = Fixture(tmp_path)
    f.inputs["recorded_root_aliases"] = ["/allowed-alias"]
    evidence = f.evidence()
    assert evidence.relative("/allowed-alias/own/file.json") == "own/file.json"
    with pytest.raises(ValueError, match="escapes"):
        evidence.relative("/other-alias/own/file.json")
    with pytest.raises(ValueError, match="escapes"):
        evidence.relative("../escape.json")


@pytest.mark.parametrize("damage", ["float_step", "missing_step", "skip", "nonfinite", "tokens", "finite_check"])
def test_native_raw_metrics_fail_without_summary_trust(damage):
    rows, expected = native_rows(2.0)
    row = rows[1]
    if damage == "float_step":
        row["step"] = 1.0
    elif damage == "missing_step":
        rows.pop(1)
    elif damage == "skip":
        row["optim/update_outcome"] = "skipped"
    elif damage == "nonfinite":
        row["timer/e2e_step_s"] = float("inf")
    elif damage == "tokens":
        row["train/valid_token_count"] = 17
    else:
        row["optim/finite_gradient_check_enabled"] = 0
    with pytest.raises(ValueError):
        select.native_metrics(rows, expected)


def test_incomplete_attempt_emits_no_selection_receipt(tmp_path, monkeypatch):
    f = fixture(tmp_path / "evidence", monkeypatch)
    path = "attempts/none/completion.json"
    value = f.read(path)
    value["completed"] = False
    f.put(path, value)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(f.inputs))
    out = tmp_path / "output"
    report = select.publish(inputs, hashlib.sha256(inputs.read_bytes()).hexdigest(), f.root, out)
    assert report["complete"] is False and len(report["cells"]) == 12
    assert (out / "decision.json").exists()
    assert not (out / "execution-selection.json").exists()
    assert not list(out.glob("best_fit_*.json"))


def test_recomputed_helpers_match_existing_runtime_contract():
    from tools.native_qlora_qualification.tests.test_runtime import (
        experiment,
        memory_fixture,
        metric_records,
    )
    from tools.native_qlora_qualification.tests.test_runtime import (
        runtime as original,
    )

    memory = memory_fixture()
    expected = original.budget_eligibility(memory)
    assert all(expected[key] == value for key, value in select.budget(memory).items())
    rows, cursor = metric_records()
    assert select.native_metrics(rows, cursor) == original.validate_metrics(rows, cursor, experiment())


def test_selector_import_does_not_load_torch_or_create_model():
    import subprocess
    import sys

    command = "import sys; import tools.native_qlora_qualification.execution_selection; assert 'torch' not in sys.modules; assert 'cosmos_framework' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_timing_cannot_win_rank(value):
    with pytest.raises(ValueError, match="rank inputs"):
        select.choose(
            [
                {
                    "eligible": True,
                    "after_warmup_mean_seconds": value,
                    "training_budget": {"observed_process_peak_lower_bound_bytes": 10},
                    "microbatch": 1,
                    "activation_checkpointing": "full",
                }
            ]
        )


@pytest.mark.parametrize("damage", ["missing_candidate", "baseline_failed", "relaxed_gate"])
def test_development_provenance_cannot_omit_or_relax_candidates(tmp_path, monkeypatch, damage):
    f = fixture(tmp_path, monkeypatch)
    path = "plans/development-selection.json"
    value = f.read(path)
    if damage == "missing_candidate":
        value["candidates"].pop()
    elif damage == "baseline_failed":
        next(row for row in value["candidates"] if row["arm"] == "nf4" and row["candidate"] == "baseline")["state"] = (
            "failed"
        )
    else:
        value["fixed_release_gates"]["accuracy_loss_one_sided_95_upper_max"] = 0.1
    f.put(path, value)
    report, _ = select.analyze(f.evidence())
    assert report["complete"] is False


def test_ac_profiles_cannot_change_runtime_mathematics(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    for arm in select.ARMS:
        for micro in (1, 2, 4):
            path = f"cells/runtime_none_{arm}_b{micro}/runtime.json"
            value = f.read(path)
            value["runtime_precision"]["tf32"] = True
            f.put(path, value)
    report, _ = select.analyze(f.evidence())
    assert report["complete"] is False
    assert "full/none calibration changed software, mathematics or measured workload" in report["errors"]


@pytest.mark.parametrize("damage", ["different_checkpoint", "different_build", "different_step"])
def test_cell_reload_is_bound_to_own_final_model(tmp_path, monkeypatch, damage):
    f = fixture(tmp_path, monkeypatch)
    root = "cells/runtime_none_nf4_b1"
    if damage == "different_checkpoint":
        path = root + "/probes/portable/probe.json"
        value = f.read(path)
        value["checkpoint"] = "/recorded/other/iter_000020"
        f.put(path, value)
    else:
        path = root + "/final-checkpoint.json"
        value = f.read(path)
        if damage == "different_build":
            value["metadata"]["identity"]["training_contract"]["code_revision"] = "9" * 40
        else:
            value["optimizer_step"] = 19
        f.put(path, value)
    report, _ = select.analyze(f.evidence())
    assert report["complete"] is False


def explicit_profile_fixture(tmp_path, *, version=2):
    """Synthetic future producer: no private revision is implicitly trusted."""
    f = Fixture(tmp_path)
    f.inputs["protocol"] = {
        2: select.PROFILE_INPUT_PROTOCOL,
        3: select.PROFILE_INPUT_PROTOCOL_V3,
        4: select.PROFILE_INPUT_PROTOCOL_V4,
        5: select.PROFILE_INPUT_PROTOCOL_V5,
        6: select.PROFILE_INPUT_PROTOCOL_V6,
        7: select.PROFILE_INPUT_PROTOCOL_V7,
    }[version]
    workflow_root, runner_root = (
        "workflows/aaaaaaa",
        {
            2: "gpu-runtime-v6",
            3: "gpu-runtime-v7",
            4: "gpu-runtime-v8",
            5: "gpu-runtime-v9",
            6: "gpu-runtime-v10",
            7: "gpu-runtime-v11",
        }[version],
    )
    files = {}

    def source(name, contents):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        sha = hashlib.sha256(contents).hexdigest()
        f.inputs["files"][name] = sha
        return sha

    for name in ("monitor.py", "monitor_host.py", "run.py"):
        relative = "tools/native_qlora_qualification/" + name
        files[relative] = source(workflow_root + "/" + relative, b"# synthetic\n")
    archive = source(workflow_root + "/source.tar", b"synthetic archive")
    workflow = {
        "files": files,
        "code_revision": "a" * 40,
        "training_package_revision": select.TRAINING_REVISION,
        "training_wheel_sha256": select.WHEEL_SHA,
        "source_archive_sha256": archive,
    }
    workflow_sha = f.put(workflow_root + "/receipt.json", workflow)
    driver_sha = source(runner_root + "/driver.py", b"# reviewed synthetic driver\n")
    runner_sha = f.put(runner_root + "/receipt.json", {"files": {"driver.py": driver_sha}})
    correction_sha = f.put(
        "correction.json",
        {
            "protocol": "cosmos.qlora-telemetry-correction-plan.v1",
            "preserve": {
                "native_training_package": select.TRAINING_REVISION,
                "all_quality_and_memory_thresholds": "unchanged",
            },
        },
    )
    profile = {
        "protocol": select.PRODUCER_PROFILE_PROTOCOL,
        "training": {"code_revision": select.TRAINING_REVISION, "wheel_sha256": select.WHEEL_SHA},
        "workflow": {
            "root": workflow_root,
            "revision": "a" * 40,
            "receipt_sha256": workflow_sha,
            "source_archive_sha256": archive,
        },
        "runner": {"root": runner_root, "receipt_sha256": runner_sha, "driver_sha256": driver_sha},
        "monitor": {
            "method": {
                2: "proc-identity-state-read-attempts-v1",
                3: "proc-identity-state-read-attempts-v1",
                4: "proc-identity-state-read-attempts-v2",
                5: "proc-identity-state-read-attempts-v3",
                6: "proc-identity-state-read-attempts-v3",
                7: "proc-identity-state-read-attempts-v3",
            }[version],
            "files": {n: files["tools/native_qlora_qualification/" + n] for n in ("monitor.py", "monitor_host.py")},
        },
        "correction_plan": {"path": "correction.json", "sha256": correction_sha},
    }
    profile_sha = f.put("producer-profile.json", profile)
    binding = {"path": "producer-profile.json", "sha256": profile_sha}
    f.inputs["producer_profile"] = binding
    identity = {
        "driver_sha256": driver_sha,
        "workflow_receipt_sha256": workflow_sha,
        "plan": {
            "producer_profile": binding,
            "workflow_receipt_sha256": workflow_sha,
            "inputs": [{"path": p, "sha256": s} for p, s in f.inputs["files"].items()],
        },
    }
    return f, identity


def test_explicit_profile_requires_exact_presealed_source_inventory(tmp_path):
    f, identity = explicit_profile_fixture(tmp_path)
    workflow, profile = select.producer_profile(f.evidence(), identity)
    assert workflow["code_revision"] == "a" * 40
    assert profile["profile_sha256"] == f.inputs["producer_profile"]["sha256"]
    evidence = f.evidence()
    select.producer_profile(evidence, identity)
    (tmp_path / "workflows/aaaaaaa/tools/native_qlora_qualification/monitor.py").write_text("# changed")
    with pytest.raises(select.InvalidExecutionSelection, match="changed"):
        evidence.stable()


@pytest.mark.parametrize(
    "change",
    [
        "unbound_profile",
        "different_plan_profile",
        "driver",
        "workflow",
        "missing_helper",
        "extra_python",
        "source_archive",
        "unknown_method",
    ],
)
def test_explicit_profile_rejects_mismatched_or_unsealed_producer(tmp_path, change):
    f, identity = explicit_profile_fixture(tmp_path)
    if change == "unbound_profile":
        identity["plan"]["inputs"] = [r for r in identity["plan"]["inputs"] if r["path"] != "producer-profile.json"]
    elif change == "different_plan_profile":
        identity["plan"]["producer_profile"] = {"path": "producer-profile.json", "sha256": "b" * 64}
    elif change in ("driver", "workflow"):
        identity[change + ("_sha256" if change == "driver" else "_receipt_sha256")] = "b" * 64
    elif change == "extra_python":
        (tmp_path / "gpu-runtime-v6/extra.py").write_text("# unreviewed")
    elif change == "source_archive":
        (tmp_path / "workflows/aaaaaaa/source.tar").write_bytes(b"changed")
    else:
        profile = f.read("producer-profile.json")
        if change == "unknown_method":
            profile["monitor"]["method"] = "best_effort"
        else:
            profile["monitor"]["files"].pop("monitor_host.py")
        sha = f.put("producer-profile.json", profile)
        f.inputs["producer_profile"]["sha256"] = sha
        identity["plan"]["inputs"] = [{"path": p, "sha256": s} for p, s in f.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection):
        select.producer_profile(f.evidence(), identity)


def test_legacy_input_cannot_opt_in_to_future_profile(tmp_path):
    f, identity = explicit_profile_fixture(tmp_path)
    f.inputs["protocol"] = select.INPUT_PROTOCOL
    with pytest.raises(select.InvalidExecutionSelection, match="versioned"):
        f.evidence()
    del f.inputs["producer_profile"]
    with pytest.raises(select.InvalidExecutionSelection, match="pinned v5"):
        select.producer_profile(f.evidence(), identity)


def test_explicit_profile_late_python_inventory_injection_is_rejected(tmp_path):
    f, identity = explicit_profile_fixture(tmp_path)
    evidence = f.evidence()
    select.producer_profile(evidence, identity)
    (tmp_path / "gpu-runtime-v6/late.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


def test_explicit_profile_unrelated_file_is_not_strict_monitor_helper(tmp_path):
    f, identity = explicit_profile_fixture(tmp_path)
    profile = f.read("producer-profile.json")
    profile["monitor"]["files"]["run.py"] = profile["monitor"]["files"].pop("monitor_host.py")
    sha = f.put("producer-profile.json", profile)
    f.inputs["producer_profile"]["sha256"] = sha
    identity["plan"]["inputs"] = [{"path": p, "sha256": s} for p, s in f.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection, match="strict host-monitor"):
        select.producer_profile(f.evidence(), identity)


def test_v3_profile_binds_exact_v7_without_changing_v2(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=3)
    workflow, result = select.producer_profile(fixture.evidence(), identity)
    assert workflow["code_revision"] == "a" * 40
    assert result["input_protocol"] == select.PROFILE_INPUT_PROTOCOL_V3
    assert result["runner"]["root"] == "gpu-runtime-v7"
    fixture.inputs["protocol"] = select.PROFILE_INPUT_PROTOCOL
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


def test_v3_cannot_reuse_v6_profile_even_with_matching_inventory(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path)
    fixture.inputs["protocol"] = select.PROFILE_INPUT_PROTOCOL_V3
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


def test_v3_rechecks_its_v7_inventory_at_publication(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=3)
    evidence = fixture.evidence()
    select.producer_profile(evidence, identity)
    (tmp_path / "gpu-runtime-v7/extra.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


def test_v4_profile_binds_exact_v8_and_exit_proof_monitor(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=4)
    evidence = fixture.evidence()
    _, result = select.producer_profile(evidence, identity)
    assert result["input_protocol"] == select.PROFILE_INPUT_PROTOCOL_V4
    assert result["runner"]["root"] == "gpu-runtime-v8"
    assert result["monitor"]["method"] == "proc-identity-state-read-attempts-v2"
    evidence.stable()
    (tmp_path / "gpu-runtime-v8/extra.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


@pytest.mark.parametrize("source_version,target_version", [(2, 4), (3, 4), (4, 2), (4, 3)])
def test_v4_cannot_reinterpret_legacy_runner_profiles(tmp_path, source_version, target_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=source_version)
    fixture.inputs["protocol"] = {
        2: select.PROFILE_INPUT_PROTOCOL,
        3: select.PROFILE_INPUT_PROTOCOL_V3,
        4: select.PROFILE_INPUT_PROTOCOL_V4,
    }[target_version]
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


@pytest.mark.parametrize("version", [2, 3, 4])
def test_profile_monitor_version_cannot_upgrade_or_downgrade_history(tmp_path, version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=version)
    profile = fixture.read("producer-profile.json")
    profile["monitor"]["method"] = (
        "proc-identity-state-read-attempts-v1" if version == 4 else "proc-identity-state-read-attempts-v2"
    )
    sha = fixture.put("producer-profile.json", profile)
    fixture.inputs["producer_profile"]["sha256"] = sha
    identity["plan"]["inputs"] = [{"path": path, "sha256": value} for path, value in fixture.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection, match="strict host-monitor"):
        select.producer_profile(fixture.evidence(), identity)


def test_v4_requires_profile_presealed_by_submitted_calibration(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=4)
    identity["plan"]["inputs"] = [row for row in identity["plan"]["inputs"] if row["path"] != "producer-profile.json"]
    with pytest.raises(select.InvalidExecutionSelection, match="did not preseal"):
        select.producer_profile(fixture.evidence(), identity)


def test_v5_profile_binds_exact_v9_and_metadata_owner_monitor(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=5)
    evidence = fixture.evidence()
    _, result = select.producer_profile(evidence, identity)
    assert result["input_protocol"] == select.PROFILE_INPUT_PROTOCOL_V5
    assert result["runner"]["root"] == "gpu-runtime-v9"
    assert result["monitor"]["method"] == "proc-identity-state-read-attempts-v3"
    evidence.stable()
    (tmp_path / "gpu-runtime-v9/extra.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


@pytest.mark.parametrize("source_version,target_version", [(2, 5), (3, 5), (4, 5), (5, 2), (5, 3), (5, 4)])
def test_v5_cannot_reinterpret_legacy_runner_profiles(tmp_path, source_version, target_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=source_version)
    fixture.inputs["protocol"] = {
        2: select.PROFILE_INPUT_PROTOCOL,
        3: select.PROFILE_INPUT_PROTOCOL_V3,
        4: select.PROFILE_INPUT_PROTOCOL_V4,
        5: select.PROFILE_INPUT_PROTOCOL_V5,
    }[target_version]
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


@pytest.mark.parametrize("version,method_version", [(2, 3), (3, 3), (4, 3), (5, 1), (5, 2)])
def test_v3_monitor_cannot_upgrade_or_downgrade_prior_input_contracts(tmp_path, version, method_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=version)
    profile = fixture.read("producer-profile.json")
    profile["monitor"]["method"] = f"proc-identity-state-read-attempts-v{method_version}"
    sha = fixture.put("producer-profile.json", profile)
    fixture.inputs["producer_profile"]["sha256"] = sha
    identity["plan"]["inputs"] = [{"path": path, "sha256": value} for path, value in fixture.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection, match="strict host-monitor"):
        select.producer_profile(fixture.evidence(), identity)


def test_v5_requires_profile_presealed_by_submitted_calibration(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=5)
    identity["plan"]["inputs"] = [row for row in identity["plan"]["inputs"] if row["path"] != "producer-profile.json"]
    with pytest.raises(select.InvalidExecutionSelection, match="did not preseal"):
        select.producer_profile(fixture.evidence(), identity)


def test_v6_profile_binds_exact_v10_with_same_strict_v3_raw_semantics(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=6)
    evidence = fixture.evidence()
    _, result = select.producer_profile(evidence, identity)
    assert result["input_protocol"] == select.PROFILE_INPUT_PROTOCOL_V6
    assert result["runner"]["root"] == "gpu-runtime-v10"
    assert result["monitor"]["method"] == "proc-identity-state-read-attempts-v3"
    evidence.stable()
    (tmp_path / "gpu-runtime-v10/extra.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


@pytest.mark.parametrize(
    "source_version,target_version", [(2, 6), (3, 6), (4, 6), (5, 6), (6, 2), (6, 3), (6, 4), (6, 5)]
)
def test_v6_cannot_relabel_previous_runners_even_with_same_v3_method(tmp_path, source_version, target_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=source_version)
    fixture.inputs["protocol"] = {
        2: select.PROFILE_INPUT_PROTOCOL,
        3: select.PROFILE_INPUT_PROTOCOL_V3,
        4: select.PROFILE_INPUT_PROTOCOL_V4,
        5: select.PROFILE_INPUT_PROTOCOL_V5,
        6: select.PROFILE_INPUT_PROTOCOL_V6,
    }[target_version]
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


@pytest.mark.parametrize("method_version", [1, 2, 4])
def test_v6_does_not_allow_other_raw_monitor_semantics(tmp_path, method_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=6)
    profile = fixture.read("producer-profile.json")
    profile["monitor"]["method"] = f"proc-identity-state-read-attempts-v{method_version}"
    sha = fixture.put("producer-profile.json", profile)
    fixture.inputs["producer_profile"]["sha256"] = sha
    identity["plan"]["inputs"] = [{"path": path, "sha256": value} for path, value in fixture.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection, match="strict host-monitor"):
        select.producer_profile(fixture.evidence(), identity)


def test_v6_requires_profile_presealed_by_submitted_calibration(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=6)
    identity["plan"]["inputs"] = [row for row in identity["plan"]["inputs"] if row["path"] != "producer-profile.json"]
    with pytest.raises(select.InvalidExecutionSelection, match="did not preseal"):
        select.producer_profile(fixture.evidence(), identity)


def test_v6_profile_cannot_backfill_the_unversioned_legacy_contract(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=6)
    fixture.inputs["protocol"] = select.INPUT_PROTOCOL
    with pytest.raises(select.InvalidExecutionSelection, match="versioned"):
        fixture.evidence()
    del fixture.inputs["producer_profile"]
    with pytest.raises(select.InvalidExecutionSelection, match="pinned v5"):
        select.producer_profile(fixture.evidence(), identity)


def test_v7_profile_binds_exact_v11_with_same_strict_v3_raw_semantics(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=7)
    evidence = fixture.evidence()
    _, result = select.producer_profile(evidence, identity)
    assert result["input_protocol"] == select.PROFILE_INPUT_PROTOCOL_V7
    assert result["runner"]["root"] == "gpu-runtime-v11"
    assert result["monitor"]["method"] == "proc-identity-state-read-attempts-v3"
    evidence.stable()
    (tmp_path / "gpu-runtime-v11/extra.py").write_text("# late unbound source")
    with pytest.raises(select.InvalidExecutionSelection, match="inventory changed"):
        evidence.stable()


@pytest.mark.parametrize(
    "source_version,target_version", [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)]
)
def test_v7_cannot_relabel_previous_runners_even_with_same_v3_method(tmp_path, source_version, target_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=source_version)
    fixture.inputs["protocol"] = {
        2: select.PROFILE_INPUT_PROTOCOL,
        3: select.PROFILE_INPUT_PROTOCOL_V3,
        4: select.PROFILE_INPUT_PROTOCOL_V4,
        5: select.PROFILE_INPUT_PROTOCOL_V5,
        6: select.PROFILE_INPUT_PROTOCOL_V6,
        7: select.PROFILE_INPUT_PROTOCOL_V7,
    }[target_version]
    with pytest.raises(select.InvalidExecutionSelection, match="observed execution"):
        select.producer_profile(fixture.evidence(), identity)


@pytest.mark.parametrize("method_version", [1, 2, 4])
def test_v7_does_not_allow_other_raw_monitor_semantics(tmp_path, method_version):
    fixture, identity = explicit_profile_fixture(tmp_path, version=7)
    profile = fixture.read("producer-profile.json")
    profile["monitor"]["method"] = f"proc-identity-state-read-attempts-v{method_version}"
    sha = fixture.put("producer-profile.json", profile)
    fixture.inputs["producer_profile"]["sha256"] = sha
    identity["plan"]["inputs"] = [{"path": path, "sha256": value} for path, value in fixture.inputs["files"].items()]
    with pytest.raises(select.InvalidExecutionSelection, match="strict host-monitor"):
        select.producer_profile(fixture.evidence(), identity)


def test_v7_requires_profile_presealed_by_submitted_calibration(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=7)
    identity["plan"]["inputs"] = [row for row in identity["plan"]["inputs"] if row["path"] != "producer-profile.json"]
    with pytest.raises(select.InvalidExecutionSelection, match="did not preseal"):
        select.producer_profile(fixture.evidence(), identity)


def test_v7_profile_cannot_backfill_the_unversioned_legacy_contract(tmp_path):
    fixture, identity = explicit_profile_fixture(tmp_path, version=7)
    fixture.inputs["protocol"] = select.INPUT_PROTOCOL
    with pytest.raises(select.InvalidExecutionSelection, match="versioned"):
        fixture.evidence()
    del fixture.inputs["producer_profile"]
    with pytest.raises(select.InvalidExecutionSelection, match="pinned v5"):
        select.producer_profile(fixture.evidence(), identity)


@pytest.mark.parametrize("source", ["contract", "completion"])
def test_diagnostic_native_trace_cannot_enter_execution_selection(tmp_path, monkeypatch, source):
    f = fixture(tmp_path, monkeypatch)
    root = "cells/runtime_none_nf4_b1"
    path = root + "/" + source + ".json"
    value = f.read(path)
    value["diagnostic_profile"] = "early"
    f.put(path, value)
    completion_path = root + "/completion.json"
    completed = f.read(completion_path)
    with pytest.raises(ValueError, match="diagnostic traces"):
        select.cell(
            f.evidence(),
            {"path": root, "completion_sha256": f.inputs["files"][completion_path]},
            Experiment(**completed["experiment"]),
            "unused-before-profile-rejection",
            plan={},
            build={},
            allocation="123",
            profile={},
            tools={},
        )
