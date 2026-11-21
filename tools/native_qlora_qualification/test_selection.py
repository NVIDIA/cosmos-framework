# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sealed-selection contract fixtures, not new training or quality evidence."""

import json
from pathlib import Path

import pytest

from tools.native_qlora_qualification import selection
from tools.native_qlora_qualification.common import digest_file, digest_json
from tools.native_qlora_qualification.protocol import Experiment
from tools.native_qlora_qualification.score import aggregate
from tools.native_qlora_qualification.test_performance import experiment_root, write

# Match the original sealed JSON specs independently of the selector constants.
# In particular, alpha was serialized as an integer in every native override.
DECLARED_CANDIDATES = {
    "lr_5e5": (16, 32, 5e-5),
    "baseline": (16, 32, 1e-4),
    "lr_2e4": (16, 32, 2e-4),
    "rank32": (32, 64, 1e-4),
}


def score_rows(ce):
    rows = []
    for i in range(584):
        identifier = f"20230929_{i % 12}_SN22_T1_Camera1_0_{i}"
        inputs = {"input_ids": {"sha256": digest_json(identifier)}}
        correct = i < 570
        rows.append(
            {
                "id": identifier,
                "source_line": i + 1,
                "raw_row_sha256": digest_json(identifier),
                "row_identity": digest_json(identifier),
                "gold": "B",
                "choices": ["A", "B"],
                "media_sha256": digest_json(str(i % 12)),
                "input_sha256": digest_json(inputs),
                "inputs": inputs,
                "candidate_token_ids": {"A": 1, "B": 2},
                "answer_token_position": 3,
                "prediction_logit_position": 2,
                "native_supervised_tokens": 2,
                "sequence_length": 5,
                "native_sft_ce": ce,
                "answer_nll": 0.5,
                "restricted_choice_nll": 0.4,
                "candidate_logits": {"A": 1.0 if correct else 2.0, "B": 2.0 if correct else 1.0},
                "restricted_choice_prediction": "B" if correct else "A",
                "restricted_choice_correct": correct,
                "strict_first_token_id": 2 if correct else 1,
                "strict_first_token_correct": correct,
                "valid_first_letter": True,
                "restricted_choice_tied_max": False,
                "strict_first_token_tied_max": False,
            }
        )
    return rows


def write_scores(root, spec, contract, manifest, manifest_sha, rows):
    selected = {
        "split_role": "development",
        "split_manifest_sha256": manifest_sha,
        "selected_rows": 584,
        "source_rows": 584,
        "group_count": 12,
        "source_sha256": manifest["data_sha256"],
        "ordered_ids": [r["id"] for r in rows],
        "ordered_row_identities": [r["row_identity"] for r in rows],
        "selected_content_sha256": digest_json(
            [{key: r[key] for key in ("id", "row_identity", "gold", "choices", "media_sha256")} for r in rows]
        ),
    }
    final = json.loads((root / "final-checkpoint.json").read_text())
    identity = {key: value for key, value in final["metadata"]["identity"].items() if key != "training_contract"}
    write(root / "stages/scoring/command.json", ["python", "score.py", "--checkpoint", final["checkpoint"]])
    adapters = {f"tensor_{i}": {"sha256": digest_json([spec.name, i])} for i in range(504)}
    write(
        root / "probes/dcp/probe.json",
        {
            "protocol": "cosmos.qlora-integration-probe.v1",
            "checkpoint": final["checkpoint"],
            "identity_sha256": digest_json(identity),
            "adapters": adapters,
            "trainable_tensors": 504,
            "optimizer_parameters": 504,
            "adapter_updates": {"A": 252, "B": 252},
        },
    )
    write(root / "scores/scores.jsonl", rows, lines=True)
    write(root / "scores/selection.json", selected)
    write(root / "scores/model_identity.json", identity)
    write(
        root / "scores/summary.json",
        {
            "protocol": "cosmos.wts-heldout.v1",
            "complete": True,
            "adapter_state": "reloaded_checkpoint",
            "checkpoint": "/private/own-checkpoint",
            "recipe": spec.recipe,
            "overrides": spec.overrides(),
            "model_seed": 42,
            "world_size": 1,
            "max_length": 4096,
            "scores_sha256": digest_file(root / "scores/scores.jsonl"),
            "selection_sha256": digest_json(selected),
            "model_identity_sha256": digest_json(identity),
            "adapter_tensors_sha256": digest_json(adapters),
            "build_receipt": {
                "COSMOS_TRAINING_CODE_REVISION": contract["build"]["code_revision"],
                "COSMOS_TRAINING_WHEEL_SHA256": contract["build"]["wheel_sha256"],
            },
            "selected_content_sha256": selected["selected_content_sha256"],
            "processor_contract": identity["processor_contract"],
            "quantized_matrices": 252 if spec.arm == "nf4" else 0,
            "metrics": aggregate(rows),
        },
    )


@pytest.fixture
def sealed(tmp_path):
    initial = score_rows(1.0)
    manifest = {
        "protocol": "cosmos.wts-group-split.v1",
        "role": "development",
        "data_sha256": "c" * 64,
        "rows": [
            {
                "id": r["id"],
                "raw_row_sha256": r["raw_row_sha256"],
                "group_sha256": digest_json(str(i % 12)),
                "media_sha256": r["media_sha256"],
            }
            for i, r in enumerate(initial)
        ],
        "group_count": 12,
    }
    manifest_path = tmp_path / "development.manifest.json"
    write(manifest_path, manifest)
    manifest_sha = digest_file(manifest_path)
    plan = {
        "protocol": "cosmos.qlora-development-plan.v1",
        "selection_rule": selection.RULE,
        "fixed_gates": selection.GATES,
        "training_seeds": [42],
        "paired_confirmation_seeds": [17, 42, 123],
        "fresh_base_required": True,
        "failed_candidates_retained": True,
        "initialization_seed": "equal training seed in both arms",
        "development_manifest_sha256": manifest_sha,
        "development_data_sha256": manifest["data_sha256"],
        "fresh_training_data_sha256": "d" * 64,
        "candidates": [],
    }
    runs = []
    for repeat, (candidate, (rank, alpha, lr)) in enumerate(DECLARED_CANDIDATES.items(), 1):
        entry = {"candidate": candidate, "ordered_runs": []}
        for arm in ("nf4", "bf16"):
            root = experiment_root(tmp_path / "runs", arm, repeat)
            spec = Experiment(f"dev_{candidate}_{arm}_seed42", arm, rank=rank, alpha=alpha, learning_rate=lr)
            spec_path = tmp_path / (spec.name + ".json")
            write(spec_path, spec.as_dict())
            entry["ordered_runs"].append({"name": spec.name, "file": spec_path.name, "sha256": digest_file(spec_path)})
            contract = json.loads((root / "contract.json").read_text())
            contract.update(experiment=spec.as_dict(), overrides=spec.overrides(), split="development", checkpoint=None)
            for index, sha in {
                0: digest_file(spec_path),
                2: manifest_sha,
                3: manifest["data_sha256"],
                4: plan["fresh_training_data_sha256"],
                5: manifest["data_sha256"],
            }.items():
                contract["files"][index]["sha256"] = sha
            write(root / "contract.json", contract)
            final = json.loads((root / "final-checkpoint.json").read_text())
            final["checkpoint"] = "/private/own-checkpoint"
            final["metadata"]["identity"]["adapter"].update(rank=rank, alpha=alpha)
            write(root / "final-checkpoint.json", final)
            completion = json.loads((root / "completion.json").read_text())
            completion["experiment"] = spec.as_dict()
            write(root / "completion.json", completion)
            # Both arms are in-budget in these synthetic fixtures; an explicit test
            # restores an over-budget BF16 control and verifies it stays visible.
            samples = [json.loads(line) for line in (root / "nvml-host.jsonl").read_text().splitlines()]
            for sample in samples:
                sample["device_compute_processes"][0]["used_gpu_memory_bytes"] = 20 * selection.GIB
            write(root / "nvml-host.jsonl", samples, lines=True)
            rows = score_rows({"lr_5e5": 0.9, "baseline": 1.0, "lr_2e4": 1.1, "rank32": 0.8}[candidate])
            write_scores(root, spec, contract, manifest, manifest_sha, rows)
            runs.append(
                {
                    "candidate": candidate,
                    "arm": arm,
                    "path": str(root),
                    "contract_sha256": digest_file(root / "contract.json"),
                }
            )
        plan["candidates"].append(entry)
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    return {
        "plan_path": plan_path,
        "plan_sha256": digest_file(plan_path),
        "development_manifest_path": manifest_path,
        "development_manifest_sha256": manifest_sha,
        "runs": runs,
    }


def item(sealed, candidate="rank32", arm="nf4"):
    return next(r for r in sealed["runs"] if r["candidate"] == candidate and r["arm"] == arm)


def alter_scores(sealed, transform, candidate="rank32", arm="nf4"):
    root = Path(item(sealed, candidate, arm)["path"])
    contract = json.loads((root / "contract.json").read_text())
    rows = [json.loads(line) for line in (root / "scores/scores.jsonl").read_text().splitlines()]
    transform(rows)
    manifest = json.loads(sealed["development_manifest_path"].read_text())
    write_scores(
        root, Experiment(**contract["experiment"]), contract, manifest, sealed["development_manifest_sha256"], rows
    )


def test_complete_eight_candidate_selection_has_no_untouched_scoring_authority(sealed):
    report = selection.select_development(**sealed)
    assert report["complete"], report
    assert {arm: row["candidate"] for arm, row in report["selected"].items()} == {"nf4": "rank32", "bf16": "rank32"}
    assert report["confirmation_seeds_required"] == [17, 42, 123]
    assert report["untouched_scoring_authorized"] is False
    assert report["confirmation_scoring_configurations_locked"] is False
    assert "selected_scoring_config_sha256" not in report
    assert report["protocol"] != "cosmos.qlora-selection.v1"
    assert "/private/" not in json.dumps(report)
    assert str(sealed["plan_path"].parent) not in json.dumps(report)


def test_original_integer_alpha_overrides_admit_all_eight_declared_candidates(sealed):
    for row in sealed["runs"]:
        contract = json.loads((Path(row["path"]) / "contract.json").read_text())
        alpha = 64 if row["candidate"] == "rank32" else 32
        assert type(contract["experiment"]["alpha"]) is int
        assert f"model.model_config.adapter.alpha={alpha}" in contract["overrides"]
    report = selection.select_development(**sealed)
    assert report["complete"] is True
    assert len(report["candidates"]) == 8
    assert all(row["state"] == "completed" for row in report["candidates"])


@pytest.mark.parametrize("candidate", ["baseline", "rank32"])
@pytest.mark.parametrize("damage", ["changed_alpha", "float_override", "float_metadata_and_override"])
def test_alpha_correction_keeps_exact_predeclared_override_contract(sealed, candidate, damage):
    entry = item(sealed, candidate=candidate)
    path = Path(entry["path"]) / "contract.json"
    contract = json.loads(path.read_text())
    alpha = contract["experiment"]["alpha"]
    if damage == "changed_alpha":
        contract["experiment"]["alpha"] = alpha + 1
        replacement = str(alpha + 1)
    else:
        replacement = str(float(alpha))
        if damage == "float_metadata_and_override":
            contract["experiment"]["alpha"] = float(alpha)
    contract["overrides"] = [
        f"model.model_config.adapter.alpha={replacement}"
        if value.startswith("model.model_config.adapter.alpha=")
        else value
        for value in contract["overrides"]
    ]
    write(path, contract)
    entry["contract_sha256"] = digest_file(path)
    report = selection.select_development(**sealed)
    rejected = next(row for row in report["candidates"] if (row["candidate"], row["arm"]) == (candidate, "nf4"))
    assert rejected["state"] == "incomplete"
    assert rejected["eligible"] is False
    assert rejected["reasons"] == ["native experiment differs from the predeclared candidate"]


@pytest.mark.parametrize("damage", ["plan_sha", "gate", "regression", "groups", "rows", "duplicate_input"])
def test_plan_or_manifest_substitution_fails(sealed, damage):
    if damage == "plan_sha":
        sealed["plan_sha256"] = "0" * 64
    elif damage == "duplicate_input":
        sealed["runs"].append(sealed["runs"][0])
    elif damage == "gate":
        plan = json.loads(sealed["plan_path"].read_text())
        plan["fixed_gates"]["relative_supervised_token_nll_one_sided_95_upper_max"] = 0.03
        write(sealed["plan_path"], plan)
        sealed["plan_sha256"] = digest_file(sealed["plan_path"])
    else:
        manifest = json.loads(sealed["development_manifest_path"].read_text())
        if damage == "regression":
            manifest["role"] = "regression"
        elif damage == "groups":
            manifest["group_count"] = 1
        else:
            manifest["rows"].pop()
        write(sealed["development_manifest_path"], manifest)
        sealed["development_manifest_sha256"] = digest_file(sealed["development_manifest_path"])
        plan = json.loads(sealed["plan_path"].read_text())
        plan["development_manifest_sha256"] = sealed["development_manifest_sha256"]
        write(sealed["plan_path"], plan)
        sealed["plan_sha256"] = digest_file(sealed["plan_path"])
    with pytest.raises(selection.InvalidSelection):
        selection.select_development(**sealed)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_candidate",
        "missing_completion",
        "unexplained_failure",
        "wrong_contract",
        "wrong_summary",
        "missing_scores",
        "changed_mask",
        "changed_media",
    ],
)
def test_incomplete_or_incompatible_evidence_never_selects(sealed, damage):
    target = item(sealed)
    root = Path(target["path"])
    if damage == "missing_candidate":
        sealed["runs"].remove(target)
    elif damage == "missing_completion":
        (root / "completion.json").unlink()
    elif damage == "unexplained_failure":
        row = json.loads((root / "completion.json").read_text())
        row["completed"] = False
        write(root / "completion.json", row)
    elif damage == "wrong_contract":
        target["contract_sha256"] = "0" * 64
    elif damage == "wrong_summary":
        row = json.loads((root / "scores/summary.json").read_text())
        row["metrics"]["token_weighted_ce"] = 0.0
        write(root / "scores/summary.json", row)
    elif damage == "missing_scores":
        alter_scores(sealed, lambda rows: rows.pop())
    elif damage == "changed_mask":

        def transform(rows):
            rows[0]["inputs"]["labels"] = "changed"
            rows[0]["input_sha256"] = digest_json(rows[0]["inputs"])

        alter_scores(sealed, transform)
    else:
        alter_scores(sealed, lambda rows: rows[0].update(media_sha256="a" * 64))
    report = selection.select_development(**sealed)
    assert report["complete"] is False
    assert report["selected"] == {"nf4": None, "bf16": None}


def test_terminal_failed_candidate_retained_and_failed_baseline_blocks_its_arm(sealed):
    for candidate, arm in [("rank32", "nf4"), ("baseline", "bf16")]:
        path = Path(item(sealed, candidate, arm)["path"]) / "completion.json"
        row = json.loads(path.read_text())
        row.update(completed=False, failure="RuntimeError: simulated failure at /private/sentinel")
        write(path, row)
    report = selection.select_development(**sealed)
    assert report["complete"]
    assert report["selected"]["nf4"]["candidate"] == "lr_5e5"
    assert report["selected"]["bf16"] is None
    assert report["arm_status"]["bf16"] == "baseline_unavailable"
    assert sum(row["state"] == "failed" for row in report["candidates"]) == 2
    assert "sentinel" not in json.dumps(report)


def test_accuracy_gate_uses_exact_count_and_baseline_retained_when_over_budget(sealed):
    def worsen(rows):
        for row in rows[:6]:
            row.update(
                candidate_logits={"A": 2.0, "B": 1.0},
                restricted_choice_prediction="A",
                restricted_choice_correct=False,
                strict_first_token_id=1,
                strict_first_token_correct=False,
            )

    alter_scores(sealed, worsen)
    root = Path(item(sealed, "baseline", "bf16")["path"])
    samples = [json.loads(line) for line in (root / "nvml-host.jsonl").read_text().splitlines()]
    samples[0]["device_compute_processes"][0]["used_gpu_memory_bytes"] = 25 * selection.GIB
    write(root / "nvml-host.jsonl", samples, lines=True)
    report = selection.select_development(**sealed)
    assert report["complete"]
    assert report["selected"]["nf4"]["candidate"] == "lr_5e5"
    baseline = next(r for r in report["candidates"] if r["candidate"] == "baseline" and r["arm"] == "bf16")
    assert baseline["state"] == "completed" and not baseline["eligible"]
    assert "over_24gib_best_fit_budget" in baseline["reasons"]
    assert report["selected"]["bf16"]["candidate"] == "rank32"


def test_exact_remaining_tie_does_not_invent_a_new_tiebreaker(sealed):
    alter_scores(sealed, lambda rows: [row.update(native_sft_ce=0.9) for row in rows])
    report = selection.select_development(**sealed)
    assert report["complete"]
    assert report["selected"]["nf4"] is None
    assert report["arm_status"]["nf4"] == "unresolved_exact_tie"
    assert report["selected"]["bf16"]["candidate"] == "rank32"


@pytest.mark.parametrize("damage", ["checkpoint", "adapters", "identity", "fresh_base", "command"])
def test_stale_trained_scoring_and_inherited_base_are_rejected(sealed, damage):
    run = item(sealed)
    root = Path(run["path"])
    summary_path = root / "scores/summary.json"
    summary = json.loads(summary_path.read_text())
    if damage == "checkpoint":
        summary["checkpoint"] = "/private/another-checkpoint"
        write(summary_path, summary)
    elif damage == "adapters":
        summary["adapter_tensors_sha256"] = "1" * 64
        write(summary_path, summary)
    elif damage == "identity":
        identity_path = root / "scores/model_identity.json"
        identity = json.loads(identity_path.read_text())
        identity["base"]["revision"] = "0" * 40
        write(identity_path, identity)
        summary["model_identity_sha256"] = digest_json(identity)
        write(summary_path, summary)
    elif damage == "command":
        write(
            root / "stages/scoring/command.json", ["python", "score.py", "--checkpoint", "/private/another-checkpoint"]
        )
    else:
        contract_path = root / "contract.json"
        contract = json.loads(contract_path.read_text())
        contract["checkpoint"] = "/private/another-checkpoint"
        write(contract_path, contract)
        run["contract_sha256"] = digest_file(contract_path)
    report = selection.select_development(**sealed)
    assert report["complete"] is False
    assert report["selected"] == {"nf4": None, "bf16": None}
    assert "another-checkpoint" not in json.dumps(report)


def test_canonical_checkpoint_alias_preserves_valid_own_run_binding(sealed, tmp_path):
    run = item(sealed)
    root = Path(run["path"])
    actual = tmp_path / "actual-checkpoint"
    actual.mkdir()
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(actual, target_is_directory=True)
    final = json.loads((root / "final-checkpoint.json").read_text())
    final["checkpoint"] = str(alias)
    write(root / "final-checkpoint.json", final)
    probe = json.loads((root / "probes/dcp/probe.json").read_text())
    probe["checkpoint"] = str(alias)
    write(root / "probes/dcp/probe.json", probe)
    summary = json.loads((root / "scores/summary.json").read_text())
    summary["checkpoint"] = str(actual)
    write(root / "scores/summary.json", summary)
    write(root / "stages/scoring/command.json", ["python", "score.py", "--checkpoint", str(alias)])
    report = selection.select_development(**sealed)
    assert report["complete"] and report["selected"]["nf4"]["candidate"] == "rank32"


def test_cli_pins_run_inputs_and_refuses_to_overwrite_decision(sealed, tmp_path, monkeypatch):
    import sys

    inputs = tmp_path / "runs.json"
    write(inputs, {"protocol": "cosmos.qlora-development-runs.v1", "runs": sealed["runs"]})
    output = tmp_path / "selection.json"
    argv = [
        "selection",
        "--plan",
        str(sealed["plan_path"]),
        "--plan-sha256",
        sealed["plan_sha256"],
        "--development-manifest",
        str(sealed["development_manifest_path"]),
        "--development-manifest-sha256",
        sealed["development_manifest_sha256"],
        "--runs",
        str(inputs),
        "--runs-sha256",
        digest_file(inputs),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    selection.main()
    original = output.read_bytes()
    result = json.loads(original)
    assert result["runs_file_sha256"] == digest_file(inputs)
    assert result["untouched_scoring_authorized"] is False
    with pytest.raises(FileExistsError):
        selection.main()
    assert output.read_bytes() == original
    inputs.write_text(inputs.read_text() + "\n")
    with pytest.raises(selection.InvalidSelection, match="SHA256"):
        selection.main()
