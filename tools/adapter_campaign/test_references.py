# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic reference receipts: never admitted as real campaign evidence."""

import copy
import hashlib
import json

import pytest

from tools.adapter_campaign.evidence import EvidenceError, Store
from tools.adapter_campaign.references import normalize


@pytest.fixture
def evidence(tmp_path):
    source = {"source_revision": "a" * 40, "wheel_sha256": "b" * 64, "image_sha256": "c" * 64}
    row = {
        "cell": "synthetic-cell",
        "full_config_sha256": "d" * 64,
        "status": "PASS",
        "spec_sha256": "e" * 64,
        "primary_receipt_sha256": "f" * 64,
        "job": "123",
        "plan": "/remote/plan.json",
        "receipt": "/remote/cell/receipt.json",
        "oracle_version": "text_micro_reduce_v3",
        **source,
    }
    steps = [
        {
            "step": i,
            "status": "PASS",
            "global_valid_tokens": sum(counts),
            "independent_micro_tokens": counts,
            "global_loss": 1.0,
            "distributed_loss": 1.0,
            "max_gradient_error": 0.0,
            "max_parameter_error": 0.0,
        }
        for i, counts in enumerate(([3, 5], [2, 4], [1, 0]), 1)
    ]
    gradients = {f"gradient-{i:06d}.pt": str(i) * 64 for i in (1, 2, 3)}
    row["steps"] = copy.deepcopy(steps)
    row["gradient_artifacts"] = {n: {"sha256": h} for n, h in gradients.items()}
    documents = {
        row["plan"]: {
            **source,
            "cells": [
                {
                    "id": row["cell"],
                    "sha256": row["spec_sha256"],
                    "primary_receipt_sha256": row["primary_receipt_sha256"],
                }
            ],
            "files_sha256": {"independent_text_update_reference_v3.py": "1" * 64},
        },
        row["receipt"]: {
            "cell": row["cell"],
            "job": row["job"],
            "status": "PASS",
            "returncode": 0,
            "steps": copy.deepcopy(steps),
            "gradient_artifacts": gradients,
        },
        "/remote/cell/oracle/results.json": {
            "source": {
                "source_revision": source["source_revision"],
                "wheel_sha256": source["wheel_sha256"],
                "container_sha256": source["image_sha256"],
                "spec_sha256": row["spec_sha256"],
                "oracle_version": row["oracle_version"],
                "slurm_job_id": row["job"],
                "verified_package_files": 1932,
                "strict_determinism": True,
                "initial_trainable_exact": True,
                "oracle_sha256": "1" * 64,
            },
            "steps": steps,
        },
    }

    def persist():
        captured = {}
        # Hash final dependencies first, then bind them into their receipts.
        for remote in [row["plan"], "/remote/cell/oracle/results.json", row["receipt"]]:
            document = documents[remote]
            if remote == row["receipt"]:
                document["plan_sha256"] = row["plan_sha256"]
                document["results_sha256"] = row["result_sha256"]
            path = tmp_path / str(len(captured))
            raw = (json.dumps(document, allow_nan=True) + "\n").encode()
            path.write_bytes(raw)
            ref = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
            captured[remote] = ref
            key = {
                row["plan"]: "plan_sha256",
                row["receipt"]: "receipt_sha256",
                "/remote/cell/oracle/results.json": "result_sha256",
            }[remote]
            row[key] = ref["sha256"]
        if row["status"] == "FAIL":
            path = tmp_path / "failure.log"
            path.write_text("Synthetic gradient mismatch at update 2\n")
            ref = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            captured["/remote/cell/oracle.log"] = ref
            row["log"] = {"path": "/remote/cell/oracle.log", "sha256": ref["sha256"]}
            row["partial_result"] = {"sha256": row["result_sha256"]}
        return Store(tmp_path), captured

    cell = {"id": row["cell"], "identity": {"config_sha256": row["full_config_sha256"]}}
    return row, cell, documents, persist


def test_three_updates_including_tail_are_a_prerequisite_only(evidence):
    row, cell, _, persist = evidence
    store, captured = persist()
    result, source = normalize(store, cell, row, captured, "9" * 64)
    assert result["uninterrupted_multi_update_reference"]
    assert result["tail_and_uneven_lengths"]
    assert result["optimizer_within_gates"]
    assert source["revision"] == "a" * 40
    assert "final_checkpoint" not in result and "full_training_completed" not in result


def test_early_gradient_failure_does_not_certify_later_updates(evidence):
    row, cell, docs, persist = evidence
    row.update(
        status="FAIL",
        first_failed_update=2,
        first_failed_check="gradient comparison before optimizer step",
        mismatched_elements=1,
        gradient_gates={"atol": 2e-6, "rtol": 2e-4},
    )
    docs[row["receipt"]].update(status="FAIL", returncode=1)
    docs["/remote/cell/oracle/results.json"]["steps"] = docs["/remote/cell/oracle/results.json"]["steps"][:1]
    store, captured = persist()
    result, _ = normalize(store, cell, row, captured, "9" * 64)
    assert not result["raw_gradients_within_gates"]
    assert not result["optimizer_within_gates"]
    assert not result["tail_and_uneven_lengths"]
    assert len(result["completed_steps"]) == 1
    assert result["failure"]["first_failed_update"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_config",
        "changed_source",
        "reinitialized",
        "different_oracle",
        "missing_step",
        "duplicate_step",
        "wrong_denominator",
        "changed_gate",
        "missing_gradient",
        "missing_review_gradient",
        "nonfinite",
        "failed_command",
        "changed_status",
        "no_tail",
    ],
)
def test_semantically_invalid_receipts_fail_with_fresh_hashes(evidence, mutation):
    row, cell, docs, persist = evidence
    result = docs["/remote/cell/oracle/results.json"]
    receipt = docs[row["receipt"]]
    if mutation == "wrong_config":
        cell["identity"]["config_sha256"] = "0" * 64
    elif mutation == "changed_source":
        result["source"]["wheel_sha256"] = "0" * 64
    elif mutation == "reinitialized":
        result["source"]["initial_trainable_exact"] = False
    elif mutation == "different_oracle":
        result["source"]["oracle_sha256"] = "0" * 64
    elif mutation == "missing_step":
        result["steps"].pop()
    elif mutation == "duplicate_step":
        result["steps"][1]["step"] = 1
    elif mutation == "wrong_denominator":
        result["steps"][0]["global_valid_tokens"] += 1
    elif mutation == "changed_gate":
        result["steps"][0]["distributed_loss"] = 2.0
    elif mutation == "missing_gradient":
        receipt["gradient_artifacts"].pop("gradient-000003.pt")
    elif mutation == "missing_review_gradient":
        row["gradient_artifacts"].pop("gradient-000003.pt")
    elif mutation == "nonfinite":
        result["steps"][0]["max_gradient_error"] = float("nan")
    elif mutation == "failed_command":
        receipt["returncode"] = 1
    elif mutation == "changed_status":
        receipt["status"] = "FAIL"
    elif mutation == "no_tail":
        for steps in (result["steps"], row["steps"], receipt["steps"]):
            steps[-1].update(independent_micro_tokens=[1, 1], global_valid_tokens=2)
    store, captured = persist()
    with pytest.raises((EvidenceError, ValueError)):
        normalize(store, cell, row, captured, "9" * 64)


def test_absent_or_mutated_raw_file_cannot_enter_gate(evidence):
    row, cell, _, persist = evidence
    store, captured = persist()
    path = store.root / captured[row["receipt"]]["path"]
    path.write_text("{}")
    with pytest.raises(EvidenceError, match="stale evidence"):
        normalize(store, cell, row, captured, "9" * 64)
    path.unlink()
    with pytest.raises(EvidenceError, match="missing or escaping"):
        normalize(store, cell, row, captured, "9" * 64)
