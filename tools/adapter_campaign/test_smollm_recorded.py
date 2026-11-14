# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic receipt checks only; no synthetic campaign credit."""

from pathlib import Path

import pytest

from tools.adapter_campaign.evidence import EvidenceError
from tools.adapter_campaign.smollm_recorded import CELL, GATES, STATE_KEYS, finite, normalize, validate_cpu_continuation


class Tensors:
    def __init__(self):
        self.seen = []

    def tensor(self, path):
        self.seen.append(path)
        return "a" * 64


def proof():
    return {
        "mid_update": 467,
        "steps": [468, 469, 470],
        "ranks": [
            {
                "rank": rank,
                "initial_state_exact": True,
                "updates": [
                    {
                        "step": step,
                        "status": "PASS",
                        "owners": 180,
                        "tensor_and_control_leaves_compared": 1001,
                        "max_errors": dict.fromkeys(STATE_KEYS, 0.0),
                        "loss_error": 0.0,
                    }
                    for step in (468, 469, 470)
                ],
            }
            for rank in range(4)
        ],
    }


def test_complete_synthetic_control_consumes_exact_witnesses():
    tensors = Tensors()
    validate_cpu_continuation(tensors, proof(), "old", "resumed", 467, 180)
    assert len(tensors.seen) == len(set(tensors.seen)) == 32
    assert "old/update-000467-rank3.pt" in tensors.seen
    assert "resumed/update-000470-rank3.pt" in tensors.seen


@pytest.mark.parametrize(
    "case",
    [
        "wrong_mid",
        "bool_mid",
        "wrong_steps",
        "missing_rank",
        "duplicate_rank",
        "bool_rank",
        "restore_false",
        "missing_update",
        "repeated_update",
        "reordered_update",
        "failed_update",
        "wrong_owner",
        "bool_owner",
        "partial_leaves",
        "missing_rng",
        "new_state_key",
        "nan_error",
        "inf_error",
        "negative_error",
        "bool_error",
        "parameter_over_gate",
        "optimizer_over_gate",
        "gradient_over_gate",
        "input_difference",
        "rng_difference",
        "cursor_difference",
        "scheduler_difference",
        "clock_difference",
        "loss_difference",
        "bool_loss",
        "nan_loss",
    ],
)
def test_incomplete_or_changed_continuation_refused(case):
    p = proof()
    u = p["ranks"][0]["updates"][0]
    if case == "wrong_mid":
        p["mid_update"] = 468
    elif case == "bool_mid":
        p["mid_update"] = True
    elif case == "wrong_steps":
        p["steps"] = [468, 469]
    elif case == "missing_rank":
        p["ranks"].pop()
    elif case == "duplicate_rank":
        p["ranks"][3]["rank"] = 2
    elif case == "bool_rank":
        p["ranks"][0]["rank"] = False
    elif case == "restore_false":
        p["ranks"][0]["initial_state_exact"] = False
    elif case == "missing_update":
        p["ranks"][0]["updates"].pop()
    elif case == "repeated_update":
        p["ranks"][0]["updates"][1]["step"] = 468
    elif case == "reordered_update":
        p["ranks"][0]["updates"].reverse()
    elif case == "failed_update":
        u["status"] = "FAIL"
    elif case == "wrong_owner":
        u["owners"] = 179
    elif case == "bool_owner":
        u["owners"] = True
    elif case == "partial_leaves":
        u["tensor_and_control_leaves_compared"] = 1
    elif case == "missing_rng":
        u["max_errors"].pop("rng")
    elif case == "new_state_key":
        u["max_errors"]["omitted"] = 0.0
    elif case in {"nan_error", "inf_error", "negative_error", "bool_error"}:
        u["max_errors"]["parameters"] = {
            "nan_error": float("nan"),
            "inf_error": float("inf"),
            "negative_error": -1.0,
            "bool_error": False,
        }[case]
    elif case.endswith("_over_gate"):
        key = {
            "parameter_over_gate": "parameters",
            "optimizer_over_gate": "optimizer",
            "gradient_over_gate": "gradients",
        }[case]
        u["max_errors"][key] = 1.00001e-7
    elif case.endswith("_difference") and not case.startswith("loss"):
        key = {
            "input_difference": "inputs",
            "rng_difference": "rng",
            "cursor_difference": "data",
            "scheduler_difference": "scheduler",
            "clock_difference": "clock",
        }[case]
        u["max_errors"][key] = 1e-12
    else:
        u["loss_error"] = {"loss_difference": 1e-12, "bool_loss": False, "nan_loss": float("nan")}[case]
    with pytest.raises((EvidenceError, KeyError)):
        validate_cpu_continuation(Tensors(), p, "old", "resumed", 467, 180)


@pytest.mark.parametrize(
    "name",
    [CELL.replace("dora", "lora"), CELL.replace("fsdp2", "hsdp"), "qwen2_hf__dora__factorized_v1__default_v1__fsdp2"],
)
def test_closed_cell_admission_before_evidence_access(name):
    cell = {"id": name, "identity": {"training_execution": ["factorized_v1", "default_v1"]}}
    with pytest.raises(EvidenceError, match="does not admit"):
        normalize(Path("/missing"), cell, {})


@pytest.mark.parametrize(
    "execution", [["merged_weight_bf16_v1", "default_v1"], ["factorized_v1", "experimental"], [True, "default_v1"]]
)
def test_changed_execution_cannot_use_old_proofs(execution):
    with pytest.raises(EvidenceError, match="does not admit"):
        normalize(Path("/missing"), {"id": CELL, "identity": {"training_execution": execution}}, {})


def test_only_three_existing_categories():
    assert GATES == ("independent_global_update", "controlled_failure_recovery", "mid_epoch_continuation")
    assert finite(0.0) and not finite(False) and not finite(float("nan"))


@pytest.mark.parametrize(
    "case",
    ["complete", "producer", "source", "review", "manifest", "repository", "supporting", "gate_hash", "review_status"],
)
def test_synthetic_emitter_admission_boundary(tmp_path, monkeypatch, case):
    import hashlib
    import json

    from tools.adapter_campaign import smollm_recorded as m

    root = tmp_path
    manifest = {"path": "synthetic-unused.json", "sha256": "a" * 64}
    producer = hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    reviewed = {
        "tools/adapter_campaign/" + n: hashlib.sha256(Path(m.__file__).with_name(n).read_bytes()).hexdigest()
        for n in m.DEPENDENCIES
    }
    review = {
        "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
        "open_findings": [],
        "recorded_capture_manifest": manifest,
        "reviewed_repository_files": reviewed,
        "reviewed_files": {},
    }
    if case == "manifest":
        review["recorded_capture_manifest"] = {"path": "other.json", "sha256": "b" * 64}
    if case == "repository":
        review["reviewed_repository_files"]["tools/adapter_campaign/smollm_recorded.py"] = "b" * 64
    if case == "review_status":
        review["status"] = "SOURCE_ONLY_PENDING"
    path = root / "synthetic-review.json"
    path.write_text(json.dumps(review))
    ref = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    requirements = []
    for gate in GATES:
        pointer, value = (
            ("/protocol_sha256", m.FAULT_SHA)
            if gate == "controlled_failure_recovery"
            else ("/gates_sha256", m.GATES_SHA)
        )
        requirements.append(
            {
                "id": gate,
                "producers": [producer],
                "sources": [m.SOURCE],
                "producer_reviews": {producer: [ref]},
                "assertions": [{"document": "measurement", "pointer": pointer, "operator": "eq", "expected": value}],
            }
        )
    if case == "producer":
        requirements[0]["producers"] = []
    if case == "source":
        requirements[0]["sources"] = []
    if case == "review":
        requirements[0]["producer_reviews"][producer] = [{"path": "other-review.json", "sha256": "b" * 64}]
    if case == "gate_hash":
        requirements[0]["assertions"][0]["expected"] = "b" * 64
    contract = {
        "synthetic": True,
        "cells": [{"id": CELL, "identity": {"synthetic_test_fixture": True}, "requirements": requirements}],
    }
    supporting = [{"path": "unreviewed.json", "sha256": "c" * 64}] if case == "supporting" else []
    monkeypatch.setattr(
        m,
        "normalize",
        lambda *args: (
            {g: {"synthetic_test_fixture": True} for g in GATES},
            m.SOURCE,
            {"synthetic_test_fixture": True},
            supporting,
        ),
    )
    output = root / "synthetic-output"
    if case != "complete":
        with pytest.raises(EvidenceError):
            m.produce(root, contract, manifest, ref, output)
        assert not output.exists()
    else:
        index = m.produce(root, contract, manifest, ref, output)
        assert set(index["cells"][CELL]["gates"]) == set(GATES)
        assert len(list(output.rglob("receipt.json"))) == 3
        assert len(list(output.rglob("measurement.json"))) == 3
