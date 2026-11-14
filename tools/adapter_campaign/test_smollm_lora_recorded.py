# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic admission inverses; no synthetic receipts enter campaign evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.adapter_campaign import smollm_lora_recorded as m
from tools.adapter_campaign.evidence import EvidenceError


@pytest.mark.parametrize("cell", [m.CELL.replace("lora", "dora"), m.CELL.replace("fsdp2", "hsdp"), "other"])
def test_cell_admission_precedes_file_reads(cell):
    with pytest.raises(EvidenceError, match="unsupported"):
        m.normalize(
            "/does-not-exist", {"id": cell, "identity": {"training_execution": ["factorized_v1", "default_v1"]}}, {}
        )


@pytest.mark.parametrize(
    "execution",
    [["factorized_v1", "smollm_fp32_fixed_mma_v2"], ["merged_weight_bf16_v1", "default_v1"], [True, "default_v1"]],
)
def test_inference_or_other_training_execution_not_relabelled(execution):
    with pytest.raises(EvidenceError, match="unsupported"):
        m.normalize("/does-not-exist", {"id": m.CELL, "identity": {"training_execution": execution}}, {})


def recipes():
    full = {
        "cell": {
            "id": m.CELL,
            "optimizer_updates": 935,
            "training_epochs": 1,
            "data_order": {"finite_epochs": 1},
            "adapter_config": {"method": "lora"},
        },
        "paths": {"train": "full/train", "validation": "full/validation"},
    }
    cell = {"id": m.CELL, "identity": {"config_sha256": "a" * 64}}
    spec = copy.deepcopy(full)
    spec["primary_config_sha256"] = "a" * 64
    spec["preflight_data_contract"] = {
        "format": "cosmos-adapter-preflight-data-v1",
        "epochs": 3,
        "updates": 9,
        "stop_after": 6,
        "checkpoint": 3,
        "resume_updates": [4, 5, 6],
    }
    spec["cell"].update(optimizer_updates=9, training_epochs=3)
    spec["cell"]["data_order"]["finite_epochs"] = 3
    spec["paths"] = {k: "/fixture/data-preflight/text-control-v2/" + k + ".jsonl" for k in ("train", "validation")}
    return cell, full, spec


def test_only_declared_bounded_schedule_and_data_inverse():
    m.recipe_inverse(*recipes())


@pytest.mark.parametrize(
    "change",
    ["method", "dataset", "budget", "epochs", "finite", "bool_epoch", "config", "unknown", "resume", "checkpoint"],
)
def test_recipe_inverse_rejects_changed_training_semantics(change):
    cell, full, spec = recipes()
    if change == "method":
        spec["cell"]["adapter_config"]["method"] = "dora"
    elif change == "dataset":
        spec["paths"]["train"] = "/arbitrary/train.jsonl"
    elif change == "budget":
        spec["cell"]["optimizer_updates"] = 6
    elif change in ("epochs", "bool_epoch"):
        spec["preflight_data_contract"]["epochs"] = True if change == "bool_epoch" else 1
    elif change == "finite":
        spec["cell"]["data_order"]["finite_epochs"] = 1
    elif change == "config":
        spec["primary_config_sha256"] = "b" * 64
    elif change == "unknown":
        spec["new_setting"] = 1
    elif change == "resume":
        spec["preflight_data_contract"]["resume_updates"] = [3, 4, 5]
    else:
        spec["preflight_data_contract"]["checkpoint"] = 2
    with pytest.raises(EvidenceError):
        m.recipe_inverse(cell, full, spec)


class SyntheticReference:
    def __init__(self):
        source = {
            "source_revision": m.SOURCE["revision"],
            "wheel_sha256": m.SOURCE["wheel_sha256"],
            "container_sha256": m.SOURCE["runtime_sha256"],
            "source_dirty": False,
            "verified_package_files": 1932,
            "tracked_package_files_verified": 1932,
            "slurm_job_id": "38363470",
        }
        primary = {**source, "campaign_spec_sha256": "a" * 64, "harness_sha256": m.PREFLIGHT}
        source.update(
            reference_provenance=primary,
            oracle_sha256=m.ORACLE,
            strict_determinism=True,
            spec_sha256="a" * 64,
            reference_files={"initial-rank0.json": "a" * 64},
            environment=None,
        )
        steps = [
            {
                "step": i,
                "status": "PASS",
                "global_valid_tokens": tokens,
                "global_loss": 1.0,
                "distributed_loss": 1.0,
                "max_gradient_error": 1e-9,
                "max_parameter_error": 1e-9,
            }
            for i, tokens in enumerate((1286, 830, 131), 1)
        ]
        self.result = {"source": source, "steps": steps}
        self.cpu = {
            "reference": {
                "initial_portable_exact": True,
                "steps": [dict(r, owners=120, max_cpu_parameter_error=1e-9, max_cpu_moment_error=1e-9) for r in steps],
            }
        }
        self.seen = []

    def name(self, name):
        return "/synthetic/" + name

    def sha(self, path):
        return "a" * 64

    def read(self, path):
        return self.result

    def tensor(self, path):
        self.seen.append(path)

    def job(self, *args):
        return {"synthetic_test_fixture": True, "job": args[0]}


def test_original_v1_reference_consumes_all_own_state_witnesses():
    c = SyntheticReference()
    result = m.independent_reference(c, {}, {}, "/synthetic/cell")
    assert len(c.seen) == len(set(c.seen)) == 20
    assert result["oracle_identity"]["sha256"] == m.ORACLE
    assert "fixed_mma" not in repr(result)


@pytest.mark.parametrize(
    "change",
    [
        "oracle",
        "version",
        "source",
        "dirty",
        "package",
        "job",
        "spec",
        "harness",
        "env",
        "strict",
        "tail",
        "bool_token",
        "steps",
        "status",
        "initial",
        "owners",
        "grad",
        "parameter",
        "moment",
        "negative",
        "nan",
        "loss",
        "foreign_path",
        "gradient_disagreement",
    ],
)
def test_original_v1_reference_cannot_clear_wrong_or_incomplete_evidence(change):
    c = SyntheticReference()
    s = c.result["source"]
    actual, replay = c.result["steps"][2], c.cpu["reference"]["steps"][2]
    if change == "oracle":
        s["oracle_sha256"] = "b" * 64
    elif change == "version":
        s["oracle_version"] = "text_micro_reduce_v3"
    elif change == "source":
        s["source_revision"] = "b" * 40
    elif change == "dirty":
        s["source_dirty"] = True
    elif change == "package":
        s["verified_package_files"] = 1931
    elif change == "job":
        s["slurm_job_id"] = "38366620"
    elif change == "spec":
        s["spec_sha256"] = "b" * 64
    elif change == "harness":
        s["reference_provenance"]["harness_sha256"] = "b" * 64
    elif change == "env":
        s["environment"] = {"new_runtime": True}
    elif change == "strict":
        s["strict_determinism"] = False
    elif change in ("tail", "bool_token"):
        actual["global_valid_tokens"] = True if change == "bool_token" else 130
    elif change == "steps":
        c.result["steps"].pop()
    elif change == "status":
        actual["status"] = "FAIL"
    elif change == "initial":
        c.cpu["reference"]["initial_portable_exact"] = False
    elif change == "owners":
        replay["owners"] = 180
    elif change == "grad":
        actual["max_gradient_error"] = replay["max_gradient_error"] = 3e-6
    elif change == "parameter":
        replay["max_cpu_parameter_error"] = 2e-7
    elif change == "moment":
        replay["max_cpu_moment_error"] = 2e-7
    elif change == "negative":
        replay["max_cpu_parameter_error"] = -1.0
    elif change == "nan":
        replay["max_cpu_parameter_error"] = float("nan")
    elif change == "loss":
        actual["global_loss"] = 3.0
    elif change == "foreign_path":
        s["reference_files"] = {"../other.json": "a" * 64}
    else:
        replay["max_gradient_error"] = 2e-9
    with pytest.raises((EvidenceError, KeyError)):
        m.independent_reference(c, {}, {}, "/synthetic/cell")


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "producer",
        "source",
        "review",
        "manifest",
        "dependency",
        "supporting",
        "gate_hash",
        "review_status",
        "synthetic_contract",
    ],
)
def test_typed_emitter_admission_and_incomplete_preflight_remains_blocked(tmp_path, monkeypatch, case):
    manifest = {"path": "synthetic-unused.json", "sha256": "a" * 64}
    producer = hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    review = {
        "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
        "open_findings": [],
        "recorded_capture_manifest": manifest,
        "reviewed_repository_files": {
            "tools/adapter_campaign/" + n: hashlib.sha256(Path(m.__file__).with_name(n).read_bytes()).hexdigest()
            for n in m.DEPENDENCIES
        },
        "reviewed_files": {},
    }
    if case == "manifest":
        review["recorded_capture_manifest"] = {}
    if case == "dependency":
        review["reviewed_repository_files"]["tools/adapter_campaign/smollm_recorded.py"] = "b" * 64
    if case == "review_status":
        review["status"] = "PENDING"
    path = tmp_path / "synthetic-review.json"
    path.write_text(json.dumps(review))
    reference = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    reqs = [
        {
            "id": gate,
            "producers": [producer],
            "sources": [m.SOURCE],
            "producer_reviews": {producer: [reference]},
            "assertions": [
                {"document": "measurement", "pointer": "/gates_sha256", "operator": "eq", "expected": m.GATES_SHA}
            ],
        }
        for gate in m.GATES
    ]
    if case == "producer":
        reqs[0]["producers"] = []
    if case == "source":
        reqs[0]["sources"] = []
    if case == "review":
        reqs[0]["producer_reviews"][producer] = [{"path": "wrong.json", "sha256": "b" * 64}]
    if case == "gate_hash":
        reqs[1]["assertions"][0]["expected"] = "b" * 64
    contract = {
        "synthetic": case == "synthetic_contract",
        "cells": [{"id": m.CELL, "identity": {"synthetic_test_fixture": True}, "requirements": reqs}],
    }
    supporting = [{"path": "unreviewed.json", "sha256": "c" * 64}] if case == "supporting" else []
    monkeypatch.setattr(
        m,
        "normalize",
        lambda *args: (
            {gate: {"synthetic_test_fixture": True} for gate in m.GATES},
            m.SOURCE,
            {"synthetic_test_fixture": True},
            supporting,
        ),
    )
    if case != "complete":
        with pytest.raises((EvidenceError, KeyError)):
            m.produce(tmp_path, contract, manifest, reference, tmp_path / "synthetic-receipts")
        return
    index = m.produce(tmp_path, contract, manifest, reference, tmp_path / "synthetic-receipts")
    gates = index["cells"][m.CELL]["gates"]
    rows = {gate: json.loads((tmp_path / refs[0]["path"]).read_bytes()) for gate, refs in gates.items()}
    assert rows[m.GATES[0]]["outcome"] == "BLOCKED" and rows[m.GATES[0]]["executed"] is False
    assert rows[m.GATES[0]]["documents"] == {} and rows[m.GATES[0]]["blocker_evidence"]
    assert all(rows[gate]["outcome"] == "PASS" for gate in m.GATES[1:])
    assert len(gates) == 3 and not any("deployment" in gate for gate in gates)
