# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic proof/admission controls, never campaign executions or receipts."""

import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools.adapter_campaign import final_checkpoint as subject
from tools.adapter_campaign.evidence import EvidenceError, Store, check_receipt, digest
from tools.adapter_campaign.primary import write


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None])
def test_positive_integer_rejects_boolean_and_nonpositive(value):
    assert not subject.positive_integer(value)


def test_positive_integer_accepts_actual_counter():
    assert subject.positive_integer(935)


@pytest.mark.parametrize("kind", ["missing", "extra", "empty", "bool", "bad_hash", "extra_field", "partial"])
def test_complete_inventory_requires_exact_files_and_typed_nonempty_hashes(kind):
    names = {"trainer.pt", "model.pt", "data.pt"}
    values = {name: {"bytes": 17, "sha256": "a" * 64} for name in names}
    assert subject.inventory(values, names) == values
    if kind == "missing":
        values.pop("trainer.pt")
    elif kind == "extra":
        values["other.pt"] = copy.deepcopy(values["model.pt"])
    elif kind == "empty":
        values["trainer.pt"]["bytes"] = 0
    elif kind == "bool":
        values["trainer.pt"]["bytes"] = True
    elif kind == "bad_hash":
        values["trainer.pt"]["sha256"] = "z" * 64
    elif kind == "extra_field":
        values["trainer.pt"]["incomplete"] = True
    else:
        values["trainer.pt.partial"] = values.pop("trainer.pt")
    with pytest.raises(EvidenceError):
        subject.inventory(values, names)


@pytest.mark.parametrize("kind", ["failed", "bool", "missing", "extra", "foreign_source", "wrong_python"])
def test_command_requires_exact_public_invocation_and_success(kind):
    expected = ["python", "producer.py", "--source", "/original", "--out", "/new"]
    measured = {"returncode": 0, "command": expected.copy()}
    subject.validate_command(measured, expected)
    if kind == "failed":
        measured["returncode"] = 1
    elif kind == "bool":
        measured["returncode"] = False
    elif kind == "missing":
        measured["command"].pop()
    elif kind == "extra":
        measured["command"] += ["--allow-partial"]
    elif kind == "foreign_source":
        measured["command"][3] = "/other"
    else:
        measured["command"][0] = "unattested-python"
    with pytest.raises(EvidenceError):
        subject.validate_command(measured, expected)


def fixture(root):
    """Clearly synthetic temporary campaign sufficient for producer-output checks."""
    producer = hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest()
    refs = {
        name: write(root, root / (name + ".json"), {"fixture": "synthetic", "name": name})
        for name in ("capture", "controls", "admission")
    }
    review = {
        "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
        "open_findings": [],
        "reviewed_files": {
            str("tools/adapter_campaign/" + n): hashlib.sha256(
                Path(subject.__file__).with_name(n).read_bytes()
            ).hexdigest()
            for n in subject.DEPENDENCIES
        },
    }
    review["reviewed_files"].update({r["path"]: r["sha256"] for r in refs.values()})
    review_ref = write(root, root / "review.json", review)
    source = {"revision": "a" * 40, "wheel_sha256": "b" * 64, "runtime_sha256": "c" * 64}
    identity = {"cell_id": "synthetic-not-a-campaign-cell"}
    lineage = {"primary_job": "synthetic", "primary_root": "/synthetic", "primary_plan_sha256": "d" * 64}
    requirement = {
        "id": subject.GATE,
        "profile": None,
        "final_checkpoint": True,
        "documents": ["measurement", "provenance"],
        "provenance_document": "provenance",
        "raw_evidence_required": True,
        "training_lineage": True,
        "producers": [producer],
        "producer_reviews": {producer: [review_ref]},
        "sources": [source],
        "checkpoint_anchor": {"document": "measurement", "pointer": "/checkpoint"},
        "assertions": [
            {
                "document": "measurement",
                "pointer": "/checkpoint",
                "operator": "complete_checkpoint",
                "expected": {"components": ["model", "optimizer", "training_state"], "update": 9},
            },
            {
                "document": "measurement",
                "pointer": "/consolidated_and_dcp_verified",
                "operator": "eq",
                "expected": True,
            },
        ],
    }
    cell = {"id": identity["cell_id"], "identity": identity, "requirements": [requirement]}
    contract = {"synthetic": False, "cells": [cell]}
    files = {"trainer.pt": "f" * 64, "model.pt": "e" * 64, "optim/.metadata": "1" * 64, "data.pt": "2" * 64}
    measurement = {
        "checkpoint": {
            "files": files,
            "inventory_sha256": digest(files),
            "update": 9,
            "components": ["model", "optimizer", "training_state"],
            "identity": identity,
            "lineage": lineage,
            "committed": True,
            "validated_by_framework": True,
        },
        "consolidated_and_dcp_verified": True,
    }
    args = {
        "cell_id": cell["id"],
        "capture_ref": refs["capture"],
        "controls_ref": refs["controls"],
        "primary_admission_ref": refs["admission"],
        "producer_review": review_ref,
    }
    return contract, args, (measurement, source, lineage, list(refs.values())), requirement


def test_emits_only_final_gate_and_original_inventory_anchor(tmp_path):
    contract, args, normalized, requirement = fixture(tmp_path)
    with mock.patch.object(subject, "normalize", return_value=normalized) as normalize:
        index = subject.produce(tmp_path, contract, tmp_path / "new", **args)
    normalize.assert_called_once()
    entry = index["cells"][args["cell_id"]]
    assert set(entry["gates"]) == {"final_checkpoint"}
    assert entry["final_checkpoint"] == normalized[0]["checkpoint"]["inventory_sha256"]
    store = Store(tmp_path)
    receipt = entry["gates"]["final_checkpoint"][0]
    result = check_receipt(
        store,
        receipt,
        requirement,
        contract["cells"][0]["identity"],
        contract,
        entry["final_checkpoint"],
        normalized[2],
    )
    assert result["outcome"] == "PASS"
    assert not (tmp_path / "campaign-final-gates-current.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "synthetic",
        "existing",
        "outside",
        "unreviewed",
        "findings",
        "dependency",
        "capture",
        "controls",
        "admission",
        "producer",
        "review",
        "source",
    ],
)
def test_producer_refuses_unreviewed_or_unsupported_inputs_before_output(tmp_path, kind):
    contract, args, normalized, _ = fixture(tmp_path)
    out = tmp_path / "new"
    if kind == "synthetic":
        contract["synthetic"] = True
    elif kind == "existing":
        out.mkdir()
    elif kind == "outside":
        out = tmp_path.parent / "synthetic-outside-test"
    elif kind in ("unreviewed", "findings", "dependency", "capture", "controls", "admission"):
        review = json.loads((tmp_path / args["producer_review"]["path"]).read_bytes())
        if kind == "unreviewed":
            review["status"] = "PENDING"
        elif kind == "findings":
            review["open_findings"] = ["unfixed"]
        elif kind == "dependency":
            review["reviewed_files"]["tools/adapter_campaign/inline_midpoint.py"] = "0" * 64
        else:
            review["reviewed_files"][kind + ".json"] = "0" * 64
        args["producer_review"] = write(tmp_path, tmp_path / "review-bad.json", review)
    else:
        requirement = contract["cells"][0]["requirements"][0]
        requirement[{"producer": "producers", "review": "producer_reviews", "source": "sources"}[kind]] = (
            [] if kind != "review" else {}
        )
    with mock.patch.object(subject, "normalize", return_value=normalized):
        with pytest.raises(EvidenceError):
            subject.produce(tmp_path, contract, out, **args)
    assert not (out / "index.json").exists()


@pytest.mark.parametrize(
    "field", ["identity", "lineage", "inventory_sha256", "committed", "validated_by_framework", "update", "files"]
)
def test_receipt_cannot_claim_another_or_incomplete_checkpoint(tmp_path, field):
    contract, args, normalized, requirement = fixture(tmp_path)
    with mock.patch.object(subject, "normalize", return_value=normalized):
        index = subject.produce(tmp_path, contract, tmp_path / "new", **args)
    entry = index["cells"][args["cell_id"]]
    receipt = Store(tmp_path).json(entry["gates"][subject.GATE][0])
    measurement = copy.deepcopy(normalized[0])
    measurement["checkpoint"][field] = {
        "identity": {},
        "lineage": {},
        "inventory_sha256": "0" * 64,
        "committed": False,
        "validated_by_framework": False,
        "update": True,
        "files": {},
    }[field]
    m = write(tmp_path, tmp_path / "wrong-measurement.json", measurement)
    prov = Store(tmp_path).json(receipt["documents"]["provenance"])
    prov["documents_sha256"]["measurement"] = m["sha256"]
    p = write(tmp_path, tmp_path / "wrong-provenance.json", prov)
    receipt["documents"] = {"measurement": m, "provenance": p}
    r = write(tmp_path, tmp_path / "wrong-receipt.json", receipt)
    with pytest.raises(EvidenceError):
        check_receipt(
            Store(tmp_path),
            r,
            requirement,
            contract["cells"][0]["identity"],
            contract,
            entry["final_checkpoint"],
            normalized[2],
        )
