# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic-only identity amendment controls; never campaign evidence."""

import copy
from pathlib import Path

import pytest

from tools.adapter_campaign.contract import apply_pretraining_identity, assertion
from tools.adapter_campaign.evidence import EvidenceError, Store, digest
from tools.adapter_campaign.primary import write


def fixture():
    cells = [
        {
            "id": f"family{family}__synthetic{number}",
            "identity": {"base": None, "processor": None, "config_sha256": "c" * 64},
            "requirements": [
                {
                    "id": "matched_base_evaluation",
                    "admission": "REQUIRED",
                    "assertions": [assertion("immutable_base_sha256", digest(None))],
                },
                {"id": "immutable_asset_seal", "admission": "REQUIRED", "pending_admission": "Missing identity"},
            ],
        }
        for family in range(16)
        for number in range(23)
    ]
    prior = {
        "format": "adapter-campaign-completion-contract-v1",
        "synthetic_fixture": True,
        "required_cells": 368,
        "cells": cells,
    }
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(prior), "cells": {}}
    after = {
        "base": {"sha256": "a" * 64, "files": {"model.safetensors": "b" * 64}},
        "processor": {"sha256": "b" * 64, "files": {"tokenizer.json": "a" * 64}},
    }
    candidate = copy.deepcopy(prior)
    candidate["cells"][0]["identity"].update(after)
    proof = {
        "format": "hf-segmented-pretraining-identity-resolution-v2",
        "cell": cells[0]["id"],
        "prior_identity": copy.deepcopy(cells[0]["identity"]),
        "prior_contract_sha256": digest(prior),
        "prior_index_sha256": digest(index),
        "candidate_contract_sha256": digest(candidate),
        "before": {"base": None, "processor": None},
        "after": after,
        "capture_origin": "/synthetic-original-root",
        "qualification_pass": False,
        "active_index_written": False,
        "gate_evidence_emitted": False,
        "producer_files_sha256": {
            "tools/adapter_campaign/" + n: "d" * 64
            for n in ("hf_segmented.py", "segmented.py", "recovered.py", "evidence.py", "primary.py")
        },
    }
    audit = {"cell": cells[0]["id"], "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS", "open_findings": []}
    review = {"status": "PASS_NO_OPEN_CONFIRMED_FINDINGS", "open_findings": [], "synthetic_fixture": True}
    return prior, index, proof, audit, review


def seal(tmp_path, value):
    prior, index, proof, audit, review = value
    refs = {}
    for name, data in (("prior_contract", prior), ("prior_index", index), ("actual_audit", audit)):
        refs[name] = write(tmp_path, tmp_path / (name + ".json"), data)
    proof.setdefault("actual_audit", refs["actual_audit"])
    proof.setdefault("supporting_evidence", [refs["actual_audit"]])
    refs["resolution"] = write(tmp_path, tmp_path / "resolution.json", proof)
    review.setdefault("capture_origins", {refs["actual_audit"]["sha256"]: proof["capture_origin"]})
    review.setdefault(
        "reviewed_files",
        {
            refs["actual_audit"]["path"]: refs["actual_audit"]["sha256"],
            refs["resolution"]["path"]: refs["resolution"]["sha256"],
            **proof["producer_files_sha256"],
        },
    )
    refs["producer_review"] = write(tmp_path, tmp_path / "producer_review.json", review)
    amendment = {
        "format": "adapter-campaign-pretraining-identity-amendment-v1",
        "cell": prior["cells"][0]["id"],
        **{k: refs[k] for k in ("prior_contract", "prior_index", "resolution", "producer_review")},
    }
    return write(tmp_path, tmp_path / "amendment.json", amendment)


def apply(tmp_path, value, *, cells=None):
    ref = seal(tmp_path, value)
    selected = copy.deepcopy(value[0]["cells"]) if cells is None else cells
    apply_pretraining_identity(
        Store(tmp_path), selected, name="pretraining_identity:" + value[0]["cells"][0]["id"], reference=ref
    )
    return selected


def test_resolution_changes_only_identity_base_assertion_and_explanatory_note(tmp_path):
    value = fixture()
    original = copy.deepcopy(value)
    result = apply(tmp_path, value)
    before = original[0]["cells"]
    assert result[1:] == before[1:]
    assert len(result) == 368 and len({r["id"].split("__")[0] for r in result}) == 16
    target = result[0]
    assert target["identity"] == {**before[0]["identity"], **value[2]["after"]}
    assert target["requirements"][0]["assertions"] == [
        assertion("immutable_base_sha256", digest(value[2]["after"]["base"]))
    ]
    assert target["requirements"][1]["admission"] == "REQUIRED"
    assert "pending_admission" not in target["requirements"][1]
    assert "unawarded" in target["identity_resolution"]["note"]
    assert set(target) == set(before[0]) | {"identity_resolution"}
    assert value[0] == original[0] and value[1] == original[1]


@pytest.mark.parametrize(
    "kind",
    [
        "existing_gate",
        "wrong_index",
        "wrong_prior",
        "wrong_prior_identity",
        "wrong_prior_index",
        "scope",
        "duplicate_cell",
        "review_failure",
        "review_findings",
        "missing_review_pin",
        "wrong_origin_review",
        "relative_origin",
        "escaping_origin",
        "foreign_audit",
        "failed_audit",
        "wrong_candidate",
        "extra_identity",
        "missing_source",
        "invalid_source_hash",
        "qualification",
        "gate_credit",
        "pointer_write",
        "missing_support",
        "duplicate_support",
        "unsealed_support",
        "traversal_asset",
        "absolute_asset",
        "missing_asset_inventory",
        "wrong_proof_format",
    ],
)
def test_inconsistent_review_history_or_proof_rejects_without_mutating_cells(tmp_path, kind):
    value = fixture()
    prior, index, proof, audit, review = value
    if kind == "existing_gate":
        index["cells"][proof["cell"]] = {"gates": {"full_schedule": [{}]}}
        proof["prior_index_sha256"] = digest(index)
    elif kind == "wrong_index":
        index["contract_sha256"] = "e" * 64
    elif kind == "wrong_prior":
        proof["prior_contract_sha256"] = "e" * 64
    elif kind == "wrong_prior_identity":
        proof["prior_identity"]["config_sha256"] = "e" * 64
    elif kind == "wrong_prior_index":
        proof["prior_index_sha256"] = "e" * 64
    elif kind == "scope":
        prior["cells"].pop()
    elif kind == "duplicate_cell":
        prior["cells"][-1] = copy.deepcopy(prior["cells"][-2])
    elif kind == "review_failure":
        review["status"] = "FAIL"
    elif kind == "review_findings":
        review["open_findings"] = ["unresolved"]
    elif kind == "missing_review_pin":
        review["reviewed_files"] = {}
    elif kind == "wrong_origin_review":
        review["capture_origins"] = {}
    elif kind == "relative_origin":
        proof["capture_origin"] = "relative"
    elif kind == "escaping_origin":
        proof["capture_origin"] = "/synthetic/../foreign"
    elif kind == "foreign_audit":
        audit["cell"] = "another"
    elif kind == "failed_audit":
        audit["status"] = "FAIL"
    elif kind == "wrong_candidate":
        proof["candidate_contract_sha256"] = "e" * 64
    elif kind == "extra_identity":
        proof["after"]["config_sha256"] = "e" * 64
    elif kind == "missing_source":
        proof["producer_files_sha256"].pop("tools/adapter_campaign/recovered.py")
    elif kind == "invalid_source_hash":
        proof["producer_files_sha256"]["tools/adapter_campaign/recovered.py"] = "not-a-hash"
    elif kind in ("qualification", "gate_credit", "pointer_write"):
        proof[
            {
                "qualification": "qualification_pass",
                "gate_credit": "gate_evidence_emitted",
                "pointer_write": "active_index_written",
            }[kind]
        ] = True
    elif kind == "missing_support":
        proof["supporting_evidence"] = []
    elif kind == "duplicate_support":
        ref = {"path": "actual_audit.json", "sha256": digest(audit)}
        proof["supporting_evidence"] = [ref, ref]
    elif kind == "unsealed_support":
        proof["supporting_evidence"] = [{"path": "missing.json", "sha256": "e" * 64}]
    elif kind == "traversal_asset":
        proof["after"]["base"]["files"] = {"../model": "a" * 64}
    elif kind == "absolute_asset":
        proof["after"]["base"]["files"] = {"/model": "a" * 64}
    elif kind == "missing_asset_inventory":
        proof["after"]["processor"]["files"] = {}
    elif kind == "wrong_proof_format":
        proof["format"] = "hf-segmented-pretraining-identity-resolution-v1"
    cells = copy.deepcopy(prior["cells"])
    before = copy.deepcopy(cells)
    with pytest.raises(EvidenceError):
        apply(tmp_path, value, cells=cells)
    assert cells == before


@pytest.mark.parametrize(
    "kind", ["existing_identity", "changed_config", "missing_asset_gate", "changed_assertion", "duplicate_assertion"]
)
def test_current_compilation_must_still_have_exact_null_identity_and_gates(tmp_path, kind):
    value = fixture()
    cells = copy.deepcopy(value[0]["cells"])
    if kind == "existing_identity":
        cells[0]["identity"]["base"] = value[2]["after"]["base"]
    elif kind == "changed_config":
        cells[0]["identity"]["config_sha256"] = "e" * 64
    elif kind == "missing_asset_gate":
        cells[0]["requirements"].pop()
    elif kind == "changed_assertion":
        cells[0]["requirements"][0]["assertions"][0]["expected"] = "e" * 64
    else:
        cells[0]["requirements"][0]["assertions"] *= 2
    before = copy.deepcopy(cells)
    with pytest.raises(EvidenceError):
        apply(tmp_path, value, cells=cells)
    assert cells == before


def test_stale_resolution_bytes_fail_at_review_boundary(tmp_path):
    value = fixture()
    ref = seal(tmp_path, value)
    (tmp_path / "resolution.json").write_text("{}")
    with pytest.raises(EvidenceError, match="stale evidence"):
        apply_pretraining_identity(
            Store(tmp_path), value[0]["cells"], name="pretraining_identity:" + value[2]["cell"], reference=ref
        )


def test_invalid_amendment_path_cannot_escape_root(tmp_path):
    value = fixture()
    ref = seal(tmp_path, value)
    ref["path"] = "../" + Path(ref["path"]).name
    with pytest.raises(EvidenceError):
        apply_pretraining_identity(
            Store(tmp_path), value[0]["cells"], name="pretraining_identity:" + value[2]["cell"], reference=ref
        )
