# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic admission proofs only; actual raw replay lives in the review packet."""

import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools.adapter_campaign import smollm_lora_final_export as subject
from tools.adapter_campaign.evidence import EvidenceError, Store, check_receipt
from tools.adapter_campaign.primary import write


def comparison(merge=False, failed=False):
    atol, rtol, mean = (5e-4, 2e-4, 5e-5) if merge else (2e-5, 2e-5, 2e-6)
    return dict(
        status="FAIL" if failed else "PASS",
        finite=True,
        all_elements_close=not failed,
        all_greedy_equal=True,
        max_absolute_error=mean * 10 if failed else 0.0,
        mean_absolute_error=mean * 2 if failed else 0.0,
        atol=atol,
        rtol=rtol,
        mean_gate=mean,
        shape=[1, 32, 17],
    )


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_rederives_original_predicate_with_distinct_merge_and_cache_gates(merge, failed):
    subject.comparison_outcome(comparison(merge, failed), expected="FAIL" if failed else "PASS", merge=merge)


@pytest.mark.parametrize(
    "kind", ["status", "mean", "finite", "atol", "rtol", "mean_gate", "shape", "boolean", "nan", "negative"]
)
def test_comparison_flags_cannot_mint_pass_or_relax_thresholds(kind):
    row = comparison()
    if kind == "status":
        row["status"] = "FAIL"
    elif kind == "mean":
        row["mean_absolute_error"] = 3e-6
    elif kind == "finite":
        row["finite"] = False
    elif kind in ("atol", "rtol", "mean_gate"):
        row[kind] *= 10
    elif kind == "shape":
        row["shape"] = [True]
    elif kind == "boolean":
        row["all_elements_close"] = 1
    elif kind == "nan":
        row["mean_absolute_error"] = float("nan")
    else:
        row["max_absolute_error"] = -1.0
    with pytest.raises(EvidenceError):
        subject.comparison_outcome(row, expected="PASS")


def tensor_fixture():
    weight_map = {f"tensor_{i}": f"model-{i % 3 + 1:05d}-of-00003.safetensors" for i in range(272)}
    proof = dict(
        all_tensors_exact=True,
        complete_assets_exact=True,
        identity_exact=True,
        tensor_count=272,
        tensors=[dict(name=n, shape=[2, 3], dtype="torch.float32", exact=True) for n in weight_map],
    )
    return proof, weight_map


def test_complete_tensor_census_checks_keys_shapes_precision_and_reuse():
    proof, weight_map = tensor_fixture()
    schema = subject.tensor_census(proof, weight_map)
    assert subject.tensor_census(proof, weight_map, schema) == schema


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "foreign",
        "count_bool",
        "count",
        "dtype",
        "inexact",
        "identity",
        "assets",
        "shape_bool",
        "shape_negative",
        "shape_zero",
        "empty_shape",
        "extra_field",
        "extra_shard",
        "wrong_dimensions",
    ],
)
def test_tensor_census_refuses_partial_malformed_or_wrong_export(kind):
    proof, weight_map = tensor_fixture()
    before = subject.tensor_census(proof, weight_map)
    if kind == "missing":
        proof["tensors"].pop()
    elif kind == "duplicate":
        proof["tensors"][1] = copy.deepcopy(proof["tensors"][0])
    elif kind == "foreign":
        proof["tensors"][0]["name"] = "other"
    elif kind.startswith("count"):
        proof["tensor_count"] = True if kind == "count_bool" else 271
    elif kind == "dtype":
        proof["tensors"][0]["dtype"] = "torch.bfloat16"
    elif kind == "inexact":
        proof["tensors"][0]["exact"] = False
    elif kind in ("identity", "assets"):
        proof[{"identity": "identity_exact", "assets": "complete_assets_exact"}[kind]] = False
    elif kind.startswith("shape") or kind == "empty_shape":
        proof["tensors"][0]["shape"] = {
            "shape_bool": [True],
            "shape_negative": [-1],
            "shape_zero": [0],
            "empty_shape": [],
        }[kind]
    elif kind == "extra_field":
        proof["tensors"][0]["invented"] = True
    elif kind == "extra_shard":
        weight_map["tensor_0"] = "model-4.safetensors"
    else:
        proof["tensors"][0]["shape"] = [3, 2]
    with pytest.raises(EvidenceError):
        subject.tensor_census(proof, weight_map, before)


def command_fixture():
    return dict(
        returncode=-9,
        exception=None,
        cleanup_error=None,
        cleanup={"remaining": []},
        samples=9,
        peak_owned_process_tree_rss_bytes=1024,
        seconds=1.5,
        memory_scope="controller RSS excluded",
    )


@pytest.mark.parametrize(
    "kind",
    ["rc", "bool_rc", "exception", "cleanup_error", "remaining", "peak", "peak_bool", "samples", "seconds", "scope"],
)
def test_owned_interruption_requires_cleanup_and_real_peaks(kind):
    row = command_fixture()
    subject.clean_command(row, -9)
    key, value = {
        "rc": ("returncode", 0),
        "bool_rc": ("returncode", False),
        "exception": ("exception", "timeout"),
        "cleanup_error": ("cleanup_error", "failure"),
        "remaining": ("cleanup", {"remaining": [123]}),
        "peak": ("peak_owned_process_tree_rss_bytes", 0),
        "peak_bool": ("peak_owned_process_tree_rss_bytes", True),
        "samples": ("samples", False),
        "seconds": ("seconds", float("inf")),
        "scope": ("memory_scope", "allocation ceiling"),
    }[kind]
    row[key] = value
    with pytest.raises(EvidenceError):
        subject.clean_command(row, -9)


def fixture(root):
    producer = hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest()
    refs = {
        n: write(root, root / (n + ".json"), {"fixture": "synthetic", "name": n})
        for n in ("extraction", "checkpoint_admission", "checkpoint_capture")
    }
    review = {
        "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
        "open_findings": [],
        "reviewed_files": {
            "tools/adapter_campaign/" + n: hashlib.sha256(Path(subject.__file__).with_name(n).read_bytes()).hexdigest()
            for n in subject.DEPENDENCIES
        },
    }
    review["reviewed_files"].update({r["path"]: r["sha256"] for r in refs.values()})
    review_ref = write(root, root / "review.json", review)
    identity = {"cell_id": subject.CELL}
    lineage = {"primary_job": "synthetic", "primary_root": "/synthetic", "primary_plan_sha256": "d" * 64}
    requirement = dict(
        id=subject.GATE,
        profile=None,
        final_checkpoint=True,
        documents=["measurement", "provenance"],
        provenance_document="provenance",
        raw_evidence_required=True,
        training_lineage=True,
        producers=[producer],
        producer_reviews={producer: [review_ref]},
        sources=[subject.SOURCE],
        assertions=[
            dict(document="measurement", pointer="/" + n, operator="eq", expected=True) for n in subject.ASSERTIONS
        ],
    )
    contract = {
        "synthetic": False,
        "cells": [{"id": subject.CELL, "identity": identity, "requirements": [requirement]}],
    }
    measurement = {n: True for n in subject.ASSERTIONS}
    measurement["preserved_numerical_failures"] = {
        "full_cached_failures": 34,
        "split_prefix_failures": 34,
        "merge_comparison_paths_failed": 2,
        "stock_returncode": 1,
    }
    normalized = (measurement, subject.SOURCE, lineage, "f" * 64, list(refs.values()))
    args = {n + "_ref": r for n, r in refs.items()}
    args["producer_review"] = review_ref
    return contract, args, normalized, requirement


def test_typed_output_is_only_export_and_retains_original_failures(tmp_path):
    contract, args, normalized, requirement = fixture(tmp_path)
    with mock.patch.object(subject, "normalize", return_value=normalized):
        index = subject.produce(tmp_path, contract, tmp_path / "new", **args)
    cell = index["cells"][subject.CELL]
    assert set(cell["gates"]) == {subject.GATE} and cell["final_checkpoint"] == normalized[3]
    store = Store(tmp_path)
    receipt_ref = cell["gates"][subject.GATE][0]
    assert (
        check_receipt(
            store, receipt_ref, requirement, contract["cells"][0]["identity"], contract, normalized[3], normalized[2]
        )["outcome"]
        == "PASS"
    )
    receipt = store.json(receipt_ref)
    measurement = store.json(receipt["documents"]["measurement"])
    assert measurement["preserved_numerical_failures"] == normalized[0]["preserved_numerical_failures"]
    assert not (tmp_path / "campaign-final-gates-current.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "synthetic",
        "outside",
        "existing",
        "unreviewed",
        "finding",
        "dependency",
        "extraction",
        "checkpoint_admission",
        "checkpoint_capture",
        "producer",
        "review",
        "source",
    ],
)
def test_produce_rejects_unreviewed_source_or_prerequisite(tmp_path, kind):
    contract, args, normalized, requirement = fixture(tmp_path)
    output = tmp_path / "new"
    if kind == "synthetic":
        contract["synthetic"] = True
    elif kind == "outside":
        output = tmp_path.parent / "outside"
    elif kind == "existing":
        output.mkdir()
    elif kind in ("producer", "review", "source"):
        requirement[{"producer": "producers", "review": "producer_reviews", "source": "sources"}[kind]] = (
            {} if kind == "review" else []
        )
    else:
        review = json.loads((tmp_path / args["producer_review"]["path"]).read_bytes())
        if kind == "unreviewed":
            review["status"] = "PENDING"
        elif kind == "finding":
            review["open_findings"] = ["unresolved"]
        elif kind == "dependency":
            review["reviewed_files"]["tools/adapter_campaign/final_checkpoint.py"] = "0" * 64
        else:
            review["reviewed_files"][kind + ".json"] = "0" * 64
        args["producer_review"] = write(tmp_path, tmp_path / "changed-review.json", review)
    with mock.patch.object(subject, "normalize", return_value=normalized), pytest.raises(EvidenceError):
        subject.produce(tmp_path, contract, output, **args)
    assert not (output / "index.json").exists()


def test_produce_rechecks_support_after_normalization(tmp_path):
    contract, args, normalized, _ = fixture(tmp_path)

    def altered(*a, **kw):
        (tmp_path / args["extraction_ref"]["path"]).write_text("changed after normalization")
        return normalized

    with (
        mock.patch.object(subject, "normalize", side_effect=altered),
        pytest.raises(EvidenceError, match="stale evidence"),
    ):
        subject.produce(tmp_path, contract, tmp_path / "new", **args)
    assert not (tmp_path / "new").exists()
