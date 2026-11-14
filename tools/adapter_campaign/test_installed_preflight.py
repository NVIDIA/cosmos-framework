# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic preflight receipts. These fixtures are never campaign evidence."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.adapter_campaign import installed_preflight as f
from tools.adapter_campaign.evidence import EvidenceError, Store, check_receipt


def fixture(layout="fsdp2", rank=0):
    settings = dict(
        method="dora",
        initialization="default",
        rank=8,
        alpha=16.0,
        use_rslora=False,
        dropout=0.0,
        dtype="float32",
        pissa_niter=4,
        lr_multiplier=1.0,
        weight_decay=0.0,
    )
    targets = [
        {
            "name": f"model.layers.{i}.self_attn.{q}_proj",
            "owner": f"model.layers.{i}.self_attn.{q}_proj",
            "component": "language",
            "type": "torch.nn.Linear",
            "shape": [576 if q == "q" else 192, 576],
            "settings": copy.deepcopy(settings),
            "scaling": 2.0,
        }
        for i in range(30)
        for q in ("q", "v")
    ]
    names = [t["name"] + "." + s for t in targets for s in ("lora_A.weight", "lora_B.weight", "magnitude")]
    manifest = {"targets": targets, "extra_parameters": [], "expected_trainable_parameters": 483840}
    peers = [
        dict(
            rank=r,
            local_rank=r % 2,
            host="host" + str(r // 2),
            gpu_uuid="gpu" + str(r),
            job="123",
            gpu_name="NVIDIA A100-SXM4-80GB",
        )
        for r in range(4)
    ]
    layout_spec = {
        "dtype": "torch.float32",
        "mesh": [[0, 1], [2, 3]] if layout == "hsdp" else [0, 1, 2, 3],
        "shard_dims": [None, 0] if layout == "hsdp" else [0],
    }
    groups = (
        {
            "0": {"placement": "R", "ranks": [rank % 2, rank % 2 + 2], "hosts": ["host0", "host1"]},
            "1": {
                "placement": "S(0)",
                "ranks": [2 * (rank // 2), 2 * (rank // 2) + 1],
                "hosts": ["host" + str(rank // 2)],
            },
        }
        if layout == "hsdp"
        else {"0": {"placement": "S(0)", "ranks": [0, 1, 2, 3], "hosts": ["host0", "host1"]}}
    )
    cell = {
        "id": f.CELLS[layout == "hsdp"],
        "identity": {"base": {"sha256": "a" * 64}, "processor": {"sha256": "b" * 64}},
    }
    spec = {
        "cell": {
            "topology": {"layout": layout},
            "adapter_config": {"components": [{**settings, "execution": "factorized_v1"}]},
        }
    }
    initial = {
        "resolved": manifest,
        "optimizer": [{"names": names}],
        "parameter_layouts": {n: copy.deepcopy(layout_spec) for n in names},
        "adapter_identity": {**copy.deepcopy(cell["identity"]), "resolved": manifest},
        "topology": {
            "peers": peers,
            "layouts": {
                "mesh": {
                    "shape": [2, 2] if layout == "hsdp" else [4],
                    "coordinate": [rank // 2, rank % 2] if layout == "hsdp" else [rank],
                    "groups": groups,
                }
            },
        },
        "initial_clock": {"optimizer_step": 0, "accum": 0},
    }
    completed = {
        "step": 6,
        "status": "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING",
        "frozen_preserved": True,
        "changed": dict.fromkeys(names, True),
        "changed_local": dict.fromkeys(names, True),
        "finite_gradients": names[:],
        "replicas": {"hybrid": True, "trainable_shards": 180, "replicas": 2, "sha256": str(rank % 2) * 64}
        if layout == "hsdp"
        else {"hybrid": False},
    }
    provenance = dict(
        source_revision=f.SOURCE["revision"],
        wheel_sha256=f.SOURCE["wheel_sha256"],
        container_sha256=f.SOURCE["runtime_sha256"],
        source_dirty=False,
        verified_package_files=1932,
        tracked_package_files_verified=1932,
        slurm_job_id="123",
        campaign_spec_sha256="c" * 64,
        harness_sha256="d" * 64,
    )
    public = {"final_step": 6, "preempted": "campaign diagnostic boundary"}
    return (
        initial,
        completed,
        provenance,
        public,
        dict(cell=cell, spec=spec, spec_sha="c" * 64, job="123", steps=6, harness="d" * 64),
    )


@pytest.mark.parametrize("layout", ["fsdp2", "hsdp"])
def test_owned_dora_topology_and_update_proof(layout):
    initial, completed = [], []
    for rank in range(4):
        i, c, p, result, kwargs = fixture(layout, rank)
        names, peers = f.validate_rank(i, c, p, result, **kwargs)
        assert len(names) == 180 and len(peers) == 4
        initial.append(i)
        completed.append(c)
    f.rank_agreement(initial, completed, layout)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i, c, p, r, k: p.update(source_dirty=True),
        lambda i, c, p, r, k: p.update(source_revision="0" * 40),
        lambda i, c, p, r, k: p.update(wheel_sha256="0" * 64),
        lambda i, c, p, r, k: p.update(container_sha256="0" * 64),
        lambda i, c, p, r, k: p.update(verified_package_files=1931),
        lambda i, c, p, r, k: p.update(slurm_job_id="456"),
        lambda i, c, p, r, k: p.update(campaign_spec_sha256="e" * 64),
        lambda i, c, p, r, k: p.update(harness_sha256="e" * 64),
        lambda i, c, p, r, k: c.update(step=5),
        lambda i, c, p, r, k: r.update(final_step=5),
        lambda i, c, p, r, k: c.update(frozen_preserved=False),
        lambda i, c, p, r, k: c["changed"].__setitem__(next(iter(c["changed"])), False),
        lambda i, c, p, r, k: c["finite_gradients"].pop(),
        lambda i, c, p, r, k: i["optimizer"][0]["names"].append(i["optimizer"][0]["names"][0]),
        lambda i, c, p, r, k: i["resolved"]["targets"][0].update(owner="foreign"),
        lambda i, c, p, r, k: i["resolved"]["targets"][0].update(type="Conv2d"),
        lambda i, c, p, r, k: i["resolved"]["targets"][0]["settings"].update(method="lora"),
        lambda i, c, p, r, k: i["resolved"]["targets"][0]["settings"].update(use_rslora=True),
        lambda i, c, p, r, k: i["resolved"]["targets"][0]["settings"].update(initialization="pissa"),
        lambda i, c, p, r, k: i["resolved"]["targets"][0]["settings"].update(dropout=True),
        lambda i, c, p, r, k: i["resolved"]["targets"][0].update(scaling=True),
        lambda i, c, p, r, k: i["resolved"].update(extra_parameters=["norm.weight"]),
        lambda i, c, p, r, k: i["adapter_identity"]["base"].update(sha256="0" * 64),
        lambda i, c, p, r, k: i["adapter_identity"]["processor"].update(sha256="0" * 64),
        lambda i, c, p, r, k: i["topology"]["peers"][2].update(host="host0"),
        lambda i, c, p, r, k: i["topology"]["peers"][0].update(gpu_uuid="gpu1"),
        lambda i, c, p, r, k: i["topology"]["peers"][1].update(rank=True),
        lambda i, c, p, r, k: next(iter(i["parameter_layouts"].values())).update(dtype="torch.bfloat16"),
        lambda i, c, p, r, k: i["initial_clock"].update(optimizer_step=1),
        lambda i, c, p, r, k: i["initial_clock"].update(accum=False),
    ],
)
def test_wrong_installed_owner_method_or_topology_rejected(mutate):
    i, c, p, r, k = fixture()
    mutate(i, c, p, r, k)
    with pytest.raises((EvidenceError, KeyError)):
        f.validate_rank(i, c, p, r, **k)


@pytest.mark.parametrize(
    "kind",
    [
        "rank_manifest",
        "wrong_coordinates",
        "replicate_within_host",
        "shard_cross_host",
        "replica_hash",
        "replica_missing",
    ],
)
def test_hybrid_consensus_rejected(kind):
    values = [fixture("hsdp", r) for r in range(4)]
    i = [v[0] for v in values]
    c = [v[1] for v in values]
    if kind == "rank_manifest":
        i[2]["optimizer"][0]["names"].pop()
    elif kind == "wrong_coordinates":
        i[0]["topology"]["layouts"]["mesh"]["coordinate"] = [1, 0]
    elif kind == "replicate_within_host":
        i[0]["topology"]["layouts"]["mesh"]["groups"]["0"]["hosts"] = ["host0"]
    elif kind == "shard_cross_host":
        i[0]["topology"]["layouts"]["mesh"]["groups"]["1"]["hosts"] = ["host0", "host1"]
    elif kind == "replica_hash":
        c[2]["replicas"]["sha256"] = "f" * 64
    else:
        c[2]["replicas"]["trainable_shards"] = 179
    with pytest.raises(EvidenceError):
        f.rank_agreement(i, c, "hsdp")


def coverage():
    return [
        {
            "step": s,
            "inputs": [{"id": f"synthetic{s}"}],
            "metrics": {
                "train/valid_token_count": f.TOKENS[str(s)],
                "train/valid_sequence_count": 1 if s == 3 else 8,
                "optim/update_outcome": "applied",
                "optim/optimizer_step": s,
                "optim/step_skipped": 0.0,
                "optim/nonfinite_skips_total": 0,
                "train/loss_avg": 1.0,
            },
        }
        for s in (1, 2, 3)
    ]


@pytest.mark.parametrize("layout", ["fsdp2", "hsdp"])
@pytest.mark.parametrize(
    "kind",
    [
        "wrong_ranks",
        "missing_ranks",
        "duplicate_rank",
        "foreign_hosts",
        "missing_group",
        "extra_group",
        "wrong_placement",
        "inconsistent_peer",
        "boolean_rank",
    ],
)
def test_exact_process_group_membership(layout, kind):
    values = [fixture(layout, r) for r in range(4)]
    initial = [v[0] for v in values]
    completed = [v[1] for v in values]
    groups = initial[0]["topology"]["layouts"]["mesh"]["groups"]
    if kind == "wrong_ranks":
        groups["0"]["ranks"] = [0, 1] if layout == "hsdp" else [3, 2, 1, 0]
    elif kind == "missing_ranks":
        groups["0"].pop("ranks")
    elif kind == "duplicate_rank":
        groups["0"]["ranks"][-1] = 0
    elif kind == "foreign_hosts":
        groups["0"]["hosts"] = ["foreign0", "foreign1"]
    elif kind == "missing_group":
        groups.pop("0")
    elif kind == "extra_group":
        groups["foreign"] = copy.deepcopy(groups["0"])
    elif kind == "wrong_placement":
        groups["0"]["placement"] = "S(0)" if layout == "hsdp" else "R"
    elif kind == "inconsistent_peer":
        initial[3]["topology"]["peers"][0]["host"] = "foreign"
    else:
        groups["0"]["ranks"][0] = False
    with pytest.raises(EvidenceError):
        f.rank_agreement(initial, completed, layout)


def test_observed_incomplete_tail_keeps_native_denominator():
    rows = coverage()
    f.validate_coverage(rows, [{k: r[k] for k in ("step", "inputs")} for r in rows])


@pytest.mark.parametrize(
    "kind",
    ["missing", "duplicate", "different_input", "tail_count", "zero_rank_count", "skipped", "nonfinite", "boolean"],
)
def test_coverage_or_objective_rejected(kind):
    rows = coverage()
    expected = [copy.deepcopy({k: r[k] for k in ("step", "inputs")}) for r in rows]
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[2]["step"] = 2
    elif kind == "different_input":
        rows[2]["inputs"][0]["id"] = "different"
    elif kind == "tail_count":
        rows[2]["metrics"]["train/valid_token_count"] = 131 * 4
    elif kind == "zero_rank_count":
        rows[2]["metrics"]["train/valid_token_count"] = 0
    elif kind == "skipped":
        rows[2]["metrics"]["optim/update_outcome"] = "skipped"
    elif kind == "nonfinite":
        rows[2]["metrics"]["train/loss_avg"] = float("nan")
    else:
        rows[2]["metrics"]["train/valid_sequence_count"] = True
    with pytest.raises(EvidenceError):
        f.validate_coverage(rows, expected)


def producer_fixture(tmp_path):
    """Synthetic admission and dummy measurements; no actual capture is forged."""

    def put(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return {"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    old = put("old.json", {})
    new = put("new.json", {})
    files = {
        "tools/adapter_campaign/" + n: hashlib.sha256(Path(f.__file__).with_name(n).read_bytes()).hexdigest()
        for n in f.DEPENDENCIES
    }
    files.update({old["path"]: old["sha256"], new["path"]: new["sha256"]})
    review = put(
        "review.json", {"status": "PASS_NO_OPEN_CONFIRMED_FINDINGS", "open_findings": [], "reviewed_files": files}
    )
    producer = files["tools/adapter_campaign/installed_preflight.py"]
    gate = {
        "id": f.GATE,
        "producers": [producer],
        "producer_reviews": {producer: [review]},
        "sources": [f.SOURCE],
        "profile": None,
        "training_lineage": False,
        "final_checkpoint": False,
        "documents": ["measurement", "provenance"],
        "provenance_document": "provenance",
        "raw_evidence_required": True,
        "assertions": [
            {"document": "measurement", "pointer": "/aliases_preserved", "operator": "eq", "expected": True}
        ],
    }
    contract = {
        "synthetic": False,
        "cells": [{"id": f.CELLS[0], "identity": {"synthetic_fixture": True}, "requirements": [gate]}],
    }
    args = dict(cell_id=f.CELLS[0], original_review=old, supplemental_review=new, producer_review=review)
    return contract, args, gate


def test_producer_round_trip_only_admitted_prerequisite(tmp_path):
    c, args, gate = producer_fixture(tmp_path)
    with patch.object(f, "normalize", return_value=({"aliases_preserved": True}, f.SOURCE, [])):
        index = f.produce(tmp_path, c, tmp_path / "output", **args)
    receipt = index["cells"][f.CELLS[0]]["gates"][f.GATE][0]
    value = check_receipt(Store(tmp_path), receipt, gate, c["cells"][0]["identity"], c, None)
    assert value["outcome"] == "PASS" and list(index["cells"]) == [f.CELLS[0]]
    assert not (tmp_path / "current.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "unadmitted_producer",
        "unadmitted_source",
        "unadmitted_review",
        "producer_bytes",
        "actual_review",
        "open_finding",
    ],
)
def test_producer_cannot_invent_admission(tmp_path, kind):
    c, args, gate = producer_fixture(tmp_path)
    if kind == "unadmitted_producer":
        gate["producers"] = []
    elif kind == "unadmitted_source":
        gate["sources"] = []
    elif kind == "unadmitted_review":
        gate["producer_reviews"] = {gate["producers"][0]: [{"path": "foreign.json", "sha256": "f" * 64}]}
    else:
        p = tmp_path / args["producer_review"]["path"]
        r = json.loads(p.read_text())
        if kind == "producer_bytes":
            r["reviewed_files"]["tools/adapter_campaign/installed_preflight.py"] = "0" * 64
        elif kind == "actual_review":
            r["reviewed_files"].pop("old.json")
        else:
            r["open_findings"] = ["unresolved"]
        p.write_text(json.dumps(r))
        args["producer_review"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    with patch.object(f, "normalize", side_effect=AssertionError("should not execute")), pytest.raises(EvidenceError):
        f.produce(tmp_path, c, tmp_path / "output", **args)
    assert not (tmp_path / "output").exists()
