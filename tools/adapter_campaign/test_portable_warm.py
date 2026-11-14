# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic receipt predicates only; these fixtures are never campaign evidence."""

from __future__ import annotations

import copy

import pytest

from tools.adapter_campaign.evidence import EvidenceError
from tools.adapter_campaign.portable_warm import CELLS, WORKER, produce, validate_identity, validate_rank


def fixture(layout="fsdp2", rank=0):
    cell = CELLS[layout == "hsdp"]
    aliases = [["lm_head.weight", "model.embed_tokens.weight"]]
    settings = {"method": "dora", "rank": 8, "alpha": 16, "use_rslora": False, "dropout": 0, "dtype": "float32"}
    resolved = {
        "targets": [{"component": "language", "settings": copy.deepcopy(settings), "scaling": 2} for _ in range(60)],
        "extra_parameters": [],
        "expected_trainable_parameters": 180,
    }
    parent = {
        "base": {"sha256": "synthetic-base"},
        "processor": {"sha256": "synthetic-processor"},
        "resolved": resolved,
    }
    peers = [
        {
            "rank": r,
            "host": "synthetic-host-" + str(r // 2),
            "local_rank": r % 2,
            "gpu_uuid": "synthetic-gpu-" + str(r),
            "job": "123",
        }
        for r in range(4)
    ]
    groups = {"0": {"ranks": list(range(4)), "hosts": ["synthetic-host-0", "synthetic-host-1"], "placement": "S(0)"}}
    mesh = {"shape": [4], "coordinate": [rank], "groups": groups}
    if layout == "hsdp":
        mesh = {
            "shape": [2, 2],
            "coordinate": [rank // 2, rank % 2],
            "groups": {
                "0": {
                    "ranks": [rank % 2, rank % 2 + 2],
                    "hosts": ["synthetic-host-0", "synthetic-host-1"],
                    "placement": "R",
                },
                "1": {
                    "ranks": [rank // 2 * 2, rank // 2 * 2 + 1],
                    "hosts": ["synthetic-host-" + str(rank // 2)],
                    "placement": "S(0)",
                },
            },
        }
    names = ["synthetic-owner-" + str(i) for i in range(180)]
    case = {
        "status": "PASS_ACTUAL_TWO_HOST_FRESH_WARM_START",
        "cell": cell,
        "plan_sha256": "synthetic-plan",
        "source_revision": "synthetic-revision",
        "wheel_sha256": "synthetic-wheel",
        "image_sha256": "synthetic-runtime",
        "job": "123",
        "aliases": aliases,
        "parameters": 180,
        "trainable_elements": 180,
        "peers": peers,
        "rank_losses": [[1.0, 0.9, 0.8] for _ in range(4)],
        "base": parent["base"],
        "processor": parent["processor"],
        "parent_portable": "/synthetic/primary/output/primary/audit/final-portable",
        "parent_tensor_sha256": "synthetic-weights",
        "logical_updates": 3,
        "per_rank_cursor": 6,
        "optimizer_moments_reset_then_steps": [1, 2, 3],
        "scheduler_reset_then_steps": [0, 1, 2, 3],
        "dora_magnitudes": 60,
        "unique_examples": 24,
        "full_training_credit": False,
        "resume_credit": False,
        "task_or_deployment_credit": False,
    }
    plan = {
        "cell": cell,
        "source_revision": case["source_revision"],
        "wheel_sha256": case["wheel_sha256"],
        "image_sha256": case["image_sha256"],
        "files_sha256": {"cell.json": "synthetic-config", "final_two_host_warm_v2.py": WORKER},
        "portable_inventory": {
            "adapter.safetensors": "synthetic-weights",
            "adapter_manifest.json": "synthetic-manifest",
        },
        "portable": case["parent_portable"],
        "primary_plan_sha256": "synthetic-primary-plan",
    }
    provenance = {
        "source_revision": case["source_revision"],
        "wheel_sha256": case["wheel_sha256"],
        "container_sha256": case["image_sha256"],
        "campaign_spec_sha256": "synthetic-config",
        "warm_plan_sha256": case["plan_sha256"],
        "harness_sha256": WORKER,
        "slurm_job_id": "123",
        "verified_package_files": 1932,
        "tracked_package_files_verified": 1932,
    }
    initial = {
        "aliases": aliases,
        "adapter_identity": parent,
        "resolved": resolved,
        "optimizer": [{"names": names}],
        "parameter_layouts": dict.fromkeys(names, {}),
        "topology": {"peers": peers, "layouts": {"synthetic-mesh": mesh}},
    }
    warm = {
        "cell": cell,
        "plan_sha256": case["plan_sha256"],
        "original_final_update": 935,
        "status": "PASS_EXACT_FINAL_PORTABLE_WITH_RESET_TRAINING_STATE",
        "warm_update": 0,
        "warm_cursor": 0,
        "negative_snapshot_guards": True,
        "parent_inventory": plan["portable_inventory"],
        "aliases": aliases,
    }
    final = {
        "cell": cell,
        "plan_sha256": case["plan_sha256"],
        "original_final_update": 935,
        "status": "PASS_THREE_TWO_HOST_PUBLIC_WARM_UPDATES",
        "warm_update": 3,
        "warm_cursor": 6,
        "schedulers": [{"step": x} for x in range(4)],
        "ownership_and_aliases_unchanged": True,
        "parent_immutable": True,
        "child_strict_state": True,
        "aliases": aliases,
    }
    completed = {
        "step": 3,
        "frozen_preserved": True,
        "changed": dict.fromkeys(names, True),
        "finite_gradients": list(names),
    }
    public = {"final_step": 3}
    return copy.deepcopy(([provenance, initial, warm, final, completed, public], plan, case, parent))


@pytest.mark.parametrize("layout", ["fsdp2", "hsdp"])
@pytest.mark.parametrize("rank", range(4))
def test_synthetic_rank_receipts_accept_each_real_topology_geometry(layout, rank):
    values, plan, case, parent = fixture(layout, rank)
    validate_rank(values, plan, case, parent, rank)


def replace(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "path,value",
    [
        ((0, "source_revision"), "wrong"),
        ((0, "wheel_sha256"), "wrong"),
        ((0, "container_sha256"), "wrong"),
        ((0, "campaign_spec_sha256"), "wrong"),
        ((0, "warm_plan_sha256"), "wrong"),
        ((0, "harness_sha256"), "wrong"),
        ((0, "slurm_job_id"), "wrong"),
        ((0, "verified_package_files"), 1),
        ((2, "cell"), CELLS[1]),
        ((2, "plan_sha256"), "wrong"),
        ((2, "original_final_update"), 934),
        ((2, "warm_update"), 935),
        ((2, "warm_update"), False),
        ((2, "warm_cursor"), 935),
        ((2, "negative_snapshot_guards"), False),
        ((2, "status"), "NOT_EXECUTED"),
        ((3, "warm_update"), 2),
        ((3, "warm_cursor"), 4),
        ((3, "schedulers"), [{"step": 0}]),
        ((3, "ownership_and_aliases_unchanged"), False),
        ((3, "parent_immutable"), False),
        ((3, "child_strict_state"), False),
        ((3, "aliases"), []),
        ((4, "frozen_preserved"), False),
        ((4, "changed"), {}),
        ((4, "finite_gradients"), []),
        ((4, "changed", "synthetic-owner-0"), False),
        ((5, "final_step"), 2),
        ((1, "optimizer", 0, "names", 0), "synthetic-owner-1"),
        ((1, "topology", "layouts"), {}),
        ((1, "topology", "peers"), []),
        ((1, "topology", "layouts", "synthetic-mesh", "coordinate"), [1]),
        ((1, "topology", "layouts", "synthetic-mesh", "shape"), [2, 2]),
        ((1, "topology", "layouts", "synthetic-mesh", "groups"), {}),
        ((1, "topology", "layouts", "synthetic-mesh", "groups", "0", "ranks"), [0, 0, 2, 3]),
        ((1, "topology", "layouts", "synthetic-mesh", "groups", "0", "hosts"), ["synthetic-host-0"]),
    ],
)
def test_invalid_installed_warm_receipts_never_pass(path, value):
    values, plan, case, parent = fixture()
    # Detach shared fixture identities to make each mutation independent.
    values = [copy.deepcopy(v) for v in values]
    replace(values, path, value)
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, case, parent, 0)


@pytest.mark.parametrize(
    "key,value",
    [("method", "pissa"), ("rank", 4), ("alpha", 8), ("use_rslora", True), ("dropout", 0.1), ("dtype", "bfloat16")],
)
def test_other_methods_and_settings_cannot_inherit_this_warm_proof(key, value):
    values, plan, case, parent = fixture()
    # Deliberately change both copies: method admission must reject even when
    # the two mutable identity documents agree with one another.
    parent["resolved"]["targets"][0]["settings"][key] = value
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, case, parent, 0)


@pytest.mark.parametrize("layout", ["fsdp2", "hsdp"])
def test_duplicate_gradient_owner_and_single_host_emulation_fail(layout):
    values, plan, case, parent = fixture(layout)
    values[4]["finite_gradients"].append(values[4]["finite_gradients"][0])
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, case, parent, 0)
    values, plan, case, parent = fixture(layout)
    for peer in case["peers"]:
        peer["host"] = "synthetic-one-host"
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, case, parent, 0)


def identity_fixture():
    _, plan, case, parent_identity = fixture()
    cell = {
        "id": CELLS[0],
        "identity": {"base": case["base"], "processor": case["processor"], "config_sha256": "synthetic-config"},
    }
    parent = {"kind": "portable_adapter", "identity": parent_identity, "weights_sha256": case["parent_tensor_sha256"]}
    checkpoint = {
        "lineage": {"primary_root": "/synthetic/primary", "primary_plan_sha256": "synthetic-primary-plan"},
        "final_checkpoint": "synthetic-checkpoint",
    }
    measurement = {"checkpoint": {"identity": cell["identity"], "inventory_sha256": checkpoint["final_checkpoint"]}}
    artifacts = {
        "direct_inventory": {name: {"sha256": sha, "bytes": 1} for name, sha in plan["portable_inventory"].items()}
    }
    return tuple(copy.deepcopy(v) for v in (cell, case, plan, parent, checkpoint, measurement, artifacts))


def test_synthetic_final_checkpoint_to_warm_identity_link():
    validate_identity(*identity_fixture())


@pytest.mark.parametrize(
    "path,value",
    [
        ((0, "id"), CELLS[1]),
        ((0, "identity", "config_sha256"), "wrong"),
        ((1, "base"), {}),
        ((1, "processor"), {}),
        ((1, "parent_portable"), "/wrong"),
        ((1, "logical_updates"), 2),
        ((1, "status"), "FAIL"),
        ((1, "optimizer_moments_reset_then_steps"), [936, 937, 938]),
        ((1, "scheduler_reset_then_steps"), [935, 936, 937, 938]),
        ((1, "dora_magnitudes"), 0),
        ((1, "unique_examples"), 23),
        ((1, "full_training_credit"), True),
        ((1, "resume_credit"), True),
        ((1, "task_or_deployment_credit"), True),
        ((2, "primary_plan_sha256"), "wrong"),
        ((2, "portable"), "/wrong"),
        ((3, "kind"), "training_checkpoint"),
        ((3, "weights_sha256"), "wrong"),
        ((4, "final_checkpoint"), "another-checkpoint"),
        ((4, "lineage", "primary_root"), "/another"),
        ((5, "checkpoint", "identity"), {}),
        ((6, "direct_inventory", "adapter.safetensors", "sha256"), "wrong"),
    ],
)
def test_cross_artifact_or_lifecycle_credit_is_rejected(path, value):
    args = list(identity_fixture())
    replace(args, path, value)
    with pytest.raises(EvidenceError):
        validate_identity(*args)


def test_synthetic_contract_cannot_emit_a_real_campaign_gate(tmp_path):
    with pytest.raises(EvidenceError, match="synthetic"):
        produce(
            tmp_path,
            {"synthetic": True},
            tmp_path / "out",
            cell_id=CELLS[0],
            review_ref={},
            checkpoint_admission_ref={},
            checkpoint_capture_ref={},
            producer_review={},
        )
