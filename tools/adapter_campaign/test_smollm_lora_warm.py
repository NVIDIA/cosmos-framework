# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic receipt predicates, never real campaign execution evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tools.adapter_campaign.evidence import EvidenceError
from tools.adapter_campaign.smollm_lora_warm import CELL, JOB, SOURCE, WORKER, Capture, produce, validate_rank


def fixture(rank=0):
    owners = [f"model.layer{i:03d}.lora_{letter}.weight" for i in range(60) for letter in ("A", "B")]
    target_names = [f"model.layer{i:03d}" for i in range(60)]
    clock = dict(
        iter=0,
        optimizer_step=0,
        accum=0,
        n_mini=1,
        attempt_id=0,
        nonfinite_skips_total=0,
        nonfinite_skips_consecutive=0,
        optimizer_pristine=1,
    )
    resolved = {"targets": target_names, "extra_parameters": [], "expected_trainable_parameters": 460800}
    identity = {"base": "synthetic-base", "processor": "synthetic-processor", "resolved": resolved}
    hosts = ["synthetic-node0", "synthetic-node1"]
    journal = {"intent.json": "synthetic-intent", "submission.json": "synthetic-ack"}
    initial = {
        "adapter_identity": identity,
        "resolved": resolved,
        "parameter_layouts": dict.fromkeys(owners, {}),
        "optimizer": [{"names": owners, "lr": 1e-4, "weight_decay": 0}],
        "initial_clock": clock,
        "strict_determinism": True,
        "data_identity": "synthetic-data",
        "finite_coverage": "synthetic-cursor",
        "validation_identity": "synthetic-validation",
        "frozen_sha256": "synthetic-frozen-state",
        "topology": {
            "peers": [
                dict(
                    rank=i,
                    local_rank=i % 2,
                    node_id=i // 2,
                    host=hosts[i // 2],
                    job=JOB,
                    gpu_uuid=f"synthetic-gpu-{i}",
                    gpu_name="synthetic A100",
                    driver="synthetic-driver",
                )
                for i in range(4)
            ],
            "layouts": {
                "[0,1,2,3]": {
                    "shape": [4],
                    "coordinate": [rank],
                    "groups": {"0": {"ranks": [0, 1, 2, 3], "hosts": hosts, "placement": "S(0)"}},
                }
            },
        },
    }
    schedulers = [
        dict(step=i, identity={"schedule": "synthetic-schedule"}, groups=[{"names": owners}]) for i in range(4)
    ]
    common = dict(
        cell=CELL,
        plan_sha256="synthetic-plan",
        original_final_update=935,
        owner_names=owners,
        target_names=target_names,
        journal_sha256=journal,
    )
    aliases = [["model.embed_tokens.weight", "lm_head.weight"]]
    start = {
        **common,
        "status": "PASS_EXACT_FINAL_PORTABLE_WITH_RESET_TRAINING_STATE",
        "warm_update": 0,
        "warm_cursor": 0,
        "initial_clock": clock,
        "optimizer_empty": True,
        "negative_snapshot_guards": True,
        "parent_inventory": {"adapter.safetensors": "synthetic-adapter"},
        "aliases": aliases,
        "initial_scheduler": schedulers[0],
    }
    final = {
        **common,
        "status": "PASS_THREE_TWO_HOST_PUBLIC_WARM_UPDATES",
        "warm_update": 3,
        "warm_cursor": 6,
        "clock": {k: v for k, v in clock.items() if k != "optimizer_pristine"},
        "optimizer_steps": [3],
        "aliases": aliases,
        "ownership_and_aliases_unchanged": True,
        "parent_immutable": True,
        "child_strict_state": True,
        "schedulers": schedulers,
    }
    final["clock"].update(iter=3, optimizer_step=3, attempt_id=3)
    done = dict(step=3, frozen_preserved=True, changed=dict.fromkeys(owners, True), finite_gradients=owners)
    public = {"final_step": 3}
    provenance = dict(
        source_revision=SOURCE["revision"],
        wheel_sha256=SOURCE["wheel_sha256"],
        container_sha256=SOURCE["runtime_sha256"],
        warm_plan_sha256="synthetic-plan",
        campaign_spec_sha256="synthetic-config",
        harness_sha256=WORKER,
        slurm_job_id=JOB,
        verified_package_files=1932,
        tracked_package_files_verified=1932,
    )
    plan = dict(
        owner_names=owners,
        target_names=target_names,
        adapter_identity=identity,
        initial_clock=clock,
        parent_initial_by_rank={str(rank): initial},
        actual_plan_sha256="synthetic-plan",
        portable_inventory=start["parent_inventory"],
        files_sha256={"cell.json": "synthetic-config", "smol_final_warm_worker_v2.py": WORKER},
    )
    # Independent copies intentionally prevent shared Python fixtures from hiding
    # an inconsistency between recorded start, final, plan and parent states.
    return (
        [copy.deepcopy(v) for v in (initial, start, final, done, public, provenance)],
        copy.deepcopy(plan),
        hosts,
        journal,
    )


@pytest.mark.parametrize("rank", range(4))
def test_synthetic_rank_geometry_and_reset(rank):
    values, plan, hosts, journal = fixture(rank)
    validate_rank(values, plan, rank, hosts, journal)


def replace(value, path, replacement):
    for part in path[:-1]:
        value = value[part]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "path,replacement",
    [
        ((0, "strict_determinism"), False),
        ((0, "initial_clock", "optimizer_pristine"), True),
        ((0, "frozen_sha256"), "changed"),
        ((0, "data_identity"), "other split"),
        ((0, "validation_identity"), "other validation"),
        ((0, "finite_coverage"), "other cursor"),
        ((0, "optimizer", 0, "names"), []),
        ((0, "parameter_layouts"), {}),
        ((0, "adapter_identity", "base"), "other base"),
        ((1, "warm_update"), 935),
        ((1, "warm_update"), False),
        ((1, "warm_cursor"), 6),
        ((1, "optimizer_empty"), False),
        ((1, "negative_snapshot_guards"), False),
        ((1, "parent_inventory"), {}),
        ((1, "original_final_update"), 934),
        ((1, "cell"), CELL.replace("__lora__", "__dora__")),
        ((1, "plan_sha256"), "another warm plan"),
        ((1, "journal_sha256"), {}),
        ((2, "clock", "optimizer_step"), 2),
        ((2, "clock", "nonfinite_skips_total"), 1),
        ((2, "optimizer_steps"), [935]),
        ((2, "warm_cursor"), 5),
        ((2, "aliases"), []),
        ((2, "ownership_and_aliases_unchanged"), False),
        ((2, "parent_immutable"), False),
        ((2, "child_strict_state"), False),
        ((2, "schedulers", 1, "step"), 2),
        ((2, "schedulers", 2, "groups", 0, "names"), []),
        ((2, "schedulers", 1, "identity"), {}),
        ((3, "changed", "model.layer000.lora_A.weight"), False),
        ((3, "finite_gradients"), []),
        ((3, "frozen_preserved"), False),
        ((4, "final_step"), 2),
        ((5, "source_revision"), "different implementation"),
        ((5, "wheel_sha256"), "different wheel"),
        ((5, "container_sha256"), "different runtime"),
        ((5, "harness_sha256"), "old mutating observer"),
        ((5, "tracked_package_files_verified"), 0),
        ((5, "slurm_job_id"), "predecessor failed job"),
        ((0, "topology", "peers", 1, "gpu_uuid"), "synthetic-gpu-0"),
        ((0, "topology", "peers", 2, "host"), "synthetic-node0"),
        ((0, "topology", "peers", 2, "local_rank"), 1),
        ((0, "topology", "peers", 0, "rank"), False),
        ((0, "topology", "peers", 0, "driver"), "different driver"),
        ((0, "topology", "layouts", "[0,1,2,3]", "shape"), [2, 2]),
        ((0, "topology", "layouts", "[0,1,2,3]", "coordinate"), [1]),
        ((0, "topology", "layouts", "[0,1,2,3]", "groups", "0", "placement"), "R"),
        ((0, "topology", "layouts", "[0,1,2,3]", "groups", "0", "ranks"), [0, 1]),
        ((0, "topology", "layouts", "[0,1,2,3]", "groups", "0", "hosts"), ["synthetic-node0"]),
    ],
)
def test_contradictory_raw_rank_records_fail(path, replacement):
    values, plan, hosts, journal = fixture()
    replace(values, path, replacement)
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, 0, hosts, journal)


def test_duplicate_gradient_and_optimizer_membership_fail():
    for record, key in ((3, "finite_gradients"), (0, "optimizer")):
        values, plan, hosts, journal = fixture()
        members = values[record][key] if record == 3 else values[record][key][0]["names"]
        members[-1] = members[0]
        with pytest.raises(EvidenceError):
            validate_rank(values, plan, 0, hosts, journal)


def test_host_aliases_cannot_emulate_two_physical_nodes():
    values, plan, _, journal = fixture()
    with pytest.raises(EvidenceError):
        validate_rank(values, plan, 0, ["same-host", "same-host.example"], journal)


def test_synthetic_contract_cannot_emit_campaign_credit(tmp_path):
    with pytest.raises(EvidenceError, match="fresh real"):
        produce(
            tmp_path,
            {"synthetic": True},
            tmp_path / "output",
            extraction_ref={},
            checkpoint_admission_ref={},
            checkpoint_capture_ref={},
            producer_review={},
        )
    assert not (tmp_path / "output").exists()


def test_capture_cannot_inherit_a_synthetic_or_escaping_review(tmp_path):
    with pytest.raises(EvidenceError, match="rooted relative"):
        Capture(tmp_path, {"path": "../outside.json", "sha256": "0" * 64})


def capture_fixture(root):
    """Synthetic copy-map only; it cannot satisfy normalize's model/run gates."""

    def save(name, value):
        raw = json.dumps(value).encode()
        (root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    source = {"status": "PASS_NO_OPEN_CONFIRMED_FINDINGS", "open_findings": [], "reviewed_files": {}}
    source_ref = save("source.json", source)
    cap = {
        "format": "smol-lora-final-warm-terminal-capture-v3",
        "job": JOB,
        "namespace_unchanged": True,
        "all_files_freshly_hashed": True,
        "new_gpu_executions": 0,
        "qualification_pass": False,
        "mirrored_small_files": [],
    }
    cap_ref = save("capture.json", cap)
    actual = {
        "format": "smol-own-final-warm-independent-actual-review-v1",
        "status": "PASS_OWN_FINAL_WARM_ONLY",
        "open_findings": [],
        "job": JOB,
        "full_campaign_pass": False,
        "new_full_schedule_credit": 0,
        "reviewed_files": {
            "/synthetic/source.json": source_ref["sha256"],
            "/synthetic/capture.json": cap_ref["sha256"],
        },
    }
    actual_ref = save("actual.json", actual)
    extraction = {
        "format": "reviewed-smol-lora-warm-copy-v1",
        "synthetic": False,
        "aliases": {"/synthetic/source.json": source_ref, "/synthetic/capture.json": cap_ref},
        "actual_review": actual_ref,
        "original_capture": "/synthetic/capture.json",
        "original_source_review": "/synthetic/source.json",
        "original_files_directory": "/synthetic/files",
    }
    return extraction, save


def test_copy_map_checks_exact_reviewed_bytes_without_claiming_execution(tmp_path):
    extraction, save = capture_fixture(tmp_path)
    value = Capture(tmp_path, save("extraction.json", extraction))
    assert len(value.verified) == 4
    assert value.capture["qualification_pass"] is False


@pytest.mark.parametrize(
    "bad",
    [
        "synthetic",
        "wrong_format",
        "missing_alias",
        "extra_alias",
        "escaping_alias",
        "absolute_alias",
        "stale_source_hash",
        "source_mutation",
        "wrong_review_job",
        "review_open_finding",
        "wrong_capture_job",
        "changed_namespace",
        "full_credit",
    ],
)
def test_bad_review_copy_cannot_be_normalized(tmp_path, bad):
    extraction, save = capture_fixture(tmp_path)
    if bad == "synthetic":
        extraction["synthetic"] = True
    elif bad == "wrong_format":
        extraction["format"] = "another collector"
    elif bad == "missing_alias":
        del extraction["aliases"]["/synthetic/source.json"]
    elif bad == "extra_alias":
        extraction["aliases"]["/synthetic/unreviewed"] = extraction["aliases"]["/synthetic/source.json"]
    elif bad in ("escaping_alias", "absolute_alias"):
        extraction["aliases"]["/synthetic/source.json"]["path"] = (
            "../source.json" if bad == "escaping_alias" else "/source.json"
        )
    elif bad == "stale_source_hash":
        extraction["aliases"]["/synthetic/source.json"]["sha256"] = "0" * 64
    elif bad == "source_mutation":
        (tmp_path / "source.json").write_text("{}")
    elif bad in ("wrong_review_job", "review_open_finding", "full_credit"):
        value = json.loads((tmp_path / "actual.json").read_bytes())
        key, replacement = {
            "wrong_review_job": ("job", "failed predecessor"),
            "review_open_finding": ("open_findings", ["missing evidence"]),
            "full_credit": ("full_campaign_pass", True),
        }[bad]
        value[key] = replacement
        extraction["actual_review"] = save("changed-review.json", value)
    else:
        # Even a rehashed copy cannot change the review's recorded capture hash.
        value = json.loads((tmp_path / "capture.json").read_bytes())
        key, replacement = ("job", "another job") if bad == "wrong_capture_job" else ("namespace_unchanged", False)
        value[key] = replacement
        extraction["aliases"]["/synthetic/capture.json"] = save("changed-capture.json", value)
    with pytest.raises(EvidenceError):
        Capture(tmp_path, save("changed-extraction.json", extraction))
