# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic producer/admission controls. None is actual campaign evidence."""

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.adapter_campaign import hf_segmented as h
from tools.adapter_campaign.evidence import EvidenceError, Store, digest
from tools.adapter_campaign.primary import write


def journal_fixture(operation="allocation"):
    payload, plan = "a" * 64, "b" * 64 if operation == "allocation" else None
    review = {"path": "/synthetic/review.json", "sha256": "c" * 64}
    index = 0 if operation == "allocation" else None
    script = "/synthetic/segment-00.sbatch" if index is not None else "/synthetic/prepare.sbatch"
    export = f"ALL,NATIVE_FULL_PAYLOAD_SHA256={payload},NATIVE_FULL_REVIEW_SHA256={review['sha256']}"
    if index is not None:
        export += f",NATIVE_SEGMENT_PLAN_SHA256={plan},NATIVE_FULL_ATTEMPT=segment-0"
    command = ["sbatch", "--parsable", "--export=" + export, script]
    intent = {
        "operation": operation,
        "index": index,
        "review": review,
        "execution": None,
        "payload_sha256": payload,
        "plan_sha256": plan,
        "command": command,
    }
    ack = {
        "returncode": 0,
        "job_id": "123",
        "stdout": "123\n",
        "stderr": "",
        "command": copy.deepcopy(command),
        "payload_sha256": payload,
        "plan_sha256": plan,
    }
    kwargs = dict(
        job="123",
        payload=payload,
        plan=plan,
        operation=operation,
        index=index,
        script=script,
        review=copy.deepcopy(review),
    )
    return intent, ack, kwargs


@pytest.mark.parametrize("operation", ["prepare", "allocation"])
def test_exact_scheduler_journal(operation):
    intent, ack, kw = journal_fixture(operation)
    h.validate_journal(intent, ack, **kw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i, a: a.update(returncode=False),
        lambda i, a: a.update(returncode="0"),
        lambda i, a: a.update(returncode=1),
        lambda i, a: a.update(job_id="124"),
        lambda i, a: a.update(stdout="123;cluster\n"),
        lambda i, a: a.update(stderr="uncertain"),
        lambda i, a: a.update(plan_sha256="d" * 64),
        lambda i, a: i.update(execution={}),
        lambda i, a: i.update(index=False),
        lambda i, a: i.update(operation="continuation"),
        lambda i, a: i["review"].update(sha256="d" * 64),
        lambda i, a: i.update(payload_sha256="d" * 64),
        lambda i, a: (
            i["command"].__setitem__(-1, "/foreign/job.sbatch"),
            a["command"].__setitem__(-1, "/foreign/job.sbatch"),
        ),
    ],
)
def test_scheduler_uncertainty_or_identity_mismatch_rejects(mutate):
    intent, ack, kw = journal_fixture()
    mutate(intent, ack)
    with pytest.raises(EvidenceError):
        h.validate_journal(intent, ack, **kw)


def predecessor_fixture():
    refs = [{"path": "/synthetic/segment-0/result.json", "sha256": "a" * 64}]
    previous = {"next_predecessor": {"checkpoint": "/synthetic/iter_000350", "checkpoint_inventory": {"x": "a"}}}
    p = {
        "format": "public-hf-wts-full-schedule-segment-v1",
        "phase": "training",
        "purpose": "authoritative_full_schedule",
        "start": 350,
        "end": 600,
        "global_updates": 695,
        "previous": {**previous["next_predecessor"], "receipt": copy.deepcopy(refs[0])},
    }
    r = {
        "phase": "training",
        "start": 350,
        "end": 600,
        "status": "PASS_PLANNED_SEGMENT_ONLY",
        "full_schedule_pass": False,
        "lineage_receipts": copy.deepcopy(refs),
    }
    return p, r, previous, refs


def test_continuation_extends_the_complete_owned_chain():
    h.validate_predecessor(*predecessor_fixture(), start=350, end=600, updates=695)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p, r: p.update(start=349),
        lambda p, r: p.update(end=601),
        lambda p, r: p.update(global_updates=600),
        lambda p, r: p.update(phase="branch"),
        lambda p, r: p.update(purpose="diagnostic"),
        lambda p, r: p.update(previous=None),
        lambda p, r: p["previous"].update(checkpoint="/synthetic/iter_000349"),
        lambda p, r: p["previous"]["receipt"].update(sha256="b" * 64),
        lambda p, r: r.update(status="FAIL"),
        lambda p, r: r.update(full_schedule_pass=True),
        lambda p, r: r.update(lineage_receipts=[]),
    ],
)
def test_omitted_or_foreign_predecessor_rejected(mutate):
    p, r, previous, refs = predecessor_fixture()
    mutate(p, r)
    with pytest.raises(EvidenceError):
        h.validate_predecessor(p, r, previous, refs, start=350, end=600, updates=695)


def phase_fixture():
    phases = [{"updates": [1, 2]}, {"updates": [3, 5]}, {"updates": [6, 7]}]
    logs = [[[{"step": s} for s in range(a, b + 1)] for _ in range(4)] for a, b in [v["updates"] for v in phases]]
    return phases, logs


def test_variable_allocation_boundaries_join_once():
    joined = h.join_phases(*phase_fixture(), 7)
    assert len(joined) == 4 and all([r["step"] for r in rank] == list(range(1, 8)) for rank in joined)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p, l: p[1].update(updates=[2, 5]),
        lambda p, l: p[1].update(updates=[4, 5]),
        lambda p, l: p.append(p[-1]),
        lambda p, l: l.pop(),
        lambda p, l: l[1].pop(),
        lambda p, l: l[1][0].pop(),
        lambda p, l: l[1][0].reverse(),
        lambda p, l: l[2][3].append({"step": 7}),
        lambda p, l: p[-1].update(updates=[6, 6]),
    ],
)
def test_replays_gaps_missing_ranks_never_become_primary(mutate):
    p, l = phase_fixture()
    mutate(p, l)
    with pytest.raises(EvidenceError):
        h.join_phases(p, l, 7)


def saved_fixture():
    rows = []
    for rank in range(4):
        rows.append(
            {
                "kind": "midpoint_restore",
                "rank": rank,
                "step": 347,
                "compared_fields": sorted(h.PERSISTENT),
                "bitwise_equal": True,
                "transient_empty_fields_checked": True,
            }
        )
        rows.extend(
            {
                "kind": "uninterrupted_next_update",
                "rank": rank,
                "step": s,
                "compared_fields": sorted(h.UPDATE_FIELDS),
                "bitwise_equal": True,
            }
            for s in (348, 349, 350)
        )
        rows.extend(
            {
                "kind": "allocation_restore",
                "rank": rank,
                "step": s,
                "compared_fields": sorted(h.PERSISTENT),
                "bitwise_equal": True,
            }
            for s in (350, 600)
        )
        rows.append(
            {"kind": "final_owner_updates", "rank": rank, "step": 695, "changed_owners": 177, "changed_magnitudes": 59}
        )
    saved = {
        "status": "PASS_EXACT_PERSISTED_CONTINUATION_AND_OWNER_UPDATES",
        "cell": "synthetic",
        "qualification_pass": False,
        "new_gpu_executions": 0,
        "comparisons": rows,
    }
    independent = {
        "saved_comparisons_repeated": 28,
        "saved_controls_rehashed": 68,
        "final_checkpoint_control_metadata_checked": True,
        "final_validation_status": "PASS_LOCAL_FINAL_VALIDATION_COMMITTED",
        "replica_consistency": [
            {"point": p, "ranks": list(r), "all_owned_parameters_moments_groups_clock_scheduler_exact": True}
            for p in ("initial", "midpoint347", "boundary350", "boundary600", "final695")
            for r in ((0, 2), (1, 3))
        ],
    }
    return saved, independent


def saved_check(saved, independent):
    h.validate_saved_audit(
        saved,
        independent,
        cell="synthetic",
        boundaries=[0, 350, 600, 695],
        midpoint=347,
        updates=695,
        owner_count=177,
        magnitudes=59,
    )


def test_complete_saved_optimizer_rng_and_hsdp_replica_population():
    saved_check(*saved_fixture())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, i: s.update(status="FAIL"),
        lambda s, i: s.update(cell="foreign"),
        lambda s, i: s.update(qualification_pass=True),
        lambda s, i: s["comparisons"].pop(),
        lambda s, i: s["comparisons"][0].update(bitwise_equal=False),
        lambda s, i: s["comparisons"][0].update(transient_empty_fields_checked=False),
        lambda s, i: s["comparisons"][0]["compared_fields"].remove("rng"),
        lambda s, i: s["comparisons"][1]["compared_fields"].remove("gradients"),
        lambda s, i: s["comparisons"][0].update(step=348),
        lambda s, i: s["comparisons"][6].update(changed_magnitudes=58),
        lambda s, i: i.update(saved_comparisons_repeated=27),
        lambda s, i: i.update(final_checkpoint_control_metadata_checked=False),
        lambda s, i: i.update(final_validation_status="PENDING"),
        lambda s, i: i["replica_consistency"].pop(),
        lambda s, i: i["replica_consistency"][0].update(ranks=[0, 1]),
        lambda s, i: i["replica_consistency"][0].update(
            all_owned_parameters_moments_groups_clock_scheduler_exact=False
        ),
    ],
)
def test_partial_or_negative_saved_state_is_not_recovery_pass(mutate):
    saved, independent = saved_fixture()
    mutate(saved, independent)
    with pytest.raises(EvidenceError):
        saved_check(saved, independent)


def owner_fixture():
    settings = {
        "method": "dora",
        "initialization": "default",
        "rank": 8,
        "alpha": 16.0,
        "use_rslora": False,
        "dropout": 0.0,
        "dtype": "float32",
        "pissa_niter": 4,
        "lr_multiplier": 0.5,
        "weight_decay": 0.0,
    }
    names = ["projection.lora_A.weight", "projection.lora_B.weight", "projection.magnitude"]
    cell = {
        "adapter_config": {
            "components": [
                {**settings, "targets": ["projection"], "execution": "factorized_v1", "component": "projector"}
            ],
            "extra_modules": [],
        },
        "resolved_target_names": ["projection"],
        "model_execution": "ordered",
        "topology": {"layout": "hsdp"},
    }
    initial = {
        "resolved": {
            "targets": [
                {
                    "name": "projection",
                    "owner": "projection",
                    "component": "projector",
                    "settings": settings,
                    "scaling": 2.0,
                }
            ],
            "extra_parameters": [],
            "model_execution": "ordered",
        },
        "optimizer": [{"names": names}],
        "parameter_layouts": {
            n: {"mesh": [[0, 1], [2, 3]], "shard_dims": [None, 0], "dtype": "torch.float32"} for n in names
        },
    }
    return initial, cell


def test_own_dora_target_rank_scaling_and_magnitude_owners():
    assert h.validate_owners(*owner_fixture()) == (3, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i, c: i["optimizer"][0]["names"].pop(),
        lambda i, c: i["optimizer"][0]["names"].append("frozen.base"),
        lambda i, c: i["resolved"]["targets"][0].update(owner="foreign"),
        lambda i, c: i["resolved"]["targets"][0]["settings"].update(rank=4),
        lambda i, c: i["resolved"]["targets"][0].update(scaling=1.0),
        lambda i, c: i["resolved"].update(model_execution="another_profile"),
        lambda i, c: i["resolved"].update(extra_parameters=["extra"]),
        lambda i, c: i["parameter_layouts"]["projection.magnitude"].update(mesh=[0, 1, 2, 3]),
        lambda i, c: i["parameter_layouts"]["projection.magnitude"].update(dtype="torch.bfloat16"),
        lambda i, c: c["adapter_config"]["components"][0].update(execution="dense_v1"),
    ],
)
def test_owner_rank_profile_or_sharding_changes_rejected(mutate):
    i, c = owner_fixture()
    mutate(i, c)
    with pytest.raises(EvidenceError):
        h.validate_owners(i, c)


def validation_fixture():
    return {
        "status": "PASS_LOCAL_FINAL_VALIDATION_COMMITTED",
        "step": 695,
        "clock": {"iter": 695, "optimizer_step": 695, "attempt_id": 695, "accum": 0, "nonfinite_skips_total": 0},
        "data_rank_positions": [1390] * 4,
        "validation": {
            "version": 2,
            "identity": {"enabled": True, "source_fingerprint": "fp"},
            "progress": {"pending_step": None, "pending_reasons": [], "pending_metrics": {}},
            "publication": None,
        },
    }


def test_exact_committed_validation_clock_and_cursor():
    h.validate_validation_state(validation_fixture(), step=695, fingerprint="fp", accumulation=2)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(status="PENDING"),
        lambda v: v.update(step=694),
        lambda v: v["clock"].update(accum=1),
        lambda v: v["clock"].update(nonfinite_skips_total=1),
        lambda v: v["data_rank_positions"].__setitem__(3, 1389),
        lambda v: v["validation"]["identity"].update(source_fingerprint="foreign"),
        lambda v: v["validation"]["progress"].update(pending_step=695),
        lambda v: v["validation"].update(publication={}),
    ],
)
def test_pending_partial_or_foreign_validation_rejected(mutate):
    v = validation_fixture()
    mutate(v)
    with pytest.raises(EvidenceError):
        h.validate_validation_state(v, step=695, fingerprint="fp", accumulation=2)


def accounting_fixture():
    def cmd(stdout):
        return {"returncode": 0, "stderr": "", "stdout": stdout}

    c = {
        "sacct": cmd("123|COMPLETED|0:0|10|gres/gpu=4,node=2|a,b|rarunachalam\n"),
        "queue": cmd("1234\n99\n"),
        "steps": cmd("1234.batch\n99.0\n"),
    }
    expected = [
        {"job": "123", "allocation_seconds": 10, "allocated_gpus": 4, "gpu_seconds": 40, "scheduler_nodes": "a,b"}
    ]
    return c, expected


def test_unrelated_job_ids_do_not_block_terminal_accounting():
    h.validate_accounting(*accounting_fixture())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c, e: c["sacct"].update(returncode=1),
        lambda c, e: c["sacct"].update(returncode=False),
        lambda c, e: c["sacct"].update(stdout=c["sacct"]["stdout"].replace("COMPLETED", "FAILED")),
        lambda c, e: c["sacct"].update(stdout=c["sacct"]["stdout"].replace("rarunachalam", "foreign")),
        lambda c, e: c["sacct"].update(stdout=c["sacct"]["stdout"].replace("node=2", "node=1")),
        lambda c, e: c["sacct"].update(stdout=c["sacct"]["stdout"] * 2),
        lambda c, e: c["queue"].update(returncode=1),
        lambda c, e: c["queue"].update(stdout="123\n"),
        lambda c, e: c["steps"].update(stdout="123.batch\n"),
        lambda c, e: e[0].update(gpu_seconds=0),
    ],
)
def test_accounting_query_failure_foreign_owner_or_active_steps_reject(mutate):
    c, e = accounting_fixture()
    mutate(c, e)
    with pytest.raises(EvidenceError):
        h.validate_accounting(c, e)


def test_owned_reader_rejects_alias_hardlink_fifo_and_changed_descriptor(tmp_path):
    p = tmp_path / "source"
    p.write_bytes(b"synthetic")
    assert h.owned_bytes(p)[0] == b"synthetic"
    alias = tmp_path / "alias"
    alias.symlink_to(p)
    with pytest.raises(EvidenceError):
        h.owned_bytes(alias)
    alias.unlink()
    alias.hardlink_to(p)
    with pytest.raises(EvidenceError):
        h.owned_bytes(p)
    alias.unlink()
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(EvidenceError):
        h.owned_bytes(fifo)
    original = os.open

    def changed(path, flags):
        if Path(path) == p:
            p.rename(tmp_path / "retained-before-open")
            p.write_bytes(b"synthetic")
        return original(path, flags)

    with patch.object(os, "open", changed), pytest.raises(EvidenceError):
        h.owned_bytes(p)


def test_capture_rechecks_previously_read_file_identity(tmp_path):
    p = tmp_path / "source.json"
    p.write_bytes(b"{}")
    c = h.Capture.__new__(h.Capture)
    c.root, c.references, c.identities = tmp_path, {}, {}
    c.bytes({"path": p.name, "sha256": h.sha(b"{}")})
    c.finish()
    p.rename(tmp_path / "retained-before-replacement.json")
    p.write_bytes(b"{}")
    with pytest.raises(EvidenceError):
        c.finish()


def test_final_content_recheck_rejects_changed_bytes_even_if_metadata_repeats(tmp_path):
    p = tmp_path / "source.json"
    p.write_bytes(b"{}")
    c = h.Capture.__new__(h.Capture)
    c.root, c.references, c.identities = tmp_path, {}, {}
    c.bytes({"path": p.name, "sha256": h.sha(b"{}")})
    previous = c.identities[str(p)]
    with patch.object(h, "owned_bytes", return_value=(b"[]", previous)), pytest.raises(EvidenceError):
        c.finish()


def test_remote_proof_cannot_invent_checkpoint_object(tmp_path):
    c = h.Capture.__new__(h.Capture)
    c.remote = {"/synthetic/model/__0_0.distcp": {"bytes": 100, "sha256": "a" * 64}}
    c.resolver = {}
    c.proof("/synthetic/model/__0_0.distcp", "a" * 64, 100)
    for name, sha, size in [
        ("/synthetic/missing", "a" * 64, 100),
        ("/synthetic/model/__0_0.distcp", "b" * 64, 100),
        ("/synthetic/model/__0_0.distcp", "a" * 64, 99),
    ]:
        with pytest.raises(EvidenceError):
            c.proof(name, sha, size)


def relocated_capture(tmp_path):
    c = h.Capture.__new__(h.Capture)
    c.root = tmp_path / "collector"
    c.capture_origin = tmp_path / "historical"
    c.root.mkdir()
    c.references, c.identities = {}, {}
    return c


def test_capture_relocation_preserves_original_bytes_and_relative_references(tmp_path):
    c = relocated_capture(tmp_path)
    folder = c.root / "actual"
    folder.mkdir()
    body = json.dumps({"historical_path": str(c.capture_origin / "actual" / "source")}).encode()
    (folder / "review.json").write_bytes(body)
    ref = c.ref(c.capture_origin / "actual" / "review.json", h.sha(body))
    assert ref == {"path": "actual/review.json", "sha256": h.sha(body)}
    assert c.bytes(ref) == body
    c.finish()
    assert not c.capture_origin.exists()  # No fallback read of the old root.


@pytest.mark.parametrize("kind", ["foreign_absolute", "sibling_prefix", "traversal", "destination_absolute"])
def test_capture_relocation_does_not_admit_new_absolute_names(tmp_path, kind):
    c = relocated_capture(tmp_path)
    paths = {
        "foreign_absolute": tmp_path / "foreign" / "review.json",
        "sibling_prefix": Path(str(c.capture_origin) + "-other") / "review.json",
        "traversal": c.capture_origin / ".." / "review.json",
        "destination_absolute": c.root / "review.json",
    }
    with pytest.raises(EvidenceError):
        c.ref(paths[kind], "a" * 64)


@pytest.mark.parametrize("kind", ["changed_bytes", "symlink", "hardlink", "ancestor_symlink"])
def test_relocated_capture_still_requires_exact_owned_regular_bytes(tmp_path, kind):
    c = relocated_capture(tmp_path)
    original = tmp_path / "original"
    original.write_bytes(b"{}")
    folder = c.root / "actual"
    folder.mkdir()
    target = folder / "source.json"
    if kind == "changed_bytes":
        target.write_bytes(b"[]")
    elif kind == "symlink":
        target.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, target)
    else:
        external = tmp_path / "external"
        folder.rmdir()
        external.mkdir()
        (external / "source.json").write_bytes(b"{}")
        folder.symlink_to(external, target_is_directory=True)
    ref = c.ref(c.capture_origin / "actual" / "source.json", h.sha(b"{}"))
    with pytest.raises(EvidenceError):
        c.bytes(ref)


@pytest.mark.parametrize("origin", ["relative", "/synthetic/../other"])
def test_capture_origin_rejects_ambiguous_roots_before_reading(tmp_path, origin):
    with pytest.raises(EvidenceError, match="capture origin"):
        h.Capture(tmp_path, {}, capture_origin=origin)


def producer_fixture(tmp_path):
    audit = write(tmp_path, tmp_path / "synthetic-audit.json", {"scope": "SYNTHETIC TEST ONLY"})
    pins = {"tools/adapter_campaign/" + n: h.sha(Path(h.__file__).with_name(n).read_bytes()) for n in h.DEPENDENCIES}
    pins[audit["path"]] = audit["sha256"]
    review = write(
        tmp_path,
        tmp_path / "synthetic-producer-review.json",
        {
            "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
            "open_findings": [],
            "reviewed_files": pins,
            "scope": "SYNTHETIC TEST ONLY; NEVER CAMPAIGN ADMISSION",
        },
    )
    producer = h.sha(Path(h.__file__).read_bytes())
    source = {"revision": "a" * 40, "wheel_sha256": "b" * 64, "runtime_sha256": "c" * 64}
    requirements = [
        {"id": g, "sources": [source], "producers": [producer], "producer_reviews": {producer: [review]}}
        for g in sorted(h.GATES)
    ]
    contract = {
        "cells": [{"id": "synthetic", "identity": {"scope": "SYNTHETIC TEST ONLY"}, "requirements": requirements}]
    }
    normalized = (
        {g: {"scope": "SYNTHETIC TEST ONLY"} for g in h.GATES},
        source,
        {"primary_root": "/synthetic", "primary_job": "123", "primary_plan_sha256": "d" * 64},
        [audit],
    )
    return contract, audit, review, normalized


def test_producer_only_emits_four_source_bound_training_gates(tmp_path):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    with patch.object(h, "normalize", return_value=normalized):
        index = h.produce(
            tmp_path, contract, tmp_path / "output", cell_id="synthetic", audit_review=audit, producer_review=review
        )
    store = Store(tmp_path)
    result = index["cells"]["synthetic"]
    assert set(result["gates"]) == h.GATES and result["final_checkpoint"] is None
    for references in result["gates"].values():
        receipt = store.json(references[0])
        assert receipt["profile"] is None and receipt["source"] == normalized[1]
        assert receipt["supporting_evidence"] == [audit, review]
    assert index["contract_sha256"] == digest(contract)


def test_relocated_production_requires_explicit_reviewed_origin(tmp_path):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    with patch.object(h, "normalize", return_value=normalized) as mocked, pytest.raises(EvidenceError, match="origin"):
        h.produce(
            tmp_path,
            contract,
            tmp_path / "output",
            cell_id="synthetic",
            audit_review=audit,
            producer_review=review,
            capture_origin="/synthetic-original",
        )
    mocked.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_reviewed_relocation_is_forwarded_to_the_strict_normalizer(tmp_path):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    body = Store(tmp_path).json(review)
    body["capture_origins"] = {audit["sha256"]: "/synthetic-original"}
    review2 = write(tmp_path, tmp_path / "synthetic-mapped-review.json", body)
    for gate in contract["cells"][0]["requirements"]:
        gate["producer_reviews"] = {gate["producers"][0]: [review2]}
    with patch.object(h, "normalize", return_value=normalized) as mocked:
        h.produce(
            tmp_path,
            contract,
            tmp_path / "output",
            cell_id="synthetic",
            audit_review=audit,
            producer_review=review2,
            capture_origin="/synthetic-original",
        )
    mocked.assert_called_once_with(tmp_path, contract["cells"][0], audit, capture_origin="/synthetic-original")


def test_legacy_primary_categories_accept_only_the_exact_contract_sealed_review(tmp_path):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    contract["sealed_inputs"] = [review]
    for row in contract["cells"][0]["requirements"]:
        if row["id"] != "mid_epoch_continuation":
            row.pop("producer_reviews")
    with patch.object(h, "normalize", return_value=normalized):
        index = h.produce(
            tmp_path, contract, tmp_path / "output", cell_id="synthetic", audit_review=audit, producer_review=review
        )
    assert set(index["cells"]["synthetic"]["gates"]) == h.GATES
    for refs in index["cells"]["synthetic"]["gates"].values():
        assert review in Store(tmp_path).json(refs[0])["supporting_evidence"]


@pytest.mark.parametrize(
    "kind", ["missing_seal", "wrong_hash", "wrong_path", "midpoint_missing_map", "map_empty", "map_wrong_review"]
)
def test_legacy_admission_cannot_bypass_sealed_review_or_existing_map(tmp_path, kind):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    contract["sealed_inputs"] = [copy.deepcopy(review)]
    rows = {row["id"]: row for row in contract["cells"][0]["requirements"]}
    for name in ("full_schedule", "complete_sample_coverage", "initial_final_validation"):
        rows[name].pop("producer_reviews")
    if kind == "missing_seal":
        contract["sealed_inputs"] = []
    elif kind == "wrong_hash":
        contract["sealed_inputs"][0]["sha256"] = "f" * 64
    elif kind == "wrong_path":
        contract["sealed_inputs"][0]["path"] = "other.json"
    elif kind == "midpoint_missing_map":
        rows["mid_epoch_continuation"].pop("producer_reviews")
    elif kind == "map_empty":
        rows["full_schedule"]["producer_reviews"] = {}
    else:
        rows["full_schedule"]["producer_reviews"] = {
            rows["full_schedule"]["producers"][0]: [{"path": "other.json", "sha256": "f" * 64}]
        }
    with patch.object(h, "normalize", return_value=normalized), pytest.raises(EvidenceError):
        h.produce(
            tmp_path, contract, tmp_path / "output", cell_id="synthetic", audit_review=audit, producer_review=review
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["source", "producer", "review", "extra_gate"])
def test_no_partial_gate_publication_when_any_admission_missing(tmp_path, mutation):
    contract, audit, review, normalized = producer_fixture(tmp_path)
    last = contract["cells"][0]["requirements"][-1]
    if mutation == "source":
        last["sources"] = []
    elif mutation == "producer":
        last["producers"] = []
    elif mutation == "review":
        last.pop("producer_reviews")
    else:
        normalized[0]["deployment_numerics"] = {"fake": True}
    with patch.object(h, "normalize", return_value=normalized), pytest.raises(EvidenceError):
        h.produce(
            tmp_path, contract, tmp_path / "output", cell_id="synthetic", audit_review=audit, producer_review=review
        )
    assert not (tmp_path / "output").exists()


class SyntheticCapture:
    """Small in-memory documents for semantic negatives, never raw admission."""

    def __init__(self, values, root=Path("/synthetic")):
        self.values, self.remote_root = values, root
        self.references, self.proofs = {}, []

    def json(self, path):
        return copy.deepcopy(self.values[str(path)])

    def raw(self, path):
        value = self.values[str(path)]
        return value if type(value) is bytes else json.dumps(value).encode()

    def sha(self, path):
        return h.sha(self.raw(path))

    def rows(self, path):
        return [json.loads(v) for v in self.raw(path).splitlines()]

    def proof(self, path, expected):
        assert self.sha(path) == expected
        self.proofs.append(str(path))

    def finish(self):
        return None


def identity_fixture():
    cid = "qwen2_vl_hf__synthetic"
    cells = [
        {"id": f"family{family}__synthetic{n}", "identity": {"unchanged": n}} for family in range(16) for n in range(23)
    ]
    # Rename one whole family so the sealed sixteen-family count is retained.
    for n in range(23):
        cells[n]["id"] = f"qwen2_vl_hf__synthetic{n}"
    cells[0]["id"] = cid
    binding = {
        "id": "qwen2_vl_hf",
        "revision": "f" * 40,
        "implementation": "hf",
        "repository": "Qwen/Qwen2-VL-2B-Instruct",
    }
    spec = {"cell": {"id": cid}, "binding": binding}
    plan = {
        "cell": cid,
        "config_sha256": h.sha(json.dumps(spec).encode()),
        "source_revision": "a" * 40,
        "wheel_sha256": "b" * 64,
        "image_sha256": "c" * 64,
    }
    actual = {
        "base": {"sha256": "d" * 64, "files": {"model.safetensors": "e" * 64}},
        "processor": {"sha256": "e" * 64, "files": {"tokenizer.json": "d" * 64}},
    }
    provenance = {
        "source_dirty": False,
        "source_revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "container_sha256": plan["image_sha256"],
        "campaign_spec_sha256": plan["config_sha256"],
    }
    values = {
        "/synthetic/inputs/plan.json": plan,
        "/synthetic/inputs/cell.json": spec,
        "/synthetic/inputs/readiness-seal.json": {
            "expected_adapter_identity": copy.deepcopy(actual),
            "cell": cid,
            "config_sha256": plan["config_sha256"],
        },
        "/synthetic/inputs/full-data-identity-plan.json": {
            "cases": [
                {
                    "binding": binding["id"],
                    "model_assets": {
                        **binding,
                        "base_identity": copy.deepcopy(actual["base"]),
                        "processor_identity": copy.deepcopy(actual["processor"]),
                    },
                }
            ]
        },
        "/synthetic/output/segment-0/training/audit/initial-portable/adapter_manifest.json": {
            "identity": actual,
            "provenance": provenance,
            "weights_sha256": h.sha(b"synthetic weights"),
        },
        "/synthetic/output/segment-0/training/audit/initial-portable/adapter.safetensors": b"synthetic weights",
    }
    for rank in range(4):
        values[f"/synthetic/output/segment-0/training/audit/initial-rank{rank}.json"] = {
            "initial_clock": {"iter": 0, "optimizer_step": 0, "attempt_id": 0, "accum": 0, "optimizer_pristine": 1},
            "adapter_identity": copy.deepcopy(actual),
        }
        values[f"/synthetic/output/segment-0/training/audit/provenance-rank{rank}.json"] = copy.deepcopy(provenance)
    cells[0]["identity"] = {
        "base": None,
        "processor": None,
        "model_revision": binding["revision"],
        "config_sha256": plan["config_sha256"],
        "inventory_cell_sha256": digest(spec["cell"]),
    }
    contract = {"cells": cells}
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(contract), "cells": {}}
    captured = SyntheticCapture(values)
    captured.audit = {"cell": cid}
    return contract, index, captured, cid


def test_identity_resolution_only_fills_two_null_fields_before_training():
    contract, index, captured, cid = identity_fixture()
    before = copy.deepcopy(contract)
    with patch.object(h, "Capture", return_value=captured):
        resolved, receipt = h.resolve_identity(
            "/synthetic",
            contract,
            index,
            cell_id=cid,
            audit_review={"path": "synthetic-review.json", "sha256": "a" * 64},
        )
    assert contract == before and resolved["cells"][1:] == contract["cells"][1:]
    old, new = contract["cells"][0], resolved["cells"][0]
    assert {k: v for k, v in old["identity"].items() if k not in ("base", "processor")} == {
        k: v for k, v in new["identity"].items() if k not in ("base", "processor")
    }
    assert new["identity"]["base"]["sha256"] == "d" * 64
    assert receipt["gate_evidence_emitted"] is False and receipt["active_index_written"] is False
    assert captured.proofs == ["/synthetic/output/segment-0/training/audit/initial-portable/adapter.safetensors"]


@pytest.mark.parametrize(
    "kind",
    [
        "existing_gate",
        "partial_identity",
        "wrong_index",
        "scope_reduction",
        "post_update",
        "boolean_clock",
        "used_optimizer",
        "wrong_revision",
        "changed_admission",
        "changed_assets",
        "changed_runtime",
    ],
)
def test_identity_cannot_rebind_evidence_or_select_trained_state(kind):
    contract, index, captured, cid = identity_fixture()
    v = captured.values
    if kind == "existing_gate":
        index["cells"][cid] = {"gates": {"full_schedule": ["existing"]}}
    elif kind == "partial_identity":
        contract["cells"][0]["identity"]["base"] = {}
        index["contract_sha256"] = digest(contract)
    elif kind == "wrong_index":
        index["contract_sha256"] = "a" * 64
    elif kind == "scope_reduction":
        contract["cells"].pop()
    elif kind == "post_update":
        v["/synthetic/output/segment-0/training/audit/initial-rank3.json"]["initial_clock"]["iter"] = 1
    elif kind == "boolean_clock":
        v["/synthetic/output/segment-0/training/audit/initial-rank3.json"]["initial_clock"]["iter"] = False
    elif kind == "used_optimizer":
        v["/synthetic/output/segment-0/training/audit/initial-rank0.json"]["initial_clock"]["optimizer_pristine"] = 0
    elif kind == "wrong_revision":
        contract["cells"][0]["identity"]["model_revision"] = "a" * 40
        index["contract_sha256"] = digest(contract)
    elif kind == "changed_admission":
        v["/synthetic/inputs/readiness-seal.json"]["expected_adapter_identity"]["base"]["sha256"] = "a" * 64
    elif kind == "changed_assets":
        v["/synthetic/inputs/full-data-identity-plan.json"]["cases"][0]["model_assets"]["base_identity"]["sha256"] = (
            "a" * 64
        )
    else:
        v["/synthetic/output/segment-0/training/audit/provenance-rank2.json"]["container_sha256"] = "a" * 64
    with patch.object(h, "Capture", return_value=captured), pytest.raises(EvidenceError):
        h.resolve_identity("/synthetic", contract, index, cell_id=cid, audit_review={})


def data_fixture():
    root, folder = Path("/synthetic"), Path("/synthetic/output/segment-0")
    identity = {
        "base": {"sha256": "a" * 64},
        "processor": {"sha256": "b" * 64},
        "media": {"processor_kwargs": {"videos_kwargs": {"num_frames": 4}}},
    }
    source = {"source_revision": "c" * 40, "wheel_sha256": "d" * 64, "image_sha256": "e" * 64, "runtime": "/runtime"}
    request = {"format": "hf-wts-canonical-full-data-capture-v1", "root": str(root), **source}
    plan = {**source, "config_sha256": h.sha(b"{}")}
    spec = {
        "binding": {"id": "qwen2_vl_hf"},
        "dataset": {"rows_train": 5, "rows_validation": 3},
        "paths": {"train": "/train.jsonl", "validation": "/validation.jsonl"},
        "cell": {"training_epochs": 1, "grad_accum": 2},
    }
    case = {
        "binding": "qwen2_vl_hf",
        "config": {"path": "/config.json", "sha256": plan["config_sha256"]},
        "assets": {"base": "a" * 64, "processor": "b" * 64},
        "processor_kwargs": {"videos_kwargs": {"num_frames": 4}},
        "historical_census": {
            "schema": "own-hf-complete-processor-census-v1",
            "binding": "qwen2_vl_hf",
            "processor_identity": identity["processor"],
            "source_revision": source["source_revision"],
            "wheel_sha256": source["wheel_sha256"],
            "source_jsonl_sha256": {},
        },
        "splits": {},
        "finite_coverage": {
            "version": 1,
            "kind": "finite_sft_zero_weight_update_tail",
            "epochs": 1,
            "samples_per_update_per_rank": 2,
            "dataset_rows": 5,
            "world_size": 4,
        },
    }
    values = {str(root / "inputs/full-data-identity-plan.json"): request, "/config.json": b"{}"}
    census = []
    for split, count in (("train", 5), ("validation", 3)):
        ids = [f"{split}-{i}" for i in range(count)]
        raw = b"\n".join(
            json.dumps(
                {"id": i, "messages": [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]}
            ).encode()
            for i in ids
        )
        values[spec["paths"][split]] = raw
        case["historical_census"]["source_jsonl_sha256"][split] = h.sha(raw)
        cursor = {"position": 0, "rank": 0, "world_size": 4}
        if split == "train":
            cursor["source"] = {"rank": 0, "world_size": 1, "position": 0}
        case["splits"][split] = {
            "rows": count,
            "jsonl_sha256": h.sha(raw),
            "ids_sha256": h.sha("\n".join(ids).encode()),
            "source_fingerprint": ("f" if split == "train" else "e") * 64,
            "media": {"/canonical/clip.mp4": "a" * 64},
            "native_loss_tokens": count * 2,
            "initial_cursor": cursor,
        }
        census.extend(
            {"split": split, "row_id": row_id, "index": index, "mask_matches": True, "reasons": [], "loss_tokens": 2}
            for index, row_id in enumerate(ids)
        )
        census.append(
            {
                "summary": {
                    "split": split,
                    "ok": True,
                    "failed": 0,
                    "unmeasured_rows": 0,
                    "over_long_rows": 0,
                    "rows": count,
                    "matched_rows": count,
                    "loss_tokens": count * 2,
                }
            }
        )
    census_raw = b"\n".join(json.dumps(v).encode() for v in census)
    values[str(root / "inputs/own-census-qwen2_vl_hf-rows.jsonl")] = census_raw
    case["historical_census"]["rows_sha256"] = h.sha(census_raw)
    c = SyntheticCapture(values)
    admission = {
        "status": "PASS_TWO_HOST_CANONICAL_FULL_DATA_IDENTITIES",
        "plan_sha256": c.sha(root / "inputs/full-data-identity-plan.json"),
        "job": "123",
        "hosts": ["host-a", "host-b"],
        "records": {},
    }
    for node, host in enumerate(admission["hosts"]):
        own = copy.deepcopy(case)
        for split in ("train", "validation"):
            own["splits"][split]["initial_cursor"]["rank"] = node * 2
        record = {
            "format": "hf-wts-canonical-full-data-host-v1",
            "status": "PASS_IDENTITIES_ONLY",
            "plan_sha256": admission["plan_sha256"],
            "job": "123",
            "host": host,
            "node_id": node,
            "runtime": {
                "source_revision": source["source_revision"],
                "wheel_sha256": source["wheel_sha256"],
                "container_sha256": source["image_sha256"],
                "source_dirty": False,
            },
            "cases": [own, {"binding": "qwen3_vl_hf", "scope": "OTHER CHECKPOINT RECEIVES NO CREDIT"}],
        }
        name = str(folder / "canonical-data" / f"full-data-{host}.json")
        values[name] = record
        admission["records"][name] = c.sha(name)
    values[str(folder / "canonical-data-admission.json")] = admission
    return c, (root, spec, plan, ["host-a", "host-b"], "123", folder, identity)


def test_own_multimodal_census_and_rank_local_cursor_join():
    c, args = data_fixture()
    datasets, tokens, fingerprints, native, case = h.validate_data(c, *args)
    assert len(datasets["train"]) == 5 and native["validation"]["loss_tokens"] == 6
    assert set(tokens) == set(fingerprints) == {"train", "validation"}
    assert case["binding"] == "qwen2_vl_hf"


@pytest.mark.parametrize(
    "kind",
    [
        "rank_cursor",
        "world_size",
        "foreign_source",
        "wrong_processor",
        "missing_media",
        "wrong_denominator",
        "duplicate_own_case",
        "missing_host",
        "changed_media_hash",
        "wrong_binding",
    ],
)
def test_own_data_cannot_borrow_another_checkpoint_or_host(kind):
    c, args = data_fixture()
    records = c.values["/synthetic/output/segment-0/canonical-data-admission.json"]["records"]
    second = "/synthetic/output/segment-0/canonical-data/full-data-host-b.json"
    record = c.values[second]
    case = record["cases"][0]
    if kind == "rank_cursor":
        case["splits"]["train"]["initial_cursor"]["rank"] = 0
    elif kind == "world_size":
        case["splits"]["validation"]["initial_cursor"]["world_size"] = 2
    elif kind == "foreign_source":
        record["runtime"]["source_revision"] = "0" * 40
    elif kind == "wrong_processor":
        case["assets"]["processor"] = "0" * 64
    elif kind == "missing_media":
        case["splits"]["train"]["media"] = {}
    elif kind == "wrong_denominator":
        case["splits"]["train"]["native_loss_tokens"] = 8
    elif kind == "duplicate_own_case":
        record["cases"].append(copy.deepcopy(case))
    elif kind == "missing_host":
        records.pop(second)
    elif kind == "changed_media_hash":
        case["splits"]["train"]["media"]["/canonical/clip.mp4"] = "0" * 64
    else:
        case["binding"] = "qwen3_vl_hf"
    if second in records:
        records[second] = c.sha(second)
    with pytest.raises(EvidenceError):
        h.validate_data(c, *args)
