# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic receipt tests; none of these fixtures is campaign evidence."""

import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.adapter_campaign import fresh_segmented as f
from tools.adapter_campaign.evidence import EvidenceError, Store


def tensor_fixture():
    return {
        "status": "PASS_ACTUAL_SEGMENT_RESTORE_MIDPOINT_DORA_AND_CONTROL_TENSORS",
        "cell": f.CELL,
        "full_campaign_pass": False,
        "boundary_restores": [{"step": s, "rank": r, "bitwise_equal": True} for s in (250, 500, 750) for r in range(4)],
        "midpoint": [{"step": 467, "rank": r, "bitwise_equal": True} for r in range(4)],
        "next_updates": [
            {"step": s, "rank": r, "bitwise_equal_all_saved_fields": True} for s in (468, 469, 470) for r in range(4)
        ],
        "checkpoint_controls": [
            {
                "step": s,
                "trainer_control_clock": {"iter": s, "optimizer_step": s, "attempt_id": s, "accum": 0},
                "rank_data_positions": [2 * s] * 4,
                "all_metadata_storage_ranges_within_hashed_files": True,
                "all192_owners_and64_magnitude_states_present": True,
            }
            for s in (250, 467, 470, 500, 750, 935)
        ],
        "final_changed_owners": 192,
        "final_changed_magnitudes": 64,
    }


def test_complete_saved_state_comparison_population():
    f.validate_tensor_audit(tensor_fixture())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.update(cell="mistral_hf__lora__factorized_v1__default_v1__fsdp2"),
        lambda t: t.update(status="FAIL"),
        lambda t: t.update(full_campaign_pass=True),
        lambda t: t["boundary_restores"].pop(),
        lambda t: t["boundary_restores"][0].update(bitwise_equal=False),
        lambda t: t["midpoint"][0].update(rank=1),
        lambda t: t["midpoint"][0].update(step=468),
        lambda t: t["next_updates"].pop(),
        lambda t: t["next_updates"][0].update(bitwise_equal_all_saved_fields=False),
        lambda t: t["next_updates"][0].update(step=471),
        lambda t: t["checkpoint_controls"].pop(),
        lambda t: t["checkpoint_controls"][0]["trainer_control_clock"].update(accum=1),
        lambda t: t["checkpoint_controls"][0]["rank_data_positions"].__setitem__(1, 0),
        lambda t: t["checkpoint_controls"][0].update(all_metadata_storage_ranges_within_hashed_files=False),
        lambda t: t["checkpoint_controls"][0].update(all192_owners_and64_magnitude_states_present=False),
        lambda t: t.update(final_changed_magnitudes=63),
        lambda t: t.update(final_changed_owners=191),
    ],
)
def test_saved_state_incomplete_or_contradictory_rejected(mutate):
    t = tensor_fixture()
    mutate(t)
    with pytest.raises(EvidenceError):
        f.validate_tensor_audit(t)


def predecessor_fixture():
    refs = [{"path": "/remote/segment-0/result.json", "sha256": "a" * 64}]
    previous = {"next_predecessor": {"checkpoint": "/remote/iter_000250", "checkpoint_inventory": {"x": "a"}}}
    p = {
        "format": "public-mistral-full-schedule-segment-v1",
        "phase": "training",
        "purpose": "authoritative_full_schedule",
        "start": 250,
        "end": 500,
        "global_updates": 935,
        "previous": {**previous["next_predecessor"], "receipt": refs[0]},
    }
    r = {
        "phase": "training",
        "start": 250,
        "end": 500,
        "status": "PASS_PLANNED_SEGMENT_ONLY",
        "full_schedule_pass": False,
        "lineage_receipts": copy.deepcopy(refs),
    }
    return p, r, previous, refs


def test_exact_predecessor_chain_and_initial_phase():
    args = predecessor_fixture()
    f.validate_predecessor(*args, start=250, end=500)
    p, r, _, _ = predecessor_fixture()
    p.update(start=0, end=250, previous=None)
    r.update(start=0, end=250, lineage_receipts=[])
    f.validate_predecessor(p, r, None, [], start=0, end=250)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p, r: p.update(start=249),
        lambda p, r: p.update(end=501),
        lambda p, r: p.update(global_updates=934),
        lambda p, r: p.update(phase="branch"),
        lambda p, r: p.update(purpose="diagnostic"),
        lambda p, r: p.update(previous=None),
        lambda p, r: p["previous"].update(checkpoint="/remote/iter_000249"),
        lambda p, r: p["previous"]["receipt"].update(sha256="b" * 64),
        lambda p, r: r.update(status="FAIL"),
        lambda p, r: r.update(full_schedule_pass=True),
        lambda p, r: r.update(lineage_receipts=[]),
    ],
)
def test_predecessor_splice_rejected(mutate):
    args = list(predecessor_fixture())
    args[0] = copy.deepcopy(args[0])
    mutate(args[0], args[1])
    with pytest.raises(EvidenceError):
        f.validate_predecessor(*args, start=250, end=500)


def validation_fixture():
    return {
        "status": "PASS_LOCAL_FINAL_VALIDATION_COMMITTED",
        "step": 935,
        "clock": {"iter": 935, "optimizer_step": 935, "attempt_id": 935, "accum": 0, "nonfinite_skips_total": 0},
        "data_rank_positions": [1870] * 4,
        "validation": {
            "version": 2,
            "identity": {"enabled": True, "source_fingerprint": "validation"},
            "progress": {"pending_step": None, "pending_reasons": [], "pending_metrics": {}},
            "publication": None,
        },
    }


def test_committed_validation_complete():
    f.validate_committed_validation(validation_fixture(), "validation")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(status="PENDING"),
        lambda v: v.update(step=934),
        lambda v: v["clock"].update(optimizer_step=934),
        lambda v: v["data_rank_positions"].pop(),
        lambda v: v["validation"]["progress"].update(pending_step=935),
        lambda v: v["validation"].update(publication={"step": 935}),
        lambda v: v["validation"]["identity"].update(source_fingerprint="foreign"),
    ],
)
def test_uncommitted_validation_rejected(mutate):
    v = validation_fixture()
    mutate(v)
    with pytest.raises(EvidenceError):
        f.validate_committed_validation(v, "validation")


def test_transport_is_streamed_and_has_no_generic_large_file_admission(tmp_path):
    data = b"synthetic transport"
    p = tmp_path / "synthetic.tar"
    p.write_bytes(data)
    ref = {"path": p.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    with pytest.raises(EvidenceError):
        f.verify_transport(tmp_path, ref)
    with patch.object(f, "ARCHIVE", ref):
        f.verify_transport(tmp_path, ref)
        p.write_bytes(b"S" + data[1:])
        with pytest.raises(EvidenceError):
            f.verify_transport(tmp_path, ref)
        p.unlink()
        p.symlink_to(tmp_path / "missing")
        with pytest.raises(EvidenceError):
            f.verify_transport(tmp_path, ref)


def test_transport_hardlink_rejected(tmp_path):
    p = tmp_path / "raw"
    p.write_bytes(b"x")
    (tmp_path / "alias").hardlink_to(p)
    ref = {"path": "raw", "sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1}
    with patch.object(f, "ARCHIVE", ref), pytest.raises(EvidenceError):
        f.verify_transport(tmp_path, ref)


def test_proof_requires_exact_source_digest_and_length(tmp_path):
    p = tmp_path / "proof.json"
    p.write_bytes(b"{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    c = f.Capture.__new__(f.Capture)
    c.root = tmp_path
    c.store = Store(tmp_path)
    c.references = {}
    c.resolver = {}
    c.reviewed = {p.name: sha}
    c.large = {"files": {"/remote/large": {"bytes": 100, "sha256": "a" * 64}}}
    c.bind("/remote/small", Path(p.name), {"bytes": 2, "sha256": sha})
    c.proof("/remote/small", sha, 2)
    c.proof("/remote/large", "a" * 64, 100)
    for name, digest, size in [
        ("/remote/missing", sha, 2),
        ("/remote/small", "b" * 64, 2),
        ("/remote/large", "a" * 64, 99),
        ("/remote/large", "b" * 64, 100),
    ]:
        with pytest.raises(EvidenceError):
            c.proof(name, digest, size)
