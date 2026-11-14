# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize the reviewed fresh four-allocation Mistral DoRA primary.

This closed lineage differs from the earlier interrupted LoRA primary. It joins
actual source-bound records; it neither trains nor loads model checkpoints.
Saved-state comparisons come from the admitted independent tensor audit. Raw
coverage, objective denominators, process lineage and validation are rederived.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import stat
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, producer_reviews, read_json, require, same
from tools.adapter_campaign.primary import file_ref, write
from tools.adapter_campaign.recovered import validate_coverage, validate_datasets, validate_rank
from tools.adapter_campaign.segmented import GATES_SHA, join_phases
from tools.adapter_campaign.segmented import Capture as LegacyCapture

CELL = "mistral_hf__dora__factorized_v1__default_v1__fsdp2"
BOUNDS = (0, 250, 500, 750, 935)
JOBS = ("38488547", "38496370", "38499022", "38505524")
GATES = {"full_schedule", "complete_sample_coverage", "initial_final_validation", "mid_epoch_continuation"}
ARCHIVE = {
    "path": "review/mistral-dora-segmented-primary-actual-v1/audit-tensors.tar",
    "sha256": "2d44d3cfc27fd1d253cc2408a295a750521e9466b8843f34db575c20da038a61",
    "bytes": 1269022720,
}
DEPENDENCIES = ("fresh_segmented.py", "segmented.py", "recovered.py", "evidence.py", "primary.py")


def verify_transport(root, reference):
    """Stream only this redundant raw transport; leave JSON Store's cap intact."""
    require(same(reference, ARCHIVE), "unadmitted large raw transport")
    root = Path(root).resolve(strict=True)
    path = root / reference["path"]
    require(all(not p.is_symlink() for p in (path, *path.parents)), "raw transport alias")
    fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == os.getuid(),
        "foreign or nonregular transport",
    )
    require(before.st_size == reference["bytes"], "wrong transport length")
    identity = tuple(getattr(before, k) for k in fields)
    hasher = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        require(tuple(getattr(os.fstat(stream.fileno()), k) for k in fields) == identity, "transport replaced")
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
        require(tuple(getattr(os.fstat(stream.fileno()), k) for k in fields) == identity, "transport changed")
    require(tuple(getattr(path.lstat(), k) for k in fields) == identity, "transport replaced after read")
    require(hasher.hexdigest() == reference["sha256"], "stale raw transport")


class Capture(LegacyCapture):
    """Use the existing exact resolver interface, with this capture's own schema."""

    def __init__(self, root, audit_ref):
        self.root, self.store, self.references = Path(root), Store(root), {}
        self.audit = self.document(audit_ref)
        a = self.audit
        require(
            a["format"] == "mistral-never-started-segmented-primary-independent-actual-audit-v1"
            and a["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and not a["open_findings"]
            and a["full_campaign_pass"] is False
            and a["active_index_written"] is False,
            "unadmitted fresh segmented audit",
        )
        self.reviewed = a["reviewed_files"]
        require(self.reviewed.get(ARCHIVE["path"]) == ARCHIVE["sha256"], "review omitted raw transport")
        verify_transport(root, ARCHIVE)
        # Individual saved witnesses stay in the normal bounded Store. The
        # redundant tar is verified above, never a large JSON evidence reference.
        for name, sha in self.reviewed.items():
            if name != ARCHIVE["path"]:
                self.bytes({"path": name, "sha256": sha})
        folder = Path(audit_ref["path"]).parent
        self.small = self.local(folder / "capture01.json")
        require(
            self.small["format"] == "mistral-dora-segmented-readonly-raw-capture-v1"
            and self.small["root"] == a["root"]
            and not self.small["links"],
            "changed capture namespace",
        )
        self.remote_root = Path(a["root"]).parent
        self.resolver = {}
        for name, item in self.small["files"].items():
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe capture path")
            remote = self.remote_root / path
            if path.parts[0] == "external-assets":
                split = path.parts[1]
                require(path == Path(f"external-assets/{split}/{split}.jsonl"), "unknown external asset")
                remote = Path(self.small["assets"][split]["path"])
                require(
                    item["original_path"] == str(remote)
                    and all(item[k] == self.small["assets"][split][k] for k in ("sha256", "bytes")),
                    "asset binding changed",
                )
            self.bind(remote, folder / "files" / path, item)
        self.large = self.local(folder / "large-object-hashes.json")
        require(self.large["status"] == "PASS_STABLE_REMOTE_OBJECT_SHA256", "unhashed remote checkpoint")
        expected_large = {str(self.remote_root / n): v for n, v in self.small["large_unhashed"].items()}
        require(set(self.large["files"]) == set(expected_large), "incomplete remote inventory")
        for name, item in self.large["files"].items():
            expected = expected_large[name]
            require(
                item["bytes"] == expected["bytes"] and same(item["pre_post_stat"], expected["stat"]),
                "remote object identity changed",
            )
        post = self.document(a["capture"]["actual_small_source_postcheck"])
        require(
            post["status"] == "PASS_ALL_CAPTURED_SOURCE_BYTES_AND_NAMESPACES_UNCHANGED"
            and a["capture"]["original_sources_immutable"] is True,
            "captured sources changed",
        )
        self.measured = self.local(folder / "local-audit.json")
        self.tensors = self.local(folder / "tensor-audit.json")
        self.validation = self.local(folder / "validation-audit.json")
        transfer = self.local(folder / "tensor-transfer.json")
        require(transfer["status"] == "PASS_EXACT_ACTUAL_AUDIT_CONTROL_TENSOR_TRANSFER", "partial tensor transfer")
        for remote, item in transfer["files"].items():
            require(remote in self.large["files"], "transferred witness lacks remote digest")
            require(
                item["sha256"] == self.large["files"][remote]["sha256"]
                and item["bytes"] == self.large["files"][remote]["bytes"],
                "tensor transfer differs from actual remote state",
            )
            self.bind(remote, Path(item["local_path"]), item)

    def proof(self, remote, sha, size=None):
        remote = str(remote)
        if remote in self.resolver:
            raw = self.raw(remote)
            require(hashlib.sha256(raw).hexdigest() == sha, "stale raw receipt")
            count = len(raw)
        else:
            require(remote in self.large["files"], "missing actual tensor inventory")
            item = self.large["files"][remote]
            require(item["sha256"] == sha, "stale remote tensor digest")
            count = item["bytes"]
        require(count > 0 and (size is None or count == size), "wrong raw witness length")


def validate_tensor_audit(tensors):
    require(
        tensors["status"] == "PASS_ACTUAL_SEGMENT_RESTORE_MIDPOINT_DORA_AND_CONTROL_TENSORS"
        and tensors["cell"] == CELL
        and tensors["full_campaign_pass"] is False,
        "tensor comparison audit failed",
    )
    for key, steps, flag in (
        ("boundary_restores", (250, 500, 750), "bitwise_equal"),
        ("midpoint", (467,), "bitwise_equal"),
        ("next_updates", (468, 469, 470), "bitwise_equal_all_saved_fields"),
    ):
        rows = tensors[key]
        require(
            len(rows) == len(steps) * 4
            and {(r["step"], r["rank"]) for r in rows} == {(s, r) for s in steps for r in range(4)}
            and all(r[flag] is True for r in rows),
            "missing, repeated or failed saved-state comparison",
        )
    controls = tensors["checkpoint_controls"]
    require(
        len(controls) == 6 and {r["step"] for r in controls} == {250, 467, 470, 500, 750, 935},
        "missing checkpoint control generation",
    )
    for row in controls:
        step = row["step"]
        require(
            row["trainer_control_clock"] == {"iter": step, "optimizer_step": step, "attempt_id": step, "accum": 0}
            and row["rank_data_positions"] == [2 * step] * 4
            and row["all_metadata_storage_ranges_within_hashed_files"] is True
            and row["all192_owners_and64_magnitude_states_present"] is True,
            "checkpoint controls, ownership or storage bounds failed",
        )
    require(
        tensors["final_changed_owners"] == 192 and tensors["final_changed_magnitudes"] == 64, "untrained DoRA owner"
    )


def validate_predecessor(plan, receipt, previous, previous_refs, *, start, end):
    require(
        plan["format"] == "public-mistral-full-schedule-segment-v1"
        and plan["phase"] == receipt["phase"] == "training"
        and plan["purpose"] == "authoritative_full_schedule"
        and plan["start"] == receipt["start"] == start
        and plan["end"] == receipt["end"] == end
        and plan["global_updates"] == 935,
        "changed segment interval or purpose",
    )
    require(
        receipt["status"] == "PASS_PLANNED_SEGMENT_ONLY"
        and receipt["full_schedule_pass"] is False
        and same(receipt["lineage_receipts"], previous_refs),
        "incomplete or reordered predecessor chain",
    )
    expected = None if previous is None else {**previous["next_predecessor"], "receipt": previous_refs[-1]}
    require(same(plan["previous"], expected), "wrong committed predecessor")


def validate_committed_validation(value, fingerprint):
    require(
        value["status"] == "PASS_LOCAL_FINAL_VALIDATION_COMMITTED" and value["step"] == 935,
        "uncommitted final validation",
    )
    require(
        value["clock"]
        == {"iter": 935, "optimizer_step": 935, "attempt_id": 935, "accum": 0, "nonfinite_skips_total": 0}
        and value["data_rank_positions"] == [1870] * 4,
        "wrong committed final clock/cursor",
    )
    state = value["validation"]
    require(
        state["version"] == 2
        and state["identity"]["enabled"] is True
        and state["identity"]["source_fingerprint"] == fingerprint
        and state["progress"] == {"pending_step": None, "pending_reasons": [], "pending_metrics": {}}
        and state["publication"] is None,
        "pending, partial or foreign validation",
    )


def normalize(root, contract_cell, audit_ref):
    c = Capture(root, audit_ref)
    a, read = c.audit, c.json
    terminal = Path(a["root"])
    plan, spec = read(terminal / "inputs/plan.json"), read(terminal / "inputs/cell.json")
    original = Path(plan["original_root"])
    old = read(original / "inputs/plan.json")
    cell, identity = spec["cell"], contract_cell["identity"]
    require(a["cell"] == plan["cell"] == cell["id"] == contract_cell["id"] == CELL, "unadmitted fresh primary")
    require(
        plan["format"] == "mistral-reviewed-never-started-full-segments-v2"
        and plan["root"] == str(terminal)
        and original != terminal
        and original.parent == c.remote_root,
        "wrong fresh primary ancestry",
    )
    require(
        digest(cell) == identity["inventory_cell_sha256"]
        and c.sha(terminal / "inputs/cell.json") == c.sha(original / "inputs/cell.json") == identity["config_sha256"]
        and plan["config_sha256"] == identity["config_sha256"],
        "changed frozen configuration",
    )
    require(c.sha(terminal / "inputs/plan.json") == a["plan_sha256"], "wrong audited plan")
    require(c.sha(original / "inputs/plan.json") == plan["original_plan_sha256"], "changed original ancestry")
    require(
        plan["boundaries"] == list(BOUNDS)
        and plan["global_updates"] == cell["optimizer_updates"] == 935
        and cell["training_epochs"] == 1
        and cell["checkpoint_contract"]["mid_epoch_update"] == 467
        and cell["micro_batch_size"] == 1
        and cell["grad_accum"] == 2
        and cell["topology"]["world_size"] == 4
        and cell["topology"]["layout"] == "fsdp2",
        "unadmitted schedule/topology",
    )
    require(spec["binding"]["id"] == "mistral_hf" and spec["binding"]["dataset"] == "gsm8k", "wrong binding")
    for base, value in [(original, old), (terminal, plan)]:
        for key in ("source_revision", "wheel_sha256", "image_sha256", "runtime", "cell"):
            require(value[key] == plan[key], "source/runtime changed")
        for name, sha in value["files_sha256"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe executed payload")
            c.proof(base / "inputs" / name, sha)
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    require(same(c.measured["source"], a["actual_source"]), "actual source audit mismatch")
    require(all(a["actual_source"][k] == plan[k] for k in a["actual_source"]), "wrong installed producer")
    require(
        c.measured["cancelled_never_started_job"] == {"job": "38458746", "gpu_seconds": 0}
        and c.measured["cell"] == CELL
        and c.measured["status"] == "PASS_ACTUAL_SEGMENTED_PRIMARY_AND_REPLAY_RECEIPTS",
        "original job may have contributed coverage",
    )
    accounts = {}
    for row in c.small["accounting"][0]["stdout"].splitlines():
        parts = row.split("|")
        require(parts[1:3] == ["rarunachalam", "41852"] and parts[0] not in accounts, "foreign/duplicate allocation")
        accounts[parts[0]] = parts
    require(set(accounts) == set(JOBS) | {"38458746"}, "wrong allocation census")
    require(
        accounts["38458746"][3:5] == ["CANCELLED by 41852", "0:0"] and int(accounts["38458746"][7]) == 0, "original ran"
    )
    require(all(accounts[j][3:5] == ["COMPLETED", "0:0"] for j in JOBS), "nonterminal or failed primary segment")
    require(
        c.small["accounting"][1]["returncode"] == 0 and c.small["accounting"][1]["stdout"] == "", "active original job"
    )
    datasets, loss_tokens, fingerprints, native = validate_datasets(c, original, old, spec)
    first_audit = terminal / "output/segment-0/training/audit"
    initial_manifest = read(first_audit / "initial-portable/adapter_manifest.json")
    expected_identity = initial_manifest["identity"]
    require(
        same(expected_identity["base"], identity["base"])
        and same(expected_identity["processor"], identity["processor"]),
        "base/processor differs from contract",
    )
    logs, first, previous_refs, phases, checkpoints = [], {}, [], [], {}
    previous = None
    for index, job in enumerate(JOBS):
        folder = terminal / "output" / f"segment-{index}"
        p, result, process = [read(folder / name) for name in ("plan.json", "result.json", "primary-process.json")]
        start, end = BOUNDS[index : index + 2]
        validate_predecessor(p, result, previous, previous_refs, start=start, end=end)
        require(result["segment_plan_sha256"] == c.sha(folder / "plan.json"), "wrong segment plan")
        require(
            same(result["process"], process) and process["returncode"] == 0 and process["stop_notice"] is None,
            "failed process",
        )
        require(process["provenance"]["job"] == job, "wrong executing allocation")
        for key in ("source_revision", "wheel_sha256", "image_sha256", "config_sha256", "cell"):
            require(result[key] == p[key] == plan[key], "segment changed source/configuration")
        require(p["parent_plan_sha256"] == result["parent_plan_sha256"] == a["plan_sha256"], "wrong parent plan")
        for name, sha in result["evidence_sha256"].items():
            c.proof(name, sha)
        ack = read(terminal / "dispatch" / f"segment-{index}/submission.json")
        intent = read(terminal / "dispatch" / f"segment-{index}/intent.json")
        require(
            ack["job_id"] == job
            and ack["returncode"] == 0
            and same(ack["command"], intent["command"])
            and ack["intent_sha256"] == c.sha(terminal / "dispatch" / f"segment-{index}/intent.json")
            and ack["plan_sha256"] == a["plan_sha256"],
            "unbound scheduler acknowledgment",
        )
        next_source = result["next_predecessor"]
        checkpoints[next_source["checkpoint"]] = next_source["checkpoint_inventory"]
        if "midpoint" in result:
            require(index == 1 and result["midpoint"]["step"] == 467, "wrong midpoint checkpoint")
            checkpoints[result["midpoint"]["checkpoint"]] = result["midpoint"]["checkpoint_inventory"]
        rank_logs = []
        measured = c.measured["phases"][index]
        require(measured["job"] == job and measured["start"] == start and measured["end"] == end, "wrong audited phase")
        for rank in range(4):
            audit = folder / "training/audit"
            initial = validate_rank(
                c,
                audit,
                spec=spec,
                plan=terminal,
                source=source,
                job=job,
                hosts={v["host"] for v in measured["rank_device_mapping"]},
                admission={"expected_adapter_identity": expected_identity},
                fingerprints=fingerprints,
                rank=rank,
                mode="full",
                final_step=end,
            )
            clock = initial["initial_clock"]
            require(
                clock["iter"] == clock["optimizer_step"] == clock["attempt_id"] == start and clock["accum"] == 0,
                "wrong initial clock",
            )
            owners = [n for g in initial["optimizer"] for n in g["names"]]
            require(
                len(owners) == 192 and sum(n.endswith(".magnitude") for n in owners) == 64, "DoRA ownership missing"
            )
            require(
                {v["name"] for v in initial["resolved"]["targets"]} == set(cell["resolved_target_names"]),
                "wrong actual targets",
            )
            require(not initial["resolved"]["extra_parameters"], "unadmitted extra owner")
            if rank in first:
                for key in ("resolved", "adapter_identity", "parameter_layouts", "frozen_sha256", "finite_coverage"):
                    require(same(first[rank][key], initial[key]), "segment changed model/state ownership")
            else:
                first[rank] = initial
            if index:
                restored = read(audit / f"segment-restored-rank{rank}.json")
                require(
                    restored["status"] == "PASS_EXACT_SEGMENT_RESTORE_BEFORE_FORWARD" and restored["step"] == start,
                    "failed full-state restore",
                )
                require(
                    restored["segment_plan_sha256"] == c.sha(folder / "plan.json")
                    and same(restored["boundary"], p["previous"]["boundaries"][str(rank)]),
                    "wrong restoration witness",
                )
            rank_logs.append(c.rows(audit / f"coverage-rank{rank}.jsonl"))
        logs.append(rank_logs)
        phases.append({"updates": [start + 1, end], "job": job, "root": str(folder)})
        previous_refs.append({"path": str(folder / "result.json"), "sha256": c.sha(folder / "result.json")})
        previous = result
    joined = join_phases(phases, logs, 935)
    counts, pads, tokens, curve = validate_coverage(joined, datasets, loss_tokens, native, cell)
    training, replay, final = [
        read(terminal / "output" / n)
        for n in ("primary-training-receipt.json", "continuation-receipt.json", "result.json")
    ]
    require(
        training["status"] == "FULL_TRAINING_COMPLETE_FROM_VERIFIED_SEGMENTS_DEPLOYMENT_PENDING"
        and same(training["lineage_receipts"], previous_refs),
        "incomplete full schedule",
    )
    require(
        final["status"] == "SEGMENTED_FULL_TRAINING_AND_RESTART_COMPLETE_DEPLOYMENT_PENDING"
        and final["full_campaign_pass"] is False,
        "wrong terminal result",
    )
    for name, key in (
        ("primary-training-receipt.json", "primary_receipt_sha256"),
        ("continuation-receipt.json", "continuation_receipt_sha256"),
    ):
        require(c.sha(terminal / "output" / name) == final[key] == a[key], "stale final receipt")
    for receipt in (training, replay):
        require(receipt["parent_plan_sha256"] == a["plan_sha256"], "wrong receipt lineage")
        for name, sha in receipt["evidence"].items():
            c.proof(name, sha)
    require(training["checkpoint"] == previous["next_predecessor"]["checkpoint"], "wrong scheduled final checkpoint")
    expected_names = {"data.pt", "trainer.pt", "model/.metadata", "optim/.metadata"} | {
        f"{k}/__{r}_0.distcp" for k in ("model", "optim") for r in range(4)
    }
    require(
        len(checkpoints) == 5
        and {Path(p).name for p in checkpoints} == {f"iter_{s:06d}" for s in (250, 467, 500, 750, 935)},
        "missing checkpoint generation",
    )
    for checkpoint, inventory in checkpoints.items():
        require(set(inventory) == expected_names, "incomplete checkpoint")
        for name, item in inventory.items():
            c.proof(Path(checkpoint) / name, item["sha256"], item["bytes"])
    coverage = read(terminal / "output/primary-coverage.json")
    require(c.sha(terminal / "output/primary-coverage.json") == training["coverage_sha256"], "stale coverage")
    require(
        coverage["per_id_counts"] == [dict(v) for v in counts]
        and coverage["valid_tokens"] == tokens
        and coverage["zero_weight_padding_per_epoch"] == pads,
        "coverage summary disagrees with raw rows",
    )
    validation = []
    for index in range(4):
        metrics = (
            terminal
            / f"output/segment-{index}/training/adapter_all_family_two_node/mistral_hf"
            / CELL
            / "campaign/metrics.jsonl"
        )
        validation.extend(v for v in c.rows(metrics) if "val/loss_avg" in v.get("metrics", v))
    require(
        [v["step"] for v in validation] == [0, 935] and same(validation, training["initial_final_validation"]),
        "wrong validation boundaries",
    )
    for row in validation:
        m = row.get("metrics", row)
        require(
            math.isfinite(m["val/loss_avg"])
            and m["val/valid_sequence_count"] == 1319
            and m["val/valid_token_count"] == native["validation"]["loss_tokens"],
            "invalid validation denominator",
        )
    for rank in range(4):
        value = read(terminal / f"output/segment-3/training/audit/final-validation-state-rank{rank}.json")
        validate_committed_validation(value, fingerprints["validation"])
        for name, sha in value["control_files_sha256"].items():
            c.proof(name, sha)
    require(
        replay["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION" and replay["replay_is_not_primary_coverage"] is True,
        "incomplete continuation",
    )
    require(
        same(replay["process"], read(terminal / "output/continuation-segment-1-process.json"))
        and replay["process"]["returncode"] == 0
        and replay["process"]["provenance"]["job"] == JOBS[1],
        "failed replay process",
    )
    for rank in range(4):
        folder = terminal / "output/continuation-segment-1/audit"
        initial = validate_rank(
            c,
            folder,
            spec=spec,
            plan=terminal,
            source=source,
            job=JOBS[1],
            hosts={v["host"] for v in c.measured["phases"][1]["rank_device_mapping"]},
            admission={"expected_adapter_identity": expected_identity},
            fingerprints=fingerprints,
            rank=rank,
            mode="branch",
            final_step=470,
        )
        require(
            initial["initial_clock"]["iter"]
            == initial["initial_clock"]["optimizer_step"]
            == initial["initial_clock"]["attempt_id"]
            == 467
            and initial["initial_clock"]["accum"] == 0,
            "wrong replay boundary",
        )
        require(read(folder / f"restored-state-rank{rank}.json")["status"] == "PASS", "failed exact replay restoration")
        rows = c.rows(folder / f"coverage-rank{rank}.jsonl")
        require([r["step"] for r in rows] == [468, 469, 470], "partial diagnostic continuation")
        for resumed in rows:
            step = resumed["step"]
            before = joined[rank][step - 1]
            comparison = read(folder / f"resume-update-{step}-rank{rank}.json")
            require(
                comparison["status"] == "PASS"
                and comparison["atol"] == 1e-7
                and comparison["rtol"] == 1e-5
                and comparison["reference"] == str(terminal / "output/midpoint-reference-segment-1"),
                "wrong continuation oracle",
            )
            require(same(resumed["inputs"], before["inputs"]), "replay consumed different examples")
            loss = before["metrics"]["train/loss_avg"]
            error = abs(loss - resumed["metrics"]["train/loss_avg"])
            require(
                math.isfinite(error) and error == comparison["loss_error"] and error <= 1e-6 + 1e-5 * abs(loss),
                "replay loss failed",
            )
    validate_tensor_audit(c.tensors)
    history = {
        "kind": "fresh_four_allocation_segmented_primary",
        "phases": phases,
        "audit": audit_ref,
        "cancelled_never_started_original": c.measured["cancelled_never_started_job"],
        "diagnostic_replayed_rows": 24,
        "allocated_gpu_hours": sum(int(accounts[j][7]) * 4 for j in JOBS) / 3600,
        "tensor_scope": "Independently reviewed actual saved-state comparisons and remote checkpoint digests; no new checkpoint load or training by this normalizer.",
    }
    lineage = {"primary_root": str(terminal), "primary_job": JOBS[-1], "primary_plan_sha256": a["plan_sha256"]}
    values = {
        "full_schedule": {
            "logical_updates": 935,
            "epochs": 1,
            "finite_applied_updates": True,
            "frozen_preserved": True,
            "two_physical_hosts": True,
            "loss_curve": curve,
        },
        "complete_sample_coverage": {
            "real_rows_per_epoch": [sum(v.values()) for v in counts],
            "zero_weight_padding_per_epoch": pads,
            "each_expected_id_once_per_epoch": True,
            "denominator_census_matches": True,
            "recovery_replays_excluded": True,
            "data_contract_sha256": identity["dataset_sha256"],
            "per_id_counts": [dict(v) for v in counts],
            "valid_tokens": tokens,
        },
        "initial_final_validation": {
            "steps": [0, 935],
            "counts": [1319, 1319],
            "complete_id_census": True,
            "native_denominators_verified": True,
            "finite_losses": True,
            "validation": validation,
            "census_basis": "Complete immutable split/native preprocessing census and exact global counts; no per-row runtime validation trace claimed.",
        },
        "mid_epoch_continuation": {
            "mid_update": 467,
            "updates_compared": 3,
            "same_topology": True,
            "input_ids_exact": True,
            "rng_exact": True,
            "cursor_exact": True,
            "clock_scheduler_exact": True,
            "loss_within_gates": True,
            "parameters_within_gates": True,
            "optimizer_within_gates": True,
            "initial_factors_unchanged": True,
            "source_immutable": True,
            "gates_sha256": GATES_SHA,
        },
    }
    for value in values.values():
        value["recovery_chain"] = history
    return values, source, lineage, list(c.references.values())


def produce(root, contract, output, *, audit_review, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output escapes evidence root")
    store = Store(root)
    reviewed = store.json(producer_review)
    require(
        reviewed["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and not reviewed["open_findings"], "unreviewed producer"
    )
    for name in DEPENDENCIES:
        require(
            reviewed["reviewed_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "producer/dependency changed after review",
        )
    require(reviewed["reviewed_files"].get(audit_review["path"]) == audit_review["sha256"], "unadmitted capture audit")
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(c for c in contract["cells"] if c["id"] == CELL)
    values, source, lineage, supporting = normalize(root, cell, audit_review)
    result = {"lineage": lineage, "final_checkpoint": None, "gates": {}}
    for gate, measurement in values.items():
        requirement = next(r for r in cell["requirements"] if r["id"] == gate)
        require(
            source in requirement["sources"] and producer in requirement["producers"], "producer/source not admitted"
        )
        reviews = producer_reviews(requirement, producer)
        require(not reviews or producer_review in reviews, "wrong producer review admission")
        closure = list({r["path"]: r for r in [*supporting, producer_review, *reviews]}.values())
        for ref in closure:
            store.bytes(ref)
        folder = output / CELL / gate
        m = write(root, folder / "measurement.json", measurement)
        common = {
            "identity": cell["identity"],
            "source": source,
            "profile": None,
            "producer_sha256": producer,
            "lineage": lineage,
            "final_checkpoint": None,
            "supporting_evidence": closure,
        }
        p = write(root, folder / "provenance.json", {**common, "documents_sha256": {"measurement": m["sha256"]}})
        record = {
            **common,
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "gate": gate,
            "outcome": "PASS",
            "executed": True,
            "documents": {"measurement": m, "provenance": p},
        }
        result["gates"][gate] = [write(root, folder / "receipt.json", record)]
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {CELL: result},
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "contract", "output", "audit-review", "producer-review"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    produce(
        root,
        read_json(Path(args.contract).read_bytes()),
        args.output,
        audit_review=file_ref(root, Path(args.audit_review).resolve()),
        producer_review=file_ref(root, Path(args.producer_review).resolve()),
    )
