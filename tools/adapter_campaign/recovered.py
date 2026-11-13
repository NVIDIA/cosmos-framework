# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize a reviewed recovered logical primary without rewriting its history.

Small records are revalidated locally. Large checkpoint/reference tensors and
remote namespace checks rely on an explicitly admitted, source-bound read-only
semantic audit, not synthetic local placeholders or another training execution.
The original failed process, discarded suffix, and diagnostic replay stay visible.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, producer_reviews, read_json, require, same
from tools.adapter_campaign.primary import file_ref, write

HELPERS = {
    "full_training_evidence_v2.py": "5e4ebd6b82ded0c325ebadc184f1d8670046bd9ecaa1a9a38d17f75a4b11ce81",
    "lost_final_state_admission_v1.py": "1c21c6140e4c4c626b7896c26a646f4fe06cc9c9055f94c2de5fb76f13eb96e6",
    "recovered_primary_evidence_v1.py": "2c28a3ceedd8911779c4120cc0a0c095cfb5ab58239170808cafcfd06f0f966f",
    "run_text_family_preflights.py": "a4319bd1d6357b3eb5c4d9c5db4556d28cba2831842753bee8298101adb60e78",
}
WRAPPERS = {
    "campaign_full_run_v1.py": "7cbc8b0282f020a00801a75693302f458925422712cddd93acf18b4f6b50d8cb",
    "campaign_recovered_full_run_v1.py": "8b75c4388c34d9dea1414fa5a6e52cea1dae2cb2de14aded0b86a34571a02438",
}


class Capture:
    """Explicit remote-name resolver; never translate or edit producer JSON."""

    def __init__(self, root, receipt, review_ref):
        self.root, self.store, self.references = Path(root), Store(root), {}
        review = self.document(review_ref)
        require(review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS", "unreviewed recovery capture")
        require(
            not review["open_confirmed_findings"] and review["capture_result"] == file_ref(self.root, receipt),
            "review is for another capture",
        )
        self.reviewed = review["reviewed_files"]
        # The admission includes the actual SSH command envelopes and post-
        # transfer namespace checks. Their presence cannot be inferred from a
        # summary flag or from the subset of records consumed below.
        for path, expected in self.reviewed.items():
            self.bytes({"path": path, "sha256": expected})
        receipt_ref = file_ref(self.root, receipt)
        require(self.reviewed.get(receipt_ref["path"]) == receipt_ref["sha256"], "review does not bind this capture")
        self.receipt = self.document(receipt_ref)
        require(self.receipt["status"] == "PASS_EXACT_RAW_EVIDENCE_CAPTURE", "incomplete recovery capture")
        self.plan = self.reviewed_document(self.receipt["plan"])
        self.resolver = self.reviewed_document(self.receipt["resolver"])
        require(self.plan["format"] == "recovered-primary-small-evidence-capture-plan-v1", "unknown capture format")
        require(self.resolver["remote_root"] == self.plan["remote_root"], "resolver changed remote root")
        self.remote_root = Path(self.plan["remote_root"])
        require(set(self.plan["files"]) == set(self.resolver["files"]), "capture resolver inventory differs")
        expected = {v["relative_path"]: v["sha256"] for v in self.plan["files"].values()}
        require(expected == self.receipt["files_sha256"], "capture receipt inventory differs")
        require(self.receipt["files"] == len(expected), "partial raw capture")
        self.reviewed_document(self.plan["prior_attestations"])
        for key in ("capture_driver", "remote_audit_program"):
            ref = self.plan[key]
            require(self.reviewed.get(ref["path"]) == ref["sha256"], "unreviewed remote audit program")
            self.bytes(ref)
        self.audits = {}
        for ref in self.plan["semantic_audits"]:
            value = self.reviewed_document(ref)
            require(value["cell"] not in self.audits, "ambiguous recovery audit")
            require(value["status"] == "PASS_PINNED_SEMANTIC_REVALIDATION_AND_RAW_MANIFEST", "semantic audit failed")
            require(all(value["helper_sha256"].get(k) == v for k, v in HELPERS.items()), "changed semantic validator")
            require(value["remote_root"] == str(self.remote_root), "semantic audit root mismatch")
            self.audits[value["cell"]] = value
        for name, value in self.plan["files"].items():
            relative = Path(name).relative_to(self.remote_root)
            require(str(relative) == value["relative_path"] and ".." not in relative.parts, "unsafe remote name")
            resolved = self.resolver["files"][name]
            require(
                resolved["sha256"] == value["sha256"] and resolved["bytes"] == value["bytes"], "changed raw binding"
            )
            require(resolved["canonical_remote_path"] == value["canonical_path"], "changed canonical source")
        require(
            same(self.plan["large_objects_not_transferred"], self.resolver["large_objects_not_transferred"])
            and same(
                self.plan["continuation_links_not_materialized"], self.resolver["continuation_links_not_materialized"]
            ),
            "resolver changed remote-only evidence",
        )

    def bytes(self, ref):
        self.references[ref["path"]] = ref
        return self.store.bytes(ref)

    def document(self, ref):
        return read_json(self.bytes(ref))

    def reviewed_document(self, ref):
        require(self.reviewed.get(ref["path"]) == ref["sha256"], "capture review omitted an attestation")
        return self.document(ref)

    def raw(self, remote):
        name = str(remote)
        require(name in self.resolver["files"], "uncaptured producer record: " + name)
        item = self.resolver["files"][name]
        raw = self.bytes({"path": item["path"], "sha256": item["sha256"]})
        require(len(raw) == item["bytes"], "raw source size changed")
        return raw

    def json(self, remote):
        return read_json(self.raw(remote))

    def rows(self, remote):
        return [read_json(row) for row in self.raw(remote).split(b"\n") if row]

    def sha(self, remote):
        return hashlib.sha256(self.raw(remote)).hexdigest()

    def remote_tensor_sha(self, remote, audit):
        name = str(remote)
        require(name in audit["large_objects"], "tensor absent from actual remote audit")
        item = audit["large_objects"][name]
        require(same(item, self.plan["large_objects_not_transferred"][name]), "remote tensor inventory changed")
        require(
            item["remote_path"] == name and item["bytes"] == item["stat"]["st_size"] > 0, "wrong remote tensor identity"
        )
        require(re.fullmatch("[0-9a-f]{64}", item["sha256"]) is not None, "invalid remote tensor digest")
        return item["sha256"]


def validate_rank(
    capture, audit_path, *, spec, plan, source, job, hosts, admission, fingerprints, rank, mode, final_step
):
    """Check real rank placement/owners for each original, recovered and replay process."""
    c = capture
    provenance = c.json(audit_path / f"provenance-rank{rank}.json")
    require(
        provenance["source_revision"] == source["revision"]
        and provenance["wheel_sha256"] == source["wheel_sha256"]
        and provenance["container_sha256"] == source["runtime_sha256"],
        "rank source mismatch",
    )
    require(provenance["campaign_spec_sha256"] == c.sha(plan / "inputs/cell.json"), "rank config mismatch")
    require(provenance["slurm_job_id"] == job and provenance["mode"] == mode, "rank execution mismatch")
    require(provenance["harness_sha256"] == WRAPPERS["campaign_full_run_v1.py"], "unreviewed training wrapper")
    require(
        provenance["verified_package_files"] > 0 and provenance["source_dirty"] is False, "unattested installed package"
    )
    initial = c.json(audit_path / f"initial-rank{rank}.json")
    require(
        initial["data_identity"] == fingerprints["train"]
        and initial["validation_identity"] == fingerprints["validation"],
        "executed split differs",
    )
    require(
        same(initial["adapter_identity"], admission["expected_adapter_identity"]), "adapter/processor identity changed"
    )
    owners = [n for group in initial["optimizer"] for n in group["names"]]
    require(
        bool(owners) and len(owners) == len(set(owners)) and set(owners) == set(initial["parameter_layouts"]),
        "optimizer ownership mismatch",
    )
    peers = initial["topology"]["peers"]
    require(
        [p["rank"] for p in peers] == list(range(4)) and len({p["gpu_uuid"] for p in peers}) == 4, "wrong actual ranks"
    )
    require(sorted(Counter(p["host"] for p in peers).values()) == [2, 2], "requires two physical equal-GPU hosts")
    require(
        {p["host"] for p in peers} == set(hosts) and all(p["job"] == job for p in peers), "wrong scheduler allocation"
    )
    require(initial["topology"]["layouts"], "missing parameter mesh")
    for layout in initial["topology"]["layouts"].values():
        groups = layout["groups"]
        for group in groups.values():
            require(
                rank in group["ranks"] and set(group["hosts"]) == {peers[r]["host"] for r in group["ranks"]},
                "group placement differs from physical ranks",
            )
        if spec["cell"]["topology"]["layout"] == "fsdp2":
            require(
                layout["shape"] == [4]
                and groups["0"]["ranks"] == list(range(4))
                and len(groups["0"]["hosts"]) == 2
                and groups["0"]["placement"].startswith("S"),
                "FSDP must shard across hosts",
            )
        else:
            require(
                layout["shape"] == [2, 2]
                and len(groups["0"]["hosts"]) == 2
                and groups["0"]["placement"] == "R"
                and len(groups["1"]["hosts"]) == 1
                and groups["1"]["placement"].startswith("S")
                and all(len(g["ranks"]) == 2 for g in groups.values()),
                "HSDP must replicate across hosts and shard inside hosts",
            )
    if final_step is not None:
        finished = c.json(audit_path / f"completed-updates-rank{rank}.json")
        require(
            finished["status"] == "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING"
            and finished["step"] == final_step
            and finished["mode"] == mode,
            "incomplete update execution",
        )
        require(
            type(finished["finite_gradients"]) is list
            and bool(finished["finite_gradients"])
            and finished["frozen_preserved"] is True
            and set(finished["finite_gradients"]) == set(finished["changed"]) == set(owners)
            and all(v is True for v in finished["changed"].values()),
            "missing gradients/updates or changed frozen state",
        )
        require(
            c.json(audit_path / f"public-result-rank{rank}.json")["final_step"] == final_step,
            "public trainer stopped early",
        )
    return initial


def normalize_recovered_primary(root, contract_cell, capture, directory, *, capture_review):
    c = Capture(root, capture, capture_review)
    raw, read = c.raw, c.json
    recovered = Path(directory)
    plan = read(recovered / "inputs/plan.json")
    original = Path(plan["original_root"])
    old_plan = read(original / "inputs/plan.json")
    identity = contract_cell["identity"]
    gates_ref = {
        "path": "inputs/campaign-gates-v1.json",
        "sha256": "75c7db901369169327bc5b3ba5508da55d55efee6e8ebef530d3d477fcf62d3b",
    }
    gates = c.document(gates_ref)
    resume_gates = gates["shared"]["resume"]
    require(
        resume_gates["updates"] == 3
        and resume_gates["parameter_and_optimizer_atol"] == 1e-7
        and resume_gates["parameter_and_optimizer_rtol"] == 1e-5
        and resume_gates["loss_atol"] == 1e-6
        and resume_gates["loss_rtol"] == 1e-5,
        "unsupported frozen continuation gates",
    )
    spec = read(recovered / "inputs/cell.json")
    cell = spec["cell"]
    require(
        plan["format"] == "recovered-full-text-primary-v1" and plan["root"] == str(recovered), "wrong recovery plan"
    )
    require(original.parent == recovered.parent == c.remote_root and original != recovered, "wrong recovery ancestry")
    require(plan["cell"] == old_plan["cell"] == cell["id"] == contract_cell["id"], "wrong recovery cell")
    require(
        spec["binding"]["id"] == "qwen2_hf" and spec["binding"]["dataset"] == "gsm8k", "unadmitted recovery binding"
    )
    require(
        c.sha(recovered / "inputs/cell.json") == c.sha(original / "inputs/cell.json") == identity["config_sha256"]
        and digest(cell) == identity["inventory_cell_sha256"],
        "changed recovery config",
    )
    require(
        cell["optimizer_updates"] == 935
        and cell["training_epochs"] == 1
        and cell["checkpoint_contract"]["mid_epoch_update"] == 467,
        "unadmitted schedule",
    )
    require(
        cell["micro_batch_size"] == 1 and cell["grad_accum"] == 2 and cell["topology"]["world_size"] == 4,
        "wrong training budget",
    )
    require(c.sha(original / "inputs/plan.json") == plan["original_plan_sha256"], "changed original plan")
    for directory_, value in [(original, old_plan), (recovered, plan)]:
        for name, expected in value["files_sha256"].items():
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and c.sha(directory_ / "inputs" / name) == expected,
                "changed executed payload",
            )
    for name, expected in WRAPPERS.items():
        require(plan["files_sha256"][name] == expected, "unreviewed restore/continuation wrapper")
    for key in ["source_revision", "wheel_sha256", "image_sha256", "runtime"]:
        require(plan[key] == old_plan[key], "recovery changed execution environment")
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    audit = c.audits[contract_cell["id"]]
    require(
        audit["recovery_root"] == str(recovered) and audit["original_root"] == str(original), "wrong semantic lineage"
    )
    for name, value in audit["files"].items():
        require(
            same(value, c.plan["files"][name]) or value["sha256"] == c.plan["files"][name]["sha256"],
            "raw audit binding changed",
        )
        require(c.sha(name) == value["sha256"], "raw record differs from semantic revalidation")
    original_submission, submission = [read(path / "submission.json") for path in [original, recovered]]
    for item in [original_submission, submission]:
        require(
            type(item["returncode"]) is int
            and item["returncode"] == 0
            and item["stderr"] == ""
            and re.fullmatch("[1-9][0-9]*", item["job_id"]) is not None,
            "invalid scheduler acknowledgement",
        )
        require(
            re.fullmatch(re.escape(item["job_id"]) + r"(?:;[A-Za-z0-9_.-]+)?\n?", item["stdout"]) is not None,
            "job ID differs from ACK",
        )
    accounting = audit["accounting"]
    require(
        type(accounting["returncode"]) is int and accounting["returncode"] == 0 and accounting["stderr"] == "",
        "failed accounting query",
    )
    require(
        accounting["command"]
        == [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            original_submission["job_id"] + "," + submission["job_id"],
            "--format=JobID,State,ExitCode,Start,End,NodeList",
        ],
        "wrong accounting provenance",
    )
    states = [row.split("|") for row in accounting["stdout"].splitlines()]
    require(len(states) == 2 and all(len(row) == 6 for row in states), "incomplete accounting")
    require(
        states[0][:3] == [original_submission["job_id"], "FAILED", "1:0"]
        and states[1][:3] == [submission["job_id"], "COMPLETED", "0:0"],
        "original failure/recovery completion changed",
    )
    training, branch, result = [
        read(recovered / "output" / name)
        for name in ["primary-training-receipt.json", "continuation-receipt.json", "result.json"]
    ]
    require(
        training["status"] == "FULL_TRAINING_COMPLETE_FROM_VERIFIED_RECOVERY_DEPLOYMENT_PENDING"
        and branch["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION",
        "incomplete recovered training",
    )
    require(
        result["status"] == "RECOVERED_FULL_TRAINING_AND_RESTART_COMPLETE_DEPLOYMENT_PENDING"
        and result["full_campaign_pass"] is False,
        "wrong recovery result",
    )
    provenance = training["provenance"]
    require(
        same(provenance, branch["provenance"]) and same(provenance, result["provenance"]),
        "inconsistent recovered lineage",
    )
    lineage = {
        "primary_job": submission["job_id"],
        "primary_root": str(recovered),
        "primary_plan_sha256": c.sha(recovered / "inputs/plan.json"),
    }
    require(
        provenance["job"] == lineage["primary_job"]
        and provenance["plan_sha256"] == lineage["primary_plan_sha256"]
        and provenance["cell_id"] == cell["id"]
        and provenance["config_sha256"] == identity["config_sha256"],
        "wrong execution identity",
    )
    require(
        provenance["original_primary"] == str(original)
        and provenance["original_plan_sha256"] == plan["original_plan_sha256"]
        and provenance["committed_boundary"] == 467,
        "wrong recovery boundary",
    )
    for key in ["source_revision", "wheel_sha256", "image_sha256"]:
        require(provenance[key] == plan[key], "recovery process runtime changed")
    for receipt_ in [training, branch]:
        process = receipt_["process"]
        require(
            type(process["returncode"]) is int
            and process["returncode"] == 0
            and process["status"] == "COMMAND_COMPLETE"
            and same(process["provenance"], provenance),
            "incomplete recovered process",
        )
        for name, expected in receipt_["evidence"].items():
            require(c.sha(name) == expected, "stale completed rank evidence")
    for name, key in [
        ("primary-training-receipt.json", "primary_receipt_sha256"),
        ("continuation-receipt.json", "continuation_receipt_sha256"),
    ]:
        require(c.sha(recovered / "output" / name) == result[key], "recovery receipt chain changed")
    semantic = audit["semantic_validation"]
    require(
        semantic["complete_primary"] is True
        and semantic["exact_continuation"] is True
        and semantic["full_campaign_pass"] is False
        and semantic["logical_updates"] == 935,
        "semantic audit did not validate this recovery",
    )
    for name, expected in semantic["evidence"].items():
        require(c.sha(name) == expected, "semantic audit result differs from producer")
    decision = provenance["lost_final_state"]
    require(
        decision["reason"] == "FINAL_STATE_LOST_DURING_QUOTA_FAILED_PORTABLE_EXPORT"
        and decision["last_committed_update"] == 467
        and decision["applied_updates_observed"] == 935
        and decision["replayed_updates"] == list(range(468, 936))
        and decision["new_logical_training_examples"] == 0,
        "wrong lost-state decision",
    )
    require(
        decision["proof_sha256"] == c.sha(recovered / "inputs/final-state-loss-evidence.json")
        and decision["original_root"] == str(original)
        and decision["cell"] == cell["id"],
        "loss evidence changed",
    )
    require(read(recovered / "intent.json")["loss_decision"] == decision, "recovery intent changed")
    require(
        all(value["lexists"] is False for value in audit["explicit_absence_checks"].values()),
        "unexpected original final state",
    )
    checkpoint_audit = read(recovered / "inputs/checkpoint-audit.json")
    checkpoint = Path(checkpoint_audit["resume_checkpoint"])
    saved = next(v for v in checkpoint_audit["checkpoints"] if v["path"] == str(checkpoint))
    require(
        saved["step"] == 467 and saved["status"] == "COMPLETE_FILES_PENDING_EXACT_RESTORE", "wrong committed checkpoint"
    )
    for name, value in saved["files"].items():
        require(c.remote_tensor_sha(checkpoint / name, audit) == value["sha256"], "committed checkpoint changed")
    # The frozen logical dataset remains the original full corpus. It is not
    # inferred from either execution's partial coverage logs.
    base, plan_for_data = original, old_plan
    datasets, loss_tokens, fingerprints, native = validate_datasets(c, base, plan_for_data, spec)
    admission = read(original / "inputs/readiness-seal.json")
    require(
        admission["config_sha256"] == identity["config_sha256"] and admission["source_revision"] == source["revision"],
        "wrong original admission",
    )
    require(
        same(admission["expected_adapter_identity"]["base"], identity["base"])
        and same(admission["expected_adapter_identity"]["processor"], identity["processor"]),
        "changed base or processor",
    )
    original_audit = original / "output/primary/audit"
    recovered_audit = recovered / "output/primary/audit"
    branch_audit = recovered / "output/continuation/audit"
    chain_path = recovered / "output/authoritative-coverage/coverage-chain.json"
    chain = read(chain_path)
    require(
        chain["format"] == "recovered-primary-coverage-chain-v1"
        and chain["boundary"] == 467
        and chain["final_step"] == 935
        and same(chain["lost_final_state"], decision),
        "wrong coverage splice",
    )
    rank_logs = []
    phase_peers = {}
    for rank in range(4):
        original_initial = read(original_audit / f"initial-rank{rank}.json")
        original_hosts = {p["host"] for p in original_initial["topology"]["peers"]}
        for location, job, hosts, mode, final in [
            (original_audit, original_submission["job_id"], original_hosts, "full", None),
            (recovered_audit, submission["job_id"], provenance["hosts"], "full", 935),
            (branch_audit, submission["job_id"], provenance["hosts"], "branch", 470),
        ]:
            initial = validate_rank(
                c,
                location,
                spec=spec,
                plan=original if location == original_audit else recovered,
                source=source,
                job=job,
                hosts=hosts,
                admission=admission,
                fingerprints=fingerprints,
                rank=rank,
                mode=mode,
                final_step=final,
            )
            peers = initial["topology"]["peers"]
            require(location not in phase_peers or same(phase_peers[location], peers), "ranks disagree on allocation")
            phase_peers[location] = peers
            require(
                initial["initial_clock"]["optimizer_step"] == (0 if location == original_audit else 467),
                "wrong initial update clock",
            )
            require(
                initial["frozen_sha256"] == original_initial["frozen_sha256"],
                "frozen base/initial factors changed during recovery",
            )
        before_path, after_path = [d / f"coverage-rank{rank}.jsonl" for d in [original_audit, recovered_audit]]
        before, after = c.rows(before_path), c.rows(after_path)
        require(
            [v["step"] for v in before] == list(range(1, 936)) and [v["step"] for v in after] == list(range(468, 936)),
            "incomplete source trajectory",
        )
        require(
            chain["rank_source_sha256"][str(rank)] == {str(p): c.sha(p) for p in [before_path, after_path]},
            "changed coverage sources",
        )
        joined_path = chain_path.parent / before_path.name
        joined = c.rows(joined_path)
        require(
            same(joined, before[:467] + after)
            and chain["combined_files_sha256"][str(joined_path)] == c.sha(joined_path),
            "wrong logical coverage splice",
        )
        rank_logs.append(joined)
        restore = read(recovered_audit / f"recovered-primary-state-rank{rank}.json")
        require(
            restore["status"] == "PASS_EXACT_BEFORE_RECOVERED_PRIMARY_FORWARD"
            and restore["step"] == 467
            and restore["checkpoint"] == str(checkpoint)
            and restore["frozen_base_restored_exactly"] is True,
            "inexact recovery restoration",
        )
        require(
            restore["wrapper_sha256"] == WRAPPERS["campaign_recovered_full_run_v1.py"]
            and restore["original_identity_sha256"] == c.sha(original_audit / f"initial-rank{rank}.json"),
            "restoration identity differs",
        )
        require(
            restore["reference_sha256"] == c.remote_tensor_sha(original_audit / f"update-000467-rank{rank}.pt", audit),
            "wrong restoration state reference",
        )
        require(
            read(branch_audit / f"restored-state-rank{rank}.json")
            == {"status": "PASS", "comparison": "exact before first resumed forward"},
            "branch state restoration failed",
        )
        replay = c.rows(branch_audit / f"coverage-rank{rank}.jsonl")
        require([row["step"] for row in replay] == [468, 469, 470], "wrong diagnostic replay interval")
        for step in range(468, 471):
            value = read(recovered_audit / f"original-continuation-update-{step}-rank{rank}.json")
            require(
                value["status"] == "PASS_ORIGINAL_UNINTERRUPTED_CONTINUATION"
                and value["step"] == step
                and value["reference_sha256"]
                == c.remote_tensor_sha(original_audit / f"update-{step:06d}-rank{rank}.pt", audit),
                "original uninterrupted comparison changed",
            )
            resumed = read(branch_audit / f"resume-update-{step}-rank{rank}.json")
            require(
                resumed["status"] == "PASS"
                and resumed["reference"] == str(recovered / "output/continuation-reference"),
                "wrong branch comparison",
            )
            for comparison in [value, resumed]:
                require(
                    comparison["atol"] == 1e-7
                    and comparison["rtol"] == 1e-5
                    and math.isfinite(comparison["loss_error"])
                    and comparison["loss_error"] >= 0,
                    "changed continuation gates",
                )
            original_row, recovered_row, replay_row = before[step - 1], after[step - 468], replay[step - 468]
            require(
                same(original_row["inputs"], recovered_row["inputs"])
                and same(recovered_row["inputs"], replay_row["inputs"]),
                "continuation changed actual input IDs or padding",
            )
            for comparison, lhs, rhs in [(value, recovered_row, original_row), (resumed, replay_row, recovered_row)]:
                loss, reference = lhs["metrics"]["train/loss_avg"], rhs["metrics"]["train/loss_avg"]
                error = abs(loss - reference)
                require(
                    math.isfinite(loss)
                    and math.isfinite(reference)
                    and error == comparison["loss_error"]
                    and error <= resume_gates["loss_atol"] + resume_gates["loss_rtol"] * abs(reference),
                    "continuation loss differs from raw controlled trajectories",
                )
    discarded = {str(rank): list(range(468, 936)) for rank in range(4)}
    require(
        chain["discarded_uncommitted_updates"] == result["original_uncommitted_updates_discarded"] == discarded,
        "discarded work counted as new coverage",
    )
    require(
        branch["primary_rows_replayed"] == 24 and branch["replay_is_not_primary_coverage"] is True,
        "diagnostic replay counted as primary",
    )
    counts, pads, tokens, curve = validate_coverage(rank_logs, datasets, loss_tokens, native, cell)
    metric_suffix = Path("adapter_all_family_two_node/qwen2_hf") / cell["id"] / "campaign/metrics.jsonl"
    old_validation = [
        v for v in c.rows(original / "output/primary" / metric_suffix) if "val/loss_avg" in v.get("metrics", v)
    ]
    new_validation = [
        v for v in c.rows(recovered / "output/primary" / metric_suffix) if "val/loss_avg" in v.get("metrics", v)
    ]
    require(
        [v["step"] for v in old_validation] == [0] and [v["step"] for v in new_validation] in [[935], [467, 935]],
        "wrong validation boundaries",
    )
    validation = old_validation + [v for v in new_validation if v["step"] == 935]
    require(same(validation, training["initial_final_validation"]), "wrong validation splice")
    for row in old_validation + new_validation:
        m = row.get("metrics", row)
        require(
            math.isfinite(m["val/loss_avg"])
            and m["val/valid_sequence_count"] == 1319
            and m["val/valid_token_count"] == native["validation"]["loss_tokens"],
            "incomplete/nonfinite validation",
        )
    requirement = next(r for r in contract_cell["requirements"] if r["id"] == "mid_epoch_continuation")
    gates_digest = gates_ref["sha256"]
    require(
        next(r["expected"] for r in requirement["assertions"] if r["pointer"] == "/gates_sha256") == gates_digest,
        "contract differs from executed frozen gates",
    )
    # An explicit rich chain supplements (and never extends) the strict
    # three-field standard lineage used by the gate engine.
    history = {
        "kind": "recovered_logical_primary",
        "original": {
            "root": str(original),
            "job": original_submission["job_id"],
            "plan_sha256": plan["original_plan_sha256"],
            "scheduler": states[0],
        },
        "recovery": {**lineage, "scheduler": states[1]},
        "loss_decision": decision,
        "segments": [{"root": str(original), "updates": [1, 467]}, {"root": str(recovered), "updates": [468, 935]}],
        "discarded_uncommitted_updates": discarded,
        "diagnostic_replayed_rows": 24,
        "remote_tensor_scope": "Pinned remote digest/namespace/semantic audit; tensor values were not downloaded or independently re-compared by this producer.",
        "capture": file_ref(Path(root), capture),
        "capture_review": capture_review,
    }
    measurements = {
        "full_schedule": {
            "logical_updates": 935,
            "epochs": 1,
            "finite_applied_updates": True,
            "frozen_preserved": True,
            "two_physical_hosts": True,
            "loss_curve": curve,
            "recovery_chain": history,
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
            "recovery_chain": history,
        },
        "initial_final_validation": {
            "steps": [0, 935],
            "counts": [1319, 1319],
            "complete_id_census": True,
            "native_denominators_verified": True,
            "finite_losses": True,
            "validation": validation,
            "additional_recovery_validation": training["additional_recovery_validation"],
            "census_basis": "Complete immutable split and native preprocessing census; exact global counts; no per-row runtime validation trace claimed.",
            "recovery_chain": history,
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
            "gates_sha256": gates_digest,
            "recovery_chain": history,
        },
    }
    return measurements, source, lineage, list(c.references.values())


def validate_datasets(capture, base, plan, spec):
    raw = capture.raw
    identities = read_json(raw(base / "inputs/proofs/data-identities.json"))
    datasets = {}
    for split in ("train", "validation"):
        item = identities[split]
        require(item["path"] == spec["paths"][split], "census does not name the actual configured split")
        data_bytes = raw(item["path"])
        require(hashlib.sha256(data_bytes).hexdigest() == item["sha256"], "dataset content changed")
        datasets[split] = [read_json(line) for line in data_bytes.split(b"\n") if line]
        ids = [row["id"] for row in datasets[split]]
        require(len(ids) == len(set(ids)) == spec["dataset"]["rows_" + split], "wrong complete split census")
        require(
            all(
                set(row) == {"id", "messages"} and all(isinstance(m["content"], str) for m in row["messages"])
                for row in datasets[split]
            ),
            "producer only admits the frozen plain-text dataset fingerprint",
        )
    fingerprints = {split: digest({"rows": rows, "media": {}}) for split, rows in datasets.items()}
    census = read_json(raw(base / "inputs/proofs/preprocessing-census.json"))
    require(census["status"] == "PASS", "preprocessing census failed")
    native = {r["split"]: r for r in census["summaries"]}
    for split in datasets:
        r = native[split]
        require(
            r["ok"] is True and r["failed"] == r["unmeasured_rows"] == r["over_long_rows"] == 0,
            "incomplete preprocessing",
        )
        require(r["rows"] == r["matched_rows"] == len(datasets[split]), "partial preprocessing census")
        require(
            plan["expected_" + split.replace("validation", "validation").replace("train", "train") + "_valid_tokens"]
            == r["loss_tokens"],
            "wrong native denominator",
        )
    report_bytes = raw(base / "inputs/proofs/preprocessing-rows.jsonl")
    require(hashlib.sha256(report_bytes).hexdigest() == census["report_sha256"], "preprocessing rows changed")
    preprocessing = [read_json(line) for line in report_bytes.split(b"\n") if line]
    loss_tokens = {}
    for split, dataset in datasets.items():
        rows = [r for r in preprocessing if r.get("split") == split]
        require(
            [r["row_id"] for r in rows] == [r["id"] for r in dataset]
            and [r["index"] for r in rows] == list(range(len(dataset))),
            "preprocessing omitted/reordered/repeated an ID",
        )
        require(
            all(
                r["mask_matches"] is True
                and not r["reasons"]
                and type(r["loss_tokens"]) is int
                and r["loss_tokens"] > 0
                for r in rows
            ),
            "invalid native mask/token census",
        )
        require(
            sum(r["loss_tokens"] for r in rows) == native[split]["loss_tokens"],
            "preprocessing token total differs from rows",
        )
        loss_tokens[split] = {r["row_id"]: r["loss_tokens"] for r in rows}
    return datasets, loss_tokens, fingerprints, native


def validate_coverage(rank_logs, datasets, loss_tokens, native, cell):
    updates, epochs = cell["optimizer_updates"], cell["training_epochs"]
    train_ids = [r["id"] for r in datasets["train"]]
    global_batch = 4 * cell["grad_accum"]
    per_epoch = math.ceil(len(train_ids) / global_batch)
    require(updates == per_epoch * epochs, "wrong full epoch schedule")
    require(
        all(
            all(type(r["step"]) is int for r in log) and [r["step"] for r in log] == list(range(1, updates + 1))
            for log in rank_logs
        ),
        "missing/repeated/invalid updates",
    )
    counts, pads, tokens = [Counter() for _ in range(epochs)], [0] * epochs, 0
    curve = []
    for i in range(updates):
        e = i // per_epoch
        reference = rank_logs[0][i]["metrics"]
        real, expected_tokens = 0, 0
        for rank in range(4):
            record = rank_logs[rank][i]
            m = record["metrics"]
            require(
                math.isfinite(m["train/loss_avg"])
                and abs(m["train/loss_avg"] - reference["train/loss_avg"])
                <= 1e-6 + 1e-5 * abs(reference["train/loss_avg"]),
                "rank loss mismatch/nonfinite",
            )
            require(
                m["optim/update_outcome"] == "applied"
                and m["optim/step_skipped"] == m["optim/nonfinite_skips_total"] == 0
                and m["optim/optimizer_step"] == m["optim/attempt_id"] == i + 1,
                "skipped/missing update",
            )
            require(
                m["train/valid_token_count"] == reference["train/valid_token_count"]
                and m["train/valid_sequence_count"] == reference["train/valid_sequence_count"],
                "rank objective denominator mismatch",
            )
            require(len(record["inputs"]) == cell["grad_accum"], "wrong microbatch census")
            for item in record["inputs"]:
                require(
                    type(item["source_index"]) is int and 0 <= item["source_index"] < len(train_ids),
                    "invalid source index",
                )
                require(
                    type(item["epoch"]) is int
                    and item["epoch"] == e
                    and item["sample_id"] == train_ids[item["source_index"]],
                    "wrong actual example",
                )
                require(
                    type(item["padding"]) is bool and item["weight"] == float(not item["padding"]),
                    "padding contributed to loss",
                )
                if item["padding"]:
                    pads[e] += 1
                else:
                    counts[e][item["sample_id"]] += 1
                    real += 1
                    expected_tokens += loss_tokens["train"][item["sample_id"]]
        require(
            real == reference["train/valid_sequence_count"] and reference["train/valid_token_count"] > 0,
            "incorrect objective census",
        )
        require(
            reference["train/valid_token_count"] == expected_tokens,
            "per-update native objective denominator differs from actual examples",
        )
        tokens += reference["train/valid_token_count"]
        curve.append(
            {"step": i + 1, "loss": reference["train/loss_avg"], "valid_tokens": reference["train/valid_token_count"]}
        )
    require(all(count == Counter(train_ids) for count in counts), "missing/duplicated logical examples")
    require(
        pads == [per_epoch * global_batch - len(train_ids)] * epochs
        and tokens == epochs * native["train"]["loss_tokens"],
        "tail or token denominator mismatch",
    )
    return counts, pads, tokens, curve


def produce(root, contract, capture, output, *, capture_review, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output must be in evidence root")
    captured = Capture(root, capture, capture_review)
    directories = sorted(a["recovery_root"] for a in captured.audits.values())
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    store = Store(root)
    reviewed = store.json(producer_review)
    require(
        reviewed["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and not reviewed["open_findings"],
        "unreviewed recovered producer",
    )
    require(
        reviewed["reviewed_files"]["tools/adapter_campaign/recovered.py"] == producer,
        "normalizer review does not bind current source",
    )
    for bound in (capture_review, file_ref(root, capture)):
        require(
            reviewed["reviewed_files"].get(bound["path"]) == bound["sha256"],
            "normalizer review does not admit this raw capture review/result",
        )
    for name in ("evidence.py", "primary.py"):
        dependency = Path(__file__).with_name(name)
        require(
            reviewed["reviewed_files"]["tools/adapter_campaign/" + name]
            == hashlib.sha256(dependency.read_bytes()).hexdigest(),
            "normalizer dependency changed after independent review",
        )
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(contract), "cells": {}}
    cells = {c["id"]: c for c in contract["cells"]}
    for directory in directories:
        plan = captured.json(Path(directory) / "inputs/plan.json")
        cell = cells[plan["cell"]]
        values, source, lineage, supporting = normalize_recovered_primary(
            root, cell, capture, directory, capture_review=capture_review
        )
        result = {"lineage": lineage, "final_checkpoint": None, "gates": {}}
        require(cell["id"] not in index["cells"], "ambiguous authoritative primary")
        for gate, measurement in values.items():
            requirement = next(r for r in cell["requirements"] if r["id"] == gate)
            require(
                source in requirement["sources"] and producer in requirement["producers"],
                "producer/source needs independent admission",
            )
            admitted_reviews = producer_reviews(requirement, producer)
            require(
                not admitted_reviews or producer_review in admitted_reviews,
                "normalizer review is not admitted by this contract",
            )
            gate_supporting = list(
                {ref["path"]: ref for ref in [*supporting, producer_review, *admitted_reviews]}.values()
            )
            for review_ref in [producer_review, *admitted_reviews]:
                store.bytes(review_ref)
            destination = output / cell["id"] / gate
            ref = write(root, destination / "measurement.json", measurement)
            provenance = {
                "identity": cell["identity"],
                "source": source,
                "profile": None,
                "producer_sha256": producer,
                "lineage": lineage,
                "final_checkpoint": None,
                "documents_sha256": {"measurement": ref["sha256"]},
                "supporting_evidence": gate_supporting,
            }
            p = write(root, destination / "provenance.json", provenance)
            record = {
                "format": "adapter-campaign-gate-v1",
                "synthetic": False,
                "identity": cell["identity"],
                "gate": gate,
                "profile": None,
                "source": source,
                "producer_sha256": producer,
                "lineage": lineage,
                "final_checkpoint": None,
                "outcome": "PASS",
                "executed": True,
                "documents": {"measurement": ref, "provenance": p},
                "supporting_evidence": gate_supporting,
            }
            result["gates"][gate] = [write(root, destination / "receipt.json", record)]
        index["cells"][cell["id"]] = result
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "capture", "capture-review", "producer-review", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    produce(
        args.root,
        read_json(Path(args.contract).read_bytes()),
        args.capture,
        args.output,
        capture_review=file_ref(Path(args.root).resolve(), Path(args.capture_review).resolve()),
        producer_review=file_ref(Path(args.root).resolve(), Path(args.producer_review).resolve()),
    )
