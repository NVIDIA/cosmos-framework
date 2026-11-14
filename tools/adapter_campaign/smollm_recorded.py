# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize three preserved Smol DoRA FSDP checks with explicit historical lineage.

The original scalar-boundary-v2 oracle, shared mixed-parameter fault controls,
and completion-only midpoint replay are different schemas from the existing
v3/inline/recovered producers. This adapter does not rerun models or reinterpret
old numerical failures. Its closed initial admission is one exact campaign cell.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import stat
from pathlib import Path

from .evidence import Store, digest, producer_reviews, read_json, require, same
from .installed_preflight import SOURCE, number, rank_agreement, source_matches, validate_spec
from .primary import write

CELL = "llama_hf__dora__factorized_v1__default_v1__fsdp2"
MIXED = "llama_hf__mixed__factorized_v1__default_v1__fsdp2"
GATES_SHA = "75c7db901369169327bc5b3ba5508da55d55efee6e8ebef530d3d477fcf62d3b"
FAULT_SHA = "a209a1e389d653f43adf48464286259b5881b163ac856759eb91a6ca7b3b0bc2"
GATES = ("independent_global_update", "controlled_failure_recovery", "mid_epoch_continuation")
DEPENDENCIES = ("smollm_recorded.py", "installed_preflight.py", "evidence.py", "primary.py")
WORKERS = {
    "independent_text_update_reference_v2.py": "829a70dd211b49293a735e86974ef87e2c2bbeb4b9e8c6d8d796dd41bb88bd06",
    "text_reference_math_v2.py": "9c5898bc1e4f09c5d60feee8afd7988d375440508d66a079a160dac40d60cf80",
    "campaign_full_run_v1.py": "7cbc8b0282f020a00801a75693302f458925422712cddd93acf18b4f6b50d8cb",
    "run_controlled_failures_v2.py": "8db00a2183e264fdc0bac040089e19e28ffe54c78c0070c837c1d0ff31e5046a",
    "fault_worker_supervisor.py": "7835ffb1f9b02e2c91bf113d3affa0066f9ebb8a76d83243e200ae3fb2f76189",
    "campaign_recipe.py": "492ccf379af61cc652b8618f9e90978701677aa196f1afad9769640b8755acb2",
}
STATE_KEYS = {
    "parameters",
    "optimizer",
    "gradients",
    "optimizer_groups",
    "scheduler",
    "data",
    "clock",
    "rng",
    "rng_before",
    "inputs",
    "batches",
    "flow_randomness",
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def hash_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Capture(Store):
    """Small immutable envelopes retain raw bytes and stable remote tensor hashes."""

    def __init__(self, root, reference):
        super().__init__(root)
        self.manifest_ref = reference
        self.manifest = self.json(reference)
        m = self.manifest
        require(m["format"] == "smollm-recorded-three-gate-capture-v1" and m["cell"] == CELL, "wrong capture")
        self.remote_root = m["remote_root"]
        require(Path(self.remote_root).is_absolute() and ".." not in Path(self.remote_root).parts, "unsafe remote root")
        self.records, self.namespaces, self.accounting = {}, {}, {}
        formats = (
            "smollm-required-existing-sources-v1",
            "smollm-existing-midpoint-source-inventory-capture-v1",
            "smollm-existing-fault-source-closure-v1",
        )
        require(len(m["captures"]) == len(formats), "missing source captures")
        for ref, expected in zip(m["captures"], formats, strict=True):
            envelope = self.json(ref)
            require(same(envelope["returncode"], 0) and not envelope["stderr"], "failed remote source capture")
            body = read_json(envelope["stdout"])
            require(body["format"] == expected and body["synthetic"] is False, "wrong raw capture schema")
            for name, item in body["files"].items():
                p = Path(name)
                require(p.is_relative_to(self.remote_root) and ".." not in p.parts, "foreign remote path")
                s = item["stable_stat"]
                require(
                    type(item["bytes"]) is int
                    and item["bytes"] == s["st_size"] >= 0
                    and stat.S_ISREG(s["st_mode"])
                    and s["st_uid"] == 41852
                    and s["st_nlink"] == 1
                    and hash_value(item["sha256"]),
                    "unowned/partial source witness",
                )
                require(
                    name not in self.records or self.records[name]["sha256"] == item["sha256"], "conflicting captures"
                )
                if "text" in item:
                    data = item["text"].encode()
                    require(
                        len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                        "changed raw text",
                    )
                self.records[name] = item
            self.namespaces.update(body.get("namespaces", {}))
            self.accounting.update(body.get("accounting", {}))
        for ref in m["programs"] + m["source_reviews"]:
            self.bytes(ref)
        self.cpu = self.json(m["cpu_comparison"])
        require(
            self.cpu["format"] == "smollm-preserved-state-cpu-recomparison-v1"
            and self.cpu["status"] == "PASS_EXISTING_TENSOR_RECOMPARISON_ONLY"
            and self.cpu["synthetic"] is False
            and self.cpu["cell"] == CELL
            and same(self.cpu["captures"], m["captures"])
            and self.cpu["program_sha256"] == m["comparison_program"]["sha256"],
            "unbound CPU comparison",
        )
        self.bytes(m["comparison_program"])
        transfer = self.json(m["tensor_transfer"])
        self.transfer_plan = self.json(m["tensor_transfer_plan"])
        require(
            transfer["status"] == "PASS_HASH_BOUND_TRANSFER_ONLY"
            and same(transfer["returncode"], 0)
            and transfer["plan_sha256"] == self.cpu["transfer_plan_sha256"] == m["tensor_transfer_plan"]["sha256"]
            and transfer["files"] == len(self.transfer_plan["files"]),
            "incomplete tensor transfer",
        )
        expected = {r["source"]: r["sha256"] for r in self.transfer_plan["files"]}
        require(len(expected) == len(self.transfer_plan["files"]), "duplicate tensor source")
        for name, h in expected.items():
            require(self.sha(name) == h, "transferred tensor differs from raw witness")
        for name, h in self.cpu["consumed_tensors"].items():
            require(expected.get(name) == h, "CPU comparison used another tensor")

    def name(self, relative):
        return self.remote_root + "/" + relative

    def item(self, path):
        require(str(path) in self.records, "uncaptured source: " + str(path))
        return self.records[str(path)]

    def raw(self, path):
        item = self.item(path)
        require("text" in item, "source bytes were not captured: " + str(path))
        return item["text"].encode()

    def read(self, path):
        return read_json(self.raw(path))

    def sha(self, path):
        return self.item(path)["sha256"]

    def tensor(self, path):
        require(self.cpu["consumed_tensors"].get(path) == self.sha(path), "tensor omitted from CPU comparison")
        return self.sha(path)

    def plan(self, root):
        p = self.read(root + "/inputs/plan.json")
        require(p["root"] == root, "plan namespace changed")
        runtime = p
        if p.get("format") == "full-text-completion-v1":
            parent = p["primary_root"]
            require(parent == self.name("full-v4-" + CELL + "-attempt01"), "foreign completion parent")
            require(self.sha(parent + "/inputs/plan.json") == p["primary_plan_sha256"], "completion parent changed")
            runtime = self.read(parent + "/inputs/plan.json")
            require(p["source_revision"] == runtime["source_revision"], "completion source changed")
        for key, value in (
            ("source_revision", "revision"),
            ("wheel_sha256", "wheel_sha256"),
            ("image_sha256", "runtime_sha256"),
        ):
            require(runtime[key] == SOURCE[value], "runtime/source changed")
        for name, sha in p["files_sha256"].items():
            require(
                Path(name).name == name and self.sha(root + "/inputs/" + name) == sha, "executed plan source differs"
            )
            if name in WORKERS:
                require(sha == WORKERS[name], "unreviewed worker mathematics/lifecycle")
        return p

    def job(self, job, state, seconds, gpus, nodes):
        row = self.accounting[job]
        require(same(row["returncode"], 0) and row["stderr"] == "", "scheduler capture failed")
        lines = row["stdout"].strip().splitlines()
        require(len(lines) == 1, "ambiguous scheduler record")
        fields = lines[0].split("|")
        require(
            len(fields) == 7 and fields[:3] == [job, "41852", state] and int(fields[4]) == seconds, "wrong actual job"
        )
        resources = dict(x.split("=", 1) for x in fields[5].split(","))
        require(
            resources["gres/gpu"] == str(gpus) and resources["node"] == str(nodes), "wrong actual resource topology"
        )
        require(fields[3] == ("0:0" if state in ("COMPLETED", "TIMEOUT") else "1:0"), "job exit evidence differs")
        return {
            "job": job,
            "state": state,
            "elapsed_seconds": seconds,
            "allocated_gpus": gpus,
            "allocated_gpu_seconds": seconds * gpus,
            "nodes": fields[6],
        }


def validate_cpu_continuation(c, report, original, resumed, mid, owners):
    require(
        same(report["mid_update"], mid) and same(report["steps"], list(range(mid + 1, mid + 4))),
        "wrong CPU comparison range",
    )
    require(same([r["rank"] for r in report["ranks"]], list(range(4))), "missing/repeated comparison rank")
    for row in report["ranks"]:
        rank = row["rank"]
        require(row["initial_state_exact"] is True, "restore state differs")
        c.tensor(original + f"/update-{mid:06d}-rank{rank}.pt")
        c.tensor(resumed + f"/initial-state-rank{rank}.pt")
        require(
            same([u["step"] for u in row["updates"]], list(range(mid + 1, mid + 4))), "partial/rebased continuation"
        )
        for update in row["updates"]:
            step = update["step"]
            require(update["status"] == "PASS" and same(update["owners"], owners), "missing owner checks")
            require(
                set(update["max_errors"]) == STATE_KEYS and update["tensor_and_control_leaves_compared"] > 1000,
                "partial state comparison",
            )
            for key, error in update["max_errors"].items():
                require(
                    finite(error) and 0 <= error <= (1e-7 if key in {"parameters", "optimizer", "gradients"} else 0),
                    "CPU state exceeds original gates",
                )
            require(number(update["loss_error"], 0), "unexpected changed recorded loss")
            for base in (original, resumed):
                c.tensor(base + f"/update-{step:06d}-rank{rank}.pt")


def rank_evidence(c, audit, *, job, spec_sha, harness, step, cell):
    initials, completed = [], []
    for rank in range(4):
        initial = c.read(audit + f"/initial-rank{rank}.json")
        end = c.read(audit + f"/completed-updates-rank{rank}.json")
        provenance = c.read(audit + f"/provenance-rank{rank}.json")
        source_matches(provenance)
        require(
            provenance["slurm_job_id"] == job
            and provenance["campaign_spec_sha256"] == spec_sha
            and provenance["harness_sha256"] == harness,
            "wrong installed worker provenance",
        )
        require(
            same(end["step"], step)
            and end["status"] == "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING"
            and end["frozen_preserved"] is True,
            "incomplete state check",
        )
        public = c.read(audit + f"/public-result-rank{rank}.json")
        if step == 935:
            require(
                set(public) == {"start", "final_step", "final_loss"}
                and same(public["start"], 0)
                and same(public["final_step"], 935)
                and finite(public["final_loss"]),
                "full public workflow stopped early",
            )
        else:
            require(
                same(public, {"final_step": step, "preempted": "campaign diagnostic boundary"}),
                "bounded public workflow stopped early",
            )
        owners = [n for g in initial["optimizer"] for n in g["names"]]
        require(
            len(set(owners)) == len(owners) > 0
            and set(owners) == set(initial["parameter_layouts"]) == set(end["changed"]) == set(end["finite_gradients"])
            and all(v is True for v in end["changed"].values()),
            "incorrect optimizer ownership/updates",
        )
        peers = initial["topology"]["peers"]
        require(
            same([p["rank"] for p in peers], [0, 1, 2, 3])
            and same([p["local_rank"] for p in peers], [0, 1, 0, 1])
            and peers[0]["host"] == peers[1]["host"] != peers[2]["host"] == peers[3]["host"]
            and len({p["gpu_uuid"] for p in peers}) == 4
            and all(p["job"] == job and p["gpu_name"] == "NVIDIA A100-SXM4-80GB" for p in peers),
            "not two physical hosts/four A100s",
        )
        require(
            same(initial["adapter_identity"]["base"], cell["identity"]["base"])
            and same(initial["adapter_identity"]["processor"], cell["identity"]["processor"]),
            "wrong immutable base/processor",
        )
        initials.append(initial)
        completed.append(end)
    rank_agreement(initials, completed, "fsdp2")
    return initials


def independent_reference(c, cell, full):
    root = c.name("text-reference-v2-admission-attempt03")
    plan = c.plan(root)
    require(plan["format"] == "text-reference-correction-v2", "unadmitted oracle format")
    entry = next(r for r in plan["cells"] if r["id"] == CELL)
    spec = c.read(root + "/inputs/" + CELL + ".json")
    validate_spec(cell, full, spec)
    require(c.sha(root + "/inputs/" + CELL + ".json") == entry["sha256"], "wrong bounded recipe")
    receipt = c.read(root + "/output/" + CELL + "/receipt.json")
    require(
        receipt["cell"] == CELL
        and receipt["job"] == "38366620"
        and receipt["plan_sha256"] == c.sha(root + "/inputs/plan.json")
        and receipt["status"] == "PASS"
        and same(receipt["returncode"], 0),
        "incomplete/wrong reference execution",
    )
    original = entry["primary_folder"]
    require(
        original == receipt["primary_folder"] == c.name("preflight-v4-llama_hf-attempt02/output/" + CELL),
        "wrong distributed primary",
    )
    require(
        c.sha(original + "/primary-receipt.json")
        == entry["primary_receipt_sha256"]
        == receipt["original_primary_receipt_sha256"],
        "primary receipt changed",
    )
    require(
        c.sha(original + "/reference-receipt.json") == receipt["historical_reference_receipt_sha256"],
        "historical failure dropped",
    )
    historical = c.read(original + "/reference-receipt.json")
    require(
        type(historical["returncode"]) is int
        and historical["returncode"] != 0
        and historical["passed"] is False
        and historical["cell"] == CELL
        and historical["spec_sha256"] == entry["sha256"]
        and historical["job"] == "38363470",
        "historical failure reinterpreted",
    )
    result_path = root + "/output/" + CELL + "/oracle/results.json"
    require(c.sha(result_path) == receipt["results_sha256"], "result differs from original receipt")
    result = c.read(result_path)
    source = result["source"]
    source_matches(source)
    require(
        source["slurm_job_id"] == "38366620"
        and source["oracle_version"] == "text_native_scalar_boundary_v2"
        and source["oracle_sha256"] == WORKERS["independent_text_update_reference_v2.py"]
        and source["spec_sha256"] == entry["sha256"]
        and source["strict_determinism"] is True
        and source["initial_trainable_exact"] is True,
        "incomplete independent oracle identity",
    )
    for path, sha in source["imported_helpers"].items():
        require(c.sha(root + "/inputs/" + Path(path).name) == sha, "oracle helper changed")
    for name, sha in source["reference_files"].items():
        require(c.sha(original + "/primary/audit/" + name) == sha, "oracle used different distributed state")
    for path, sha in receipt["primary_evidence"].items():
        require(c.sha(path) == sha, "distributed primary proof changed")
    rank_evidence(
        c,
        original + "/primary/audit",
        job="38363470",
        spec_sha=entry["sha256"],
        harness="597cb9389c4ca4e7c0c246688000e09adc12bc7ee48fc7767952deac4b9f321f",
        step=6,
        cell=cell,
    )
    require(
        same(result["steps"], receipt["steps"]) and same([s["step"] for s in result["steps"]], [1, 2, 3]),
        "reference reset/skipped an update",
    )
    require(
        set(receipt["gradient_artifacts"]) == {f"gradient-{s:06d}.pt" for s in (1, 2, 3)},
        "missing saved independent gradients",
    )
    for name, sha in receipt["gradient_artifacts"].items():
        require(c.tensor(root + "/output/" + CELL + "/oracle/" + name) == sha, "independent gradient witness changed")
    cpu = c.cpu["reference"]["steps"]
    require(same([s["step"] for s in cpu], [1, 2, 3]), "CPU optimizer replay incomplete")
    for s, r, tokens, counts in zip(
        result["steps"], cpu, (1286, 830, 131), ([527, 759], [405, 425], [131, 0]), strict=True
    ):
        require(
            s["status"] == r["status"] == "PASS"
            and same(s["global_valid_tokens"], tokens)
            and same(r["global_valid_tokens"], tokens)
            and same(s["independent_micro_tokens"], counts)
            and same(r["owners"], 180),
            "wrong objective/tail population",
        )
        require(
            all(finite(s[k]) for k in ("global_loss", "distributed_loss", "max_gradient_error", "max_parameter_error")),
            "nonfinite oracle",
        )
        require(
            abs(s["global_loss"] - s["distributed_loss"]) <= 1e-6 + 1e-5 * abs(s["distributed_loss"]),
            "global objective differs",
        )
        require(
            all(finite(r[k]) for k in ("max_gradient_error", "max_cpu_parameter_error", "max_cpu_moment_error"))
            and r["max_gradient_error"] == s["max_gradient_error"]
            and 0 <= r["max_gradient_error"] <= 2e-6
            and 0 <= r["max_cpu_parameter_error"] <= 1e-7
            and 0 <= r["max_cpu_moment_error"] <= 1e-7,
            "saved gradient/Adam replay differs",
        )
    return {
        **dict.fromkeys(
            (
                "same_global_inputs",
                "independent_objective_weighting",
                "uninterrupted_multi_update_reference",
                "tail_and_uneven_lengths",
                "raw_gradients_within_gates",
                "optimizer_within_gates",
                "native_rng_replayed",
            ),
            True,
        ),
        "gates_sha256": GATES_SHA,
        "completed_steps": result["steps"],
        "oracle_version": source["oracle_version"],
        "cpu_saved_gradient_replay": cpu,
        "historical_reference_failure": {
            "path": original + "/reference-receipt.json",
            "sha256": receipt["historical_reference_receipt_sha256"],
        },
        "accounting": c.job("38366620", "COMPLETED", 425, 1, 1),
        "scope": "Preserved independent global-update prerequisite only. No new model execution; old raw-sum oracle failure retained.",
    }


def midpoint(c, cell, full):
    root = c.name("full-v4-" + CELL + "-attempt01")
    continuation = c.name("completion-v4-" + CELL + "-attempt01")
    plan = c.plan(continuation)
    require(plan["primary_root"] == root, "foreign full-run continuation")
    receipt = c.read(root + "/output/continuation-receipt.json")
    p = receipt["provenance"]
    require(
        receipt["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION"
        and p["job"] == "38449671"
        and p["cell_id"] == CELL
        and p["config_sha256"] == cell["identity"]["config_sha256"]
        and p["plan_sha256"] == c.sha(continuation + "/inputs/plan.json"),
        "wrong midpoint execution",
    )
    require(
        p["original_training"]["job"] == "38375170"
        and p["original_training"]["plan_sha256"] == c.sha(root + "/inputs/plan.json"),
        "foreign primary lineage",
    )
    process = receipt["process"]
    require(
        same(process["returncode"], 0)
        and process["status"] == "COMMAND_COMPLETE"
        and process["failure"] is None
        and same(process["provenance"], p),
        "continuation command failed",
    )
    command = process["command"]
    for flag, value in (
        ("--mode", "branch"),
        ("--branch-mid", "467"),
        ("--stop-after", "470"),
        ("--reference", root + "/output/primary/audit"),
    ):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, "wrong resume command")
    checkpoint = (
        root + "/output/primary/adapter_all_family_two_node/llama_hf/" + CELL + "/campaign/checkpoints/iter_000467"
    )
    require(command[command.index("--resume") + 1] == checkpoint, "warm start substituted for exact resume")
    inventory = c.read(continuation + "/output/original-source-inventory.json")
    require(len(inventory) == 118, "incomplete original namespace")
    for name, h in inventory.items():
        require(c.sha(root + "/" + name) == h, "original full training/checkpoint changed")
    for required in ("model/.metadata", "optim/.metadata", "trainer.pt", "data.pt"):
        require(c.item(checkpoint + "/" + required)["bytes"] > 0, "midpoint checkpoint incomplete")
    for path, sha in receipt["evidence"].items():
        require(c.sha(path) == sha, "continuation evidence changed")
    originals = rank_evidence(
        c,
        root + "/output/primary/audit",
        job="38375170",
        spec_sha=cell["identity"]["config_sha256"],
        harness=WORKERS["campaign_full_run_v1.py"],
        step=935,
        cell=cell,
    )
    resumed = rank_evidence(
        c,
        continuation + "/output/continuation/audit",
        job="38449671",
        spec_sha=cell["identity"]["config_sha256"],
        harness=WORKERS["campaign_full_run_v1.py"],
        step=470,
        cell=cell,
    )
    require(same(originals[0]["resolved"], resumed[0]["resolved"]), "resume changed adapter ownership")
    for rank in range(4):
        base = continuation + "/output/continuation/audit"
        require(
            same(
                c.read(base + f"/restored-state-rank{rank}.json"),
                {"status": "PASS", "comparison": "exact before first resumed forward"},
            ),
            "missing exact pre-forward restore",
        )
        for step in (468, 469, 470):
            row = c.read(base + f"/resume-update-{step}-rank{rank}.json")
            require(
                row["status"] == "PASS"
                and row["reference"] == root + "/output/primary/audit"
                and number(row["atol"], 1e-7)
                and number(row["rtol"], 1e-5)
                and number(row["loss_error"], 0),
                "wrong original continuation check",
            )
    validate_cpu_continuation(
        c, c.cpu["midpoint"], root + "/output/primary/audit", continuation + "/output/continuation/audit", 467, 180
    )
    return {
        "mid_update": 467,
        "updates_compared": 3,
        **dict.fromkeys(
            (
                "same_topology",
                "input_ids_exact",
                "rng_exact",
                "cursor_exact",
                "clock_scheduler_exact",
                "loss_within_gates",
                "parameters_within_gates",
                "optimizer_within_gates",
                "initial_factors_unchanged",
                "source_immutable",
            ),
            True,
        ),
        "gates_sha256": GATES_SHA,
        "cpu_state_recomparison": c.cpu["midpoint"],
        "initial_factors_scope": "Ordinary DoRA has no saved PiSSA initial factors; unchanged method/base and exact restored parameters are checked.",
        "accounting": [c.job("38375170", "FAILED", 1037, 4, 2), c.job("38449671", "COMPLETED", 115, 4, 2)],
        "scope": "Completed primary is preserved; separate recovery-only allocation replays 467→470. Original post-training host-audit failure retained.",
    }, {"primary_job": "38375170", "primary_root": root, "primary_plan_sha256": p["original_training"]["plan_sha256"]}


def fault_case(c, root, row, source, spec, cell, cpu):
    kind = row["kind"]
    require(
        kind in ("worker_microbatch", "node_workers_microbatch", "partial_checkpoint")
        and row["cell"] == MIXED
        and row["status"] == "PASS"
        and row["source_unchanged"] is True,
        "incomplete/foreign fault case",
    )
    plan = c.plan(root)
    provenance = row["provenance"]
    require(
        provenance["plan_sha256"] == c.sha(root + "/inputs/plan.json")
        and provenance["protocol_sha256"] == FAULT_SHA
        and same(provenance["files"], plan["files_sha256"]),
        "fault sources changed",
    )
    for key in ("source_revision", "wheel_sha256", "image_sha256"):
        require(provenance[key] == plan[key], "fault runtime differs")
    folder = root + "/output/" + MIXED
    audit = folder + "/" + kind + "/audit"
    checkpoint = source + "/primary/adapter_all_family_two_node/llama_hf/" + MIXED + "/campaign/checkpoints/iter_000003"
    inventory_path = folder + "/source-checkpoint-inventory.json"
    require(c.sha(inventory_path) == row["source_inventory_sha256"], "changed fault source inventory")
    inventory = c.read(inventory_path)
    require(
        {"model/.metadata", "optim/.metadata", "trainer.pt", "data.pt"} <= set(inventory), "uncommitted fault source"
    )
    for name, item in inventory.items():
        captured = c.item(checkpoint + "/" + name)
        require(
            captured["sha256"] == item["sha256"] and captured["bytes"] == item["bytes"] > 0,
            "fault changed committed source",
        )
    process = c.read(folder + "/" + kind + "-process.json")
    require(
        type(process["returncode"]) is int and process["returncode"] != 0 and process["watchdog_terminated"] is False,
        "fault did not occur or timed out",
    )
    arms = [c.read(audit + f"/fault-armed-rank{r}.json") for r in range(4)]
    job = "38364736" if root.endswith("-v1-attempt01") else "38373728"
    require(
        same([r["rank"] for r in arms], list(range(4)))
        and all(r["job_id"] == job and r["kind"] == kind and r["source_checkpoint"] == checkpoint for r in arms),
        "wrong owned worker fault",
    )
    require(
        len({r["host"] for r in arms}) == 2
        and len({r["step_id"] for r in arms}) == 1
        and same([r["node_id"] for r in arms], [0, 0, 1, 1])
        and arms[0]["host"] == arms[1]["host"] != arms[2]["host"] == arms[3]["host"],
        "fault was not on two physical hosts",
    )
    for rank, arm in enumerate(arms):
        require(
            same(arm["peers"], arms[0]["peers"])
            and same({k: v for k, v in arm.items() if k != "peers"}, arms[0]["peers"][rank]),
            "fault peer ownership differs",
        )
        require(
            same(arm["optimizer_step"], 4 if kind == "partial_checkpoint" else 3)
            and same(arm["accumulation_count"], 0 if kind == "partial_checkpoint" else 1),
            "wrong committed/accumulation boundary",
        )
    victims = [1] if kind == "worker_microbatch" else [2, 3]
    require(
        [r["rank"] for r in arms if r["target"]] == victims and all(type(r["target"]) is bool for r in arms),
        "wrong worker/node victims",
    )
    exits = []
    for rank in victims:
        arm = arms[rank]
        exit_row = c.read(audit + f"/worker-exit-rank{rank}.json")
        require(
            exit_row["pid"] == arm["pid"]
            and exit_row["rank"] == rank
            and exit_row["job_id"] == job
            and exit_row["step_id"] == arm["step_id"]
            and same(exit_row["returncode"], -9)
            and same(exit_row["forwarded_signals"], []),
            "worker failure was not the owned injected SIGKILL",
        )
        supervisor = c.read(audit + f"/supervisor-exit-rank{rank}.json")
        require(
            same(supervisor["victims"], victims) and supervisor["victim_receipts_complete"] is True,
            "supervisor did not observe owned deaths",
        )
        signals = supervisor["forwarded_signals"]
        signal_files = [n for n in c.namespaces[audit]["files"] if n.startswith(f"supervisor-signal-rank{rank}-")]
        require(len(signals) == len(signal_files), "unrecorded supervisor signal")
        for i, signal in enumerate(signals, 1):
            require(
                signal["time_ns"] > exit_row["time_ns"]
                and signal["signal"] in (15, 2, 1)
                and same(c.read(audit + f"/supervisor-signal-rank{rank}-{i}.json"), signal),
                "scheduler caused the death instead of the test",
            )
        exits.append(exit_row)
    duration = (process["ended_ns"] - min(r["time_ns"] for r in arms)) / 1e9
    require(
        0 <= duration <= 180
        and same(row["failure"]["victim_wait_status"], exits)
        and row["failure"]["global_failure_seconds"] == duration
        and row["failure"]["status"] == "PASS",
        "global failure detection did not pass",
    )
    require(
        same(
            row["failure"]["fault_receipts"],
            {f"fault-armed-rank{r}.json": c.sha(audit + f"/fault-armed-rank{r}.json") for r in range(4)},
        ),
        "fault witness hashes changed",
    )
    for node in (0, 1):
        cleanup = c.read(audit + f"/cleanup-node{node}.json")
        peers = [r for r in arms if r["node_id"] == node]
        require(
            cleanup["status"] == "PASS"
            and cleanup["job"] == job
            and {r["host"] for r in peers} == {cleanup["host"]}
            and same(
                sorted(cleanup["workers"], key=lambda r: r["rank"]),
                [{"rank": r["rank"], "pid": r["pid"], "same_live_process": False} for r in peers],
            ),
            "global teardown incomplete",
        )
    for phase in (kind + "-cleanup", kind + "-recovery"):
        process_row = c.read(folder + "/" + phase + "-process.json")
        require(
            same(process_row["returncode"], 0) and process_row["watchdog_terminated"] is False,
            "cleanup/recovery failed",
        )
    if kind == "partial_checkpoint":
        partial = arms[0]["partial_checkpoint"]
        require(all(r["partial_checkpoint"] == partial for r in arms), "different partial checkpoint per rank")
        namespace = c.namespaces[partial]
        require(
            "model/.metadata" in namespace["files"]
            and "trainer.pt" not in namespace["files"]
            and not any(
                n == "optim" or n.startswith("optim/") for n in namespace["directories"] + list(namespace["files"])
            ),
            "partial state was complete",
        )
        rejected = c.read(folder + "/" + kind + "-public-rejection-process.json")
        require(
            type(rejected["returncode"]) is int
            and rejected["returncode"] != 0
            and rejected["watchdog_terminated"] is False,
            "partial checkpoint accepted",
        )
        rejected_audit = folder + "/" + kind + "-public-rejection/audit"
        for rank in range(4):
            error = c.read(rejected_audit + f"/failure-rank{rank}.json")["traceback"]
            require(
                partial in error
                and ("incomplete" in error or "requires trainer state" in error)
                and f"update-000004-rank{rank}.pt" not in c.namespaces[rejected_audit]["files"],
                "wrong rejection or update after partial restore",
            )
    recovery = folder + "/" + kind + "-recovery/resume/audit"
    for path, h in row["three_update_continuation"].items():
        require(path.startswith(recovery + "/") and c.sha(path) == h, "fault recovery changed witnesses")
    rank_evidence(
        c,
        recovery,
        job=job,
        spec_sha=c.sha(root + "/inputs/" + MIXED + ".json"),
        harness=plan["files_sha256"]["campaign_public_run.py"],
        step=6,
        cell=cell,
    )
    for rank in range(4):
        require(
            same(
                c.read(recovery + f"/restored-state-rank{rank}.json"),
                {"status": "PASS", "comparison": "exact before first resumed forward"},
            ),
            "fault restore not exact",
        )
        for step in (4, 5, 6):
            check = c.read(recovery + f"/resume-update-{step}-rank{rank}.json")
            require(
                check["status"] == "PASS"
                and check["reference"] == source + "/primary/audit"
                and number(check["atol"], 1e-7)
                and number(check["rtol"], 1e-5)
                and number(check["loss_error"], 0),
                "fault continuation failed",
            )
    validate_cpu_continuation(c, cpu, source + "/primary/audit", recovery, 3, 141)
    return {
        "kind": kind,
        "job": job,
        "victims": victims,
        "global_failure_seconds": duration,
        "source_inventory_sha256": row["source_inventory_sha256"],
        "cpu_state_recomparison": cpu,
    }


def controlled_recovery(c, cell, full):
    root = c.name("controlled-failure-llama-v2-attempt01")
    plan = c.plan(root)
    require(
        plan["format"] == "controlled-failure-continuation-v2"
        and plan["binding"] == "llama_hf"
        and plan["physical_nodes"] == 2
        and plan["protocol_sha256"] == FAULT_SHA,
        "wrong fault lifecycle",
    )
    require(c.sha(root + "/inputs/controlled-failure-protocol-v1.json") == FAULT_SHA, "fault protocol changed")
    protocol = c.read(root + "/inputs/controlled-failure-protocol-v1.json")
    item = next(r for r in protocol["diagnostic_profiles"] if r["cell"] == MIXED)
    require(
        item["reason"] == "HF text tied embeddings/LLMTrainer/portable and DCP"
        and same(item["diagnostics"], ["worker_microbatch", "node_workers_microbatch", "partial_checkpoint"]),
        "wrong shared lifecycle applicability",
    )
    entry = next(r for r in plan["cells"] if r["id"] == MIXED)
    source = entry["reference_folder"]
    spec = c.read(root + "/inputs/" + MIXED + ".json")
    require(
        c.sha(root + "/inputs/" + MIXED + ".json") == entry["sha256"]
        and same(spec["binding"], full["binding"])
        and same(spec["cell"]["topology"], full["cell"]["topology"]),
        "mixed control does not share binding/trainer/topology",
    )
    initial = c.read(source + "/primary/audit/initial-rank0.json")
    require(
        initial["resolved"]["extra_parameters"]
        and {r["settings"]["method"] for r in initial["resolved"]["targets"]} == {"lora", "dora", "pissa"}
        and any(r["settings"]["use_rslora"] for r in initial["resolved"]["targets"]),
        "mixed parameter families not exercised",
    )
    components = spec["cell"]["adapter_config"]["components"]
    configured = {name: component for component in components for name in component["targets"]}
    require(
        len(configured) == sum(len(component["targets"]) for component in components) == 60,
        "mixed target selectors overlap",
    )
    require(
        {target["name"] for target in initial["resolved"]["targets"]} == set(configured),
        "mixed targets differ from fixed recipe",
    )
    for target in initial["resolved"]["targets"]:
        component = configured[target["name"]]
        require(
            same(target["settings"], {key: component[key] for key in target["settings"]})
            and target["owner"] == target["name"],
            "mixed independent settings changed",
        )
    require(
        same(
            initial["resolved"]["extra_parameters"],
            [
                {
                    "name": "model.norm.weight",
                    "component": "language",
                    "shape": [576],
                    "dtype": "float32",
                    "lr_multiplier": 0.07,
                    "weight_decay": 0.01,
                    "owner": "model.norm.weight",
                }
            ],
        ),
        "extra norm ownership/settings changed",
    )
    aggregate = c.read(root + "/output/controlled-failure-results.json")
    require(aggregate["job"] == "38373728" and len(aggregate["rows"]) == 6, "incomplete original fault aggregate")
    selected = []
    for wrapper in aggregate["rows"]:
        row = wrapper.get("record", wrapper)
        if row["cell"] != MIXED:
            continue
        actual_root = root
        if "record" in wrapper:
            reuse = wrapper["reused_from"]
            require(
                same(reuse, plan["completed_diagnostics"][0])
                and c.sha(reuse["path"]) == reuse["sha256"]
                and same(c.read(reuse["path"]), row),
                "unbound reused diagnostic",
            )
            for name, h in reuse["evidence_files"].items():
                require(c.sha(name) == h, "reused diagnostic witness changed")
            actual_root = c.name("controlled-failure-llama-v1-attempt01")
        selected.append(fault_case(c, actual_root, row, source, spec, cell, c.cpu["controlled_recovery"][row["kind"]]))
    require(
        same([r["kind"] for r in selected], ["worker_microbatch", "node_workers_microbatch", "partial_checkpoint"]),
        "missing/duplicate fault case",
    )
    return {
        **dict.fromkeys(
            (
                "worker_loss",
                "job_workers_one_node_loss",
                "global_teardown",
                "partial_checkpoint_rejected",
                "committed_boundary_recovered",
                "mixed_native_parameters_covered",
                "lifecycle_and_topology_match",
            ),
            True,
        ),
        "protocol_sha256": FAULT_SHA,
        "actual_mixed_cell": MIXED,
        "cases": selected,
        "accounting": [c.job("38364736", "FAILED", 1306, 4, 2), c.job("38373728", "COMPLETED", 1280, 4, 2)],
        "scope": "Shared fixed mixed-parameter HF trainer/FSDP lifecycle control; the homogeneous DoRA primary was not itself killed. Original failed allocation and reused case stay separate; shared allocation costs are not per-cell costs.",
    }


def normalize(root, cell, manifest_reference):
    require(
        cell["id"] == CELL and same(cell["identity"]["training_execution"], ["factorized_v1", "default_v1"]),
        "producer does not admit this cell/profile",
    )
    c = Capture(root, manifest_reference)
    full_root = c.name("full-v4-" + CELL + "-attempt01")
    full = c.read(full_root + "/inputs/cell.json")
    require(
        c.sha(full_root + "/inputs/cell.json") == cell["identity"]["config_sha256"]
        and digest(full["cell"]) == cell["identity"]["inventory_cell_sha256"],
        "frozen full recipe changed",
    )
    reference = independent_reference(c, cell, full)
    recovery = controlled_recovery(c, cell, full)
    continuation, lineage = midpoint(c, cell, full)
    return (
        {GATES[0]: reference, GATES[1]: recovery, GATES[2]: continuation},
        SOURCE,
        lineage,
        [{"path": p, "sha256": h} for p, h in sorted(c.verified.items())],
    )


def produce(root, contract, manifest_reference, review_reference, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output escapes evidence root")
    store = Store(root)
    review = store.json(review_reference)
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
        and not review["open_findings"]
        and same(review["recorded_capture_manifest"], manifest_reference),
        "source/capture needs independent review",
    )
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in DEPENDENCIES:
        relative = "tools/adapter_campaign/" + name
        require(
            review["reviewed_repository_files"].get(relative)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "unreviewed producer/dependency bytes",
        )
    cell = next(r for r in contract["cells"] if r["id"] == CELL)
    requirements = {r["id"]: r for r in cell["requirements"]}
    for gate in GATES:
        require(
            producer in requirements[gate]["producers"]
            and SOURCE in requirements[gate]["sources"]
            and review_reference in producer_reviews(requirements[gate], producer),
            "producer/source/review needs explicit admission",
        )
        pointer, expected = (
            ("/protocol_sha256", FAULT_SHA) if gate == "controlled_failure_recovery" else ("/gates_sha256", GATES_SHA)
        )
        require(
            any(
                clause.get("document") == "measurement"
                and clause.get("pointer") == pointer
                and clause.get("operator") == "eq"
                and clause.get("expected") == expected
                for clause in requirements[gate]["assertions"]
            ),
            "changed frozen numerical or fault contract",
        )
    measurements, source, lineage, supporting = normalize(root, cell, manifest_reference)
    for ref in supporting:
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "review omitted a consumed source")
    supporting += [review_reference]
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {CELL: {"gates": {}}},
    }
    for gate, measurement in measurements.items():
        directory = output / CELL / gate
        m = write(root, directory / "measurement.json", measurement)
        actual_lineage = lineage if gate == "mid_epoch_continuation" else None
        provenance = {
            "identity": cell["identity"],
            "source": source,
            "profile": None,
            "producer_sha256": producer,
            "lineage": actual_lineage,
            "final_checkpoint": None,
            "documents_sha256": {"measurement": m["sha256"]},
            "supporting_evidence": supporting,
        }
        p = write(root, directory / "provenance.json", provenance)
        receipt = {
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "identity": cell["identity"],
            "gate": gate,
            "profile": None,
            "source": source,
            "producer_sha256": producer,
            "lineage": actual_lineage,
            "final_checkpoint": None,
            "outcome": "PASS",
            "executed": True,
            "documents": {"measurement": m, "provenance": p},
            "supporting_evidence": supporting,
        }
        index["cells"][CELL]["gates"][gate] = [write(root, directory / "receipt.json", receipt)]
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "contract", "manifest", "manifest-sha256", "review", "review-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    produce(
        args.root,
        read_json(Path(args.contract).read_bytes()),
        {"path": args.manifest, "sha256": args.manifest_sha256},
        {"path": args.review, "sha256": args.review_sha256},
        args.output,
    )
