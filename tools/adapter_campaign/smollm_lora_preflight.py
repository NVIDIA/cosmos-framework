# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Join own Smol LoRA six-update preflight with its executed AC/alias supplement.

Only the existing installed_cell_preflight gate is emitted. The old BLOCKED
receipt remains immutable: selecting this new attempt requires a separately
reviewed contract supersession. No model, optimizer or scheduler is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import stat
from pathlib import Path

from .evidence import Store, check_receipt, digest, producer_reviews, read_json, require, same
from .installed_preflight import ALIASES, GATES, SOURCE, TOKENS, number, validate_coverage
from .primary import write
from .smollm_lora_recorded import CELL, Capture as OriginalCapture, owners, preflight
from .smollm_recorded import rank_evidence

GATE = "installed_cell_preflight"
JOB = "38543030"
CAPTURE = "a55ed6edb5e40a290e768efac1536702383f9afd1ccf1661eb2fe17fa49accc1"
PLAN = "7ca8a18f3ce99a353d46a3b6977fc92a2d95890a43df619a185aa7bc1cfb5935"
COMPARISON = "cff53c9cf068b0fe40e169ab720574e0e625a6fff1c522337d3aac7cf98503cf"
SOURCE_REVIEW = "a37e9188b478e17d2cda49afdb475cf866acd29b5245d9e97a656c6e70b81963"
PROTOCOL = "61a54c70d04506f67b171fa6569761819ef59f397d79f71272bd7646e2677797"
BLOCKED = "8f7060905441b87426fcb2146e6c85f4534ec83917950104602999f97d88223e"
DEPENDENCIES = (
    "smollm_lora_preflight.py", "smollm_lora_recorded.py", "smollm_recorded.py",
    "installed_preflight.py", "evidence.py", "primary.py",
)
ASSERTIONS = (
    "installed_bytes_verified", "two_physical_hosts", "topology_verified", "all_targets_resolved",
    "owners_equal_optimizer", "aliases_preserved", "all_methods_verified", "frozen_preserved",
    "finite_updates", "all_whole_trainables_changed", "rank_agreement", "activation_checkpointing_checked",
)


def hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe captured relative path")
    return name


def captured_file(item):
    info = item["stat"]
    require(type(item["bytes"]) is int and item["bytes"] >= 0
            and same(item["bytes"], info["st_size"]) and hash_string(item["sha256"])
            and stat.S_ISREG(info["st_mode"]) and same(info["st_uid"], 41852)
            and same(info["st_gid"], 30) and same(info["st_nlink"], 1),
            "unowned or invalid captured file")


def completed_accounting(cap):
    for key in ("queue", "steps", "sacct", "all_step_accounting"):
        row = cap[key]
        require(same(row["returncode"], 0) and row["stderr"] == "", "failed scheduler capture")
    for key in ("queue", "steps"):
        require(not any(n.split(".", 1)[0] == JOB for n in cap[key]["stdout"].splitlines()),
                "job/step still active")
    rows = [line.split("|") for line in cap["all_step_accounting"]["stdout"].strip().splitlines()]
    expected = {JOB, JOB + ".batch", JOB + ".extern", *(JOB + "." + str(i) for i in range(3))}
    require(len(rows) == len(expected) and {r[0] for r in rows} == expected
            and all(len(r) == 7 and r[1:3] == ["COMPLETED", "0:0"] for r in rows),
            "incomplete or failed allocation/step closure")
    root = next(r for r in rows if r[0] == JOB)
    require(root == [JOB, "COMPLETED", "0:0", "279",
                     "billing=4,cpu=16,gres/gpu=4,mem=128G,node=2",
                     "batch-block5-[03724,03942]", "rarunachalam"], "foreign allocation/resources")
    require(cap["sacct"]["stdout"].strip().splitlines() == ["|".join(root)], "allocation captures disagree")
    return {"job": JOB, "state": "COMPLETED", "elapsed_seconds": 279, "allocated_gpus": 4,
            "physical_nodes": 2, "allocated_gpu_seconds": 1116, "allocated_gpu_hours": 0.31,
            "hosts": ["batch-block5-03724", "batch-block5-03942"],
            "scope": "Whole allocation charged once; phase elapsed times are not extra allocations."}


class Capture(Store):
    """Relocatable exact raw copies, bound to one immutable actual capture."""

    def __init__(self, root, reference):
        super().__init__(root)
        self.manifest = m = self.json(reference)
        require(m["format"] == "smollm-own-lora-preflight-supplement-v1" and m["cell"] == CELL,
                "wrong supplemental manifest")
        require(m["capture"]["sha256"] == CAPTURE and m["source_review"]["sha256"] == SOURCE_REVIEW,
                "wrong actual capture/source review")
        self.capture = cap = self.json(m["capture"])
        self.source_review = review = self.json(m["source_review"])
        require(cap["format"] == "smol-own-lora-orthogonal-terminal-capture-v1"
                and cap["job"] == JOB and cap["plan_sha256"] == PLAN
                and cap["source_revision"] == SOURCE["revision"]
                and cap["namespace_unchanged"] is True and cap["all_files_freshly_hashed"] is True
                and cap["qualification_pass"] is False and same(cap["new_gpu_executions"], 0),
                "incomplete/wrong terminal capture")
        require(review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == []
                and review["plan_sha256"] == PLAN, "source review does not admit executed plan")
        require(len(cap["files"]) == 175 and len(cap["mirrored_small_files"]) == 131
                and len(set(cap["mirrored_small_files"])) == 131
                and set(m["raw_files"]) == set(cap["mirrored_small_files"]),
                "incomplete/duplicate raw capture")
        for name, item in cap["files"].items():
            relative(name)
            captured_file(item)
        require(same(sum(r["bytes"] for r in cap["files"].values()), cap["total_bytes"]), "byte census changed")
        for name in m["raw_files"]:
            self.raw_bytes(name)
        self.plan = plan = self.read("inputs/plan.json")
        require(self.hash("inputs/plan.json") == PLAN and plan["root"] == cap["root"]
                and plan["cell"] == CELL, "wrong execution namespace/cell")
        require(set(n for n in cap["files"] if n.startswith("inputs/"))
                == {"inputs/plan.json", *("inputs/" + n for n in plan["files_sha256"])},
                "executed payload namespace differs")
        stage = Path(review["stage"])
        for name, sha in {"plan.json": PLAN, **plan["files_sha256"]}.items():
            require(Path(name).name == name and self.hash("inputs/" + name) == sha
                    and review["reviewed_files"].get(str(stage / name)) == sha,
                    "executed worker differs from source review")
        self.accounting = completed_accounting(cap)

    def hash(self, name):
        return self.capture["files"][relative(name)]["sha256"]

    def raw_bytes(self, name):
        relative(name)
        ref = self.manifest["raw_files"][name]
        require(ref["sha256"] == self.hash(name), "raw copy digest differs from actual capture")
        value = self.bytes(ref)
        require(len(value) == self.capture["files"][name]["bytes"], "raw copy length differs")
        return value

    def read(self, name):
        return read_json(self.raw_bytes(name))

    def finish(self):
        for name, sha in tuple(self.verified.items()):
            self.bytes({"path": name, "sha256": sha})


def plan_original(c, old, cell, spec, folder):
    p = c.plan
    original_root = str(Path(folder).parents[1])
    spec_sha = old.sha(original_root + "/inputs/" + CELL + ".json")
    initial = old.read(folder + "/primary/audit/initial-rank0.json")
    require(p["format"] == "smollm-lora-preflight-orthogonal-v1" and p["original_method"] == "lora"
            and p["original_primary_job"] == "38363470" and p["layout"] == "fsdp2"
            and p["original_config_sha256"] == c.hash("inputs/cell.json") == spec_sha
            and p["original_plan_sha256"] == old.sha(original_root + "/inputs/plan.json")
            and same(c.read("inputs/cell.json"), spec)
            and same(p["original_adapter_identity"], initial["adapter_identity"]),
            "supplement changes original config/targets/identity")
    for rank in range(4):
        name = f"initial-state-rank{rank}.pt"
        original_hash = old.tensor(folder + "/primary/audit/" + name)
        require(all(c.hash(f"output/{variant}/audit/" + name) == original_hash for variant in ("off", "on")),
                "supplement did not start from exact original rank state")
    require(p["source_revision"] == SOURCE["revision"] and p["wheel_sha256"] == SOURCE["wheel_sha256"]
            and p["image_sha256"] == SOURCE["runtime_sha256"]
            and same(p["steps"], [1, 2, 3]) and same(p["variants"], ["off", "on"])
            and same(p["gates"], GATES) and same(p["global_valid_tokens"], TOKENS)
            and same(p["qualification_credit"], 0)
            and p["protocol_sha256"] == c.hash("inputs/protocol.json") == PROTOCOL,
            "changed installed source or frozen supplemental protocol")
    cover = c.json(c.manifest["original_coverage"])
    require(c.manifest["original_coverage"]["sha256"]
            == p["original_source_manifest"]["coverage_supplement"]["sha256"]
            and cover["format"] == "smol-own-lora-original-coverage-supplement-v1"
            and cover["root"] == original_root and cover["job"] == "38363470"
            and cover["source_immutable"] is True and cover["read_only"] is True
            and same(cover["new_gpu_executions"], 0), "wrong original coverage")
    for name, item in cover["files"].items():
        relative(name)
        captured_file(item)
        raw = item["text"].encode()
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "original coverage text changed")
    original_plan = cover["files"]["inputs/plan.json"]
    require(original_plan["sha256"] == p["original_plan_sha256"], "coverage belongs to another preflight")
    samples = []
    for rank in range(4):
        name = "output/" + CELL + f"/primary/audit/coverage-rank{rank}.jsonl"
        item = cover["files"][name]
        require(p["worker_immutable_inputs"].get(original_root + "/" + name) == item["sha256"],
                "unbound original inputs")
        rows = [read_json(line) for line in item["text"].splitlines()]
        require(same([r["step"] for r in rows], list(range(1, 7))), "original preflight coverage incomplete")
        selected = [{k: r[k] for k in ("step", "inputs")} for r in rows[:3]]
        require(same(selected, p["expected_inputs"][str(rank)]), "supplement substituted original examples")
        samples.extend(x for r in selected for x in r["inputs"])
    real = [r for r in samples if r["padding"] is False]
    pads = [r for r in samples if r["padding"] is True]
    require(len(real) == 17 and len(pads) == 7 and len(samples) == 24
            and {r["source_index"] for r in real} == set(range(17))
            and len({r["sample_id"] for r in real}) == 17
            and all(number(r["weight"], 1) and same(r["epoch"], 0) for r in real)
            and all(number(r["weight"], 0) and same(r["epoch"], 0) for r in pads),
            "incomplete or incorrectly weighted tail")
    return spec_sha


def execution(c):
    plan, root = c.plan, c.plan["root"]
    intent, ack = c.read("intent.json"), c.read("submission.json")
    command = ["sbatch", "--parsable", "--export=ALL,SMOL_TRACE_PLAN_SHA256=" + PLAN
               + ",SMOL_TRACE_REVIEW_SHA256=" + SOURCE_REVIEW, root + "/inputs/job.sbatch"]
    require(same(intent, {"plan_sha256": PLAN, "review": {"path": plan["dispatch_review_path"],
                                                          "sha256": SOURCE_REVIEW}, "command": command})
            and same(ack, {"command": command, "plan_sha256": PLAN, "returncode": 0,
                           "stdout": JOB + "\n", "stderr": "", "job_id": JOB}),
            "wrong execution journals")
    result = c.read("output/result.json")
    require(result["job"] == JOB and result["cell"] == CELL and result["plan_sha256"] == PLAN
            and result["passed"] is True and result["source_immutable"] is True
            and result["full_campaign_pass"] is False and same(result["qualification_credit"], 0)
            and result["comparison_sha256"] == c.hash("output/comparison.json") == COMPARISON,
            "incomplete supplemental controller")
    require([r["phase"] for r in result["phases"]] == ["off", "on", "comparison"], "missing execution phase")
    for r in result["phases"]:
        require(same(r, c.read("output/" + r["phase"] + "-receipt.json"))
                and r["job"] == JOB and r["plan_sha256"] == PLAN and r["passed"] is True
                and same(r["returncode"], 0) and r["source_revision"] == SOURCE["revision"]
                and r["image_sha256"] == SOURCE["runtime_sha256"]
                and same(r["nodes"], c.accounting["hosts"])
                and type(r["wall_seconds"]) in (int, float) and math.isfinite(r["wall_seconds"])
                and 0 < r["wall_seconds"] <= 279, "failed/wrong/unmeasured phase")
    comparison = c.read("output/comparison.json")
    require(comparison["status"] == "PASS_SUPPLEMENTAL_ALIAS_AND_MATCHED_AC"
            and comparison["plan_sha256"] == PLAN and comparison["cell"] == CELL
            and comparison["source_revision"] == SOURCE["revision"]
            and comparison["native_replay_observed"] is True and comparison["full_training_credit"] is False
            and comparison["all_cells_qualified"] is False and same(comparison["gates"], GATES)
            and same(comparison["whole_model_aliases"], ALIASES)
            and same(comparison["real_rows"], 17) and same(comparison["zero_weight_tail_slots"], 7)
            and same(comparison["global_valid_tokens"], TOKENS), "incomplete actual comparison")
    leaves = {f"{name}-rank{r}.json" for name in ("initial", "completed-updates", "orthogonal-initial",
              "provenance", "public-result") for r in range(4)}
    leaves |= {f"orthogonal-step{s}-rank{r}.json" for s in (1, 2, 3) for r in range(4)}
    leaves |= {f"initial-state-rank{r}.pt" for r in range(4)}
    leaves |= {f"update-{s:06d}-rank{r}.pt" for s in (1, 2, 3) for r in range(4)}
    require(set(comparison["evidence"]) == {root + f"/output/{v}/audit/" + n for v in ("off", "on") for n in leaves},
            "partial or duplicate comparison witnesses")
    for name, sha in comparison["evidence"].items():
        require(c.hash(name[len(root) + 1:]) == sha, "changed comparison witness")
    validate_exact_pairs(comparison["steps"])
    return comparison


def validate_exact_pairs(rows):
    require(len(rows) == 12 and all(type(r["rank"]) is int and type(r["step"]) is int for r in rows)
            and {(r["rank"], r["step"]) for r in rows} == {(r, s) for r in range(4) for s in (1, 2, 3)},
            "incomplete/duplicate rank-update comparison")
    for row in rows:
        require(row["status"] == "PASS" and set(row["errors"]) == {"gradients", "parameters", "optimizer", "loss"}
                and all(type(x) is float and math.isfinite(x) and x == 0 for x in row["errors"].values()),
                "actual exact AC comparison failed")


def observed_step(receipt, variant, step):
    require(receipt["cell"] == CELL and receipt["variant"] == variant and receipt["plan_sha256"] == PLAN
            and same(receipt["step"], step) and receipt["whole_model_parameter_objects_unchanged"] is True
            and same(receipt["optimizer_owners"], 120) and same(receipt["aliases"], ALIASES),
            "changed updated ownership/aliases")
    population = {(f"model.layers.{i}", m) for i in range(30) for m in range(2)}
    entries = receipt["layer_entries"]
    require(len(entries) == 60 and {(r["layer"], r["micro"]) for r in entries} == population
            and all(type(r["micro"]) is int and type(r["entries"]) is int for r in entries)
            and all(r["entries"] >= 2 if variant == "on" else r["entries"] == 1 for r in entries),
            "missing actual checkpoint replay/incorrect off control")
    records = receipt["records"]
    require(len(records) == (60 if variant == "on" else 0), "wrong replay cardinality")
    if variant == "on":
        require({(r["layer"], r["micro"]) for r in records} == population
                and all(same(r["step"], step) and type(r["micro"]) is int
                        and type(r["entries"]) is int and r["entries"] >= 2
                        and type(r["replays"]) is int and r["replays"] >= 1
                        and hash_string(r["input_sha256"]) and hash_string(r["rng_sha256"]) for r in records),
                "missing replay of the same native call")
    require(number(receipt["metrics"]["train/valid_token_count"], TOKENS[str(step)]), "wrong loss denominator")


def observed_ranks(c, cell, spec, spec_sha, comparison):
    expected_owners = {f"model.layers.{i}.self_attn.{q}_proj.lora_{f}.weight"
                       for i in range(30) for q in ("q", "v") for f in ("A", "B")}
    for variant in ("off", "on"):
        prefix = "output/" + variant + "/audit"
        initials = rank_evidence(c, prefix, job=JOB, spec_sha=spec_sha, step=3, cell=cell,
                                harness=c.plan["files_sha256"]["smollm_preflight_orthogonal_v1.py"])
        ends = [c.read(prefix + f"/completed-updates-rank{r}.json") for r in range(4)]
        owners(initials, ends, spec, cell)
        for rank, initial in enumerate(initials):
            require(same(initial["adapter_identity"], c.plan["original_adapter_identity"])
                    and same(initial["topology"]["peers"], comparison["topology"])
                    and sorted({r["host"] for r in initial["topology"]["peers"]}) == c.accounting["hosts"],
                    "different original identity or physical ranks")
            provenance = c.read(prefix + f"/provenance-rank{rank}.json")
            require(provenance["orthogonal_plan_sha256"] == PLAN and provenance["variant"] == variant
                    and provenance["mode"] == "supplemental_alias_ac"
                    and provenance["public_grad_ckpt"] is (variant == "on"), "wrong public AC configuration")
            rows = [read_json(line) for line in c.raw_bytes(prefix + f"/coverage-rank{rank}.jsonl").splitlines()]
            validate_coverage(rows, c.plan["expected_inputs"][str(rank)])
            initial_orthogonal = c.read(prefix + f"/orthogonal-initial-rank{rank}.json")
            o = initial_orthogonal
            require(o["cell"] == CELL and o["variant"] == variant and o["plan_sha256"] == PLAN
                    and o["original_config_sha256"] == spec_sha and o["public_grad_ckpt"] is (variant == "on")
                    and same(o["layer_count"], 30) and same(o["aliases"], ALIASES)
                    and len(o["optimizer_owners"]) == 120 and set(o["optimizer_owners"]) == expected_owners
                    # This field counts every unique model parameter object:
                    # 272 original owners plus120 adapters, including one tied
                    # embedding/head owner. It is not the optimizer count.
                    and same(o["initial_owners"], 392),
                    "missing exact initial AC/owner witness")
            require([r["phase"] for r in o["lifecycle"]]
                    == ["loaded_base_before_injection", "after_injection_before_sharding", "before_fsdp", "after_fsdp"]
                    and all(same(r["aliases"], ALIASES) for r in o["lifecycle"]), "alias lifecycle incomplete")
            for step in (1, 2, 3):
                observed_step(c.read(prefix + f"/orthogonal-step{step}-rank{rank}.json"), variant, step)


def cpu_replay(c, comparison):
    m = c.manifest
    cpu = c.json(m["cpu_comparison"])
    c.bytes(m["comparison_program"])
    transfer, plan = c.json(m["tensor_transfer"]), c.json(m["tensor_transfer_plan"])
    require(cpu["format"] == "smollm-own-lora-ac-independent-cpu-v1"
            and cpu["status"] == "PASS_SAVED_STATE_COMPARISON_ONLY" and cpu["synthetic"] is False
            and cpu["job"] == JOB and cpu["cell"] == CELL and cpu["capture_sha256"] == CAPTURE
            and cpu["plan_sha256"] == PLAN and same(cpu["gates"], GATES)
            and cpu["program_sha256"] == m["comparison_program"]["sha256"]
            and cpu["initial_states_exact"] is True and cpu["initial_uncomputed_loss_nan_sentinel"] is True
            and cpu["all_sources_rehashed_after"] is True and same(cpu["new_gpu_executions"], 0),
            "incomplete/unbound independent CPU replay")
    require(transfer["status"] == "PASS_HASH_BOUND_TRANSFER_ONLY" and same(transfer["returncode"], 0)
            and same(transfer["files"], 24) and plan["cell"] == CELL and plan["job"] == JOB
            and plan["capture_sha256"] == CAPTURE
            and cpu["transfer_plan_sha256"] == transfer["plan_sha256"] == m["tensor_transfer_plan"]["sha256"],
            "incomplete/wrong saved witness transfer")
    expected = {name: sha for name, sha in comparison["evidence"].items() if name.endswith(".pt")}
    require(len(expected) == 32 and same(cpu["consumed_tensors"], expected)
            and set(m["tensor_files"]) == set(expected), "CPU replay omitted/substituted a saved state")
    update_names = {n for n in expected if "/update-" in n}
    transferred = {r["source"]: r for r in plan["files"]}
    require(len(plan["files"]) == len(transferred) == 24 and set(transferred) == update_names,
            "duplicate/missing transferred update")
    for name, sha in expected.items():
        ref = m["tensor_files"][name]
        require(ref["sha256"] == sha, "replayed tensor hash differs")
        raw = c.bytes(ref)
        relative_name = name[len(c.plan["root"]) + 1:]
        require(len(raw) == c.capture["files"][relative_name]["bytes"], "replayed tensor length changed")
        if name in transferred:
            row = transferred[name]
            require(row["sha256"] == sha and same(row["bytes"], len(raw)), "changed transfer plan")
    require(len(cpu["local_tensors"]) == 32
            and same(sorted(cpu["local_tensors"].values()), sorted(expected.values())), "CPU local closure incomplete")
    validate_exact_pairs(cpu["steps"])
    recorded = {(r["rank"], r["step"]): r for r in comparison["steps"]}
    for row in cpu["steps"]:
        require(row["complete_serialized_state_exact"] is True
                and same(row["errors"], recorded[row["rank"], row["step"]]["errors"]),
                "independent CPU results disagree")
    return {"initial_states_exact": True, "compared_saved_states": 32, "rank_update_pairs": 12,
            "all_update_serializations_exact": True, "comparison_sha256": m["cpu_comparison"]["sha256"],
            "scope": "Independent CPU re-read/comparison; no fresh forward/backward or global-objective reference."}


def normalize(root, cell, manifest_reference):
    require(cell["id"] == CELL and same(cell["identity"]["training_execution"], ["factorized_v1", "default_v1"]),
            "unsupported cell/profile substitution")
    c = Capture(root, manifest_reference)
    old = OriginalCapture(root, c.manifest["original_manifest"])
    full_root = old.name("full-v4-" + CELL + "-attempt01")
    full = old.read(full_root + "/inputs/cell.json")
    require(old.sha(full_root + "/inputs/cell.json") == cell["identity"]["config_sha256"]
            and digest(full["cell"]) == cell["identity"]["inventory_cell_sha256"], "wrong full recipe identity")
    partial, spec, folder = preflight(old, cell, full)
    spec_sha = plan_original(c, old, cell, spec, folder)
    comparison = execution(c)
    observed_ranks(c, cell, spec, spec_sha, comparison)
    cpu = cpu_replay(c, comparison)
    prior_ref = c.manifest["historical_blocked_receipt"]
    require(prior_ref["sha256"] == BLOCKED, "historical blocked attempt changed")
    prior = c.json(prior_ref)
    require(prior["format"] == "adapter-campaign-gate-v1" and prior["synthetic"] is False
            and same(prior["identity"], cell["identity"]) and prior["gate"] == GATE
            and prior["profile"] is None and same(prior["source"], SOURCE)
            and prior["outcome"] == "BLOCKED" and prior["executed"] is False
            and prior["documents"] == {} and prior["blocker_kind"] == "implementation",
            "wrong original unresolved gate")
    measurement = dict.fromkeys(ASSERTIONS, True)
    measurement.update(original_job="38363470", original_steps=6, supplemental_job=JOB,
                       supplemental_steps=[1, 2, 3], original_spec_sha256=spec_sha,
                       full_config_sha256=cell["identity"]["config_sha256"], supplemental_plan_sha256=PLAN,
                       source=SOURCE, target_count=60, optimizer_owner_count=120, trainable_elements=460800,
                       magnitudes=0, aliases=ALIASES, real_rows=17, zero_weight_tail_slots=7,
                       original_initial_serializations_exact=True,
                       global_valid_tokens=TOKENS, gates=GATES, cpu_state_recomparison=cpu,
                       original_allocation=partial["accounting"], supplemental_allocation=c.accounting,
                       historical_blocked_receipt=prior_ref, historical_blocked_receipt_preserved=True,
                       scope="Only own ordinary-LoRA FSDP2 installed preflight. No new full schedule, reference, midpoint, task or deployment credit.")
    c.finish()
    for name, sha in tuple(old.verified.items()):
        old.bytes({"path": name, "sha256": sha})
    refs = {**old.verified, **c.verified}
    return measurement, SOURCE, [{"path": p, "sha256": h} for p, h in sorted(refs.items())]


def produce(root, contract, manifest_reference, review_reference, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists() and contract.get("synthetic") is False,
            "fresh output under real evidence root required")
    store = Store(root)
    review = store.json(review_reference)
    require(review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == []
            and same(review["supplement_manifest"], manifest_reference), "independent producer/actual review required")
    for name in DEPENDENCIES:
        require(review["reviewed_repository_files"].get("tools/adapter_campaign/" + name)
                == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
                "unreviewed producer dependency")
    cell = next(c for c in contract["cells"] if c["id"] == CELL)
    gate = next(r for r in cell["requirements"] if r["id"] == GATE)
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(producer in gate["producers"] and SOURCE in gate["sources"]
            and review_reference in producer_reviews(gate, producer), "exact producer/source/review not admitted")
    require(all(any(same(a, {"document": "measurement", "pointer": "/" + key, "operator": "eq", "expected": True})
                    for a in gate["assertions"]) for key in ASSERTIONS), "existing preflight assertions changed")
    measurement, source, support = normalize(root, cell, manifest_reference)
    for ref in support:
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "actual review omitted consumed evidence")
    support.append(review_reference)
    directory = output / CELL / GATE
    measured = write(root, directory / "measurement.json", measurement)
    common = {"identity": cell["identity"], "source": source, "profile": None, "lineage": None,
              "final_checkpoint": None, "producer_sha256": producer, "supporting_evidence": support}
    provenance = write(root, directory / "provenance.json",
                       {**common, "documents_sha256": {"measurement": measured["sha256"]}})
    receipt = {"format": "adapter-campaign-gate-v1", "synthetic": False, **common,
               "gate": GATE, "outcome": "PASS", "executed": True,
               "documents": {"measurement": measured, "provenance": provenance}}
    reference = write(root, directory / "receipt.json", receipt)
    checked = check_receipt(Store(root), reference, gate, cell["identity"], contract, None)
    require(checked["outcome"] == "PASS" and checked["executed"] is True, "emitted receipt did not validate")
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(contract),
             "cells": {CELL: {"gates": {GATE: [reference]}}}}
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "contract", "manifest", "manifest-sha256", "review", "review-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    produce(args.root, read_json(Path(args.contract).read_bytes()),
            {"path": args.manifest, "sha256": args.manifest_sha256},
            {"path": args.review, "sha256": args.review_sha256}, args.output)
