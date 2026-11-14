# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Admit preserved Smol LoRA FSDP v1 reference and completion-only midpoint.

This one-cell producer preserves the original oracle, weights, source and failed
allocation states. It reports installed-preflight evidence as blocked until its
missing own AC/alias applicability proof is supplied by a separately reviewed
producer. No legacy deployment result or fixed-MMA inference identity is changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import stat
from pathlib import Path

from .evidence import Store, digest, producer_reviews, read_json, require, same
from .installed_preflight import SOURCE, number, source_matches
from .primary import write
from .smollm_recorded import GATES_SHA, rank_evidence, validate_cpu_continuation
from .smollm_recorded import Capture as RecordedCapture

CELL = "llama_hf__lora__factorized_v1__default_v1__fsdp2"
GATES = ("installed_cell_preflight", "independent_global_update", "mid_epoch_continuation")
DEPENDENCIES = ("smollm_lora_recorded.py", "smollm_recorded.py", "installed_preflight.py", "evidence.py", "primary.py")
ORACLE = "d47c2e3531c68124a0c69625b8526aa6189a1165ebdcecce35493b7c5680954a"
PREFLIGHT = "597cb9389c4ca4e7c0c246688000e09adc12bc7ee48fc7767952deac4b9f321f"
FULL_WORKER = "7cbc8b0282f020a00801a75693302f458925422712cddd93acf18b4f6b50d8cb"
WORKERS = {
    "independent_text_update_reference.py": ORACLE,
    "campaign_public_run.py": PREFLIGHT,
    "campaign_full_run_v1.py": FULL_WORKER,
    "campaign_recipe.py": "492ccf379af61cc652b8618f9e90978701677aa196f1afad9769640b8755acb2",
}
ORIGINAL = {
    "primary-receipt.json": "14ce49bf03c2830d9d052427324c3ce61e829376a0b6102f6f67beda9b7506c6",
    "resume-receipt.json": "38ce78bdf3d50787428c470d4f8bf78ff70e9fef7565f679e5c519344c66db16",
    "reference-receipt.json": "a153ff48afa9d0789f05c947b2294075d040a2da43b5136f187c2f35fd7a09f2",
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Capture(Store):
    """Read captured bytes and reviewed CPU comparisons; never imports a worker."""

    def __init__(self, root, reference):
        super().__init__(root)
        m = self.json(reference)
        require(m["format"] == "smollm-lora-recorded-gates-v1" and m["cell"] == CELL, "wrong one-cell capture")
        self.manifest = m
        self.remote_root = m["remote_root"]
        require(Path(self.remote_root).is_absolute() and ".." not in Path(self.remote_root).parts, "unsafe source root")
        self.records = {}
        require(len(m["captures"]) == 2, "missing original worker/source closure")
        for ref, schema in zip(
            m["captures"],
            ("smollm-lora-recorded-source-capture-v1", "smollm-lora-original-worker-closure-v1"),
            strict=True,
        ):
            envelope = self.json(ref)
            require(same(envelope["returncode"], 0) and envelope["stderr"] == "", "failed read-only capture")
            body = read_json(envelope["stdout"])
            require(
                body["format"] == schema
                and body["cell"] == CELL
                and body["synthetic"] is False
                and body["missing"] == [],
                "wrong or incomplete source capture",
            )
            for name, item in body["files"].items():
                p, s = Path(name), item["stable_stat"]
                require(p.is_relative_to(self.remote_root) and ".." not in p.parts, "foreign captured source")
                require(
                    type(item["bytes"]) is int
                    and item["bytes"] == s["st_size"] >= 0
                    and stat.S_ISREG(s["st_mode"])
                    and s["st_uid"] == 41852
                    and s["st_nlink"] == 1
                    and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                    "unowned or invalid source witness",
                )
                require(
                    name not in self.records or self.records[name]["sha256"] == item["sha256"], "conflicting captures"
                )
                if "text" in item:
                    raw = item["text"].encode()
                    require(
                        len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                        "changed raw source",
                    )
                self.records[name] = item
        discovery = self.json(m["original_discovery"])
        require(same(discovery["returncode"], 0) and discovery["stderr"] == "", "failed historical discovery")
        old = read_json(discovery["stdout"])
        require(old["cell"] == CELL, "foreign original allocation")
        self.accounting = old["accounting"]
        for name, item in old["files"].items():
            require(self.sha(name) == item["sha256"], "original captured receipt changed")
        for ref in m["programs"]:
            self.bytes(ref)
        self.cpu = self.json(m["cpu_comparison"])
        require(
            self.cpu["format"] == "smollm-lora-preserved-state-cpu-v1"
            and self.cpu["status"] == "PASS_EXISTING_TENSOR_RECOMPARISON_ONLY"
            and self.cpu["synthetic"] is False
            and self.cpu["cell"] == CELL
            and self.cpu["capture_sha256"] == m["captures"][0]["sha256"]
            and self.cpu["program_sha256"] == m["comparison_program"]["sha256"]
            and self.cpu["comparison_helper_sha256"] == m["comparison_helper"]["sha256"],
            "wrong or unbound saved-state CPU comparison",
        )
        self.bytes(m["comparison_program"])
        self.bytes(m["comparison_helper"])
        transfer, plan = self.json(m["tensor_transfer"]), self.json(m["tensor_transfer_plan"])
        require(
            transfer["status"] == "PASS_HASH_BOUND_TRANSFER_ONLY"
            and same(transfer["returncode"], 0)
            and transfer["plan_sha256"] == self.cpu["transfer_plan_sha256"] == m["tensor_transfer_plan"]["sha256"]
            and plan["capture_sha256"] == m["captures"][0]["sha256"]
            and plan["cell"] == CELL,
            "foreign or incomplete tensor transfer",
        )
        expected = {r["source"]: r["sha256"] for r in plan["files"]}
        require(len(expected) == len(plan["files"]) == transfer["files"] == 52, "incomplete/duplicate state population")
        require(same(expected, self.cpu["consumed_tensors"]), "CPU comparison omitted or substituted a tensor")
        for name, value in expected.items():
            require(self.sha(name) == value, "transferred tensor differs from captured original")

    name = RecordedCapture.name
    item = RecordedCapture.item
    raw = RecordedCapture.raw
    read = RecordedCapture.read
    sha = RecordedCapture.sha
    tensor = RecordedCapture.tensor
    job = RecordedCapture.job

    def plan(self, root):
        p = self.read(root + "/inputs/plan.json")
        require(p["root"] == root, "plan namespace changed")
        require(p.get("format") in ("full-text-cell-v1", "full-text-completion-v1"), "unadmitted execution format")
        runtime = p
        if p.get("format") == "full-text-completion-v1":
            parent = p["primary_root"]
            require(parent == self.name("full-v4-" + CELL + "-attempt01"), "wrong primary parent")
            require(self.sha(parent + "/inputs/plan.json") == p["primary_plan_sha256"], "changed parent plan")
            runtime = self.read(parent + "/inputs/plan.json")
            require(p["source_revision"] == runtime["source_revision"], "completion source changed")
        require(p["cell"] == CELL, "wrong plan cell")
        for key, target in (
            ("source_revision", "revision"),
            ("wheel_sha256", "wheel_sha256"),
            ("image_sha256", "runtime_sha256"),
        ):
            require(runtime[key] == SOURCE[target], "wrong original installed runtime")
        for name, value in p["files_sha256"].items():
            require(".." not in Path(name).parts and not Path(name).is_absolute(), "unsafe input path")
            require(self.sha(root + "/inputs/" + name) == value, "changed executed source")
            if name in WORKERS:
                require(value == WORKERS[name], "wrong original worker mathematics")
        return p


def recipe_inverse(cell, full, spec):
    require(cell["id"] == full["cell"]["id"] == spec["cell"]["id"] == CELL, "unsupported recipe binding")
    require(spec["primary_config_sha256"] == cell["identity"]["config_sha256"], "wrong full recipe")
    bounded = copy.deepcopy(spec)
    contract = bounded.pop("preflight_data_contract")
    bounded.pop("primary_config_sha256")
    require(
        contract["format"] == "cosmos-adapter-preflight-data-v1"
        and same([contract[k] for k in ("epochs", "updates", "stop_after", "checkpoint")], [3, 9, 6, 3])
        and same(contract["resume_updates"], [4, 5, 6]),
        "wrong bounded schedule",
    )
    require(
        same(
            [
                spec["cell"]["optimizer_updates"],
                spec["cell"]["training_epochs"],
                spec["cell"]["data_order"]["finite_epochs"],
            ],
            [9, 3, 3],
        ),
        "changed bounded budget",
    )
    for key in ("train", "validation"):
        require(spec["paths"][key].endswith("/data-preflight/text-control-v2/" + key + ".jsonl"), "wrong fixed inputs")
        bounded["paths"][key] = full["paths"][key]
    for key in ("optimizer_updates", "training_epochs"):
        bounded["cell"][key] = full["cell"][key]
    bounded["cell"]["data_order"]["finite_epochs"] = full["cell"]["data_order"]["finite_epochs"]
    require(same(bounded, full), "unreviewed full/preflight recipe difference")


def owners(initials, completed, spec, cell):
    targets = {f"model.layers.{i}.self_attn.{q}_proj" for i in range(30) for q in ("q", "v")}
    expected = {n + "." + leaf for n in targets for leaf in ("lora_A.weight", "lora_B.weight")}
    component = spec["cell"]["adapter_config"]["components"][0]
    require(
        component["method"] == "lora"
        and component["initialization"] == "default"
        and component["execution"] == "factorized_v1"
        and component["use_rslora"] is False,
        "wrong method/algebra",
    )
    for value, end in zip(initials, completed, strict=True):
        manifest = value["resolved"]
        names = [n for g in value["optimizer"] for n in g["names"]]
        require(
            len(names) == len(set(names)) == 120 and set(names) == expected == set(value["parameter_layouts"]),
            "wrong trainable/optimizer owners",
        )
        require(
            manifest["extra_parameters"] == []
            and same(manifest["expected_trainable_parameters"], 460800)
            and len(manifest["targets"]) == 60,
            "wrong target population",
        )
        require({t["name"] for t in manifest["targets"]} == targets, "target set changed")
        require(same(value["adapter_identity"]["resolved"], manifest), "artifact/resolved ownership differs")
        for target in manifest["targets"]:
            settings = target["settings"]
            require(
                same(settings, {k: component[k] for k in settings})
                and set(settings)
                == {
                    "method",
                    "initialization",
                    "rank",
                    "alpha",
                    "use_rslora",
                    "dropout",
                    "dtype",
                    "pissa_niter",
                    "lr_multiplier",
                    "weight_decay",
                },
                "wrong resolved settings",
            )
            require(
                target["owner"] == target["name"]
                and target["component"] == "language"
                and target["type"] == "torch.nn.Linear"
                and same(target["shape"], [576 if target["name"].endswith("q_proj") else 192, 576]),
                "wrong adaptable module",
            )
            require(
                same(settings["rank"], 8)
                and number(settings["alpha"], 16)
                and number(settings["dropout"], 0)
                and settings["dtype"] == "float32"
                and number(target["scaling"], 2),
                "wrong scaling/storage",
            )
        require(
            set(end["finite_gradients"]) == expected == set(end["changed_local"])
            and all(v is True for v in end["changed_local"].values()),
            "incomplete finite/changed owner checks",
        )
        for name, layout in value["parameter_layouts"].items():
            shape = [8, 576] if name.endswith("lora_A.weight") else [576 if ".q_proj." in name else 192, 8]
            require(
                layout["dtype"] == "torch.float32"
                and same(layout["mesh"], [0, 1, 2, 3])
                and same(layout["shard_dims"], [0])
                and same(layout["shape"], shape)
                and same(layout["local_shape"], [shape[0] // 4, shape[1]]),
                "wrong shard ownership",
            )
    require(full_identity(cell, spec), "wrong immutable checkpoint or execution")


def full_identity(cell, spec):
    return (
        spec["binding"]["repository"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        and spec["binding"]["revision"]
        == cell["identity"]["model_revision"]
        == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
        and same(cell["identity"]["training_execution"], ["factorized_v1", "default_v1"])
    )


def preflight(c, cell, full):
    root = c.name("preflight-v4-llama_hf-attempt02")
    folder = root + "/output/" + CELL
    plan, spec = c.read(root + "/inputs/plan.json"), c.read(root + "/inputs/" + CELL + ".json")
    recipe_inverse(cell, full, spec)
    spec_sha = c.sha(root + "/inputs/" + CELL + ".json")
    require(
        plan["root"] == root and plan["source_revision"] == SOURCE["revision"] and same(plan["physical_nodes"], 2),
        "foreign preflight plan",
    )
    require([r["sha256"] for r in plan["cells"] if r["id"] == CELL] == [spec_sha], "wrong plan/spec join")
    for name, value in WORKERS.items():
        if name != "campaign_full_run_v1.py":
            require(c.sha(root + "/inputs/" + name) == value, "wrong original preflight worker")
    for name, expected in ORIGINAL.items():
        require(c.sha(folder + "/" + name) == expected, "original receipt identity changed")
        row = c.read(folder + "/" + name)
        require(
            row["cell"] == CELL
            and row["phase"] == name.split("-")[0]
            and row["job"] == "38363470"
            and row["spec_sha256"] == spec_sha
            and row["image_sha256"] == SOURCE["runtime_sha256"]
            and row["passed"] is True
            and same(row["returncode"], 0),
            "wrong/incomplete own preflight phase",
        )
        for path, sha in row["evidence"].items():
            require(c.sha(path) == sha, "changed original phase evidence")
    audit = folder + "/primary/audit"
    initials = rank_evidence(c, audit, job="38363470", spec_sha=spec_sha, harness=PREFLIGHT, step=6, cell=cell)
    completed = [c.read(audit + f"/completed-updates-rank{r}.json") for r in range(4)]
    owners(initials, completed, spec, cell)
    portable = c.read(audit + "/initial-portable/adapter_manifest.json")
    require(
        same(portable["identity"], initials[0]["adapter_identity"])
        and portable["weights_sha256"] == c.sha(audit + "/initial-portable/adapter.safetensors"),
        "original warm initialization identity/weights differ",
    )
    require(
        all(same(v["initial_clock"]["optimizer_step"], 0) and same(v["initial_clock"]["accum"], 0) for v in initials),
        "not an original initial state",
    )
    return (
        {
            "status": "OWN_PREFLIGHT_PASS_ORTHOGONAL_PROOF_INCOMPLETE",
            "cell": CELL,
            "job": "38363470",
            "updates": 6,
            "owners": 120,
            "targets": 60,
            "source": SOURCE,
            "missing": ["own or explicitly applicable AC off/on comparison", "whole-model alias lifecycle proof"],
            "accounting": c.job("38363470", "TIMEOUT", 7220, 4, 2),
            "scope": "Completed own phases survive the later shared-batch timeout. The full installed-preflight gate stays BLOCKED; DoRA controls are not implicitly transferred.",
        },
        spec,
        folder,
    )


def independent_reference(c, cell, spec, folder):
    root = folder + "/independent-reference"
    result = c.read(root + "/results.json")
    source = result["source"]
    source_matches(source)
    source_matches(source["reference_provenance"])
    require(
        source["slurm_job_id"] == source["reference_provenance"]["slurm_job_id"] == "38363470"
        and source["oracle_sha256"] == ORACLE
        and source["strict_determinism"] is True,
        "wrong original v1 oracle",
    )
    require("oracle_version" not in source, "later oracle relabeled as original v1")
    require(
        source["spec_sha256"]
        == source["reference_provenance"]["campaign_spec_sha256"]
        == c.sha(c.name("preflight-v4-llama_hf-attempt02/inputs/" + CELL + ".json")),
        "wrong oracle input config",
    )
    require(
        source["reference_provenance"]["harness_sha256"] == PREFLIGHT and source["environment"] is None,
        "wrong original environment",
    )
    audit = folder + "/primary/audit"
    for name, value in source["reference_files"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts and c.sha(audit + "/" + name) == value,
            "different original reference state",
        )
    require(c.cpu["reference"]["initial_portable_exact"] is True, "initial portable does not match original owners")
    require(
        same([r["step"] for r in result["steps"]], [1, 2, 3])
        and same([r["step"] for r in c.cpu["reference"]["steps"]], [1, 2, 3]),
        "discontinuous reference",
    )
    for rank in range(4):
        c.tensor(audit + f"/initial-state-rank{rank}.pt")
    c.tensor(audit + "/initial-portable/adapter.safetensors")
    for s, r, tokens in zip(result["steps"], c.cpu["reference"]["steps"], (1286, 830, 131), strict=True):
        require(
            s["status"] == r["status"] == "PASS"
            and same(s["global_valid_tokens"], tokens)
            and same(r["global_valid_tokens"], tokens)
            and same(r["owners"], 120),
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
        require(number(r["max_gradient_error"], s["max_gradient_error"]), "saved gradient report disagrees")
        for key, limit in (
            ("max_gradient_error", 2e-6),
            ("max_cpu_parameter_error", 1e-7),
            ("max_cpu_moment_error", 1e-7),
        ):
            require(finite(r[key]) and 0 <= r[key] <= limit, "saved gradient/Adam replay exceeds gates")
        require(0 <= s["max_parameter_error"] <= 1e-7, "original parameter gate exceeded")
        c.tensor(root + f"/gradient-{s['step']:06d}.pt")
        for rank in range(4):
            c.tensor(audit + f"/update-{s['step']:06d}-rank{rank}.pt")
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
        "oracle_identity": {
            "original_filename": "independent_text_update_reference.py",
            "sha256": ORACLE,
            "schema": "original_v1_no_version_field",
        },
        "cpu_saved_gradient_replay": c.cpu["reference"],
        "accounting": c.job("38363470", "TIMEOUT", 7220, 4, 2),
        "scope": "Own original raw valid-token sum/global denominator oracle; historical completed phases in a later timed-out shared allocation. Shared allocation cost must be counted once, not once per gate.",
    }


def midpoint(c, cell, full):
    root = c.name("full-v4-" + CELL + "-attempt01")
    continuation = c.name("completion-v4-" + CELL + "-attempt01")
    plan = c.plan(continuation)
    require(plan["primary_root"] == root, "foreign full-run continuation")
    failure = plan["original_failure"]
    require(
        failure["job"] == "38375161"
        and failure["state"] == "FAILED"
        and failure["plan_sha256"] == plan["primary_plan_sha256"] == c.sha(root + "/inputs/plan.json")
        and failure["primary_process_sha256"]
        == plan["primary_process_sha256"]
        == c.sha(root + "/output/primary-process.json")
        and failure["failure_log_sha256"] == c.sha(root + "/output/main-38375161.err"),
        "original post-training failure was omitted or reinterpreted",
    )
    receipt = c.read(root + "/output/continuation-receipt.json")
    p = receipt["provenance"]
    require(
        receipt["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION"
        and p["job"] == "38450046"
        and p["cell_id"] == CELL
        and p["config_sha256"] == cell["identity"]["config_sha256"]
        and p["plan_sha256"] == c.sha(continuation + "/inputs/plan.json"),
        "wrong midpoint execution",
    )
    require(
        p["original_training"]["job"] == "38375161"
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
        job="38375161",
        spec_sha=cell["identity"]["config_sha256"],
        harness=FULL_WORKER,
        step=935,
        cell=cell,
    )
    resumed = rank_evidence(
        c,
        continuation + "/output/continuation/audit",
        job="38450046",
        spec_sha=cell["identity"]["config_sha256"],
        harness=FULL_WORKER,
        step=470,
        cell=cell,
    )
    require(same(originals[0]["resolved"], resumed[0]["resolved"]), "resume changed adapter ownership")
    for audit, rows in (
        (root + "/output/primary/audit", originals),
        (continuation + "/output/continuation/audit", resumed),
    ):
        owners(rows, [c.read(audit + f"/completed-updates-rank{r}.json") for r in range(4)], full, cell)
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
        c, c.cpu["midpoint"], root + "/output/primary/audit", continuation + "/output/continuation/audit", 467, 120
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
        "initial_factors_scope": "Ordinary LoRA has no saved PiSSA initial factors; unchanged method/base and exact restored parameters are checked.",
        "accounting": [c.job("38375161", "FAILED", 914, 4, 2), c.job("38450046", "COMPLETED", 92, 4, 2)],
        "scope": "Completed primary is preserved; separate recovery-only allocation replays 467→470. Original post-training host-audit failure retained.",
    }, {"primary_job": "38375161", "primary_root": root, "primary_plan_sha256": p["original_training"]["plan_sha256"]}


def normalize(root, cell, manifest_reference):
    require(
        cell["id"] == CELL and same(cell["identity"]["training_execution"], ["factorized_v1", "default_v1"]),
        "unsupported cell or inference substitution",
    )
    c = Capture(root, manifest_reference)
    full_root = c.name("full-v4-" + CELL + "-attempt01")
    full = c.read(full_root + "/inputs/cell.json")
    require(
        c.sha(full_root + "/inputs/cell.json") == cell["identity"]["config_sha256"]
        and digest(full["cell"]) == cell["identity"]["inventory_cell_sha256"],
        "full config identity changed",
    )
    partial, spec, folder = preflight(c, cell, full)
    reference = independent_reference(c, cell, spec, folder)
    continuation, lineage = midpoint(c, cell, full)
    return (
        {GATES[0]: partial, GATES[1]: reference, GATES[2]: continuation},
        SOURCE,
        lineage,
        [{"path": p, "sha256": h} for p, h in sorted(c.verified.items())],
    )


def produce(root, contract, manifest_reference, review_reference, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and contract.get("synthetic") is False, "foreign output or synthetic admission")
    store = Store(root)
    review = store.json(review_reference)
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
        and review["open_findings"] == []
        and same(review["recorded_capture_manifest"], manifest_reference),
        "producer/capture requires independent review",
    )
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in DEPENDENCIES:
        require(
            review["reviewed_repository_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "unreviewed source dependency",
        )
    cell = next(c for c in contract["cells"] if c["id"] == CELL)
    requirements = {r["id"]: r for r in cell["requirements"]}
    for gate in GATES:
        require(
            producer in requirements[gate]["producers"]
            and SOURCE in requirements[gate]["sources"]
            and review_reference in producer_reviews(requirements[gate], producer),
            "missing exact producer/source/review admission",
        )
        if gate != GATES[0]:
            require(
                any(
                    clause.get("document") == "measurement"
                    and clause.get("pointer") == "/gates_sha256"
                    and clause.get("operator") == "eq"
                    and clause.get("expected") == GATES_SHA
                    for clause in requirements[gate]["assertions"]
                ),
                "changed frozen numerical gates",
            )
    measurements, source, lineage, supporting = normalize(root, cell, manifest_reference)
    for ref in supporting:
        require(
            review["reviewed_files"].get(ref["path"]) == ref["sha256"], "independent review omitted consumed evidence"
        )
    supporting += [review_reference]
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {CELL: {"gates": {}}},
    }
    for gate, measurement in measurements.items():
        directory = output / CELL / gate
        receipt = {
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "identity": cell["identity"],
            "gate": gate,
            "profile": None,
            "source": source,
            "producer_sha256": producer,
            "lineage": lineage if gate == GATES[2] else None,
            "final_checkpoint": None,
            "supporting_evidence": supporting,
        }
        if gate == GATES[0]:
            # The incomplete composite gate must not count as an executed PASS.
            partial = write(root, directory / "partial-evidence.json", measurement)
            receipt.update(
                outcome="BLOCKED",
                executed=False,
                documents={},
                blocker_kind="implementation",
                reason="Own six-update installed preflight passed, but exact LoRA AC off/on and whole-model alias lifecycle applicability are unproved. Existing DoRA supplemental evidence is not transferred.",
                blocker_evidence=[partial, *supporting],
            )
        else:
            m = write(root, directory / "measurement.json", measurement)
            provenance = {
                "identity": cell["identity"],
                "source": source,
                "profile": None,
                "producer_sha256": producer,
                "lineage": receipt["lineage"],
                "final_checkpoint": None,
                "documents_sha256": {"measurement": m["sha256"]},
                "supporting_evidence": supporting,
            }
            p = write(root, directory / "provenance.json", provenance)
            receipt.update(outcome="PASS", executed=True, documents={"measurement": m, "provenance": p})
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
