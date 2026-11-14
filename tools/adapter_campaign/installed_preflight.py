# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize original installed Smol DoRA checks plus reviewed AC/tie controls.

This producer accepts the two actually executed bindings only. It rechecks the
captured JSON/source closure and the installed workers' tensor-comparison
receipts, not model execution. Missing producer admission remains a blocker.
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
from .primary import write

GATE = "installed_cell_preflight"
DEPENDENCIES = ("installed_preflight.py", "evidence.py", "primary.py")
SOURCE = {
    "revision": "5fbdf7d189a7f1be84af1d5175d8936b6e5a47e2",
    "wheel_sha256": "4907e00c809b31f74885bf185a674f9dc6dbfb29f2ff33f1a2625aa49fd6d9f5",
    "runtime_sha256": "2ec9f4d3d2a079705de15a8bc5c141fa681f84aabbf7f996f20b18e4b71d2a8e",
}
CELLS = tuple("llama_hf__dora__factorized_v1__default_v1__" + layout for layout in ("fsdp2", "hsdp"))
ALIASES = [["lm_head.weight", "model.embed_tokens.weight"]]
GATES = {
    "gradient_atol": 2e-6,
    "gradient_rtol": 2e-4,
    "loss_atol": 1e-6,
    "loss_rtol": 1e-5,
    "parameter_optimizer_atol": 1e-7,
    "parameter_optimizer_rtol": 1e-5,
}
TOKENS = {"1": 1286, "2": 830, "3": 131}
PROTOCOL_SHA = "54d2395cd259cfc52be9106dcf660fb91414b83593f8e2ca13c7c507c47cdc21"


def number(value, expected):
    return type(value) in (int, float) and math.isfinite(value) and value == expected


def accounting(row, reviewed):
    value = row["accounting"]
    require(
        type(value["returncode"]) is int and value["returncode"] == 0 and value["stderr"] == "",
        "failed scheduler capture",
    )
    lines = value["stdout"].strip().splitlines()
    require(len(lines) == 1, "ambiguous scheduler allocation")
    fields = lines[0].split("|")
    require(
        len(fields) == 7 and fields[:4] == [row["job"], "41852", "COMPLETED", "0:0"],
        "incomplete/foreign supplemental job",
    )
    resources = dict(item.split("=", 1) for item in fields[5].split(","))
    require(resources["node"] == "2" and resources["gres/gpu"] == "4", "wrong allocated physical resources")
    elapsed = int(fields[4])
    require(
        type(reviewed["elapsed_seconds"]) is int
        and reviewed["elapsed_seconds"] == elapsed > 0
        and number(reviewed["allocated_gpu_hours"], elapsed * 4 / 3600),
        "reviewed scheduler accounting differs",
    )
    return elapsed


def validate_coverage(rows, expected):
    require(len(rows) == 3 and same([r["step"] for r in rows], [1, 2, 3]), "incomplete observed update population")
    require(
        same([{k: r[k] for k in ("step", "inputs")} for r in rows], expected),
        "observed samples differ from frozen inputs",
    )
    for row in rows:
        metrics = row["metrics"]
        require(
            number(metrics["train/valid_token_count"], TOKENS[str(row["step"])])
            and number(metrics["train/valid_sequence_count"], 1 if row["step"] == 3 else 8)
            and metrics["optim/update_outcome"] == "applied"
            and number(metrics["optim/optimizer_step"], row["step"])
            and number(metrics["optim/step_skipped"], 0)
            and number(metrics["optim/nonfinite_skips_total"], 0)
            and type(metrics["train/loss_avg"]) in (int, float)
            and math.isfinite(metrics["train/loss_avg"]),
            "invalid objective/update observation",
        )


def source_matches(value):
    require(
        value["source_revision"] == SOURCE["revision"]
        and value["wheel_sha256"] == SOURCE["wheel_sha256"]
        and value["container_sha256"] == SOURCE["runtime_sha256"]
        and value["source_dirty"] is False
        and value["verified_package_files"] == value["tracked_package_files_verified"] == 1932,
        "preflight installed source bytes are not the admitted historical runtime",
    )


def validate_spec(cell, full, spec):
    require(
        cell["id"] in CELLS and full["cell"]["id"] == spec["cell"]["id"] == cell["id"], "unsupported preflight cell"
    )
    require(
        spec["primary_config_sha256"] == cell["identity"]["config_sha256"], "preflight belongs to another full recipe"
    )
    bounded = copy.deepcopy(spec)
    preflight = bounded.pop("preflight_data_contract")
    bounded.pop("primary_config_sha256")
    require(
        preflight["format"] == "cosmos-adapter-preflight-data-v1"
        and (preflight["epochs"], preflight["updates"], preflight["stop_after"], preflight["checkpoint"])
        == (3, 9, 6, 3)
        and same(preflight["resume_updates"], [4, 5, 6]),
        "wrong bounded lifecycle",
    )
    require(
        spec["cell"]["optimizer_updates"] == 9
        and spec["cell"]["training_epochs"] == spec["cell"]["data_order"]["finite_epochs"] == 3,
        "bounded schedule changed",
    )
    for key in ("train", "validation"):
        require(
            spec["paths"][key].endswith("/data-preflight/text-control-v2/" + key + ".jsonl"), "wrong preflight dataset"
        )
        bounded["paths"][key] = full["paths"][key]
    bounded["cell"]["optimizer_updates"] = full["cell"]["optimizer_updates"]
    bounded["cell"]["training_epochs"] = full["cell"]["training_epochs"]
    bounded["cell"]["data_order"]["finite_epochs"] = full["cell"]["data_order"]["finite_epochs"]
    require(same(bounded, full), "unreviewed preflight/full-recipe difference")
    require(
        full["binding"]["repository"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        and full["binding"]["revision"]
        == cell["identity"]["model_revision"]
        == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
        and same(cell["identity"]["training_execution"], ["factorized_v1", "default_v1"]),
        "wrong checkpoint or execution semantics",
    )


def validate_rank(initial, completed, provenance, public, *, cell, spec, spec_sha, job, steps, harness):
    source_matches(provenance)
    require(
        provenance["slurm_job_id"] == job
        and provenance["campaign_spec_sha256"] == spec_sha
        and provenance["harness_sha256"] == harness,
        "wrong installed worker/recipe/job",
    )
    require(
        same(public, {"final_step": steps, "preempted": "campaign diagnostic boundary"})
        and same(completed["step"], steps),
        "incomplete preflight updates",
    )
    require(
        completed["status"] == "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING" and completed["frozen_preserved"] is True,
        "missing frozen-base or update checks",
    )
    expected = {f"model.layers.{i}.self_attn.{q}_proj" for i in range(30) for q in ("q", "v")}
    names = {name + "." + leaf for name in expected for leaf in ("lora_A.weight", "lora_B.weight", "magnitude")}
    manifest = initial["resolved"]
    owners = [name for group in initial["optimizer"] for name in group["names"]]
    require(
        len(owners) == len(set(owners)) == 180 and set(owners) == names == set(initial["parameter_layouts"]),
        "trainable owners and optimizer differ",
    )
    require(
        set(completed["changed"]) == set(completed["changed_local"]) == set(completed["finite_gradients"]) == names
        and all(v is True for v in completed["changed"].values())
        and all(v is True for v in completed["changed_local"].values()),
        "missing finite or changed trainables",
    )
    require(
        manifest["extra_parameters"] == []
        and manifest["expected_trainable_parameters"] == 483840
        and len(manifest["targets"]) == 60
        and {t["name"] for t in manifest["targets"]} == expected,
        "wrong resolved component coverage",
    )
    for target in manifest["targets"]:
        settings = target["settings"]
        component = spec["cell"]["adapter_config"]["components"][0]
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
            }
            and component["execution"] == "factorized_v1"
            and target["type"] == "torch.nn.Linear"
            and same(target["shape"], [576 if target["name"].endswith("q_proj") else 192, 576]),
            "actual targets/settings differ from full recipe",
        )
        require(
            target["owner"] == target["name"]
            and target["component"] == "language"
            and settings["method"] == "dora"
            and same(settings["rank"], 8)
            and number(settings["alpha"], 16)
            and settings["use_rslora"] is False
            and settings["dtype"] == "float32"
            and number(settings["dropout"], 0)
            and number(target["scaling"], 2),
            "wrong method, magnitude, scaling or component settings",
        )
    require(
        same(initial["adapter_identity"]["resolved"], manifest)
        and same(initial["adapter_identity"]["base"], cell["identity"]["base"])
        and same(initial["adapter_identity"]["processor"], cell["identity"]["processor"]),
        "base/processor/resolved identity changed",
    )
    layout = spec["cell"]["topology"]["layout"]
    peers = initial["topology"]["peers"]
    require(
        same([p["rank"] for p in peers], list(range(4)))
        and len({p["host"] for p in peers}) == 2
        and len({p["gpu_uuid"] for p in peers}) == 4
        and same([p["local_rank"] for p in peers], [0, 1, 0, 1])
        and peers[0]["host"] == peers[1]["host"] != peers[2]["host"] == peers[3]["host"]
        and all(p["job"] == job and p["gpu_name"] == "NVIDIA A100-SXM4-80GB" for p in peers),
        "not the required two physical hosts/four A100s",
    )
    for value in initial["parameter_layouts"].values():
        require(
            value["dtype"] == "torch.float32"
            and same(value["mesh"], [[0, 1], [2, 3]] if layout == "hsdp" else [0, 1, 2, 3])
            and same(value["shard_dims"], [None, 0] if layout == "hsdp" else [0]),
            "wrong physical parameter sharding",
        )
    require(
        same(initial["initial_clock"]["optimizer_step"], 0) and same(initial["initial_clock"]["accum"], 0),
        "unexpected resume/warm clock",
    )
    return names, peers


def rank_agreement(initial, completed, layout):
    require(
        all(
            same(value["resolved"], initial[0]["resolved"])
            and same(value["adapter_identity"], initial[0]["adapter_identity"])
            and same(value["optimizer"], initial[0]["optimizer"])
            and same(value["topology"]["peers"], initial[0]["topology"]["peers"])
            for value in initial
        ),
        "rank configuration/ownership disagrees",
    )
    for rank, value in enumerate(initial):
        require(bool(value["topology"]["layouts"]), "missing actual process groups")
        for mesh in value["topology"]["layouts"].values():
            require(
                same(mesh["shape"], [2, 2] if layout == "hsdp" else [4])
                and same(mesh["coordinate"], [rank // 2, rank % 2] if layout == "hsdp" else [rank]),
                "wrong rank mesh coordinate",
            )
            members = (
                [("R", [rank % 2, rank % 2 + 2]), ("S(0)", [2 * (rank // 2), 2 * (rank // 2) + 1])]
                if layout == "hsdp"
                else [("S(0)", [0, 1, 2, 3])]
            )
            expected_groups = {
                str(axis): {
                    "placement": placement,
                    "ranks": ranks,
                    "hosts": sorted({value["topology"]["peers"][r]["host"] for r in ranks}),
                }
                for axis, (placement, ranks) in enumerate(members)
            }
            require(
                same(mesh["groups"], expected_groups),
                "process-group membership or host identities differ from the mesh and peers",
            )
    if layout == "hsdp":
        require(
            all(
                v["replicas"]["hybrid"] is True
                and v["replicas"]["trainable_shards"] == 180
                and v["replicas"]["replicas"] == 2
                for v in completed
            ),
            "missing replica comparison",
        )
        require(
            all(completed[a]["replicas"]["sha256"] == completed[b]["replicas"]["sha256"] for a, b in ((0, 2), (1, 3))),
            "hybrid replicas diverged",
        )
    else:
        require(all(same(v["replicas"], {"hybrid": False}) for v in completed), "wrong full-shard replica semantics")


class Capture(Store):
    def __init__(self, root, original_review, supplemental_review):
        super().__init__(root)
        self.old_ref = original_review
        self.old = self.json(original_review)
        require(
            self.old["format"] == "independent-installed-preflight-evidence-sufficiency-v1"
            and self.old["status"] == "REVIEW_COMPLETE_EVIDENCE_GAPS_REMAIN",
            "wrong original preflight review",
        )
        self.folder = Path(original_review["path"]).parent / "files"
        self.new = self.json(supplemental_review)
        require(
            self.new["status"] == "PASS_ACTUAL_SUPPLEMENTAL_RECEIPTS"
            and self.new["open_findings"] == []
            and self.new["full_campaign_pass"] is False
            and same(self.new["qualification_credit"], 0),
            "missing actual supplemental review",
        )
        source_review = self.json(self.new["source_review"])
        require(
            source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and source_review["open_findings"] == [],
            "unreviewed supplemental worker",
        )
        self.source_review = source_review
        for name, sha in self.old["reviewed_files"].items():
            require(
                name not in source_review["reviewed_files"] or source_review["reviewed_files"][name] == sha,
                "source reviews contradict one another",
            )
        for name, sha in {**self.old["reviewed_files"], **source_review["reviewed_files"]}.items():
            self.bytes({"path": name, "sha256": sha})
        envelope = self.json(self.new["capture"])
        require(
            type(envelope["returncode"]) is int and envelope["returncode"] == 0 and envelope["stderr"] == "",
            "terminal capture failed",
        )
        self.capture = read_json(envelope["stdout"].encode())
        require(
            self.capture["synthetic"] is False and same(self.capture["qualification_credit"], 0), "synthetic capture"
        )
        require(
            {r["layout"] for r in self.capture["rows"]} == {"fsdp2", "hsdp"} and len(self.capture["rows"]) == 2,
            "wrong supplemental population",
        )

    def original(self, name):
        path = str(self.folder / name)
        require(path in self.old["reviewed_files"], "uncaptured original preflight witness")
        return self.json({"path": path, "sha256": self.old["reviewed_files"][path]})

    @staticmethod
    def raw(row, name):
        value = row["files"][name]
        raw = value["text"].encode()
        require(
            len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"],
            "captured raw bytes changed",
        )
        return read_json(raw)


def normalize(root, contract, cell_id, *, original_review, supplemental_review):
    require(contract.get("synthetic") is False and cell_id in CELLS, "unsupported/synthetic installed preflight")
    cell = next(c for c in contract["cells"] if c["id"] == cell_id)
    layout = cell_id.rsplit("__", 1)[1]
    store = Capture(root, original_review, supplemental_review)
    full = store.json({"path": "configs/v4/" + cell_id + ".json", "sha256": cell["identity"]["config_sha256"]})
    spec = store.original("inputs/" + cell_id + ".json")
    spec_sha = store.old["reviewed_files"][str(store.folder / "inputs" / (cell_id + ".json"))]
    validate_spec(cell, full, spec)
    old_plan = store.original("inputs/plan.json")
    old_plan_sha = store.old["reviewed_files"][str(store.folder / "inputs/plan.json")]
    require([r["sha256"] for r in old_plan["cells"] if r["id"] == cell_id] == [spec_sha], "wrong original plan cell")
    old_receipt = store.original("output/" + cell_id + "/primary-receipt.json")
    require(
        old_receipt["passed"] is True
        and type(old_receipt["returncode"]) is int
        and old_receipt["returncode"] == 0
        and old_receipt["cell"] == cell_id
        and old_receipt["spec_sha256"] == spec_sha
        and old_receipt["job"] == "38363470"
        and old_receipt["image_sha256"] == SOURCE["runtime_sha256"],
        "original preflight failed or changed",
    )
    initials, completions = [], []
    for rank in range(4):
        prefix = "output/" + cell_id + "/primary/audit/"
        initial, completed, provenance, public = (
            store.original(prefix + name + f"-rank{rank}.json")
            for name in ("initial", "completed-updates", "provenance", "public-result")
        )
        validate_rank(
            initial,
            completed,
            provenance,
            public,
            cell=cell,
            spec=spec,
            spec_sha=spec_sha,
            job="38363470",
            steps=6,
            harness=store.old["reviewed_files"][str(store.folder / "inputs/campaign_public_run.py")],
        )
        require(
            provenance["mode"] == completed["mode"] == "preflight"
            and same(public, {"final_step": 6, "preempted": "campaign diagnostic boundary"}),
            "wrong preflight phase",
        )
        initials.append(initial)
        completions.append(completed)
    rank_agreement(initials, completions, layout)
    row = next(r for r in store.capture["rows"] if r["layout"] == layout)
    reviewed = next(r for r in store.new["results"] if r["cell"] == cell_id)
    require(row["root"] == reviewed["root"] and row["job"] == reviewed["job"], "wrong supplemental allocation")
    elapsed = accounting(row, reviewed)
    for name, item in row["files"].items():
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "unsafe or unhashed captured witness",
        )
        info = item["stable_stat"]
        require(
            type(item["bytes"]) is int
            and item["bytes"] == info["st_size"]
            and stat.S_ISREG(info["st_mode"])
            and info["st_uid"] == 41852
            and info["st_nlink"] == 1,
            "unowned captured source",
        )
        if "text" in item:
            raw = item["text"].encode()
            require(
                len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "capture content mismatch",
            )

    def read(name):
        return store.raw(row, name)

    plan, comparison, result = (
        read(name) for name in ("inputs/plan.json", "output/comparison.json", "output/result.json")
    )
    plan_sha = row["files"]["inputs/plan.json"]["sha256"]
    stage = store.source_review["stages"][layout]
    require(stage["plan_sha256"] == plan_sha == reviewed["plan_sha256"], "unreviewed supplemental stage")
    for name, sha in plan["files_sha256"].items():
        require(
            row["files"]["inputs/" + name]["sha256"] == sha
            and store.source_review["reviewed_files"].get(str(Path(stage["path"]) / name)) == sha,
            "unreviewed executed supplemental source",
        )
    require(
        {name[7:] for name in row["files"] if name.startswith("inputs/")} == set(plan["files_sha256"]) | {"plan.json"},
        "incomplete executed payload",
    )
    require(
        plan["original_config_sha256"] == spec_sha
        and plan["original_primary_job"] == "38363470"
        and plan["original_plan_sha256"] == old_plan_sha
        and plan["cell"] == cell_id
        and same(plan["original_adapter_identity"], initials[0]["adapter_identity"])
        and same(read("inputs/cell.json"), spec),
        "supplemental/original binding differs",
    )
    require(
        plan["source_revision"] == SOURCE["revision"]
        and plan["wheel_sha256"] == SOURCE["wheel_sha256"]
        and plan["image_sha256"] == SOURCE["runtime_sha256"]
        and plan["root"] == row["root"]
        and plan["layout"] == layout
        and same(plan["steps"], [1, 2, 3])
        and same(plan["variants"], ["off", "on"])
        and same(plan["gates"], GATES)
        and same(plan["global_valid_tokens"], TOKENS)
        and plan["protocol_sha256"] == row["files"]["inputs/protocol.json"]["sha256"] == PROTOCOL_SHA,
        "changed installed runtime or frozen supplemental protocol",
    )
    intent, ack = read("intent.json"), read("submission.json")
    require(
        intent["plan_sha256"] == ack["plan_sha256"] == plan_sha
        and ack["job_id"] == row["job"]
        and type(ack["returncode"]) is int
        and ack["returncode"] == 0
        and ack["stdout"] == row["job"] + "\n"
        and ack["stderr"] == ""
        and intent["review"]["sha256"] == store.new["source_review"]["sha256"]
        and intent["review"]["path"] == str(Path(root) / store.new["source_review"]["path"])
        and same(intent["command"], ack["command"])
        and intent["command"][-1] == row["root"] + "/inputs/job.sbatch",
        "wrong completed job journal",
    )
    require(
        result["job"] == row["job"]
        and result["cell"] == cell_id
        and result["plan_sha256"] == plan_sha
        and result["passed"] is True
        and result["source_immutable"] is True
        and result["full_campaign_pass"] is False
        and same(result["qualification_credit"], 0),
        "incomplete supplemental controller",
    )
    require(
        result["comparison_sha256"]
        == reviewed["comparison_sha256"]
        == row["files"]["output/comparison.json"]["sha256"],
        "wrong comparison receipt",
    )
    require(
        [p["phase"] for p in result["phases"]] == ["off", "on", "comparison"]
        and all(
            type(p["returncode"]) is int
            and p["returncode"] == 0
            and p["passed"] is True
            and p["job"] == row["job"]
            and p["plan_sha256"] == plan_sha
            for p in result["phases"]
        ),
        "missing AC/comparison execution",
    )
    require(
        comparison["status"] == "PASS_SUPPLEMENTAL_ALIAS_AND_MATCHED_AC"
        and comparison["plan_sha256"] == plan_sha
        and comparison["cell"] == cell_id
        and comparison["native_replay_observed"] is True
        and comparison["full_training_credit"] is False
        and comparison["all_cells_qualified"] is False
        and comparison["source_revision"] == SOURCE["revision"]
        and same(comparison["whole_model_aliases"], ALIASES)
        and same(comparison["real_rows"], 17)
        and same(comparison["zero_weight_tail_slots"], 7)
        and same(comparison["global_valid_tokens"], TOKENS)
        and same(comparison["gates"], plan["gates"]),
        "incomplete supplemental evidence",
    )
    require(
        len(comparison["steps"]) == 12
        and all(type(r["rank"]) is int and type(r["step"]) is int for r in comparison["steps"])
        and {(r["rank"], r["step"]) for r in comparison["steps"]} == {(r, s) for r in range(4) for s in (1, 2, 3)},
        "incomplete AC update comparisons",
    )
    require(
        all(
            r["status"] == "PASS"
            and set(r["errors"]) == {"parameters", "optimizer", "gradients", "loss"}
            and all(type(v) is float and math.isfinite(v) and v == 0 for v in r["errors"].values())
            for r in comparison["steps"]
        ),
        "recorded exact AC comparisons disagree",
    )
    for remote, sha in comparison["evidence"].items():
        prefix = row["root"] + "/"
        require(
            remote.startswith(prefix) and row["files"][remote[len(prefix) :]]["sha256"] == sha,
            "missing comparison witness",
        )
    witness_names = ["initial", "completed-updates", "orthogonal-initial", "provenance", "public-result"]
    witnesses = {f"{name}-rank{r}.json" for name in witness_names for r in range(4)}
    witnesses |= {f"orthogonal-step{s}-rank{r}.json" for s in (1, 2, 3) for r in range(4)}
    witnesses |= {f"initial-state-rank{r}.pt" for r in range(4)}
    witnesses |= {f"update-{s:06d}-rank{r}.pt" for s in (1, 2, 3) for r in range(4)}
    require(
        set(comparison["evidence"])
        == {row["root"] + f"/output/{v}/audit/" + n for v in ("off", "on") for n in witnesses},
        "incomplete actual tensor-comparison witnesses",
    )
    for variant in ("off", "on"):
        initial_rows, final_rows = [], []
        for rank in range(4):
            prefix = "output/" + variant + "/audit/"
            coverage = [
                read_json(line) for line in row["files"][prefix + f"coverage-rank{rank}.jsonl"]["text"].splitlines()
            ]
            validate_coverage(coverage, plan["expected_inputs"][str(rank)])
            initial, completed, provenance, public = (
                read(prefix + name + f"-rank{rank}.json")
                for name in ("initial", "completed-updates", "provenance", "public-result")
            )
            names, peers = validate_rank(
                initial,
                completed,
                provenance,
                public,
                cell=cell,
                spec=spec,
                spec_sha=spec_sha,
                job=row["job"],
                steps=3,
                harness=plan["files_sha256"]["smollm_preflight_orthogonal_v1.py"],
            )
            require(
                same(peers, comparison["topology"]) and same(peers, reviewed["peers"]), "reviewed rank mapping differs"
            )
            require(
                provenance["mode"] == "supplemental_alias_ac"
                and provenance["variant"] == variant
                and provenance["public_grad_ckpt"] is (variant == "on")
                and provenance["orthogonal_plan_sha256"] == plan_sha,
                "wrong public checkpointing mode",
            )
            require(
                same(initial["adapter_identity"], initials[rank]["adapter_identity"]), "supplemental adapter differs"
            )
            orthogonal = read(prefix + f"orthogonal-initial-rank{rank}.json")
            require(
                orthogonal["plan_sha256"] == plan_sha
                and orthogonal["public_grad_ckpt"] is (variant == "on")
                and orthogonal["layer_count"] == 30
                and len(orthogonal["optimizer_owners"]) == 180
                and set(orthogonal["optimizer_owners"]) == names,
                "incomplete AC initial ownership",
            )
            lifecycle = orthogonal["lifecycle"]
            require(
                [r["phase"] for r in lifecycle]
                == ["loaded_base_before_injection", "after_injection_before_sharding", "before_fsdp", "after_fsdp"]
                and all(same(r["aliases"], ALIASES) for r in lifecycle),
                "missing whole-model alias lifecycle",
            )
            for step in (1, 2, 3):
                receipt = read(prefix + f"orthogonal-step{step}-rank{rank}.json")
                require(
                    receipt["cell"] == cell_id
                    and receipt["plan_sha256"] == plan_sha
                    and same(receipt["step"], step)
                    and receipt["variant"] == variant
                    and receipt["whole_model_parameter_objects_unchanged"] is True
                    and receipt["optimizer_owners"] == 180
                    and same(receipt["aliases"], ALIASES),
                    "changed updated ownership",
                )
                expected = {(f"model.layers.{i}", m) for i in range(30) for m in range(2)}
                entries = receipt["layer_entries"]
                require(
                    len(entries) == 60
                    and {(r["layer"], r["micro"]) for r in entries} == expected
                    and all(type(r["micro"]) is int and type(r["entries"]) is int for r in entries)
                    and all(r["entries"] >= 2 if variant == "on" else r["entries"] == 1 for r in entries),
                    "public AC replay not observed",
                )
                records = receipt["records"]
                require(len(records) == (60 if variant == "on" else 0), "wrong replay cardinality")
                if variant == "on":
                    require(
                        {(r["layer"], r["micro"]) for r in records} == expected
                        and all(
                            same(r["step"], step)
                            and type(r["entries"]) is int
                            and type(r["replays"]) is int
                            and r["entries"] >= 2
                            and r["replays"] >= 1
                            for r in records
                        ),
                        "missing non-reentrant replay",
                    )
                require(
                    number(receipt["metrics"]["train/valid_token_count"], TOKENS[str(step)]),
                    "changed native denominator",
                )
            initial_rows.append(initial)
            final_rows.append(completed)
        rank_agreement(initial_rows, final_rows, layout)
    measurement = {
        key: True
        for key in (
            "installed_bytes_verified",
            "two_physical_hosts",
            "topology_verified",
            "all_targets_resolved",
            "owners_equal_optimizer",
            "aliases_preserved",
            "all_methods_verified",
            "frozen_preserved",
            "finite_updates",
            "all_whole_trainables_changed",
            "rank_agreement",
            "activation_checkpointing_checked",
        )
    }
    measurement.update(
        original_job="38363470",
        supplemental_job=row["job"],
        original_steps=6,
        supplemental_steps=[1, 2, 3],
        original_spec_sha256=spec_sha,
        full_config_sha256=cell["identity"]["config_sha256"],
        supplemental_plan_sha256=plan_sha,
        supplemental_elapsed_seconds=elapsed,
        retained_failed_supplemental_job=plan["retry"]["previous_job"],
        aliases=ALIASES,
        proof_limit="Captured original assertions and independently reviewed supplemental tensor comparisons; this normalizer rehashes captured text and verifies tensor hash/receipt closure, not a fresh tensor numerical replay.",
        scope="Only installed preflight for the exact Smol DoRA cell. No full-data, resume, deployment, task or other-method credit.",
    )
    return measurement, SOURCE, [{"path": name, "sha256": sha} for name, sha in store.verified.items()]


def produce(root, contract, output, *, cell_id, original_review, supplemental_review, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "fresh output under evidence root required")
    store = Store(root)
    review = store.json(producer_review)
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == [], "producer not reviewed"
    )
    for name in DEPENDENCIES:
        require(
            review["reviewed_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "producer dependency changed",
        )
    for ref in (original_review, supplemental_review):
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "actual capture review not admitted")
    cell = next(c for c in contract["cells"] if c["id"] == cell_id)
    gate = next(g for g in cell["requirements"] if g["id"] == GATE)
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(
        producer in gate["producers"]
        and producer_review in producer_reviews(gate, producer)
        and SOURCE in gate["sources"],
        "installed-preflight producer/source needs independent admission",
    )
    measurement, source, support = normalize(
        root, contract, cell_id, original_review=original_review, supplemental_review=supplemental_review
    )
    support.append(producer_review)
    folder = output / cell_id / GATE
    measured = write(root, folder / "measurement.json", measurement)
    common = {
        "identity": cell["identity"],
        "source": source,
        "profile": None,
        "final_checkpoint": None,
        "producer_sha256": producer,
        "supporting_evidence": support,
    }
    provenance = write(
        root, folder / "provenance.json", {**common, "documents_sha256": {"measurement": measured["sha256"]}}
    )
    receipt = write(
        root,
        folder / "receipt.json",
        {
            **common,
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "gate": GATE,
            "outcome": "PASS",
            "executed": True,
            "documents": {"measurement": measured, "provenance": provenance},
        },
    )
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {cell_id: {"gates": {GATE: [receipt]}}},
    }
    write(root, output / "index.json", index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    for name in ("original-review", "supplemental-review", "producer-review"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    refs = {
        name: {"path": getattr(args, name), "sha256": getattr(args, name + "_sha256")}
        for name in ("original_review", "supplemental_review", "producer_review")
    }
    produce(args.root, read_json(args.contract.read_bytes()), args.output, cell_id=args.cell, **refs)


if __name__ == "__main__":
    main()
