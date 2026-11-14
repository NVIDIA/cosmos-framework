# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compile the sealed all-family inventory into explicit completion requirements.

This is a contract compiler, not a result collector. It never changes a gate in
response to observed results. Producer/source admissions are reviewed separately;
an empty admission is deliberately unexecutable, not an architectural exclusion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from .evidence import Store, digest, producer_reviews, read_json, require, same

REQUIRED_DEPLOYMENTS = {"REQUIRED", "REQUIRED_LEGACY_APPROXIMATION_CHECK", "REQUIRED_PRECISION_MATCHED_MERGE_CHECK"}
EXCLUDED_DEPLOYMENTS = {"EXPLICITLY_REJECTED", "LEGACY_REJECTED_NOT_A_PASS", "UNADMITTED_PROFILE_COMBINATION"}
PHYSICAL_DATASETS = {"libero_pi05", "libero_gr00t", "physicalai_alpamayo"}
IDENTITY_PROTOCOL_PREFIX = "pretraining_identity:"


def reference(root, name):
    path = Path(root) / name
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def assertion(field, expected, operator="eq"):
    return {"document": "measurement", "pointer": "/" + field, "operator": operator, "expected": expected}


def apply_pretraining_identity(store, cells, *, name, reference):
    """Resolve one previously empty identity from a separately reviewed proof.

    The inventory/config and all gates remain sealed. This is not an asset-seal
    receipt, nor permission to reinterpret evidence under a different identity.
    The caller still needs independent review of the complete compiled digest.
    """
    cell_id = name.removeprefix(IDENTITY_PROTOCOL_PREFIX)
    require(name.startswith(IDENTITY_PROTOCOL_PREFIX) and cell_id, "invalid identity amendment name")
    amendment = store.json(reference)
    require(
        set(amendment) == {"format", "cell", "prior_contract", "prior_index", "resolution", "producer_review"}
        and amendment["format"] == "adapter-campaign-pretraining-identity-amendment-v1"
        and amendment["cell"] == cell_id,
        "invalid cell-scoped identity amendment",
    )
    prior = store.json(amendment["prior_contract"])
    index = store.json(amendment["prior_index"])
    proof = store.json(amendment["resolution"])
    review = store.json(amendment["producer_review"])
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == [],
        "identity producer has unresolved review findings",
    )
    require(
        review["reviewed_files"].get(amendment["resolution"]["path"]) == amendment["resolution"]["sha256"],
        "identity resolution has no exact independent review",
    )
    require(
        prior["format"] == "adapter-campaign-completion-contract-v1"
        and prior["required_cells"] == len(prior["cells"]) == len(cells) == 368
        and len({v["id"].split("__")[0] for v in prior["cells"]}) == 16
        and len({v["id"] for v in cells}) == len({v["id"] for v in prior["cells"]}) == 368
        and {v["id"] for v in cells} == {v["id"] for v in prior["cells"]},
        "identity amendment changed the sealed scope",
    )
    require(
        index["format"] == "adapter-campaign-evidence-index-v1"
        and index["contract_sha256"] == digest(prior)
        and (cell_id not in index["cells"] or index["cells"][cell_id].get("gates") == {}),
        "identity amendment has stale history or existing target gates",
    )
    selected = [v for v in cells if v["id"] == cell_id]
    previous = [v for v in prior["cells"] if v["id"] == cell_id]
    require(len(selected) == len(previous) == 1, "unknown identity amendment cell")
    cell = selected[0]
    require(
        same(cell["identity"], previous[0]["identity"])
        and cell["identity"]["base"] is None
        and cell["identity"]["processor"] is None,
        "identity amendment may only resolve the original two null fields",
    )
    require(
        proof["format"] == "hf-segmented-pretraining-identity-resolution-v2"
        and proof["cell"] == cell_id
        and same(proof["prior_identity"], cell["identity"])
        and proof["prior_contract_sha256"] == digest(prior)
        and proof["prior_index_sha256"] == digest(index)
        and same(proof["before"], {"base": None, "processor": None})
        and set(proof["after"]) == {"base", "processor"}
        and proof["qualification_pass"] is False
        and proof["active_index_written"] is False
        and proof["gate_evidence_emitted"] is False,
        "unbound or post-execution identity resolution",
    )
    origin = Path(proof["capture_origin"])
    require(origin.is_absolute() and ".." not in origin.parts, "unsafe identity capture origin")
    audit_ref = proof["actual_audit"]
    require(
        review["reviewed_files"].get(audit_ref["path"]) == audit_ref["sha256"]
        and review.get("capture_origins", {}).get(audit_ref["sha256"]) == str(origin),
        "identity audit or capture relocation has no exact independent review",
    )
    audit = store.json(audit_ref)
    require(
        audit["cell"] == cell_id
        and audit["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
        and audit["open_findings"] == [],
        "identity proof belongs to another or unresolved actual audit",
    )
    sources = proof["producer_files_sha256"]
    require(
        set(sources)
        == {
            "tools/adapter_campaign/" + source
            for source in ("hf_segmented.py", "segmented.py", "recovered.py", "evidence.py", "primary.py")
        },
        "incomplete identity producer source closure",
    )
    for path, expected in sources.items():
        require(
            isinstance(expected, str)
            and re.fullmatch("[0-9a-f]{64}", expected)
            and review["reviewed_files"].get(path) == expected,
            "identity producer source was not independently reviewed",
        )
    support = proof["supporting_evidence"]
    require(
        isinstance(support, list)
        and support
        and len({r["path"] for r in support}) == len(support)
        and audit_ref in support,
        "incomplete identity evidence closure",
    )
    for ref in support:
        store.bytes(ref)
    for asset in proof["after"].values():
        require(
            set(asset) == {"sha256", "files"}
            and isinstance(asset["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", asset["sha256"])
            and isinstance(asset["files"], dict)
            and asset["files"],
            "incomplete immutable identity inventory",
        )
        for path, expected in asset["files"].items():
            require(
                isinstance(path, str)
                and Path(path).parts
                and not Path(path).is_absolute()
                and ".." not in Path(path).parts
                and isinstance(expected, str)
                and re.fullmatch("[0-9a-f]{64}", expected),
                "unsafe or unsealed immutable identity member",
            )
    proposal = copy.deepcopy(prior)
    next(v for v in proposal["cells"] if v["id"] == cell_id)["identity"].update(proof["after"])
    require(proof["candidate_contract_sha256"] == digest(proposal), "identity proposal changed other fields")
    base_gates = [v for v in cell["requirements"] if v["id"] == "matched_base_evaluation"]
    asset_gates = [v for v in cell["requirements"] if v["id"] == "immutable_asset_seal"]
    require(len(base_gates) == len(asset_gates) == 1, "identity amendment cannot remove a required gate")
    base_assertions = [v for v in base_gates[0]["assertions"] if v["pointer"] == "/immutable_base_sha256"]
    require(
        len(base_assertions) == 1 and same(base_assertions[0], assertion("immutable_base_sha256", digest(None))),
        "identity amendment cannot change matched-base semantics",
    )
    # All validation precedes the allowed identity/assertion/note changes to
    # this one cell. The required asset gate and its admissions remain intact.
    cell["identity"].update(copy.deepcopy(proof["after"]))
    base_assertions[0]["expected"] = digest(proof["after"]["base"])
    asset_gates[0].pop("pending_admission", None)
    cell["identity_resolution"] = {
        "protocol": reference,
        "resolution": amendment["resolution"],
        "note": "Pretraining identities resolved; the required own-cell immutable_asset_seal receipt remains unawarded.",
    }


def closed_loop_contract(store, protocols, dataset, data):
    """A data preparation contract is never a simulator protocol.

    Missing runtime/suite sealing leaves the required loader-specific check
    unadmitted. No count or seed list is invented by this compiler.
    """
    ref = protocols.get("closed_loop:" + dataset)
    if ref is None:
        return None
    protocol = store.json(ref)
    require(protocol.get("format") == "adapter-campaign-closed-loop-protocol-v1", "wrong simulator protocol")
    require(protocol.get("dataset_sha256") == digest(data), "simulator protocol dataset mismatch")
    trials = protocol.get("trials")
    require(isinstance(trials, list) and trials, "simulator protocol has no frozen trials")
    ids = []
    for trial in trials:
        require(isinstance(trial, dict), "invalid simulator trial")
        require(
            set(trial) == {"id", "task_or_scene", "seed", "initial_state_index", "initial_state_sha256"},
            "incomplete simulator trial",
        )
        require(isinstance(trial["id"], str) and trial["id"], "invalid trial ID")
        require(
            type(trial["seed"]) is int and isinstance(trial["task_or_scene"], str) and bool(trial["task_or_scene"]),
            "invalid simulator seed/task",
        )
        require(type(trial["initial_state_index"]) is int, "invalid simulator initial-state index")
        require(
            isinstance(trial["initial_state_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", trial["initial_state_sha256"]),
            "unsealed simulator initial state",
        )
        ids.append(trial["id"])
    require(len(set(ids)) == len(ids), "duplicate simulator trial")
    coordinates = {(t["task_or_scene"], t["initial_state_index"], t["seed"]) for t in trials}
    require(len(coordinates) == len(trials), "duplicated simulator task/state/seed")
    if dataset.startswith("libero_"):
        # These are the already frozen inventory's 10 tasks/suite, first50
        # initial states, seed42. A later protocol cannot shrink that census.
        require(
            "50 fixed initial states" in data["closed_loop"] and "seed42" in data["closed_loop"].replace(" ", ""),
            "unreviewed LIBERO schedule",
        )
        expected = {
            (f"{suite}:{task}", state, 42)
            for suite in data["evaluation_suite"]
            for task in range(10)
            for state in range(50)
        }
    else:
        preparation = store.json(protocols[dataset])
        split = preparation["splits"]["alpasim"]
        scene_ids = store.json(
            {"path": str(Path(protocols[dataset]["path"]).parent / split["filename"]), "sha256": split["ids_sha256"]}
        )
        require(
            len(scene_ids) == len(set(scene_ids)) == data["closed_loop_scenarios"],
            "wrong complete driving scene census",
        )
        require(type(protocol.get("seed")) is int, "driving simulator seed must be frozen")
        expected = {(scene, 0, protocol["seed"]) for scene in scene_ids}
    require(coordinates == expected, "simulator protocol does not cover the frozen full suite/state/seed census")
    for key in ("suite", "runtime", "initial_states", "native_action_contract", "metrics_and_gates", "pre_output_seal"):
        require(protocol.get(key), "simulator protocol missing " + key)
        store.bytes(protocol[key])
    states = store.json(protocol["initial_states"])
    require(
        same(
            states,
            {f"{t['task_or_scene']}/{t['initial_state_index']}/{t['seed']}": t["initial_state_sha256"] for t in trials},
        ),
        "simulator initial-state bytes differ from sealed census",
    )
    return ref, {"count": len(ids), "sorted_ids_sha256": digest(sorted(ids))}


def compile_contract(root, *, inventory, deployment, gates, admissions, protocols):
    """All input references must be independently reviewed with the output digest.

    The compiler emits explicit required rows for absent plumbing, unreviewed
    producers, missing data and failed legacy profiles. Only a rejection already
    recorded by the deployment contract can exclude an additional profile.
    """
    store = Store(root)
    manifest = store.json(inventory)
    table = store.json(deployment)
    limits = store.json(gates)
    admission = store.json(admissions)
    require(admission.get("format") == "adapter-campaign-producer-admissions-v1", "wrong producer admissions")
    require(table["campaign_manifest_sha256"] == inventory["sha256"], "deployment belongs to another inventory")
    require(table["gates_sha256"] == gates["sha256"], "deployment belongs to another gate contract")
    # v3 -> v4 changed strict determinism, not numerical or task gates. Bind both
    # inventories and validate the unchanged fields instead of ignoring a hash.
    if limits["inventory_sha256"] != inventory["sha256"]:
        old_ref = protocols["gate_inventory"]
        require(old_ref["sha256"] == limits["inventory_sha256"], "missing original numerical-gate inventory")
        old = store.json(old_ref)
        for key in ("datasets", "bindings", "gates", "total_required_cells"):
            require(same(old[key], manifest[key]), f"unreviewed gate inventory change: {key}")
        previous = {c["id"]: c for c in old["cells"]}
        require(set(previous) == {c["id"] for c in manifest["cells"]}, "amendment changed primary cells")
        for cell in manifest["cells"]:
            changed = {
                k for k in set(cell) | set(previous[cell["id"]]) if not same(cell.get(k), previous[cell["id"]].get(k))
            }
            require(changed <= {"deterministic_strict"}, f"unreviewed cell amendment: {cell['id']}: {changed}")
            require(cell["deterministic_strict"] is True, "strict deterministic amendment is absent")
    for ref in protocols.values():
        store.bytes(ref)
    bindings = {b["id"]: b for b in manifest["bindings"]}
    deployed = {c["cell_id"]: c for c in table["cells"]}
    require(len(deployed) == len(table["cells"]) == manifest["total_required_cells"], "incomplete deployment table")
    require(set(deployed) == {c["id"] for c in manifest["cells"]}, "deployment inventory mismatch")
    cells = []
    config_refs = []
    for cell in manifest["cells"]:
        cell_id, binding = cell["id"], bindings[cell["binding"]]
        data = manifest["datasets"][cell["data_contract"]]
        config_ref = reference(root, f"configs/v4/{cell_id}.json")
        config = store.json(config_ref)
        require(same(config["cell"], cell), "resolved config differs from inventory")
        # The frozen runnable configs deliberately contain a compact binding
        # and data view; identity below still binds the entire sealed records.
        binding_keys = {
            "architecture",
            "dataset",
            "environment",
            "family",
            "id",
            "implementation",
            "native_model_contract",
            "objective",
            "repository",
            "revision",
        }
        if "snapshot_sha256" in binding:
            binding_keys.add("snapshot_sha256")
        data_keys = {"rows_train", "rows_validation", "epochs", "global_batch", "updates"}
        data_keys.update(
            {
                "wts": {"max_length"},
                "gsm8k": {"max_length"},
                "libero_pi05": {"horizon"},
                "libero_gr00t": {"horizon"},
                "physicalai_alpamayo": {"history_steps", "future_steps", "dt_seconds"},
            }[cell["data_contract"]]
        )
        require(same(config["binding"], {k: binding[k] for k in binding_keys}), "config changed binding view")
        require(same(config["dataset"], {k: data[k] for k in data_keys}), "config changed data view")
        config_refs.append(config_ref)
        identity = {
            "cell_id": cell_id,
            "inventory_cell_sha256": digest(cell),
            "config_sha256": config_ref["sha256"],
            "binding_sha256": digest(binding),
            "dataset_sha256": digest(data),
            "base": binding["base_identity"],
            "processor": binding["processor_identity"],
            "media_sha256": digest(binding.get("media_identity")),
            "native_contract_sha256": digest(binding.get("native_model_contract")),
            "model_revision": binding.get("revision") or binding.get("snapshot_sha256"),
            "training_execution": [cell["component_execution"], cell["model_execution"]],
            "topology": cell["topology"],
        }
        requirements, exclusions = [], []

        def add(name, assertions, *, final=False, profile=None, anchor=False, pending=None):
            category = name.split(":", 1)[0]
            allowed = admission["gates"].get(category, {"producers": [], "sources": []})
            require(
                {"producers", "sources"} <= set(allowed) <= {"producers", "sources", "producer_reviews"},
                "unsupported admission fields",
            )
            requirement = {
                "id": name,
                "admission": "REQUIRED",
                "profile": profile,
                "final_checkpoint": final,
                "documents": ["measurement", "provenance"],
                "provenance_document": "provenance",
                "raw_evidence_required": True,
                "training_lineage": category
                not in {
                    "installed_cell_preflight",
                    "independent_global_update",
                    "controlled_failure_recovery",
                    "matched_base_evaluation",
                },
                "producers": allowed["producers"],
                "sources": allowed["sources"],
                "assertions": assertions,
            }
            if anchor:
                requirement["checkpoint_anchor"] = {"document": "measurement", "pointer": "/checkpoint"}
            if pending:
                requirement["pending_admission"] = pending
            if "producer_reviews" in allowed:
                requirement["producer_reviews"] = allowed["producer_reviews"]
                require(allowed["producers"], "review admission has no producer")
                for producer in allowed["producers"]:
                    for ref in producer_reviews(requirement, producer):
                        store.bytes(ref)
            requirements.append(requirement)

        missing_assets = [key for key in ("base_identity", "processor_identity") if not binding.get(key)]
        if missing_assets:
            add(
                "immutable_asset_seal",
                [],
                pending="Missing immutable "
                + ", ".join(missing_assets)
                + "; requires reviewed binding-identity amendment before execution admission.",
            )

        add(
            "installed_cell_preflight",
            [
                assertion(k, True)
                for k in (
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
            ],
        )
        add(
            "independent_global_update",
            [
                assertion(k, True)
                for k in (
                    "same_global_inputs",
                    "independent_objective_weighting",
                    "uninterrupted_multi_update_reference",
                    "tail_and_uneven_lengths",
                    "raw_gradients_within_gates",
                    "optimizer_within_gates",
                    "native_rng_replayed",
                )
            ]
            + [assertion("gates_sha256", gates["sha256"])],
        )
        add(
            "controlled_failure_recovery",
            [
                assertion(k, True)
                for k in (
                    "worker_loss",
                    "job_workers_one_node_loss",
                    "global_teardown",
                    "partial_checkpoint_rejected",
                    "committed_boundary_recovered",
                    "mixed_native_parameters_covered",
                    "lifecycle_and_topology_match",
                )
            ]
            + [assertion("protocol_sha256", protocols["controlled_failure"]["sha256"])],
        )
        add(
            "matched_base_evaluation",
            [
                assertion("complete_fixed_task_split", True),
                assertion("native_unadapted_checkpoint", True),
                assertion("matched_preprocessing_precision_decoding", True),
                assertion("all_invalid_failures_in_denominator", True),
                assertion("immutable_base_sha256", digest(binding["base_identity"])),
            ],
        )
        add(
            "full_schedule",
            [
                assertion("logical_updates", cell["optimizer_updates"]),
                assertion("epochs", cell["training_epochs"]),
                assertion("finite_applied_updates", True),
                assertion("frozen_preserved", True),
                assertion("two_physical_hosts", True),
            ],
        )
        add(
            "complete_sample_coverage",
            [
                assertion("real_rows_per_epoch", [data["rows_train"]] * cell["training_epochs"]),
                assertion("zero_weight_padding_per_epoch", [data["padding_slots_per_epoch"]] * cell["training_epochs"]),
                assertion("each_expected_id_once_per_epoch", True),
                assertion("denominator_census_matches", True),
                assertion("recovery_replays_excluded", True),
                assertion("data_contract_sha256", digest(data)),
            ],
        )
        add(
            "initial_final_validation",
            [
                assertion("steps", [0, cell["optimizer_updates"]]),
                assertion("counts", [data["rows_validation"]] * 2),
                assertion("complete_id_census", True),
                assertion("native_denominators_verified", True),
                assertion("finite_losses", True),
            ],
        )
        add(
            "mid_epoch_continuation",
            [
                assertion("mid_update", cell["checkpoint_contract"]["mid_epoch_update"]),
                assertion("updates_compared", cell["checkpoint_contract"]["continuation_updates"]),
                *[
                    assertion(k, True)
                    for k in (
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
                    )
                ],
                assertion("gates_sha256", gates["sha256"]),
            ],
        )
        add(
            "final_checkpoint",
            [
                assertion(
                    "checkpoint",
                    {"components": cell["checkpoint_contract"]["components"], "update": cell["optimizer_updates"]},
                    "complete_checkpoint",
                ),
                assertion("consolidated_and_dcp_verified", True),
            ],
            final=True,
            anchor=True,
        )
        add(
            "final_portable_warm_reload",
            [
                assertion(k, True)
                for k in (
                    "strict_reload",
                    "base_processor_identity",
                    "initial_factors_extras_ties",
                    "source_immutable",
                    "warm_start_reset",
                    "warm_updates_and_ownership",
                    "warm_two_node_topology",
                    "incomplete_and_wrong_base_rejected",
                )
            ],
            final=True,
        )
        add(
            "final_merged_export",
            [
                assertion(k, True)
                for k in (
                    "stock_or_native_supported_loader",
                    "complete_tensors_and_assets",
                    "precision_identity",
                    "live_and_source_immutable",
                    "idempotent_reuse",
                    "interruption_recovery",
                    "peaks_measured",
                )
            ],
            final=True,
        )
        if not any(
            p["path"].startswith("merged") and p["admission"] in REQUIRED_DEPLOYMENTS
            for p in deployed[cell_id]["deployment_paths"]
        ):
            requirements[-1]["known_limitation"] = (
                "The frozen table admits no merged path for this legacy cell. Its generic required merged-export "
                "criterion remains unresolved; the explicit legacy rejection is not a successful export. "
                "Preserve the negative outcome without repeating an unchanged experiment."
            )
        add(
            "final_resources",
            [
                assertion(k, True)
                for k in (
                    "all_allocated_gpus_counted",
                    "startup_and_warm_separate",
                    "replays_failures_accounted",
                    "per_rank_node_peaks",
                    "training_export_evaluation_costs",
                    "native_throughput_denominator",
                )
            ],
            final=True,
        )
        closed_loop = (
            closed_loop_contract(store, protocols, cell["data_contract"], data)
            if cell["data_contract"] in PHYSICAL_DATASETS
            else None
        )
        for path in deployed[cell_id]["deployment_paths"]:
            require(path["admission"] in REQUIRED_DEPLOYMENTS | EXCLUDED_DEPLOYMENTS, "unknown deployment admission")
            if path["admission"] in EXCLUDED_DEPLOYMENTS:
                require(path.get("reason"), "unsupported profile requires architectural/format reason")
                exclusions.append(
                    {
                        "admission": "EXPLICITLY_UNSUPPORTED_ADDITIONAL_PROFILE",
                        "profile": path,
                        "reason": path["reason"],
                        "contract_pointer": f"deployment/cells/{cell_id}/{path['path']}",
                    }
                )
                continue
            profile = {k: v for k, v in path.items() if k != "reason"}
            add(
                "deployment_numerics:" + path["path"],
                [
                    assertion(k, True)
                    for k in (
                        "complete_frozen_population",
                        "declared_loader",
                        "own_reference_precision",
                        "output_within_frozen_gates",
                        "cache_or_native_state_lifecycle",
                        "ownership_and_ties",
                        "required_generation_or_sampling",
                        "resource_gates",
                    )
                ]
                + [assertion("gates_sha256", gates["sha256"])],
                final=True,
                profile=profile,
            )
            count = data.get("rows_evaluation", data.get("questions_evaluation", data["rows_validation"]))
            add(
                "full_task_evaluation:" + path["path"],
                [
                    assertion("rows", count),
                    assertion("expected_unique_ids", count),
                    assertion("missing_ids", 0),
                    assertion("duplicate_ids", 0),
                    assertion("native_metric_computed", True),
                    assertion("invalid_outputs_counted", True),
                    assertion("own_reference_drift_gates", True),
                    assertion("protocol_sha256", protocols[cell["data_contract"]]["sha256"]),
                ],
                final=True,
                profile=profile,
            )
            if cell["data_contract"] in PHYSICAL_DATASETS:
                checks = [
                    assertion(k, True)
                    for k in (
                        "all_fixed_tasks_scenarios",
                        "all_fixed_seeds_episodes",
                        "failed_episodes_in_denominator",
                        "native_action_contract",
                        "paired_uncertainty",
                        "source_suite_identity",
                        "native_policy_loader",
                        "reset_and_rollout_reproducibility",
                    )
                ]
                if closed_loop:
                    protocol_ref, trial_ids = closed_loop
                    checks.extend(
                        [
                            assertion("protocol_sha256", protocol_ref["sha256"]),
                            assertion("trial_ids", trial_ids, "unique_ids"),
                            assertion("missing_trials", 0),
                            assertion("duplicate_trials", 0),
                        ]
                    )
                add(
                    "family_closed_loop:" + path["path"],
                    checks,
                    final=True,
                    profile=profile,
                    pending=None
                    if closed_loop
                    else (
                        "Exact simulator suite, initial states/seeds, runtime and native policy protocol "
                        "must be sealed before this required loader-specific gate can pass."
                    ),
                )
        if cell["data_contract"] not in PHYSICAL_DATASETS:
            exclusions.append(
                {
                    "admission": "NOT_APPLICABLE",
                    "reason": "This binding's native objective is text/video conversation CE; it has no simulator action interface.",
                    "contract_pointer": f"inventory/bindings/{binding['id']}/objective",
                    "gate": "family_closed_loop",
                }
            )
        cells.append(
            {
                "id": cell_id,
                "identity": identity,
                "requirements": requirements,
                "exclusions": exclusions,
                "training_gates": ["full_schedule", "complete_sample_coverage", "initial_final_validation"],
            }
        )
    require(len(cells) == manifest["total_required_cells"], "cell count changed")
    for name, ref in protocols.items():
        if name.startswith(IDENTITY_PROTOCOL_PREFIX):
            apply_pretraining_identity(store, cells, name=name, reference=ref)
    return {
        "format": "adapter-campaign-completion-contract-v1",
        "synthetic": False,
        "inventory": inventory,
        "required_cells": len(cells),
        "sealed_inputs": [deployment, gates, admissions, *protocols.values(), *config_refs],
        "compiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": cells,
        "compatibility_receipts": admission.get("compatibility_receipts", []),
        "supersession_receipts": admission.get("supersession_receipts", []),
        "trust_boundary": "Only independently reviewed producer/source admissions may normalize raw measurements. The reviewed canonical contract digest must be provided by the caller; hashes alone do not validate scientific measurements.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = compile_contract(a.root, **read_json(a.inputs.read_bytes()))
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(digest(result))


if __name__ == "__main__":
    main()
