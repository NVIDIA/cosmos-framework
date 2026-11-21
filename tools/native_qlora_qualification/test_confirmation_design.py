# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy
import json

import pytest

from tools.native_qlora_qualification import confirmation_design as design
from tools.native_qlora_qualification.common import digest_json
from tools.native_qlora_qualification.quality_confirmation import _resample, _totals
from tools.native_qlora_qualification.test_quality_confirmation import data


def packet():
    trained, frozen, _ = data()
    tasks = [f"task{i:02}" for i in range(22)]
    spec = design.proposal(tasks, {"nf4": "a" * 64, "bf16": "b" * 64})
    spec["paired_seeds"] = {"pilot": [17, 42, 123], "confirmation": [2001, 2002]}
    spec["groups"] = {"pilot": 2, "confirmation": 2}
    spec["minimum_groups_per_task"] = {"pilot": 1, "confirmation": 1}
    spec["provenance_audit_sha256"] = "c" * 64
    spec["task_taxonomy_sha256"] = "e" * 64
    inventory = []
    anchors = []
    for role, offset in (("exposed", 100), ("pilot", 200), ("confirmation", 300)):
        for group in range(2):
            for i, task in enumerate(tasks):
                identifier = f"20230929_{offset + group}_SN22_T1_Camera1_0_{i}"
                media = digest_json([offset, group])
                inventory.append(
                    {
                        "id": identifier,
                        "sha256": media,
                        "raw_row_sha256": digest_json(identifier),
                        "role": role,
                        "task": task,
                        "session": f"session{offset + group}",
                        "event": f"event{offset + group}",
                    }
                )
                if role == "pilot":
                    row = copy.deepcopy(trained[17]["nf4"][i % 6])
                    row.update(
                        id=identifier,
                        raw_row_sha256=digest_json(identifier),
                        row_identity=digest_json(identifier),
                        media_sha256=media,
                    )
                    anchors.append(row)
    trained = {seed: {arm: copy.deepcopy(anchors) for arm in ("nf4", "bf16")} for seed in spec["paired_seeds"]["pilot"]}
    frozen = {arm: copy.deepcopy(anchors) for arm in ("nf4", "bf16")}
    spec["inventory_sha256"] = digest_json(inventory)
    return spec, inventory, trained, frozen


def test_proposal_is_concrete_but_refuses_unbound_provenance():
    spec, inventory, _, _ = packet()
    unbound = design.proposal(spec["task_types"], spec["recipe_sha256"])
    assert len(unbound["paired_seeds"]["pilot"]) == 8
    assert len(unbound["paired_seeds"]["confirmation"]) == 20
    assert unbound["groups"] == {"pilot": 40, "confirmation": 400}
    with pytest.raises(ValueError, match="unbound"):
        design.validate_design(unbound, inventory, role="pilot")


@pytest.mark.parametrize("damage", ["recording", "video", "session", "event", "transitive"])
def test_related_component_cannot_cross_pilot_final_or_exposed(damage):
    spec, inventory, _, _ = packet()
    a, b = inventory[0], inventory[44]
    if damage == "recording":
        b["id"] = a["id"].replace("Camera1", "Camera2")
    elif damage == "video":
        b["sha256"] = a["sha256"]
    elif damage in {"session", "event"}:
        b[damage] = a[damage]
    else:
        # pilot -> second pilot family via media -> exposed via session
        inventory[66]["sha256"] = b["sha256"]
        inventory[66]["session"] = a["session"]
    spec["inventory_sha256"] = digest_json(inventory)
    with pytest.raises(ValueError, match="crosses"):
        design.validate_design(spec, inventory, role="pilot")


@pytest.mark.parametrize(
    "damage",
    ["seed_overlap", "seed_duplicate", "task", "coverage", "unknown", "hash", "margin", "weights", "pilot_missing"],
)
def test_incomplete_prospective_contract_refuses(damage):
    spec, inventory, _, _ = packet()
    role = "pilot"
    if damage == "seed_overlap":
        spec["paired_seeds"]["confirmation"][0] = 17
    elif damage == "seed_duplicate":
        spec["paired_seeds"]["pilot"] = [17, 17]
    elif damage == "task":
        spec["task_types"].pop()
    elif damage == "coverage":
        inventory = [r for r in inventory if not (r["role"] == "pilot" and r["task"] == "task00")]
        spec["inventory_sha256"] = digest_json(inventory)
    elif damage == "unknown":
        inventory[0]["session"] = "unknown"
        spec["inventory_sha256"] = digest_json(inventory)
    elif damage == "hash":
        inventory[0]["event"] += "changed"
    elif damage == "margin":
        spec["margins"]["accuracy_loss"] = 0.011
    elif damage == "weights":
        spec["weighting"]["tasks"] = "equal_macro"
    elif damage == "pilot_missing":
        role = "confirmation"
    with pytest.raises(ValueError):
        design.validate_design(spec, inventory, role=role)


@pytest.mark.parametrize("crossed", [False, True])
def test_exact_original_bootstrap_replay_without_changing_history(crossed):
    trained, frozen, groups = data()
    for seed in trained:
        for i, row in enumerate(trained[seed]["nf4"]):
            row["native_sft_ce"] *= 1 + i * seed / 2000
    original_totals = _totals(trained, frozen, groups)
    totals = design.paired_totals(trained, frozen, groups, dict.fromkeys(groups, "task"), seeds=[17, 42, 123])
    assert totals == original_totals
    old = _resample(totals, crossed=crossed, repetitions=200)
    new = design.resample(totals, crossed=crossed, repetitions=200)
    assert new["bounds"] == old["bounds"]
    assert new["ni_criteria_passed"]["relative_native_token_nll"] == old["criteria_passed"]["native_token_nll"]
    assert new["ni_criteria_passed"]["accuracy_loss"] == old["criteria_passed"]["accuracy_preservation"]


@pytest.mark.parametrize("damage", ["seed", "arm", "mask", "order", "tokens", "inventory", "media"])
def test_actual_score_pairing_and_bound_inventory_refuse(damage):
    spec, inventory, trained, frozen = packet()
    if damage == "seed":
        trained[99] = trained.pop(17)
    elif damage == "arm":
        trained[17].pop("nf4")
    elif damage == "mask":
        trained[17]["nf4"][0]["inputs"]["labels"] = "changed"
    elif damage == "order":
        trained[42]["bf16"].reverse()
    elif damage == "tokens":
        trained[123]["nf4"][0]["native_supervised_tokens"] = 2.5
    elif damage == "inventory":
        inventory[44]["id"] += "9"
        spec["inventory_sha256"] = digest_json(inventory)
    else:
        for arms in [*trained.values(), frozen]:
            for rows in arms.values():
                rows[0]["media_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        design.analyze(trained, frozen, inventory, spec, role="pilot", repetitions=20)


def test_pilot_and_test_repetitions_never_pass_final_confirmation():
    spec, inventory, trained, frozen = packet()
    result = design.analyze(trained, frozen, inventory, spec, role="pilot", repetitions=30)
    assert result["conditional_recipe_ni_passed"] is False
    assert result["release_qualified"] is False
    assert result["provenance_truth_or_population_independence_verified"] is False
    assert len(result["task_slices_exploratory"]) == 22
    assert all(v["both_ni_passed"] for v in result["uncertainty"].values())
    assert not any(v["accuracy_improvement_supported"] for v in result["uncertainty"].values())
    assert all(not p["calibrated_bootstrap_power"] for p in result["planning"])


def test_pooling_and_influence_decomposition_retain_token_weights_and_seed_floor():
    trained, frozen, groups = data()
    for seed in trained:
        for i, row in enumerate(trained[seed]["nf4"]):
            row["native_sft_ce"] = float(seed) * (10 if i % 2 == 0 else 1)
        for row in trained[seed]["bf16"]:
            row["native_sft_ce"] = 10.0
    totals = design.paired_totals(trained, frozen, groups, dict.fromkeys(groups, "task"), seeds=[17, 42, 123])
    diagnostics = design.variance_diagnostics(totals)
    assert diagnostics["point_estimates"]["relative_native_token_nll"] == pytest.approx(
        4 * (17 + 42 + 123) / 3 / 10 - 1
    )
    c = diagnostics["variance_components"]["relative_native_token_nll"]
    assert c["unshrunk_seed_mean_variance"] > 0
    large = design.joint_power(diagnostics, seeds=3, groups=1000000)
    assert large["tests"]["crossed:relative_native_token_nll"]["seed_variance_floor"] > 0
    assert large["tests"]["crossed:relative_native_token_nll"]["approximate_power"] < 0.8


def test_joint_power_is_union_bound_not_product_or_single_guard():
    c = {
        "seed_variance": 0.0004,
        "group_variance": 0.003,
        "interaction_variance": 0.001,
        "unshrunk_seed_mean_variance": 0.0005,
        "unshrunk_group_mean_variance": 0.004,
    }
    diagnostics = {"variance_components": {k: dict(c) for k in design.MARGINS}}
    p = design.joint_power(diagnostics, seeds=12, groups=180)
    powers = [v["approximate_power"] for v in p["tests"].values()]
    assert p["approximate_joint_lower_bound"] == pytest.approx(max(0, sum(powers) - 3))
    assert p["approximate_joint_lower_bound"] <= min(powers)
    assert (
        design.joint_power(diagnostics, seeds=12, groups=180, variance_multiplier=2)["approximate_joint_lower_bound"]
        <= p["approximate_joint_lower_bound"]
    )
    assert not p["group_sufficiency_established"]


def test_one_group_cannot_be_resampled_as_independent_questions():
    trained, frozen, groups = data()
    totals = design.paired_totals(
        trained, frozen, dict.fromkeys(groups, "same"), dict.fromkeys(groups, "task"), seeds=[17, 42, 123]
    )
    with pytest.raises(ValueError, match="at least two"):
        design.resample(totals, crossed=True)


def test_bound_metadata_reader_refuses_wrong_hash_and_tensor_paths(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"ok": True}))
    with pytest.raises(ValueError, match="hash"):
        design._read_bound({"path": str(p), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="metadata"):
        design._read_bound({"path": str(tmp_path / "model.pt"), "sha256": "0" * 64})


def bound_file(tmp_path, name, value, *, jsonl=False):
    import hashlib

    p = tmp_path / name
    raw = ("\n".join(json.dumps(x) for x in value) + "\n") if jsonl else json.dumps(value)
    p.write_text(raw)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def bundle_fixture(tmp_path):
    spec, inventory, trained, frozen = packet()
    taxonomy = bound_file(
        tmp_path,
        "taxonomy.json",
        {"protocol": "cosmos.qlora-confirmation-task-taxonomy.v1", "task_types": spec["task_types"]},
    )
    relationships = bound_file(
        tmp_path,
        "relationships.json",
        {
            "inventory_sha256": spec["inventory_sha256"],
            "evidence": "synthetic test fixture",
            "known_exposed_rows": [r for r in inventory if r["role"] == "exposed"],
        },
    )
    spec["task_taxonomy_sha256"] = taxonomy["sha256"]
    provenance = {
        "protocol": "cosmos.qlora-confirmation-provenance.v1",
        "complete": True,
        "inventory_sha256": spec["inventory_sha256"],
        "unresolved_relationships": [],
        "exposure_and_relationship_source_sha256": relationships["sha256"],
        "task_taxonomy_sha256": taxonomy["sha256"],
    }
    bundle = {
        "role": "pilot",
        "task_taxonomy": taxonomy,
        "exposure_and_relationship_source": relationships,
        "inventory": bound_file(tmp_path, "inventory.json", inventory),
        "provenance_audit": bound_file(tmp_path, "provenance.json", provenance),
    }
    spec["provenance_audit_sha256"] = bundle["provenance_audit"]["sha256"]
    bundle["spec"] = bound_file(tmp_path, "spec.json", spec)
    bundle["trained"] = {
        str(seed): {a: bound_file(tmp_path, f"{seed}-{a}.jsonl", rows, jsonl=True) for a, rows in arms.items()}
        for seed, arms in trained.items()
    }
    bundle["frozen"] = {a: bound_file(tmp_path, f"frozen-{a}.jsonl", rows, jsonl=True) for a, rows in frozen.items()}
    proof = {
        "protocol": "cosmos.qlora-confirmation-score-artifacts.v1",
        "complete": True,
        "recipe_sha256": spec["recipe_sha256"],
        "paired_seeds": spec["paired_seeds"]["pilot"],
        "inventory_sha256": spec["inventory_sha256"],
        "score_bundle_sha256": digest_json({k: bundle[k] for k in ("trained", "frozen")}),
    }
    bundle["artifact_validation"] = bound_file(tmp_path, "artifacts.json", proof)
    return spec, bundle, proof


def test_bound_pilot_result_has_design_ancestry_and_never_qualifies(tmp_path):
    spec, bundle, _ = bundle_fixture(tmp_path)
    result = design.analyze_bundle(bundle)
    assert result["external_bindings_validated"] is True
    assert result["design_identity_sha256"] == design.design_identity(spec)
    assert result["conditional_recipe_ni_passed"] is False
    assert result["release_qualified"] is False
    assert result["provenance_truth_or_population_independence_verified"] is False


@pytest.mark.parametrize("damage", ["failed", "wrong_scores", "wrong_recipe", "wrong_seed"])
def test_artifact_attestation_is_successful_and_score_specific(tmp_path, damage):
    _, bundle, proof = bundle_fixture(tmp_path)
    if damage == "failed":
        proof["complete"] = False
    elif damage == "wrong_scores":
        proof["score_bundle_sha256"] = "f" * 64
    elif damage == "wrong_recipe":
        proof["recipe_sha256"]["nf4"] = "f" * 64
    else:
        proof["paired_seeds"] = [17, 42, 124]
    bundle["artifact_validation"] = bound_file(tmp_path, "altered-artifacts.json", proof)
    with pytest.raises(ValueError, match="artifact proof"):
        design.analyze_bundle(bundle)


@pytest.mark.parametrize("damage", ["ancestry", "recipe", "seeds", "groups", "inventory", "unvalidated"])
def test_unrelated_or_modified_pilot_cannot_unlock_confirmation(tmp_path, damage):
    spec, bundle, _ = bundle_fixture(tmp_path)
    pilot = {
        "protocol": design.PROTOCOL,
        "role": "pilot",
        "design_identity_sha256": design.design_identity(spec),
        "paired_seeds": spec["paired_seeds"]["pilot"],
        "coverage": {"groups": 2},
        "recipe_sha256": dict(spec["recipe_sha256"]),
        "inventory_sha256": spec["inventory_sha256"],
        "external_bindings_validated": True,
    }
    if damage == "ancestry":
        pilot["design_identity_sha256"] = "f" * 64
    elif damage == "recipe":
        pilot["recipe_sha256"]["bf16"] = "f" * 64
    elif damage == "seeds":
        pilot["paired_seeds"] = [17, 42, 124]
    elif damage == "groups":
        pilot["coverage"]["groups"] = 3
    elif damage == "inventory":
        pilot["inventory_sha256"] = "f" * 64
    else:
        pilot["external_bindings_validated"] = False
    bundle["pilot_analysis"] = bound_file(tmp_path, "pilot.json", pilot)
    spec["pilot_analysis_sha256"] = bundle["pilot_analysis"]["sha256"]
    bundle["spec"] = bound_file(tmp_path, "final-spec.json", spec)
    bundle["role"] = "confirmation"
    with pytest.raises(ValueError, match="separate pilot"):
        design.analyze_bundle(bundle)


def test_complete_bound_confirmation_reports_ni_but_never_release(tmp_path):
    spec, bundle, proof = bundle_fixture(tmp_path)
    pilot = design.analyze_bundle(bundle)
    bundle["pilot_analysis"] = bound_file(tmp_path, "pilot-result.json", pilot)
    spec["pilot_analysis_sha256"] = bundle["pilot_analysis"]["sha256"]
    bundle["spec"] = bound_file(tmp_path, "final-spec.json", spec)
    inventory = design._read_bound(bundle["inventory"])
    original = design._read_bound(bundle["trained"]["17"]["nf4"], jsonl=True)
    final_rows = []
    for prototype, row in zip(original, (r for r in inventory if r["role"] == "confirmation"), strict=True):
        value = copy.deepcopy(prototype)
        value.update(
            id=row["id"],
            raw_row_sha256=row["raw_row_sha256"],
            row_identity=digest_json(row["id"]),
            media_sha256=row["sha256"],
        )
        final_rows.append(value)
    bundle["role"] = "confirmation"
    bundle["trained"] = {
        str(seed): {a: bound_file(tmp_path, f"final-{seed}-{a}.jsonl", final_rows, jsonl=True) for a in ("nf4", "bf16")}
        for seed in spec["paired_seeds"]["confirmation"]
    }
    bundle["frozen"] = {
        a: bound_file(tmp_path, f"final-frozen-{a}.jsonl", final_rows, jsonl=True) for a in ("nf4", "bf16")
    }
    proof["paired_seeds"] = spec["paired_seeds"]["confirmation"]
    proof["score_bundle_sha256"] = digest_json({k: bundle[k] for k in ("trained", "frozen")})
    bundle["artifact_validation"] = bound_file(tmp_path, "final-artifacts.json", proof)
    result = design.analyze_bundle(bundle)
    assert result["conditional_recipe_ni_passed"] is True
    assert result["conditional_original_quality_guards_passed"] is False
    assert not any(v["useful_adaptation_supported"] for v in result["uncertainty"].values())
    assert result["release_qualified"] is False
    assert result["provenance_truth_or_population_independence_verified"] is False
    assert not any(v["accuracy_improvement_supported"] for v in result["uncertainty"].values())


@pytest.mark.parametrize("damage", ["taxonomy", "relationship_inventory", "relationship_hash"])
def test_actual_taxonomy_and_relationship_receipts_are_read_and_bound(tmp_path, damage):
    _, bundle, _ = bundle_fixture(tmp_path)
    if damage == "taxonomy":
        value = design._read_bound(bundle["task_taxonomy"])
        value["task_types"][0] = "another_task"
        bundle["task_taxonomy"] = bound_file(tmp_path, "wrong-taxonomy.json", value)
    elif damage == "relationship_inventory":
        bundle["exposure_and_relationship_source"] = bound_file(
            tmp_path, "wrong-relationship.json", {"inventory_sha256": "f" * 64}
        )
    else:
        bundle["exposure_and_relationship_source"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        design.analyze_bundle(bundle)


@pytest.mark.parametrize("damage", ["nan_effect", "target", "omit_stress"])
def test_planning_assumptions_must_be_finite_and_predeclared(damage):
    spec, inventory, _, _ = packet()
    if damage == "nan_effect":
        spec["planning"]["assumed_relative_nll"] = float("nan")
    elif damage == "target":
        spec["planning"]["target_joint_power"] = 1
    else:
        spec["planning"]["variance_multipliers"] = [1.0]
    with pytest.raises(ValueError):
        design.validate_design(spec, inventory, role="pilot")


@pytest.mark.parametrize("damage", ["empty", "omitted"])
def test_known_exposure_cannot_be_omitted_from_actual_bound_inventory(tmp_path, damage):
    spec, bundle, _ = bundle_fixture(tmp_path)
    source = design._read_bound(bundle["exposure_and_relationship_source"])
    if damage == "empty":
        source["known_exposed_rows"] = []
    else:
        source["known_exposed_rows"][0]["id"] = "missing-known-row"
    bundle["exposure_and_relationship_source"] = bound_file(tmp_path, "altered-exposure.json", source)
    provenance = design._read_bound(bundle["provenance_audit"])
    provenance["exposure_and_relationship_source_sha256"] = bundle["exposure_and_relationship_source"]["sha256"]
    bundle["provenance_audit"] = bound_file(tmp_path, "updated-provenance.json", provenance)
    spec["provenance_audit_sha256"] = bundle["provenance_audit"]["sha256"]
    bundle["spec"] = bound_file(tmp_path, "updated-spec.json", spec)
    with pytest.raises(ValueError, match="exposure|exposed"):
        design.analyze_bundle(bundle)


def test_failed_pilot_planning_cannot_become_final_recipe_acceptance(tmp_path):
    spec, bundle, _ = bundle_fixture(tmp_path)
    pilot = design.analyze_bundle(bundle)
    for values in pilot["diagnostics"]["variance_components"].values():
        values["unshrunk_seed_mean_variance"] = 1.0
        values["unshrunk_group_mean_variance"] = 1.0
    pilot["planning"], pilot["pilot_planning_eligible"] = design.pilot_planning(spec, pilot["diagnostics"])
    assert pilot["pilot_planning_eligible"] is False
    bundle["pilot_analysis"] = bound_file(tmp_path, "failed-pilot.json", pilot)
    spec["pilot_analysis_sha256"] = bundle["pilot_analysis"]["sha256"]
    bundle["spec"] = bound_file(tmp_path, "final-spec.json", spec)
    bundle["role"] = "confirmation"
    with pytest.raises(ValueError, match="pilot planning rejects"):
        design.analyze_bundle(bundle)


def test_original_adaptation_guard_is_separate_from_two_margin_ni():
    trained, frozen, groups = data()
    totals = design.paired_totals(trained, frozen, groups, dict.fromkeys(groups, "task"), seeds=[17, 42, 123])
    positive = design.resample(totals, crossed=True, repetitions=100)
    assert positive["both_ni_passed"] is True
    assert positive["useful_adaptation_supported"] is True
    assert positive["all_original_quality_guards_passed"] is True
    frozen["nf4"] = copy.deepcopy(trained[17]["nf4"])
    totals = design.paired_totals(trained, frozen, groups, dict.fromkeys(groups, "task"), seeds=[17, 42, 123])
    zero = design.resample(totals, crossed=True, repetitions=100)
    assert zero["both_ni_passed"] is True
    assert zero["useful_adaptation_supported"] is False
    assert zero["all_original_quality_guards_passed"] is False
