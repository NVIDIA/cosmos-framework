# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Manual sampling/weighting fixtures; these are not real policy-quality results."""

import copy
import hashlib
import json

import pytest

from tools.native_qlora_qualification import action_quality as quality


def tables(factor=1.0):
    return {
        17: {
            "task-a": {"ep-a": 1.0 * factor, "ep-b": 3.0 * factor},
            "task-b": {"ep-c": 2.0 * factor, "ep-d": 4.0 * factor},
        },
        42: {
            "task-a": {"ep-a": 5.0 * factor, "ep-b": 7.0 * factor},
            "task-b": {"ep-c": 6.0 * factor, "ep-d": 8.0 * factor},
        },
    }


def row(identifier, value, count=70):
    return {"ids": [identifier], "metrics": {quality.METRIC: {"sum": value * count, "count": count}}}


def test_windows_are_averaged_before_episode_weighting():
    groups = {"a": ("task-a", "ep-a"), **{f"b{i}": ("task-a", "ep-b") for i in range(9)}}
    result = quality.episode_means([row("a", 100), *[row(f"b{i}", 0) for i in range(9)]], groups)
    assert result == {"task-a": {"ep-a": 100, "ep-b": 0}}
    assert sum(result["task-a"].values()) / 2 == 50  # Row-natural mean would be10.


def test_equal_tasks_not_global_episode_count_weighting():
    reference = {seed: {"a": {"a1": 1.0, "a2": 1.0}, "b": {f"b{i}": 9.0 for i in range(4)}} for seed in (17, 42)}
    result = quality.compare_recipes(reference, reference, draws=30)
    assert result["reference_equal_seed_task_episode_mse"] == 5
    assert result["relative_increase_upper_one_sided95"] == 0


def test_shared_episode_indices_across_training_seeds_preserve_correlation():
    cube = [[[0.0, 100.0], [20.0, 40.0]], [[100.0, 0.0], [40.0, 20.0]]]
    # Opposite episode effects across two trained seeds cancel because both
    # seeds see the SAME episode draw. Independent per-seed draws need not.
    assert quality.draw_mean(cube, [0, 1], [0], [[0, 0]]) == 50
    assert quality.draw_mean(cube, [0, 1], [0], [[1, 1]]) == 50
    assert quality.draw_mean(cube, [0, 0], [0], [[0, 0]]) == 0  # Training-seed uncertainty remains.
    # Two selected occurrences of a task may have distinct episode draws;
    # each occurrence's draw still remains common across all seeds.
    assert quality.draw_mean(cube, [0, 1], [0, 0], [[0, 0], [1, 1]]) == 50


def test_identical_arms_have_zero_bootstrap_uncertainty_despite_large_cluster_variation():
    reference = tables()
    result = quality.compare_recipes(copy.deepcopy(reference), reference, draws=500)
    assert result["relative_increase"] == 0
    assert result["relative_increase_upper_one_sided95"] == 0


def test_manual_proportional_effect_and_seed_task_episode_means():
    result = quality.compare_recipes(tables(1.1), tables(), draws=200)
    assert result["reference_equal_seed_task_episode_mse"] == 4.5
    assert result["candidate_equal_seed_task_episode_mse"] == pytest.approx(4.95)
    assert result["relative_increase"] == pytest.approx(0.1)
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(0.1)


def test_exact_manual_rng_draw_plan_and_reproducibility(monkeypatch):
    class FixedDraw:
        def __init__(self, seed):
            assert seed == 6809
            self.calls = 0

        def choices(self, values, *, k):
            self.calls += 1
            # One seed draw, one task draw, then one episode draw per task
            # occurrence. There is no inside-seed episode sampling loop.
            return {1: [0, 1], 2: [0, 1], 3: [0, 0], 4: [1, 1]}[self.calls]

    monkeypatch.setattr(quality.random, "Random", FixedDraw)
    candidate = tables()
    candidate[17]["task-a"]["ep-a"] = 9.0
    # Reference draw=(1+4+5+8)/4=4.5; candidate=(9+4+5+8)/4=6.5.
    result = quality.compare_recipes(candidate, tables(), draws=1)
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(6.5 / 4.5 - 1)


@pytest.mark.parametrize("change", ["duplicate", "missing", "unknown", "denominator", "nonfinite"])
def test_bad_native_row_inventory_or_objective_reduction_fails(change):
    groups = {"a": ("t", "e"), "b": ("t", "e")}
    rows = [row("a", 1), row("b", 2)]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "unknown":
        rows[1]["ids"] = ["unknown"]
    elif change == "denominator":
        rows[1]["metrics"][quality.METRIC]["count"] = 69
    else:
        rows[0]["metrics"][quality.METRIC]["sum"] = float("nan")
    with pytest.raises(ValueError):
        quality.episode_means(rows, groups, expected_count=70)


@pytest.mark.parametrize("change", ["seed", "task", "episode", "zero-reference"])
def test_missing_pairing_and_undefined_relative_statistic_fail(change):
    before, after = tables(), tables()
    if change == "seed":
        after.pop(17)
    elif change == "task":
        after[17].pop("task-a")
    elif change == "episode":
        after[42]["task-b"].pop("ep-d")
    else:
        before = tables(0)
    with pytest.raises(ValueError):
        quality.compare_recipes(after, before, draws=10)


def test_all_declared_seeds_required_before_recipe_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "load_groups", lambda *args: {"a": ("t", "e")})
    # Deliberately no completed results: do not manufacture nine runs or native
    # checkpoints merely to test admission of the analysis inventory.
    with pytest.raises(ValueError, match="all nine"):
        quality.analyze([], validation=tmp_path / "unused", manifest=tmp_path / "unused")


def rollout_case(task, initial=0, **extra):
    return {"suite": "libero_spatial", "task_id": task, "init_state_id": initial, **extra}


def test_rollout_equal_tasks_not_control_frames_or_number_of_initial_states():
    cases = [rollout_case(0), *[rollout_case(1, index) for index in range(9)]]
    candidate = [{**case, "success": case["task_id"] == 0, "steps": 256} for case in cases]
    reference = [{**case, "success": True, "steps": 8} for case in cases]
    result = quality.compare_rollout_checkpoints(candidate, reference, cases, draws=100)
    assert result["candidate_equal_task_success"] == 0.5  # Case-natural rate would be0.1.
    assert result["reference_equal_task_success"] == 1
    assert result["success_loss_percentage_points"] == 50
    assert not result["recipe_preservation_established"]
    assert not result["useful_adaptation_established"]


def test_rollout_identical_arms_cancel_task_variability_and_all_fail_is_not_usefulness():
    cases = [rollout_case(0), rollout_case(1)]
    for outcomes in ((True, False), (False, False)):
        episodes = [{**case, "success": outcome} for case, outcome in zip(cases, outcomes, strict=True)]
        result = quality.compare_rollout_checkpoints(episodes, copy.deepcopy(episodes), cases, draws=100)
        assert result["success_loss_upper_one_sided95_percentage_points"] == 0
        assert not result["recipe_preservation_established"]
        assert not result["useful_adaptation_established"]


def test_rollout_manual_paired_whole_task_draw(monkeypatch):
    class FixedDraw:
        def __init__(self, seed):
            assert seed == 6809

        def choices(self, values, *, k):
            assert values == [100, -100] and k == 2
            return [values[1], values[1]]

    monkeypatch.setattr(quality.random, "Random", FixedDraw)
    cases = [rollout_case(0), rollout_case(1)]
    before = [rollout_case(0, success=True), rollout_case(1, success=False)]
    after = [rollout_case(0, success=False), rollout_case(1, success=True)]
    result = quality.compare_rollout_checkpoints(after, before, cases, draws=1)
    assert result["success_loss_percentage_points"] == 0
    assert result["success_loss_upper_one_sided95_percentage_points"] == -100


@pytest.mark.parametrize("change", ["duplicate", "missing", "unknown", "nonboolean", "case-duplicate", "boolean-id"])
def test_rollout_requires_complete_prospective_case_pairing(change):
    cases = [rollout_case(0), rollout_case(1)]
    rows = [rollout_case(0, success=True), rollout_case(1, success=False)]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "unknown":
        rows[1]["task_id"] = 2
    elif change == "nonboolean":
        rows[0]["success"] = 1
    elif change == "case-duplicate":
        cases[1] = cases[0]
    else:
        rows[0]["task_id"] = False
    with pytest.raises(ValueError):
        quality.compare_rollout_checkpoints(rows, rows, cases, draws=10)


def test_rollout_does_not_manufacture_task_uncertainty_from_many_states():
    cases = [rollout_case(0, index) for index in range(100)]
    rows = [{**case, "success": True} for case in cases]
    with pytest.raises(ValueError, match="multiple task"):
        quality.compare_rollout_checkpoints(rows, rows, cases)


def test_read_json_hash_binds_exact_bytes(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"value": 1}))
    value, first = quality.read_json(path)
    path.write_text(json.dumps({"value": 2}))
    _, second = quality.read_json(path)
    assert value == {"value": 1} and first != second


@pytest.mark.parametrize("memory", [False, None, 1])
def test_missing_attributed_process_memory_blocks_combined_comparison(tmp_path, monkeypatch, memory):
    monkeypatch.setattr(quality, "load_groups", lambda *args: {"a": ("t", "e")})
    path = tmp_path / "development-result.json"
    path.write_text(
        json.dumps(
            {
                "mode": "nf4_lora",
                "seed": 17,
                "protocol_sha256": quality.PROTOCOL_SHA,
                "fixed_work": {"updates": 300},
                "endpoint": {"fresh_portable_exact_primary_parity": True},
                "attributed_process_memory_valid": memory,
            }
        )
    )
    with pytest.raises(ValueError, match="attributed process-memory"):
        quality.analyze([path], validation=tmp_path / "unused", manifest=tmp_path / "unused")


@pytest.fixture
def continuation_fixture(tmp_path, monkeypatch):
    """Synthetic ingestion fixtures only: no manufactured model-quality evidence."""
    groups = {f"{task}-{episode}": (task, episode) for task in ("a", "b") for episode in ("0", "1")}
    monkeypatch.setattr(quality, "load_groups", lambda *args: groups)
    target_sha = "e" * 64
    protocols = {17: "a" * 64, 42: "b" * 64, 123: "c" * 64}
    contract = {
        "reused_cases": [],
        "executed_orders": {"42": ["bf16_frozen"], "123": list(quality.MODES)},
        "scientific_protocol_sha256": "c" * 64,
        "target_sha256": target_sha,
    }
    paths = []
    for seed in quality.SEEDS:
        for mode in quality.MODES:
            reused = seed == 17 or (seed == 42 and mode != "bf16_frozen")
            path = tmp_path / f"{mode}-{seed}.json"
            score_path = tmp_path / f"{mode}-{seed}-score.json"
            score = {
                "step": 300,
                "metrics": {quality.METRIC: 0.04},
                "rows": [row(identifier, 0.04) for identifier in groups],
            }
            score_path.write_text(json.dumps(score))
            value = {
                "mode": mode,
                "seed": seed,
                "protocol_sha256": protocols[seed] if reused else "c" * 64,
                "fixed_work": {
                    "updates": 300,
                    "history": [
                        {"step": step, "metrics": {quality.METRIC: loss}}
                        for step, loss in zip((0, 100, 200, 300), (0.06, 0.05, 0.04, 0.04), strict=True)
                    ],
                },
                "endpoint": {"fresh_portable_exact_primary_parity": True, "resident_score": str(score_path)},
                "attributed_process_memory_valid": True,
                "target_sha256": target_sha,
                "target": {
                    "metric": quality.METRIC,
                    "threshold": 0.049561327725898344,
                    "attained": True,
                    "step": 200,
                    "native_seconds": 200.0,
                    "preflight_inclusive_seconds": 220.0,
                },
                "calibration_cost": {"seconds": 10.0},
                "sampled_memory": {},
            }
            path.write_text(json.dumps(value))
            paths.append(path)
            if reused:
                contract["reused_cases"].append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "protocol_sha256": value["protocol_sha256"],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(quality, "CONTINUATION_CONTRACT_SHA", hashlib.sha256(contract_path.read_bytes()).hexdigest())
    return paths, {
        "validation": tmp_path / "unused",
        "manifest": tmp_path / "unused",
        "continuation_contract": contract_path,
    }


def test_continuation_preserves_original_evidence_and_the_statistic(continuation_fixture):
    paths, kwargs = continuation_fixture
    original_bytes = {path: path.read_bytes() for path in paths}
    result = quality.analyze(paths, **kwargs)
    primary = result["primary_nf4_vs_matched_bf16_lora"]
    assert primary["relative_increase"] == 0
    assert primary["relative_increase_upper_one_sided95"] == 0
    assert primary["predeclared_offline_preservation_margin"] == 0.05
    assert primary["training_seeds"] == [17, 42, 123]
    assert [r["seed"] for r in result["input_receipts"] if r["reused_completed_arm"]] == [17, 17, 17, 42, 42]
    assert len({r["original_protocol_sha256"] for r in result["input_receipts"]}) == 3
    assert not result["independent_confirmation"]["established"]
    assert not result["closed_loop_preservation"]["established"]
    assert all(path.read_bytes() == before for path, before in original_bytes.items())


def test_continuation_contract_requires_exact_bytes(continuation_fixture):
    paths, kwargs = continuation_fixture
    contract = kwargs["continuation_contract"]
    contract.write_bytes(contract.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sealed remaining-arm contract"):
        quality.analyze(paths, **kwargs)


def test_continuation_cannot_substitute_a_timing_candidate_for_reused_reference(continuation_fixture):
    paths, kwargs = continuation_fixture
    path = next(path for path in paths if path.name == "nf4_lora-42.json")
    value = json.loads(path.read_text())
    # Same protocol, step and metrics are insufficient to replace the original
    # content-bound reference result with another run from the finite study.
    value["endpoint"]["checkpoint"] = "/synthetic/candidate/iter_000300"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="admitted original bytes"):
        quality.analyze(paths, **kwargs)


@pytest.mark.parametrize("change", ["missing", "duplicate", "protocol", "target", "crossing", "nonfinite"])
def test_continuation_rejects_incomplete_or_changed_panel(continuation_fixture, change):
    paths, kwargs = continuation_fixture
    if change == "missing":
        paths = paths[:-1]
    elif change == "duplicate":
        paths = paths + [paths[-1]]
    else:
        path = paths[-1]  # Newly executed seed123, not a reused hash-bound result.
        value = json.loads(path.read_text())
        if change == "protocol":
            value["protocol_sha256"] = quality.PROTOCOL_SHA
        elif change == "target":
            value["target_sha256"] = "wrong"
        elif change == "crossing":
            value["target"]["step"] = 300
        else:
            value["fixed_work"]["history"][1]["metrics"][quality.METRIC] = float("nan")
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        quality.analyze(paths, **kwargs)


def test_unreached_target_is_retained_and_cannot_acquire_a_clock():
    run = {
        "target_sha256": "fixed",
        "fixed_work": {"history": [{"step": step, "metrics": {quality.METRIC: 0.06}} for step in (0, 100, 200, 300)]},
        "target": {
            "metric": quality.METRIC,
            "threshold": 0.049561327725898344,
            "attained": False,
            "step": None,
            "native_seconds": None,
            "preflight_inclusive_seconds": None,
        },
    }
    quality.validate_continuation_target(run, "fixed")
    run["target"]["native_seconds"] = 0.0
    with pytest.raises(ValueError, match="fabricated target clock"):
        quality.validate_continuation_target(run, "fixed")
