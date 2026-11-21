# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Manual prospective statistical/timing fixtures, not trained policy evidence."""

import copy
import json

import pytest

from tools.native_qlora_qualification import alpamayo_quality as q


def row(identifier, value, *, count=1):
    return {
        "ids": [identifier],
        "metrics": {q.ADE: {"sum": value, "count": count}, q.FDE: {"sum": 2 * value, "count": count}},
    }


def tables(factor=1.0):
    return {
        17: {("p", "A", 1): 1.0 * factor, ("p", "B", 1): 3.0 * factor},
        42: {("p", "A", 1): 5.0 * factor, ("p", "B", 1): 7.0 * factor},
    }


def test_equal_clip_then_block_then_seed_weighting_not_pooled_rows():
    groups = {"a": ("A",), **{f"b{i}": ("B",) for i in range(9)}}
    means = q.block_means([row("a", 100), *[row(f"b{i}", 10) for i in range(9)]], groups, q.ADE)
    assert means == {("A",): 100, ("B",): 10}
    result = q.compare_recipes({17: means, 42: means}, {17: means, 42: means}, draws=2)
    assert result["reference_equal_seed_block_clip_meters"] == 55  # Pooled row mean would be19.


def test_whole_block_draw_is_common_across_seeds_and_no_inner_clip_resampling():
    cube = [[0, 100], [100, 0]]
    assert q.draw_mean(cube, [0, 1], [0, 0]) == 50
    assert q.draw_mean(cube, [0, 1], [1, 1]) == 50
    assert q.draw_mean(cube, [0, 0], [0, 0]) == 0  # Seed uncertainty remains.


def test_exact_manual_two_draw_plan_and_shared_metric_sampling(monkeypatch):
    class Fixed:
        def __init__(self, seed):
            assert seed == 70241
            self.calls = 0

        def choices(self, values, *, k):
            self.calls += 1
            return {1: [0, 1], 2: [0, 0]}[self.calls]  # Exactlyseed/block, neverclip draws.

    monkeypatch.setattr(q.random, "Random", Fixed)
    before = tables()
    after = tables()
    after[17][("p", "A", 1)] = 9
    result = q.compare_recipes(after, before, draws=1)
    assert result["reference_equal_seed_block_clip_meters"] == 4
    assert result["absolute_difference_upper_one_sided95_meters"] == 4
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(7 / 3 - 1)


def test_proportional_and_identical_arms_have_expected_effects():
    result = q.compare_recipes(tables(1.1), tables(), draws=200)
    assert result["candidate_equal_seed_block_clip_meters"] == pytest.approx(4.4)
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(0.1)
    same = q.compare_recipes(tables(), tables(), draws=200)
    assert same["absolute_difference_upper_one_sided95_meters"] == 0
    assert same["relative_increase_upper_one_sided95"] == 0


@pytest.mark.parametrize("failure", ["duplicate", "missing", "unknown", "denominator", "negative", "nonfinite"])
def test_native_row_inventory_and_physical_trajectory_reduction_fail_early(failure):
    rows = [row("a", 1), row("b", 2)]
    groups = {"a": ("A",), "b": ("B",)}
    if failure == "duplicate":
        rows[1] = rows[0]
    if failure == "missing":
        rows.pop()
    if failure == "unknown":
        rows[1]["ids"] = ["unknown"]
    if failure == "denominator":
        rows[1]["metrics"][q.ADE]["count"] = 64
    if failure == "negative":
        rows[1]["metrics"][q.ADE]["sum"] = -1
    if failure == "nonfinite":
        rows[1]["metrics"][q.ADE]["sum"] = float("nan")
    with pytest.raises(ValueError):
        q.block_means(rows, groups, q.ADE)


@pytest.mark.parametrize("failure", ["seed", "block", "zero_reference"])
def test_missing_pairing_or_undefined_relative_comparison_refused(failure):
    before, after = tables(), tables()
    if failure == "seed":
        after.pop(42)
    if failure == "block":
        after[42].pop(("p", "B", 1))
    if failure == "zero_reference":
        before = tables(0)
    with pytest.raises(ValueError):
        q.compare_recipes(after, before, draws=2)


def run(times, *, target=None):
    steps = ([0] if target == 0 else []) + [128, 256, 384, 512]
    return {
        "completion": {
            "step": 512,
            "horizon_reached": True,
            "resumed": False,
            "timing_scope": "uninterrupted",
            "target": None if target is None else {"step": target},
            "history": [{"step": step, "metrics": {q.ADE: 1.0}} for step in q.STEPS],
        },
        "timing": {
            "checkpoint_publications": [
                {"step": step, "seconds": seconds, "checkpoint": f"/run/checkpoints/iter_{step:06d}"}
                for step, seconds in zip(steps, times, strict=True)
            ]
        },
    }


def test_equal_compute_selects_latest_publication_under_earlier_final_and_reports_lag():
    candidate = run([12, 24, 36, 48])
    reference = run([10, 20, 30, 40])
    value = q.equal_compute_endpoints(candidate, reference)
    assert value["budget_seconds"] == 40
    assert value["nf4_lora"]["step"] == 384 and value["nf4_lora"]["publication_lag_seconds"] == 4
    assert value["bf16_lora"]["step"] == 512 and value["bf16_lora"]["publication_lag_seconds"] == 0
    # Scores can favor an earliercheckpoint; the rule never consults them.
    candidate["completion"]["history"][1]["metrics"][q.ADE] = 0.0
    assert q.equal_compute_endpoints(candidate, reference) == value


def test_exact_budget_boundary_included_without_interpolation():
    result = q.equal_compute_endpoints(run([10, 20, 30, 40]), run([10, 20, 30, 40]))
    assert result["nf4_lora"]["step"] == result["bf16_lora"]["step"] == 512
    with pytest.raises(ValueError, match="no published"):
        q.equal_compute_endpoints(run([60, 70, 80, 90]), run([10, 20, 30, 40]))


@pytest.mark.parametrize("failure", ["resume", "missing", "duplicate", "nonfinite", "unordered", "wrong_path"])
def test_partial_or_spliced_publication_clocks_refused(failure):
    value = run([10, 20, 30, 40])
    times = value["timing"]["checkpoint_publications"]
    if failure == "resume":
        value["completion"]["resumed"] = True
    if failure == "missing":
        times.pop()
    if failure == "duplicate":
        times[1] = times[0]
    if failure == "nonfinite":
        times[1]["seconds"] = float("nan")
    if failure == "unordered":
        times[1]["seconds"] = 5
    if failure == "wrong_path":
        times[0]["checkpoint"] = "/other/iter_000999"
    with pytest.raises(ValueError):
        q.publications(value)


def test_initial_target_publication_remains_explicit():
    value = run([1, 10, 20, 30, 40], target=0)
    assert q.publications(value)[0]["step"] == 0


def test_joint_guards_require_absolute_and_relative_for_both_metrics(monkeypatch):
    results = iter(
        [
            {"relative_increase_upper_one_sided95": 0.01, "absolute_difference_upper_one_sided95_meters": 0.101},
            {"relative_increase_upper_one_sided95": 0.01, "absolute_difference_upper_one_sided95_meters": 0.01},
        ]
    )
    monkeypatch.setattr(q, "compare_recipes", lambda *args: next(results))
    value = q.comparison({q.ADE: None, q.FDE: None}, {q.ADE: None, q.FDE: None})
    assert not value["joint_development_preservation_guards_met"]
    assert not value["development_ADE_improvement_upper_bound_below_zero"]


def identity_fixture():
    source = {"format": "physicalai-alpamayo-flow-v1", "sha256": "validation"}
    processor = {"physical_contract": "fixture"}
    target = {
        "threshold": 0.95,
        "source": source,
        "scorer": {"factories": "fixture"},
        "identity": {
            k: {"fixture": k}
            for k in (
                "base",
                "task_contract",
                "precision",
                "dependencies",
                "implementation",
                "targets",
                "auxiliary_parameters",
            )
        },
    }
    target["identity"]["processor_contract"] = processor
    identity = copy.deepcopy(target["identity"])
    identity.update(
        format="cosmos.family-backbone.v1",
        adapter={"mode": "nf4_lora", "rank": 32, "alpha": 64.0, "dropout": 0.0, "seed": 17},
    )
    identity["quantization"] = {
        "backend": "bitsandbytes",
        "backend_version": "0.49.2",
        "format": "nf4",
        "block_size": 64,
        "nested": True,
        "storage_dtype": "uint8",
        "compute_dtype": "bfloat16",
    }
    identity["training_contract"] = {
        "format": "cosmos-native-action-sft-v1",
        "seed": 17,
        "max_steps": 512,
        "evaluation_every": 128,
        "evaluation_seed": 123,
        "evaluation_max_samples": 0,
        "target": [q.ADE, 0.95, "min", False],
        "validation": source,
        "lr_schedule": {"kind": "cosine", "f_start": 0.01, "f_min": 0.01, "cycle_steps": 512},
        "optimizer": {
            "name": "adamw",
            "impl": "fused",
            "decay_groups": "",
            "lr": 1e-4,
            "warmup_steps": 100,
            "grad_clip_norm": 1.0,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "train_keys": [],
            "lr_multipliers": {},
        },
        "objective": {
            "algorithm": "flow_bc",
            "action_dim": 2,
            "training_prefix": 64,
            "micro_batch_size": 1,
            "training_source": {"format": "physicalai-alpamayo-flow-v1", "sha256": "train"},
            "task": processor,
        },
        "factories": target["scorer"],
        "runtime": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "cublas_workspace_config": ":4096:8",
            "optimizer_state_dtype": "float32",
        },
    }
    return identity, target, {"native_sources": {"train": {"format": "physicalai-alpamayo-flow-v1", "sha256": "train"}}}


@pytest.mark.parametrize(
    "failure", [None, "horizon", "optimizer", "objective", "scorer", "runtime", "base", "task", "mode", "quantization"]
)
def test_training_identity_validated_before_frozen_base_comparison(failure):
    identity, target, admitted = identity_fixture()
    if failure == "horizon":
        identity["training_contract"]["max_steps"] = 300
    if failure == "optimizer":
        identity["training_contract"]["optimizer"]["grad_clip_norm"] = 0
    if failure == "objective":
        identity["training_contract"]["objective"]["action_dim"] = 7
    if failure == "scorer":
        identity["training_contract"]["factories"] = {}
    if failure == "runtime":
        identity["training_contract"]["runtime"]["deterministic_warn_only"] = True
    if failure == "base":
        identity["base"] = {"other": "base"}
    if failure == "task":
        identity["training_contract"]["objective"]["task"] = {}
    if failure == "mode":
        identity["adapter"]["mode"] = "bf16_lora"
    if failure == "quantization":
        identity["quantization"]["nested"] = False
    if failure is None:
        q.validate_training_contract(identity, mode="nf4_lora", seed=17, target=target, admitted=admitted)
    else:
        with pytest.raises(ValueError):
            q.validate_training_contract(identity, mode="nf4_lora", seed=17, target=target, admitted=admitted)


def test_does_not_drop_unreached_or_missing_recipe_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "load_groups", lambda *args: ({}, {"native_sources": {"validation": {}}}))
    monkeypatch.setattr(q, "load_target", lambda *args: ({"source": {}}, "target", {}))
    with pytest.raises(ValueError, match="all nine"):
        q.analyze([], validation=tmp_path / "unused", manifest=tmp_path / "unused", target_path=tmp_path / "unused")


def test_manifest_and_validation_byte_mismatch_refused(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")
    with pytest.raises(ValueError, match="exact admitted"):
        q.load_groups(validation, manifest)


def test_admitted_media_receipts_reproduce_actual_native_content_fingerprint(tmp_path):
    import hashlib

    from PIL import Image

    from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset

    folder = tmp_path / "clip"
    folder.mkdir()
    image = folder / "camera.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image)
    sample = {
        "id": "row",
        "episode_id": "episode",
        "observation": {},
        "actions": [[1.0, 2.0]],
        "media": {"camera": [str(image)]},
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    admitted = {
        "jsonl_sha256": {"train": hashlib.sha256(path.read_bytes()).hexdigest()},
        "clip_receipts": [
            {
                "split": "train",
                "row_id": "row",
                "episode_id": "episode",
                "clip_id": "clip",
                "receipt": {"files": {"camera.png": hashlib.sha256(image.read_bytes()).hexdigest()}},
            }
        ],
    }
    reconstructed = q.admitted_source(path, admitted, "train")
    assert reconstructed["sha256"] == FlowSFTDataset(str(path)).fingerprint
    assert reconstructed["sha256"] != admitted["jsonl_sha256"]["train"]


@pytest.mark.parametrize("crossing", [None, 0, 128])
def test_native_target_shape_ingestion_retains_unreached_and_initial_crossing(tmp_path, monkeypatch, crossing):
    from cosmos_framework.posttrain.checkpoint import adapter

    identity, target, admitted = identity_fixture()
    value = run([1, 10, 20, 30, 40] if crossing == 0 else [10, 20, 30, 40], target=crossing)
    value.update(
        complete=True,
        mode="nf4_lora",
        seed=17,
        target_sha256="fixed-target",
        memory_evidence={"samples": 10, "complete_host_samples": 10, "positive_owned_gpu_observations": 4},
    )
    value["completion"].update(training_source=admitted["native_sources"]["train"], validation_source=target["source"])
    for entry in value["completion"]["history"]:
        entry["metrics"][q.ADE] = 0.9 if crossing is not None and entry["step"] >= crossing else 1.1
    publication = (
        None
        if crossing is None
        else next(row for row in value["timing"]["checkpoint_publications"] if row["step"] == crossing)
    )
    value["completion"]["target"] = (
        None
        if publication is None
        else {
            "step": crossing,
            "checkpoint": publication["checkpoint"],
            "seconds": publication["seconds"],
            "timing_scope": "uninterrupted",
        }
    )
    value["timing"].update(
        unreached=crossing is None,
        first_target_publication_seconds=None if publication is None else publication["seconds"],
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value))
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {
                "protocol_sha256": q.PROTOCOL_SHA,
                "mode": "nf4_lora",
                "seed": 17,
                "implementation": q.CODE,
                "wheel_sha256": q.WHEEL,
            }
        )
    )
    (tmp_path / "workflow.json").write_text(
        json.dumps({"monitor_alive_at_end": True, "monitor_exit": 0, "elapsed_seconds": 45.0})
    )
    monkeypatch.setattr(adapter, "read_adapter_metadata", lambda *args, **kwargs: {"identity": identity})
    ingested, _ = q.load_run(path, target=target, target_sha="fixed-target", admitted=admitted)
    assert ingested["timing"]["unreached"] is (crossing is None)
    value["timing"]["first_target_publication_seconds"] = 42.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="clock or censoring"):
        q.load_run(path, target=target, target_sha="fixed-target", admitted=admitted)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "factor",
        "seed",
        "evaluator",
        "reference_bytes",
        "aggregate",
        "incomplete",
        "partial_rows",
        "monitor_dead",
        "monitor_exit",
        "workflow_time",
    ],
)
def test_frozen_target_requires_exact_calibration_metric_and_completion(tmp_path, failure):
    import hashlib

    groups = {"a": ("A",), "b": ("B",)}
    reference = {
        "artifact_kind": "fresh-frozen-base",
        "identity": {"adapter": {"mode": "bf16_frozen"}},
        "source": {"sha256": "native"},
        "scorer": {"factory": "bound"},
        "metrics": {q.ADE: 1.0, q.FDE: 2.0},
        "rows": [row("a", 1.0), row("b", 1.0)],
    }
    if failure == "aggregate":
        reference["metrics"][q.ADE] = 2.0
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(reference))
    target = {
        "format": "native-family-frozen-base-target-v1",
        "family": "alpamayo",
        "metric": q.ADE,
        "factor": 0.95,
        "direction": "min",
        "seed": 18 if failure == "seed" else 17,
        "evaluation_seed": 123,
        "evaluation_entrypoint_sha256": "changed" if failure == "evaluator" else q.CALIBRATION_EVALUATOR_SHA,
        "protocol_sha256": q.PROTOCOL_SHA,
        "cohort_manifest_sha256": q.MANIFEST_SHA,
        "build": {"implementation_sha": q.CODE, "wheel_sha256": q.WHEEL},
        "baseline": reference["metrics"][q.ADE],
        "threshold": 0.95 * reference["metrics"][q.ADE],
        "reference_evaluation": str(path),
        "reference_evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **{key: reference[key] for key in ("identity", "source", "scorer")},
    }
    if failure == "factor":
        target["factor"] = 0.96
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target))
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "complete": failure != "incomplete",
                "family": "alpamayo",
                "mode": "bf16_frozen",
                "target_sha256": digest,
                "scored_rows": 1 if failure == "partial_rows" else len(groups),
            }
        )
    )
    (tmp_path / "workflow.json").write_text(
        json.dumps(
            {
                "monitor_was_alive_at_end": failure != "monitor_dead",
                "monitor_exit": 1 if failure == "monitor_exit" else 0,
                "elapsed_seconds": 0 if failure == "workflow_time" else 3.0,
            }
        )
    )
    if failure == "reference_bytes":
        path.write_text("{}")
    if failure is None:
        assert q.load_target(target_path, groups)[1] == digest
    else:
        with pytest.raises((ValueError, KeyError)):
            q.load_target(target_path, groups)


def test_complete_nine_run_analysis_keeps_both_endpoints_and_unreached(tmp_path, monkeypatch):
    """Synthetic fixtures exercise ingestion; they are never promoted as experiment evidence."""
    from cosmos_framework.posttrain.checkpoint import adapter

    base_identity, target, admitted = identity_fixture()
    admitted["native_sources"]["validation"] = target["source"]
    groups = {"a": ("platform", "A", 1), "b": ("platform", "B", 1)}
    target["baseline"] = 1.0
    baseline = {"metrics": {q.ADE: 1.0, q.FDE: 2.0}}
    monkeypatch.setattr(q, "load_groups", lambda *args: (groups, admitted))
    monkeypatch.setattr(q, "load_target", lambda *args: (target, "fixed-target", baseline))
    metadata_by_checkpoint = {}
    paths = []
    for mode in q.MODES:
        for seed in q.SEEDS:
            folder = tmp_path / f"{mode}-{seed}"
            folder.mkdir()
            multiplier = {"nf4_lora": 1.01, "bf16_lora": 1.0, "bf16_frozen": 1.2}[mode]
            times = [12, 24, 36, 48] if mode == "nf4_lora" else [10, 20, 30, 40]
            result = run(times)
            result.update(
                complete=True,
                mode=mode,
                seed=seed,
                target_sha256="fixed-target",
                memory_evidence={"samples": 10, "complete_host_samples": 10, "positive_owned_gpu_observations": 4},
            )
            completion = result["completion"]
            completion.update(
                training_source=admitted["native_sources"]["train"],
                validation_source=target["source"],
                clocks={},
                workflow_seconds=55.0,
            )
            identity = copy.deepcopy(base_identity)
            identity["adapter"].update(mode=mode, seed=seed)
            identity["training_contract"]["seed"] = seed
            if mode != "nf4_lora":
                identity["quantization"] = None
            for entry in completion["history"]:
                step = entry["step"]
                number = (1.2 if seed == 123 or step == 0 else 0.8) * multiplier
                entry["metrics"] = {q.ADE: number, q.FDE: 2 * number}
                (folder / f"evaluation-{step:06d}.json").write_text(
                    json.dumps(
                        {"step": step, "metrics": entry["metrics"], "rows": [row("a", number), row("b", number)]}
                    )
                )
            for publication in result["timing"]["checkpoint_publications"]:
                publication["checkpoint"] = str(folder / "checkpoints" / f"iter_{publication['step']:06d}")
                metadata_by_checkpoint[publication["checkpoint"]] = {"identity": identity}
            attained = next(
                (entry for entry in completion["history"] if entry["metrics"][q.ADE] <= target["threshold"]), None
            )
            publication = (
                None
                if attained is None
                else next(row for row in result["timing"]["checkpoint_publications"] if row["step"] == attained["step"])
            )
            completion["target"] = (
                None
                if publication is None
                else {
                    "step": attained["step"],
                    "checkpoint": publication["checkpoint"],
                    "seconds": publication["seconds"],
                    "timing_scope": "uninterrupted",
                }
            )
            result["timing"].update(
                unreached=publication is None,
                first_target_publication_seconds=None if publication is None else publication["seconds"],
            )
            path = folder / "result.json"
            path.write_text(json.dumps(result))
            paths.append(path)
            (folder / "protocol.json").write_text(
                json.dumps(
                    {
                        "protocol_sha256": q.PROTOCOL_SHA,
                        "mode": mode,
                        "seed": seed,
                        "implementation": q.CODE,
                        "wheel_sha256": q.WHEEL,
                    }
                )
            )
            (folder / "workflow.json").write_text(
                json.dumps({"monitor_alive_at_end": True, "monitor_exit": 0, "elapsed_seconds": 60.0})
            )
    monkeypatch.setattr(adapter, "read_adapter_metadata", lambda path, **kwargs: metadata_by_checkpoint[str(path)])
    (tmp_path / "workflow.json").write_text(json.dumps({"elapsed_seconds": 5.0}))
    result = q.analyze(
        paths, validation=tmp_path / "unused", manifest=tmp_path / "unused", target_path=tmp_path / "target.json"
    )
    assert len(result["input_receipts"]) == 9
    assert len(result["individual_target_runtime_memory"]) == 9
    assert (
        result["paired_time_to_target"][-1]["nf4_unreached"] and result["paired_time_to_target"][-1]["bf16_unreached"]
    )
    assert result["equal_compute_publications_and_lags"][0]["nf4_lora"]["step"] == 384
    assert result["fixed512_nf4_vs_bf16_lora"][q.ADE]["draws"] == 10000
    assert result["equal_elapsed_nf4_vs_bf16_lora"][q.ADE]["bootstrap_seed"] == 70241
    assert result["independent_confirmation"]["established"] is False
    json.dumps(result, allow_nan=False)
