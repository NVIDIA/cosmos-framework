# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from cosmos_framework.scripts.prepare_student_config import _add_missing_defaults, main, prepare_student_model

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _saved_model() -> dict[str, Any]:
    return {
        "_target_": "historical.SelfForcingDMDModel",
        "_recursive_": False,
        "config": {
            "_type": "historical.SelfForcingDMDConfig",
            "video_temporal_causal": True,
            "transfer_control_attention_mode": "causal_control_with_rgb_history",
            "teacher_forcing_frames_per_chunk": 1,
            "parallelism": {"context_parallel_size": 3},
            "compile": {"enabled": True},
            "fixed_step_sampler_config": {"sample_type": "sde", "t_list": [1.0, 0.9375, 0.625]},
            "kv_cache_inference_size": 51,
            "kv_cache_dtype": None,
            "simulation_mode": "self_forcing",
            "teacher_load_from": {"load_path": "private-teacher"},
        },
    }


def test_missing_defaults_preserve_customized_nested_values() -> None:
    saved = {"nested": {"_type": "Config", "null": None, "false": False, "zero": 0, "list": [3, 1]}}
    defaults = {"nested": {"_type": "Config", "null": 9, "false": True, "zero": 5, "list": [0], "new": {"x": 2}}}
    additions: dict[str, Any] = {}
    skipped: list[str] = []
    original = copy.deepcopy(saved)
    _add_missing_defaults(saved, defaults, additions, skipped)

    assert saved["nested"] == {**original["nested"], "new": {"x": 2}}
    assert additions == {"nested.new": {"x": 2}}
    assert skipped == []
    saved["nested"]["new"]["x"] = 8
    assert defaults["nested"]["new"]["x"] == 2
    assert additions["nested.new"]["x"] == 2


@pytest.mark.parametrize(
    "saved_class,default_class",
    [
        ({"_type": "Custom"}, {"_type": "Default"}),
        ({"_target_": "Custom"}, {"_target_": "Default"}),
        ({"_target_": "Custom"}, {"_type": "Default"}),
        ({"_target_": "Same", "_type": "Custom"}, {"_target_": "Same", "_type": "Default"}),
        ({"_type": "Custom"}, {}),
        ({}, {"_type": "Default"}),
    ],
)
def test_different_config_classes_skip_entire_subtree(
    saved_class: dict[str, str], default_class: dict[str, str]
) -> None:
    saved = {"tokenizer": {**saved_class, "custom": False}, "sibling": {}}
    defaults = {"tokenizer": {**default_class, "unrelated": 42}, "sibling": {"needed": 1}}
    original = copy.deepcopy(saved)
    additions: dict[str, Any] = {}
    skipped: list[str] = []
    _add_missing_defaults(saved, defaults, additions, skipped)

    assert saved["tokenizer"] == original["tokenizer"]
    assert saved["sibling"] == {"needed": 1}
    assert additions == {"sibling.needed": 1}
    assert skipped == ["tokenizer"]


def test_preparation_projects_migrates_and_reports_defaults_without_mutating_input() -> None:
    saved = _saved_model()
    original = copy.deepcopy(saved)
    model, report = prepare_student_model(saved)
    config = model["config"]

    assert saved == original
    assert model["_target_"].endswith(".OmniMoTCausalModel")
    assert model["_recursive_"] is False
    assert config["teacher_forcing_replay_policy"]["control_visibility"] == "causal"
    assert config["teacher_forcing_replay_policy"]["controls_read_strict_past_clean_rgb"] is True
    assert config["teacher_forcing_kv_implementation"] == "singleview_threeway_kv"
    assert config["teacher_forcing_frames_per_chunk"] == 1
    assert config["parallelism"]["context_parallel_size"] == 3
    assert config["parallelism"]["vae_load_balance_group_size"] == 1
    assert config["fixed_step_sampler_config"] == saved["config"]["fixed_step_sampler_config"]
    assert config["kv_cache_dtype"] is None
    assert config["compile"]["enabled"] is False
    assert report["original_target"] == saved["_target_"]
    assert report["projected_target"] == model["_target_"]
    assert report["dropped_config_fields"] == [
        "simulation_mode",
        "teacher_load_from",
        "transfer_control_attention_mode",
    ]
    assert report["projection_changes"]["config.compile.enabled"] == {
        "operation": "changed",
        "before": True,
        "after": False,
    }
    # Defaults the projection did not produce are filled afterwards and reported separately.
    projected_policy = {
        key: value
        for key, value in config["teacher_forcing_replay_policy"].items()
        if f"teacher_forcing_replay_policy.{key}" not in report["added_defaults"]
    }
    assert report["projection_changes"]["config.teacher_forcing_replay_policy"]["after"] == projected_policy
    assert "teacher_forcing_replay_policy" not in report["added_defaults"]
    assert report["added_defaults"]["parallelism.vae_load_balance_group_size"] == 1
    assert "private-teacher" not in json.dumps(report)


def test_preparation_rejects_conflicting_legacy_and_current_replay_settings() -> None:
    saved = _saved_model()
    saved["config"]["teacher_forcing_replay_policy"] = {"control_visibility": "global"}
    with pytest.raises(ValueError, match="conflicts with teacher_forcing_replay_policy.control_visibility"):
        prepare_student_model(saved)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_cli_extracts_model_and_records_exact_source_hash(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "saved.yaml"
    output = tmp_path / "prepared" / f"student{suffix}"
    report_path = tmp_path / "report.json"
    original = yaml.safe_dump({"model": _saved_model(), "trainer": {"ignored": True}}).encode()
    source.write_bytes(original)
    main(["--config-file", str(source), "--output-config", str(output), "--report", str(report_path)])

    prepared = yaml.safe_load(output.read_text())
    report = json.loads(report_path.read_text())
    assert list(prepared) == ["model"]
    assert report["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original
    assert report["input_config"] == str(source.resolve())
    assert prepared["model"]["config"]["parallelism"]["vae_load_balance_group_size"] == 1


def test_cli_does_not_overwrite_source_config(tmp_path: Path) -> None:
    source = tmp_path / "saved.yaml"
    source.write_text("model: {}\n")
    with pytest.raises(ValueError, match="must be distinct"):
        main(["--config-file", str(source), "--output-config", str(source), "--report", str(tmp_path / "report.json")])
    assert source.read_text() == "model: {}\n"


def test_preparation_does_not_directly_import_the_distillation_stack() -> None:
    syntax_tree = ast.parse(Path(__file__).with_name("prepare_student_config.py").read_text())
    imported_modules = {
        node.module for node in ast.walk(syntax_tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("cosmos_framework.model.generator.distillation") for module in imported_modules)
