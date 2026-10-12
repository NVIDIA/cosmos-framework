# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only adapter contracts; not a training/evaluation integration test."""

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomli_w
import yaml
from automl_core.contracts import Execution, Objective

from cosmos_framework.automl.evaluate_trial import prepare, validate_dcp
from cosmos_framework.automl.workload import SFTWorkload, load
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recipe():
    return {
        "job": {"task": "vlm", "experiment": "pre_exp012_llava_ov_mapstyle_dataloader", "wandb_mode": "disabled"},
        "optimizer": {"lr": 1e-5},
        "model": {"lora_enabled": False},
        "trainer": {"max_iter": 20},
        "checkpoint": {"load_path": "/assets/base", "save_iter": 20},
    }


@pytest.fixture
def workload():
    return SFTWorkload(
        ROOT, {"task": {"type": "mcq"}, "dataset": {"annotation_path": "/assets/validation.json"}}, 2, 42, "demo"
    )


def dcp(run):
    checkpoint = run / "checkpoints/iter_000000020"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata fixture")
    (model / "__0_0.distcp").write_bytes(b"shard fixture")
    (run / "checkpoints/latest_checkpoint.txt").write_text(checkpoint.name)
    (run / "config.yaml").write_text("model: {}\n")
    return checkpoint


def test_plan_native_commands_and_isolation(tmp_path, recipe, workload):
    workload.validate(recipe)
    before = copy.deepcopy(recipe)
    trial = tmp_path / "trial-00000"
    plan = workload.plan(recipe, trial, Execution(kind="venv", python=sys.executable))
    assert recipe == before
    assert "cosmos_framework.scripts.train" in plan.commands[0]
    assert "--nproc-per-node=2" in plan.commands[0]
    assert "trainer.seed=42" in plan.commands[0]
    assert "checkpoint.load_training_state=false" in plan.commands[0]
    assert "cosmos_framework.automl.evaluate_trial" in plan.commands[1]
    assert plan.environment["IMAGINAIRE_OUTPUT_ROOT"] == str(trial / "train")
    assert not trial.exists()
    assert "optimizer.lr=1e-05" in build_hydra_overrides(recipe)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("job", "task", "vfm"),
        ("trainer", "num_epochs", 2),
        ("trainer", "max_iter", 21),
        ("checkpoint", "save_freq_in_epoch", 1),
        ("checkpoint", "load_path", "???"),
        ("optimizer", "lr", -1),
        ("trainer", "grad_accum_iter", 0),
    ],
)
def test_invalid_configs(recipe, workload, section, key, value):
    recipe.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        workload.validate(recipe)


def test_prepare_native_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = dcp(run)
    evaluation, selected = prepare(run, {"dataset": {"annotation_path": "validation.json"}}, tmp_path / "eval", 20)
    assert selected == checkpoint
    assert evaluation["model"]["config_file"] == str(run / "config.yaml")
    assert evaluation["model"]["model_name"] == str(checkpoint)
    assert evaluation["model"]["save_folder"] == "metrics"
    assert evaluation["evaluation"]["skip_saved"] is False
    with pytest.raises(ValueError, match="budget"):
        prepare(run, {}, tmp_path / "eval", 40)
    (run / "checkpoints/latest_checkpoint.txt").write_text("../../elsewhere")
    with pytest.raises(ValueError, match="direct child"):
        prepare(run, {}, tmp_path / "eval", 20)


def test_result_requires_nonempty_eval_and_checkpoint(tmp_path, workload):
    trial = tmp_path / "trial"
    checkpoint = dcp(trial / "train/automl/demo/trial")
    metrics = trial / "evaluation/metrics/metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"overall": {"total": 10, "accuracy": 0.7}}))
    (trial / "evaluation/checkpoint.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
    (trial / "config.toml").write_text("[job]\n")
    objective = Objective(metric="overall.accuracy", direction="maximize")
    assert workload.collect(trial, objective).metric == 0.7
    metrics.write_text(json.dumps({"overall": {"total": 0, "accuracy": 0.7}}))
    with pytest.raises(ValueError, match="empty"):
        workload.collect(trial, objective)
    (checkpoint / "model/__0_0.distcp").write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        validate_dcp(checkpoint)


def test_manifest_embeds_sources_and_rejects_unsafe_axes(tmp_path, recipe, workload):
    (tmp_path / "recipe.toml").write_text(tomli_w.dumps(recipe))
    (tmp_path / "evaluation.toml").write_text(tomli_w.dumps(workload.evaluation))
    manifest = {
        "search": {
            "experiment_id": "demo",
            "seed": 42,
            "objective": {"metric": "overall.accuracy", "direction": "maximize"},
            "budget": {"max_trials": 2, "trial_timeout_seconds": 30, "total_timeout_seconds": 60},
            "parameters": {"optimizer.lr": {"kind": "float", "low": 1e-6, "high": 1e-4, "log": True}},
        },
        "workload": {"recipe": "recipe.toml", "evaluation": "evaluation.toml", "cwd": str(ROOT), "nproc_per_node": 2},
        "execution": {"kind": "venv", "python": sys.executable},
    }
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(manifest))
    experiment = load(path, tmp_path / "runs")
    plan = experiment.plan()
    assert plan["identity"]["search"]["base_config"] == recipe
    assert not (tmp_path / "runs").exists()
    manifest["search"]["parameters"] = {"trainer.max_iter": {"kind": "int", "low": 20, "high": 40}}
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="adapter support"):
        load(path, tmp_path / "runs")


def test_import_is_cpu_only():
    code = "import sys; import cosmos_framework.automl.workload; assert 'torch' not in sys.modules; assert 'tao_automl' not in sys.modules; assert not any(x.startswith('nvidia_tao') for x in sys.modules)"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_discovery_and_documentation_links():
    canonical = ROOT / ".agents/skills/cosmos-automl/SKILL.md"
    mirror = ROOT / ".claude/skills/cosmos-automl/SKILL.md"
    assert mirror.resolve() == canonical
    assert yaml.safe_load(canonical.read_text().split("---", 2)[1])["name"] == "cosmos-automl"
    for source in [canonical, ROOT / "docs/automl.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
            assert (source.parent / target).exists(), target


def test_packaged_planning_example(tmp_path):
    experiment = load(ROOT / "examples/automl/reasoner.yaml", tmp_path / "runs")
    plan = experiment.plan()
    assert plan["identity"]["search"]["objective"]["direction"] == "maximize"
    assert not (tmp_path / "runs").exists()
