# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reasoner SFT → native DCP checkpoint → held-out reasoner evaluation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import tomli_w
from automl_core.contracts import Contract, Execution, Result, Search, TrialPlan, lookup
from automl_core.recovery import attempt_root
from automl_core.runner import Experiment
from automl_core.workload import local_artifact, read_manifest
from pydantic import Field

from cosmos_framework.automl.evaluate_trial import read_toml, validate_dcp
from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig

# These knobs are consumed by both Framework's TOML schema and its VLM remapping.
# Fidelity, topology, data, identity and checkpoint lineage are not search axes.
SEARCHABLE = {
    "optimizer.lr",
    "optimizer.weight_decay",
    "trainer.grad_accum_iter",
    "dataloader_train.max_samples_per_batch",
}


class SFTWorkload:
    def __init__(
        self,
        cwd: Path,
        evaluation: dict,
        nproc: int,
        seed: int,
        experiment_id: str,
        expected_validation_records: int | None = None,
    ):
        self.cwd = cwd
        self.evaluation = evaluation
        self.nproc = nproc
        self.seed = seed
        self.experiment_id = experiment_id
        self.expected_validation_records = expected_validation_records

    def identity(self) -> dict:
        return {
            "adapter": "cosmos-reasoner-sft-v1",
            "cwd": str(self.cwd),
            "evaluation": self.evaluation,
            "nproc": self.nproc,
            "training_seed": self.seed,
            "expected_validation_records": self.expected_validation_records,
        }

    def validate(self, config: dict) -> None:
        parsed = SFTExperimentConfig.model_validate(config)
        if parsed.job.task != "vlm" or not parsed.job.experiment:
            raise ValueError("Reasoner AutoML needs a VLM SFT recipe with an explicit job.experiment")
        if "lora_enabled" not in config.get("model", {}):
            raise ValueError("Set model.lora_enabled explicitly so checkpoint evaluation matches training")
        if parsed.trainer.num_epochs is not None:
            raise ValueError("This adapter compares a fixed trainer.max_iter budget; remove num_epochs")
        maximum = config.get("trainer", {}).get("max_iter", 0)
        interval = config.get("checkpoint", {}).get("save_iter", 0)
        if maximum <= 0 or interval <= 0 or maximum % interval:
            raise ValueError("Set positive max_iter and save_iter with a checkpoint at the final iteration")
        if parsed.checkpoint.save_freq_in_epoch:
            raise ValueError("Use iteration-based checkpointing for a fixed max_iter search")
        if "load_path" not in config.get("checkpoint", {}) or "???" in json.dumps(config):
            raise ValueError("Resolve the initial checkpoint and all recipe placeholders before planning")
        # Native VLM initialization loads HF backbone weights when load_path is
        # explicitly empty; a nonempty value instead selects a DCP warm start.
        # Do not force a safetensors directory through the DCP loader.
        if not parsed.checkpoint.load_path and not config.get("model", {}).get("backbone", {}).get("model_name"):
            raise ValueError(
                "HF initialization needs an explicit model.backbone.model_name and checkpoint.load_path=''"
            )
        if parsed.optimizer.lr <= 0 or parsed.optimizer.weight_decay < 0:
            raise ValueError("Learning rate must be positive; weight decay must be non-negative")
        batch_size = parsed.dataloader_train.max_samples_per_batch
        if parsed.trainer.grad_accum_iter < 1 or (batch_size is not None and batch_size < 1):
            raise ValueError("Batch size and gradient accumulation must be positive")
        if self.evaluation.get("task", {}).get("type") not in {
            "its_directionality",
            "metropolis_sgd",
            "mcq",
            "bcq",
            "binary",
        }:
            raise ValueError("Select a scored reasoner task: mcq, bcq, binary, its_directionality or metropolis_sgd")
        if not self.evaluation.get("dataset", {}).get("annotation_path"):
            raise ValueError("An explicit held-out evaluation dataset is required")
        if parsed.model.lora_enabled and not self.evaluation.get("model", {}).get("base_model_path"):
            raise ValueError("LoRA evaluation needs model.base_model_path for native checkpoint preparation")

    def validate_search(self, search) -> None:
        if search.fidelity:
            if search.base_config["trainer"]["max_iter"] != search.fidelity.maximum:
                raise ValueError(
                    "Framework fidelity uses cumulative optimizer steps; maximum must equal trainer.max_iter"
                )
            cycles = search.base_config.get("scheduler", {}).get("cycle_lengths")
            if not cycles or any(type(c) is not int or c <= 0 for c in cycles) or sum(cycles) < search.fidelity.maximum:
                raise ValueError(
                    "Set explicit scheduler.cycle_lengths covering the full fidelity.maximum horizon; "
                    "stage limits must not shorten the LR schedule"
                )
            algorithms = {search.algorithm}
            if search.llm:
                algorithms.update(search.llm.phase_algorithms)
            if "pbt" in algorithms and set(search.parameters) - {"optimizer.lr", "optimizer.weight_decay"}:
                raise ValueError(
                    "Framework PBT mutates optimizer.lr/weight_decay only; preserve batch/accumulation and data-loader checkpoint semantics"
                )
            if search.base_config.get("checkpoint", {}).get("keys_to_skip_loading"):
                raise ValueError("Promotions must restore the complete native checkpoint; remove keys_to_skip_loading")

    def plan_stage(self, config, trial, execution, stage, parent) -> TrialPlan:
        recipe = copy.deepcopy(config)
        resource = stage["resource"]
        recipe["trainer"]["max_iter"] = resource
        recipe["checkpoint"].update(save_iter=resource, dcp_async_mode_enabled=False)
        recipe["trainer"].setdefault("callbacks", {}).setdefault("workflow_status", {})["enabled"] = True
        checkpoint = None
        if parent:
            checkpoint = local_artifact(attempt_root(trial.parent, parent), parent["result"]["artifacts"]["checkpoint"])
            validate_dcp(checkpoint, components=("model", "optim", "scheduler", "trainer"))
            if checkpoint.name != f"iter_{parent['resource']:09d}":
                raise ValueError("Parent checkpoint iteration does not match recorded resource")
            recipe["checkpoint"]["load_path"] = str(checkpoint)
        plan = self.plan(recipe, trial, execution)
        command = plan.commands[0]
        command[command.index("checkpoint.load_training_state=false")] = (
            f"checkpoint.load_training_state={'true' if parent else 'false'}"
        )
        if parent:
            # Preset warm-start exclusions must not turn a promotion into a
            # partial restore even when the TOML did not explicitly name them.
            command += [
                "checkpoint.keys_not_to_resume=[]",
                "checkpoint.keys_to_skip_loading=[]",
                "checkpoint.strict_resume=true",
            ]
        command += [
            "++trainer.callbacks.automl_stage._target_=cosmos_framework.automl.stage_callback.StageCallback",
            f"++trainer.callbacks.automl_stage.stage_file={json.dumps(str(trial / 'stage.json'))}",
        ]
        # Full-horizon scheduler settings are retained from the base recipe.
        # Native optimizer/scheduler containers retain the current recipe's
        # per-group LR/WD while restoring progress; the callback verifies this.
        plan.files["stage.json"] = json.dumps(
            {
                "resource": resource,
                "source_trial": parent["id"] if parent else None,
                "source_checkpoint": str(checkpoint) if checkpoint else None,
                "expected_start_iteration": parent["resource"] if parent else 0,
                "resume_mode": stage.get("resume_mode", "fresh"),
            }
        )
        if checkpoint:
            plan.read_only_mounts.append(checkpoint)
        return plan

    def plan(self, config: dict, trial: Path, execution: Execution) -> TrialPlan:
        if execution.kind in {"kubernetes", "brev", "lepton"}:
            if getattr(execution, execution.kind).gpus != self.nproc:
                raise ValueError("Managed Framework trials require one allocated GPU per training rank")
        recipe = copy.deepcopy(config)
        recipe.setdefault("job", {}).update(project="automl", group=self.experiment_id, name=trial.name)
        recipe["job"].setdefault("wandb_mode", "disabled")
        recipe.setdefault("trainer", {}).setdefault("callbacks", {}).setdefault("workflow_status", {})["enabled"] = True
        output = trial / "train"
        run_dir = output / "automl" / self.experiment_id / trial.name
        evaluation = copy.deepcopy(self.evaluation)
        evaluation.setdefault("model", {})["enable_lora"] = recipe.get("model", {}).get("lora_enabled", False)
        command = [execution.python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={self.nproc}"]
        command += ["-m", "cosmos_framework.scripts.train", f"--sft-toml={trial / 'config.toml'}"]
        command += [
            "--",
            f"trainer.seed={self.seed}",
            "trainer.num_epochs=null",
            "checkpoint.save_freq_in_epoch=0",
            "checkpoint.load_training_state=false",
        ]
        return TrialPlan(
            commands=[
                command,
                [
                    execution.python,
                    "-m",
                    "cosmos_framework.automl.evaluate_trial",
                    "--run-dir",
                    str(run_dir),
                    "--evaluation-config",
                    str(trial / "evaluation.toml"),
                    "--output-dir",
                    str(trial / "evaluation"),
                    "--expected-iteration",
                    str(recipe["trainer"]["max_iter"]),
                ],
            ],
            cwd=self.cwd,
            files={"config.toml": tomli_w.dumps(recipe), "evaluation.toml": tomli_w.dumps(evaluation)},
            environment={"IMAGINAIRE_OUTPUT_ROOT": str(output), "PYTHONHASHSEED": str(self.seed)},
            progress_files={0: [str(run_dir.relative_to(trial) / "status.json")], 1: ["evaluation/status.json"]},
        )

    def collect(self, trial: Path, objective) -> Result:
        metrics_path = local_artifact(trial, "evaluation/metrics/metrics.json")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("overall", {}).get("total", 0) <= 0:
            raise ValueError("An empty evaluation is not an observation")
        stage_path = trial / "stage.json"
        if self.expected_validation_records is not None:
            if metrics["overall"]["total"] != self.expected_validation_records:
                raise ValueError("Evaluation did not score the expected full validation dataset")
        if self.expected_validation_records is not None or stage_path.exists():
            run = trial / "train" / "automl" / self.experiment_id / trial.name
            status = local_artifact(trial, str(run / "status.json"))
            lines = [line for line in status.read_text().splitlines() if line.strip()]
            if not lines or json.loads(lines[-1]).get("status") != "SUCCESS":
                raise ValueError("Native training status is not SUCCESS")
        value = lookup(metrics, objective.metric)
        if type(value) not in (float, int):
            raise ValueError("The objective must be a finite JSON number")
        handoff = json.loads(local_artifact(trial, "evaluation/checkpoint.json").read_text())
        checkpoint = local_artifact(trial, handoff["checkpoint"])
        validate_dcp(checkpoint)
        if stage_path.exists():
            stage = json.loads(stage_path.read_text())
            if checkpoint.name != f"iter_{stage['resource']:09d}":
                raise ValueError("Stage result is not at the requested cumulative optimizer step")
            for rank in range(self.nproc):
                receipt = json.loads(local_artifact(trial, f"resume-rank-{rank}.json").read_text())
                if (
                    receipt.get("iteration") != stage["expected_start_iteration"]
                    or receipt.get("source_checkpoint") != stage["source_checkpoint"]
                ):
                    raise ValueError("Native checkpoint restore receipt disagrees with the stage lineage")
                if stage["resume_mode"] == "pbt" and not receipt.get("optimizer_configuration_verified"):
                    raise ValueError("PBT optimizer configuration was not verified after restoring donor state")
        return Result(
            metric=value,
            artifacts={
                "checkpoint": str(checkpoint),
                "config": str(local_artifact(trial, "config.toml")),
                "metrics": str(metrics_path),
            },
        )


class WorkloadSettings(Contract):
    recipe: Path
    evaluation: Path
    cwd: Path
    nproc_per_node: int = Field(ge=1, strict=True)
    expected_validation_records: int | None = Field(default=None, gt=0, strict=True)


class Manifest(Contract):
    search: Search
    workload: WorkloadSettings
    execution: Execution


def load(path: Path, workspace: Path) -> Experiment:
    raw = read_manifest(path)
    if "base_config" in raw.get("search", {}):
        raise ValueError("Use workload.recipe as the single source of the Framework base configuration")
    recipe_path = (path.parent / raw["workload"]["recipe"]).resolve()
    raw["search"]["base_config"] = read_toml(recipe_path)
    manifest = Manifest.model_validate(raw)
    unsupported = set(manifest.search.parameters) - SEARCHABLE
    if unsupported:
        raise ValueError(f"Search axes need adapter support: {sorted(unsupported)}; supported: {sorted(SEARCHABLE)}")
    evaluation = read_toml((path.parent / manifest.workload.evaluation).resolve())
    # Hydra reads these only inside the selected execution environment; persist names, not values.
    required = set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)[^}]*\}", json.dumps(raw)))
    if not required.issubset(manifest.execution.environment):
        raise ValueError(f"Declare recipe environment names in execution.environment: {sorted(required)}")
    # These are both host source and container destination names. Resolving a
    # shared-storage symlink (e.g. /shared/fs -> /shared/fs2) would leave the
    # recipe's declared /shared/fs paths unmounted inside the container.
    manifest.execution.mounts = [(path.parent / p).absolute() for p in manifest.execution.mounts]
    if manifest.execution.kind == "slurm":
        if not manifest.execution.image or manifest.execution.slurm.gpus != manifest.workload.nproc_per_node:
            raise ValueError("Framework SLURM trials require a prepared image and one GPU per training rank")
        if not manifest.execution.slurm.record_open:
            raise ValueError("Framework SLURM trials require the existing job-record writer's open/mark hooks")
    workload = SFTWorkload(
        (path.parent / manifest.workload.cwd).resolve(),
        evaluation,
        manifest.workload.nproc_per_node,
        manifest.search.seed,
        manifest.search.experiment_id,
        manifest.workload.expected_validation_records,
    )
    return Experiment(manifest.search, workload, manifest.execution, workspace)
