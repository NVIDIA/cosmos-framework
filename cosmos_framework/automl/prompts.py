# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Evaluation-only GEPA adapter using native Framework annotations and metrics."""

import json
import re
from pathlib import Path
from typing import Literal

import tomli_w
from automl_core.contracts import Contract, Execution, TrialPlan, fingerprint, lookup, override
from automl_core.prompt_runner import PromptExperiment, PromptSettings
from automl_core.workload import local_artifact, read_manifest
from pydantic import Field

from cosmos_framework.automl.evaluate_trial import read_toml
from cosmos_framework.automl.prompt_metrics import aggregate


class Dataset(Contract):
    annotations: Path
    media_dir: Path


class PromptWorkloadSettings(Contract):
    evaluation: Path
    cwd: Path
    reflection: Dataset
    validation: Dataset
    test: Dataset | None = None
    metric: Literal["accuracy", "macro_f1"] = "accuracy"
    provenance: dict[str, str] = Field(default_factory=dict)


class Manifest(Contract):
    prompts: PromptSettings
    workload: PromptWorkloadSettings
    execution: Execution


def read_items(dataset, root, task):
    path = (root / dataset.annotations).resolve()
    media = (root / dataset.media_dir).resolve()
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload["items"]
    items = []
    for index, record in enumerate(records):
        if record.get("task", record.get("task_type", task)) != task:
            raise ValueError("Prompt optimization currently requires one consistent scored task per experiment")
        if "conversations" in record:
            turns = record["conversations"]
            if len(turns) != 2:
                raise ValueError("Prompt annotations must contain exactly one question/answer pair")
            query = re.sub(r"(\n)?</?(image|video)>(\n)?", "", turns[0]["value"]).strip()
            refs = [str(turns[1]["value"])]
        else:
            query = record.get("prompt", record.get("question", ""))
            answer = record.get("references", record.get("answer", []))
            refs = [str(v) for v in answer] if isinstance(answer, list) else [str(answer)]
        if not isinstance(query, str) or not query.strip() or not refs or not all(refs):
            raise ValueError("Prompt examples need a nonempty query and reference labels")
        annotation = {
            "id": "automl_" + fingerprint({"path": str(path), "index": index})[:24],
            "question": query,
            "references": refs,
            "task": task,
        }
        images = record.get("image") or record.get("images")
        videos = record.get("video") or record.get("video_id")
        if images and videos:
            raise ValueError("Use one media modality per prompt example")
        values = images or videos or []
        values = [values] if isinstance(values, str) else values
        resolved = [str((media / value).resolve()) for value in values]
        if any(not Path(value).is_file() for value in resolved):
            raise ValueError("Prompt annotation references missing local media")
        if resolved:
            annotation["image" if images else "video"] = resolved
        items.append(
            {"query": query, "gold": {"references": refs, "task": task}, "annotation": annotation, "media": resolved}
        )
    if not items:
        raise ValueError("Prompt dataset is empty")
    return items


class ReasonerPromptWorkload:
    def __init__(self, config, cwd, settings):
        self.config, self.cwd, self.settings = config, cwd, settings
        self.metric_key = settings.metric

    def identity(self):
        return {
            "adapter": "cosmos-reasoner-prompts-v1",
            "evaluation": self.config,
            "cwd": str(self.cwd),
            "metric": self.metric_key,
            "provenance": self.settings.provenance,
        }

    def validate_candidate(self, candidate):
        allowed = {"dataset.system_prompt", "vision.num_frames", "vision.max_pixels"}
        if set(candidate) - allowed or "dataset.system_prompt" not in candidate:
            raise ValueError(
                "Prompt candidates require dataset.system_prompt; joint choices support vision.num_frames/max_pixels"
            )
        if not isinstance(candidate["dataset.system_prompt"], str) or len(candidate["dataset.system_prompt"]) > 16000:
            raise ValueError("System prompt must be text of at most 16000 characters")
        for name in candidate.keys() - {"dataset.system_prompt"}:
            lookup(self.config, name)
            if str(int(candidate[name])) != candidate[name] or int(candidate[name]) <= 0:
                raise ValueError("Joint vision choices must be positive integer strings")
        if "vision.num_frames" in candidate and any(key in self.config.get("vision", {}) for key in ("nframes", "fps")):
            raise ValueError("Use only canonical vision.num_frames when optimizing frame count")
        if "vision.max_pixels" in candidate and int(candidate["vision.max_pixels"]) < self.config.get("vision", {}).get(
            "min_pixels", 0
        ):
            raise ValueError("Candidate max_pixels cannot be smaller than the configured min_pixels")

    def plan_batch(self, candidate, items, trial, execution):
        self.validate_candidate(candidate)
        typed = {name: int(value) if name.startswith("vision.") else value for name, value in candidate.items()}
        config = override(self.config, typed)
        config["dataset"]["annotation_path"] = str(trial / "annotations.json")
        config["dataset"]["media_dir"] = ""
        config["results_dir"] = str(trial / "evaluation")
        config.setdefault("model", {}).update(save_folder="metrics", export_dir=str(trial / "hf"))
        config.setdefault("evaluation", {}).update(skip_saved=False, total_shard=1, shard_id=0)
        config["evaluation"].pop("limit", None)
        return TrialPlan(
            commands=[
                [
                    execution.python,
                    "-m",
                    "cosmos_framework.automl.evaluate_prompts",
                    "--config",
                    str(trial / "evaluation.toml"),
                ]
            ],
            cwd=self.cwd,
            files={
                "evaluation.toml": tomli_w.dumps(config),
                "annotations.json": json.dumps([item["annotation"] for item in items]),
            },
            progress_files={0: ["evaluation/status.json"]},
        )

    def collect_batch(self, items, trial):
        metrics = json.loads(local_artifact(trial, "evaluation/metrics/metrics.json").read_text())
        if metrics.get("overall", {}).get("total") != len(items):
            raise ValueError("Native prompt evaluation did not score every requested example")
        rows = json.loads(local_artifact(trial, "evaluation/metrics/general/results.json").read_text())
        by_id = {str(row["video_id"]): row for row in rows}
        if (
            len(rows) != len(items)
            or len(by_id) != len(rows)
            or set(by_id) != {item["annotation"]["id"] for item in items}
        ):
            raise ValueError("Missing, duplicate or unexpected native prediction IDs")
        outputs = []
        for item in items:
            row = by_id[item["annotation"]["id"]]
            if row["question"] != item["query"]:
                raise ValueError("Native prediction/query alignment mismatch")
            outputs.append(str(row.get("full_response", row["response"])))
        return outputs

    def per_item(self, output, gold):
        score = aggregate([output], [gold])["accuracy"]
        return score, {
            "correct": bool(score),
            "guidance": "Use reusable visual and temporal reasoning; do not memorize example-specific answers.",
        }

    def aggregate(self, outputs, golds):
        return aggregate(outputs, golds)


def load(path, workspace):
    manifest = Manifest.model_validate(read_manifest(path))
    config = read_toml((path.parent / manifest.workload.evaluation).resolve())
    task = config.get("task", {}).get("type")
    if task not in {"mcq", "bcq", "binary"}:
        raise ValueError("Framework GEPA needs a scored MCQ/BCQ/binary evaluator")
    if manifest.workload.metric == "macro_f1" and task not in {"bcq", "binary"}:
        raise ValueError("Macro-F1 requires a binary task")
    if not config.get("model", {}).get("model_name"):
        raise ValueError("Declare an immutable evaluation checkpoint")
    config.setdefault("dataset", {}).setdefault("system_prompt", "")
    config.setdefault("evaluation", {})["seed"] = manifest.prompts.seed
    config.setdefault("generation", {})["seed"] = manifest.prompts.seed
    roles = []
    seen = set()
    for dataset in (manifest.workload.reflection, manifest.workload.validation, manifest.workload.test):
        if dataset is None:
            roles.append(None)
            continue
        items = read_items(dataset, path.parent, task)
        keys = {"media:" + media for item in items for media in item["media"]}
        keys.update("query:" + fingerprint(item["query"]) for item in items if not item["media"])
        if seen & keys:
            raise ValueError(
                "Reflection, validation and test roles must be media-disjoint (or query-disjoint without media)"
            )
        seen.update(keys)
        roles.append(items)
    manifest.execution.mounts = [(path.parent / p).absolute() for p in manifest.execution.mounts]
    if manifest.execution.kind == "slurm" and (
        not manifest.execution.image or not manifest.execution.slurm.record_open
    ):
        raise ValueError("Framework SLURM prompt batches need the prepared image and job-record hooks")
    work = ReasonerPromptWorkload(config, (path.parent / manifest.workload.cwd).resolve(), manifest.workload)
    for name, choices in manifest.prompts.config_choices.items():
        if name == "dataset.system_prompt":
            raise ValueError("System prompt is a reflected text component, not a joint numeric setting")
        for choice in choices:
            work.validate_candidate({**manifest.prompts.seed_candidate, name: choice})
    return PromptExperiment(manifest.prompts, work, manifest.execution, workspace, *roles)
