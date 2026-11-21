# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Persistent, isolated WTS scoring on the live SFT model at durable boundaries."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.data.datasets.sft import row_identity, sha256_file
from cosmos_framework.posttrain.roles.trainer.validation.runtime import validation_module_inventory
from cosmos_framework.posttrain.roles.trainer.validation.wts import aggregate, evaluate_row, json_digest, validate_wts

INPUT_KEYS = (
    "id",
    "source_line",
    "raw_row_sha256",
    "row_identity",
    "gold",
    "choices",
    "media_sha256",
    "input_sha256",
    "inputs",
    "candidate_token_ids",
    "answer_token_position",
    "prediction_logit_position",
    "native_supervised_tokens",
    "sequence_length",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class DevelopmentEvaluation:
    """Own only evaluation inputs/cache and deterministic scoring history, never training state.

    Wall clocks and artifact locations are attempt-local sidecars. The checkpoint journal contains
    only content and numerical observations, so interruption recovery can compare trainer state
    without deleting timing or output-location fields from the production identity.
    """

    def __init__(self, settings: Any, rl: Any, training: Any, *, model_source: str, trust: bool, output: str) -> None:
        from cosmos_framework.posttrain.algorithm.sft import make_objective
        from cosmos_framework.posttrain.data.datasets.sft import refuse_source_overlap, source_overlap
        from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
        from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
        from cosmos_framework.posttrain.roles.env.sft import build_sft_env
        from cosmos_framework.posttrain.utils.adapter_identity import adapter_processor_identity
        from cosmos_framework.posttrain.utils.adapter_source import verify_adapter_snapshot

        self.settings, self.horizon = settings, rl.max_steps
        self.output = Path(output).resolve()
        self.model_source = model_source
        self.history: list[dict[str, Any]] = []
        self.artifacts: dict[int, dict[str, Any]] = {}
        self.pipeline = None
        self.source = build_sft_env(settings.data, rl.validation.seed, training=False)
        self.training = training
        refuse_source_overlap(source_overlap(training.dataset, self.source.dataset))
        if len(self.source.dataset) != settings.expected_rows:
            raise ValueError("WTS source row count differs from its declared complete partition")
        self.media = dict(self.source.dataset.media_hashes)
        self.snapshot = verify_adapter_snapshot(model_source)
        self.verify_sources()
        lines = Path(settings.data.path).read_text().splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("WTS source must have one sealed row on every line")
        raw = [json.loads(line) for line in lines]
        manifest = json.loads(Path(settings.manifest_path).read_text())
        if manifest.get("protocol") != "cosmos.wts-group-split.v1" or manifest.get("role") != "development":
            raise ValueError("WTS requires a sealed development manifest")
        if manifest.get("data_sha256") != settings.source_sha256:
            raise ValueError("WTS manifest names different source bytes")
        sealed = manifest.get("rows")
        if not isinstance(sealed, list) or len(sealed) != len(raw):
            raise ValueError("WTS manifest must seal every source row in order")
        groups = {row["group_sha256"] for row in sealed}
        if type(manifest.get("group_count")) is not int or len(groups) != manifest["group_count"]:
            raise ValueError("WTS sealed group count differs")
        self.selection_base = {
            "protocol": manifest["protocol"],
            "source": str(Path(settings.data.path).resolve()),
            "source_sha256": settings.source_sha256,
            "source_rows": len(raw),
            "selected_rows": len(raw),
            "selection_rule": "all_rows_from_precommitted_group_split_in_source_order",
            "split_manifest_sha256": settings.manifest_sha256,
            "split_role": "development",
            "group_count": manifest["group_count"],
            "release_qualified": False,
        }
        self.selected = []
        for i, original in enumerate(raw):
            row = self.source.dataset[i]
            identifier, answer, choices, media_path = validate_wts(row)
            record = {
                "source_line": i + 1,
                "id": identifier,
                "row_identity": row_identity(row, media_digest=self.media.__getitem__)[0],
                "raw_row_sha256": json_digest(original),
                "normalized_row_sha256": json_digest(row),
                "gold": answer,
                "choices": list(choices),
                "media_sha256": self.media[media_path],
                "media_size_bytes": Path(media_path).stat().st_size,
                "media_path": media_path,
                "row": row,
            }
            if any(record[key] != sealed[i][key] for key in ("id", "raw_row_sha256", "media_sha256")):
                raise ValueError("WTS source rows/media differ from the sealed development manifest")
            self.selected.append(record)
        if len({r["id"] for r in self.selected}) != len(self.selected):
            raise ValueError("WTS full-development IDs must be unique")
        packer = build_sft_packer(
            model_source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        self.processor_identity = adapter_processor_identity(
            packer, source_path=model_source, source_identity=self.snapshot
        )
        if settings.cache_bytes:
            packer = self.pipeline = build_sft_preprocessing(
                packer,
                name_or_path=model_source,
                cache_bytes=settings.cache_bytes,
                prefetch=False,
                media_hashes=self.media,
                max_pending=1,
                workers=1,
            )
        self.packer = packer
        self.objective = make_objective(
            packer,
            ce_exponent=rl.ce_exponent,
            moe_lbl_coeff=rl.moe_lbl_coeff,
            moe_lbl_method=rl.moe_lbl_method,
            serve_contract=rl.serve_contract,
        )
        self.identity = {
            "protocol": "cosmos.native-sft-wts.v1",
            "source_fingerprint": self.source.source_fingerprint,
            "source_sha256": settings.source_sha256,
            "manifest_sha256": settings.manifest_sha256,
            "expected_rows": settings.expected_rows,
            "steps": list(settings.steps),
            "cache_bytes": settings.cache_bytes,
            "min_correct": settings.min_correct,
            "max_token_ce": settings.max_token_ce,
            "stop_on_target": settings.stop_on_target,
            "processor": self.processor_identity,
            "objective": self.objective.identity,
            "selection_sha256": json_digest(self.selected),
        }

    def verify_sources(self) -> None:
        from cosmos_framework.posttrain.utils.adapter_source import verify_adapter_snapshot

        for path, expected in (
            (self.settings.data.path, self.settings.source_sha256),
            (self.settings.manifest_path, self.settings.manifest_sha256),
        ):
            if sha256_file(path) != expected:
                raise ValueError("WTS source or sealed manifest changed")
        if any(sha256_file(path) != expected for path, expected in self.media.items()):
            raise ValueError("WTS media changed")
        if verify_adapter_snapshot(self.model_source) != self.snapshot:
            raise ValueError("WTS model/processor source identity changed")

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "observations": copy.deepcopy(self.history)}

    def load_state_dict(self, state: Any, *, iteration: int, pending_step: int | None = None) -> None:
        if not isinstance(state, dict) or set(state) != {"version", "observations"} or state["version"] != 1:
            raise RuntimeError("unsupported WTS evaluation journal")
        observations = state["observations"]
        if not isinstance(observations, list):
            raise RuntimeError("WTS observations must be a list")
        expected = [s for s in self.settings.steps if s <= iteration and s != pending_step]
        if any(not isinstance(r, dict) for r in observations) or [r.get("step") for r in observations] != expected:
            raise RuntimeError("WTS journal is not a completed declared step prefix")
        for i, row in enumerate(observations):
            if set(row) != {
                "step",
                "rows",
                "correct",
                "token_weighted_ce",
                "target_met",
                "metrics",
                "scores_sha256",
                "selected_content_sha256",
                "processed_inputs_sha256",
            }:
                raise RuntimeError("WTS journal observation has unsupported fields")
            if (
                type(row["step"]) is not int
                or type(row["rows"]) is not int
                or row["rows"] != self.settings.expected_rows
                or type(row["correct"]) is not int
                or not 0 <= row["correct"] <= row["rows"]
            ):
                raise RuntimeError("WTS journal row count is invalid")
            metrics = row["metrics"]
            if (
                not isinstance(metrics, dict)
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in metrics.values()
                )
                or metrics.get("rows") != row["rows"]
                or metrics.get("restricted_choice_accuracy") != row["correct"] / row["rows"]
                or metrics.get("token_weighted_ce") != row["token_weighted_ce"]
                or metrics.get("native_supervised_tokens", 0) < 1
            ):
                raise RuntimeError("WTS journal metrics are invalid")
            for name in ("scores_sha256", "selected_content_sha256", "processed_inputs_sha256"):
                if not isinstance(row[name], str) or not re.fullmatch("[0-9a-f]{64}", row[name]):
                    raise RuntimeError("WTS journal content digest is invalid")
            if (
                type(row["token_weighted_ce"]) not in (int, float)
                or not math.isfinite(row["token_weighted_ce"])
                or row["token_weighted_ce"] < 0
            ):
                raise RuntimeError("WTS journal CE must be finite and nonnegative")
            passed = (
                row["correct"] >= self.settings.min_correct and row["token_weighted_ce"] <= self.settings.max_token_ce
            )
            if (
                type(row["target_met"]) is not bool
                or row["target_met"] != passed
                or (
                    self.settings.stop_on_target and passed and (i != len(observations) - 1 or row["step"] != iteration)
                )
            ):
                raise RuntimeError("WTS journal contains an invalid first-pass decision")
            if i and row["processed_inputs_sha256"] != observations[0]["processed_inputs_sha256"]:
                raise RuntimeError("WTS journal processed input identity changed")
        self.history = copy.deepcopy(observations)

    def evaluate(self, model: Any, optimizer: Any, step: int) -> dict[str, int | float]:
        if step not in self.settings.steps:
            return {}
        if len(self.history) >= len(self.settings.steps) or step != self.settings.steps[len(self.history)]:
            raise RuntimeError("WTS evaluation must follow the declared grid exactly once")
        if self.settings.stop_on_target and self.history and self.history[-1]["target_met"]:
            raise RuntimeError("WTS cannot evaluate after its first target pass")
        from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
        from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase

        rng = collect_rng_state()
        train_cursor, eval_cursor = self.training.state_dict(), self.source.state_dict()
        _, modules = validation_module_inventory(model)
        cleanup = ExitStack()
        cleanup.callback(restore_rng_state, rng)
        cleanup.callback(self.training.load_state_dict, train_cursor)
        cleanup.callback(self.source.load_state_dict, eval_cursor)
        for module in modules:
            cleanup.callback(setattr, module, "training", module.training)
        codec = model.net._cosmos_adapter_checkpoint_codec
        directory = self.output / "wts" / f"step_{step:06d}" / uuid.uuid4().hex
        try:
            directory.mkdir(parents=True, exist_ok=False)
            self.verify_sources()
            codec.assert_live(model.net, optimizer)
            before = {name: tensor_digest(value) for name, value in codec.parameters.items()}
            selected_file = directory / "selected_rows.jsonl"
            _write_rows(selected_file, self.selected)
            selected_content = json_digest(
                [
                    {key: row[key] for key in ("id", "row_identity", "gold", "choices", "media_sha256")}
                    for row in self.selected
                ]
            )
            selection = {
                **self.selection_base,
                "ordered_ids": [r["id"] for r in self.selected],
                "ordered_row_identities": [r["row_identity"] for r in self.selected],
                "selected_content_sha256": selected_content,
                "selected_rows_file_sha256": sha256_file(str(selected_file)),
            }
            _write_json(directory / "selection.json", selection)
            record_qlora_phase(f"wts_scoring_{step:06d}_start", reset_peak=True)
            rows = []
            for selected in self.selected:
                scores = evaluate_row(model, self.packer, self.objective, selected["row"])
                rows.append({**{k: v for k, v in selected.items() if k != "row"}, **scores})
            self.verify_sources()
            codec.validate_frozen_base()
            if before != {name: tensor_digest(value) for name, value in codec.parameters.items()}:
                raise RuntimeError("WTS evaluation mutated adapter parameters")
            if self.training.state_dict() != train_cursor or self.source.state_dict() != eval_cursor:
                raise RuntimeError("WTS evaluation mutated a data cursor")
            _write_rows(directory / "scores.jsonl", rows)
            metrics = aggregate(rows)
            inputs = json_digest([{key: row[key] for key in INPUT_KEYS} for row in rows])
            if self.history and inputs != self.history[0]["processed_inputs_sha256"]:
                raise RuntimeError("WTS processed inputs changed across validation steps")
            correct = sum(row["restricted_choice_correct"] for row in rows)
            passed = correct >= self.settings.min_correct and metrics["token_weighted_ce"] <= self.settings.max_token_ce
            observation = {
                "step": step,
                "rows": len(rows),
                "correct": correct,
                "token_weighted_ce": metrics["token_weighted_ce"],
                "target_met": passed,
                "metrics": metrics,
                "scores_sha256": sha256_file(str(directory / "scores.jsonl")),
                "selected_content_sha256": selected_content,
                "processed_inputs_sha256": inputs,
            }
            checkpoint = str(self.output / "checkpoints" / f"iter_{step:06d}")
            _write_json(
                directory / "summary.json",
                {
                    "protocol": "cosmos.native-sft-wts-score.v1",
                    "complete": True,
                    "checkpoint": checkpoint,
                    "metrics": metrics,
                    "selection_sha256": json_digest(selection),
                    "scores_sha256": observation["scores_sha256"],
                    "selected_content_sha256": selected_content,
                    "adapter_tensors_sha256": json_digest(before),
                    "processor_contract": self.processor_identity,
                    "objective_identity": self.objective.identity,
                    "model_identity_sha256": json_digest(
                        {k: v for k, v in codec.identity.items() if k != "training_contract"}
                    ),
                    "checkpoint_publication_required": True,
                    **(
                        {
                            "preprocessing_cache": {
                                "policy": {
                                    "cache_bytes": self.settings.cache_bytes,
                                    "prefetch": False,
                                    "workers": 1,
                                    "max_pending": 1,
                                },
                                "stats": copy.deepcopy(self.pipeline.preprocessing_stats),
                                "memory_scope": "bounded retained CPU cache entries, not total host memory",
                                "counter_scope": "Cumulative within this native evaluator attempt, including earlier scheduled evaluations; not restored from a checkpoint. Successive snapshots must not be summed.",
                                "elapsed_scope": "Processing counters include cache hits and exclude model evaluation; they are nested in validation wall time, not an additional elapsed-time charge.",
                            }
                        }
                        if self.pipeline is not None
                        else {}
                    ),
                },
            )
            record_qlora_phase(f"wts_scoring_{step:06d}_complete")
        finally:
            cleanup.close()
        # Only successful scoring AND restoration may alter the checkpoint journal.
        self.history.append(observation)
        self.artifacts[step] = {"directory": str(directory.relative_to(self.output)), "checkpoint": checkpoint}
        return {
            "val/wts_correct": correct,
            "val/wts_rows": len(rows),
            "val/wts_token_weighted_ce": metrics["token_weighted_ce"],
            "val/wts_target_met": int(passed),
        }

    def publish(self, *, step: int, checkpoint: str, started_at: float, resumed: bool) -> dict[str, Any]:
        """Called only after validation completion has been durably published by the trainer."""
        if self.history and self.history[-1]["step"] == step and step in self.artifacts:
            self.artifacts[step].setdefault("available_monotonic_seconds", time.monotonic())
        observations = []
        for row in self.history:
            artifact = self.artifacts.get(row["step"], {})
            observations.append(
                {
                    **row,
                    "directory": artifact.get("directory"),
                    "checkpoint": artifact.get("checkpoint"),
                    "available_monotonic_seconds": artifact.get("available_monotonic_seconds"),
                    "available_on_this_attempt": bool(artifact),
                }
            )
        passing = next((row for row in observations if row["target_met"]), None)
        complete = (self.settings.stop_on_target and passing is not None) or step == self.horizon
        result = {
            "protocol": "cosmos.native-sft-wts-target.v1",
            "complete": complete,
            "final_step": step,
            "horizon": self.horizon,
            "stop_on_target": self.settings.stop_on_target,
            "first_passing_step": passing["step"] if passing else None,
            "horizon_censored": complete and passing is None,
            "checkpoint": str(Path(checkpoint).resolve()),
            "observations": observations,
            "target": {
                "expected_rows": self.settings.expected_rows,
                "minimum_correct": self.settings.min_correct,
                "maximum_token_ce": self.settings.max_token_ce,
                "validation_steps": self.settings.steps,
            },
            "attempt_elapsed_seconds": time.monotonic() - started_at,
            "resumed_from_checkpoint": resumed,
            "prior_attempt_availability_not_reconstructed": True,
            "availability_clock_scope": "After score files and full validation checkpoint publication/readback verification; excludes final target-index serialization and later cleanup, which remain in complete native-process elapsed time.",
        }
        _write_json(self.output / "wts" / "target-result.json", result)
        return result

    def verify_checkpoint(self, checkpoint: str, manager: Any, model: Any, optimizer: Any) -> None:
        from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata

        metadata = read_adapter_metadata(checkpoint, require_resume=True, manager=manager)
        model.net._cosmos_adapter_checkpoint_codec.validate_metadata(metadata, optimizer, purpose="resume")
        trainer = manager.load_component(checkpoint, "trainer")
        validation = trainer.get("validation", {}) if isinstance(trainer, dict) else {}
        if (
            validation.get("wts") != self.state_dict()
            or validation.get("progress", {}).get("pending_step") is not None
            or validation.get("publication") is not None
        ):
            raise RuntimeError("WTS checkpoint readback lacks its completed scoring journal")
        for artifact in self.artifacts.values():
            directory = self.output / artifact["directory"]
            # The UUID and step directory entries must also reach durable storage
            # before exposing availability, not only the files in the UUID leaf.
            while True:
                _fsync_directory(directory)
                if directory == self.output:
                    break
                directory = directory.parent

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()
