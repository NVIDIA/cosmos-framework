# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-light, explicit contracts for bounded native QLoRA experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    seed: int = 42
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: float = 32.0
    steps: int = 300
    warmup_steps: int = 30
    microbatch: int = 1
    effective_batch: int = 8
    max_length: int = 4096
    frames: int = 8
    video_min_pixels: int = 32768
    video_max_pixels: int = 524288
    activation_checkpointing: str = "full"
    adapter_compute_dtype: str = "float32"
    logits_mode: str = "full"
    compile_language_model: bool = False
    cache_bytes: int = 1073741824
    prefetch: bool = True
    save_every: int = 50
    validation_every: int = 50
    validation_samples: int = 64
    initialization: str = "kaiming_A_zero_B"
    initialization_artifact: str = ""
    initialization_artifact_sha256: str = ""
    dropout: float = 0.0
    wts_answer_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,95}", self.name):
            raise ValueError("experiment name must be a bounded lowercase slug")
        if self.arm not in {"nf4", "bf16"}:
            raise ValueError("only the native nf4 and matched bf16 arms are supported")
        for field in (
            "rank",
            "steps",
            "microbatch",
            "effective_batch",
            "max_length",
            "frames",
            "video_min_pixels",
            "video_max_pixels",
            "save_every",
            "validation_every",
            "validation_samples",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("seed", "warmup_steps", "cache_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in ("learning_rate", "alpha"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if type(self.dropout) not in (int, float) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if (
            type(self.wts_answer_smoothing) not in (int, float)
            or not math.isfinite(self.wts_answer_smoothing)
            or not 0 <= self.wts_answer_smoothing < 1
        ):
            raise ValueError("wts_answer_smoothing must be finite and in [0, 1)")
        if self.effective_batch % self.microbatch or self.warmup_steps >= self.steps:
            raise ValueError("microbatch must divide effective batch; warmup must precede completion")
        if self.video_min_pixels > self.video_max_pixels or self.frames % 2:
            raise ValueError("invalid temporal pixel budgets or odd frame count")
        if self.activation_checkpointing not in {"full", "none", "selective"}:
            raise ValueError("unsupported native activation checkpointing mode")
        if self.adapter_compute_dtype not in {"float32", "bfloat16"} or self.logits_mode not in {"full", "supervised"}:
            raise ValueError("unsupported adapter math or logits mode")
        if type(self.prefetch) is not bool or type(self.compile_language_model) is not bool:
            raise ValueError("prefetch and compile must be booleans")
        if self.initialization not in {"kaiming_A_zero_B", "nf4_residual_svd"}:
            raise ValueError("unsupported native adapter initialization")
        if not isinstance(self.initialization_artifact, str) or not isinstance(
            self.initialization_artifact_sha256, str
        ):
            raise ValueError("initializer path and SHA256 must be strings")
        if self.initialization == "kaiming_A_zero_B":
            if self.initialization_artifact or self.initialization_artifact_sha256:
                raise ValueError("zero-B initialization forbids a residual initializer artifact")
        elif (
            self.arm != "nf4"
            or not Path(self.initialization_artifact).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", self.initialization_artifact_sha256) is None
        ):
            raise ValueError("residual initialization requires NF4 and an absolute SHA256-bound artifact")

    @classmethod
    def read(cls, path: str | Path) -> Experiment:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("experiment must be a JSON object")
        return cls(**value)

    @property
    def recipe(self) -> str:
        suffix = "qlora" if self.arm == "nf4" else "lora"
        return f"cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_{suffix}.py"

    def overrides(self) -> list[str]:
        values = {
            "job.project": "native_qlora_qualification",
            "job.group": "wts",
            "job.name": self.name,
            "job.wandb_mode": "disabled",
            "rl.run_ts": "qualification_v1",
            "rl.seed": self.seed,
            "rl.deterministic": True,
            "rl.max_steps": self.steps,
            "rl.micro_batch_size": self.microbatch,
            "rl.grad_accum": self.effective_batch // self.microbatch,
            "rl.max_length": self.max_length,
            "rl.optimizer.lr": self.learning_rate,
            "rl.optimizer.warmup_steps": self.warmup_steps,
            "rl.lr_schedule.cycle_steps": self.steps,
            "rl.save_every": self.save_every,
            "rl.checkpoint_format": "dcp",
            "rl.save_async": False,
            "rl.checkpoint_components": ["model", "optimizer", "training_state"],
            "rl.emergency_checkpoint_components": "full",
            "rl.keep_optim_last": 0,
            "rl.keep_model_last": 0,
            "rl.validation.enabled": True,
            "rl.validation.before_training": True,
            "rl.validation.every_steps": self.validation_every,
            "rl.validation.max_samples": self.validation_samples,
            "rl.validation.batch_size": 1,
            "rl.validation.seed": self.seed,
            "rl.validation.save_checkpoint": True,
            "rl.validation.checkpoint_components": ["model", "optimizer", "training_state"],
            "rl.preprocessing_cache_bytes": self.cache_bytes,
            "rl.preprocessing_prefetch": self.prefetch,
            "model.model_config.adapter.rank": self.rank,
            "model.model_config.adapter.alpha": self.alpha,
            "model.model_config.adapter.initialization_seed": self.seed,
            "model.model_config.activation_checkpointing": self.activation_checkpointing,
            "model.model_config.adapter_compute_dtype": self.adapter_compute_dtype,
            "model.model_config.sft_logits_mode": self.logits_mode,
            "model.model_config.compile_language_model": self.compile_language_model,
            "rl.processor_kwargs.videos_kwargs.num_frames": self.frames,
            "rl.processor_kwargs.videos_kwargs.size.shortest_edge": self.video_min_pixels,
            "rl.processor_kwargs.videos_kwargs.size.longest_edge": self.video_max_pixels,
        }
        # Omit the historical zero default to preserve sealed specification and
        # override identities. Any stochastic adapter recipe is explicit.
        if self.dropout:
            values["model.model_config.adapter.dropout"] = self.dropout
        if self.wts_answer_smoothing:
            values["rl.wts_answer_smoothing"] = self.wts_answer_smoothing
        if self.initialization != "kaiming_A_zero_B":
            values.update(
                {
                    "model.model_config.adapter.initialization": self.initialization,
                    "model.model_config.adapter.initialization_artifact": self.initialization_artifact,
                    "model.model_config.adapter.initialization_artifact_sha256": self.initialization_artifact_sha256,
                }
            )
        return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self)
        if self.dropout == 0:
            del result["dropout"]
        if self.wts_answer_smoothing == 0:
            del result["wts_answer_smoothing"]
        if self.initialization == "kaiming_A_zero_B":
            for name in ("initialization", "initialization_artifact", "initialization_artifact_sha256"):
                del result[name]
        return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Exclusive publication: failed attempts and prior evidence are never overwritten."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def assert_budget(records: list[dict], *, next_gpu_hours: float, limit_gpu_hours: float = 48.0) -> float:
    """Count completed allocation time plus full reservations for nonterminal jobs.

    The scheduler must call this before submission. Failed attempts count. Unknown
    accounting is an error, never a free reservation. This is not a Slurm client.
    """
    if not math.isfinite(next_gpu_hours) or next_gpu_hours < 0:
        raise ValueError("invalid next allocation reservation")
    total = 0.0
    seen = set()
    for record in records:
        identity = record["job_id"]
        if identity in seen:
            raise ValueError("duplicate allocation in budget ledger")
        seen.add(identity)
        hours = record["elapsed_gpu_hours"] if record["terminal"] else record["reserved_gpu_hours"]
        if type(hours) not in (int, float) or not math.isfinite(hours) or hours < 0:
            raise ValueError("missing or invalid allocation accounting")
        total += hours
    if total + next_gpu_hours > limit_gpu_hours:
        raise ValueError("bounded GPU allocation budget would be exceeded")
    return total
