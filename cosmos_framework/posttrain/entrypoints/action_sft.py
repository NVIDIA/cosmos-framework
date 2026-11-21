# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native supervised driving/robotics training with family-owned inputs and objectives.

The existing LLMTrainer owns every backward, optimizer update and checkpoint transaction.
This entry owns deterministic data traversal and completed-update evaluation boundaries.
It intentionally supports one process and no gradient accumulation until task-specific
distributed reductions and recovery are qualified.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _factory(path: str):
    if not isinstance(path, str) or "." not in path:
        raise ValueError("action SFT factories must be explicit import paths")
    module, name = path.rsplit(".", 1)
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise TypeError(f"action SFT factory is not callable: {path}")
    return result


class _Source:
    """Give the shared IndexedCursor a content-bound family dataset."""

    def __init__(self, dataset):
        from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

        self.dataset = dataset
        self.identity = json.loads(json.dumps(dataset.identity, allow_nan=False))
        if not self.identity or not len(dataset):
            raise ValueError("action SFT requires a nonempty, content-bound map-style dataset")
        self.fingerprint = canonical_json_sha256(self.identity, context="action SFT source")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def cursor_fingerprint(self):
        return self.fingerprint


@contextmanager
def preserved_evaluation(policy, seed: int):
    """Restore mixed module modes and all training RNG streams, including on failure."""
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state

    modes = [(module, module.training) for module in policy.net.modules()]
    rng = collect_rng_state()
    try:
        policy.net.eval()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            yield
    finally:
        try:
            for module, mode in modes:
                module.training = mode
        finally:
            restore_rng_state(rng)


def evaluate(policy, objective, source, config, callback=None):
    """Full finite evaluation; loss is weighted by the objective's native denominator."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import evaluation_batch_scope

    limit = config.rl.evaluation_max_samples or len(source)
    count = min(limit, len(source))
    totals: dict[str, list[float]] = {}
    rows = []
    with preserved_evaluation(policy, config.rl.evaluation_seed):
        for start in range(0, count, config.rl.micro_batch_size):
            batch = [source[index] for index in range(start, min(start + config.rl.micro_batch_size, count))]
            with evaluation_batch_scope(policy, batch):
                loss, metrics = objective(policy, batch)
                weight = float(metrics.get(OBJECTIVE_WEIGHT_SUM, len(batch)))
                values = {"loss": {"sum": float(loss.detach()) * weight, "count": weight}}
                if callback is not None:
                    extra = callback(policy, batch, config)
                    if "loss" in extra:
                        raise ValueError("task evaluation callback must not replace the native loss")
                    values.update(extra)
            for key, item in values.items():
                if set(item) != {"sum", "count"}:
                    raise ValueError(f"evaluation metric {key} requires explicit sum and count")
                numerator, denominator = float(item["sum"]), float(item["count"])
                if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"nonfinite or empty action evaluation metric {key}")
                total = totals.setdefault(key, [0.0, 0.0])
                total[0] += numerator
                total[1] += denominator
            rows.append({"ids": [item.id for item in batch], "metrics": values})
    return {key: numerator / denominator for key, (numerator, denominator) in totals.items()}, rows


def find_target(history, rl):
    """First prospective target crossing; no substitution of loss or retrospective target selection."""
    if not rl.target_metric:
        return None
    for item in history:
        value = item["metrics"].get(rl.target_metric)
        if value is None or not math.isfinite(value):
            raise ValueError("declared quality target metric is missing or nonfinite")
        passed = value <= rl.target_threshold if rl.target_direction == "min" else value >= rl.target_threshold
        if passed:
            return item
    return None


def _factory_identity(config):
    import inspect

    from cosmos_framework.posttrain.utils.object_store import file_sha256

    result = {}
    for name in ("policy_factory", "data_factory", "objective_factory", "evaluation_callback"):
        path = getattr(config, name)
        if path:
            source = inspect.getsourcefile(_factory(path))
            if source is None:
                raise ValueError(f"cannot bind native action factory source: {path}")
            result[name] = {"factory": path, "source_sha256": file_sha256(source)}
    result["entrypoint_sha256"] = file_sha256(__file__)
    return result


def validate_config(config):
    from cosmos_framework.utils.lazy_config import instantiate

    rl = instantiate(config.rl)
    rl.__post_init__()
    if config.grid.backend != "local" or config.grid.grouping != "single":
        raise ValueError("action SFT requires the native local single-process torchrun path")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("action SFT backbone adaptation does not yet support distributed execution")
    if rl.checkpoint_store.backend != "local" or rl.upload.enabled:
        raise ValueError("action SFT adaptation currently requires local durable checkpoint publication")
    for name in ("policy_factory", "data_factory", "objective_factory"):
        if not isinstance(getattr(config, name), str) or "." not in getattr(config, name):
            raise ValueError(f"action SFT requires {name}")
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import validate_evaluation_input_reuse
    from cosmos_framework.posttrain.utils.gr00t_evaluation_backbone import validate_reuse

    validate_evaluation_input_reuse(config, rl)
    validate_reuse(config, rl)
    return rl


def _json(path: Path, value: Any):
    from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes

    atomic_write_bytes(str(path), (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def prepare_sources(config):
    """Validate owned input files and split separation before allocating a model.

    Native task data factories are model-independent and receive None during
    admission; processors/packers remain attached to the constructed policy.
    """
    data_factory = _factory(config.data_factory)
    train = _Source(data_factory(config, "train", None))
    validation = _Source(data_factory(config, "validation", None))
    for attribute in ("groups", "ids", "media_hashes"):
        left, right = getattr(train.dataset, attribute, None), getattr(validation.dataset, attribute, None)
        if left is not None and right is not None:
            if attribute == "media_hashes":
                left, right = left.values(), right.values()
            if set(left) & set(right):
                raise ValueError(f"action SFT train/validation overlap in {attribute}")
    return train, validation


def _preflight_backbone_runtime(config, device):
    """Reject unavailable selected backbone execution before reading data or model weights."""
    adaptation = getattr(config.model, "backbone_adaptation", None)
    if adaptation is None:
        return  # The native official-head path has its own precision/runtime contract.
    mode = getattr(adaptation, "mode", None)
    if mode not in {"nf4_lora", "nf4_frozen", "bf16_lora", "bf16_frozen"}:
        raise ValueError("native action backbone mode must be nf4_lora|nf4_frozen|bf16_lora|bf16_frozen")

    from importlib.metadata import PackageNotFoundError

    from cosmos_framework.posttrain.model.quantization.family import _runtime

    try:
        _runtime(device, quantized=mode.startswith("nf4_"))
    except PackageNotFoundError as error:
        if error.name != "bitsandbytes":
            raise
        raise ValueError(
            "native action NF4 requires bitsandbytes==0.49.2; follow docker/posttrain/native_action/README.md "
            "for the pinned runtime, or install it with "
            "`uv pip install --python <runtime-python> bitsandbytes==0.49.2`. No dense fallback is permitted."
        ) from error


def training_identity(config, rl, objective, validation):
    """Bind resolved native configuration, including OmegaConf container leaves."""
    import torch

    from cosmos_framework.posttrain.utils.action_evaluation_inputs import reuse_identity
    from cosmos_framework.posttrain.utils.action_training_identity import action_training_source_identity
    from cosmos_framework.posttrain.utils.gr00t_evaluation_backbone import reuse_identity as backbone_reuse_identity
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    return canonical_config_value(
        {
            "format": "cosmos-native-action-sft-v1",
            "factories": _factory_identity(config),
            "implementation": action_training_source_identity(),
            "objective": objective.identity,
            "validation": validation.identity,
            "seed": rl.seed,
            "max_steps": rl.max_steps,
            "runtime": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "optimizer_state_dtype": "float32",
            },
            "optimizer": asdict(rl.optimizer),
            "lr_schedule": asdict(rl.lr_schedule),
            "evaluation_every": rl.evaluation_every,
            "evaluation_seed": rl.evaluation_seed,
            "evaluation_max_samples": rl.evaluation_max_samples,
            "evaluation_input_reuse": reuse_identity(rl),
            "evaluation_backbone_reuse": backbone_reuse_identity(rl),
            "target": [rl.target_metric, rl.target_threshold, rl.target_direction, rl.stop_on_target],
        }
    )


def run(config, *, config_path: str, opts: list[str]):
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint.adapter import publish_adapter_artifact, read_adapter_metadata
    from cosmos_framework.posttrain.config.build import build_optimizer, build_sft_lr_scheduler
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import (
        install_evaluation_input_reuse,
        preparation_observations,
        validate_evaluation_input_reuse,
    )
    from cosmos_framework.posttrain.utils.action_training_identity import (
        action_training_source_identity,
        require_matching_training_source,
    )
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.entry import dump_resolved, out_dir, pin_run_attempt
    from cosmos_framework.posttrain.utils.gr00t_evaluation_backbone import install_reuse, observations
    from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank
    from cosmos_framework.utils import misc

    started = time.monotonic()
    rl = validate_config(config)
    if "LOCAL_RANK" not in os.environ or not torch.cuda.is_available():
        raise ValueError("native action SFT requires torchrun and a CUDA device")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    enable_determinism(strict=True)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    attempt = pin_run_attempt(config, opts, out_dir(config))
    output = Path(attempt.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump_resolved(config, str(output), provenance=True)
    lifecycle = CheckpointLifecycle.from_config(
        config, rl, run_out_dir=str(output), resume_checkpoint=attempt.resume_checkpoint
    )
    if attempt.resume_checkpoint:
        metadata = read_adapter_metadata(attempt.resume_checkpoint, require_resume=True, manager=lifecycle.manager)
        require_matching_training_source(
            metadata["identity"].get("training_contract"), action_training_source_identity()
        )
    _preflight_backbone_runtime(config, device)
    record_qlora_phase("data_preparation_begin", reset_peak=True)
    train, validation = prepare_sources(config)
    validate_evaluation_input_reuse(config, rl, validation)
    record_qlora_phase("data_preparation_complete", reset_peak=True)
    misc.set_random_seed(seed=rl.seed, by_rank=False)
    record_qlora_phase("model_loading_begin", reset_peak=True)
    policy = _factory(config.policy_factory)(config, device=device, output=str(output))
    codec = getattr(policy.net, "_cosmos_adapter_checkpoint_codec", None)
    if codec is None:
        raise RuntimeError("action SFT family did not install the native checkpoint codec")
    record_qlora_phase("model_loading_complete", reset_peak=True)
    cursor = IndexedCursor(train, shuffle=True, seed=rl.seed, cursor_resume="strict", label="action SFT")
    install_evaluation_input_reuse(policy, config, rl)
    install_reuse(policy, config, rl)
    objective = _factory(config.objective_factory)(policy, config)
    objective.identity = {
        **objective.identity,
        "training_source": train.identity,
        "micro_batch_size": rl.micro_batch_size,
    }
    callback = _factory(config.evaluation_callback) if config.evaluation_callback else None
    training_contract = training_identity(config, rl, objective, validation)
    codec.bind_training_identity(training_contract)
    optimizer = build_optimizer(policy.net, rl.optimizer)
    codec.assert_live(optimizer=optimizer)
    trainer = LLMTrainer(
        policy,
        objective,
        optimizer,
        grad_accum=1,
        n_mini=1,
        max_grad_norm=rl.optimizer.grad_clip_norm,
        lr_scheduler=build_sft_lr_scheduler(optimizer, rl.optimizer, rl.lr_schedule, rl.max_steps),
        require_gradients=True,
        nonfinite_update="fail_stop",
        checkpoint_manager=lifecycle.manager,
    )
    history: list[dict[str, Any]] = []
    step = 0
    if attempt.resume_checkpoint:
        restored = lifecycle.load_trainer_and_component(trainer, "data")
        if restored is None or (state := state_for_current_rank(restored[1])) is None:
            raise ValueError("action SFT resume requires its committed data/evaluation state")
        if set(state) != {"format", "cursor", "evaluation"} or state["format"] != "action-sft-state-v1":
            raise ValueError("unsupported action SFT data state")
        cursor.load_state_dict(state["cursor"])
        history = state["evaluation"]
        step = restored[0]
        if cursor.consumed != step * rl.micro_batch_size:
            raise ValueError("action SFT resumed cursor and optimizer clock disagree")
    clocks = {
        "startup_seconds": time.monotonic() - started,
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }
    watcher = TerminationWatch()
    last_checkpoint = attempt.resume_checkpoint
    crossing = find_target(history, rl)
    target = (
        None
        if crossing is None
        else {
            "step": crossing["step"],
            "checkpoint": str(Path(attempt.resume_checkpoint).parent / f"iter_{crossing['step']:06d}"),
            "seconds": None,
            "timing_scope": "restored-crossing-uninterrupted-time-unavailable",
        }
    )
    if target is not None:
        read_adapter_metadata(
            target["checkpoint"], expected_identity=codec.identity, require_resume=False, manager=lifecycle.manager
        )

    def save(update):
        nonlocal last_checkpoint
        if update != trainer.iteration():
            raise RuntimeError("action checkpoint boundary differs from the native optimizer clock")
        path = str(Path(lifecycle.checkpoint_root) / f"iter_{update:06d}")
        if path == last_checkpoint:
            return path
        tick = time.monotonic()
        state = collect_rank_state(
            {"format": "action-sft-state-v1", "cursor": cursor.state_dict(), "evaluation": history}
        )
        record_qlora_phase("checkpoint_begin", reset_peak=True)
        trainer.save(path, data_state=state)
        trainer.finalize_checkpoints()
        read_adapter_metadata(path, expected_identity=codec.identity, require_resume=True, manager=lifecycle.manager)
        record_qlora_phase("checkpoint_complete", reset_peak=True)
        clocks["checkpoint_seconds"] += time.monotonic() - tick
        last_checkpoint = path
        return path

    def score(update):
        tick = time.monotonic()
        record_qlora_phase("evaluation_begin", reset_peak=True)
        metrics, rows = evaluate(policy, objective, validation, config, callback)
        preparation = preparation_observations(policy)
        if preparation is not None:
            _json(output / "evaluation_preparation.json", preparation)
        backbone_reuse = observations(policy)
        if backbone_reuse is not None:
            _json(output / "evaluation_backbone.json", backbone_reuse)
        record_qlora_phase("evaluation_complete", reset_peak=True)
        clocks["evaluation_seconds"] += time.monotonic() - tick
        history.append({"step": update, "metrics": metrics})
        _json(output / f"evaluation-{update:06d}.json", {"step": update, "metrics": metrics, "rows": rows})
        return metrics

    try:
        if step == 0 and not history:
            score(0)
            if find_target(history, rl) is not None:
                path = save(0)
                target = {
                    "step": 0,
                    "checkpoint": path,
                    "seconds": time.monotonic() - started,
                    "timing_scope": "uninterrupted",
                }
        while step < rl.max_steps and not (target is not None and rl.stop_on_target):
            tick = time.monotonic()
            batch = [train[cursor.next_index()] for _ in range(rl.micro_batch_size)]
            metrics = trainer.train(batch)
            if metrics.get("optim/update_outcome") != "applied" or trainer.iteration() != step + 1:
                raise RuntimeError("native action update did not commit exactly one optimizer step")
            step += 1
            torch.cuda.synchronize(device)
            update_seconds = time.monotonic() - tick
            clocks["training_seconds"] += update_seconds
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"step": step, "update_seconds": update_seconds, **metrics}, allow_nan=False) + "\n"
                )
            reason = watcher.should_stop()
            boundary = step == rl.max_steps or step == rl.pause_after_step or bool(reason)
            evaluated = score(step) if step % rl.evaluation_every == 0 or boundary else None
            if evaluated is not None or boundary or (rl.save_every and step % rl.save_every == 0):
                path = save(step)
                if evaluated is not None and target is None and find_target(history, rl) is not None:
                    target = {
                        "step": step,
                        "checkpoint": path,
                        "seconds": None if attempt.resume_checkpoint else time.monotonic() - started,
                        "timing_scope": "resumed-not-uninterrupted" if attempt.resume_checkpoint else "uninterrupted",
                    }
            if boundary or (target is not None and rl.stop_on_target):
                break
        codec.validate_frozen_base()
        codec.assert_live(optimizer=optimizer)
        export_started = time.monotonic()
        record_qlora_phase("portable_export_begin", reset_peak=True)
        prior_completion = output / "completion.json"
        prior = json.loads(prior_completion.read_text()) if prior_completion.is_file() else {}
        if prior.get("step") == step and prior.get("checkpoint") == last_checkpoint:
            portable = prior["portable"]
            read_adapter_metadata(portable, expected_identity=codec.identity, manager=lifecycle.manager)
        else:
            portable = publish_adapter_artifact(last_checkpoint, str(output / "portable" / f"iter_{step:06d}"))
        record_qlora_phase("portable_export_complete", reset_peak=True)
        clocks["portable_export_seconds"] = time.monotonic() - export_started
        result = {
            "format": "cosmos-native-action-sft-v1",
            "step": step,
            "checkpoint": last_checkpoint,
            "portable": portable,
            "target": target,
            "resumed": bool(attempt.resume_checkpoint),
            "timing_scope": "current-process-only" if attempt.resume_checkpoint else "uninterrupted",
            "horizon_reached": step == rl.max_steps,
            "history": history,
            "clocks": clocks,
            "workflow_seconds": time.monotonic() - started,
            "training_source": train.identity,
            "validation_source": validation.identity,
        }
        _json(output / "completion.json", result)
        return result
    finally:
        trainer.finalize_checkpoints()
        if objective.close is not None:
            objective.close()


def main():
    from cosmos_framework.posttrain.utils.entry import cli_flags
    from cosmos_framework.utils.config import load_config

    config_path, opts, flags = cli_flags(
        "Native supervised action adaptation",
        "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py",
        extra_flags={"--validate-data": "Check native input files and split separation without constructing a model."},
    )
    config = load_config(config_path, ["--", *opts])
    if flags["validate_data"]:
        validate_config(config)
        train, validation = prepare_sources(config)
        print(
            json.dumps(
                {
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train": train.identity,
                    "validation": validation.identity,
                },
                sort_keys=True,
            )
        )
        return
    if flags["validate_manifest"]:
        validate_config(config)
        print("Native action SFT configuration accepted; model/data/kernel checks run at construction.")
        return
    print(json.dumps(run(config, config_path=config_path, opts=opts), sort_keys=True))


if __name__ == "__main__":
    main()
