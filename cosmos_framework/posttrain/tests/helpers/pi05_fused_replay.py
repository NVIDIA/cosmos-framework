# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test-only capture and evidence for fixed-record FSDP2 fused AdamW replay."""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.posttrain.config.config_joint_ppo import resolve_joint_config
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.entrypoints.joint_ppo import setup_policy as ordinary_setup_policy
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.posttrain.utils.object_store import file_sha256


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def qualification_config(path: str, output: Path):
    from cosmos_framework.utils.config import load_config

    config = load_config(path, [])
    config.rl.validation.enabled = False
    config.rl.resume = "off"
    config.rl.max_steps = 12
    config.job.group = "pi05_fused_optimizer_qualification"
    config.job.name = "fixed_real_window"
    config.job.wandb_mode = "disabled"
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(output)
    config = resolve_joint_config(config)
    recipe = instantiate(config.rl)
    if (recipe.num_envs, recipe.window_decisions, recipe.global_minibatch, recipe.microbatch) != (64, 48, 512, 16):
        raise ValueError("qualification requires real 64x48 records, global512 and micro16")
    if recipe.optimizer.name != "adamw" or recipe.optimizer.impl != "fused":
        raise ValueError("qualification config must already select AdamW impl=fused")
    if recipe.verify_behavior_logprobs:
        raise ValueError("fixed-record optimizer replay requires verify_behavior_logprobs=False")
    return config, recipe


def capture_targets(self, trajectories):
    """Observe a real, fully validated source batch without changing its records or training."""
    receipt = JointPPOTrainer.prepare_ppo_targets(self, trajectories)
    root = Path(self._qualification_records)
    rank = dist.get_rank()

    def save():
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"rank-{rank:05d}.pt"
        if path.exists():
            raise FileExistsError(f"fixed-record tape already exists: {path}")
        if len(trajectories) != 384 or any(sample.pool_leases for sample in trajectories):
            raise ValueError("capture requires 384 durable CPU records without live transport leases per rank")
        torch.save(tuple(trajectories), path)
        write_json(
            root / f"rank-{rank:05d}.json",
            {
                "rank": rank,
                "world_size": dist.get_world_size(),
                "records": len(trajectories),
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "batch_id": receipt["batch_id"],
                "recipe_identity": self.recipe.identity(),
                "model_identity": self.model.ppo_identity(),
                "progress": self.progress(),
            },
        )

    run_rank_converged(save, "qualification real-record capture")
    return receipt


def capture_setup_policy(config, output, resume, recipe):
    bodies = ordinary_setup_policy(config, output, resume, recipe)
    trainer = bodies["policy"]
    trainer._qualification_records = str(Path(output).parent / "records")
    trainer.prepare_ppo_targets = MethodType(capture_targets, trainer)
    return bodies


def capture(config, recipe, output: Path):
    import ray

    from cosmos_framework.posttrain import entrypoints
    from cosmos_framework.posttrain.control.loop.decision_window import collect_decision_window
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical

    entry = entrypoints.joint_ppo
    entry.validate_config(config, 8)
    with ExitStack() as cleanup:
        cleanup.callback(ray.shutdown)
        with patch.object(entry, "setup_policy", capture_setup_policy):
            units = entry.assemble(config, recipe, output / "capture-run", None, 8, evaluation=False)
        for name in ("rollout", "collector", "sim"):
            cleanup.callback(lambda name=name: resolve_all(units[name].close()))
        cleanup.callback(CheckpointLifecycle.finalize_units, {"policy": units["policy"]})
        policy = units["policy"]
        weight_sync(units["rollout"], policy, validation_step=0)
        raw = collect_decision_window(units["collector"], units["sim"], version=0, decisions=48)
        scored = units["rm"].score_with_provider_report(raw)
        resolve_all(units["replay"].put_samples(scored))
        batch = units["replay"].get_exact_samples()
        receipt = resolve_identical(policy.prepare_ppo_targets(batch), operation="capture targets")
        progress = resolve_identical(policy.progress(), operation="capture unchanged learner counters")
        if progress != {"optimizer_step": 0, "publication_version": 0}:
            raise AssertionError("capture advanced the learner")
        resolve_all(policy.clear_batch(receipt["batch_id"]))
        shards = [json.loads((output / "records" / f"rank-{rank:05d}.json").read_text()) for rank in range(8)]
        if sum(item["records"] for item in shards) != 3072:
            raise AssertionError("capture lacks the complete real window")
        report = {
            "passed": True,
            "progress": progress,
            "records": 3072,
            "ranks": 8,
            "batch_id": receipt["batch_id"],
            "shards": shards,
            "scope": "One actual public collector/reward/prepare_targets window; zero optimizer updates.",
        }
        write_json(output / "records/capture.json", report)
        return report


def tensor_fingerprint(tensor):
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    value = local.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite parameter or optimizer state")
    return {
        "dtype": str(value.dtype),
        "global_shape": list(tensor.shape),
        "local_shape": list(value.shape),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy()).hexdigest(),
    }


def learner_snapshot(trainer):
    attestation = trainer.evaluation_state()  # Includes the production frozen-VLM guard.
    parameters = {
        name: parameter for name, parameter in trainer.model.net.named_parameters() if parameter.requires_grad
    }
    if len(parameters) != 216:
        raise AssertionError("expected the complete 216-parameter Pi05Heads inventory")
    parameter_rows, state_rows, scalar_steps = {}, {}, {}
    for name, parameter in parameters.items():
        parameter_rows[name] = tensor_fingerprint(parameter)
        state = trainer.opt.state.get(parameter, {})
        if trainer.optimizer_step and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise AssertionError(f"optimizer state inventory differs at {name}: {set(state)}")
        state_rows[name] = {key: tensor_fingerprint(value) for key, value in state.items()}
        if state:
            step = state["step"]
            step = step.to_local() if isinstance(step, DTensor) else step
            scalar_steps[name] = float(step.item())
            if scalar_steps[name] != trainer.optimizer_step:
                raise AssertionError(f"Adam step differs at {name}: {scalar_steps[name]}")
    names_by_id = {id(value): name for name, value in parameters.items()}
    groups = [
        {
            **{key: value for key, value in group.items() if key != "params"},
            "parameter_names": [names_by_id[id(value)] for value in group["params"]],
        }
        for group in trainer.opt.param_groups
    ]
    if len(groups) != 2 or any(group.get("fused") is not True for group in groups):
        raise AssertionError("both production actor/value groups must select fused=True")
    return {
        "progress": trainer.progress(),
        "parameters": parameter_rows,
        "optimizer": state_rows,
        "adam_steps": scalar_steps,
        "groups": groups,
        "attestation": attestation,
        "objective_identity": trainer.objective_identity,
        "all_finite": True,
    }


def replay_phase(
    config, recipe, output: Path, phase: str, records_root: Path | None = None, export_records: Path | None = None
):
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.utils.dist import init_local_process_group

    records_root = records_root if records_root is not None else output / "records"
    init_local_process_group()
    trainer = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8:
            raise ValueError("qualification requires exactly 8 FSDP2 ranks")
        random.seed(recipe.seed)
        np.random.seed(recipe.seed)
        torch.manual_seed(recipe.seed)
        dimensions = ParallelDims(world_size=world, dp_shard=world)
        dimensions.build_device_mesh("cuda")
        model = instantiate(config.policy, output=str(output / phase), parallel_dims=dimensions)
        lifecycle = CheckpointLifecycle.from_config(
            config, recipe, run_out_dir=str(output / phase), resume_checkpoint=None
        )
        trainer = JointPPOTrainer(
            model,
            recipe,
            parallel_dims=dimensions,
            checkpoint_manager=lifecycle.manager,
            source_identity=instantiate(config.environment).identity(),
        )
        checkpoint = output / "checkpoints/iter_000006"
        if phase == "resume":
            trainer.load(str(checkpoint))
        expected_start = 6 if phase == "resume" else 0
        if trainer.progress() != {"optimizer_step": expected_start, "publication_version": 0}:
            raise AssertionError("fixed-record replay restored unexpected learner counters")

        def load_records():
            manifest = json.loads((records_root / "capture.json").read_text())
            header = manifest["shards"][rank]
            path = records_root / f"rank-{rank:05d}.pt"
            if (
                header["rank"] != rank
                or header["recipe_identity"] != recipe.identity()
                or header["model_identity"] != model.ppo_identity()
            ):
                raise ValueError("record provenance differs from the fused qualification recipe/model")
            if file_sha256(path) != header["sha256"]:
                raise ValueError("fixed-record bytes changed after capture")
            # Trusted tape was produced by the public collector in this same qualification.
            records = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if len(records) != 384:
                raise ValueError("fixed-record shard must contain 384 decisions")
            return list(records), header

        records, header = run_rank_converged(load_records, "qualification tape validation")
        snapshots = {str(expected_start): learner_snapshot(trainer)}
        prepared = trainer.prepare_ppo_targets(records)  # Recompute through the public target owner.
        batch_id = prepared["batch_id"]
        if batch_id != header["batch_id"]:
            raise AssertionError("replaying the raw tape changed its public batch identity")
        if export_records is not None:
            from cosmos_framework.posttrain.tests.helpers.pi05_prepared import PREPARED16_INDICES, prepared_coverage

            def export():
                if rank != 0:
                    return
                selected = trainer._ppo_cache.select(batch_id, [records[index].id for index in PREPARED16_INDICES])
                selection = prepared_coverage(selected)
                if export_records.exists() or export_records.with_suffix(".json").exists():
                    raise FileExistsError(f"prepared16 export already exists: {export_records}")
                export_records.parent.mkdir(parents=True, exist_ok=True)
                torch.save(selected, export_records)
                write_json(
                    export_records.with_suffix(".json"),
                    {
                        "records_sha256": file_sha256(export_records),
                        "batch_id": batch_id,
                        "source_capture": {
                            "path": str((records_root / "capture.json").resolve()),
                            "sha256": file_sha256(records_root / "capture.json"),
                        },
                        "assets": {
                            "checkpoint_sha256": file_sha256(Path(model.config.checkpoint_path) / "model.safetensors"),
                            "tokenizer_sha256": file_sha256(model.config.tokenizer_path),
                            "norm_stats_sha256": file_sha256(model.config.norm_stats_path),
                        },
                        "selection": selection,
                    },
                )

            run_rank_converged(export, "compact prepared16 export")
        fused_calls = 0
        original_fused = torch._fused_adamw_

        def observed_fused(*args, **kwargs):
            nonlocal fused_calls
            fused_calls += 1
            return original_fused(*args, **kwargs)

        updates = []
        with patch.object(torch, "_fused_adamw_", observed_fused):
            for _ in range(2 if phase == "continuous" else 1):
                update = trainer.train_batch(batch_id)
                updates.append(update)
                snapshots[str(trainer.optimizer_step)] = learner_snapshot(trainer)
        trainer.clear_batch(batch_id)
        if not fused_calls:
            raise AssertionError("no actual fused AdamW call was observed")
        if phase == "save":
            trainer.save(str(checkpoint))
            trainer.finalize_checkpoints()
        expected_end = 6 if phase == "save" else 12
        if trainer.optimizer_step != expected_end:
            raise AssertionError("wrong final optimizer counter")
        report = {
            "passed": True,
            "phase": phase,
            "rank": rank,
            "world_size": world,
            "record_sha256": header["sha256"],
            "batch_id": batch_id,
            "fused_adamw_calls": fused_calls,
            "snapshots": snapshots,
            "updates": updates,
            "scope": "Fixed real behavior-version0 window replay. No second on-policy collection, no publication advancement and no environment-RNG restore claim.",
        }
        run_rank_converged(
            lambda: write_json(output / phase / f"rank-{rank:05d}.json", report), "qualification evidence write"
        )
        return {
            "phase": phase,
            "rank": rank,
            "optimizer_step": trainer.optimizer_step,
            "fused_adamw_calls": fused_calls,
        }
    finally:
        if trainer is not None:
            trainer.finalize_checkpoints()
        if dist.is_initialized():
            dist.destroy_process_group()


def compare_replays(output: Path):
    results = []
    for rank in range(8):
        reports = {
            phase: json.loads((output / phase / f"rank-{rank:05d}.json").read_text())
            for phase in ("continuous", "save", "resume")
        }
        if not all(
            report["passed"] and report["rank"] == rank and report["world_size"] == 8 for report in reports.values()
        ):
            raise ValueError(f"qualification rank {rank} did not complete")
        if (
            len({report["record_sha256"] for report in reports.values()}) != 1
            or len({report["batch_id"] for report in reports.values()}) != 1
        ):
            raise ValueError("qualification phases did not use the same immutable records")
        checks = {}
        for label, left, right in (
            ("step6_continuous_vs_save", reports["continuous"]["snapshots"]["6"], reports["save"]["snapshots"]["6"]),
            ("step6_saved_vs_restored", reports["save"]["snapshots"]["6"], reports["resume"]["snapshots"]["6"]),
            (
                "step12_continuous_vs_resumed",
                reports["continuous"]["snapshots"]["12"],
                reports["resume"]["snapshots"]["12"],
            ),
        ):
            checks[label] = {
                key: left[key] == right[key]
                for key in ("progress", "parameters", "optimizer", "adam_steps", "groups", "objective_identity")
            }
            checks[label]["frozen_vlm"] = left["attestation"]["frozen_vlm"] == right["attestation"]["frozen_vlm"]
            checks[label]["learner_rng"] = left["attestation"]["rng"] == right["attestation"]["rng"]
            checks[label]["scheduler"] = left["attestation"]["scheduler"] == right["attestation"]["scheduler"]
        results.append(
            {"rank": rank, "checks": checks, "passed": all(all(values.values()) for values in checks.values())}
        )
    report = {
        "passed": all(row["passed"] for row in results),
        "ranks": results,
        "parameters_per_rank": 216,
        "continuous_optimizer_steps": [0, 6, 12],
        "checkpoint_optimizer_step": 6,
        "restored_optimizer_step": 12,
        "publication_version": 0,
        "scope": "Learner/optimizer checkpoint equivalence on one fixed real window; not a new on-policy training run or environment-RNG qualification.",
    }
    write_json(output / "comparison.json", report)
    return report
