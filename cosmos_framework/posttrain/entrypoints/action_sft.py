# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native supervised driving/robotics training with family-owned inputs and objectives.

The existing LLMTrainer owns every backward, optimizer update and checkpoint transaction.
This entry owns deterministic data traversal and completed-update evaluation boundaries.
It supports one process and no gradient accumulation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def prepare_sources(config):
    """Validate owned input files and split separation before allocating a model.

    Native task data factories are model-independent and receive None during
    admission; processors/packers remain attached to the constructed policy.
    """
    from cosmos_framework.posttrain.entrypoints.action_common import ActionSource, load_factory

    data_factory = load_factory(config.data_factory)
    train = ActionSource(data_factory(config, "train", None))
    validation = ActionSource(data_factory(config, "validation", None))
    for attribute in ("groups", "ids", "media_hashes"):
        left, right = getattr(train.dataset, attribute, None), getattr(validation.dataset, attribute, None)
        if left is not None and right is not None:
            if attribute == "media_hashes":
                left, right = left.values(), right.values()
            if set(left) & set(right):
                raise ValueError(f"action SFT train/validation overlap in {attribute}")
    return train, validation


def preflight_backbone_runtime(config, device):
    """Reject unavailable selected backbone execution before reading data or model weights."""
    mode = getattr(getattr(config.model, "backbone_adaptation", None), "mode", None)
    if mode not in {"nf4_lora", "nf4_frozen", "bf16_lora", "bf16_frozen"}:
        raise ValueError("native action backbone mode must be nf4_lora|nf4_frozen|bf16_lora|bf16_frozen")

    from importlib.metadata import PackageNotFoundError

    from cosmos_framework.posttrain.model.quantization.family import require_family_runtime

    try:
        require_family_runtime(device, quantized=mode.startswith("nf4_"))
    except PackageNotFoundError as error:
        if error.name != "bitsandbytes":
            raise
        raise ValueError(
            "native action NF4 requires bitsandbytes; install the `qlora` dependency group or follow "
            "docker/posttrain/native_action/README.md"
        ) from error


def training_identity(config, rl, objective):
    """Bind what changes the training trajectory, including OmegaConf container leaves.

    As in ``entrypoints/sft.py``: validation data, evaluation cadence/seed/sample count and
    evaluation input reuse do not change the trajectory and are free to differ on resume; run
    length enters only as the horizon of a cosine schedule without ``cycle_steps``.
    """
    import torch

    from cosmos_framework.posttrain.entrypoints.action_common import factory_identity
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    adaptation = config.model.backbone_adaptation
    factories = factory_identity(config)
    factories.pop("evaluation_callback", None)
    horizon = (rl.lr_schedule.cycle_steps or rl.max_steps) if rl.lr_schedule.kind == "cosine" else None
    return canonical_config_value(
        {
            "format": "cosmos-native-action-sft-v1",
            "factories": factories,
            "objective": objective.identity,
            "seed": rl.seed,
            # Training-only model settings; reload/inference identities omit them.
            "adapter_seed": adaptation.seed,
            "adapter_dropout": adaptation.dropout,
            "activation_checkpointing": getattr(config.model, "activation_checkpointing", None),
            "runtime": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "optimizer_state_dtype": "float32",
            },
            "optimizer": asdict(rl.optimizer),
            "lr_schedule": {**asdict(rl.lr_schedule), "resolved_cycle_steps": horizon},
        }
    )


def training_provenance() -> dict[str, str]:
    """Optional build provenance, as in the Qwen path; recorded but never compared."""
    names = {"code_revision": "COSMOS_TRAINING_CODE_REVISION", "wheel_sha256": "COSMOS_TRAINING_WHEEL_SHA256"}
    return {key: os.environ[name] for key, name in names.items() if os.environ.get(name)}


def run(config, *, config_path: str, opts: list[str]):
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import publish_adapter_artifact, read_adapter_metadata
    from cosmos_framework.posttrain.config.build import build_optimizer, build_sft_lr_scheduler
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from cosmos_framework.posttrain.entrypoints.action_common import (
        configure_tf32,
        evaluate,
        load_factory,
        single_process_group,
        validate_config,
        write_json,
    )
    from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import (
        install_evaluation_input_reuse,
        preparation_observations,
        validate_evaluation_input_reuse,
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
    configure_tf32(config)
    torch.cuda.set_device(device)
    with single_process_group():
        attempt = pin_run_attempt(config, opts, out_dir(config))
        output = Path(attempt.out_dir)
        output.mkdir(parents=True, exist_ok=True)
        dump_resolved(config, str(output), provenance=True)
        lifecycle = CheckpointLifecycle.from_config(
            config, rl, run_out_dir=str(output), resume_checkpoint=attempt.resume_checkpoint
        )
        if attempt.resume_checkpoint:
            # Refuse a different adapter before preparing data or allocating the model; the codec
            # compares the complete identity once the model is built.
            adaptation = getattr(config.model, "backbone_adaptation", None)
            expected = (
                None
                if adaptation is None
                else {"adapter": {"mode": adaptation.mode, "rank": adaptation.rank, "alpha": adaptation.alpha}}
            )
            read_adapter_metadata(
                attempt.resume_checkpoint,
                expected_identity=expected,
                require_resume=True,
                manager=lifecycle.manager,
            )
        preflight_backbone_runtime(config, device)
        record_qlora_phase("data_preparation_begin", reset_peak=True)
        train, validation = prepare_sources(config)
        validate_evaluation_input_reuse(config, rl, validation)
        record_qlora_phase("data_preparation_complete", reset_peak=True)
        misc.set_random_seed(seed=rl.seed, by_rank=False)
        record_qlora_phase("model_loading_begin", reset_peak=True)
        policy = load_factory(config.policy_factory)(config, device=device, output=str(output))
        codec = getattr(policy.net, "_cosmos_adapter_checkpoint_codec", None)
        if codec is None:
            raise RuntimeError("action SFT family did not install the native checkpoint codec")
        record_qlora_phase("model_loading_complete", reset_peak=True)
        cursor = IndexedCursor(train, shuffle=True, seed=rl.seed, cursor_resume="strict", label="action SFT")
        install_evaluation_input_reuse(policy, config, rl)
        install_reuse(policy, config, rl)
        objective = load_factory(config.objective_factory)(policy, config)
        objective.identity = {
            **objective.identity,
            "training_source": train.identity,
            "micro_batch_size": rl.micro_batch_size,
        }
        callback = load_factory(config.evaluation_callback) if config.evaluation_callback else None
        codec.bind_training_identity(training_identity(config, rl, objective))
        codec.bind_provenance(training_provenance())
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

        def save(update):
            nonlocal last_checkpoint
            if update != trainer.iteration():
                raise RuntimeError("action checkpoint boundary differs from the native optimizer clock")
            path = str(Path(lifecycle.checkpoint_root) / f"iter_{update:06d}")
            if path == last_checkpoint:
                return
            tick = time.monotonic()
            state = collect_rank_state(
                {"format": "action-sft-state-v1", "cursor": cursor.state_dict(), "evaluation": history}
            )
            record_qlora_phase("checkpoint_begin", reset_peak=True)
            trainer.save(path, data_state=state)
            trainer.finalize_checkpoints()
            read_adapter_metadata(
                path, expected_identity=codec.identity, require_resume=True, manager=lifecycle.manager
            )
            record_qlora_phase("checkpoint_complete", reset_peak=True)
            clocks["checkpoint_seconds"] += time.monotonic() - tick
            last_checkpoint = path

        def score(update):
            tick = time.monotonic()
            record_qlora_phase("evaluation_begin", reset_peak=True)
            metrics, rows = evaluate(policy, objective, validation, config, callback)
            preparation = preparation_observations(policy)
            if preparation is not None:
                write_json(output / "evaluation_preparation.json", preparation)
            backbone_reuse = observations(policy)
            if backbone_reuse is not None:
                write_json(output / "evaluation_backbone.json", backbone_reuse)
            record_qlora_phase("evaluation_complete", reset_peak=True)
            clocks["evaluation_seconds"] += time.monotonic() - tick
            history.append({"step": update, "metrics": metrics})
            write_json(output / f"evaluation-{update:06d}.json", {"step": update, "metrics": metrics, "rows": rows})

        try:
            if step == 0 and not history:
                score(0)
            while step < rl.max_steps:
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
                boundary = step == rl.max_steps or bool(reason)
                evaluated = step % rl.evaluation_every == 0 or boundary
                if evaluated:
                    score(step)
                if evaluated or (rl.save_every and step % rl.save_every == 0):
                    save(step)
                if boundary:
                    break
            codec.validate_frozen_base()
            codec.assert_live(optimizer=optimizer)
            export_started = time.monotonic()
            record_qlora_phase("portable_export_begin", reset_peak=True)
            prior_completion = output / "completion.json"
            prior = json.loads(prior_completion.read_text()) if prior_completion.is_file() else {}
            if prior.get("step") == step and prior.get("checkpoint") == last_checkpoint:
                portable = prior["portable"]
                # Build provenance of an earlier process may differ; it is never compared.
                expected = {key: value for key, value in codec.identity.items() if key != "provenance"}
                read_adapter_metadata(portable, expected_identity=expected, manager=lifecycle.manager)
            else:
                portable = publish_adapter_artifact(last_checkpoint, str(output / "portable" / f"iter_{step:06d}"))
            record_qlora_phase("portable_export_complete", reset_peak=True)
            clocks["portable_export_seconds"] = time.monotonic() - export_started
            result = {
                "format": "cosmos-native-action-sft-v1",
                "step": step,
                "checkpoint": last_checkpoint,
                "portable": portable,
                "resumed": bool(attempt.resume_checkpoint),
                "timing_scope": "current-process-only" if attempt.resume_checkpoint else "uninterrupted",
                "horizon_reached": step == rl.max_steps,
                "history": history,
                "clocks": clocks,
                "workflow_seconds": time.monotonic() - started,
                "training_source": train.identity,
                "validation_source": validation.identity,
            }
            write_json(output / "completion.json", result)
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
    from cosmos_framework.posttrain.entrypoints.action_common import validate_config

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
