# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The native flow objective retains its action denominator across validation resumes."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTValidationConfig
from cosmos_framework.posttrain.control.validation.continuation import PROTOCOL, topology_identity
from cosmos_framework.posttrain.entrypoints.sft import loop
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.roles.env.flow_sft import build_flow_sft_env
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.tests.test_flow_sft import Policy, dataset, pack
from cosmos_framework.posttrain.tests.test_roles_trainer_sft_validation import _assert_equal

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_public_flow_loop_preserves_complete_action_weighted_validation_and_next_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import termination
    from cosmos_framework.posttrain.utils import report as reporting

    train, validation = dataset(tmp_path / "train", "train-episode", 4), dataset(tmp_path / "val", "val-episode", 3)
    monkeypatch.setattr(termination, "TerminationWatch", lambda: SimpleNamespace(should_stop=lambda: None))
    outcomes = {}
    for label, chunk in (("whole", None), ("chunks", 1)):
        reports, consumed = [], []
        monkeypatch.setattr(
            reporting,
            "make_reporters",
            lambda *a, **kw: [
                SimpleNamespace(
                    start=lambda: None,
                    finish=lambda: None,
                    report=lambda metrics, *, step: reports.append((step, copy.deepcopy(metrics))),
                )
            ],
        )
        for _ in range(3):
            rl = SFTConfig(
                task="flow_bc",
                flow_training_prefix=2,
                flow_action_dim=1,
                max_steps=2,
                data=train,
                save_async=False,
                save_every=1,
                checkpoint_format="dcp",
                validation=SFTValidationConfig(
                    enabled=True,
                    before_training=True,
                    batch_size=2,
                    save_checkpoint=True,
                    data=validation,
                    continuation_batches=chunk,
                ),
            )
            config = SimpleNamespace(
                rl=rl, job=SimpleNamespace(wandb_mode="disabled", project="flow", group="continuation", name=label)
            )
            data = build_flow_sft_env(train, 7)
            held = build_flow_sft_env(validation, rl.validation.seed, training=False)
            policy = Policy()
            objective = make_objective(pack, training_prefix=2, action_dim=1, sft_metrics=True)
            manager = CheckpointManager(checkpoint_format="dcp")
            trainer = LLMTrainer(
                policy,
                objective,
                torch.optim.AdamW(policy.net.parameters(), lr=0.01),
                grad_accum=1,
                checkpoint_manager=manager,
                validation_state=ValidationState(),
                objective_evaluator=evaluate_sft_batch,
            )
            output = tmp_path / label
            checkpoint = manager.resolve_resume("auto", str(output / "checkpoints"))
            if checkpoint:
                trainer.load(checkpoint)
                data.load_state_dict(manager.load_component(checkpoint, "data"))
            identity = {
                "enabled": True,
                "source_fingerprint": held.source_fingerprint,
                "objective_identity": dict(objective.identity),
                "batch_size": 2,
                "max_samples": None,
                "seed": rl.validation.seed,
                "schedule": {"before_training": True, "every_steps": 0, "train_total_steps": 2},
            }
            if chunk is not None:
                identity["continuation"] = {"protocol": PROTOCOL, "topology": topology_identity(policy)}
            trainer.configure_validation(identity)
            original = held.get_validation_batch

            def observed(index, batch_size, max_samples):
                batch = original(index, batch_size, max_samples)
                consumed.append((trainer.iteration(), index, [row.id for row in batch]))
                return batch

            held.get_validation_batch = observed
            units = {
                "data": roles.env("data", body=data),
                "validation": roles.env("validation", body=held),
                "policy": roles.trainer("policy", body=trainer),
            }
            result = loop(units, config, str(output))
            trainer.finalize_checkpoints()
            if "preempted" not in result:
                break
            assert trainer.validation_continuation["next_batch"] == 1
        assert "preempted" not in result
        frames = [
            (step, {key: value for key, value in metrics.items() if not key.startswith("timer/")})
            for step, metrics in reports
        ]
        validations = [(step, metrics) for step, metrics in frames if "val/loss_avg" in metrics]
        assert [step for step, _ in validations] == [0, 2]
        for _, metrics in validations:
            assert metrics["val/valid_action_count"] == 6
            assert metrics["val/valid_sequence_count"] == 3
            assert "val/valid_token_count" not in metrics
        outcomes[label] = {
            "model": copy.deepcopy(policy.net.state_dict()),
            "optimizer": copy.deepcopy(trainer.opt.state_dict()),
            "rng": trainer._trainer_state()["rng"],
            "data": data.state_dict(),
            "consumed": consumed,
            "reports": frames,
        }
    _assert_equal(outcomes["chunks"], outcomes["whole"])
