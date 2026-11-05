# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Continuous-action validation journals established by real collection, updates and DCP saves."""

import pytest

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest
from cosmos_framework.posttrain.entrypoints.joint_ppo import CollectorCheckpointState, validation_identity
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
    GaussianPolicy,
    LocalWindowGateway,
    collect_scored,
    make_collector,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_failed_continuous_validation_restores_journal_without_repeating_updates(tmp_path):
    for split in ("train", "eval"):
        fixed_manifest(split).write(tmp_path / f"{split}.json")
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config

    config = make_config()
    config.rl = JointPPOConfig(num_envs=2, global_minibatch=16)
    config.environment.config.train_manifest = str(tmp_path / "train.json")
    config.environment.config.eval_manifest = str(tmp_path / "eval.json")
    recipe = config.rl
    model = GaussianPolicy()
    collector = make_collector(model, LocalWindowGateway(terminal_decision=15), recipe, 0, 2)
    learner = JointPPOTrainer(model, recipe)
    learner.configure_validation(validation_identity(config))
    receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
    learner.train_batch(receipt["batch_id"])
    learner.clear_batch(receipt["batch_id"])
    learner.set_publication_version(1)
    lifecycle = CheckpointLifecycle(learner.checkpoint_manager, str(tmp_path / "checkpoints"), None)
    coordinator = lifecycle.coordinator(str(tmp_path), learner, data=CollectorCheckpointState(collector))
    transaction = ValidationTransaction(learner, coordinator, [], set())
    reasons = frozenset({ValidationTrigger.PERIODIC})

    def interrupted(step, triggers):
        assert step == learner.optimizer_step and triggers == reasons
        raise RuntimeError("injected evaluation interruption")

    before = learner.evaluation_state()
    with pytest.raises(RuntimeError, match="injected evaluation"):
        transaction.run(learner.optimizer_step, reasons, interrupted)
    restored = JointPPOTrainer(GaussianPolicy(), recipe)
    checkpoint = coordinator.path(learner.optimizer_step)
    recovery = CheckpointLifecycle(restored.checkpoint_manager, str(tmp_path / "checkpoints"), checkpoint)
    recovery.load_trainer(restored)
    restored.configure_validation(validation_identity(config))
    assert restored.pending_validation_step() == learner.optimizer_step
    assert restored.evaluation_state() == before

    def evaluate(step, triggers):
        value = sum(parameter.detach().square().sum().item() for parameter in restored.model.net.parameters())
        return ValidationReport(
            step, 1, {"val/parameter_norm_squared": value}, fixed_manifest("eval").digest, triggers, 0.0
        )

    resumed = ValidationTransaction(restored, recovery.coordinator(str(tmp_path), restored), [], set())
    report = resumed.resume(learner.optimizer_step, reasons, evaluate)
    assert report.metrics["val/parameter_norm_squared"] > 0
    assert restored.pending_validation_step() is None
    assert restored.evaluation_state() == before
