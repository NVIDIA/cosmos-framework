# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit behavior verification preserves validation, updates and resume boundaries."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.batched_joint_ppo import (
    BatchedFixedEpisodePolicy,
    assert_checkpoint_identity_boundary,
    batch_recipe,
)
from cosmos_framework.posttrain.tests.helpers.fixed_episode import collect_fixed

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_behavior_verification_requires_an_explicit_boolean(invalid):
    with pytest.raises(ValueError, match="rl.verify_behavior_logprobs must be a boolean"):
        batch_recipe(verify_behavior_logprobs=invalid)


def test_disabling_behavior_replay_skips_only_the_extra_forward():
    recipe = batch_recipe()
    models = [BatchedFixedEpisodePolicy(), BatchedFixedEpisodePolicy()]
    samples = collect_fixed(models[0], recipe)
    learners = [
        JointPPOTrainer(model, replace(recipe, verify_behavior_logprobs=verify))
        for model, verify in zip(models, (True, False), strict=True)
    ]
    receipts = [learner.prepare_ppo_targets(samples) for learner in learners]
    assert receipts[0]["behavior_verified"] is True
    assert receipts[0]["max_logprob_delta"] <= models[0].ppo_logprob_tolerance
    assert receipts[1]["behavior_verified"] is False
    assert receipts[1]["max_logprob_delta"] is None
    assert receipts[1]["initial_clip_fraction"] is None
    assert receipts[0]["target_diagnostics"] == receipts[1]["target_diagnostics"]
    assert len(models[0].scored_batches) == 2
    assert models[1].scored_batches == []
    updates = [learner.train_batch(item["batch_id"]) for learner, item in zip(learners, receipts, strict=True)]
    assert updates[0] == updates[1]
    assert all(enabled for enabled, _ in models[1].scored_batches)
    for actual, expected in zip(models[1].net.parameters(), models[0].net.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for learner, receipt in zip(learners, receipts, strict=True):
        learner.clear_batch(receipt["batch_id"])


def test_default_verification_still_rejects_changed_behavior_scores():
    recipe = batch_recipe()
    model = BatchedFixedEpisodePolicy()
    samples = collect_fixed(model, recipe)
    last = samples[-1].transitions[0]
    stats = last.algo_extra[STATS_KEY]
    last.algo_extra[STATS_KEY] = replace(stats, old_logprob=stats.old_logprob + 0.2)
    checked = JointPPOTrainer(model, recipe)
    with pytest.raises(ValueError, match="on-policy mismatch"):
        checked.prepare_ppo_targets(samples)
    unchecked = JointPPOTrainer(BatchedFixedEpisodePolicy(), replace(recipe, verify_behavior_logprobs=False))
    receipt = unchecked.prepare_ppo_targets(samples)
    assert receipt["behavior_verified"] is False and receipt["max_logprob_delta"] is None
    assert unchecked.model.scored_batches == []
    unchecked.clear_batch(receipt["batch_id"])


@pytest.mark.parametrize("corruption", ["trace", "version", "statistics"])
def test_disabled_behavior_replay_still_rejects_late_invalid_records(corruption):
    recipe = batch_recipe(verify_behavior_logprobs=False)
    model = BatchedFixedEpisodePolicy()
    samples = collect_fixed(model, recipe)
    last = samples[-1].transitions[0]
    if corruption == "trace":
        trace = last.policy_info["gaussian_trace"]
        last.policy_info["gaussian_trace"] = replace(trace, row=len(trace.observations))
        message = "row is out of range"
    elif corruption == "version":
        last.policy_info["weight_version"] += 1
        message = "stale behavior version"
    else:
        last.algo_extra[STATS_KEY] = replace(last.algo_extra[STATS_KEY], old_logprob=float("nan"))
        message = "invalid old statistics"
    learner = JointPPOTrainer(model, recipe)
    with pytest.raises(ValueError, match=message):
        learner.prepare_ppo_targets(samples)
    assert model.scored_batches == []
    assert learner.progress() == {"optimizer_step": 0, "publication_version": 0}


def test_checkpoint_rejects_changed_verification_mode_without_mutation(tmp_path):
    recipe = batch_recipe()
    assert_checkpoint_identity_boundary(
        JointPPOTrainer(BatchedFixedEpisodePolicy(), recipe),
        JointPPOTrainer(BatchedFixedEpisodePolicy(), recipe),
        JointPPOTrainer(BatchedFixedEpisodePolicy(), replace(recipe, verify_behavior_logprobs=False)),
        str(tmp_path / "verified-checkpoint"),
    )
