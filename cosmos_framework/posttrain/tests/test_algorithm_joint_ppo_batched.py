# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Optional batch scoring preserves public joint PPO arithmetic and DP schedules."""

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from cosmos_framework.posttrain.algorithm.joint_ppo.objective import make_joint_objective
from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, TARGET_KEY
from cosmos_framework.posttrain.model.protocol import BatchedDecisionPPOPolicy
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.batched_joint_ppo import (
    BatchedFixedEpisodePolicy,
    assert_checkpoint_identity_boundary,
    batch_recipe,
)
from cosmos_framework.posttrain.tests.helpers.fixed_episode import FixedEpisodePolicy, collect_fixed

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _objective(recipe):
    return make_joint_objective(
        policy_clip=recipe.policy_clip,
        value_clip=recipe.value_clip,
        dual_clip=recipe.dual_clip,
        huber_delta=recipe.huber_delta,
        loss_reduction=recipe.loss_reduction,
        microbatch=recipe.microbatch,
        identity=recipe.identity(),
    )


def _assert_update_matches(left, right, scalar, batched):
    assert left["optimizer_step"] == right["optimizer_step"]
    assert left["learning_rates"] == right["learning_rates"]
    assert left["metrics"].keys() == right["metrics"].keys()
    for key in left["metrics"]:
        assert right["metrics"][key] == pytest.approx(left["metrics"][key], abs=2e-6, rel=2e-6), key
    for actual, expected in zip(batched.net.parameters(), scalar.net.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-6)


@pytest.mark.parametrize("microbatch", [4, 16, 128])
def test_batch_replay_preserves_targets_loss_and_update_with_fixed_tail(microbatch):
    recipe = batch_recipe(microbatch=microbatch)
    scalar, batched = FixedEpisodePolicy(), BatchedFixedEpisodePolicy()
    assert not isinstance(scalar, BatchedDecisionPPOPolicy)
    assert isinstance(batched, BatchedDecisionPPOPolicy)
    samples = collect_fixed(scalar, recipe)
    learners = [JointPPOTrainer(model, recipe) for model in (scalar, batched)]
    prepared = [learner.prepare_ppo_targets(samples) for learner in learners]
    assert prepared[0]["target_diagnostics"] == prepared[1]["target_diagnostics"]
    assert prepared[1]["max_logprob_delta"] <= batched.ppo_logprob_tolerance
    updates = [learner.train_batch(item["batch_id"]) for learner, item in zip(learners, prepared, strict=True)]
    _assert_update_matches(*updates, scalar, batched)
    expected_forwards = (recipe.decisions + microbatch - 1) // microbatch
    assert [enabled for enabled, _ in batched.scored_batches].count(False) == expected_forwards
    assert [enabled for enabled, _ in batched.scored_batches].count(True) == expected_forwards
    assert all(len(ids) == microbatch for _, ids in batched.scored_batches)
    for learner, item in zip(learners, prepared, strict=True):
        learner.clear_batch(item["batch_id"])


@pytest.mark.parametrize("loss_reduction", ["decision_mean", "fixed_episode_mean"])
def test_batch_objective_keeps_position_denominator_and_masked_zero_graph(loss_reduction):
    recipe = batch_recipe()
    scalar, batched = FixedEpisodePolicy(), BatchedFixedEpisodePolicy()
    learner = JointPPOTrainer(scalar, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(scalar, recipe))
    samples = list(learner._ppo_cache.select(receipt["batch_id"], learner._ppo_cache.episode_ids(receipt["batch_id"])))
    valid = [sample for sample in samples if sample.transitions[0].algo_extra[TARGET_KEY].loss_mask]
    masked = [sample for sample in samples if not sample.transitions[0].algo_extra[TARGET_KEY].loss_mask]
    batch = [valid[0], masked[0], valid[-1]]
    batched.ppo_padding = scalar.ppo_padding
    objective = _objective(replace(recipe, loss_reduction=loss_reduction))
    old_loss, old_metrics = objective(scalar, batch)
    new_loss, new_metrics = objective(batched, batch)
    torch.testing.assert_close(new_loss, old_loss, atol=1e-7, rtol=1e-6)
    for key in old_metrics:
        assert float(new_metrics[key]) == pytest.approx(float(old_metrics[key]), abs=1e-7, rel=1e-6)
    assert new_metrics["ppo_valid_decisions"] == 2
    if loss_reduction == "fixed_episode_mean":
        assert new_metrics["ppo_decisions"] == 3
    naive_loss, _ = objective(batched, [valid[0], valid[-1]])
    assert not torch.isclose(new_loss, naive_loss, atol=1e-5)
    zero, metrics = objective(batched, masked)
    assert float(zero.detach()) == 0
    zero.backward()
    assert all(parameter.grad is not None and not bool(parameter.grad.any()) for parameter in batched.net.parameters())
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
    learner.clear_batch(receipt["batch_id"])


def test_late_invalid_batch_record_is_rejected_before_forward_and_recovers():
    recipe = batch_recipe()
    model = BatchedFixedEpisodePolicy()
    learner = JointPPOTrainer(model, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(model, recipe))
    samples = list(learner._ppo_cache.select(receipt["batch_id"], learner._ppo_cache.episode_ids(receipt["batch_id"])))[
        :4
    ]
    last = samples[-1].transitions[0]
    original = last.policy_info["gaussian_trace"]
    last.policy_info["gaussian_trace"] = replace(original, row=len(original.observations))
    before = len(model.scored_batches)
    with pytest.raises(ValueError, match="row is out of range"):
        _objective(recipe)(model, samples)
    assert len(model.scored_batches) == before
    last.policy_info["gaussian_trace"] = original
    loss, _ = _objective(recipe)(model, samples)
    loss.backward()
    assert len(model.scored_batches) == before + 1
    learner.clear_batch(receipt["batch_id"])


def test_on_policy_tail_excludes_padding_statistics_and_rejects_real_drift():
    recipe = batch_recipe(microbatch=4)
    model = BatchedFixedEpisodePolicy()
    learner = JointPPOTrainer(model, recipe)
    samples = collect_fixed(model, recipe)
    receipt = learner.prepare_ppo_targets(samples)
    original = model.ppo_padding
    model.ppo_padding = replace(
        original,
        algo_extra={**original.algo_extra, STATS_KEY: replace(original.algo_extra[STATS_KEY], old_logprob=100.0)},
    )
    assert learner._on_policy(samples)["max_logprob_delta"] <= model.ppo_logprob_tolerance
    last = samples[-1].transitions[0]
    original_stats = last.algo_extra[STATS_KEY]
    last.algo_extra[STATS_KEY] = replace(original_stats, old_logprob=original_stats.old_logprob + 0.1)
    with pytest.raises(ValueError, match="on-policy mismatch"):
        learner._on_policy(samples)
    last.algo_extra[STATS_KEY] = original_stats
    model.ppo_padding = original
    learner.clear_batch(receipt["batch_id"])


def test_batch_geometry_fails_before_forward():
    recipe = batch_recipe()
    model = BatchedFixedEpisodePolicy()
    learner = JointPPOTrainer(model, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(model, recipe))
    samples = list(learner._ppo_cache.select(receipt["batch_id"], learner._ppo_cache.episode_ids(receipt["batch_id"])))
    before = len(model.scored_batches)
    with pytest.raises(ValueError, match="microbatch covering"):
        _objective(recipe)(model, samples)
    with pytest.raises(ValueError, match="globally empty"):
        _objective(recipe)(model, [])
    model.ppo_padding = None
    with pytest.raises(ValueError, match="genuine padding"):
        _objective(recipe)(model, samples[:1])
    assert len(model.scored_batches) == before
    learner.clear_batch(receipt["batch_id"])


def test_checkpoint_rejects_changed_scoring_layout_before_state_mutation(tmp_path):
    recipe = batch_recipe()
    original = JointPPOTrainer(FixedEpisodePolicy(), recipe)
    changed = JointPPOTrainer(BatchedFixedEpisodePolicy(), recipe)
    # The same weights and recipe are insufficient: the replay layout is an exact-resume boundary.
    assert changed.ppo_identity() == original.ppo_identity()
    assert_checkpoint_identity_boundary(
        original, JointPPOTrainer(FixedEpisodePolicy(), recipe), changed, str(tmp_path / "scalar-checkpoint")
    )


def _distributed_batch_worker(rank, rendezvous, global_minibatch):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        recipe = batch_recipe(num_envs=3, global_minibatch=global_minibatch)
        scalar, batched = FixedEpisodePolicy(distributed=True), BatchedFixedEpisodePolicy(distributed=True)
        samples = collect_fixed(scalar, recipe, successes=((4,), (2, 8))[rank], replica=rank)
        learners = [JointPPOTrainer(model, recipe) for model in (scalar, batched)]
        prepared = [learner.prepare_ppo_targets(samples) for learner in learners]
        assert prepared[0]["target_diagnostics"] == prepared[1]["target_diagnostics"]
        batch_id = prepared[1]["batch_id"]
        cached = list(learners[1]._ppo_cache.select(batch_id, learners[1]._ppo_cache.episode_ids(batch_id)))[:4]
        last = cached[-1].transitions[0]
        original = last.policy_info["gaussian_trace"]
        if rank == 1:
            last.policy_info["gaussian_trace"] = replace(original, row=len(original.observations))
        before = len(batched.scored_batches)
        with pytest.raises(RuntimeError, match="row is out of range"):
            _objective(recipe)(batched, cached)
        assert len(batched.scored_batches) == before
        last.policy_info["gaussian_trace"] = original
        updates = [learner.train_batch(item["batch_id"]) for learner, item in zip(learners, prepared, strict=True)]
        _assert_update_matches(*updates, scalar, batched)
        assert all(len(ids) == recipe.microbatch for _, ids in batched.scored_batches)
        sizes = [None, None]
        dist.all_gather_object(sizes, len(batched.scored_batches))
        assert sizes[0] == sizes[1]
        if global_minibatch == 9:
            assert sizes[rank] == 4  # Two on-policy batches and two train batches, including rank 0's empty tail.
            if rank == 0:
                assert batched.scored_batches[-1][1] == (batched.ppo_padding.source.sample_id,) * recipe.microbatch
        elif rank == 0:
            masks = {sample.id: sample.transitions[0].algo_extra[TARGET_KEY].loss_mask for sample in cached}
            assert any(not masks[ids[0]] for enabled, ids in batched.scored_batches if enabled)
        for learner, item in zip(learners, prepared, strict=True):
            learner.clear_batch(item["batch_id"])
    finally:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.parametrize("global_minibatch", [1, 9])
def test_distributed_batch_uneven_empty_and_masked_shards_match_scalar_update(tmp_path, global_minibatch):
    torch.multiprocessing.spawn(
        _distributed_batch_worker,
        args=(f"file://{tmp_path / 'batch-rendezvous'}", global_minibatch),
        nprocs=2,
        join=True,
    )
