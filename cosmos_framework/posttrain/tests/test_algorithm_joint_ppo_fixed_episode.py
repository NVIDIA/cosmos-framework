# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-duration episodes preserve terminal facts while masking padded decisions."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.algorithm.ppo.losses import normalize_advantages
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.embodied.window import collect_window
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.schema import EnvironmentFeedback
from cosmos_framework.posttrain.tests.helpers.fixed_episode import (
    FixedEpisodeEnvironment,
    FixedEpisodePolicy,
    FixedEpisodeSession,
    collect_fixed,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _recipe(**kwargs):
    return JointPPOConfig(
        num_envs=2,
        window_decisions=3,
        global_minibatch=kwargs.pop("global_minibatch", 6),
        microbatch=kwargs.pop("microbatch", 1),
        dual_clip=None,
        loss_reduction="fixed_episode_mean",
        time_limit_bootstrap="zero",
        **kwargs,
    )


def _reference(samples, successes, recipe):
    """Independent RLinf-style recurrence and control-step mask, before any update."""
    old_values = torch.tensor([sample.transitions[0].algo_extra[STATS_KEY].old_value for sample in samples])
    rewards = torch.tensor([sample.transitions[0].reward["final"] for sample in samples])
    values = old_values.reshape(3, len(successes))
    rewards = rewards.reshape_as(values)
    returns = torch.zeros_like(values)
    valid = torch.zeros_like(values, dtype=torch.bool)
    weights = torch.empty_like(values)
    for lane, success in enumerate(successes):
        first_done_chunk = (success - 1) // FixedEpisodePolicy.horizon
        gae = torch.tensor(0.0)
        for index in reversed(range(first_done_chunk + 1)):
            next_value = values[index + 1, lane] if index < first_done_chunk else 0.0
            delta = rewards[index, lane] + recipe.gamma * next_value - values[index, lane]
            gae = delta + recipe.gamma * recipe.gae_lambda * gae
            returns[index, lane] = gae + values[index, lane]
        valid[: first_done_chunk + 1, lane] = True
        weights[:, lane] = 3 * FixedEpisodePolicy.horizon / success
    raw = returns - values
    advantages = (raw - raw[valid].mean()) / (raw[valid].std() + 1e-5)
    return returns.flatten(), advantages.flatten(), valid.flatten(), weights.flatten()


def test_targets_use_inclusive_control_endings_and_complete_episode_weights():
    recipe = _recipe()
    model = FixedEpisodePolicy()
    samples = collect_fixed(model, recipe)
    learner = JointPPOTrainer(model, recipe)
    expected_returns, advantages, valid, weights = _reference(samples, (2, 8), recipe)
    receipt = learner.prepare_ppo_targets(samples)
    rows = receipt["target_diagnostics"]
    for index, (sample, row) in enumerate(zip(samples, rows, strict=True)):
        assert row["loss_mask"] == bool(valid[index])
        assert row["loss_weight"] == pytest.approx(float(weights[index]))
        assert row["returns"] == pytest.approx(float(expected_returns[index]), abs=1e-7)
        assert row["normalized_advantage"] == pytest.approx(float(advantages[index]) if valid[index] else 0.0, abs=1e-6)
        assert sample.transitions[0].feedback.control_terminated is not None
    assert samples[0].transitions[0].done
    assert samples[0].transitions[0].feedback.episode_complete is False
    # A chunk-based valid length would give 3 here; the second control step ended the task.
    assert rows[0]["loss_weight"] != 3.0
    learner.clear_batch(receipt["batch_id"])


def test_fixed_episode_update_matches_weighted_full_grid_and_not_valid_mean():
    recipe = _recipe()
    model = FixedEpisodePolicy()
    samples = collect_fixed(model, recipe)
    learner = JointPPOTrainer(model, recipe)
    receipt = learner.prepare_ppo_targets(samples)
    expected = FixedEpisodePolicy()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter for name, parameter in expected.net.named_parameters() if not name.startswith("value.")
                ],
                "lr": recipe.actor_lr,
            },
            {"params": expected.net.value.parameters(), "lr": recipe.value_lr},
        ],
        betas=recipe.optimizer.betas,
        eps=recipe.optimizer.eps,
        weight_decay=recipe.optimizer.weight_decay,
    )
    returns, advantages, valid, weights = _reference(samples, (2, 8), recipe)
    losses = []
    for index, sample in enumerate(samples):
        scores = expected.score_ppo_decision(sample.transitions[0])
        if valid[index]:
            old = sample.transitions[0].algo_extra[STATS_KEY].old_logprob
            actor = -(scores.logprobs - old).exp().mean() * advantages[index]
            critic = torch.nn.functional.huber_loss(scores.values, returns[index : index + 1], delta=recipe.huber_delta)
            losses.append(actor + critic)
        else:
            losses.append(scores.zero_loss)
    weighted = (torch.stack(losses) * weights).mean()
    naive = torch.stack(losses)[valid].mean()
    assert not torch.isclose(weighted, naive, atol=1e-5)
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(expected.net.parameters(), recipe.optimizer.grad_clip_norm)
    optimizer.step()
    result = learner.train_batch(receipt["batch_id"])
    assert result["optimizer_step"] == 1
    assert result["metrics"]["joint_loss"] == pytest.approx(float(weighted.detach()), abs=1e-6)
    for actual, reference in zip(model.net.parameters(), expected.net.parameters(), strict=True):
        torch.testing.assert_close(actual, reference, atol=3e-8, rtol=1e-6)
    learner.clear_batch(receipt["batch_id"])


def test_invalid_late_episode_is_rejected_before_forward_then_recovers():
    recipe = _recipe()
    model = FixedEpisodePolicy()
    samples = collect_fixed(model, recipe)
    learner = JointPPOTrainer(model, recipe)
    last = samples[-1].transitions[0]
    original = last.feedback
    last.feedback = replace(original, episode_complete=False)
    calls = []
    with model.net.register_forward_pre_hook(lambda *_: calls.append(True)):
        with pytest.raises(ValueError, match="complete physical episode"):
            learner.prepare_ppo_targets(samples)
    assert not calls
    last.feedback = original
    receipt = learner.prepare_ppo_targets(samples)
    learner.train_batch(receipt["batch_id"])
    learner.clear_batch(receipt["batch_id"])


def test_all_masked_minibatch_has_zero_loss_and_finite_metrics():
    recipe = _recipe(global_minibatch=1)
    model = FixedEpisodePolicy()
    learner = JointPPOTrainer(model, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(model, recipe))
    result = learner.train_batch(receipt["batch_id"])
    zero = [row for row in result["optimizer_metrics"] if row["joint_loss"] == 0]
    assert len(zero) == 2
    assert all(row["clip_fraction"] == row["ratio_mean"] == 0 for row in zero)
    assert result["optimizer_step"] == recipe.decisions
    learner.clear_batch(receipt["batch_id"])


def test_feedback_flags_and_fixed_reduction_configuration_are_strict():
    with pytest.raises(ValueError, match="both control"):
        EnvironmentFeedback((0.0,), 1, control_terminated=(False,))
    with pytest.raises(ValueError, match=r"control_truncated\[0\]"):
        EnvironmentFeedback((0.0,), 1, (False,), (1,))
    with pytest.raises(ValueError, match="one flag"):
        EnvironmentFeedback((0.0,), 1, (), ())
    with pytest.raises(ValueError, match="time_limit_bootstrap"):
        JointPPOConfig(loss_reduction="fixed_episode_mean")
    with pytest.raises(ValueError, match="loss_reduction"):
        JointPPOConfig(loss_reduction="typo")
    with pytest.raises(ValueError, match="dual_clip"):
        JointPPOConfig(dual_clip=1.0)
    assert _recipe().dual_clip is None
    assert JointPPOConfig().dual_clip == 3.0


def test_native_advantage_statistics_match_float32_with_float64_negative_control():
    recipe = _recipe()
    model = FixedEpisodePolicy()
    samples = collect_fixed(model, recipe, successes=(4,))
    returns, _, valid, _ = _reference(samples, (4,), recipe)
    old_values = torch.tensor([sample.transitions[0].algo_extra[STATS_KEY].old_value for sample in samples])
    raw = returns - old_values
    expected = (raw[valid] - raw[valid].mean()) / (raw[valid].std() + 1e-5)
    learner = JointPPOTrainer(model, recipe)
    receipt = learner.prepare_ppo_targets(samples)
    actual = torch.tensor([row["normalized_advantage"] for row in receipt["target_diagnostics"]])[valid]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    former = normalize_advantages(raw, valid, correction=1, eps=1e-5, eps_inside_sqrt=False)
    assert float((former[valid] - expected).abs().max()) > 1e-6
    learner.clear_batch(receipt["batch_id"])


def test_late_invalid_control_feedback_commits_nothing_and_recovers():
    class CorruptSession(FixedEpisodeSession):
        def step_chunk(self, actions):
            results = super().step_chunk(actions)
            results[-1] = replace(results[-1], control_truncated=(False, False, 1))
            return results

    recipe = _recipe()
    model = FixedEpisodePolicy()
    collector = PPOWindowCollector(recipe, 0, recipe.num_envs, environment=FixedEpisodeEnvironment())
    collector.bind_rollout(model)
    before = collector.state_dict()
    collector.begin_window(0)
    error = None
    try:
        collect_window(CorruptSession(), collector, version=0, decisions=3)
    except ValueError as caught:
        error = caught
    assert error is not None and "control_truncated[2]" in str(error)
    assert collector.window_statistics()["reward"] == 0
    collector.abort_window()
    assert collector.state_dict() == before
    collector.begin_window(0)
    report = collect_window(FixedEpisodeSession(), collector, version=0, decisions=3)
    samples = collector.finish_window(report)
    assert len(samples) == recipe.decisions
    collector.close()


def _distributed_fixed_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        recipe = JointPPOConfig(
            num_envs=3,
            window_decisions=3,
            global_minibatch=9,
            microbatch=1,
            dual_clip=None,
            loss_reduction="fixed_episode_mean",
            time_limit_bootstrap="zero",
        )
        successes = ((4,), (2, 8))[rank]
        model = FixedEpisodePolicy(distributed=True)
        learner = JointPPOTrainer(model, recipe)
        samples = collect_fixed(model, recipe, successes=successes, replica=rank)
        receipt = learner.prepare_ppo_targets(samples)
        expected = FixedEpisodePolicy()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for name, parameter in expected.net.named_parameters()
                        if not name.startswith("value.")
                    ],
                    "lr": recipe.actor_lr,
                },
                {"params": expected.net.value.parameters(), "lr": recipe.value_lr},
            ],
            betas=recipe.optimizer.betas,
            eps=recipe.optimizer.eps,
            weight_decay=recipe.optimizer.weight_decay,
        )
        losses = []
        for replica, events in enumerate(((4,), (2, 8))):
            batch = collect_fixed(expected, recipe, successes=events, replica=replica)
            returns, advantages, valid, weights = _reference(batch, events, recipe)
            for index, sample in enumerate(batch):
                scores = expected.score_ppo_decision(sample.transitions[0])
                if valid[index]:
                    old = sample.transitions[0].algo_extra[STATS_KEY].old_logprob
                    actor = -(scores.logprobs - old).exp().mean() * advantages[index]
                    critic = torch.nn.functional.huber_loss(
                        scores.values, returns[index : index + 1], delta=recipe.huber_delta
                    )
                    losses.append((actor + critic) * weights[index])
                else:
                    losses.append(scores.zero_loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.net.parameters(), recipe.optimizer.grad_clip_norm)
        optimizer.step()
        result = learner.train_batch(receipt["batch_id"])
        assert result["metrics"]["joint_loss"] == pytest.approx(float(loss.detach()), abs=1e-6), (
            result["metrics"]["joint_loss"],
            float(loss.detach()),
            result["optimizer_metrics"],
        )
        for actual, reference in zip(model.net.parameters(), expected.net.parameters(), strict=True):
            torch.testing.assert_close(actual, reference, atol=3e-8, rtol=1e-6)
        learner.clear_batch(receipt["batch_id"])
    finally:
        dist.destroy_process_group()


def test_uneven_fixed_episode_shards_and_masked_microbatches_match_global_update(tmp_path: Path):
    torch.multiprocessing.spawn(
        _distributed_fixed_worker, args=(f"file://{tmp_path / 'fixed-rendezvous'}",), nprocs=2, join=True
    )
