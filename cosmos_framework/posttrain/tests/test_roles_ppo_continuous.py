# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public collector-to-learner checks for the joint PPO path."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.algorithm.ppo.losses import ppo_policy_loss
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.model.gr00t.ppo import TRACE_KEY
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
    GaussianPolicy,
    LocalWindowGateway,
    collect_scored,
    make_collector,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_late_invalid_trace_prevents_forward_and_cache_can_be_rebuilt() -> None:
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    model = GaussianPolicy()
    collector = make_collector(model, LocalWindowGateway(), recipe, replica=0, n_lanes=2)
    learner = JointPPOTrainer(model, recipe)
    batch = collect_scored(collector, 0)
    original = batch[-1].transitions[0].policy_info[TRACE_KEY]
    before = {name: parameter.detach().clone() for name, parameter in model.net.named_parameters()}
    invalid = (
        (replace(original, sampler_identity="corrupt"), "sampler identity"),
        (replace(original, latent_after=original.latent_after[:, :16, :7]), "native"),
        (
            replace(
                original, denoise_index=torch.full_like(original.denoise_index, 4), time=torch.ones_like(original.time)
            ),
            "must lie in",
        ),
        (replace(original, time=original.time + 0.25), "recorded time"),
        (replace(original, latent_after=torch.full_like(original.latent_after, float("nan"))), "nonfinite"),
    )
    forwarded = []
    with model.net.register_forward_pre_hook(lambda *_: forwarded.append(True)):
        for trace, message in invalid:
            batch[-1].transitions[0].policy_info[TRACE_KEY] = trace
            with pytest.raises(ValueError, match=message):
                learner.prepare_ppo_targets(batch)
        assert not forwarded
        batch[-1].transitions[0].policy_info[TRACE_KEY] = original
        receipt = learner.prepare_ppo_targets(batch)
        assert forwarded
    for name, parameter in model.net.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    learner.abort_batch(receipt["batch_id"])
    assert learner.model.ppo_padding is None


def test_dual_clip_bounds_negative_advantage_loss_and_gradient() -> None:
    current = torch.tensor([4.0], requires_grad=True)
    old, advantages, mask = torch.zeros(1), -torch.ones(1), torch.ones(1, dtype=torch.bool)
    unclipped, _ = ppo_policy_loss(current, old, advantages, mask)
    clipped, _ = ppo_policy_loss(current, old, advantages, mask, dual_clip=3)
    assert float(unclipped.detach()) > 3
    torch.testing.assert_close(clipped, torch.tensor(3.0))
    clipped.backward()
    assert float(current.grad) == 0


def test_checkpoint_refuses_changed_processor_and_collector_suite(tmp_path):
    from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint

    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    model = GaussianPolicy()
    collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
    learner = JointPPOTrainer(model, recipe, source_identity={"processor": "original-source"})
    receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
    learner.train_batch(receipt["batch_id"])
    learner.clear_batch(receipt["batch_id"])
    learner.set_publication_version(1)
    checkpoint = str(tmp_path / "checkpoint")
    save_policy_checkpoint(learner, checkpoint, data=collector)
    changed = JointPPOTrainer(GaussianPolicy(), recipe, source_identity={"processor": "changed-source"})
    before = changed.evaluation_state()
    with pytest.raises(RuntimeError, match="identity differs"):
        changed.load(checkpoint)
    assert changed.evaluation_state() == before
    from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
    from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig

    other_suite = make_collector(
        model,
        LocalWindowGateway(),
        recipe,
        0,
        2,
        environment=LiberoEnvironment(LiberoConfig(suite="libero_10", execution_horizon=8, max_episode_steps=720)),
    )
    before = other_suite.state_dict()
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        other_suite.load_state_dict(collector.state_dict())
    assert other_suite.state_dict() == before


def _uneven_joint_worker(rank: int, rendezvous: str, result_path: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist

    from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
    from cosmos_framework.posttrain.model.gr00t.ppo import CONDITION_KEY

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        recipe = JointPPOConfig(num_envs=6, window_decisions=1, global_minibatch=6, microbatch=1)
        model = GaussianPolicy(distributed=True)
        learner = JointPPOTrainer(model, recipe)
        count, offset = ((2, 0), (4, 2))[rank]
        collector = make_collector(model, LocalWindowGateway(count, 0, offset), recipe, rank, count)
        samples = collect_scored(collector, 0)
        receipt = learner.prepare_ppo_targets(samples)
        result = learner.train_batch(receipt["batch_id"])
        assert result["optimizer_step"] == 1
        learner.clear_batch(receipt["batch_id"])

        # Independent single-process loss over the same two actor-local normalized shards.
        reference = GaussianPolicy()
        actor = list(reference.net.action_head.actor.parameters())
        value = list(reference.net.action_head.value_head.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": actor, "lr": recipe.actor_lr}, {"params": value, "lr": recipe.value_lr}],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        losses = []
        for replica, (n_lanes, first) in enumerate(((2, 0), (4, 2))):
            source = make_collector(reference, LocalWindowGateway(n_lanes, 0, first), recipe, replica, n_lanes)
            batch = collect_scored(source, 0)
            returns = torch.tensor([sum(t.transitions[0].feedback.control_rewards) for t in batch])
            old_values = torch.tensor([t.transitions[0].algo_extra[STATS_KEY].old_value for t in batch])
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std(correction=1) + 1e-5)
            for sample, target, advantage in zip(batch, returns, advantages, strict=True):
                step = sample.transitions[0]
                scores = reference.ppo_score(step.obs.representations[CONDITION_KEY], step.policy_info[TRACE_KEY])
                index = step.policy_info["sample_index"]
                ratio = (scores.logprobs[index : index + 1] - step.algo_extra[STATS_KEY].old_logprob).exp()
                critic = torch.nn.functional.huber_loss(scores.values[index : index + 1], target.reshape(1), delta=10)
                losses.append(-ratio.mean() * advantage + critic)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.net.parameters(), 1.0)
        optimizer.step()
        maximum = 0.0
        for actual, expected in zip(model.net.parameters(), reference.net.parameters(), strict=True):
            maximum = max(maximum, float((actual - expected).abs().max().detach()))
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
        if rank == 0:
            Path(result_path).write_text(str(maximum))
    finally:
        dist.destroy_process_group()


def test_uneven_joint_update_with_empty_microbatches_matches_global_loss(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    result = tmp_path / "comparison.txt"
    mp.spawn(_uneven_joint_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(result)), nprocs=2, join=True)
    assert float(result.read_text()) <= 2e-7


def test_refactor_preserves_frozen_reference_records_targets_and_update() -> None:
    import json

    from cosmos_framework.posttrain.tests.helpers.parity import assert_loss_parity

    def capture(recipe: JointPPOConfig) -> dict:
        model = GaussianPolicy()
        collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
        learner = JointPPOTrainer(model, recipe)
        receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
        try:
            update = learner.train_batch(receipt["batch_id"])
            assert learner.publication_version() == 0
        finally:
            learner.clear_batch(receipt["batch_id"])
        assert learner.model.ppo_padding is None
        return {
            "receipt": {key: value for key, value in receipt.items() if key != "batch_id"},
            "update": update,
        }

    def compare(actual: object, expected: object, path: str = "record") -> None:
        assert type(actual) is type(expected), path
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys(), path
            for key, value in expected.items():
                compare(actual[key], value, f"{path}.{key}")
        elif isinstance(expected, list):
            assert len(actual) == len(expected), path
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                compare(left, right, f"{path}[{index}]")
        elif isinstance(expected, float):
            # CPU math libraries differ at float32 rounding precision. Keep structure, zero
            # drift, clocks and learning rates exact; update norms have no absolute slack.
            if expected == 0 or path.endswith(("_step", "_lr", "_defined", "_decisions")) or ".learning_rates." in path:
                assert actual == expected, path
            else:
                assert_loss_parity([actual], [expected], atol=0 if "parameter_update_l2" in path else 1e-6)
        else:
            assert actual == expected, path

    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    frozen = json.loads((Path(__file__).parent / "fixtures/gr00t_ppo_refactor_reference.json").read_text())
    expected = {
        "receipt": {key: value for key, value in frozen["receipt"].items() if key != "batch_id"},
        "update": {key: value for key, value in frozen["update"].items() if key != "metrics"},
    }
    # The frozen receipt predates the explicit behavior-verification field.
    expected["receipt"]["behavior_verified"] = True
    actual = capture(recipe)
    metrics = actual["update"].pop("metrics")
    compare(actual, expected)
    # Opposite minibatch policy losses nearly cancel in the window mean. Compare the
    # individual updates to the golden, then require exact aggregation in this runtime.
    assert metrics.keys() == frozen["update"]["metrics"].keys()
    updates = actual["update"]["optimizer_metrics"]
    for name, value in metrics.items():
        reference = (
            float(actual["update"]["optimizer_step"])
            if name == "policy_optimizer_step"
            else sum(frame[name] for frame in updates) / len(updates)
        )
        assert value == reference, name
    # Exercise actual target construction and optimizer updates as negative controls.
    changed_targets = capture(replace(recipe, gamma=recipe.gamma / 2))
    with pytest.raises(AssertionError):
        compare(changed_targets["receipt"], expected["receipt"])
    changed_updates = capture(replace(recipe, optimizer=replace(recipe.optimizer, lr=recipe.actor_lr * 2)))
    with pytest.raises(AssertionError):
        compare(
            changed_updates["update"]["parameter_update_l2"],
            expected["update"]["parameter_update_l2"],
            path="update.parameter_update_l2",
        )


@pytest.mark.parametrize("reward", [None, {"final": float("nan")}, {"final": True}])
def test_missing_or_invalid_training_reward_rejects_whole_batch_before_scoring(reward, monkeypatch) -> None:
    from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY

    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    model = GaussianPolicy()
    learner = JointPPOTrainer(model, recipe)
    collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
    batch = collect_scored(collector, 0)
    valid = batch[-1].transitions[0].reward
    invalid = {FINAL_REWARD_KEY: reward["final"]} if isinstance(reward, dict) and "final" in reward else reward
    batch[-1].transitions[0].reward = invalid
    before = learner.evaluation_state()
    forwarded = []
    with model.net.register_forward_pre_hook(lambda *_: forwarded.append(True)):
        with pytest.raises((TypeError, ValueError), match=batch[-1].id):
            learner.prepare_ppo_targets(batch)
    assert not forwarded and learner.evaluation_state() == before
    assert learner.model.ppo_padding is None and learner._global_ids == ()
    batch[-1].transitions[0].reward = valid
    receipt = learner.prepare_ppo_targets(batch)
    assert learner.train_batch(receipt["batch_id"])["optimizer_step"] == 2
    learner.clear_batch(receipt["batch_id"])


@pytest.mark.parametrize("stage", ["hash", "state_dict"])
def test_parameter_reader_failure_releases_helper_aliases_and_preserves_exception_chain(stage: str) -> None:
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import parameter_failure_release

    learner = JointPPOTrainer(GaussianPolicy(), JointPPOConfig())
    before = learner.evaluation_state()
    result = parameter_failure_release(learner, stage)
    assert result["released_bindings"] > 0 and result["exception_chain_preserved"]
    assert learner.evaluation_state() == before
