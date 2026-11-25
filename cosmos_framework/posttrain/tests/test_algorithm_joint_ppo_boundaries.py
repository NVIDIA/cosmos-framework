# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Collection boundaries preserve task facts and fixed-reference PPO updates."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.algorithm.reward.decision import build_decision_reward_provider
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.embodied.sims.libero.reward import LiberoDecisionReward
from cosmos_framework.posttrain.embodied.window import ChunkResult, collect_window
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Transition
from cosmos_framework.posttrain.tests.helpers.joint_ppo import (
    GaussianEnvironment,
    GaussianPolicy,
    GaussianSession,
    reward_provider,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class BoundarySession(GaussianSession):
    """Reset lanes independently, exposing different final and next-episode observations."""

    def __init__(self):
        super().__init__(n_lanes=4)
        self.episodes = [0] * self.n_lanes
        self.positions = [0.0] * self.n_lanes

    def observe(self):
        return [
            {"features": np.array([lane + 1, self.positions[lane]], dtype=np.float32)} for lane in range(self.n_lanes)
        ]

    def sources(self, window):
        return [
            {**source, "episode_id": f"episode-{lane}:{self.episodes[lane]}"}
            for lane, source in enumerate(super().sources(window))
        ]

    def step_chunk(self, actions):
        self.turn += 1
        results = []
        for lane, action in enumerate(actions):
            self.positions[lane] += 0.25
            final = self.observe()[lane]
            truncated = (lane, self.turn) in ((0, 1), (1, 3))
            terminated = (lane, self.turn) == (2, 2)
            if terminated or truncated:
                self.episodes[lane] += 1
                self.positions[lane] = -5.0
            results.append(ChunkResult(action.copy(), self.observe()[lane], final, (0.0,), terminated, truncated))
        return results


class RecordingPolicy(GaussianPolicy):
    """Observe genuine objective forwards without changing their values or gradients."""

    def __init__(self):
        super().__init__()
        self.training_ids = []

    def score_ppo_decision(self, step: Transition):
        if torch.is_grad_enabled():
            self.training_ids.append(step.source.sample_id)
        return super().score_ppo_decision(step)


def _collector(model, recipe):
    collector = PPOWindowCollector(recipe, 0, recipe.num_envs, environment=GaussianEnvironment())
    collector.bind_rollout(model)
    return collector


def _collect(collector, session, version, decisions):
    collector.begin_window(version)
    try:
        report = collect_window(session, collector, version=version, decisions=decisions)
        return collector.finish_window(report)
    except BaseException:
        collector.abort_window()
        raise


@pytest.mark.parametrize("mode", ["value", "zero"])
def test_timeout_targets_preserve_reset_observations_and_libero_failure_rewards(mode):
    recipe = JointPPOConfig(num_envs=4, window_decisions=3, global_minibatch=4, microbatch=1, time_limit_bootstrap=mode)
    model, session = GaussianPolicy(), BoundarySession()
    collector = _collector(model, recipe)
    learner = JointPPOTrainer(model, recipe)
    reward = RewardWorker([build_decision_reward_provider(LiberoDecisionReward("libero_10"))], training_group_size=1)
    try:
        samples = reward.score(_collect(collector, session, 0, recipe.window_decisions))
        receipt = learner.prepare_ppo_targets(samples)
        diagnostics = {row["sample_id"]: row for row in receipt["target_diagnostics"]}
        timeouts = []
        terminals = []
        live_ends = []
        for sample in samples:
            step = sample.transitions[0]
            stats = step.algo_extra[STATS_KEY]
            row = diagnostics[sample.id]
            if step.truncated:
                timeouts.append(step)
                assert step.done and not row["terminated"]
                assert step.reward[FINAL_REWARD_KEY] == -1.0
                predicted = model.bootstrap_values(
                    [step.next_obs.representations["raw"]], seed=recipe.seed, weight_version=0
                )[0]
                assert predicted != 0.0
                expected = predicted if mode == "value" else 0.0
                assert stats.bootstrap_value == pytest.approx(expected)
                assert row["returns"] == pytest.approx(-1.0 + recipe.gamma * expected)
                assert step.next_obs.representations["raw"]["features"][1] > 0.0
            elif step.done:
                terminals.append(step)
                assert row["terminated"] and not row["truncated"]
                assert stats.bootstrap_value == row["returns"] == 0.0
            elif step.collection_end:
                live_ends.append(step)
                predicted = model.bootstrap_values(
                    [step.next_obs.representations["raw"]], seed=recipe.seed, weight_version=0
                )[0]
                assert predicted != 0.0
                assert stats.bootstrap_value == pytest.approx(predicted)
                assert row["returns"] == pytest.approx(recipe.gamma * predicted)
            else:
                assert stats.bootstrap_value == 0.0
        assert {step.collection_end for step in timeouts} == {False, True}
        assert terminals and live_ends
        assert session.observe()[1]["features"][1] < 0.0
        learner.clear_batch(receipt["batch_id"])
    finally:
        collector.close()


@pytest.mark.parametrize("update_policy", ["parameter", "fp32_master"])
def test_reference_shuffle_is_fixed_across_publications_and_checkpoint_resume(tmp_path, update_policy):
    recipe = JointPPOConfig(num_envs=2, window_decisions=4, global_minibatch=4, microbatch=1)
    recipe.optimizer.update_policy = update_policy
    model, session = RecordingPolicy(), GaussianSession()
    learner = JointPPOTrainer(model, recipe)
    collector = _collector(model, recipe)
    control_model = RecordingPolicy()
    control_model.net.load_state_dict(model.net.state_dict())
    control = JointPPOTrainer(control_model, recipe)
    control_collector, control_session = _collector(control_model, recipe), GaussianSession()
    reward = RewardWorker([reward_provider()], training_group_size=1)
    expected = torch.randperm(recipe.decisions, generator=torch.Generator().manual_seed(recipe.seed)).tolist()

    def train_window(active_learner, active_collector, active_session, version):
        samples = reward.score(_collect(active_collector, active_session, version, recipe.window_decisions))
        positions = {sample.id: index for index, sample in enumerate(samples)}
        active_learner.model.training_ids.clear()
        receipt = active_learner.prepare_ppo_targets(samples)
        active_learner.train_batch(receipt["batch_id"])
        actual = [positions[key] for key in active_learner.model.training_ids]
        active_learner.clear_batch(receipt["batch_id"])
        active_learner.set_publication_version(version + 1)
        assert actual == expected
        return actual

    restored_collector = None
    try:
        first = train_window(learner, collector, session, 0)
        assert train_window(control, control_collector, control_session, 0) == first
        assert learner.evaluation_state() == control.evaluation_state()
        checkpoint_state = learner.evaluation_state()
        checkpoint = str(tmp_path / "checkpoint")
        assert any(parameter.grad is not None for parameter in learner.opt.param_groups[0]["params"])
        save_policy_checkpoint(learner, checkpoint, data=collector)
        # FP32-master runs release the completed update's gradients before writing; default
        # runs keep the upstream save path untouched.
        released = all(parameter.grad is None for group in learner.opt.param_groups for parameter in group["params"])
        assert released is (update_policy == "fp32_master")
        if update_policy == "fp32_master":
            assert all(parameter.grad is None for parameter in learner.opt.masters)
        assert learner.evaluation_state() == checkpoint_state
        second = train_window(learner, collector, session, 1)
        assert train_window(control, control_collector, control_session, 1) == second
        assert learner.evaluation_state() == control.evaluation_state()
        learner.finalize_checkpoints()
        restored = JointPPOTrainer(RecordingPolicy(), recipe)
        restored.load(checkpoint)
        assert restored.evaluation_state() == checkpoint_state
        restored_collector = _collector(restored.model, recipe)
        restored_collector.load_state_dict(restored.checkpoint_manager.load_component(checkpoint, "data"))
        resumed = train_window(restored, restored_collector, GaussianSession(), 1)
        assert first == second == resumed
        assert restored.optimizer_step == learner.optimizer_step

        changed = JointPPOTrainer(GaussianPolicy(), replace(recipe, time_limit_bootstrap="zero"))
        before = changed.evaluation_state()
        with pytest.raises(RuntimeError, match="identity differs"):
            changed.load(checkpoint)
        assert changed.evaluation_state() == before

    finally:
        collector.close()
        control_collector.close()
        if restored_collector is not None:
            restored_collector.close()


@pytest.mark.parametrize("update_policy", ["parameter", "fp32_master"])
def test_checkpoint_memory_cleanup_never_discards_partial_gradients(tmp_path, update_policy):
    recipe = JointPPOConfig()
    recipe.optimizer.update_policy = update_policy
    learner = JointPPOTrainer(GaussianPolicy(), recipe)
    gradients = []
    for group in learner.opt.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
            gradients.append((parameter, parameter.grad))
    learner._accum = 1
    with pytest.raises(RuntimeError, match="completed update boundary"):
        learner.save(str(tmp_path / "unsafe"))
    assert all(
        parameter.grad is gradient and torch.equal(gradient, torch.ones_like(gradient))
        for parameter, gradient in gradients
    )
    assert not (tmp_path / "unsafe").exists()


# The OptimizerConfig fields present before the opt-in eight-bit/master work. A default
# Joint PPO identity must contain exactly these, so pre-existing checkpoints resume exactly.
_PRE_QUANTIZED_OPTIMIZER_IDENTITY_KEYS = {
    "name",
    "impl",
    "lr",
    "weight_decay",
    "betas",
    "eps",
    "train_keys",
    "grad_clip_norm",
    "force_finite_grads",
    "warmup_steps",
    "decay_groups",
    "lr_multipliers",
    "muon_momentum",
    "muon_ns_steps",
    "muon_rms_match",
}


def test_default_identity_keeps_the_pre_existing_optimizer_key_set():
    from dataclasses import replace

    recipe = JointPPOConfig()
    assert set(recipe.identity()["optimizer"]) == _PRE_QUANTIZED_OPTIMIZER_IDENTITY_KEYS
    # Each opt-in field joins the identity only when it is actually selected.
    master = replace(recipe, optimizer=replace(recipe.optimizer, update_policy="fp32_master"))
    assert set(master.identity()["optimizer"]) == _PRE_QUANTIZED_OPTIMIZER_IDENTITY_KEYS | {"update_policy"}
    eightbit = replace(
        recipe,
        optimizer=replace(
            recipe.optimizer,
            name="adamw_bnb_8bit",
            update_policy="fp32_master",
            bnb_min_8bit_size=16384,
            bnb_fp32_state_names=["head.bias"],
        ),
    )
    assert set(eightbit.identity()["optimizer"]) == _PRE_QUANTIZED_OPTIMIZER_IDENTITY_KEYS | {
        "update_policy",
        "bnb_min_8bit_size",
        "bnb_fp32_state_names",
    }
