# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Scoring-prefix and finite-horizon reward boundaries for the long-task recipe."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.ppo.batch import prepare_sequence_targets
from cosmos_framework.posttrain.algorithm.ppo.records import PPOStepSource, PPOTargetStep
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LiberoTrial,
    LiberoTrialManifest,
    fixed_manifest,
    libero_seed,
)
from cosmos_framework.posttrain.embodied.sims.libero.reward import decision_reward
from cosmos_framework.posttrain.model.gr00t.ppo import (
    PREFIX_SAMPLER_IDENTITY,
    Gr00tPPOCondition,
    Gr00tPPOTrace,
    validate_trace,
)
from cosmos_framework.posttrain.model.ppo import selected_transition_logprobs

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_native_trace_scores_eight_actions_independently_of_unused_suffix():
    generator = torch.Generator().manual_seed(1234)
    sample = torch.randn((2, 40, 132), generator=generator)
    mean, std = torch.zeros_like(sample), torch.ones_like(sample)
    score = selected_transition_logprobs(sample, mean, std, score_horizon=8)
    expected = torch.distributions.Normal(mean[:, :8, :7], std[:, :8, :7]).log_prob(sample[:, :8, :7]).sum((1, 2))
    torch.testing.assert_close(score, expected)
    changed = sample.clone()
    changed[:, 8:] += 5
    torch.testing.assert_close(selected_transition_logprobs(changed, mean, std, score_horizon=8), score, rtol=0, atol=0)
    assert not torch.equal(
        selected_transition_logprobs(changed, mean, std, score_horizon=16),
        selected_transition_logprobs(sample, mean, std, score_horizon=16),
    )
    features = torch.zeros((2, 3, 4))
    condition = Gr00tPPOCondition(
        features,
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 1, 132),
        torch.zeros(2, dtype=torch.long),
    )
    trace = Gr00tPPOTrace(mean, sample, torch.zeros(2, dtype=torch.long), torch.zeros(2), PREFIX_SAMPLER_IDENTITY, 8)
    validate_trace(condition, trace)
    with pytest.raises(ValueError, match="sampler identity"):
        validate_trace(condition, replace(trace, score_horizon=16))


def test_window_and_episode_budgets_are_independent_and_spatial_identity_is_preserved() -> None:
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as libero10
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config as spatial

    old, new = spatial(), libero10()
    assert (old.model.execution_horizon, new.model.execution_horizon) == (16, 8)
    assert new.model.backbone_buffer_dtype is None
    assert old.rl.decisions == new.rl.decisions == 128
    assert new.rl.window_decisions * new.model.execution_horizon == 128
    assert new.environment.config.max_episode_steps == 720
    assert old.rl.identity()["algorithm"] == "joint-ppo-v3"
    assert "suite" not in old.rl.identity()
    assert fixed_manifest("train").digest == "b894b2a90a7283d951ef38d2745d438b62f1b71ac7dcc441210a39b3ea8a94d0"
    assert fixed_manifest("eval").digest == "aed0f9076237eef54de65e63184306d15006940122e51cb0318b4a88de379826"
    train, periodic = fixed_manifest("train", "libero_10"), fixed_manifest("eval", "libero_10")
    assert len(train.cases) == 500 and set(periodic.cases) <= set(train.cases)
    trials = LiberoTrialManifest(
        "libero_10",
        tuple(LiberoTrial(case, seed) for case in (*train.cases[:4], *train.cases[-4:]) for seed in (0, 7, 13)),
    ).trials
    assert len(trials) == len({trial.id for trial in trials}) == 24
    assert (
        len(
            {
                libero_seed(
                    "libero_10", "evaluation-policy", trial.seed, trial.case.task_id, trial.case.init_state_id, decision
                )
                for trial in trials
                for decision in range(90)
            }
        )
        == len(trials) * 90
    )


@pytest.mark.parametrize(
    "success,timeout,raw,expected",
    [
        (False, False, (0.0,) * 8, 0.0),
        (True, False, (0.0, 1.0), 0.0),
        (False, True, (0.0,), -1.0),
    ],
)
def test_recipe_reward_preserves_raw_feedback_and_time_limit_zero_bootstrap(success, timeout, raw, expected):
    reward = decision_reward(raw, suite="libero_10", terminated=success, truncated=timeout)
    assert reward == expected
    step = PPOTargetStep(
        PPOStepSource("sample", "episode", "fragment", 0),
        (reward,),
        (0.3, 0.0 if success or timeout else 0.7),
        (True,),
        success,
        timeout,
        True,
    )
    target = prepare_sequence_targets([step], gamma=0.99, gae_lambda=0.95)[0]
    assert target.returns[0] == pytest.approx(expected if success or timeout else 0.99 * 0.7)
    assert decision_reward(raw, suite="libero_spatial", terminated=success, truncated=timeout) == sum(raw)
