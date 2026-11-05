# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cross-decision targets, independent return checks and source negative controls."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.ppo.batch import prepare_sequence_targets
from cosmos_framework.posttrain.algorithm.ppo.records import PPOStepSource, PPOTargetStep

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _steps() -> tuple[PPOTargetStep, ...]:
    return (
        PPOTargetStep(
            PPOStepSource("turn-0", "episode", "fragment-a", 4),
            (0.25, 0.75, 99.0),
            (0.2, 0.3, 91.0, 92.0),
            (True, True, False),
            False,
            False,
        ),
        PPOTargetStep(
            PPOStepSource("turn-1", "episode", "fragment-b", 5),
            (2.0,),
            (0.8, 93.0),
            (True,),
            False,
            False,
        ),
        PPOTargetStep(
            PPOStepSource("turn-2", "episode", "fragment-b", 6),
            (-0.5, 3.0),
            (0.4, 0.6, 7.0),
            (True, True),
            True,
            False,
        ),
    )


def test_returns_match_discounted_reward_sums_across_turns_and_fragments() -> None:
    steps = _steps()
    gamma = 0.87
    result = prepare_sequence_targets(steps, gamma=gamma, gae_lambda=1.0)
    rewards = [reward for step in steps for reward, valid in zip(step.rewards, step.mask) if valid]
    expected = [
        sum(gamma**offset * reward for offset, reward in enumerate(rewards[start:])) for start in range(len(rewards))
    ]
    actual = [value for item, step in zip(result, steps) for value, valid in zip(item.returns, step.mask) if valid]
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64))
    assert result[0].returns[-1] == result[0].advantages[-1] == 0.0


def test_next_turn_value_uses_post_feedback_condition() -> None:
    steps = _steps()
    gamma = 0.9
    result = prepare_sequence_targets(steps, gamma=gamma, gae_lambda=0.0)
    assert result[0].advantages[1] == pytest.approx(
        steps[0].rewards[1] + gamma * steps[1].values[0] - steps[0].values[1]
    )
    assert result[1].advantages[0] == pytest.approx(
        steps[1].rewards[0] + gamma * steps[2].values[0] - steps[1].values[0]
    )
    assert result[-1].returns[-1] == pytest.approx(steps[-1].rewards[-1])


@pytest.mark.parametrize("boundary", ["truncated", "window_end"])
def test_nonterminal_boundaries_bootstrap_without_extending_gae(boundary: str) -> None:
    steps = _steps()
    last = replace(steps[-1], terminated=False, **{boundary: True})
    gamma = 0.9
    result = prepare_sequence_targets((*steps[:-1], last), gamma=gamma, gae_lambda=1.0)
    assert result[-1].returns[-1] == pytest.approx(last.rewards[-1] + gamma * last.values[-1])
    baseline = prepare_sequence_targets(steps, gamma=gamma, gae_lambda=1.0)
    assert result[0].returns[0] - baseline[0].returns[0] == pytest.approx(gamma**5 * last.values[-1])


def test_storage_repartition_and_input_order_preserve_targets_and_sample_ids() -> None:
    steps = _steps()
    expected = {item.source.sample_id: item for item in prepare_sequence_targets(steps, gamma=0.99, gae_lambda=0.95)}
    repartitioned = tuple(replace(step, source=replace(step.source, fragment_id="joined")) for step in reversed(steps))
    actual = prepare_sequence_targets(repartitioned, gamma=0.99, gae_lambda=0.95)
    assert [item.source.sample_id for item in actual] == [step.source.sample_id for step in repartitioned]
    for item in actual:
        assert item.advantages == expected[item.source.sample_id].advantages
        assert item.returns == expected[item.source.sample_id].returns


def test_reset_and_session_identity_keep_unrelated_sequences_separate() -> None:
    first = _steps()
    second = tuple(
        replace(
            step,
            source=replace(step.source, sample_id=f"other-{step.source.sample_id}", session="other"),
            rewards=tuple(2 * value for value in step.rewards),
        )
        for step in first
    )
    result = prepare_sequence_targets((*first, *second), gamma=0.99, gae_lambda=0.95)
    for steps, actual in ((first, result[:3]), (second, result[3:])):
        assert actual == prepare_sequence_targets(steps, gamma=0.99, gae_lambda=0.95)


def test_unfinished_sequence_requires_a_collection_boundary() -> None:
    steps = _steps()
    with pytest.raises(ValueError, match="missing a final collection boundary"):
        prepare_sequence_targets((*steps[:-1], replace(steps[-1], terminated=False)), gamma=0.99, gae_lambda=0.95)
