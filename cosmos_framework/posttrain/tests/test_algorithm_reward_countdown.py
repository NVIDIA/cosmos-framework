# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the built-in Countdown reference parser and verifier."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError
from cosmos_framework.posttrain.algorithm.reward.countdown import countdown_reference, countdown_reward_fn
from cosmos_framework.posttrain.algorithm.reward.group import RewardCandidate, RewardContext
from cosmos_framework.posttrain.algorithm.reward.reward import RULE_REWARDS


def _score(completion: str, nums: tuple[int, ...] = (3, 7, 11), target: int = 26) -> float:
    """Score one immutable candidate against the structured Countdown context."""
    result = countdown_reward_fn(
        RewardCandidate(trajectory_id="g0_k0", completion=completion),
        RewardContext(reference={"nums": list(nums), "target": target}),
    )
    return result.terms["task"]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_correct_equation_scores_one() -> None:
    """All three conditions hold: true equation, the offered numbers, the target."""
    assert _score("3*11-7=26", nums=(3, 7, 11), target=26) == 1.0
    # `<answer>` tags are stripped, matching the reference, so tagging is not punished.
    assert _score("thinking... <answer>3*11-7=26</answer>", nums=(3, 7, 11), target=26) == 1.0
    # sympy evaluates the arithmetic, so any true rearrangement counts.
    assert _score("(11-7)*3=12", nums=(3, 7, 11), target=12) == 1.0


@pytest.mark.L0
@pytest.mark.CPU
def test_each_failure_mode_scores_zero() -> None:
    """The three checks are independent, and each must be able to fail on its own."""
    assert _score("3*11-7=25", nums=(3, 7, 11), target=25) == 0.0  # false equation
    assert _score("2*13=26", nums=(3, 7, 11), target=26) == 0.0  # not the offered numbers
    assert _score("3*11-7=26", nums=(3, 7, 11), target=24) == 0.0  # not the target
    # a repeated number cannot be silently dropped: sorted-multiset comparison
    assert _score("3*3=9", nums=(3, 3, 3), target=9) == 0.0
    # unparseable output scores 0 rather than raising, matching the reference's guard
    assert _score("I don't know") == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_a_malformed_reference_raises_rather_than_scoring_zero() -> None:
    """A wiring error must not look like a model that never solves anything — the same rule the
    other rule rewards follow for a missing completion."""
    with pytest.raises(RewardContractError, match="nums"):
        countdown_reference("24")
    with pytest.raises(RewardContractError, match="nums"):
        countdown_reference({"target": 24})
    with pytest.raises(RewardContractError, match="integer"):
        countdown_reference({"nums": ["x"], "target": 24})
    with pytest.raises(RewardContractError, match="non-empty"):
        countdown_reference({"nums": [], "target": 24})


@pytest.mark.L0
@pytest.mark.CPU
def test_countdown_is_selectable_like_any_other_built_in() -> None:
    """Registered in `RULE_REWARDS`, so `rl.reward='countdown'` resolves through the default
    catalog. Its `reference` need selects a bounded generic capability; the reference SHAPE remains
    the verifier's own contract."""
    assert "countdown" in RULE_REWARDS
    assert RULE_REWARDS["countdown"].needs == ("reference",)
