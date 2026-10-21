# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest

from cosmos_framework.posttrain.algorithm.reward import RewardCandidate, RewardContext, gsm8k_reward_fn


def _score(completion: str, gold: str) -> float:
    result = gsm8k_reward_fn(
        RewardCandidate(trajectory_id="t", completion=completion),
        RewardContext(reference=gold),
    )
    return result.terms["task"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "completion,gold,expected",
    [
        ("The answer is #### 42", "#### 42", 1.0),  # exact strict match
        ("...\n#### 1,234", "#### 1234", 1.0),  # comma stripped both sides
        ("#### $18", "#### 18", 0.0),  # leading '$' not in strict char class -> None (verl/cosmos-rl parity)
        ("#### 41", "#### 42", 0.0),  # wrong number
        ("no marker here", "#### 42", 0.0),  # no #### in completion -> None != gold
    ],
)
def test_gsm8k_reward_matches_cosmos_rl_regex(completion: str, gold: str, expected: float) -> None:
    assert _score(completion, gold) == expected
