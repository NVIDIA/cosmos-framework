# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the FIELD rejection class, exercised through the real GRPO recipe.

The source/reward mismatch is rejected before runtime reward evaluation. Runtime reward checks remain
the second line of defense for malformed rows or trajectories."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
from cosmos_framework.posttrain.control.bind import CheckError, bind_check
from cosmos_framework.posttrain.control.manifest import Plane
from cosmos_framework.posttrain.entrypoints.grpo import manifest

_PROMPTS_ONLY = DatasetConfig(path="prompts.jsonl", prompt_column="question")
_WITH_ANSWERS = DatasetConfig(path="prompts.jsonl", prompt_column="question", response_column="answer")
_CUSTOM_PROMPTS_ONLY = DatasetConfig(factory="pkg.data.build", prompt_column="question")
_CUSTOM_WITH_ANSWERS = DatasetConfig(
    factory="pkg.data.build",
    prompt_column="question",
    response_column="answer",
)


def _field_errors(rl: GRPOConfig) -> list[CheckError]:
    return [e for e in bind_check(manifest(rl)).errors if e.category == "field"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("reward", ["arithmetic", "exact_match"])
def test_rejects_field_mismatch(reward: str) -> None:
    """The message names producer, consumer, the missing field, and the config key that fixes it."""
    errors = _field_errors(GRPOConfig(data=_PROMPTS_ONLY, reward=reward))
    assert len(errors) == 1
    assert errors[0].where == "rollout->rm"
    assert errors[0].message == (
        "consumer needs 'reference' (ground_truth); producer emits ['completion_text', 'prompt']; "
        "name it in rl.data.response_column, or select a reward provider whose needs omit 'reference'"
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "data,reward",
    [
        (DatasetConfig(), "arithmetic"),  # the built-in source generates its own answers
        (_WITH_ANSWERS, "exact_match"),  # a named response column feeds the answer-checking reward
        (_PROMPTS_ONLY, "nonempty"),  # the reward that reads no ground truth
        (_PROMPTS_ONLY, "constant"),  # pure distillation: task reward is identically zero
        (_CUSTOM_PROMPTS_ONLY, "nonempty"),
        (_CUSTOM_WITH_ANSWERS, "exact_match"),
    ],
)
def test_accepts_structurally_compatible_source_reward_pairings(data: DatasetConfig, reward: str) -> None:
    """Structural compatibility is narrower than reward usefulness or within-group variance."""
    assert bind_check(manifest(GRPOConfig(data=data, reward=reward))).ok


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_declares_the_loop_dataflow() -> None:
    """The manifest must describe the calls the control loop actually makes, not a shortcut edge."""
    edges = {(edge.src, edge.dst, edge.plane) for edge in manifest(GRPOConfig()).edges}
    assert edges == {
        ("data", "rollout", Plane.DATA),
        ("rollout", "rm", Plane.DATA),
        ("rm", "replay", Plane.DATA),
        ("replay", "policy", Plane.DATA),
        ("policy", "rollout", Plane.WEIGHT),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_reward_fails_catalog_resolution_before_manifest_binding() -> None:
    """A provider name must resolve from the run's catalog before the generic capability is selected."""
    with pytest.raises(KeyError, match="unknown reward provider 'no_such_reward'"):
        manifest(GRPOConfig(reward="no_such_reward"))
