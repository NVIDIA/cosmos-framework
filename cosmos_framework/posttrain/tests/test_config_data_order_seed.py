# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for the prompt-order seed contract.

`rl.seed` orders the GRPO prompt source unless `rl.data_seed_follows_rollout_seed` keys it on the
top-level `rollout.seed`; the preflight and the data role resolve the seed through one function, and
the preflight warns when arms that vary only `rollout.seed` would all read one prompt order.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.config import build
from cosmos_framework.posttrain.config.rl import (
    DatasetConfig,
    GRPOConfig,
    RolloutConfig,
    check_data_order_seed,
    resolve_data_seed,
)
from cosmos_framework.posttrain.roles.env.dataset import DatasetEnv
from cosmos_framework.posttrain.roles.env.prompt import build_prompt_env

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

PROMPT_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_prompt_dataset"
SHARED_ORDER = "all arms share one prompt order"
FLAG_NEEDS_SEED = "requires an integer rollout.seed"


def _source(*, shuffle: bool, count: int = 2) -> DatasetConfig:
    return DatasetConfig(
        factory=PROMPT_FACTORY,
        factory_kwargs={"count": count},
        prompt_column="question",
        response_column="answer",
        shuffle=shuffle,
    )


def _prompt_order(env: Any, count: int) -> list[str]:
    return [observation.representations["prompt"] for observation in env.get_samples(count)]


def test_resolved_data_seed_is_rl_seed_unless_the_flag_follows_rollout_seed() -> None:
    """Default resolution is `rl.seed`; the flag substitutes `rollout.seed` and refuses an unset one."""
    rollout = RolloutConfig(seed=42)
    assert resolve_data_seed(GRPOConfig(), rollout) == 0
    assert resolve_data_seed(GRPOConfig(seed=7), rollout) == 7
    assert resolve_data_seed(GRPOConfig(seed=7), None) == 7
    assert resolve_data_seed(GRPOConfig(seed=7, data_seed_follows_rollout_seed=True), rollout) == 42
    for missing in (RolloutConfig(), None):
        with pytest.raises(ValueError, match=FLAG_NEEDS_SEED):
            resolve_data_seed(GRPOConfig(data_seed_follows_rollout_seed=True), missing)
    with pytest.raises(TypeError, match="data_seed_follows_rollout_seed must be bool"):
        GRPOConfig(data_seed_follows_rollout_seed="yes")  # type: ignore[arg-type]


def test_shared_prompt_order_warning_fires_for_rollout_seed_replicates_only() -> None:
    """Shuffle + default rl.seed + a per-arm rollout.seed warns; the flag, a per-arm rl.seed, no shuffle,
    or an unset/zero rollout.seed are silent."""
    with pytest.warns(UserWarning, match=SHARED_ORDER):
        assert check_data_order_seed(GRPOConfig(data=_source(shuffle=True)), RolloutConfig(seed=42)) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        following = GRPOConfig(data=_source(shuffle=True), data_seed_follows_rollout_seed=True)
        assert check_data_order_seed(following, RolloutConfig(seed=42)) == 42
        assert check_data_order_seed(GRPOConfig(data=_source(shuffle=True), seed=3), RolloutConfig(seed=42)) == 3
        assert check_data_order_seed(GRPOConfig(data=_source(shuffle=False)), RolloutConfig(seed=42)) == 0
        assert check_data_order_seed(GRPOConfig(data=_source(shuffle=True)), RolloutConfig(seed=0)) == 0
        assert check_data_order_seed(GRPOConfig(data=_source(shuffle=True)), RolloutConfig()) == 0


def test_online_preflight_runs_the_prompt_order_seed_contract() -> None:
    """The shared GRPO/OPD identity preflight warns and refuses through the function the data role resolves with."""
    with pytest.warns(UserWarning, match=SHARED_ORDER):
        build.validate_online_policy_backend(
            GRPOConfig(data=_source(shuffle=True)),
            RolloutConfig(seed=42),
            model_backend=None,
            model_config=None,
        )
    with pytest.raises(ValueError, match=FLAG_NEEDS_SEED):
        build.validate_online_policy_backend(
            GRPOConfig(data_seed_follows_rollout_seed=True),
            RolloutConfig(),
            model_backend=None,
            model_config=None,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build.validate_online_policy_backend(
            GRPOConfig(data=_source(shuffle=True), data_seed_follows_rollout_seed=True),
            RolloutConfig(seed=42),
            model_backend=None,
            model_config=None,
        )


def test_build_prompt_env_orders_prompts_with_the_resolved_data_seed() -> None:
    """The data role permutes the corpus with the resolved seed: rollout.seed under the flag, rl.seed otherwise."""
    count = 8
    reference_orders = {
        seed: _prompt_order(
            DatasetEnv(
                [{"prompt": f"question-{index}"} for index in range(count)], shuffle=True, seed=seed, repeat=True
            ),
            count,
        )
        for seed in (0, 42)
    }
    assert reference_orders[0] != reference_orders[42]  # the two seeds are distinguishable on this corpus
    config = SimpleNamespace(rollout=RolloutConfig(seed=42))

    following = GRPOConfig(data=_source(shuffle=True, count=count), data_seed_follows_rollout_seed=True)
    env = build_prompt_env(following, config)
    assert env.state_dict()["cursor"]["seed"] == 42  # the checkpointed cursor identity carries the resolved seed
    assert _prompt_order(env, count) == reference_orders[42]

    default = GRPOConfig(data=_source(shuffle=True, count=count))
    assert build_prompt_env(default, config).state_dict()["cursor"]["seed"] == 0
    assert _prompt_order(build_prompt_env(default, config), count) == reference_orders[0]
    assert _prompt_order(build_prompt_env(default), count) == reference_orders[0]  # no root config: rl.seed
    with pytest.raises(ValueError, match=FLAG_NEEDS_SEED):
        build_prompt_env(following)
