# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Custom dataset factory contracts for online and preference training."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from cosmos_framework.utils.config import Config
from cosmos_framework.utils.config_helper import override
from cosmos_framework.posttrain.config.config_dpo import PosttrainDPOConfig
from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig
from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig, PreferenceEnvConfig

PROMPT_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_prompt_dataset"
PREFERENCE_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_preference_dataset"
IDENTIFIED_PREFERENCE_FACTORY = (
    "cosmos_framework.posttrain.tests.helpers.custom_data.build_identified_preference_dataset"
)
STATEFUL_PREFERENCE_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_stateful_preference_dataset"
EMPTY_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_empty_dataset"
ITERABLE_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_iterable_dataset"
ITEMS_FN = "cosmos_framework.posttrain.tests.helpers.custom_data.format_prompt"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("config_type", [PosttrainDPOConfig, PosttrainGRPOConfig, PosttrainOPDConfig])
def test_factory_kwargs_survive_real_root_override(config_type: type[Config]) -> None:
    """The production OmegaConf round trip must normalize factory kwargs back to plain Python."""
    authored = config_type(model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None)
    config = override(authored, ["--"], remove_defaults=True)

    factory_kwargs = getattr(config.rl, "data").factory_kwargs
    assert type(factory_kwargs) is dict


@pytest.mark.L0
@pytest.mark.CPU
def test_online_dataset_config_accepts_one_custom_factory() -> None:
    """Removing the factory fields would make custom source selection impossible."""
    config = DatasetConfig(
        factory=PROMPT_FACTORY,
        factory_kwargs={"count": 4},
        prompt_column="question",
        response_column="answer",
    )

    assert config.factory == PROMPT_FACTORY
    assert config.factory_kwargs == {"count": 4}


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_kwargs_recursively_normalize_omegaconf_containers() -> None:
    """A shallow dict conversion would leave nested ListConfig values in source identity."""
    config = DatasetConfig(
        factory=PROMPT_FACTORY,
        factory_kwargs=OmegaConf.create({"count": 2, "tags": ["a", "b"]}),
        prompt_column="question",
    )

    assert type(config.factory_kwargs) is dict
    assert type(config.factory_kwargs["tags"]) is list


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_kwargs_normalize_omegaconf_nested_in_plain_mapping() -> None:
    """A plain mapping must not retain nested OmegaConf containers."""
    config = DatasetConfig(
        factory=PROMPT_FACTORY,
        factory_kwargs={"count": 2, "tags": OmegaConf.create(["a", "b"])},
        prompt_column="question",
    )

    assert type(config.factory_kwargs) is dict
    assert type(config.factory_kwargs["tags"]) is list


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("builtin", ["path", "name"])
def test_online_dataset_config_rejects_custom_and_builtin_sources(builtin: str) -> None:
    """Dropping selector exclusivity would make source precedence ambiguous."""
    kwargs = {
        "factory": PROMPT_FACTORY,
        "prompt_column": "question",
        builtin: "rows.jsonl" if builtin == "path" else "owner/dataset",
    }
    if builtin == "name":
        kwargs["subset"] = "main"
        kwargs["response_column"] = "answer"

    with pytest.raises(ValueError, match="at most one"):
        DatasetConfig(**kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_online_dataset_config_requires_factory_for_kwargs() -> None:
    """Ignoring orphan kwargs would hide a misspelled factory field."""
    with pytest.raises(ValueError, match="factory_kwargs"):
        DatasetConfig(factory_kwargs={"count": 4})


@pytest.mark.L0
@pytest.mark.CPU
def test_online_dataset_config_requires_prompt_column_for_custom_rows() -> None:
    """Without a prompt column, a raw custom row cannot enter the canonical contract."""
    with pytest.raises(ValueError, match="prompt_column"):
        DatasetConfig(factory=PROMPT_FACTORY)


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_prompt_dataset_uses_existing_lazy_row_mapping() -> None:
    """Bypassing column/items mapping would give downstream roles the wrong row schema."""
    from cosmos_framework.posttrain.roles.env.prompt import dataset_items

    config = DatasetConfig(
        factory=PROMPT_FACTORY,
        factory_kwargs={"count": 2},
        prompt_column="question",
        response_column="answer",
        items_fn=ITEMS_FN,
    )

    items = dataset_items(config)

    assert len(items) == 2
    assert items[1] == {"prompt": "formatted: question-1", "reference": "answer-1"}


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_prompt_cursor_identity_includes_factory_kwargs() -> None:
    """Equal-length custom sources must not share an exact-resume fingerprint."""
    from cosmos_framework.posttrain.roles.env.prompt import dataset_items

    common = {
        "factory": PROMPT_FACTORY,
        "prompt_column": "question",
        "response_column": "answer",
    }
    first = dataset_items(DatasetConfig(**common, factory_kwargs={"count": 2, "prefix": "first"}))
    second = dataset_items(DatasetConfig(**common, factory_kwargs={"count": 2, "prefix": "second"}))

    assert first.cursor_fingerprint() != second.cursor_fingerprint()


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_prompt_cursor_identity_includes_dataset_fingerprint() -> None:
    """Mutable external data must invalidate resume even when factory configuration is unchanged."""
    from cosmos_framework.posttrain.roles.env.dataset import map_prompt_items

    class _Source(list):
        def __init__(self, fingerprint: str) -> None:
            super().__init__([{"question": "same"}])
            self._fingerprint: str = fingerprint

        def cursor_fingerprint(self) -> str:
            return self._fingerprint

    first = map_prompt_items(_Source("revision-a"), "question", source_identity="same-factory-config")
    second = map_prompt_items(_Source("revision-b"), "question", source_identity="same-factory-config")

    assert first.cursor_fingerprint() != second.cursor_fingerprint()


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_prompt_cursor_identity_includes_dataset_length() -> None:
    """A rebuilt custom source with a different row count must invalidate exact resume."""
    from cosmos_framework.posttrain.roles.env.dataset import map_prompt_items

    short = map_prompt_items([{"question": "same"}], "question", source_identity="same-factory-config")
    long = map_prompt_items(
        [{"question": "same"}, {"question": "new"}],
        "question",
        source_identity="same-factory-config",
    )

    assert short.cursor_fingerprint() != long.cursor_fingerprint()


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_prompt_source_builds_the_standard_dataset_env() -> None:
    """A factory branch that never reaches build_prompt_env is not a usable production source."""
    from cosmos_framework.posttrain.roles.env.prompt import build_prompt_env, prompt_source

    rl = GRPOConfig(
        data=DatasetConfig(
            factory=PROMPT_FACTORY,
            factory_kwargs={"count": 2},
            prompt_column="question",
            response_column="answer",
        )
    )

    assert prompt_source(rl) == ("custom", True)
    observation = build_prompt_env(rl).reset()
    assert observation.representations == {"prompt": "question-0", "reference": "answer-0"}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("factory", [EMPTY_FACTORY, ITERABLE_FACTORY])
def test_custom_dataset_rejects_sources_without_stable_nonempty_indices(factory: str) -> None:
    """Accepting empty or iterable-only sources would break cursor and sharding invariants."""
    from cosmos_framework.posttrain.data.datasets.factory import build_custom_dataset

    with pytest.raises((TypeError, ValueError), match="custom dataset factory"):
        build_custom_dataset(factory, {})


@pytest.mark.L0
@pytest.mark.CPU
def test_preference_config_rejects_custom_and_builtin_sources() -> None:
    """DPO must not silently choose between custom and JSONL preference data."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PreferenceEnvConfig(factory=PREFERENCE_FACTORY, jsonl_path="pairs.jsonl")


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_preference_dataset_uses_existing_env_cursor() -> None:
    """Wrapping custom items in another Env would bypass standard shuffle and resume behavior."""
    from cosmos_framework.posttrain.roles.env.preference import build_preference_env

    config = PreferenceEnvConfig(
        factory=PREFERENCE_FACTORY,
        factory_kwargs={"count": 2},
        shuffle=False,
    )

    env = build_preference_env(config, seed=7)

    trajectories = env.get_samples(2)
    assert [trajectory.id for trajectory in trajectories] == ["pref-0", "pref-1"]
    assert [trajectory.transitions[0].obs.representations["prompt"] for trajectory in trajectories] == [
        "prompt-0",
        "prompt-1",
    ]
    assert [trajectory.transitions[0].targets.winner for trajectory in trajectories] == [
        {"sample": "winner-0"},
        {"sample": "winner-1"},
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_preference_validation_requires_metadata_only_membership_identity() -> None:
    """Leakage checks must not enumerate custom rows that may decode media in __getitem__."""
    from cosmos_framework.posttrain.roles.env.preference import build_preference_env

    config = PreferenceEnvConfig(
        factory=PREFERENCE_FACTORY,
        factory_kwargs={"count": 2},
        shuffle=False,
    )
    env = build_preference_env(config, seed=7, include_source_identity=True)

    state = env.state_dict()
    assert state is not None
    with pytest.raises(TypeError, match="membership_ids"):
        env.preference_source_identity()


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_preference_validation_uses_dataset_membership_identity() -> None:
    """A custom dataset can supply cheap semantic aliases without media reads."""
    from cosmos_framework.posttrain.roles.env.preference import build_preference_env

    config = PreferenceEnvConfig(
        factory=IDENTIFIED_PREFERENCE_FACTORY,
        factory_kwargs={"count": 2},
        shuffle=False,
    )
    identity = build_preference_env(config, seed=7, include_source_identity=True).preference_source_identity()

    assert identity.raw_fingerprint.startswith("custom-preference-sha256:")
    assert identity.membership_ids == ("custom-pair-0", "custom-pair-1")


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_preference_cursor_identity_includes_dataset_length() -> None:
    """Preference cursor identity must notice a changed row count without reading media."""
    from cosmos_framework.posttrain.roles.env.preference import _CustomPreferenceDataset

    short = _CustomPreferenceDataset([("prompt", {}, {})], "same-factory-config")
    long = _CustomPreferenceDataset([("prompt", {}, {}), ("new", {}, {})], "same-factory-config")

    assert short.cursor_fingerprint() != long.cursor_fingerprint()


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_preference_dataset_forwards_internal_cursor_state() -> None:
    """Framework wrapping must preserve a custom source's stochastic preprocessing cursor."""
    from cosmos_framework.posttrain.roles.env.preference import build_preference_env

    config = PreferenceEnvConfig(factory=STATEFUL_PREFERENCE_FACTORY, shuffle=False)
    env = build_preference_env(config, seed=7)
    env.get_samples(1)

    state = env.state_dict()
    assert state is not None
    assert state["dataset_state"] == {"offset": 1}

    restored = build_preference_env(config, seed=7)
    assert restored.load_state_dict(state) is True
    restored_state = restored.state_dict()
    assert restored_state is not None
    assert restored_state["dataset_state"] == {"offset": 1}
