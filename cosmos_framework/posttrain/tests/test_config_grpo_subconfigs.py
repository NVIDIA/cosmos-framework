# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GRPO's clipped-objective knobs + prompt source on `GRPOConfig`, the additive optimizer/model
fields, and the `export_*` relocation onto `RunConfig`.

Mostly typed carriers, pinned for SCHEMA (shape, defaults, guards, placement) rather than behaviour.
The exception is `data`, which the GRPO entry now reads to build its prompt source — its behaviour
lives in tests/test_roles_env_dataset.py and tests/test_control_bind_field.py."""

import pytest

from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    DatasetConfig,
    DPOConfig,
    GRPOConfig,
    OptimizerConfig,
    PreferenceEnvConfig,
    RolloutConfig,
    RunConfig,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_knobs_are_flat_beside_group_size() -> None:
    # FLAT, not nested under an "algo" object. An earlier draft had a GRPOAlgoConfig holding these,
    # which was incoherent twice over: it contained no GRPO-specific field (epsilon/kl_beta/
    # mu_iterations are the PPO-family surrogate that GRPO inherits), while `group_size` — the one
    # parameter that makes GRPO group-relative — sat OUTSIDE it. cosmos-rl keeps them flat under
    # [train.train_policy] for the same reason.
    c = GRPOConfig()
    assert (c.epsilon_low, c.epsilon_high) == (0.2, 0.2)
    assert c.kl_beta == 0.0 and c.mu_iterations == 1
    # min_filter_prefix_tokens defaults to None = shared-prefix filtering OFF (cosmos-rl default)
    assert c.temperature == 1.0 and c.min_filter_prefix_tokens is None
    assert c.group_size == 8 and c.batch_per_replica == 32  # main's loop knobs, same level
    assert not hasattr(c, "algo"), "objective knobs must stay flat, not nested under `algo`"


@pytest.mark.L0
@pytest.mark.CPU
def test_data_source_slot_matches_dpo() -> None:
    d = DatasetConfig()
    assert d.shuffle is False  # nothing set ⇒ v0 built-in source
    assert d.name == "" and d.subset == "" and d.split == "train"
    assert d.path == "" and d.prompt_column == "" and d.response_column == ""
    # Same slot name as DPOConfig.data, so both algorithms expose their source at rl.data
    # (types differ per algorithm: DatasetConfig here, PreferenceEnvConfig for DPO).
    assert isinstance(GRPOConfig().data, DatasetConfig)
    assert isinstance(DPOConfig().data, PreferenceEnvConfig)
    assert not hasattr(GRPOConfig(), "dataset"), "slot is `data`, aligned with DPO"


@pytest.mark.L0
@pytest.mark.CPU
def test_prompt_source_guards() -> None:
    """What one config object can decide alone fails here; what depends on the reward it is paired
    with is left to bind_check, which sees both roles."""
    with pytest.raises(ValueError, match="needs rl.data.prompt_column"):
        DatasetConfig(path="p.jsonl")
    with pytest.raises(ValueError, match="require rl.data.path"):
        DatasetConfig(prompt_column="question")
    with pytest.raises(ValueError, match="require rl.data.path"):
        DatasetConfig(response_column="answer")
    with pytest.raises(ValueError, match="require rl.data.path"):
        DatasetConfig(shuffle=True)
    with pytest.raises(ValueError, match="either rl.data.path or rl.data.name"):
        DatasetConfig(name="org/data", path="p.jsonl", prompt_column="question")
    # ANY hub dataset is accepted now. The allowlist that used to reject this named ONE dataset
    # inside the dependency-light typed schema, i.e. config knew about an env. What still fails
    # loud is an INCOMPLETE spec, whichever dataset it is.
    with pytest.raises(ValueError, match="needs rl.data.subset"):
        DatasetConfig(name="org/data", prompt_column="question", response_column="answer")
    # A named source needs its columns too: without them the config would VALIDATE and then fail at
    # run time, since the builtin column mapping reads both off this object.
    with pytest.raises(ValueError, match="needs rl.data.prompt_column"):
        DatasetConfig(name="openai/gsm8k")
    # response_column is required because `prompt_source` reports this source as carrying ground
    # truth -- the claim a verifiable reward is validated against.
    with pytest.raises(ValueError, match="needs rl.data.response_column"):
        DatasetConfig(name="openai/gsm8k", subset="main", prompt_column="question")
    # subset is the HF *config* name; empty is not "no config" -- it reaches load_dataset as "" and
    # fails at ACTOR SETUP, i.e. after --validate passed and after GPUs are already held.
    with pytest.raises(ValueError, match="needs rl.data.subset"):
        DatasetConfig(name="openai/gsm8k", prompt_column="question", response_column="answer")
    # ...and a fully-specified source is accepted, with its HF config name and its columns.
    DatasetConfig(name="openai/gsm8k", subset="main", prompt_column="question", response_column="answer")
    with pytest.raises(ValueError, match="require rl.data.name"):
        DatasetConfig(subset="main")
    with pytest.raises(ValueError, match="require rl.data.name"):
        DatasetConfig(split="test")
    DatasetConfig(path="p.jsonl", prompt_column="question")  # the minimum valid JSONL source


@pytest.mark.L0
@pytest.mark.CPU
def test_additive_fields() -> None:
    assert OptimizerConfig().warmup_steps == 0
    assert HFModelConfig().fsdp_reduce_dtype == "float32"


@pytest.mark.L0
@pytest.mark.CPU
def test_export_fields_live_on_runconfig_not_the_model_spec() -> None:
    # Exporting is an OUTPUT concern with the run's lifetime, so it sits beside save_every /
    # checkpoint_format on RunConfig — not on HFModelConfig, which specifies how weights are LOADED.
    r = RunConfig()
    assert r.export_dir == "" and r.export_every == 0  # defaults carried verbatim from HFModelConfig
    assert not hasattr(HFModelConfig(), "export_dir")
    assert not hasattr(HFModelConfig(), "export_every")
    # On RunConfig ⇒ every algorithm inherits it, not just GRPO.
    assert GRPOConfig().export_dir == "" and DPOConfig().export_dir == ""


@pytest.mark.L0
@pytest.mark.CPU
def test_no_duplicate_save_cadence_knob() -> None:
    # Guard against re-introducing a second save-cadence field: `save_every` on RunConfig is THE
    # cadence knob (an earlier draft added a CheckpointConfig.save_freq that silently duplicated it,
    # alongside an `enable` implied by save_every>0 and a save_mode duplicating DPOConfig.save_async).
    c = GRPOConfig()
    assert c.save_every == 0
    for dead in ("save_freq", "ckpt"):
        assert not hasattr(c, dead), f"{dead} duplicates existing RunConfig/DPOConfig state"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_guards() -> None:
    # group_size guard is main's; the two objective guards moved here when the knobs were flattened.
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        GRPOConfig(group_size=1)
    # mu_iterations is pinned at 1 in BOTH directions. 0 is nonsense; >1 names a feature whose
    # surrogate exists but whose driver does not, so accepting it would train on-policy while the
    # run config advertised inner epochs.
    for bad in (0, 2):
        with pytest.raises(ValueError, match="mu_iterations must be 1"):
            GRPOConfig(mu_iterations=bad)
    assert GRPOConfig(mu_iterations=1).mu_iterations == 1
    with pytest.raises(ValueError, match="kl_beta must be >= 0"):
        GRPOConfig(kl_beta=-0.1)


@pytest.mark.L0
@pytest.mark.CPU
def test_dual_temperature_is_not_auto_synced() -> None:
    # Two distinct temperatures: rollout SAMPLING (RolloutConfig, top-level `config.rollout`) vs
    # policy-loss LOGIT (here). They remain independently configured; bind-time validation rejects
    # a mismatch for on-policy GRPO instead of silently coupling the fields (the reference uses 0.9
    # for each). Pinned so a future alias cannot hide which semantic field a recipe selected.
    assert RolloutConfig().temperature == 1.0
    assert GRPOConfig().temperature == 1.0
    c = GRPOConfig(temperature=0.9)
    assert RolloutConfig().temperature == 1.0  # unchanged — no hidden coupling


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_temperature_must_be_positive() -> None:
    """`GRPOConfig.temperature` divides the training logits — 0 gives inf/NaN, negatives invert."""
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            GRPOConfig(temperature=bad)
    GRPOConfig(temperature=1.0)  # the default stays valid


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_objective_identity_records_each_loss_semantic() -> None:
    from cosmos_framework.posttrain.config.build import build_objective

    legacy = build_objective(
        GRPOConfig(grpo_surrogate="legacy_pg", temperature=0.9),
        "unused",
        packer=object(),
    )
    assert legacy.identity == {
        "schema_version": 1,
        "algorithm": "grpo",
        "surrogate": "legacy_pg",
        "temperature": 0.9,
        "loss_type": "token_sum",
    }

    clipped = build_objective(
        GRPOConfig(grpo_surrogate="clipped", temperature=0.9, epsilon_low=0.1),
        "unused",
        packer=object(),
    )
    token_level_keys = {
        "schema_version",
        "algorithm",
        "surrogate",
        "temperature",
        "loss_type",
        "logprob_dtype",
        "epsilon_low",
        "epsilon_high",
        "kl_beta",
        "lower_bound_ratio",
        "use_rollout_logprobs_for_loss",
    }
    # Token level is recorded by ABSENCE: every earlier clipped checkpoint was trained under it, and
    # an unconditional key would refuse their exact resume on a byte that carries no information.
    assert set(clipped.identity) == token_level_keys
    assert clipped.identity["surrogate"] == "clipped"
    assert clipped.identity["temperature"] == 0.9
    assert clipped.identity["epsilon_low"] == 0.1
    assert clipped.identity["use_rollout_logprobs_for_loss"] is False

    sequence = build_objective(
        GRPOConfig(
            grpo_surrogate="clipped",
            importance_sampling_level="sequence",
            lower_bound_ratio=0.0,
            batch_per_replica=128,
            group_size=16,
            mini_batch_per_replica=32,
            certified_deterministic_forward=True,
            epsilon_low=3e-4,
            epsilon_high=4e-4,
        ),
        "unused",
        packer=object(),
    )
    assert sequence.identity["importance_sampling_level"] == "sequence"
    assert set(sequence.identity) == token_level_keys | {"importance_sampling_level"}
