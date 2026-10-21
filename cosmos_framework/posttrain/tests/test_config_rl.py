# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for the posttrain RL config schema + construction helpers.

Covers the load-bearing properties: typed validation, the OmegaConf structured round-trip
(author → YAML → reload-with-schema → instantiate back to the typed dataclass), typed
dotlist override (+ unknown-key rejection), RL-owned optimizer freezing, and the reserved
HF loader's multi-GPU guard.
"""

from __future__ import annotations

import warnings

import pytest
from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.posttrain.config import build
from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    CURSOR_RESUME_OFF,
    FIXED_NORMALISER_OPTIMIZER_EPS_CEILING,
    RECIPE_GUARD_REFUSAL_SENTENCE,
    RECIPE_GUARD_SEVERITIES,
    CheckpointStoreConfig,
    CheckpointUploadConfig,
    DataPoolConfig,
    DatasetConfig,
    DPOConfig,
    DPOValidationConfig,
    GridConfig,
    GRPOConfig,
    GRPOValidationConfig,
    NFTConfig,
    OPDConfig,
    OptimizerConfig,
    PreferenceEnvConfig,
    RolloutConfig,
    RunConfig,
    ValidationSamplingConfig,
    ValidationScheduleConfig,
)
from cosmos_framework.posttrain.control.loop.batching import informative_quota_per_cell, prompts_per_data_cell
from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS, CheckpointComponentPlan
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration


@pytest.mark.L0
@pytest.mark.CPU
class TestDPOConfigValidation:
    def test_defaults(self):
        c = DPOConfig()
        assert c.beta == 500.0 and c.grad_accum == 1
        assert c.optimizer.train_keys == []  # model-agnostic default (train all); recipes set the freeze subset
        assert c.ref.update_every == 0
        assert c.data.media == "video"
        assert c.save_async is True
        assert c.checkpoint_components == list(FULL_CHECKPOINT_COMPONENTS)
        assert c.keep_optim_last is None and c.keep_model_last == 0
        assert c.checkpoint_store == CheckpointStoreConfig()
        assert c.upload == CheckpointUploadConfig()
        assert c.init_from == ""

    def test_bad_beta(self):
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(beta=0)

    def test_bad_grad_accum(self):
        with pytest.raises(ValueError, match="grad_accum"):
            DPOConfig(grad_accum=0)

    @pytest.mark.parametrize("kwargs", [{"max_steps": -1}, {"max_steps": True}, {"save_every": -1}])
    def test_bad_run_clock(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DPOConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
class TestGRPOConfigValidation:
    def test_defaults(self):
        c = GRPOConfig()
        assert c.group_size == 8 and c.batch_per_replica == 32 and c.max_operand == 50
        assert c.weight_sync_transport is None
        assert c.weight_sync_r2r_broadcast is False
        assert c.weight_sync_bucket_bytes is None
        assert c.weight_sync_probe_bytes == 256 * 1024 * 1024
        assert c.weight_sync_min_gbps == 20.0
        assert c.grpo_surrogate == "clipped"
        # rollout sampling/engine lives in config.rollout (RolloutConfig), not config.rl — see
        # tests/test_config_vllm_rollout.py.
        assert c.optimizer.train_keys == []  # HF path trains all params (cosmos3 keys match none)
        assert isinstance(c, RunConfig) and c.checkpoint_format == "consolidated"

    def test_arithmetic_root_pins_legacy_pg_without_changing_the_production_default(self) -> None:
        root = PosttrainGRPOConfig(
            model=None,
            optimizer=None,
            scheduler=None,
            dataloader_train=None,
            dataloader_val=None,
        )

        assert root.rl.grpo_surrogate == "legacy_pg"
        assert GRPOConfig().grpo_surrogate == "clipped"

    def test_root_rejects_unknown_model_backend_during_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown model_backend"):
            PosttrainGRPOConfig(
                model=None,
                optimizer=None,
                scheduler=None,
                dataloader_train=None,
                dataloader_val=None,
                model_backend="unknown",
            )

    def test_selector_migration_is_one_way_and_unambiguous(self) -> None:
        import dataclasses

        grpo_fields = {item.name for item in dataclasses.fields(GRPOConfig)}
        assert "grpo_surrogate" in grpo_fields
        assert "objective" not in grpo_fields
        schema = OmegaConf.structured(GRPOConfig)
        with pytest.raises(Exception):
            OmegaConf.merge(schema, OmegaConf.from_dotlist(["objective=degenerate"]))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"logprob_dtype": "bfloat16"},
            {"epsilon_low": 0.1},
            {"epsilon_high": 0.3},
            {"kl_beta": 0.1},
            {"lower_bound_ratio": 0.0},
            {"loss_type": "seq-mean-token-sum"},
            {"use_rollout_logprobs_for_loss": True},
        ],
    )
    def test_legacy_pg_rejects_configured_clipped_only_knobs(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="does not consume clipped-surrogate settings"):
            GRPOConfig(grpo_surrogate="legacy_pg", **kwargs)

    def test_legacy_pg_accepts_an_explicit_policy_temperature(self) -> None:
        assert GRPOConfig(grpo_surrogate="legacy_pg", temperature=0.9).temperature == 0.9

    def test_validation_top_k_rejects_backend_divergent_negative_sentinel(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            ValidationSamplingConfig(top_k=-2)

    def test_bad_group_size(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            GRPOConfig(group_size=1)
        with pytest.raises(TypeError, match="group_size"):
            GRPOConfig(group_size=True)

    def test_data_pool_config(self) -> None:
        config = GRPOConfig(data_pool=DataPoolConfig(data_plane="ray_nixl_pool"))
        assert config.data_pool.data_plane == "ray_nixl_pool"
        with pytest.raises(ValueError, match="data_pool.data_plane"):
            DataPoolConfig(data_plane="invalid")
        with pytest.raises(ValueError, match="data_pool.cuda_cache_bytes"):
            DataPoolConfig(page_bytes=256, cuda_cache_bytes=257)
        with pytest.raises(ValueError, match="data_pool.page_bytes must be a multiple of the 256-byte"):
            DataPoolConfig(page_bytes=64, cuda_cache_bytes=64)

    def test_runconfig_validation_inherited(self):
        # grad_accum / checkpoint_format checks live on RunConfig and fire for GRPO too.
        with pytest.raises(ValueError, match="grad_accum"):
            GRPOConfig(grad_accum=0)
        with pytest.raises(ValueError, match="checkpoint_format"):
            GRPOConfig(checkpoint_format="bogus")

    def test_weight_sync_settings_are_validated_for_grpo_and_opd(self) -> None:
        from cosmos_framework.posttrain.config.rl import OPDConfig

        for config_type in (GRPOConfig, OPDConfig):
            with pytest.raises(ValueError, match="weight_sync_transport"):
                config_type(weight_sync_transport="nccl")
            with pytest.raises(TypeError, match="weight_sync_r2r_broadcast"):
                config_type(weight_sync_r2r_broadcast="yes")
            for invalid_bucket_bytes in (True, 0, -1, 1.5):
                with pytest.raises(ValueError, match="weight_sync_bucket_bytes"):
                    config_type(weight_sync_bucket_bytes=invalid_bucket_bytes)
            for invalid_probe_bytes in (True, 0, -1, 1.5):
                with pytest.raises(ValueError, match="weight_sync_probe_bytes"):
                    config_type(weight_sync_probe_bytes=invalid_probe_bytes)
            for invalid_min_gbps in (True, -0.1, float("nan"), float("inf"), "20"):
                with pytest.raises(ValueError, match="weight_sync_min_gbps"):
                    config_type(weight_sync_min_gbps=invalid_min_gbps)

    def test_colocated_offload_defaults_on_and_must_be_bool(self) -> None:
        """The colocated hand-off cycle stays on unless a run opts out; the dotlist path yields a real bool."""
        for config_type in (GRPOConfig, OPDConfig):
            assert config_type().colocated_offload is True
            assert config_type(colocated_offload=False).colocated_offload is False
            with pytest.raises(TypeError, match="colocated_offload"):
                config_type(colocated_offload="no")
        merged = OmegaConf.merge(OmegaConf.structured(GRPOConfig), OmegaConf.from_dotlist(["colocated_offload=false"]))
        assert instantiate(merged).colocated_offload is False

    @pytest.mark.parametrize("config_type", [GRPOConfig, OPDConfig])
    def test_init_from_remains_absent_from_llm_schemas(self, config_type: type[GRPOConfig]) -> None:
        schema = OmegaConf.structured(config_type)
        with pytest.raises(Exception):
            OmegaConf.merge(schema, OmegaConf.from_dotlist(["init_from=/tmp/base_checkpoint"]))

    def test_optimizer_shared_and_model_agnostic(self):
        # optimizer is a shared RunConfig field, and its default is model-agnostic (train all) — no model
        # param names leak into the schema. The freeze subset is a recipe choice set by the experiment.
        assert RunConfig().optimizer.train_keys == []
        assert DPOConfig().optimizer.train_keys == []
        assert GRPOConfig().optimizer.train_keys == []

    def test_seed_is_the_run_level_seed(self):
        # seed is the single run seed on RunConfig now (DPO's dataset + GRPO's env/rollout derive from
        # it); PreferenceEnvConfig no longer carries its own.
        import dataclasses

        assert RunConfig().seed == 0 and DPOConfig().seed == 0 and GRPOConfig().seed == 0
        assert "seed" not in {f.name for f in dataclasses.fields(PreferenceEnvConfig)}

    def test_dotlist_override_and_unknown_key(self):
        schema = OmegaConf.structured(GRPOConfig)
        merged = OmegaConf.merge(schema, OmegaConf.from_dotlist(["group_size=16", "max_operand=99", "max_steps=50"]))
        obj = instantiate(merged)
        assert obj.group_size == 16 and obj.max_operand == 99 and obj.max_steps == 50
        with pytest.raises(Exception):
            OmegaConf.merge(schema, OmegaConf.from_dotlist(["beta=1.0"]))  # a DPO field, not on GRPO


@pytest.mark.L0
@pytest.mark.CPU
class TestCheckpointStoragePolicyConfig:
    @pytest.mark.parametrize("config_type", [RunConfig, DPOConfig, GRPOConfig, OPDConfig, NFTConfig])
    def test_shared_defaults(self, config_type: type[RunConfig]) -> None:
        config = config_type()
        assert config.save_async is True
        assert config.checkpoint_components == list(FULL_CHECKPOINT_COMPONENTS)
        assert config.keep_optim_last is None
        assert config.keep_model_last == 0
        assert config.checkpoint_store == CheckpointStoreConfig()
        assert config.upload == CheckpointUploadConfig()

    @pytest.mark.parametrize("config_type", [DPOConfig, GRPOConfig, OPDConfig, NFTConfig])
    def test_shared_dotlist_paths_round_trip(self, config_type: type[RunConfig]) -> None:
        schema = OmegaConf.structured(config_type, flags={"allow_objects": True})
        merged = OmegaConf.merge(
            schema,
            OmegaConf.from_dotlist(
                [
                    "save_async=false",
                    "checkpoint_components=[training_state,model]",
                    "keep_optim_last=2",
                    "keep_model_last=3",
                    "upload.enabled=true",
                    "upload.bucket=test-checkpoints",
                    "upload.prefix=posttrain/run",
                ]
            ),
        )
        config = instantiate(merged)
        assert isinstance(config, config_type)
        assert config.save_async is False
        assert config.checkpoint_components == ["model", "training_state"]
        assert config.keep_optim_last == 2
        assert config.keep_model_last == 3
        assert config.upload == CheckpointUploadConfig(
            enabled=True,
            bucket="test-checkpoints",
            prefix="posttrain/run",
        )

    def test_direct_object_store_dotlist_round_trip(self) -> None:
        schema = OmegaConf.structured(RunConfig, flags={"allow_objects": True})
        merged = OmegaConf.merge(
            schema,
            OmegaConf.from_dotlist(
                [
                    "checkpoint_format=dcp",
                    "checkpoint_store.backend=object_store",
                    "checkpoint_store.bucket=direct-checkpoints",
                    "checkpoint_store.credentials=credentials/<credential>.secret",
                    "checkpoint_store.prefix=posttrain/direct",
                ]
            ),
        )

        config = instantiate(merged)

        assert isinstance(config, RunConfig)
        assert config.checkpoint_store == CheckpointStoreConfig(
            backend="object_store",
            bucket="direct-checkpoints",
            credentials="credentials/<credential>.secret",
            prefix="posttrain/direct",
        )
        assert config.upload.enabled is False

    @pytest.mark.parametrize("config_type", [RunConfig, DPOConfig, GRPOConfig, OPDConfig])
    @pytest.mark.parametrize("field_name", ["keep_optim_last", "keep_model_last"])
    @pytest.mark.parametrize("value", [True, False, 1.5, "1"])
    def test_retention_rejects_non_integer_values(
        self,
        config_type: type[RunConfig],
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises(TypeError, match=field_name):
            config_type(**{field_name: value})  # type: ignore[arg-type]

    def test_model_retention_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="keep_model_last"):
            RunConfig(keep_model_last=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("config_type", [RunConfig, DPOConfig, GRPOConfig, OPDConfig, NFTConfig])
    @pytest.mark.parametrize("field_name", ["keep_optim_last", "keep_model_last"])
    def test_retention_rejects_negative_values(
        self,
        config_type: type[RunConfig],
        field_name: str,
    ) -> None:
        with pytest.raises(ValueError, match=field_name):
            config_type(**{field_name: -1})  # type: ignore[arg-type]

    @pytest.mark.parametrize("config_type", [RunConfig, DPOConfig, GRPOConfig, OPDConfig, NFTConfig])
    def test_emergency_checkpoint_components_defaults_to_full_and_accepts_run_only(
        self, config_type: type[RunConfig]
    ) -> None:
        """The emergency-save plan selector lives on the shared RunConfig: "full" by default (a termination
        trigger upgrades to the exact-resume plan), "run" opts into the run's own cadence plan, anything
        else is refused at config time — including through the `rl.*` dotlist path recipes use."""
        assert config_type().emergency_checkpoint_components == "full"
        assert config_type(emergency_checkpoint_components="run").emergency_checkpoint_components == "run"
        with pytest.raises(ValueError, match="emergency_checkpoint_components must be one of"):
            config_type(emergency_checkpoint_components="model")
        with pytest.raises(ValueError, match="emergency_checkpoint_components must be one of"):
            config_type(emergency_checkpoint_components="")
        if config_type is RunConfig:
            return  # the dotlist path below needs a structured leaf config
        schema = OmegaConf.structured(config_type, flags={"allow_objects": True})
        merged = OmegaConf.merge(
            schema,
            OmegaConf.from_dotlist(["checkpoint_components=[model]", "emergency_checkpoint_components=run"]),
        )
        config = instantiate(merged)
        assert isinstance(config, config_type)
        assert config.checkpoint_components == ["model"]
        assert config.emergency_checkpoint_components == "run"

    def test_emergency_run_plan_requires_the_model_in_a_saving_validation_plan(self) -> None:
        """Review of b7fa3ebe4a (thread a4491f8c): under "run", a step's only artifact before the emergency save can be
        an off-cadence validation artifact; a committed artifact is never re-committed, so if it lacked the weights the
        emergency save could only write the full plan -- the TiB-scale save "run" exists to avoid. The config refuses
        that combination; a validation plan with the model, a validation that saves nothing, or the default "full"
        emergency plan are accepted."""
        dpo_data = PreferenceEnvConfig(factory="pkg.data.build_preferences", shuffle=False)
        grpo_data = DatasetConfig(factory="pkg.data.build_prompts", prompt_column="question")
        for config_type, validation_type, data in (
            (DPOConfig, DPOValidationConfig, dpo_data),
            (GRPOConfig, GRPOValidationConfig, grpo_data),
        ):
            accepted = config_type(
                checkpoint_components=["model"],
                emergency_checkpoint_components="run",
                validation=validation_type(
                    enabled=True, save_checkpoint=True, checkpoint_components=["model", "training_state"], data=data
                ),
            )
            assert accepted.validation.checkpoint_components == ["model", "training_state"]
            config_type(  # saves no validation artifact: nothing can shadow the emergency save
                emergency_checkpoint_components="run",
                validation=validation_type(
                    enabled=True, save_checkpoint=False, checkpoint_components=["training_state"], data=data
                ),
            )
            config_type(  # "full": the emergency save upgrades to the full plan whatever was saved
                emergency_checkpoint_components="full",
                validation=validation_type(
                    enabled=True, save_checkpoint=True, checkpoint_components=["training_state"], data=data
                ),
            )
            with pytest.raises(ValueError, match=r"to cover checkpoint_components plus 'model'.*missing \['model'\]"):
                config_type(
                    checkpoint_components=["model"],
                    emergency_checkpoint_components="run",
                    validation=validation_type(
                        enabled=True, save_checkpoint=True, checkpoint_components=["training_state"], data=data
                    ),
                )

    def test_emergency_run_plan_requires_the_validation_plan_to_cover_the_run_plan(self) -> None:
        """Review of 418cfb184da (LDCR 2026-10-01, mrs-ours V11): requiring only the model in a saving validation plan left
        the TiB-scale escalation reachable. Run plan `[model, training_state]` with validation plan `[model]` was
        ACCEPTED, and the coordinator's emergency save of a step holding that validation artifact wrote the FULL plan --
        the optimizer added, the one component "run" exists to avoid. The emergency plan under "run" is the run plan
        UNION {model}; a saving validation plan must cover every component of it, so its artifact stands in for the
        emergency save. A covering plan, the default (full) validation plan and the default "full" selector are
        accepted."""
        dpo_data = PreferenceEnvConfig(factory="pkg.data.build_preferences", shuffle=False)
        grpo_data = DatasetConfig(factory="pkg.data.build_prompts", prompt_column="question")
        for config_type, validation_type, data in (
            (DPOConfig, DPOValidationConfig, dpo_data),
            (GRPOConfig, GRPOValidationConfig, grpo_data),
        ):
            with pytest.raises(
                ValueError, match=r"to cover checkpoint_components plus 'model'.*missing \['training_state'\]"
            ):
                config_type(
                    checkpoint_components=["model", "training_state"],
                    emergency_checkpoint_components="run",
                    validation=validation_type(
                        enabled=True, save_checkpoint=True, checkpoint_components=["model"], data=data
                    ),
                )
            with pytest.raises(ValueError, match=r"missing \['optimizer'\]"):
                config_type(
                    checkpoint_components=["model", "optimizer"],
                    emergency_checkpoint_components="run",
                    validation=validation_type(
                        enabled=True, save_checkpoint=True, checkpoint_components=["model"], data=data
                    ),
                )
            with pytest.raises(ValueError, match=r"missing \['model', 'optimizer'\]"):
                config_type(
                    checkpoint_components=["optimizer"],
                    emergency_checkpoint_components="run",
                    validation=validation_type(
                        enabled=True, save_checkpoint=True, checkpoint_components=["training_state"], data=data
                    ),
                )
            covering = config_type(  # the validation plan holds every component of the emergency plan
                checkpoint_components=["model", "training_state"],
                emergency_checkpoint_components="run",
                validation=validation_type(
                    enabled=True, save_checkpoint=True, checkpoint_components=["model", "training_state"], data=data
                ),
            )
            assert covering.validation.checkpoint_components == ["model", "training_state"]
            full_validation = config_type(  # the default validation plan is the full plan: it covers any run plan
                checkpoint_components=["model", "optimizer"],
                emergency_checkpoint_components="run",
                validation=validation_type(enabled=True, save_checkpoint=True, data=data),
            )
            assert full_validation.validation.checkpoint_components == ["model", "optimizer", "training_state"]
            config_type(  # "full": the emergency save upgrades to the full plan whatever was saved
                checkpoint_components=["model", "training_state"],
                emergency_checkpoint_components="full",
                validation=validation_type(
                    enabled=True, save_checkpoint=True, checkpoint_components=["model"], data=data
                ),
            )

    @pytest.mark.parametrize(
        ("components", "error", "message"),
        [
            ("model", TypeError, "iterable of component names"),
            ([], ValueError, "at least one component"),
            (["model", "model"], ValueError, "duplicates"),
            (["model", "optim"], ValueError, "unsupported checkpoint components"),
            (["model", 1], TypeError, "names must be strings"),
        ],
    )
    def test_checkpoint_components_reject_invalid_selections(
        self,
        components: object,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            CheckpointComponentPlan.from_value(components)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_on_policy_temperature_mismatch_fails_before_backend_or_placement() -> None:
    """The generic preflight checks policy identity only after the distribution contract."""
    with pytest.raises(ValueError, match="rollout.temperature == rl.temperature"):
        build.validate_online_policy_backend(
            GRPOConfig(temperature=0.9),
            RolloutConfig(temperature=1.0),
            model_backend=None,
            model_config=None,
        )

    build.validate_online_policy_backend(
        GRPOConfig(temperature=0.9),
        RolloutConfig(temperature=0.9),
        model_backend=None,
        model_config=None,
    )

    build.validate_online_policy_backend(
        GRPOConfig(grpo_surrogate="legacy_pg", temperature=0.9),
        RolloutConfig(temperature=0.9),
        model_backend=None,
        model_config=None,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("max_tokens", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_rollout_max_tokens_is_a_positive_integer_contract(
    max_tokens: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="rollout.max_tokens"):
        RolloutConfig(max_tokens=max_tokens)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_max_tokens_preflight_catches_post_construction_overrides() -> None:
    rollout = RolloutConfig()
    rollout.max_tokens = 0

    with pytest.raises(ValueError, match="rollout.max_tokens"):
        build.validate_online_policy_backend(
            GRPOConfig(),
            rollout,
            model_backend=None,
            model_config=None,
        )


@pytest.mark.L0
@pytest.mark.CPU
class TestModelValidationConfig:
    def test_validation_defaults_disabled_and_final_is_implicit(self) -> None:
        dpo = DPOConfig()
        grpo = GRPOConfig()
        assert isinstance(dpo.validation, DPOValidationConfig)
        assert isinstance(grpo.validation, GRPOValidationConfig)
        assert dpo.validation.enabled is False and grpo.validation.enabled is False
        assert dpo.validation.before_training is False
        assert dpo.validation.save_checkpoint is False
        assert dpo.validation.checkpoint_components == list(FULL_CHECKPOINT_COMPONENTS)
        assert grpo.validation.every_steps == 20
        assert not hasattr(grpo.validation, "at_end")
        assert grpo.validation.sampling == ValidationSamplingConfig()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"every_steps": -1}, "every_steps"),
            ({"batch_size": 0}, "batch_size"),
            ({"max_samples": 0}, "max_samples"),
        ],
    )
    def test_schedule_rejects_invalid_values(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ValidationScheduleConfig(**kwargs)

    def test_enabling_requires_explicit_held_out_source(self) -> None:
        with pytest.raises(ValueError, match="explicit rl.validation.data"):
            DPOConfig(validation=DPOValidationConfig(enabled=True))
        with pytest.raises(ValueError, match="explicit rl.validation.data"):
            GRPOConfig(validation=GRPOValidationConfig(enabled=True))

    def test_dpo_validation_accepts_a_custom_factory_source(self) -> None:
        dpo_data = PreferenceEnvConfig(factory="pkg.data.build_preferences", shuffle=False)
        dpo = DPOConfig(validation=DPOValidationConfig(enabled=True, data=dpo_data))

        assert dpo.validation.data is dpo_data

    def test_grpo_validation_accepts_a_custom_factory_source(self) -> None:
        grpo_data = DatasetConfig(factory="pkg.data.build_prompts", prompt_column="question")
        grpo = GRPOConfig(validation=GRPOValidationConfig(enabled=True, data=grpo_data))

        assert grpo.validation.data is grpo_data

    def test_dpo_synthetic_validation_uses_an_independent_seed(self) -> None:
        val_data = PreferenceEnvConfig(
            synthetic_pairs=4,
            shuffle=False,
            cursor_resume=CURSOR_RESUME_OFF,
        )
        validation = DPOValidationConfig(enabled=True, seed=1, data=val_data)
        config = DPOConfig(data=PreferenceEnvConfig(synthetic_pairs=8), validation=validation, seed=0)
        assert config.validation.data.synthetic_pairs == 4
        with pytest.raises(ValueError, match="must be distinct"):
            DPOConfig(
                data=PreferenceEnvConfig(synthetic_pairs=8),
                validation=DPOValidationConfig(enabled=True, seed=0, data=val_data),
                seed=0,
            )

    def test_grpo_rejects_same_jsonl_for_training_and_validation(self) -> None:
        source = DatasetConfig(path="pairs.jsonl", prompt_column="question", response_column="answer")
        with pytest.raises(ValueError, match="must be distinct"):
            GRPOConfig(data=source, validation=GRPOValidationConfig(enabled=True, data=source))
        config = GRPOConfig(
            data=source,
            validation=GRPOValidationConfig(
                enabled=True,
                data=DatasetConfig(path="held_out.jsonl", prompt_column="question", response_column="answer"),
            ),
        )
        assert config.validation.data.path == "held_out.jsonl"
        with pytest.raises(ValueError, match="every_steps.*n_mini"):
            GRPOConfig(
                batch_per_replica=8,
                mini_batch_per_replica=4,
                validation=GRPOValidationConfig(
                    enabled=True,
                    every_steps=1,
                    data=DatasetConfig(path="held_out.jsonl", prompt_column="question", response_column="answer"),
                ),
            )

    def test_grpo_rejects_same_hf_split_even_when_row_transform_differs(self) -> None:
        train = DatasetConfig(
            name="org/dataset",
            subset="main",
            split="train",
            prompt_column="question",
            response_column="answer",
            items_fn="pkg.train.transform",
        )
        held_out = DatasetConfig(
            name="org/dataset",
            subset="main",
            split="train",
            prompt_column="question",
            response_column="answer",
            items_fn="pkg.validation.transform",
        )
        with pytest.raises(ValueError, match="must be distinct"):
            GRPOConfig(data=train, validation=GRPOValidationConfig(enabled=True, data=held_out))

    def test_validation_sampling_guards(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            ValidationSamplingConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="top_p"):
            ValidationSamplingConfig(top_p=0.0)
        with pytest.raises(ValueError, match="n_generation"):
            ValidationSamplingConfig(n_generation=0)

    @pytest.mark.parametrize(
        ("factory", "kwargs", "error"),
        [
            (ValidationScheduleConfig, {"enabled": 1}, TypeError),
            (ValidationScheduleConfig, {"save_checkpoint": 1}, TypeError),
            (ValidationScheduleConfig, {"every_steps": True}, TypeError),
            (ValidationScheduleConfig, {"max_samples": 1.5}, TypeError),
            (ValidationSamplingConfig, {"temperature": float("nan")}, ValueError),
            (ValidationSamplingConfig, {"repetition_penalty": float("inf")}, ValueError),
            (ValidationSamplingConfig, {"top_k": True}, TypeError),
            (ValidationSamplingConfig, {"n_generation": 1.5}, TypeError),
        ],
    )
    def test_validation_rejects_bool_fractional_and_nonfinite_values(
        self,
        factory: type,
        kwargs: dict[str, object],
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            factory(**kwargs)

    def test_validation_structured_round_trip(self) -> None:
        authored = OmegaConf.structured(
            GRPOConfig(
                validation=GRPOValidationConfig(
                    enabled=True,
                    every_steps=5,
                    save_checkpoint=True,
                    checkpoint_components=["training_state", "model"],
                    data=DatasetConfig(path="held_out.jsonl", prompt_column="q", response_column="a"),
                )
            )
        )
        obj = instantiate(
            OmegaConf.merge(OmegaConf.structured(GRPOConfig), OmegaConf.create(OmegaConf.to_yaml(authored)))
        )
        assert isinstance(obj.validation, GRPOValidationConfig)
        assert obj.validation.enabled is True and obj.validation.every_steps == 5
        assert obj.validation.save_checkpoint is True
        assert obj.validation.checkpoint_components == ["model", "training_state"]


@pytest.mark.L0
@pytest.mark.CPU
class TestStructuredRoundTrip:
    """author → to_yaml → reload-with-schema → instantiate → typed dataclass (the real
    save_yaml/load path, at the rl-node level)."""

    def test_yaml_round_trip_preserves_typed_values(self):
        authored = OmegaConf.structured(DPOConfig(beta=1000.0, optimizer=OptimizerConfig(lr=5e-7)))
        y = OmegaConf.to_yaml(authored)
        # reload mirrors how the typed Config field re-binds the schema on load
        schema = OmegaConf.structured(DPOConfig)
        reloaded = OmegaConf.merge(schema, OmegaConf.create(y))
        obj = instantiate(reloaded)
        assert isinstance(obj, DPOConfig)
        assert obj.beta == 1000.0
        assert isinstance(obj.optimizer, OptimizerConfig)
        assert obj.optimizer.lr == 5e-7

    def test_typed_dotlist_override(self):
        schema = OmegaConf.structured(DPOConfig)
        merged = OmegaConf.merge(schema, OmegaConf.from_dotlist(["beta=2000.0", "grad_accum=4", "optimizer.lr=1e-7"]))
        obj = instantiate(merged)
        assert obj.beta == 2000.0 and obj.grad_accum == 4 and obj.optimizer.lr == 1e-7

    def test_unknown_key_rejected(self):
        schema = OmegaConf.structured(DPOConfig)
        with pytest.raises(Exception):
            OmegaConf.merge(schema, OmegaConf.from_dotlist(["nonexistent_key=1"]))


@pytest.mark.L0
@pytest.mark.CPU
class TestBuildOptimizer:
    def _toy(self):
        import torch.nn as nn

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_gen = nn.Linear(4, 4)
                self.backbone = nn.Linear(4, 4)

        return Toy()

    def test_freezes_non_train_keys(self):
        m = self._toy()
        opt = build.build_optimizer(m, OptimizerConfig(lr=1e-5, train_keys=["moe_gen"]))
        assert m.moe_gen.weight.requires_grad is True
        assert m.backbone.weight.requires_grad is False
        assert sum(len(g["params"]) for g in opt.param_groups) == 2  # moe_gen weight+bias
        assert opt.param_groups[0]["lr"] == 1e-5

    def test_empty_train_keys_trains_all(self):
        m = self._toy()
        opt = build.build_optimizer(m, OptimizerConfig(train_keys=[]))
        assert sum(len(g["params"]) for g in opt.param_groups) == 4
        assert all(p.requires_grad for p in m.parameters())


@pytest.mark.L0
@pytest.mark.CPU
class TestHFBackend:
    def test_hf_loader_takes_pdims_not_world_size(self):
        # HF multi-GPU FSDP is wired now: build_hf_model no longer raises NotImplementedError
        # nor takes world_size — the mesh arrives via `pdims` (HFModel.setup). See the
        # single-device + FSDP coverage in test_config_build_hf.py / test_model_hf.py.
        import inspect

        from cosmos_framework.posttrain.model.hf import build_hf_model

        params = inspect.signature(build_hf_model).parameters
        assert "world_size" not in params
        assert "pdims" in params


@pytest.mark.L0
@pytest.mark.CPU
def test_items_fn_is_accepted_and_defaults_empty():
    from cosmos_framework.posttrain.config.rl import DatasetConfig

    assert DatasetConfig(name="any/dataset", subset="main", prompt_column="q", response_column="a").items_fn == ""
    cfg = DatasetConfig(
        name="any/dataset", subset="main", prompt_column="q", response_column="a", items_fn="pkg.mod.fn"
    )
    assert cfg.items_fn == "pkg.mod.fn"


@pytest.mark.L0
@pytest.mark.CPU
def test_any_hub_dataset_is_allowed():
    """The allowlist is gone: a dataset is columns + an optional items_fn, not a wired-in env."""
    from cosmos_framework.posttrain.config.rl import DatasetConfig

    DatasetConfig(name="squad", subset="plain_text", prompt_column="question", response_column="answers")


@pytest.mark.L0
@pytest.mark.CPU
def test_items_fn_and_jsonl_path_are_mutually_exclusive():
    from cosmos_framework.posttrain.config.rl import DatasetConfig

    with pytest.raises(ValueError, match="items_fn"):
        DatasetConfig(path="/tmp/p.jsonl", prompt_column="q", items_fn="pkg.mod.fn")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "batch,mini,micro,match,error",
    [
        (64, 24, 8, "mini_batch_per_replica", ValueError),  # 64 % 24 != 0
        (64, 32, 12, "micro_batch_per_replica", ValueError),  # 32 % 12 != 0
        (64, 96, 8, "mini_batch_per_replica", ValueError),  # mini > batch ⇒ n_mini == 0
        (64, 32, 64, "micro_batch_per_replica", ValueError),  # micro > mini ⇒ n_micro == 0
        (True, 0, 0, "batch_per_replica", TypeError),
        (64, -32, 0, "mini_batch_per_replica", ValueError),
        (64, 32, -8, "micro_batch_per_replica", ValueError),
        (64, 32, 8.0, "micro_batch_per_replica", TypeError),
    ],
)
def test_batching_fields_reject_non_dividing_combinations(
    batch: int | float | bool,
    mini: int | float | bool,
    micro: int | float | bool,
    match: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=match):
        GRPOConfig(
            batch_per_replica=batch,
            group_size=8,
            mini_batch_per_replica=mini,
            micro_batch_per_replica=micro,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_zero_means_degenerate_to_today():
    cfg = GRPOConfig(batch_per_replica=64, group_size=8)  # mini/micro both default 0
    assert (cfg.n_mini, cfg.n_micro) == (1, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_rejects_a_non_default_grad_accum():
    """Retired as a GRPO knob — the trainer's value is derived from n_micro (§5.2)."""
    with pytest.raises(ValueError, match="grad_accum"):
        GRPOConfig(batch_per_replica=64, group_size=8, grad_accum=4)


@pytest.mark.L0
@pytest.mark.CPU
def test_prompts_per_data_replica_derives_from_batch_per_replica():
    """batch_per_replica is the PRIMARY knob; the data role's per-replica prompt draw is DERIVED
    from it by the conservation law (N_data == N_policy today, so the counts cancel to
    batch // group_size). One knob, so no second count can silently disagree."""
    cfg = GRPOConfig(batch_per_replica=64, group_size=8)
    assert cfg.prompts_per_data_replica == 8  # 64 // 8
    assert (cfg.n_mini, cfg.n_micro) == (1, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_grad_accum_is_unaffected():
    """`grad_accum` is retired as a GRPO knob only. DPO really uses it —
    wallace_8b_hpdv3_repro_v1.py:51 sets 32, ga_32b_hpdv3_repro_v1.py:58 sets 4 — and DPO's
    MEAN reduction is untouched by any of this (§5.2)."""
    assert DPOConfig(grad_accum=4).grad_accum == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_unimportable_items_fn_fails_on_the_driver():
    """An items_fn typo must fail GPU-lessly at --validate, not as an ImportError inside a Ray
    actor after placement and model construction have already reserved the cluster."""
    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
    from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources

    rl = GRPOConfig(
        data=DatasetConfig(
            name="openai/gsm8k",
            subset="main",
            prompt_column="question",
            response_column="answer",
            items_fn="no.such.module.build_items",
        )
    )
    with pytest.raises(ValueError, match="no.such.module.build_items"):
        validate_rollout_reward_sources(rl)


@pytest.mark.L0
@pytest.mark.CPU
def test_batch_per_replica_must_divide_by_group_size():
    """Each data replica draws batch_per_replica // group_size prompts, so the division must be
    exact (else the policy's batch is silently mis-filled) and batch_per_replica >= group_size
    (else zero prompts). group_size is the GRPO group; batch_per_replica is the trajectory count."""
    with pytest.raises(ValueError, match="multiple of"):
        GRPOConfig(batch_per_replica=60, group_size=8)  # 60 % 8 != 0
    with pytest.raises(ValueError, match="multiple of"):
        GRPOConfig(batch_per_replica=4, group_size=8)  # 4 < 8 -> zero prompts
    cfg = GRPOConfig(batch_per_replica=64, group_size=8)  # exact
    assert cfg.prompts_per_data_replica == 8


# --------------------------------------------------------------------------- §8.2/§8.4 grid.shapes


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_fallback_separates_policy_and_rollout_replicas() -> None:
    grid = GridConfig(num_replicas=4, ranks=2)
    assert grid.policy_shape() == (1, 2)
    assert grid.rollout_shape() == (4, 2)
    assert grid.gpu_placement == "colocated"


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_gpu_placement_is_an_explicit_axis() -> None:
    assert GridConfig(gpu_placement="disaggregated").gpu_placement == "disaggregated"
    with pytest.raises(ValueError, match="gpu_placement"):
        GridConfig(gpu_placement="nixl")


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_shapes_admits_unequal_products_for_transport_aware_bind() -> None:
    """The schema normalizes shapes; placement decides whether IPC or NIXL can serve them."""
    grid = GridConfig(shapes={"policy": [1, 4], "rollout": [1, 2]})
    assert grid.policy_shape() == (1, 4)
    assert grid.rollout_shape() == (1, 2)


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_shapes_rejects_policy_execution_replication() -> None:
    with pytest.raises(ValueError, match="POLICY_REPLICAS must be 1"):
        GridConfig(shapes={"policy": [2, 4], "rollout": [4, 2]})


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_shapes_unknown_key_raises_not_silently_ignored():
    """A typo'd key must RAISE, not fall back to a default — the 'configured, then ignored' failure
    this branch has already paid for twice."""
    with pytest.raises(ValueError, match="polciy|unknown|policy.*rollout"):
        GridConfig(shapes={"polciy": [2, 4], "rollout": [4, 2]})


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_shapes_survives_omegaconf_dotlist_roundtrip():
    """The real user path: `grid.shapes.policy=[R,K]` via `--` dotlist. __post_init__ mutates
    `shapes` to (R,K) tuples, so this pins that OmegaConf's structured schema + to_object round-trip
    preserves it (a regression here would silently drop or mangle a user's topology)."""
    base = OmegaConf.structured(GridConfig())
    merged = OmegaConf.merge(
        base,
        OmegaConf.from_dotlist(["gpu_placement=disaggregated", "shapes.policy=[1,8]", "shapes.rollout=[4,2]"]),
    )
    g = OmegaConf.to_object(merged)  # re-instantiates GridConfig -> __post_init__ validates
    assert g.gpu_placement == "disaggregated"
    assert g.policy_shape() == (1, 8)
    assert g.rollout_shape() == (4, 2)
    unequal = OmegaConf.merge(base, OmegaConf.from_dotlist(["shapes.policy=[1,4]", "shapes.rollout=[2,1]"]))
    unequal_grid = OmegaConf.to_object(unequal)
    assert unequal_grid.policy_shape() == (1, 4)
    assert unequal_grid.rollout_shape() == (2, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_shapes_rejects_zero_or_negative_axes():
    """[R, K] must be positive: a 0 or negative axis is an invalid topology (empty/negative actor
    ranges downstream), and the equal-product check alone lets 0==0 through."""
    for bad in ([0, 4], [-2, 4], [2, 0]):
        with pytest.raises(ValueError, match="positive|>= 1|R, K"):
            GridConfig(shapes={"policy": bad, "rollout": [4, 2]})


@pytest.mark.L0
@pytest.mark.CPU
def test_gsm8k_recipe_topology_can_deliver_its_own_param_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe asking for storage != compute must ship a topology that installs FSDP.

    `parallelize()` refuses to emulate `param_dtype` without a wrap (no MixedPrecisionPolicy means
    the forward silently runs in the storage dtype), so a split-dtype recipe left on the
    single-device default cannot start at all. dp_shard is the grid's ranks-per-replica, which is
    the rollout TP.
    """
    from cosmos_framework.posttrain.configs import grpo_qwen3b_gsm8k as recipe

    skip_recipe_registration(monkeypatch, recipe._make_root_config, PosttrainGRPOConfig)
    cfg = recipe.build_config()
    assert cfg.model_backend == "hf"
    assert cfg.rl.grpo_surrogate == "clipped"
    if cfg.rl.master_dtype != cfg.rl.param_dtype:
        assert cfg.rollout.tensor_parallel_size > 1, (
            f"recipe wants master={cfg.rl.master_dtype} != param={cfg.rl.param_dtype} but ships "
            f"dp_shard={cfg.rollout.tensor_parallel_size}: parallelize() will refuse it"
        )
        assert cfg.grid.backend == "ray"


@pytest.mark.L0
@pytest.mark.CPU
def test_gsm8k_recipe_does_not_load_the_unrelated_experiment_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A concrete HF recipe must not import every Cosmos3 experiment and its optional dependencies."""
    from cosmos_framework.posttrain.configs import grpo_qwen3b_gsm8k as recipe

    register_experiments: bool | None = None

    def make_root(*, register_experiment_configs: bool = True) -> PosttrainGRPOConfig:
        nonlocal register_experiments
        register_experiments = register_experiment_configs
        return PosttrainGRPOConfig(
            model=None,
            optimizer=None,
            scheduler=None,
            dataloader_train=None,
            dataloader_val=None,
        )

    monkeypatch.setattr(recipe, "_make_root_config", make_root)
    recipe.build_config()

    assert register_experiments is False


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_pg_may_mini_batch_without_the_determinism_attestation():
    """`grpo_surrogate="legacy_pg"` carries no π_old cache.

    `config/build.py` hands it `make_objective` — a plain policy gradient with no importance ratio —
    and the driver skips the ahead pass for it via `needs_old_logprobs()`. Demanding the attestation
    there took mini-batching away from the legacy objective entirely, for a hazard it cannot have.
    """
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    cfg = GRPOConfig(grpo_surrogate="legacy_pg")
    cfg.batch_per_replica = 8
    cfg.mini_batch_per_replica = 2  # n_mini = 4 > 1
    cfg.__post_init__()  # must not raise
    assert not cfg._objective_consults_pi_old()


@pytest.mark.L0
@pytest.mark.CPU
def test_clipped_objective_still_requires_the_determinism_attestation():
    """The clipped surrogate is the one that forms a ratio against a cached π_old, so it keeps the
    gate — this is the path the ahead pass actually runs on."""
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    cfg = GRPOConfig()
    cfg.grpo_surrogate = "clipped"
    cfg.batch_per_replica = 8
    cfg.mini_batch_per_replica = 2
    with pytest.raises(ValueError, match="certified_deterministic_forward"):
        cfg.__post_init__()
    cfg.certified_deterministic_forward = True
    cfg.__post_init__()  # attested -> accepted


@pytest.mark.L0
@pytest.mark.CPU
def test_single_rank_collapses_the_dtype_split_whatever_the_rollout_flavor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANY single-rank launch installs no FSDP — hf, or vllm at TP=1.

    Without a wrapper there is no MixedPrecisionPolicy, so `parallelize()` refuses a `param_dtype`
    that differs from storage rather than silently computing in the storage dtype. The recipe's
    fp32-master/bf16-compute split is a parity feature of the MULTI-GPU configuration, so the HF
    flavor collapses it instead of forcing a multi-rank launch — otherwise the documented
    `--nproc_per_node=1` command cannot start at all.
    """
    from cosmos_framework.posttrain.configs import grpo_qwen3b_gsm8k as recipe

    skip_recipe_registration(monkeypatch, recipe._make_root_config, PosttrainGRPOConfig)
    cfg = recipe.build_config()
    assert cfg.rl.master_dtype != cfg.rl.param_dtype, "recipe no longer splits dtypes; test is stale"

    # The rule keys on TOPOLOGY rather than on any launcher flavor: the same collapse must apply
    # at `rollout.tensor_parallel_size=1`, which was the third instance of this bug. The rank count
    # comes from the ROLLOUT TP; deriving it from a separate launcher knob collapsed the split on
    # the default two-GPU run.
    assert cfg.rollout.tensor_parallel_size == 2, "recipe topology changed; this test is stale"
    single_rank_probe = recipe.build_config()
    if single_rank_probe.rl.master_dtype != single_rank_probe.rl.param_dtype:
        single_rank_probe.rl.master_dtype = single_rank_probe.rl.param_dtype
    assert single_rank_probe.rl.master_dtype == single_rank_probe.rl.param_dtype, "single rank must not keep a split"
    assert cfg.rl.master_dtype != cfg.rl.param_dtype, "multi-rank must KEEP the parity split"
    if cfg.rl.master_dtype != cfg.rl.param_dtype:
        cfg.rl.master_dtype = cfg.rl.param_dtype
    assert cfg.rl.master_dtype == cfg.rl.param_dtype == "bfloat16", (
        "compute dtype must win: the forward should still be the bf16 the recipe asked for"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_logprobs_mode_is_not_asked_for_a_determinism_attestation():
    """`use_rollout_logprobs_for_loss` cannot have the hazard the attestation guards, so do not ask.

    Under that flag the objective takes the `_gather_old` branch, never touches `OldLogprobCache`,
    and the driver runs no ahead pass -- there is no eval-mode pi_old to compare against a train-mode
    forward. Demanding the attestation there rejects a dropout-enabled model that is perfectly valid
    in this mode, which is a wrong refusal even though it fails closed.

    `needs_cached_old_logprobs` already answers "is there an ahead pass" and already carries the
    clause. This pins that the config gate asks IT rather than re-deriving `n_mini > 1` on its own.
    """
    from cosmos_framework.posttrain.config.rl import GRPOConfig

    def _cfg(use_rollout_logprobs):
        c = GRPOConfig()
        c.grpo_surrogate = "clipped"
        c.batch_per_replica, c.group_size = 128, 16
        c.mini_batch_per_replica = 32  # n_mini = 4
        c.use_rollout_logprobs_for_loss = use_rollout_logprobs
        c.certified_deterministic_forward = False
        return c

    # No ahead pass in this mode, so no attestation demanded.
    cfg = _cfg(True)
    assert cfg.needs_cached_old_logprobs is False
    cfg.__post_init__()

    # The ahead pass DOES run without the flag, so the attestation is still demanded there.
    with pytest.raises(ValueError, match="certified_deterministic_forward"):
        _cfg(False).__post_init__()


def _sequence_level_kwargs(**overrides: object) -> dict[str, object]:
    """Legal GSPO-shaped `rl` section: n_mini = 128 / 32 = 4, attested forward, paper epsilons, no dual clip."""
    kwargs: dict[str, object] = {
        "importance_sampling_level": "sequence",
        "lower_bound_ratio": 0.0,
        "batch_per_replica": 128,
        "group_size": 16,
        "mini_batch_per_replica": 32,
        "certified_deterministic_forward": True,
        "epsilon_low": 3e-4,
        "epsilon_high": 4e-4,
    }
    return kwargs | overrides


@pytest.mark.L0
@pytest.mark.CPU
def test_routing_replay_accepts_off_and_rollout() -> None:
    """`rl.routing_replay` is off|rollout, refused by legacy_pg, OPD and a policy that cannot replay; both roles read it."""
    from types import SimpleNamespace

    assert GRPOConfig().routing_replay == "off"
    assert GRPOConfig(routing_replay="rollout").routing_replay == "rollout"
    for value in ("on", "rollout "):
        with pytest.raises(ValueError, match=r"routing_replay must be one of \['off', 'rollout'\]"):
            GRPOConfig(routing_replay=value)
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings: routing_replay='rollout'"):
        GRPOConfig(routing_replay="rollout", grpo_surrogate="legacy_pg")
    with pytest.raises(ValueError, match="OPD does not consume the GRPO-only rl.routing_replay"):
        OPDConfig(routing_replay="rollout")

    # Trainer side: the source binds only to a ForwardReplayPolicy (a dense or HF policy is refused at assembly).
    replaying = GRPOConfig(routing_replay="rollout")
    capable = SimpleNamespace(replay_pinned=lambda: False, replay_pinned_extent=lambda: (0, 0), replay_identity=dict)
    assert build.forward_replay_source(GRPOConfig(), object()) is None
    assert build.forward_replay_source(replaying, capable) == "rollout"
    with pytest.raises(ValueError, match="ForwardReplayPolicy"):
        build.forward_replay_source(replaying, object())
    # Rollout side: the same knob is the one reason the engine returns its routing.
    assert (build.rollout_records_routing(GRPOConfig()), build.rollout_records_routing(replaying)) == (False, True)
    assert "enable_return_routed_experts" not in build.vllm_engine_kwargs(RolloutConfig())
    assert build.vllm_engine_kwargs(RolloutConfig(), return_routed_experts=True)["enable_return_routed_experts"] is True


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_level_config_rejections() -> None:
    """Unknown level, the class-default dual clip (3.0 is ON), legacy_pg, and n_mini > 1 without the
    determinism attestation are all refused at construction."""
    assert GRPOConfig().importance_sampling_level == "token"
    with pytest.raises(ValueError, match="importance_sampling_level must be 'token' or 'sequence'"):
        GRPOConfig(importance_sampling_level="response")
    with pytest.raises(ValueError, match="applies no dual clip"):
        GRPOConfig(importance_sampling_level="sequence")
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings"):
        GRPOConfig(grpo_surrogate="legacy_pg", importance_sampling_level="sequence")
    with pytest.raises(ValueError, match="certified_deterministic_forward"):
        GRPOConfig(**_sequence_level_kwargs(certified_deterministic_forward=False))


@pytest.mark.L0
@pytest.mark.CPU
def test_sequence_level_config_warnings() -> None:
    """Honored-but-inert knobs warn: a unit ratio at n_mini == 1, token-scale epsilons, bf16 logprobs.
    Rollout logprobs at n_mini == 1 and the full GSPO shape are silent."""
    on_policy = _sequence_level_kwargs(mini_batch_per_replica=0)  # n_mini == 1; the attestation is unread there
    with pytest.warns(UserWarning, match="exactly 1"):
        GRPOConfig(**on_policy)
    with pytest.warns(UserWarning, match="response-MEAN"):
        GRPOConfig(**_sequence_level_kwargs(epsilon_low=0.2, epsilon_high=0.2))
    with pytest.warns(UserWarning, match="bf16"):
        GRPOConfig(**_sequence_level_kwargs(logprob_dtype="bfloat16"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not GRPOConfig(**on_policy, use_rollout_logprobs_for_loss=True).needs_cached_old_logprobs
        assert GRPOConfig(**_sequence_level_kwargs()).n_mini == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_fixed_normaliser_optimizer_eps_warning() -> None:
    """`seq-mean-token-sum-fixed` divides by the fixed `rollout.max_tokens`, so per-parameter gradients fall
    below an Adam eps sized for token-mean (the recipes' 1e-6) and the update degenerates to (lr/eps)*m.
    The config warns for exactly that pair, once, naming the finding and the fix. Token-mean at the same
    eps, the fixed reduction at torch's 1e-8 default (also the OptimizerConfig default) and eps exactly at
    the ceiling are silent, and a section OPD refuses (it never consumes the fixed reduction) is refused
    without also warning."""
    fixed = "seq-mean-token-sum-fixed"
    assert FIXED_NORMALISER_OPTIMIZER_EPS_CEILING == 1e-7
    with pytest.warns(UserWarning, match="eps-dominated") as record:
        GRPOConfig(loss_type=fixed, optimizer=OptimizerConfig(eps=1e-6))
    messages = [str(w.message) for w in record if "eps-dominated" in str(w.message)]
    assert len(messages) == 1
    assert "rl.optimizer.eps=1e-06" in messages[0]
    assert "(lr/eps)*m" in messages[0] and "1e-8" in messages[0] and "re-tune lr" in messages[0]
    with pytest.warns(UserWarning, match="eps-dominated"):
        GRPOConfig(loss_type=fixed, optimizer=OptimizerConfig(eps=2e-7))  # just above the ceiling
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GRPOConfig(loss_type="token-mean", optimizer=OptimizerConfig(eps=1e-6))
        GRPOConfig(loss_type="seq-mean-token-mean", optimizer=OptimizerConfig(eps=1e-6))
        GRPOConfig(loss_type=fixed, optimizer=OptimizerConfig(eps=1e-8))
        GRPOConfig(loss_type=fixed, optimizer=OptimizerConfig(eps=FIXED_NORMALISER_OPTIMIZER_EPS_CEILING))
        assert GRPOConfig(loss_type=fixed).optimizer.eps == 1e-8
        with pytest.raises(ValueError, match="Dr-GRPO-only loss_type"):
            OPDConfig(loss_type=fixed, optimizer=OptimizerConfig(eps=1e-6))


@pytest.mark.L0
@pytest.mark.CPU
def test_dynamic_sampling_knobs_validate() -> None:
    """The share must fit the batch, the policy is proceed|raise, and OPD refuses the inherited knob."""
    assert GRPOConfig(min_informative_groups=4, group_size=8, batch_per_replica=32).min_informative_groups == 4
    rejected = [
        (GRPOConfig, {"min_informative_groups": True}, TypeError, "min_informative_groups"),
        (GRPOConfig, {"dynamic_sampling_max_extra_rollouts": -1}, ValueError, "dynamic_sampling_max_extra_rollouts"),
        (
            GRPOConfig,
            {"min_informative_groups": 5, "group_size": 8, "batch_per_replica": 32},
            ValueError,
            "more informative groups than the batch holds",
        ),
        (GRPOConfig, {"dynamic_sampling_on_budget_exhausted": "warn"}, ValueError, "proceed|raise"),
        (OPDConfig, {"min_informative_groups": 1}, ValueError, "does not consume min_informative_groups"),
    ]
    for config_type, kwargs, error, match in rejected:
        with pytest.raises(error, match=match):
            config_type(**kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_dynamic_sampling_refill_mode_is_opt_in_and_survives_structured_roundtrip() -> None:
    assert GRPOConfig().dynamic_sampling_refill_mode == "all"
    cfg = OmegaConf.structured(GRPOConfig(dynamic_sampling_refill_mode="short_cells"))
    assert (
        instantiate(OmegaConf.merge(OmegaConf.structured(GRPOConfig), cfg)).dynamic_sampling_refill_mode
        == "short_cells"
    )
    with pytest.raises(ValueError, match="dynamic_sampling_refill_mode must be"):
        GRPOConfig(dynamic_sampling_refill_mode="subset")


@pytest.mark.L0
@pytest.mark.CPU
def test_dynamic_sampling_quota_scope_is_opt_in_and_survives_structured_roundtrip() -> None:
    assert GRPOConfig().dynamic_sampling_quota_scope == "cell"
    cfg = OmegaConf.structured(GRPOConfig(dynamic_sampling_quota_scope="pooled"))
    assert instantiate(OmegaConf.merge(OmegaConf.structured(GRPOConfig), cfg)).dynamic_sampling_quota_scope == "pooled"
    with pytest.raises(ValueError, match="dynamic_sampling_quota_scope must be"):
        GRPOConfig(dynamic_sampling_quota_scope="global")


@pytest.mark.L0
@pytest.mark.CPU
def test_dynamic_sampling_refill_dispatch_validates() -> None:
    """`per_round` (default) or `whole_budget`; anything else is refused by type then by value."""
    layout = {"min_informative_groups": 4, "group_size": 8, "batch_per_replica": 32}
    assert GRPOConfig(**layout).dynamic_sampling_refill_dispatch == "per_round"
    assert (
        GRPOConfig(dynamic_sampling_refill_dispatch="whole_budget", **layout).dynamic_sampling_refill_dispatch
        == "whole_budget"
    )
    for value, error in (
        (True, TypeError),
        (1, TypeError),
        ("concurrent", ValueError),
        ("Whole_budget", ValueError),
    ):
        with pytest.raises(error, match="dynamic_sampling_refill_dispatch"):
            GRPOConfig(dynamic_sampling_refill_dispatch=value, **layout)


@pytest.mark.L0
@pytest.mark.CPU
def test_min_informative_groups_frac_resolves_against_groups_per_step() -> None:
    """The fraction sizes the quota as ceil(f * groups_per_step), groups_per_step = batch_per_replica // group_size.

    That capacity is placement-independent -- the data cells split the one policy batch -- so b32g4 is 8 groups
    and b256g4 is 64 whether it runs on one cell or sixteen. The stored fields keep what was authored; the loop
    reads the resolved quota.
    """
    narrow = {"batch_per_replica": 32, "group_size": 4}  # b32g4: 8 groups per step
    wide = {"batch_per_replica": 256, "group_size": 4}  # b256g4: 64 groups per step, e.g. 16 data cells x 4 prompts
    assert GRPOConfig(**narrow).groups_per_step == 8
    assert GRPOConfig(**wide).groups_per_step == 64
    assert 16 * prompts_per_data_cell(policy_batch_size=256, rollouts_per_prompt=4, data_replicas=16) == 64
    for kwargs, frac, quota in [(narrow, 1.0, 8), (narrow, 0.5, 4), (wide, 1.0, 64), (wide, 0.5, 32)]:
        cfg = GRPOConfig(**kwargs, min_informative_groups_frac=frac)
        assert cfg.effective_min_informative_groups == quota
        assert cfg.min_informative_groups == 0 and cfg.min_informative_groups_frac == frac  # stored as authored
    # Over the 16-cell wide layout the resolved quota splits exactly: 64 groups -> 4 groups x 4 rollouts per cell,
    # and half of it -> 2 groups x 4 rollouts per cell.
    for frac, per_cell in [(1.0, 16), (0.5, 8)]:
        quota = GRPOConfig(**wide, min_informative_groups_frac=frac).effective_min_informative_groups
        quotas = informative_quota_per_cell(min_informative_groups=quota, rollouts_per_prompt=4, data_replicas=16)
        assert quotas == (per_cell,) * 16
    # Ceil over the fraction read as a rational: a share naming a whole number of groups is exactly that number --
    # 0.14 of b200g4's 50 groups is 7, where the float product overshoots and a naive ceil would give 8; an odd share
    # rounds up; any positive fraction is at least one group, so the fraction is always ON.
    assert 0.14 * 50 > 7  # the float trap the rational read avoids
    b200g4 = GRPOConfig(batch_per_replica=200, group_size=4, min_informative_groups_frac=0.14)
    assert b200g4.groups_per_step == 50 and b200g4.effective_min_informative_groups == 7
    b40g4 = GRPOConfig(batch_per_replica=40, group_size=4, min_informative_groups_frac=0.7)
    assert b40g4.groups_per_step == 10 and b40g4.effective_min_informative_groups == 7
    assert GRPOConfig(**narrow, min_informative_groups_frac=0.3).effective_min_informative_groups == 3  # ceil(2.4)
    assert GRPOConfig(**wide, min_informative_groups_frac=0.001).effective_min_informative_groups == 1
    # Below the rational read's resolution (denominator <= 1e6) the share rounds to 0; the quota is floored at one
    # group, so a set fraction never resolves to OFF while the config says ON.
    assert GRPOConfig(**wide, min_informative_groups_frac=1e-9).effective_min_informative_groups == 1
    # The literal path is unchanged: the count is the quota and 0 is OFF.
    assert GRPOConfig(**narrow, min_informative_groups=8).effective_min_informative_groups == 8
    default = GRPOConfig(**narrow)
    assert default.effective_min_informative_groups == 0 and default.min_informative_groups_frac is None
    # The structured round trip carries the fraction (author -> YAML -> reload with schema -> instantiate), and a
    # dotlist integer coerces to the float share.
    authored = OmegaConf.structured(GRPOConfig(**wide, min_informative_groups_frac=0.5))
    obj = instantiate(OmegaConf.merge(OmegaConf.structured(GRPOConfig), OmegaConf.create(OmegaConf.to_yaml(authored))))
    assert obj.min_informative_groups_frac == 0.5 and obj.effective_min_informative_groups == 32
    dotted = OmegaConf.merge(
        OmegaConf.structured(GRPOConfig),
        OmegaConf.from_dotlist(["batch_per_replica=256", "group_size=4", "min_informative_groups_frac=1"]),
    )
    assert instantiate(dotted).effective_min_informative_groups == 64


@pytest.mark.L0
@pytest.mark.CPU
def test_ref_model_dir_lives_on_the_grpo_reference_config_only(tmp_path) -> None:
    """`model_dir` has one consumer, the GRPO entry, so it lives on `GRPORefConfig`, not on the `RefConfig` contract
    DPO and NFT share: the shared dataclass has no such field, and a dotted `rl.ref.model_dir=...` override on a
    DPO or NFT config is refused by the structured schema as an unknown key instead of being silently ignored.
    GRPO accepts it at config load (adopting a plain `RefConfig` with `model_dir=None`); the directory's existence
    is `build_trainer`'s check."""
    import dataclasses

    from omegaconf.errors import OmegaConfBaseException

    from cosmos_framework.posttrain.config.rl import GRPORefConfig, RefConfig

    assert "model_dir" not in {f.name for f in dataclasses.fields(RefConfig)}
    with pytest.raises(TypeError, match="model_dir"):
        RefConfig(build_ref=True, model_dir=str(tmp_path))  # type: ignore[call-arg]
    assert not hasattr(DPOConfig().ref, "model_dir") and not hasattr(NFTConfig().ref, "model_dir")
    # The structured node every DPO/NFT recipe stores at `rl.ref` refuses the key (the NFT root schema
    # itself cannot be built by OmegaConf.structured because of an unrelated Literal annotation, so the
    # shared node is exercised directly and the DPO root through its schema).
    with pytest.raises(OmegaConfBaseException, match="model_dir"):
        OmegaConf.merge(OmegaConf.structured(RefConfig), OmegaConf.from_dotlist([f"model_dir={tmp_path}"]))
    with pytest.raises(OmegaConfBaseException, match="model_dir"):
        OmegaConf.merge(OmegaConf.structured(DPOConfig), OmegaConf.from_dotlist([f"ref.model_dir={tmp_path}"]))
    # GRPO: its own subclass carries the knob, and a plain shared RefConfig is adopted.
    assert isinstance(GRPOConfig().ref, GRPORefConfig) and GRPOConfig().ref.model_dir is None
    adopted = GRPOConfig(kl_beta=0.01, kl_coeff_clamp=5.0, ref=RefConfig(build_ref=True))
    assert isinstance(adopted.ref, GRPORefConfig) and adopted.ref.build_ref and adopted.ref.model_dir is None
    missing = str(tmp_path / "not-materialized-on-this-host")
    warm = GRPOConfig(kl_beta=0.01, kl_coeff_clamp=5.0, ref=GRPORefConfig(build_ref=True, model_dir=missing))
    assert warm.ref.model_dir == missing, "the dataclass never probes the filesystem; build_trainer does, on every rank"
    dotted = OmegaConf.merge(
        OmegaConf.structured(GRPOConfig),
        OmegaConf.from_dotlist(
            ["kl_beta=0.01", "kl_coeff_clamp=5.0", "ref.build_ref=true", f"ref.model_dir={missing}"]
        ),
    )
    assert instantiate(dotted).ref.model_dir == missing, "the GRPO override key is unchanged: rl.ref.model_dir"


@pytest.mark.L0
@pytest.mark.CPU
def test_min_informative_groups_frac_rejections() -> None:
    """The fraction is a real in (0, 1]; it and a literal count > 0 cannot both be set; OPD refuses it like the count."""
    wide = {"batch_per_replica": 256, "group_size": 4}
    rejected = [
        (GRPOConfig, {**wide, "min_informative_groups_frac": True}, TypeError, "min_informative_groups_frac"),
        (GRPOConfig, {**wide, "min_informative_groups_frac": "1.0"}, TypeError, "min_informative_groups_frac"),
        (GRPOConfig, {**wide, "min_informative_groups_frac": 0.0}, ValueError, "0 < frac <= 1"),
        (GRPOConfig, {**wide, "min_informative_groups_frac": -0.5}, ValueError, "0 < frac <= 1"),
        (GRPOConfig, {**wide, "min_informative_groups_frac": 1.5}, ValueError, "0 < frac <= 1"),
        (GRPOConfig, {**wide, "min_informative_groups_frac": float("nan")}, ValueError, "0 < frac <= 1"),
        (
            GRPOConfig,
            {**wide, "min_informative_groups": 8, "min_informative_groups_frac": 1.0},
            ValueError,
            "set exactly one",
        ),
        (OPDConfig, {"min_informative_groups_frac": 1.0}, ValueError, "does not consume min_informative_groups"),
    ]
    for config_type, kwargs, error, match in rejected:
        with pytest.raises(error, match=match):
            config_type(**kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_min_informative_groups_below_half_of_groups_per_step_warns() -> None:
    """A literal count below half of groups_per_step warns and names groups_per_step, the count and the fraction key:
    8 is every group of b32g4 but 1/8 of b256g4's 64. Every group, exactly half, an explicit fraction and OFF are silent;
    a refused OPD config does not also warn."""
    with pytest.warns(
        UserWarning, match=r"min_informative_groups=8 is below half of groups_per_step=64 .*min_informative_groups_frac"
    ):
        GRPOConfig(batch_per_replica=256, group_size=4, min_informative_groups=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GRPOConfig(batch_per_replica=32, group_size=4, min_informative_groups=8)  # full DAPO on the narrow layout
        GRPOConfig(batch_per_replica=32, group_size=4, min_informative_groups=4)  # exactly half
        GRPOConfig(batch_per_replica=256, group_size=4, min_informative_groups=32)  # exactly half, wide
        GRPOConfig(batch_per_replica=256, group_size=4, min_informative_groups_frac=0.125)  # 8 of 64 by intent
        GRPOConfig(batch_per_replica=256, group_size=4)  # OFF
        with pytest.raises(ValueError, match="does not consume min_informative_groups"):
            OPDConfig(batch_per_replica=256, group_size=4, min_informative_groups=8)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_backend_refuses_autocast_mode() -> None:
    """A native builder delivers the compute dtype only through its own FSDP cast, so
    `rl.mixed_precision_mode="autocast"` would be silently ignored there; refuse it instead."""
    from types import SimpleNamespace

    native = SimpleNamespace(dtype="bfloat16", fsdp_reduce_dtype="float32")
    build._validate_native_policy_precision(GRPOConfig(), native)
    with pytest.raises(ValueError, match="mode="):
        build._validate_native_policy_precision(GRPOConfig(mixed_precision_mode="autocast"), native)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_backend_refuses_autocast_mode_without_any_dtype_field() -> None:
    """The mode check asks whether the BUILDER has an autocast path, which no native builder has; it must not hide
    behind the dtype comparisons, so a native config exposing neither `dtype` nor `fsdp_reduce_dtype` is still
    refused (`docs/reference/config.md` promises the refusal unconditionally for `model_backend="i4"`)."""
    from types import SimpleNamespace

    bare = SimpleNamespace()
    build._validate_native_policy_precision(GRPOConfig(), bare)
    with pytest.raises(ValueError, match="mode="):
        build._validate_native_policy_precision(GRPOConfig(mixed_precision_mode="autocast"), bare)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_backend_preflight_refuses_fp16_autocast_before_placement() -> None:
    """`autocast + float16` is refused by `resolve_precision`; the HF preflight has to resolve once on the driver
    so the refusal lands before Ray placement instead of inside the trainer worker. Both routes to fp16 (an explicit
    `rl.param_dtype` and the inherited `model_hf.dtype`) must be visible there; mode `fsdp` with the same dtypes
    passes, and so does the autocast recipe with a bf16 compute dtype."""
    rollout = RolloutConfig(temperature=0.9)
    hf_cfg = HFModelConfig(name_or_path=rollout.name_or_path, dtype="bfloat16")
    fp16_cfg = HFModelConfig(name_or_path=rollout.name_or_path, dtype="float16")
    refused = [
        (GRPOConfig(temperature=0.9, mixed_precision_mode="autocast", param_dtype="float16"), hf_cfg),
        (GRPOConfig(temperature=0.9, mixed_precision_mode="autocast"), fp16_cfg),
    ]
    for rl, model_config in refused:
        with pytest.raises(ValueError, match="GradScaler"):
            build.validate_online_policy_backend(rl, rollout, "hf", model_config)

    build.validate_online_policy_backend(
        GRPOConfig(temperature=0.9, mixed_precision_mode="fsdp", param_dtype="float16"), rollout, "hf", hf_cfg
    )
    build.validate_online_policy_backend(
        GRPOConfig(temperature=0.9, mixed_precision_mode="autocast", param_dtype="bfloat16"), rollout, "hf", fp16_cfg
    )


_FIXED = "seq-mean-token-sum-fixed"
_WIDE_LITERAL_QUOTA = dict(batch_per_replica=256, group_size=4, min_informative_groups=8)  # 8 of b256g4's 64 groups


def _construct_recording_warnings(**kwargs: object) -> tuple[GRPOConfig, list[str]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cfg = GRPOConfig(**kwargs)  # type: ignore[arg-type]
    return cfg, [str(w.message) for w in record if issubclass(w.category, UserWarning)]


@pytest.mark.L0
@pytest.mark.CPU
def test_recipe_guard_severity_warn_keeps_the_two_warnings_and_error_refuses_every_hazard_in_one_error() -> None:
    """`recipe_guard_severity` defaults to "warn": a b256g4 line with the literal quota 8 and eps=1e-6 under the
    fixed reduction still CONSTRUCTS with exactly the two warnings it always had, their text unchanged, and
    resolves the same quota and eps. Under "error" every hazard found is refused in ONE ValueError — both texts,
    in warning order, joined by a blank line, then the one sentence naming the knob — without also warning; a
    single hazard's error STARTS with that hazard's text unchanged. The thresholds do not move (exactly half,
    the fraction spelling, OFF, token-mean at 1e-6 and the ceiling stay silent under "error")."""
    assert GRPOConfig().recipe_guard_severity == "warn"
    assert RECIPE_GUARD_SEVERITIES == ("warn", "error")
    cfg, hazards = _construct_recording_warnings(
        **_WIDE_LITERAL_QUOTA, loss_type=_FIXED, optimizer=OptimizerConfig(eps=1e-6)
    )
    assert len(hazards) == 2
    quota_text, eps_text = hazards
    assert quota_text.startswith("min_informative_groups=8 is below half of groups_per_step=64 ")
    assert "rl.min_informative_groups_frac" in quota_text and quota_text.endswith("min_informative_groups at 0.")
    assert eps_text.startswith("loss_type='seq-mean-token-sum-fixed' with rl.optimizer.eps=1e-06: ")
    assert "eps-dominated" in eps_text and eps_text.endswith("once eps stops dominating.")
    assert (cfg.effective_min_informative_groups, cfg.groups_per_step, cfg.optimizer.eps) == (8, 64, 1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a refused config must not also warn
        with pytest.raises(ValueError) as both:
            GRPOConfig(
                **_WIDE_LITERAL_QUOTA,
                loss_type=_FIXED,
                optimizer=OptimizerConfig(eps=1e-6),
                recipe_guard_severity="error",
            )
        with pytest.raises(ValueError) as quota_only:
            GRPOConfig(
                **_WIDE_LITERAL_QUOTA,
                loss_type=_FIXED,
                optimizer=OptimizerConfig(eps=1e-8),
                recipe_guard_severity="error",
            )
        with pytest.raises(ValueError) as eps_only:
            GRPOConfig(loss_type=_FIXED, optimizer=OptimizerConfig(eps=1e-6), recipe_guard_severity="error")
        with pytest.raises(ValueError, match="rl.optimizer.eps=2e-07"):
            GRPOConfig(loss_type=_FIXED, optimizer=OptimizerConfig(eps=2e-7), recipe_guard_severity="error")
        for batch, literal in [(64, 7), (32, 3)]:  # below half on other layouts
            with pytest.raises(ValueError, match=f"min_informative_groups={literal} is below half"):
                GRPOConfig(
                    batch_per_replica=batch, group_size=4, min_informative_groups=literal, recipe_guard_severity="error"
                )
        # Inert where nothing ever warned: exactly half, the fraction spelling, OFF, token-mean at 1e-6,
        # the fixed reduction at the ceiling or below, and the OptimizerConfig default.
        GRPOConfig(batch_per_replica=64, group_size=4, min_informative_groups=8, recipe_guard_severity="error")
        GRPOConfig(batch_per_replica=256, group_size=4, min_informative_groups=32, recipe_guard_severity="error")
        GRPOConfig(
            batch_per_replica=256, group_size=4, min_informative_groups_frac=0.125, recipe_guard_severity="error"
        )
        GRPOConfig(batch_per_replica=256, group_size=4, recipe_guard_severity="error")
        GRPOConfig(loss_type="token-mean", optimizer=OptimizerConfig(eps=1e-6), recipe_guard_severity="error")
        GRPOConfig(
            loss_type=_FIXED,
            optimizer=OptimizerConfig(eps=FIXED_NORMALISER_OPTIMIZER_EPS_CEILING),
            recipe_guard_severity="error",
        )
        GRPOConfig(loss_type=_FIXED, recipe_guard_severity="error")
    assert RECIPE_GUARD_REFUSAL_SENTENCE == (
        "Refused because rl.recipe_guard_severity=error; set it to warn to run this pairing with a warning instead."
    )
    # ONE error, both hazards in warning order, one closing sentence.
    assert str(both.value) == f"{quota_text}\n\n{eps_text}\n\n{RECIPE_GUARD_REFUSAL_SENTENCE}"
    assert str(quota_only.value) == f"{quota_text}\n\n{RECIPE_GUARD_REFUSAL_SENTENCE}"
    assert str(eps_only.value) == f"{eps_text}\n\n{RECIPE_GUARD_REFUSAL_SENTENCE}"
    assert str(quota_only.value).startswith(quota_text) and str(eps_only.value).startswith(eps_text)


@pytest.mark.L0
@pytest.mark.CPU
def test_recipe_guard_severity_rejects_unknown_values_and_reaches_the_validator_through_the_dotlist() -> None:
    """Only "warn" and "error" are accepted (a bool or a third word is refused, like `cursor_resume`), and the
    dotlist spelling a launcher uses (`rl.recipe_guard_severity=error`) reaches the validator through the
    OmegaConf structured round-trip — the path `--validate-manifest` takes — refusing both hazards in one error,
    while the plain round-trip keeps the default and still warns twice, as main does."""
    with pytest.raises(TypeError, match="recipe_guard_severity must be str"):
        GRPOConfig(recipe_guard_severity=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recipe_guard_severity must be 'warn' or 'error', got 'strict'"):
        GRPOConfig(recipe_guard_severity="strict")

    cfg, hazards = _construct_recording_warnings(
        **_WIDE_LITERAL_QUOTA, loss_type=_FIXED, optimizer=OptimizerConfig(eps=1e-6)
    )
    quota_text, eps_text = hazards
    schema = OmegaConf.structured(cfg)  # structuring itself does not re-validate
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="recipe_guard_severity=error") as via_dotlist:
            OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.from_dotlist(["recipe_guard_severity=error"])))
        with pytest.raises(ValueError, match="got 'loud'"):
            OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.from_dotlist(["recipe_guard_severity=loud"])))
    assert quota_text in str(via_dotlist.value) and eps_text in str(via_dotlist.value)
    with warnings.catch_warnings(record=True) as round_trip:
        warnings.simplefilter("always")
        assert OmegaConf.to_object(schema).recipe_guard_severity == "warn"  # the round-trip keeps the default
    assert [str(w.message)[:40] for w in round_trip if issubclass(w.category, UserWarning)] == [
        quota_text[:40],
        eps_text[:40],
    ]  # and still warns, twice, as main does


@pytest.mark.L0
@pytest.mark.CPU
def test_require_full_emergency_plan_refuses_run_for_a_loop_that_always_writes_the_full_plan() -> None:
    """Review of c2af017642d (thread 0069d0e0): the shared selector must fail loud wherever a loop cannot honour it.
    `require_full_emergency_plan` is the one check the PPO validators call; "full" (and an absent field) pass."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.rl import require_full_emergency_plan

    require_full_emergency_plan(SimpleNamespace(emergency_checkpoint_components="full"), algorithm="PPO", reason="x")
    require_full_emergency_plan(SimpleNamespace(), algorithm="PPO", reason="x")
    with pytest.raises(
        ValueError, match="joint PPO ignores rl.emergency_checkpoint_components='run': always full; set 'full'"
    ):
        require_full_emergency_plan(
            SimpleNamespace(emergency_checkpoint_components="run"), algorithm="joint PPO", reason="always full"
        )
