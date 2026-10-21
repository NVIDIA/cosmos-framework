# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-aware custom DataPacker factory selection."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.model import DataPackerConfig

DEFAULT_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_default_packer"
TEACHER_MODEL_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_teacher_model_packer"
NON_PACKER_FACTORY = "cosmos_framework.posttrain.tests.helpers.custom_data.build_non_packer"


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_builds_fresh_equivalent_instances_and_setup_stays_centralized() -> None:
    """Reusing one factory must not mean sharing mutable packer objects across actors."""
    from cosmos_framework.posttrain.data.packer import init_data_packer_from_factory

    rollout = init_data_packer_from_factory(
        "model/student",
        DEFAULT_FACTORY,
        trust_remote_code=True,
    )
    policy = init_data_packer_from_factory("model/student", DEFAULT_FACTORY)

    assert rollout is not policy
    assert rollout.name == policy.name == "default"
    assert rollout.setup_name_or_path == policy.setup_name_or_path == "model/student"
    assert rollout.setup_trust_remote_code is True
    assert policy.setup_trust_remote_code is False


@pytest.mark.L0
@pytest.mark.CPU
def test_data_packer_config_keeps_student_and_teacher_model_factories_separate() -> None:
    """A distinct teacher model may require a different tokenizer and processor contract."""
    config = DataPackerConfig(
        factory=DEFAULT_FACTORY,
        teacher_model_factory=TEACHER_MODEL_FACTORY,
    )

    assert config.factory == DEFAULT_FACTORY
    assert config.teacher_model_factory == TEACHER_MODEL_FACTORY


@pytest.mark.L0
@pytest.mark.CPU
def test_data_packer_config_rejects_non_string_teacher_model_factory() -> None:
    """A malformed teacher factory must fail before distributed actor setup."""
    with pytest.raises(TypeError, match="teacher_model_factory must be a string"):
        DataPackerConfig(teacher_model_factory=[])  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_resolver_rejects_result_that_is_not_a_data_packer() -> None:
    """A wrong factory result must fail at construction rather than on the first rollout."""
    from cosmos_framework.posttrain.data.packer import init_data_packer_from_factory

    with pytest.raises(TypeError, match="must return a DataPacker"):
        init_data_packer_from_factory("model", NON_PACKER_FACTORY)


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_importability_validation_reports_the_configured_path() -> None:
    """An import typo must fail in driver preflight instead of after actor placement."""
    from cosmos_framework.posttrain.data.packer import validate_data_packer_factories

    factory = "no.such.module.build_packer"
    with pytest.raises(ValueError, match="no.such.module.build_packer"):
        validate_data_packer_factories(DataPackerConfig(factory=factory))


@pytest.mark.L0
@pytest.mark.CPU
def test_factory_importability_validation_checks_teacher_model_factory() -> None:
    """A teacher-model import typo must fail before the teacher actor is placed."""
    from cosmos_framework.posttrain.data.packer import validate_data_packer_factories

    factory = "no.such.module.build_teacher_packer"
    with pytest.raises(ValueError, match="no.such.module.build_teacher_packer"):
        validate_data_packer_factories(DataPackerConfig(teacher_model_factory=factory))


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_and_opd_roots_expose_the_same_data_packer_config() -> None:
    """Putting the config on only one root would make the shared OPD entry bypass customization."""
    from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
    from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig

    required = {
        "model": None,
        "optimizer": None,
        "scheduler": None,
        "dataloader_train": None,
        "dataloader_val": None,
    }
    assert isinstance(PosttrainGRPOConfig(**required).data_packer, DataPackerConfig)
    assert isinstance(PosttrainOPDConfig(**required).data_packer, DataPackerConfig)
