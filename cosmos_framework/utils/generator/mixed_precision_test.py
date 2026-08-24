# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator.mixed_precision import use_w8a16_step


def test_config_defaults_disable_mixed_precision() -> None:
    config = QuantizationConfig()
    assert config.mixed_precision_first_steps == 0
    assert config.mixed_precision_last_steps == 0
    assert config.mixed_precision_reasoner_policy == "high_precision"
    assert config.mixed_precision_w8a16_cache == "gpu_block"
    assert not config.mixed_precision_enabled


def test_config_enabled_when_any_width_positive() -> None:
    assert QuantizationConfig(mixed_precision_first_steps=1).mixed_precision_enabled
    assert QuantizationConfig(mixed_precision_last_steps=2).mixed_precision_enabled


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_first_steps=-1)
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_reasoner_policy="fast")
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_w8a16_cache="huge")


def test_schedule_selects_first_and_last_steps() -> None:
    selected = [i for i in range(10) if use_w8a16_step(2, 4, i, 10)]
    assert selected == [0, 1, 6, 7, 8, 9]


def test_schedule_overlap_selects_every_step() -> None:
    assert all(use_w8a16_step(3, 3, i, 4) for i in range(4))


def test_schedule_single_step_is_base_precision() -> None:
    # FSDP alignment pads slow ranks with dummy num_steps=1 samples; keep them cheap.
    assert not use_w8a16_step(3, 3, 0, 1)


def test_schedule_validates_bounds() -> None:
    with pytest.raises(ValueError):
        use_w8a16_step(1, 1, 0, 0)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, 5, 5)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, -1, 5)
