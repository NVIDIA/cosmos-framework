# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Current temporal-window configs restored from checkpoint dictionaries."""

from typing import Any

import attrs
import pytest
from omegaconf import OmegaConf

from cosmos_framework.configs.base.defaults.multiview_attention import (
    CrossViewTemporalWindowConfig,
    MultiviewAttentionMaskConfig,
    SameViewTemporalWindowConfig,
    upgrade_saved_multiview_attention,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _restore_saved_mask(saved_mask: dict[str, Any]) -> Cosmos3VFMNetworkConfig:
    # Exported checkpoint JSON need not carry the attrs object metadata.
    config = OmegaConf.create({"backend": "maskless", "mask": saved_mask})
    return Cosmos3VFMNetworkConfig(multiview_attention_config=config)


def test_nested_temporal_windows_survive_checkpoint_roundtrip() -> None:
    mask = MultiviewAttentionMaskConfig(
        cross_view_temporal_window=CrossViewTemporalWindowConfig(bounds=(-0.25, 0.125), includes_first_frame=True),
        same_view_temporal_window=SameViewTemporalWindowConfig(
            sensor_to_sensor_bounds=(-0.5, 0.5),
            sensor_to_control_bounds=(-0.25, 0.0),
            control_to_control_bounds=(-0.125, 0.0),
            control_to_sensor_bounds=(0.0, 0.0),
        ),
    )
    saved_mask = attrs.asdict(mask)
    restored = _restore_saved_mask(saved_mask).multiview_attention_config.mask

    assert OmegaConf.to_container(restored) == OmegaConf.to_container(OmegaConf.create(saved_mask))


@pytest.mark.parametrize(
    ("window_config", "field"),
    [
        ("cross_view_temporal_window", "bounds"),
        ("same_view_temporal_window", "sensor_to_sensor_bounds"),
    ],
)
def test_saved_temporal_window_rejects_invalid_bounds(window_config: str, field: str) -> None:
    saved_mask = attrs.asdict(MultiviewAttentionMaskConfig())
    saved_mask[window_config][field] = [0.25, -0.25]

    with pytest.raises(ValueError, match="start <= end"):
        _restore_saved_mask(saved_mask)


def _saved_before_temporal_window_configs(**mask_overrides: Any) -> Any:
    """``model.config.multiview_attention`` as exported before the temporal-window configs."""
    mask = {
        "attention_scope": "all_views",
        "control_attends_sensor": False,
        "control_to_control_window": None,
        "control_to_sensor_window": None,
        "decomposed_temporal_window_includes_first_frame": False,
        "decomposed_temporal_window_seconds": None,
        "lidar_attends_captions": True,
        "radar_attends_captions": True,
        "sensor_to_control_window": None,
        "sensor_to_sensor_window": None,
    }
    return OmegaConf.create({"backend": "auto", "deduplicate_cross_view": False, "mask": mask | mask_overrides})


@pytest.mark.parametrize(("saved_window", "bounds"), [(None, None), (0.4, [-0.4, 0.0]), ([-0.2, 0.2], [-0.2, 0.2])])
def test_mask_saved_before_temporal_window_configs_is_upgraded(saved_window: Any, bounds: Any) -> None:
    config = _saved_before_temporal_window_configs(decomposed_temporal_window_seconds=saved_window)
    upgrade_saved_multiview_attention(config)
    upgrade_saved_multiview_attention(config)  # Idempotent: the model and its network both upgrade.

    expected_mask = attrs.asdict(MultiviewAttentionMaskConfig()) | {
        "cross_view_temporal_window": {"bounds": bounds, "includes_first_frame": False}
    }
    assert OmegaConf.to_container(config.mask) == OmegaConf.to_container(OmegaConf.create(expected_mask))
    assert "deduplicate_cross_view" not in config
    assert Cosmos3VFMNetworkConfig(multiview_attention_config=config).multiview_attention_config is config


@pytest.mark.parametrize(
    ("mask_overrides", "message"),
    [
        ({"sensor_to_control_window": [-2, 0]}, "frame-count windows"),
        ({"decomposed_temporal_window_seconds": -1}, "non-negative"),
    ],
)
def test_saved_config_that_cannot_be_upgraded_is_left_unchanged(mask_overrides: dict[str, Any], message: str) -> None:
    config = _saved_before_temporal_window_configs(**mask_overrides)
    saved = OmegaConf.to_container(config)

    for _ in range(2):  # A retry must fail the same way rather than find the windows deleted.
        with pytest.raises(ValueError, match=message):
            upgrade_saved_multiview_attention(config)
        assert OmegaConf.to_container(config) == saved
