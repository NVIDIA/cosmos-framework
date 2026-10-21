# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU coverage for same-view temporal bounds expressed in seconds."""

import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import SameViewTemporalWindowConfig
from cosmos_framework.model.generator.mot import multiview_maskless_attention


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("period", "radius"), [(0.1, 4), (0.2, 2), (0.15, 2)])
def test_same_view_seconds_resolve_each_direction_using_the_latent_frame_period(period: float, radius: int) -> None:
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [1, 1],
        [(6, 1, 1), (6, 1, 1)],
        device=torch.device("cpu"),
        items_per_sample=[2],
        is_control=[True, False],
        control_attends_sensor=True,
        seconds_per_frame=[period, period],
        same_view_temporal_window=SameViewTemporalWindowConfig(
            sensor_to_sensor_bounds=(-0.4, 0.4),
            sensor_to_control_bounds=(-0.4, 0.0),
            control_to_control_bounds=(-0.05, 0.05),
            control_to_sensor_bounds=(-0.05, 0.0),
        ),
    )

    assert plan.sensor_to_sensor_window == (-radius, radius)
    assert plan.sensor_to_control_window == (-radius, 0)
    assert plan.control_to_control_window == (0, 0)
    assert plan.control_to_sensor_window == (0, 0)
    assert multiview_maskless_attention._natten_window(plan.control_to_control_window, (6, 1, 1))[0] == (1, 1, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_seconds_require_an_explicit_latent_clock() -> None:
    with pytest.raises(ValueError, match="same_view_temporal_window requires explicit seconds_per_frame"):
        multiview_maskless_attention.build_multiview_maskless_plan(
            [1],
            [(6, 1, 1)],
            device=torch.device("cpu"),
            same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-0.4, 0.4)),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_seconds_reject_incompatible_frame_windows_across_items() -> None:
    with pytest.raises(ValueError, match="must resolve to one common frame window"):
        multiview_maskless_attention.build_multiview_maskless_plan(
            [1, 1],
            [(6, 1, 1), (6, 1, 1)],
            device=torch.device("cpu"),
            seconds_per_frame=[0.1, 0.2],
            same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-0.4, 0.4)),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_seconds_allow_different_clocks_with_the_same_frame_window() -> None:
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [1, 1],
        [(6, 1, 1), (6, 1, 1)],
        device=torch.device("cpu"),
        seconds_per_frame=[0.1, 0.11],
        same_view_temporal_window=SameViewTemporalWindowConfig(sensor_to_sensor_bounds=(-0.25, 0.25)),
    )
    assert plan.sensor_to_sensor_window == (-2, 2)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("sensor_queries_control", [True, False])
def test_same_view_cross_role_seconds_reject_different_clocks(sensor_queries_control: bool) -> None:
    with pytest.raises(ValueError, match="require matching seconds_per_frame"):
        multiview_maskless_attention.build_multiview_maskless_plan(
            [1, 1],
            [(16, 1, 1), (16, 1, 1)],
            device=torch.device("cpu"),
            items_per_sample=[2],
            is_control=[True, False],
            control_attends_sensor=not sensor_queries_control,
            seconds_per_frame=[0.11, 0.1],
            same_view_temporal_window=SameViewTemporalWindowConfig(
                sensor_to_control_bounds=(-0.15, 0.15) if sensor_queries_control else None,
                control_to_sensor_bounds=None if sensor_queries_control else (-0.15, 0.15),
            ),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_cross_role_seconds_allow_different_clocks_between_view_groups() -> None:
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [1, 1, 1, 1],
        [(16, 1, 1)] * 4,
        device=torch.device("cpu"),
        items_per_sample=[4],
        is_control=[True, False, True, False],
        control_attends_sensor=True,
        view_axis=[0, 0, 1, 1],
        seconds_per_frame=[0.1, 0.1, 0.11, 0.11],
        same_view_temporal_window=SameViewTemporalWindowConfig(
            sensor_to_control_bounds=(-0.15, 0.15),
            control_to_sensor_bounds=(-0.15, 0.15),
        ),
    )
    assert plan.sensor_to_control_window == (-1, 1)
    assert plan.control_to_sensor_window == (-1, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_self_role_seconds_allow_different_sensor_control_clocks() -> None:
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [1, 1],
        [(16, 1, 1), (16, 1, 1)],
        device=torch.device("cpu"),
        items_per_sample=[2],
        is_control=[True, False],
        control_attends_sensor=True,
        seconds_per_frame=[0.11, 0.1],
        same_view_temporal_window=SameViewTemporalWindowConfig(
            sensor_to_sensor_bounds=(-0.15, 0.15),
            control_to_control_bounds=(-0.15, 0.15),
        ),
    )
    assert plan.sensor_to_sensor_window == (-1, 1)
    assert plan.control_to_control_window == (-1, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_same_view_disabled_control_to_sensor_seconds_allow_different_clocks() -> None:
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [1, 1],
        [(16, 1, 1), (16, 1, 1)],
        device=torch.device("cpu"),
        items_per_sample=[2],
        is_control=[True, False],
        control_attends_sensor=False,
        seconds_per_frame=[0.11, 0.1],
        same_view_temporal_window=SameViewTemporalWindowConfig(control_to_sensor_bounds=(-0.15, 0.15)),
    )
    assert not plan.control_attends_sensor
    assert plan.control_to_sensor_window == (-1, 1)
