# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest

from cosmos_framework.model.generator.tokenizers.lidar.range_projection import (
    V0_TRANSFER_RANGE_PROJECTION,
    V1_TRANSFER_RANGE_PROJECTION,
    LidarRangeProjectionConfig,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_every_projection_shares_the_drive_dreams_metric_span() -> None:
    """The span fixes the [-1, 1] affine, so a stray default silently breaks metric decode."""
    assert (LidarRangeProjectionConfig().min_range_m, LidarRangeProjectionConfig().max_range_m) == (5.0, 100.0)
    for projection in (V0_TRANSFER_RANGE_PROJECTION, V1_TRANSFER_RANGE_PROJECTION):
        assert (projection.min_range_m, projection.max_range_m) == (5.0, 100.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_range_projection_round_trip() -> None:
    projection = LidarRangeProjectionConfig(
        semantic_width=1800,
        model_width=1808,
        max_range_m=105.0,
        validity_threshold=0.6,
    )

    restored = LidarRangeProjectionConfig.from_dict(projection.to_dict())

    assert restored == projection
    assert restored.column_pool_factor == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_range_projection_rejects_inconsistent_circular_layout() -> None:
    with pytest.raises(ValueError, match="must be even"):
        LidarRangeProjectionConfig(semantic_width=1800, model_width=1801)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("validity_threshold", [0.0, 1.0])
def test_range_projection_rejects_degenerate_validity_threshold(validity_threshold: float) -> None:
    with pytest.raises(ValueError, match="validity_threshold"):
        LidarRangeProjectionConfig(validity_threshold=validity_threshold)


@pytest.mark.L0
@pytest.mark.CPU
def test_range_projection_records_legacy_resize() -> None:
    projection = LidarRangeProjectionConfig(
        semantic_width=1200,
        model_width=1024,
        model_width_transform="resize",
        min_range_m=5.0,
        max_range_m=100.0,
    )

    assert projection.model_width == 1024
    assert projection.column_pool_factor == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_range_projection_rejects_resize_from_native_width() -> None:
    with pytest.raises(ValueError, match="nearest-return pooling"):
        LidarRangeProjectionConfig(
            model_width=1024,
            model_width_transform="resize",
            min_range_m=5.0,
            max_range_m=100.0,
        )
