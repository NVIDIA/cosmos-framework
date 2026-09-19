# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared definition of the LiDAR tokenizer's physical range-view contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import attrs


@attrs.frozen
class LidarRangeProjectionConfig:
    """Settings that determine the contents and interpretation of a range view.

    This object is deliberately independent of Hydra, model construction, and
    storage. Training uses it directly and inference records the resolved object
    so evaluation can reconstruct the exact deterministic projection contract.
    """

    # Geometry currently uses the Pandar128 beam-elevation calibration. Keep
    # the sensor identity explicit so incompatible 128-row data fails early.
    sensor: str = "pandar128"
    native_height: int = 128
    native_width: int = 3600
    semantic_height: int = 128
    semantic_width: int = 3600
    model_width: int = 3600
    model_width_transform: str = "circular_pad"
    azimuth_start_degrees: float = 180.0
    azimuth_end_degrees: float = -180.0
    azimuth_endpoint: bool = False
    return_selection: str = "nearest"
    # Drive-Dreams metric span shared by every LiDAR tokenizer checkpoint. The
    # span fixes the [-1, 1] affine, so it must match the span a checkpoint was
    # trained with or metric decode is silently wrong.
    min_range_m: float = 5.0
    max_range_m: float = 100.0
    intensity_encoding: str = "unit"
    invalid_range_m: float = 0.0
    validity_threshold: float = 0.5
    coordinate_system: str = "x_forward_y_left_z_up"

    def __attrs_post_init__(self) -> None:
        if self.sensor != "pandar128":
            raise ValueError(f"Unsupported LiDAR sensor {self.sensor!r}")
        if self.native_height < 1 or self.native_width < 1:
            raise ValueError(
                f"Native range-view dimensions must be positive, got {(self.native_height, self.native_width)}"
            )
        if self.semantic_height != self.native_height:
            raise ValueError(
                "Vertical resampling is not supported; "
                f"semantic_height must equal native_height, got {self.semantic_height} != {self.native_height}"
            )
        if self.semantic_width < 1 or self.native_width % self.semantic_width:
            raise ValueError(
                f"semantic_width must be a positive divisor of native_width {self.native_width}, "
                f"got {self.semantic_width}"
            )
        if self.model_width_transform == "circular_pad":
            if self.model_width < self.semantic_width:
                raise ValueError(
                    f"model_width must be at least semantic_width, got {self.model_width} < {self.semantic_width}"
                )
            if (self.model_width - self.semantic_width) % 2:
                raise ValueError(
                    "model_width - semantic_width must be even for symmetric circular padding, "
                    f"got {self.model_width} - {self.semantic_width}"
                )
        elif self.model_width_transform == "resize":
            if self.model_width < 1:
                raise ValueError(f"model_width must be positive, got {self.model_width}")
            if self.semantic_width == self.native_width and self.model_width != self.semantic_width:
                raise ValueError(
                    "Resizing the native azimuth canvas skips nearest-return pooling. "
                    "Set semantic_width to a divisor of native_width first "
                    f"(V0 transfer uses semantic_width=1200, model_width=1024); "
                    f"got semantic_width={self.semantic_width} and model_width={self.model_width}"
                )
        else:
            raise ValueError(f"Unsupported model width transform {self.model_width_transform!r}")
        if self.return_selection != "nearest":
            raise ValueError(f"Unsupported return selection {self.return_selection!r}")
        if self.max_range_m <= self.min_range_m:
            raise ValueError(f"max_range_m must exceed min_range_m, got {self.min_range_m=} and {self.max_range_m=}")
        if self.intensity_encoding != "unit":
            raise ValueError(f"Unsupported intensity encoding {self.intensity_encoding!r}")
        if self.invalid_range_m != 0.0:
            raise ValueError(f"Only zero invalid range is supported, got {self.invalid_range_m}")
        if not 0.0 < self.validity_threshold < 1.0:
            raise ValueError(f"validity_threshold must lie in (0,1), got {self.validity_threshold}")
        if self.coordinate_system != "x_forward_y_left_z_up":
            raise ValueError(f"Unsupported coordinate system {self.coordinate_system!r}")

    @property
    def column_pool_factor(self) -> int:
        """Number of adjacent native columns considered for each semantic ray."""
        return self.native_width // self.semantic_width

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved projection for a generated artifact manifest."""
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LidarRangeProjectionConfig:
        """Reconstruct and validate a serialized projection configuration."""
        return cls(**dict(value))


# Azimuth canvases used by transfer recipes. All inherit the class metric span;
# they differ only in how many columns the tokenizer's network consumes.
V1_TRANSFER_RANGE_PROJECTION = LidarRangeProjectionConfig(
    semantic_width=1800,
    model_width=1800,
)
V0_TRANSFER_RANGE_PROJECTION = LidarRangeProjectionConfig(
    semantic_width=1200,
    model_width=1024,
    model_width_transform="resize",
)
# V1.2 keeps V1.1's 1800 semantic rays and widens the canvas to the next
# multiple of its 16x compression. Padding rather than resizing is what keeps
# the 1800 rays on their native angular grid, and wrapping the azimuth means the
# eight added columns hold real neighbors instead of edge replicas.
V1P2_TRANSFER_RANGE_PROJECTION = LidarRangeProjectionConfig(
    semantic_width=1800,
    model_width=1808,
)


def transfer_range_projection(*, tokenizer_v1: bool) -> LidarRangeProjectionConfig:
    """Range-view contract for a transfer recipe's LiDAR tokenizer version.

    The boolean reaches only the two canvases that also differ in channel
    layout. V1.2 shares V1's three channels and differs from it in canvas width
    alone, so a recipe on that tokenizer passes
    ``V1P2_TRANSFER_RANGE_PROJECTION`` explicitly and still counts as V1 here.
    """
    return V1_TRANSFER_RANGE_PROJECTION if tokenizer_v1 else V0_TRANSFER_RANGE_PROJECTION
