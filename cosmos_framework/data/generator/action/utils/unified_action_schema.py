# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical tensor contract for heterogeneous Action datasets.

``unified_v1`` is a 59D raw action layout::

    [ego_pose9d,
     right_wrist_pose9d, right_fingertips_xyz_15d, right_openness,
     left_wrist_pose9d,  left_fingertips_xyz_15d,  left_openness]

Missing semantics are zero-filled and excluded by ``action_valid_mask``. The
schema represents fingertip positions in the corresponding wrist frame and
openness as a unitless fraction (0=closed, 1=open).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from cosmos_framework.data.generator.action.utils.action_processing import ActionNormalizer

ActionSchema = Literal["native", "unified_v1"]
UNIFIED_ACTION_SCHEMA = "unified_v1"
UNIFIED_ACTION_DIM = 59

EGO_POSE = slice(0, 9)
RIGHT_WRIST_POSE = slice(9, 18)
RIGHT_FINGERTIPS = slice(18, 33)
RIGHT_OPENNESS = 33
LEFT_WRIST_POSE = slice(34, 43)
LEFT_FINGERTIPS = slice(43, 58)
LEFT_OPENNESS = 58

UNIFIED_ACTION_SLOT_GROUPS: tuple[tuple[str, slice], ...] = (
    ("ego_pose", EGO_POSE),
    ("right_wrist_pose", RIGHT_WRIST_POSE),
    ("right_fingertips", RIGHT_FINGERTIPS),
    ("right_openness", slice(RIGHT_OPENNESS, RIGHT_OPENNESS + 1)),
    ("left_wrist_pose", LEFT_WRIST_POSE),
    ("left_fingertips", LEFT_FINGERTIPS),
    ("left_openness", slice(LEFT_OPENNESS, LEFT_OPENNESS + 1)),
)


@dataclass(frozen=True)
class ActionComponentSource:
    """Map one contiguous native action block into a canonical block."""

    source: slice
    target: slice


@dataclass(frozen=True)
class OpennessSource:
    """Map one native scalar into canonical 0=closed, 1=open semantics."""

    source_index: int
    target_index: int
    closed_value: float
    open_value: float
    clamp: bool = False


@dataclass(frozen=True)
class UnifiedActionMapping:
    """Declarative native-to-``unified_v1`` mapping for one dataset."""

    native_dim: int
    components: tuple[ActionComponentSource, ...]
    openness: tuple[OpennessSource, ...] = ()
    allow_unmapped_source: bool = False

    def __post_init__(self) -> None:
        if self.native_dim <= 0:
            raise ValueError(f"native_dim must be positive, got {self.native_dim}")
        source_occupied = torch.zeros(self.native_dim, dtype=torch.bool)
        target_occupied = torch.zeros(UNIFIED_ACTION_DIM, dtype=torch.bool)
        for component in self.components:
            source_width = _slice_width(component.source, self.native_dim, "source")
            target_width = _slice_width(component.target, UNIFIED_ACTION_DIM, "target")
            if source_width != target_width:
                raise ValueError(f"component source/target widths differ: {source_width} != {target_width}")
            if bool(source_occupied[component.source].any()):
                raise ValueError(f"overlapping native source slice: {component.source}")
            if bool(target_occupied[component.target].any()):
                raise ValueError(f"overlapping canonical target slice: {component.target}")
            source_occupied[component.source] = True
            target_occupied[component.target] = True
        for source in self.openness:
            if not 0 <= source.source_index < self.native_dim:
                raise ValueError(f"openness source_index {source.source_index} outside native_dim={self.native_dim}")
            if source.target_index not in {RIGHT_OPENNESS, LEFT_OPENNESS}:
                raise ValueError(f"invalid openness target_index={source.target_index}")
            if source.open_value == source.closed_value:
                raise ValueError("open_value and closed_value must differ")
            if source_occupied[source.source_index]:
                raise ValueError(f"overlapping native source index: {source.source_index}")
            if target_occupied[source.target_index]:
                raise ValueError(f"overlapping canonical target index: {source.target_index}")
            source_occupied[source.source_index] = True
            target_occupied[source.target_index] = True
        if not self.allow_unmapped_source and not bool(source_occupied.all()):
            missing = torch.where(~source_occupied)[0].tolist()
            raise ValueError(f"unmapped native source indices: {missing}")

    def valid_mask(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return the static canonical slot-validity mask, shape ``[59]``."""
        mask = torch.zeros(UNIFIED_ACTION_DIM, dtype=torch.bool, device=device)
        for component in self.components:
            mask[component.target] = True
        for source in self.openness:
            mask[source.target_index] = True
        return mask

    def to_unified(self, native_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter a native ``[...,D]`` action into ``[...,59]`` plus ``[59]`` mask."""
        self._validate_native_action(native_action)
        unified = native_action.new_zeros(*native_action.shape[:-1], UNIFIED_ACTION_DIM)
        for component in self.components:
            unified[..., component.target] = native_action[..., component.source]
        for source in self.openness:
            value = (native_action[..., source.source_index] - source.closed_value) / (
                source.open_value - source.closed_value
            )
            if source.clamp:
                value = value.clamp(0.0, 1.0)
            unified[..., source.target_index] = value
        return unified, self.valid_mask(device=native_action.device)

    def from_unified(self, unified_action: torch.Tensor) -> torch.Tensor:
        """Project canonical raw actions back to the native action layout."""
        self._validate_unified_action(unified_action, label="unified_action")
        native = unified_action.new_zeros(*unified_action.shape[:-1], self.native_dim)
        for component in self.components:
            native[..., component.source] = unified_action[..., component.target]
        for source in self.openness:
            openness = unified_action[..., source.target_index]
            native[..., source.source_index] = (
                openness * (source.open_value - source.closed_value) + source.closed_value
            )
        return native

    def scatter_normalized(self, native_normalized: torch.Tensor) -> torch.Tensor:
        """Scatter normalized native values without applying raw-space openness calibration."""
        self._validate_native_action(native_normalized, label="native_normalized")
        unified_normalized = native_normalized.new_zeros(  # [...,59]
            *native_normalized.shape[:-1], UNIFIED_ACTION_DIM
        )
        for component in self.components:
            unified_normalized[..., component.target] = native_normalized[..., component.source]  # [...,59]
        for source in self.openness:
            unified_normalized[..., source.target_index] = native_normalized[  # [...,59]
                ..., source.source_index
            ]
        return unified_normalized  # [...,59]

    def gather_normalized(self, unified_normalized: torch.Tensor) -> torch.Tensor:
        """Gather normalized native values without applying raw-space openness calibration."""
        self._validate_unified_action(unified_normalized, label="unified_normalized")
        native_normalized = unified_normalized.new_zeros(  # [...,D_native]
            *unified_normalized.shape[:-1], self.native_dim
        )
        for component in self.components:
            native_normalized[..., component.source] = unified_normalized[  # [...,D_native]
                ..., component.target
            ]
        for source in self.openness:
            native_normalized[..., source.source_index] = unified_normalized[  # [...,D_native]
                ..., source.target_index
            ]
        return native_normalized  # [...,D_native]

    def _validate_native_action(self, native_action: torch.Tensor, *, label: str = "native_action") -> None:
        if not isinstance(native_action, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(native_action).__name__}")
        if native_action.ndim < 1 or native_action.shape[-1] != self.native_dim:
            raise ValueError(f"{label} must end in {self.native_dim}, got shape={tuple(native_action.shape)}")
        if not bool(torch.isfinite(native_action).all()):
            raise ValueError(f"{label} contains NaN or Inf")

    @staticmethod
    def _validate_unified_action(unified_action: torch.Tensor, *, label: str) -> None:
        if not isinstance(unified_action, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(unified_action).__name__}")
        if unified_action.ndim < 1 or unified_action.shape[-1] != UNIFIED_ACTION_DIM:
            raise ValueError(f"{label} must end in {UNIFIED_ACTION_DIM}, got shape={tuple(unified_action.shape)}")
        if not bool(torch.isfinite(unified_action).all()):
            raise ValueError(f"{label} contains NaN or Inf")


@dataclass(frozen=True)
class EncodedAction:
    """Canonical raw and normalized representations produced from one native action tensor."""

    raw: torch.Tensor  # [...,59]
    normalized: torch.Tensor  # [...,59]
    valid_mask: torch.Tensor  # [59]


@dataclass(frozen=True)
class UnifiedActionCodec:
    """Encode native actions and decode canonical normalized values explicitly.

    A codec accepts either a native-layout normalizer or a canonical-layout
    normalizer. Native normalization happens before scattering, while
    canonical normalization happens after mapping into ``unified_v1``.
    """

    native_normalizer: ActionNormalizer | None
    mapping: UnifiedActionMapping
    unified_normalizer: ActionNormalizer | None = None

    def __post_init__(self) -> None:
        if self.native_normalizer is not None and self.unified_normalizer is not None:
            raise ValueError("UnifiedActionCodec accepts either a native or unified normalizer, not both")

    def encode(
        self, native_raw: torch.Tensor
    ) -> EncodedAction:  # native_raw: [...,D_native], returns raw/normalized [...,59] and mask [59]
        """Encode native raw values into separate canonical raw and normalized tensors."""
        unified_raw, valid_mask = self.mapping.to_unified(native_raw)  # [...,59], [59]
        if self.unified_normalizer is not None:
            unified_normalized = self.unified_normalizer.normalize_action(unified_raw)  # [...,59]
            unified_normalized = torch.where(  # [...,59]
                valid_mask, unified_normalized, torch.zeros_like(unified_normalized)
            )
        elif self.native_normalizer is None:
            unified_normalized = unified_raw  # [...,59]
        else:
            native_normalized = self.native_normalizer.normalize_action(native_raw)  # [...,D_native]
            unified_normalized = self.mapping.scatter_normalized(native_normalized)  # [...,59]
        return EncodedAction(raw=unified_raw, normalized=unified_normalized, valid_mask=valid_mask)

    def decode_to_unified(
        self, unified_normalized: torch.Tensor
    ) -> torch.Tensor:  # unified_normalized: [...,59], returns [...,59]
        """Decode canonical normalized values into canonical raw values."""
        if self.unified_normalizer is not None:
            unified_raw = self.unified_normalizer.denormalize_action(unified_normalized)  # [...,59]
            valid_mask = self.mapping.valid_mask(device=unified_raw.device)  # [59]
            return torch.where(valid_mask, unified_raw, torch.zeros_like(unified_raw))  # [...,59]
        if self.native_normalizer is None:
            self.mapping._validate_unified_action(unified_normalized, label="unified_normalized")
            return unified_normalized  # [...,59]
        native_normalized = self.mapping.gather_normalized(unified_normalized)  # [...,D_native]
        native_raw = self.native_normalizer.denormalize_action(native_normalized)  # [...,D_native]
        unified_raw, _ = self.mapping.to_unified(native_raw)  # [...,59], [59]
        return unified_raw  # [...,59]


def get_dataset_action_mapping(dataset: object) -> UnifiedActionMapping:
    """Return and validate a dataset's declarative ``unified_v1`` mapping."""
    factory = getattr(dataset, "get_unified_action_mapping", None)
    if factory is None or not callable(factory):
        raise ValueError(f"Dataset {type(dataset).__name__} does not declare a unified_v1 action mapping")
    mapping = factory()
    if not isinstance(mapping, UnifiedActionMapping):
        raise TypeError(
            f"{type(dataset).__name__}.get_unified_action_mapping() must return UnifiedActionMapping, "
            f"got {type(mapping).__name__}"
        )
    return mapping


def ego_pose_mapping() -> UnifiedActionMapping:
    """Map an existing 9D ego-pose action into the canonical ego slots."""
    return UnifiedActionMapping(
        native_dim=9,
        components=(ActionComponentSource(slice(0, 9), EGO_POSE),),
    )


def single_arm_pose_mapping(
    *,
    native_dim: int = 10,
    side: Literal["right", "left"] = "right",
    openness_index: int | None = 9,
    closed_value: float = 0.0,
    open_value: float = 1.0,
) -> UnifiedActionMapping:
    """Build a 9D wrist-pose plus optional scalar-openness mapping."""
    pose_target = RIGHT_WRIST_POSE if side == "right" else LEFT_WRIST_POSE
    openness_target = RIGHT_OPENNESS if side == "right" else LEFT_OPENNESS
    openness = ()
    if openness_index is not None:
        openness = (OpennessSource(openness_index, openness_target, closed_value, open_value),)
    return UnifiedActionMapping(
        native_dim=native_dim,
        components=(ActionComponentSource(slice(0, 9), pose_target),),
        openness=openness,
    )


def dual_arm_pose_mapping(
    *,
    order: Literal["left_right", "right_left"] = "left_right",
    left_closed_value: float = 0.0,
    left_open_value: float = 1.0,
    right_closed_value: float = 0.0,
    right_open_value: float = 1.0,
) -> UnifiedActionMapping:
    """Build a mapping for two concatenated 10D pose+openness arm blocks."""
    if order == "left_right":
        left_start, right_start = 0, 10
    else:
        right_start, left_start = 0, 10
    return UnifiedActionMapping(
        native_dim=20,
        components=(
            ActionComponentSource(slice(right_start, right_start + 9), RIGHT_WRIST_POSE),
            ActionComponentSource(slice(left_start, left_start + 9), LEFT_WRIST_POSE),
        ),
        openness=(
            OpennessSource(right_start + 9, RIGHT_OPENNESS, right_closed_value, right_open_value),
            OpennessSource(left_start + 9, LEFT_OPENNESS, left_closed_value, left_open_value),
        ),
    )


def humanoid_pose_mapping() -> UnifiedActionMapping:
    """Map a 29D humanoid ``[ego9,right9,right_open,left9,left_open]`` layout."""
    return UnifiedActionMapping(
        native_dim=29,
        components=(
            ActionComponentSource(slice(0, 9), EGO_POSE),
            ActionComponentSource(slice(9, 18), RIGHT_WRIST_POSE),
            ActionComponentSource(slice(19, 28), LEFT_WRIST_POSE),
        ),
        openness=(
            OpennessSource(18, RIGHT_OPENNESS, 0.0, 1.0),
            OpennessSource(28, LEFT_OPENNESS, 0.0, 1.0),
        ),
    )


def head_dual_arm_pose_mapping(
    *,
    left_open_value: float = 1.0,
    right_open_value: float = 1.0,
) -> UnifiedActionMapping:
    """Map a 29D ``[head9, left9, left_open, right9, right_open]`` layout.

    Distinct from :func:`humanoid_pose_mapping`, which is also 29D but ordered
    ``[ego9, right9, right_open, left9, left_open]``. ManipArena mobile puts the
    left arm first, following its own 20D contract, so the block offsets differ
    and the two mappings cannot be shared.
    """
    return UnifiedActionMapping(
        native_dim=29,
        components=(
            ActionComponentSource(slice(0, 9), EGO_POSE),
            ActionComponentSource(slice(9, 18), LEFT_WRIST_POSE),
            ActionComponentSource(slice(19, 28), RIGHT_WRIST_POSE),
        ),
        openness=(
            OpennessSource(18, LEFT_OPENNESS, 0.0, left_open_value),
            OpennessSource(28, RIGHT_OPENNESS, 0.0, right_open_value),
        ),
    )


def hand_pose_57d_mapping(*, include_ego: bool = True) -> UnifiedActionMapping:
    """Map the existing 57D ego/wrist/fingertip layout into 59D."""
    components = (
        ActionComponentSource(slice(0, 9), EGO_POSE),
        ActionComponentSource(slice(9, 18), RIGHT_WRIST_POSE),
        ActionComponentSource(slice(18, 33), RIGHT_FINGERTIPS),
        ActionComponentSource(slice(33, 42), LEFT_WRIST_POSE),
        ActionComponentSource(slice(42, 57), LEFT_FINGERTIPS),
    )
    return UnifiedActionMapping(
        native_dim=57,
        components=components if include_ego else components[1:],
        allow_unmapped_source=not include_ego,
    )


def _slice_width(value: slice, upper_bound: int, label: str) -> int:
    start, stop, step = value.indices(upper_bound)
    if step != 1:
        raise ValueError(f"{label} slice step must be 1, got {value}")
    if start >= stop:
        raise ValueError(f"{label} slice must be non-empty, got {value}")
    if (value.start is not None and value.start < 0) or (value.stop is not None and value.stop > upper_bound):
        raise ValueError(f"{label} slice {value} outside width={upper_bound}")
    return stop - start
