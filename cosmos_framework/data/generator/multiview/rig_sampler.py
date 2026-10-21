# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Camera samplers for Cosmos3 multiview AV training.

``RandomCameraSampler`` preserves the existing behavior: sample a camera count,
then choose an arbitrary subset of that size. ``ExplicitRigSampler`` samples a
configured catalog of camera lists. ``RigSampler`` samples the 32 structural
MADS rigs built from a front-wide anchor and five independently enabled units.
"""

import math
import random
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import product
from typing import Protocol, TypeAlias, TypeVar

from cosmos_framework.data.generator.multiview.schema import MADS_CAMERAS

CameraCountSelection: TypeAlias = int | Sequence[int] | Mapping[int, float] | None
ExplicitRigKey: TypeAlias = tuple[str, ...] | frozenset[str]
ExplicitRigConfig: TypeAlias = Sequence[Collection[str]] | Mapping[ExplicitRigKey, float]
RandomSeed: TypeAlias = int | float | str | bytes | bytearray | None
_Choice = TypeVar("_Choice")

UNITS: dict[str, tuple[str, ...]] = {
    "front_tele": ("camera_front_tele_30fov",),
    "rear_tele": ("camera_rear_tele_30fov",),
    "rear_pair": ("camera_rear_left_70fov", "camera_rear_right_70fov"),
    "cross_pair": ("camera_cross_left_120fov", "camera_cross_right_120fov"),
    "fisheye_mod": (
        "camera_front_fisheye_200fov",
        "camera_rear_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
    ),
}
ANCHOR: tuple[str, ...] = ("camera_front_wide_120fov",)


class CameraSampler(Protocol):
    """Interface for selecting an ordered camera subset per sample."""

    camera_names: tuple[str, ...]

    def sample(self) -> tuple[str, ...]:
        """Sample one camera-membership tuple; consumers restore canonical order."""


def _validate_camera_count(count: int, *, max_cameras: int) -> None:
    if isinstance(count, bool) or not 1 <= count <= max_cameras:
        raise ValueError(f"camera count must be in [1, {max_cameras}], got {count}")


def validate_camera_count_selection(selection: CameraCountSelection, *, max_cameras: int) -> None:
    """Validate a fixed, uniform, or weighted camera-count selection."""
    if selection is None:
        return
    if isinstance(selection, int):
        _validate_camera_count(selection, max_cameras=max_cameras)
        return
    if isinstance(selection, Mapping):
        if not selection:
            raise ValueError("num_cameras_to_keep mapping must not be empty")
        total_weight = 0.0
        for count, weight in selection.items():
            _validate_camera_count(count, max_cameras=max_cameras)
            if weight < 0:
                raise ValueError(f"num_cameras_to_keep weight must be non-negative, got {weight} for {count}")
            total_weight += float(weight)
        if total_weight <= 0:
            raise ValueError("num_cameras_to_keep mapping must have positive total weight")
        return
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        if not selection:
            raise ValueError("num_cameras_to_keep sequence must not be empty")
        for count in selection:
            _validate_camera_count(count, max_cameras=max_cameras)
        return
    raise TypeError(
        "num_cameras_to_keep must be None, an int, a sequence of ints, or a mapping from int counts to weights"
    )


class RandomCameraSampler:
    """Sample a camera count, then an arbitrary subset in canonical order.

    Optionally require one camera to appear in every sampled subset.
    """

    def __init__(
        self,
        camera_names: Sequence[str],
        num_cameras_to_keep: CameraCountSelection,
        required_camera_name: str | None = None,
        seed: RandomSeed = None,
    ) -> None:
        self.camera_names: tuple[str, ...] = tuple(camera_names)
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        validate_camera_count_selection(num_cameras_to_keep, max_cameras=len(self.camera_names))
        self.num_cameras_to_keep: CameraCountSelection = num_cameras_to_keep
        if required_camera_name is not None and required_camera_name not in self.camera_names:
            raise ValueError(f"required camera {required_camera_name!r} is not present in camera_names")
        self.required_camera_name: str | None = required_camera_name
        # With no explicit seed, use Python's process-global RNG. PyTorch seeds
        # it independently in each dataloader worker.
        self.rng: random.Random | None = random.Random(seed) if seed is not None else None

    def _sample_camera_count(self) -> int | None:
        selection = self.num_cameras_to_keep
        if selection is None:
            return None
        if isinstance(selection, int):
            return selection
        if isinstance(selection, Mapping):
            counts = list(selection)
            weights = [float(selection[count]) for count in counts]
            if self.rng is None:
                return random.choices(counts, weights=weights, k=1)[0]
            return self.rng.choices(counts, weights=weights, k=1)[0]
        if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
            choices = list(selection)
            if self.rng is None:
                return random.choice(choices)
            return self.rng.choice(choices)
        raise TypeError(
            "num_cameras_to_keep must be None, an int, a sequence of ints, or a mapping from int counts to weights"
        )

    def sample(self) -> tuple[str, ...]:
        """Sample a subset while preserving the configured camera order."""
        camera_count = self._sample_camera_count()
        if camera_count is None or camera_count == len(self.camera_names):
            return self.camera_names
        required_camera_names = {self.required_camera_name} if self.required_camera_name is not None else set()
        secondary_camera_names = [camera for camera in self.camera_names if camera not in required_camera_names]
        secondary_camera_count = camera_count - len(required_camera_names)
        if self.rng is None:
            selected_camera_names = required_camera_names | set(
                random.sample(secondary_camera_names, secondary_camera_count)
            )
        else:
            selected_camera_names = required_camera_names | set(
                self.rng.sample(secondary_camera_names, secondary_camera_count)
            )
        return tuple(camera for camera in self.camera_names if camera in selected_camera_names)


def _normalize_explicit_rig(rig: Collection[str], *, label: str) -> frozenset[str]:
    if isinstance(rig, (str, bytes)):
        raise TypeError(f"{label} must be a collection of camera names, not a string")
    configured_cameras = tuple(rig)
    if not configured_cameras:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(camera, str) or not camera for camera in configured_cameras):
        raise ValueError(f"{label} camera names must be non-empty strings")
    normalized_rig = frozenset(configured_cameras)
    if len(normalized_rig) != len(configured_cameras):
        raise ValueError(f"{label} contains duplicate cameras")
    return normalized_rig


class ExplicitRigSampler:
    """Sample from an explicit camera-rig catalog.

    Args:
        rigs: Either a sequence of camera-name collections, sampled uniformly unless weights are supplied,
            or a mapping from hashable camera-name tuples or frozensets to
            non-negative sampling weights. Camera order within a rig has no
            semantic meaning. Lists remain supported because Hydra/OmegaConf
            configs do not reliably represent sets. Weights need not be
            normalized, but their total must be positive.
        seed: Optional deterministic seed. When omitted, use Python's global
            RNG so each PyTorch dataloader worker follows its worker seed.
        weights: Optional weights aligned with a sequence of rigs, for configs
            that cannot represent tuple-keyed mappings. Omit for mapping rigs.
    """

    def __init__(
        self, rigs: ExplicitRigConfig, seed: RandomSeed = None, *, weights: Sequence[float] | None = None
    ) -> None:
        if not rigs:
            raise ValueError("rigs must not be empty")
        if isinstance(rigs, Mapping):
            if weights is not None:
                raise ValueError("weights must be omitted when rigs is a weighted mapping")
            configured_rigs = list(rigs)
            raw_weights: list[float] | None = list(rigs.values())
        elif isinstance(rigs, Sequence) and not isinstance(rigs, (str, bytes)):
            configured_rigs = list(rigs)
            raw_weights = list(weights) if weights is not None else None
        else:
            raise TypeError(
                "rigs must be a sequence of camera collections or a mapping from hashable camera collections to weights"
            )

        rig_memberships = tuple(
            _normalize_explicit_rig(rig, label=f"rigs[{index}]") for index, rig in enumerate(configured_rigs)
        )
        if len(set(rig_memberships)) != len(rig_memberships):
            raise ValueError("rigs contains duplicate rigs; use weights to control sampling probability")
        self.rigs: tuple[tuple[str, ...], ...] = tuple(tuple(sorted(rig)) for rig in rig_memberships)

        self.weights: tuple[float, ...] | None = None
        if raw_weights is not None:
            if len(raw_weights) != len(self.rigs):
                raise ValueError("weights must contain exactly one weight per rig")
            normalized_weights: list[float] = []
            for rig, raw_weight in zip(self.rigs, raw_weights):
                if isinstance(raw_weight, bool):
                    raise ValueError(f"weight for rig {rig} must be a finite number")
                weight = float(raw_weight)
                if not math.isfinite(weight):
                    raise ValueError(f"weight for rig {rig} must be finite")
                if weight < 0:
                    raise ValueError(f"weight for rig {rig} must be non-negative")
                normalized_weights.append(weight)
            if sum(normalized_weights) <= 0:
                raise ValueError("rigs must have positive total weight")
            self.weights = tuple(normalized_weights)

        self.camera_names: tuple[str, ...] = tuple(sorted({camera for rig in rig_memberships for camera in rig}))
        # With no explicit seed, use the worker-seeded process-global RNG.
        self.rng: random.Random | None = random.Random(seed) if seed is not None else None

    def sample(self) -> tuple[str, ...]:
        """Sample one configured rig according to its uniform or weighted policy."""
        if self.weights is None:
            if self.rng is None:
                return random.choice(self.rigs)
            return self.rng.choice(self.rigs)
        if self.rng is None:
            return random.choices(self.rigs, weights=self.weights, k=1)[0]
        return self.rng.choices(self.rigs, weights=self.weights, k=1)[0]

    def all_rigs(self) -> list[tuple[str, ...]]:
        """Return every configured rig in configuration order."""
        return list(self.rigs)


def buckets() -> dict[int, list[tuple[str, ...]]]:
    """Build camera-count buckets containing all 32 structural MADS rigs."""
    rigs_by_count: defaultdict[int, list[tuple[str, ...]]] = defaultdict(list)
    unit_names = list(UNITS)
    for bits in product((0, 1), repeat=len(unit_names)):
        enabled_units = {name for name, enabled in zip(unit_names, bits) if enabled}
        cameras = ANCHOR + tuple(
            camera for unit_name in unit_names if unit_name in enabled_units for camera in UNITS[unit_name]
        )
        rigs_by_count[len(cameras)].append(cameras)
    return dict(rigs_by_count)


class RigSampler:
    """Sample all 32 structural MADS rigs uniformly by count, then by rig.

    Args:
        seed: Optional deterministic seed. When omitted, use Python's global
            RNG so each PyTorch dataloader worker follows its worker seed.

    Every rig always contains front-wide. Rear, cross, and fisheye cameras are
    sampled as symmetric modules rather than independent cameras.
    """

    def __init__(self, seed: RandomSeed = None) -> None:
        self.camera_names: tuple[str, ...] = MADS_CAMERAS
        self.buckets: dict[int, list[tuple[str, ...]]] = buckets()
        self.ks: list[int] = sorted(self.buckets)
        # See RandomCameraSampler: the global RNG is worker-seeded, while an
        # explicit seed provides deterministic standalone sampling and tests.
        self.rng: random.Random | None = random.Random(seed) if seed is not None else None

    def _choice(self, values: Sequence[_Choice]) -> _Choice:
        if self.rng is None:
            return random.choice(values)
        return self.rng.choice(values)

    def sample(self) -> tuple[str, ...]:
        """Sample uniformly over camera count, then uniformly within its bucket."""
        camera_count = self._choice(self.ks)
        rig = self._choice(self.buckets[camera_count])
        return rig

    def sample_without_replacement(self, n: int) -> list[tuple[str, ...]]:
        """Round-robin over counts so short runs still span the size axis."""
        pools = {
            camera_count: (random.sample(rigs, len(rigs)) if self.rng is None else self.rng.sample(rigs, len(rigs)))
            for camera_count, rigs in self.buckets.items()
        }
        sampled_rigs: list[tuple[str, ...]] = []
        index = 0
        while len(sampled_rigs) < n and any(pools.values()):
            camera_count = self.ks[index % len(self.ks)]
            if pools[camera_count]:
                sampled_rigs.append(pools[camera_count].pop())
            index += 1
        return sampled_rigs

    def all_rigs(self) -> list[tuple[str, ...]]:
        """Return every allowed rig in ascending camera-count order."""
        return [rig for camera_count in self.ks for rig in self.buckets[camera_count]]

    def probability(self, camera_count: int) -> float:
        """Return the probability of each rig in one camera-count bucket."""
        return 1 / len(self.ks) / len(self.buckets[camera_count])

    def histogram(self) -> dict[int, int]:
        """Return the number of allowed rigs for each camera count."""
        return {camera_count: len(rigs) for camera_count, rigs in sorted(self.buckets.items())}
