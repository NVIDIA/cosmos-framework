# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed reward-provider outputs with an explicit optimization/diagnostics boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType

from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY
from cosmos_framework.posttrain.schema.metrics import MetricContribution, MetricReduction, WeightedMean

_SAMPLE_COUNT_DIAGNOSTIC = "sample_count"
_SAFE_LOCAL_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class RewardProviderResult:
    """The sole return contract for every :class:`RewardProvider`.

    ``terms`` is the only channel consumed by :class:`RewardCombiner`. ``token_terms`` carries
    immutable per-completion-token signals without changing scalar combination. ``diagnostics`` is
    a tensor-free reporting channel and is deliberately absent from ``Transition.reward``.
    """

    terms: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, MetricContribution] = field(default_factory=dict)
    token_terms: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.terms, Mapping):
            raise RewardContractError(f"reward provider terms must be a mapping, got {type(self.terms).__name__}")
        normalized: dict[str, float] = {}
        for name, value in self.terms.items():
            if not isinstance(name, str) or not name:
                raise RewardContractError(f"reward term names must be non-empty strings, got {name!r}")
            if name == FINAL_REWARD_KEY:
                raise RewardContractError(
                    f"reward providers cannot emit reserved term {FINAL_REWARD_KEY!r}; RewardCombiner owns it"
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RewardContractError(f"reward term {name!r} must be numeric and non-bool, got {value!r}")
            try:
                result = float(value)
            except OverflowError as error:
                raise RewardContractError(f"reward term {name!r} returned non-finite value {value!r}") from error
            if not isfinite(result):
                raise RewardContractError(f"reward term {name!r} returned non-finite value {result!r}")
            normalized[name] = result
        if not isinstance(self.token_terms, Mapping):
            raise RewardContractError(
                f"reward provider token terms must be a mapping, got {type(self.token_terms).__name__}"
            )
        normalized_token_terms: dict[str, tuple[float, ...]] = {}
        for name, values in self.token_terms.items():
            if not isinstance(name, str) or not name:
                raise RewardContractError(f"reward token term names must be non-empty strings, got {name!r}")
            if name == FINAL_REWARD_KEY:
                raise RewardContractError(
                    f"reward providers cannot emit reserved term {FINAL_REWARD_KEY!r}; RewardCombiner owns it"
                )
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise RewardContractError(f"reward token term {name!r} must be a sequence, got {values!r}")
            normalized_values: list[float] = []
            for index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise RewardContractError(
                        f"reward token term {name!r}[{index}] must be numeric and non-bool, got {value!r}"
                    )
                try:
                    result = float(value)
                except OverflowError as error:
                    raise RewardContractError(
                        f"reward token term {name!r}[{index}] returned non-finite value {value!r}"
                    ) from error
                if not isfinite(result):
                    raise RewardContractError(
                        f"reward token term {name!r}[{index}] returned non-finite value {result!r}"
                    )
                normalized_values.append(result)
            if not normalized_values:
                raise RewardContractError(f"reward token term {name!r} must not be empty")
            normalized_token_terms[name] = tuple(normalized_values)
        collisions = sorted(set(normalized).intersection(normalized_token_terms))
        if collisions:
            raise RewardContractError(
                f"reward providers cannot emit names as both scalar and token terms: {collisions}"
            )
        if not normalized and not normalized_token_terms:
            raise RewardContractError("reward providers must emit at least one optimized term")
        if not isinstance(self.diagnostics, Mapping):
            raise RewardContractError(
                f"reward provider diagnostics must be a mapping, got {type(self.diagnostics).__name__}"
            )
        diagnostics: dict[str, MetricContribution] = {}
        for name, contribution in self.diagnostics.items():
            if not isinstance(name, str) or _SAFE_LOCAL_NAME.fullmatch(name) is None:
                raise RewardContractError(f"reward diagnostic names must match '[a-z][a-z0-9_]*', got {name!r}")
            if name == _SAMPLE_COUNT_DIAGNOSTIC:
                raise RewardContractError(
                    f"reward diagnostic name {_SAMPLE_COUNT_DIAGNOSTIC!r} is reserved for the framework"
                )
            if not isinstance(contribution, MetricContribution):
                raise RewardContractError(
                    f"reward diagnostic {name!r} must be a MetricContribution, got {contribution!r}"
                )
            if contribution.reduction is MetricReduction.MOMENTS:
                raise RewardContractError(
                    f"reward diagnostic {name!r} cannot use MOMENTS without a fixed output projection"
                )
            if contribution.reduction is MetricReduction.IDENTICAL and (
                isinstance(contribution.payload, bool) or not isinstance(contribution.payload, Real)
            ):
                raise RewardContractError(
                    f"reward diagnostic {name!r} IDENTICAL payload must be a non-bool number, "
                    f"got {contribution.payload!r}"
                )
            if contribution.reduction is MetricReduction.WEIGHTED_MEAN and (
                not isinstance(contribution.payload, WeightedMean) or contribution.payload.denominator <= 0.0
            ):
                raise RewardContractError(
                    f"reward diagnostic {name!r} weighted mean must carry a positive observation weight"
                )
            diagnostics[name] = contribution
        object.__setattr__(self, "terms", MappingProxyType(normalized))
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        object.__setattr__(self, "token_terms", MappingProxyType(normalized_token_terms))
