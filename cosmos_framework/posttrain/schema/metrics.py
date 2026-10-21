# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-light metric contributions and exact merge/finalize helpers.

Metric reduction is part of the value's schema.  Reducers must never infer semantics from a key
suffix such as ``_count`` or ``_std``: a count is a :class:`Reduction.SUM` contribution, a global
mean carries its numerator and denominator, and a distribution carries mergeable Welford moments.

The types in this module intentionally depend only on the Python standard library.  They can cross
worker/process boundaries without importing torch, Ray, W&B, or an algorithm implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import TypeAlias, cast

MetricScalar: TypeAlias = int | float | str | bool
FinalMetricReport: TypeAlias = dict[str, MetricScalar]

PUBLIC_METRIC_NAMESPACES = frozenset({"train", "rollout", "val", "optim", "timer"})

# Integer `train/*` counts that aggregate by SUM over data-parallel ranks (`LLMTrainer._reduce_metrics`)
# and over the chunks of one collected batch (`utils/report.py::merge_training_metric_window`, called by
# the entry loop); every other numeric metric is a mean unless the objective's metric adapter declares it
# `max_reduced` (`algorithm/primitives/objective.py::MetricAdapter`), the one declaration point for maxima.
# `routing_replay_pinned_tokens` is emitted only under `rl.routing_replay=rollout` and
# `logprob_gap_token_count` only when rollout logprobs were recorded; absent keys are simply not summed.
# `response_count` is the completions the clipped GRPO objective was handed, the weight of its
# truncation and response-length ratios.
SUMMED_TRAIN_COUNT_KEYS: tuple[str, ...] = (
    "train/valid_sequence_count",
    "train/valid_token_count",
    "train/routing_replay_pinned_tokens",
    "train/logprob_gap_token_count",
    "train/response_count",
    "train/sc_token_count",
)
_LEGACY_METRIC_PREFIXES = ("grpo_", "dpo_", "GRPO/", "DPO/")
_LEGACY_METRIC_KEYS = frozenset({"optim/grad_norm", "optim/lr", "timer/step_s"})


class MetricReduction(str, Enum):
    """How independent contributions for one metric are combined."""

    SUM = "sum"
    MAX = "max"
    IDENTICAL = "identical"
    WEIGHTED_MEAN = "weighted_mean"
    MOMENTS = "moments"


# Short module-local spelling.  The package exports both, while ``MetricReduction`` avoids colliding
# with ``algorithm.primitives.objective.Reduction`` in trainer code that needs both contracts.
Reduction = MetricReduction


def _finite_real(value: object, *, field: str) -> int | float:
    """Normalize one finite non-bool real while preserving integer counts."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, got {value!r}")
    normalized: int | float = int(value) if isinstance(value, Integral) else float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field} must be finite, got {normalized!r}")
    return normalized


def _primitive(value: object, *, field: str) -> MetricScalar:
    """Normalize a small, serializable control-plane scalar."""
    if isinstance(value, (bool, str)):
        return value
    try:
        return _finite_real(value, field=field)
    except TypeError as error:
        raise TypeError(f"{field} must be an int, float, bool, or str, got {value!r}") from error


@dataclass(frozen=True)
class WeightedMean:
    """Mergeable numerator and denominator for an exact global weighted mean."""

    numerator: float
    denominator: float

    def __post_init__(self) -> None:
        numerator = float(_finite_real(self.numerator, field="weighted-mean numerator"))
        denominator = float(_finite_real(self.denominator, field="weighted-mean denominator"))
        if denominator < 0.0:
            raise ValueError(f"weighted-mean denominator must be >= 0, got {denominator}")
        if denominator == 0.0 and numerator != 0.0:
            raise ValueError("an empty weighted mean must have numerator == 0")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    @classmethod
    def from_value(cls, value: float, weight: float = 1.0) -> WeightedMean:
        """Build one contribution representing ``value`` observed with ``weight``."""
        normalized_value = float(_finite_real(value, field="weighted-mean value"))
        normalized_weight = float(_finite_real(weight, field="weighted-mean weight"))
        if normalized_weight < 0.0:
            raise ValueError(f"weighted-mean weight must be >= 0, got {normalized_weight}")
        return cls(numerator=normalized_value * normalized_weight, denominator=normalized_weight)

    def merge(self, other: WeightedMean) -> WeightedMean:
        """Merge two independently accumulated weighted means."""
        return WeightedMean(
            numerator=self.numerator + other.numerator,
            denominator=self.denominator + other.denominator,
        )

    @property
    def value(self) -> float:
        """Finalize the mean; a globally unobserved metric is an error, not zero."""
        if self.denominator == 0.0:
            raise ValueError("cannot finalize a weighted mean with no observations")
        return self.numerator / self.denominator


@dataclass(frozen=True)
class Moments:
    """Mergeable scalar Welford state with population-variance semantics.

    ``m2`` is the sum of squared deviations from ``mean``.  Empty physical shards are represented
    by ``count == 0`` and merge as the identity; an empty *global* distribution cannot be finalized.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, Integral):
            raise TypeError(f"moments count must be an integer, got {self.count!r}")
        count = int(self.count)
        if count < 0:
            raise ValueError(f"moments count must be >= 0, got {count}")
        mean = float(_finite_real(self.mean, field="moments mean"))
        m2 = float(_finite_real(self.m2, field="moments m2"))
        if m2 < 0.0:
            raise ValueError(f"moments m2 must be >= 0, got {m2}")
        if count == 0:
            if mean != 0.0 or m2 != 0.0 or self.minimum is not None or self.maximum is not None:
                raise ValueError("empty moments must use mean=m2=0 and minimum=maximum=None")
        else:
            if self.minimum is None or self.maximum is None:
                raise ValueError("non-empty moments require minimum and maximum")
            minimum = float(_finite_real(self.minimum, field="moments minimum"))
            maximum = float(_finite_real(self.maximum, field="moments maximum"))
            if minimum > maximum:
                raise ValueError(f"moments minimum {minimum} exceeds maximum {maximum}")
            if mean < minimum or mean > maximum:
                raise ValueError(f"moments mean {mean} is outside [{minimum}, {maximum}]")
            object.__setattr__(self, "minimum", minimum)
            object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "m2", m2)

    @classmethod
    def empty(cls) -> Moments:
        """Return the identity contribution for an empty physical shard."""
        return cls()

    @classmethod
    def from_values(cls, values: Iterable[float]) -> Moments:
        """Accumulate values with Welford's stable one-pass update."""
        count = 0
        mean = 0.0
        m2 = 0.0
        minimum: float | None = None
        maximum: float | None = None
        for raw_value in values:
            value = float(_finite_real(raw_value, field="moment observation"))
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)
        # Floating-point round-off can leave a tiny negative M2 for an otherwise constant stream.
        m2 = max(m2, 0.0)
        return cls(count=count, mean=mean, m2=m2, minimum=minimum, maximum=maximum)

    def merge(self, other: Moments) -> Moments:
        """Pairwise merge using Chan et al.'s parallel Welford formula."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / count
        self_minimum, self_maximum = self._require_bounds()
        other_minimum, other_maximum = other._require_bounds()
        return Moments(
            count=count,
            mean=mean,
            m2=max(m2, 0.0),
            minimum=min(self_minimum, other_minimum),
            maximum=max(self_maximum, other_maximum),
        )

    @classmethod
    def merge_many(cls, moments: Sequence[Moments]) -> Moments:
        """Merge a possibly empty inventory; empty input yields empty moments."""
        merged = cls()
        for item in moments:
            merged = merged.merge(item)
        return merged

    def _require_observations(self) -> None:
        if self.count == 0:
            raise ValueError("cannot finalize moments with no observations")

    def _require_bounds(self) -> tuple[float, float]:
        """Return validated extrema for a non-empty moment state."""
        self._require_observations()
        if self.minimum is None or self.maximum is None:
            raise ValueError("non-empty moments require minimum and maximum")
        return self.minimum, self.maximum

    @property
    def population_variance(self) -> float:
        """Population variance (``ddof=0``)."""
        self._require_observations()
        return self.m2 / self.count

    @property
    def population_std(self) -> float:
        """Population standard deviation (``ddof=0``)."""
        return sqrt(self.population_variance)


@dataclass(frozen=True)
class MomentOutputs:
    """Explicit public keys onto which one moment family is projected."""

    count: str | None = None
    mean: str | None = None
    std: str | None = None
    minimum: str | None = None
    maximum: str | None = None

    def __post_init__(self) -> None:
        keys = [key for key in (self.count, self.mean, self.std, self.minimum, self.maximum) if key is not None]
        if not keys:
            raise ValueError("moment outputs must name at least one public metric")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"moment output keys must be non-empty strings, got {keys!r}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"moment output keys must be unique, got {keys!r}")

    def finalize(self, moments: Moments) -> FinalMetricReport:
        """Project non-empty moments into explicitly named scalar metrics."""
        minimum, maximum = moments._require_bounds()
        out: FinalMetricReport = {}
        if self.count is not None:
            out[self.count] = moments.count
        if self.mean is not None:
            out[self.mean] = moments.mean
        if self.std is not None:
            out[self.std] = moments.population_std
        if self.minimum is not None:
            out[self.minimum] = minimum
        if self.maximum is not None:
            out[self.maximum] = maximum
        return out


MetricPayload: TypeAlias = MetricScalar | WeightedMean | Moments


def _normalize_payload(reduction: object, payload: object) -> MetricPayload:
    """Validate a contribution payload, including objects mutated after construction."""
    if not isinstance(reduction, MetricReduction):
        raise TypeError(f"metric reduction must be a MetricReduction, got {reduction!r}")
    if reduction is Reduction.WEIGHTED_MEAN:
        if not isinstance(payload, WeightedMean):
            raise TypeError("WEIGHTED_MEAN requires a WeightedMean payload")
        return payload
    if reduction is Reduction.MOMENTS:
        if not isinstance(payload, Moments):
            raise TypeError("MOMENTS requires a Moments payload")
        return payload
    if isinstance(payload, (WeightedMean, Moments)):
        raise TypeError(f"{reduction.value} requires a scalar payload")
    if reduction in (Reduction.SUM, Reduction.MAX):
        return _finite_real(payload, field=f"{reduction.value} payload")
    if reduction is Reduction.IDENTICAL:
        return _primitive(payload, field=f"{reduction.value} payload")
    raise RuntimeError(f"unsupported metric reduction {reduction}")


@dataclass(frozen=True)
class MetricContribution:
    """One typed contribution to a logical metric or metric family."""

    reduction: MetricReduction
    payload: MetricPayload

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _normalize_payload(self.reduction, self.payload))

    @classmethod
    def sum(cls, value: int | float) -> MetricContribution:
        return cls(Reduction.SUM, value)

    @classmethod
    def maximum(cls, value: int | float) -> MetricContribution:
        return cls(Reduction.MAX, value)

    @classmethod
    def identical(cls, value: MetricScalar) -> MetricContribution:
        return cls(Reduction.IDENTICAL, value)

    @classmethod
    def weighted_mean(cls, value: float, weight: float = 1.0) -> MetricContribution:
        return cls(Reduction.WEIGHTED_MEAN, WeightedMean.from_value(value, weight))

    @classmethod
    def moments(cls, moments: Moments) -> MetricContribution:
        return cls(Reduction.MOMENTS, moments)


MetricReport: TypeAlias = dict[str, MetricContribution]


def _merge_contributions(key: str, left: MetricContribution, right: MetricContribution) -> MetricContribution:
    left_payload = _normalize_payload(left.reduction, left.payload)
    right_payload = _normalize_payload(right.reduction, right.payload)
    if left.reduction is not right.reduction:
        raise ValueError(f"metric {key!r} changed reduction from {left.reduction.value} to {right.reduction.value}")
    reduction = left.reduction
    if reduction is Reduction.WEIGHTED_MEAN:
        return MetricContribution(
            reduction,
            cast(WeightedMean, left_payload).merge(cast(WeightedMean, right_payload)),
        )
    if reduction is Reduction.MOMENTS:
        return MetricContribution(reduction, cast(Moments, left_payload).merge(cast(Moments, right_payload)))
    if reduction is Reduction.IDENTICAL:
        if type(left_payload) is not type(right_payload) or left_payload != right_payload:
            raise ValueError(f"metric {key!r} must be identical, got {left_payload!r} and {right_payload!r}")
        return left
    left_value = cast(int | float, left_payload)
    right_value = cast(int | float, right_payload)
    if reduction is Reduction.SUM:
        return MetricContribution(reduction, left_value + right_value)
    if reduction is Reduction.MAX:
        return MetricContribution(reduction, max(left_value, right_value))
    raise RuntimeError(f"unsupported metric reduction {reduction}")


def merge_metric_reports(reports: Sequence[Mapping[str, MetricContribution]]) -> MetricReport:
    """Merge reports in order; missing keys remain unobserved rather than becoming zero."""
    merged: MetricReport = {}
    for report in reports:
        validate_public_metric_keys(report)
        for key, contribution in report.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"metric keys must be non-empty strings, got {key!r}")
            if not isinstance(contribution, MetricContribution):
                raise TypeError(f"metric {key!r} must be a MetricContribution, got {contribution!r}")
            merged[key] = contribution if key not in merged else _merge_contributions(key, merged[key], contribution)
    return merged


def finalize_metric_report(
    report: Mapping[str, MetricContribution],
    *,
    moment_outputs: Mapping[str, MomentOutputs] | None = None,
) -> FinalMetricReport:
    """Finalize a merged typed report into the scalar public payload consumed by Reporter.

    A ``MOMENTS`` contribution is a family rather than one scalar, so its output keys must be
    supplied explicitly.  This is what permits ``val/reward_avg`` and ``train/reward_mean`` to share
    one reduction implementation without guessing names from suffixes.
    """
    outputs = {} if moment_outputs is None else dict(moment_outputs)
    unknown_outputs = set(outputs).difference(report)
    if unknown_outputs:
        raise ValueError(f"moment outputs supplied for absent metrics: {sorted(unknown_outputs)!r}")
    finalized: FinalMetricReport = {}
    for key, contribution in report.items():
        payload = _normalize_payload(contribution.reduction, contribution.payload)
        if contribution.reduction is Reduction.MOMENTS:
            if key not in outputs:
                raise ValueError(f"MOMENTS metric {key!r} requires explicit MomentOutputs")
            projected = outputs[key].finalize(cast(Moments, payload))
        elif contribution.reduction is Reduction.WEIGHTED_MEAN:
            projected = {key: cast(WeightedMean, payload).value}
        else:
            projected = {key: cast(MetricScalar, payload)}
        collisions = set(finalized).intersection(projected)
        if collisions:
            raise ValueError(f"metric finalization produced duplicate keys: {sorted(collisions)!r}")
        finalized.update(projected)
    validate_public_metric_keys(finalized)
    return finalized


def validate_public_metric_keys(metrics: Mapping[str, object]) -> None:
    """Fail loudly unless every key belongs to the schema-v2 public namespaces."""
    for key in metrics:
        if not isinstance(key, str) or not key:
            raise ValueError(f"public metric keys must be non-empty strings, got {key!r}")
        if key in _LEGACY_METRIC_KEYS or key.startswith(_LEGACY_METRIC_PREFIXES):
            raise ValueError(f"legacy metric key {key!r} is forbidden by metrics schema v2")
        namespace, separator, name = key.partition("/")
        if not separator or namespace not in PUBLIC_METRIC_NAMESPACES or not name:
            allowed = ", ".join(f"{item}/*" for item in sorted(PUBLIC_METRIC_NAMESPACES))
            raise ValueError(f"public metric key {key!r} must use one of: {allowed}")


__all__ = [
    "FinalMetricReport",
    "MetricContribution",
    "MetricReduction",
    "MetricReport",
    "MetricScalar",
    "MomentOutputs",
    "Moments",
    "PUBLIC_METRIC_NAMESPACES",
    "Reduction",
    "SUMMED_TRAIN_COUNT_KEYS",
    "WeightedMean",
    "finalize_metric_report",
    "merge_metric_reports",
    "validate_public_metric_keys",
]
