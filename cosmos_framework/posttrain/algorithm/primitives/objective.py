# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trainer-facing objective contracts and loss-reduction modes.

An `Objective` pairs `(model, batch) -> (loss, metrics)` with the generic reduction contract that
the trainer honors during gradient accumulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

from cosmos_framework.posttrain.schema.trajectory import Trajectory

if TYPE_CHECKING:
    import torch


OBJECTIVE_WEIGHT_SUM = "_objective_weight_sum"


@dataclass(frozen=True)
class MetricAdapter:
    """Map algorithm-local objective metrics into one public phase namespace."""

    internal_prefix: str
    names: Mapping[str, str]
    phases: frozenset[str]
    ignored: frozenset[str] = frozenset()
    weighted_by: Mapping[str, str] = field(default_factory=dict)
    # Source names whose contributions compose by MAX rather than by (weighted) mean: a per-shard
    # maximum is only meaningful as the maximum over shards and chunks. Disjoint from `weighted_by`.
    max_reduced: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.internal_prefix, str) or not self.internal_prefix:
            raise ValueError("metric adapter internal_prefix must be non-empty")
        names = dict(self.names)
        if any(not isinstance(key, str) or not key for key in names):
            raise ValueError("metric adapter source names must be non-empty strings")
        if any(not isinstance(leaf, str) or not leaf or "/" in leaf for leaf in names.values()):
            raise ValueError("metric adapter public metric leaves must be non-empty and contain no '/'")
        if len(set(names.values())) != len(names):
            raise ValueError("metric adapter public metric leaves must be unique")
        phases = frozenset(self.phases)
        if not phases or any(phase not in {"train", "val"} for phase in phases):
            raise ValueError(f"metric adapter phases must be a non-empty subset of train/val, got {phases!r}")
        ignored = frozenset(self.ignored)
        if any(not isinstance(key, str) or not key for key in ignored):
            raise ValueError("metric adapter ignored keys must be non-empty strings")
        overlap = set(names).intersection(ignored)
        if overlap:
            raise ValueError(f"metric adapter cannot both map and ignore metrics: {sorted(overlap)!r}")
        weighted_by = dict(self.weighted_by)
        if any(not isinstance(key, str) or not key for key in weighted_by):
            raise ValueError("metric adapter weighted metric names must be non-empty strings")
        if any(not isinstance(key, str) or not key for key in weighted_by.values()):
            raise ValueError("metric adapter weight metric names must be non-empty strings")
        unmapped_metrics = set(weighted_by).difference(names)
        unmapped_weights = set(weighted_by.values()).difference(names)
        if unmapped_metrics or unmapped_weights:
            raise ValueError(
                "metric adapter weighted_by must reference mapped metrics; "
                f"unmapped metrics={sorted(unmapped_metrics)!r}, unmapped weights={sorted(unmapped_weights)!r}"
            )
        max_reduced = frozenset(self.max_reduced)
        if any(not isinstance(key, str) or not key for key in max_reduced):
            raise ValueError("metric adapter max_reduced names must be non-empty strings")
        unmapped_max = max_reduced.difference(names)
        if unmapped_max:
            raise ValueError(
                f"metric adapter max_reduced must reference mapped metrics; unmapped={sorted(unmapped_max)!r}"
            )
        max_and_weighted = max_reduced.intersection(weighted_by)
        if max_and_weighted:
            raise ValueError(f"metric adapter cannot both max-reduce and weight metrics: {sorted(max_and_weighted)!r}")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "ignored", ignored)
        object.__setattr__(self, "weighted_by", weighted_by)
        object.__setattr__(self, "max_reduced", max_reduced)

    def __call__(self, metrics: dict[str, Any], phase: str) -> dict[str, Any]:
        if phase not in self.phases:
            raise ValueError(f"objective metrics support phases {sorted(self.phases)!r}, got {phase!r}")
        public: dict[str, Any] = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError(f"objective metric keys must be strings, got {key!r}")
            if key in self.ignored:
                continue
            leaf = self.names.get(key)
            if leaf is not None:
                public_key = f"{phase}/{leaf}"
            elif key.startswith(self.internal_prefix):
                raise ValueError(f"unmapped internal objective metric {key!r}")
            else:
                public_key = key
            if public_key in public:
                raise ValueError(f"objective metric mapping produced duplicate key {public_key!r}")
            public[public_key] = value
        return public

    def public_weight_map(self, phase: str) -> dict[str, str]:
        """Return public metric keys mapped to their public aggregation-weight keys."""
        if phase not in self.phases:
            raise ValueError(f"objective metrics support phases {sorted(self.phases)!r}, got {phase!r}")
        return {
            f"{phase}/{self.names[metric]}": f"{phase}/{self.names[weight]}"
            for metric, weight in self.weighted_by.items()
        }

    def public_max_keys(self, phase: str) -> frozenset[str]:
        """Return the public metric keys that aggregate by MAX across ranks and chunks."""
        if phase not in self.phases:
            raise ValueError(f"objective metrics support phases {sorted(self.phases)!r}, got {phase!r}")
        return frozenset(f"{phase}/{self.names[metric]}" for metric in self.max_reduced)


def _normalize_sample_weights(sample_weights: list[float] | None) -> list[float] | None:
    if sample_weights is None:
        return None
    if not isinstance(sample_weights, list):
        raise TypeError(f"sample_weights must be a list or None, got {type(sample_weights).__name__}")
    normalized: list[float] = []
    for value in sample_weights:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"validation_weight values must be non-bool real numbers, got {value!r}")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("validation_weight values must be finite and non-negative")
        normalized.append(weight)
    return normalized


def _normalize_sample_seeds(sample_seeds: list[int] | None) -> list[int] | None:
    if sample_seeds is None:
        return None
    if not isinstance(sample_seeds, list):
        raise TypeError(f"sample_seeds must be a list or None, got {type(sample_seeds).__name__}")
    normalized: list[int] = []
    for value in sample_seeds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"validation_noise_seed values must be non-bool integers, got {value!r}")
        if value < 0 or value >= 2**64:
            raise ValueError("validation_noise_seed values must be unsigned 64-bit integers")
        normalized.append(value)
    return normalized


@dataclass(frozen=True)
class ValidationBatchMetadata:
    """Validated per-sample controls for held-out objective calls."""

    sample_weights: list[float] | None = None
    sample_seeds: list[int] | None = None

    def __post_init__(self) -> None:
        weights = _normalize_sample_weights(self.sample_weights)
        seeds = _normalize_sample_seeds(self.sample_seeds)
        if weights is not None and seeds is not None and len(weights) != len(seeds):
            raise ValueError(
                f"validation sample_weights and sample_seeds must have equal lengths, got {len(weights)} and {len(seeds)}"
            )
        object.__setattr__(self, "sample_weights", weights)
        object.__setattr__(self, "sample_seeds", seeds)


def read_validation_batch_metadata(trajectories: list[Trajectory]) -> ValidationBatchMetadata:
    """Read optional validation controls, requiring each control on all samples or none."""
    raw_weights: list[object | None] = []
    raw_seeds: list[object | None] = []
    for trajectory in trajectories:
        if not trajectory.transitions:
            raise ValueError("validation metadata requires every trajectory to contain a transition")
        metadata = trajectory.transitions[0].algo_extra
        raw_weights.append(metadata.get("validation_weight"))
        raw_seeds.append(metadata.get("validation_noise_seed"))

    has_weights = any(value is not None for value in raw_weights)
    if has_weights and any(value is None for value in raw_weights):
        raise ValueError("validation_weight must be present on every trajectory in a masked batch")
    has_seeds = any(value is not None for value in raw_seeds)
    if has_seeds and any(value is None for value in raw_seeds):
        raise ValueError("validation_noise_seed must be present on every held-out trajectory")
    return ValidationBatchMetadata(
        sample_weights=list(raw_weights) if has_weights else None,  # type: ignore[arg-type]
        sample_seeds=list(raw_seeds) if has_seeds else None,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class PerSampleReducer:
    """Apply one optional per-sample mask consistently to loss and diagnostics."""

    _weights: torch.Tensor | None
    weight_sum: torch.Tensor | None
    _context: str

    @classmethod
    def from_weights(
        cls,
        reference: torch.Tensor,
        sample_weights: list[float] | None,
        *,
        context: str,
    ) -> PerSampleReducer:  # reference: [B]
        import torch

        if not isinstance(reference, torch.Tensor):
            raise TypeError(f"{context} reference must be a tensor, got {type(reference).__name__}")
        normalized = _normalize_sample_weights(sample_weights)
        if normalized is None:
            return cls(_weights=None, weight_sum=None, _context=context)
        weights = torch.as_tensor(normalized, device=reference.device, dtype=reference.dtype)  # [B]
        if tuple(weights.shape) != tuple(reference.shape):
            raise ValueError(
                f"{context} sample_weights shape {tuple(weights.shape)} does not match reference shape "
                f"{tuple(reference.shape)}"
            )
        weight_sum = weights.sum()  # []
        return cls(_weights=weights, weight_sum=weight_sum, _context=context)

    def mean(self, values: torch.Tensor) -> torch.Tensor:  # values: [B], returns []
        import torch

        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{self._context} values must be a tensor, got {type(values).__name__}")
        if self._weights is None or self.weight_sum is None:
            return values.mean()  # []
        if tuple(values.shape) != tuple(self._weights.shape):
            raise ValueError(
                f"{self._context} values shape {tuple(values.shape)} does not match sample_weights shape "
                f"{tuple(self._weights.shape)}"
            )
        divisor = torch.where(self.weight_sum == 0, torch.ones_like(self.weight_sum), self.weight_sum)  # []
        return (values * self._weights).sum() / divisor  # []


class Reduction(Enum):
    """How an objective's normalized loss composes across micro-batches.

    TOKEN_SUM uses a DP-global valid-unit weight, which may count tokens, sequences, or another
    objective-defined unit. The trainer recovers each chunk numerator and normalizes once per step.
    """

    MEAN = "mean"
    TOKEN_SUM = "token_sum"


def normalization_scale(weight_sum: float) -> float:
    """Return the inverse weight, using one only for a truly empty reduction."""
    if isinstance(weight_sum, bool) or not isinstance(weight_sum, Real):
        raise TypeError(f"objective weight sum must be a real number, got {weight_sum!r}")
    value = float(weight_sum)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"objective weight sum must be finite and non-negative, got {weight_sum!r}")
    return 1.0 if value == 0.0 else 1.0 / value


def backward(loss: "torch.Tensor", reduction: Reduction, grad_accum: int, denom: float | None = None) -> None:
    """Fold `loss` into gradient accumulation per the objective's reduction, then backprop.

    Args:
        loss: the objective's returned loss — already normalized by the objective.
        reduction: the objective's declared `Reduction`.
        grad_accum: number of `train` calls per optimizer step (the micro count).
        denom: the DP-mean divisor the objective reported as ``OBJECTIVE_WEIGHT_SUM``. Required
            for TOKEN_SUM under accumulation, ignored otherwise.

    Raises:
        NotImplementedError: TOKEN_SUM with `grad_accum > 1` and no `denom` — a per-chunk
            denominator is a mean-of-means, which is the bug this guard has always existed to stop.
    """
    if reduction is Reduction.MEAN:
        (loss / grad_accum).backward()
    elif reduction is Reduction.TOKEN_SUM:
        if grad_accum == 1:
            loss.backward()  # today's path, byte-for-byte
        elif denom is None:
            raise NotImplementedError(
                "token_sum with grad_accum>1 requires the objective weight sum; without it the "
                "per-chunk division is a mean-of-means."
            )
        else:
            # Recover the chunk SUM. The trainer divides ONCE, by the accumulated divisor,
            # before clip-and-step — see roles/trainer/llm.py and spec §3.5.
            (loss * denom).backward()
    else:
        raise AssertionError(f"unhandled reduction {reduction}")


@dataclass
class Objective:
    """A trainer-facing objective: the callable `(model, batch) -> (loss, metrics)` plus the
    `Reduction` it declares (both explicit).

    Batch precondition: objectives consume annotations prepared by upstream data/reward stages; they
    validate required annotations but do not derive group- or source-level state from a rank shard.
    """

    fn: Callable[..., "tuple[torch.Tensor, dict[str, Any]]"]
    reduction: Reduction
    # π_old frozen across an optimizer-step window, or None for objectives that have no ratio.
    # Carried HERE rather than passed to the trainer separately because the trainer receives the
    # objective and nothing else — the same reason `reduction` rides along. `LLMTrainer` reads it
    # with `getattr(..., None)`, so an objective that omits it behaves exactly as before.
    # See `algorithm/grpo.py::OldLogprobCache`.
    old_logp_cache: Any = None
    # Scorer of the frozen reference's per-token logprobs for a loss-side KL anchor (`rl.kl_beta > 0`),
    # or None for objectives without one. The objective consumes reference SCORES through it and
    # nothing else: which network computes them, and where that network lives between updates, is
    # the entry's and the trainer's business (`entrypoints/grpo.py::build_trainer` attaches the
    # network it built and registers it with `LLMTrainer(auxiliary_models=...)` for device
    # residency). The trainer never reads this field. See `algorithm/grpo.py::FrozenReference`.
    reference_scorer: Any = None
    metric_adapter: Callable[[dict[str, Any], str], dict[str, Any]] | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    # Optional deterministic CPU preprocessing over ONE completed-update window.
    # The driver owns cursor advancement; these hooks cannot read the data source.
    prepare_window: Callable[[list[list[Any]]], None] | None = None
    finish_window: Callable[[], None] | None = None
    close: Callable[[], None] | None = None
    pipeline_stats: Callable[[], dict[str, Any]] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> "tuple[torch.Tensor, dict[str, Any]]":
        return self.fn(*args, **kwargs)

    def adapt_metrics(self, metrics: dict[str, Any], phase: str) -> dict[str, Any]:
        """Convert phase-neutral objective metrics to the public stage namespace.

        The objective math may keep algorithm-local names so the same callable is reusable for
        training and held-out evaluation. Its optional adapter owns the one hard-cut mapping to
        ``train/*`` or ``val/*``; trainers stay algorithm-agnostic.
        """
        if phase not in ("train", "val"):
            raise ValueError(f"objective metric phase must be 'train' or 'val', got {phase!r}")
        return dict(metrics) if self.metric_adapter is None else self.metric_adapter(metrics, phase)

    def metric_weight_map(self, phase: str) -> dict[str, str]:
        """Return declared public metric-to-weight relationships for downstream aggregation."""
        if phase not in ("train", "val"):
            raise ValueError(f"objective metric phase must be 'train' or 'val', got {phase!r}")
        adapter = self.metric_adapter
        if adapter is None or not callable(adapter):
            return {}
        public_weight_map = getattr(adapter, "public_weight_map", None)
        if not callable(public_weight_map):
            return {}
        return dict(public_weight_map(phase))

    def metric_max_keys(self, phase: str) -> frozenset[str]:
        """Return the declared public metric keys that aggregate by MAX (see ``MetricAdapter.max_reduced``)."""
        if phase not in ("train", "val"):
            raise ValueError(f"objective metric phase must be 'train' or 'val', got {phase!r}")
        adapter = self.metric_adapter
        if adapter is None or not callable(adapter):
            return frozenset()
        public_max_keys = getattr(adapter, "public_max_keys", None)
        if not callable(public_max_keys):
            return frozenset()
        return frozenset(public_max_keys(phase))
