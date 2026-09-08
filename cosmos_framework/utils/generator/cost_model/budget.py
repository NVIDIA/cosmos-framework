# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""An iteration-time ceiling for the sequence packer, priced by the token cost model.

The packer's token ceiling (``max_num_tokens_after_packing``) is a poor proxy for
step time, because attention is quadratic: 40k tokens spent on one long clip cost
far more than the same 40k spread over forty stills. This module adds a second,
time-denominated ceiling, so a batch stops growing on whichever binds first.

From GPU-seconds per step to wall-clock seconds per rank
-------------------------------------------------------
``TokenCostModel`` is fitted in GPU-seconds summed over the whole job::

    gpu_sec_per_step = c0 + sum over every sample on every rank of m(x, y)

where ``m`` is :meth:`TokenCostModel.marginal_gpu_sec` and ``c0`` is the
token-independent floor (optimizer step, EMA, FSDP collectives). Each rank packs
and prices only its own batch, so for balanced ranks::

    wall_sec_per_step = c0 / benchmark_world_size + sum over this rank's samples of m(x, y)

A sample's marginal needs no conversion: the rank that owns a sample is the one
that spends the time on it, so a per-sample GPU-second already *is* a per-rank
second. ``c0`` does, being replicated work that all the benchmark's ranks did at
once, and dividing by the scale it was measured at leaves what one of them waited
for -- see :attr:`TokenCostModel.fixed_sec_per_step_per_rank`.

The divisor is always the benchmark's world size, never the training job's, and it
is applied where the coefficients are produced: a config states the quotient, a
calibration file carries the scale to divide by. That is what makes a target
portable, a 4-rank fit pricing a batch on 256 ranks, since the target and every
term feeding it belong to one GPU. Dividing by the training scale instead would
shrink the floor by the ratio between the two -- 64x here -- and reserve almost
nothing out of the target.

Projecting needs no collective, which is what lets the packing loop use it as an
admission test.

Three consequences worth knowing before setting a target:

- The target is enforced *per rank*, but a step is as slow as its slowest rank, so
  the realised iteration time tracks the target only as far as ranks pack alike.
- The projection inherits the fit's error bars. Check the ``quality`` line of
  ``cost_model.report`` for the shapes you actually train on: a fit off by 50% on
  them misses the target by about as much.
- Far above the benchmark's scale the reserved fixed cost is a lower bound, since
  collective latency and straggler spread both grow with the rank count, so the
  target errs towards over-packing.
- ``c0`` covers only the per-step work the benchmark actually timed, and the
  parameter update is opt-in there while the EMA average is never built at all.
  The ``measured`` block of ``cost_model.report`` says which of them a fit
  includes; whatever it skipped is missing from the reserve as well.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_framework.utils import log
from cosmos_framework.utils.generator.cost_model.estimator import Calibration, TokenCostModel, fit_token_cost_model

# The six per-sample coefficients, in the order the report prints them. Named here so
# the config can tell "no coefficients given" from "a term the fit dropped, left at
# zero", and shared verbatim with the fit and the report's JSON, since a per-sample cost
# means the same thing everywhere: seconds added to whichever rank owns the sample.
# ``c0`` is kept apart, being the one term a config states per rank while the fit
# reports it summed over ranks.
MARGINAL_COEFFICIENT_FIELDS: tuple[str, ...] = (
    "fixed_gpu_sec_per_sample",
    "gpu_sec_per_und_area",
    "gpu_sec_per_gen_area",
    "gpu_sec_per_cross_area",
    "gpu_sec_per_und_token",
    "gpu_sec_per_gen_token",
)


@dataclass
class IterationTimeBudgetConfig:
    """Config for the packer's iteration-time ceiling.

    Disabled by default: with ``iteration_time_target=None`` the packer is bounded
    by tokens alone, exactly as before.

    The coefficients come from ``cost_model.benchmark`` on the model and GPU you are
    training, so they do not survive a change of either -- but they do survive a
    change of world size, every one of them being stated per GPU. Supply them as a
    path to the benchmark's calibration JSON (refitted here) or as explicit numbers
    from the report, not both.

    Attributes:
        iteration_time_target: Wall-clock seconds per iteration that a packed batch
            may project to; ``None`` disables the ceiling. Samples are added while
            the projection stays at or below it, but none is ever dropped and no
            batch is emitted empty to honour it, so too small a target degrades to
            one sample per batch rather than to a stalled input stream.
        calibration: Path (or sequence of paths) to calibration JSON written by
            ``cost_model.benchmark``. Several are pooled and refitted together, as
            ``cost_model.report --calibration`` does.
        fixed_gpu_sec_per_step: ``c0``, as the seconds ONE rank waits for it and not
            the GPU-second total the fit reports over its ranks -- 1.917 GPU-s from
            a 4-rank benchmark, read as 1.917 s of a rank's iteration, over-reserves
            fourfold. The name is the fitted field's, the value is not, so this is
            the one coefficient not copied off the ``c0`` line: take the per-rank
            figure printed beneath it, which ``--json`` emits as
            ``fixed_sec_per_step_per_rank``.
        fixed_gpu_sec_per_sample: ``F``, the per-sample floor.
        gpu_sec_per_und_area: ``A``, understanding self-attention.
        gpu_sec_per_gen_area: ``B``, generation self-attention.
        gpu_sec_per_cross_area: ``C``, the cross quadrant.
        gpu_sec_per_und_token: ``D``, linear understanding work.
        gpu_sec_per_gen_token: ``E``, linear generation work.
    """

    iteration_time_target: float | None = None
    # Typed ``Any`` rather than ``str | list[str] | None`` because OmegaConf rejects a
    # union containing a container, and it wraps this dataclass as soon as a config holds
    # an instance of it -- which made the union an import error for every config in the
    # tree. ``_calibration_paths`` enforces at runtime what it would state.
    calibration: Any = None
    fixed_gpu_sec_per_step: float = 0.0
    fixed_gpu_sec_per_sample: float = 0.0
    gpu_sec_per_und_area: float = 0.0
    gpu_sec_per_gen_area: float = 0.0
    gpu_sec_per_cross_area: float = 0.0
    gpu_sec_per_und_token: float = 0.0
    gpu_sec_per_gen_token: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether a time ceiling was requested."""
        return self.iteration_time_target is not None

    def build(self) -> IterationTimeBudget | None:
        """Resolve the config into a usable budget.

        Returns:
            The budget, or ``None`` when no target was set.

        Raises:
            ValueError: If the target is non-positive, if both a calibration path
                and explicit coefficients are given, or if a target is set but no
                coefficients are available to price it with.
        """
        if self.iteration_time_target is None:  # No target, i.e. not self.enabled.
            return None

        target = float(self.iteration_time_target)
        if target <= 0.0:
            raise ValueError(f"iteration_time_target must be positive, got {target}.")

        # Typed loosely because the keys are field names resolved at runtime, so nothing
        # can check them against the constructor's parameters anyway.
        marginal: dict[str, Any] = {name: float(getattr(self, name)) for name in MARGINAL_COEFFICIENT_FIELDS}
        fixed = float(self.fixed_gpu_sec_per_step)
        explicit = {"fixed_gpu_sec_per_step": fixed, **marginal}
        any_explicit = any(value != 0.0 for value in explicit.values())

        if self.calibration is not None and any_explicit:
            raise ValueError(
                "Set either iteration_time_budget.calibration or the explicit coefficients, not both: "
                f"got calibration={self.calibration!r} alongside "
                f"{sorted(name for name, value in explicit.items() if value != 0.0)}."
            )

        if self.calibration is not None:
            cost_model = fit_token_cost_model(Calibration.load_all(self._calibration_paths()))
        elif any_explicit:
            # A world size of one says the configured fixed cost is already per rank: it
            # is the scale the number is expressed at, so nothing is left to divide out.
            # The fit's own world size is deliberately absent from this config, being one
            # more number to get wrong when only the quotient is ever used.
            cost_model = TokenCostModel(fixed_gpu_sec_per_step=fixed, world_size=1, **marginal, source="config")
        else:
            raise ValueError(
                f"iteration_time_target={target} was set but no cost model was given, so a batch cannot be "
                "priced. Set iteration_time_budget.calibration to a cost_model.benchmark JSON, or copy the "
                "coefficients from cost_model.report into the explicit fields."
            )

        return IterationTimeBudget(cost_model=cost_model, iteration_time_target=target)

    def _calibration_paths(self) -> list[str | Path]:
        """``calibration`` as a list of paths, whichever form the config wrote it in.

        Raises:
            ValueError: If the value is neither a path nor a sequence of paths.
        """
        value = self.calibration
        if isinstance(value, (str, Path)):
            return [value]
        # A list written in an experiment config arrives as an OmegaConf ListConfig, so
        # iterate rather than test for list. Mappings are excluded because iterating one
        # yields its keys, which would quietly fit a dict of paths.
        if isinstance(value, Iterable) and not isinstance(value, Mapping):
            paths = list(value)
            if all(isinstance(path, (str, Path)) for path in paths):
                return paths
        raise ValueError(
            "iteration_time_budget.calibration must be a path or a sequence of paths to "
            f"cost_model.benchmark JSON, got {value!r}."
        )


@dataclass(frozen=True)
class IterationTimeBudget:
    """A per-rank wall-clock ceiling on one packed batch.

    Everything here is seconds on one GPU, which is why no training world size
    appears: the same budget prices a batch identically on four ranks or four
    thousand. The only scale that enters travels inside the cost model.

    Attributes:
        cost_model: Fitted token cost model, in GPU-seconds, carrying the world
            size it was fitted at.
        iteration_time_target: Wall-clock seconds per iteration the packer aims at.
    """

    cost_model: TokenCostModel
    iteration_time_target: float

    @property
    def fixed_seconds(self) -> float:
        """Wall-clock seconds this rank spends on work no sample is charged for.

        ``c0`` is the optimizer step, EMA and FSDP collectives, which every rank runs
        concurrently on its own shard, so a rank waits for the share measured at the
        calibration's scale -- not at this job's, which nothing here is ever told.
        """
        return self.cost_model.fixed_sec_per_step_per_rank

    def sample_seconds(self, und_tokens: int, gen_tokens: int) -> float:
        """Wall-clock seconds one sample adds to this rank's step.

        Attention is block-diagonal across packed samples, so a sample's cost does
        not depend on what else shares the batch: this marginal is exact rather than
        an approximation that degrades as the batch fills.

        Args:
            und_tokens: Understanding-pathway (caption) tokens.
            gen_tokens: Generation-pathway (vision, action, sound) tokens.

        Returns:
            Seconds added to this rank's iteration time.
        """
        return self.cost_model.marginal_gpu_sec(und_tokens, gen_tokens)

    def projected_seconds(self, packed_sample_seconds: float) -> float:
        """Projected iteration time for a batch whose samples sum to ``packed_sample_seconds``."""
        return self.fixed_seconds + packed_sample_seconds

    def has_room_for(self, packed_sample_seconds: float, sample_seconds: float) -> bool:
        """Whether one more sample keeps the projection within the target.

        Args:
            packed_sample_seconds: Sum of ``sample_seconds`` over the batch so far.
            sample_seconds: Cost of the candidate sample.

        Returns:
            True when the batch may grow by this sample.
        """
        return self.projected_seconds(packed_sample_seconds + sample_seconds) <= self.iteration_time_target

    def describe(self) -> str:
        """One-line summary for the startup log."""
        return (
            f"iteration_time_target={self.iteration_time_target:.3f}s/iter per rank "
            f"(fixed={self.fixed_seconds:.3f}s/iter per rank, cost model {self.cost_model.source})"
        )

    def warn_if_unreachable(self) -> None:
        """Log once when the target cannot be met even by an empty batch."""
        if self.fixed_seconds >= self.iteration_time_target:
            log.warning(
                f"iteration_time_target={self.iteration_time_target:.3f}s is at or below the "
                f"{self.fixed_seconds:.3f}s/iter fixed cost of a step, so no batch can meet it. "
                "Packing will fall back to one sample per batch; raise the target or re-check "
                "the fitted c0."
            )
