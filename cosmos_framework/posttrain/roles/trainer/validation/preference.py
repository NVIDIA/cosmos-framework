# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Finite held-out execution for trainer-bound preference objectives."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import math
import random
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, Objective
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.model.protocol import FlowMatchingPolicy
from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv, PreferenceSourceIdentity
from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_tensor_finite
from cosmos_framework.posttrain.roles.trainer.validation.runtime import (
    reshard_fsdp_modules,
    validation_module_inventory,
)
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.schema import (
    MetricContribution,
    MetricReport,
    Trajectory,
    finalize_metric_report,
    merge_metric_reports,
)
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed, run_rank_converged
from cosmos_framework.posttrain.utils.scalar import as_scalar

_Result = TypeVar("_Result")


def _run_converged(action: Callable[[], _Result], context: str) -> _Result:
    return run_rank_converged(action, context, raise_errors=raise_if_any_rank_failed)


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively rebuild a nested batch with every tensor on ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)  # [*obj.shape]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(value, device) for value in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        for field in dataclasses.fields(obj):
            # Frozen dataclasses reject normal setattr; mutate only the shallow
            # copy's storage without changing the source object.
            object.__setattr__(out, field.name, move_to_device(getattr(obj, field.name), device))
        return out
    return obj


def build_objective_validation_identity(
    source: DatasetPreferenceEnv | None,
    *,
    batch_size: int,
    max_samples: int | None,
    seed: int,
    objective: Objective,
    before_training: bool,
    every_steps: int,
    train_total_steps: int,
) -> dict[str, Any]:
    """Build the exact held-out objective identity persisted with trainer state."""
    if source is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "source_fingerprint": source.source_fingerprint,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "seed": seed,
        "objective_identity": dict(objective.identity),
        "schedule": {
            "before_training": before_training,
            "every_steps": every_steps,
            "train_total_steps": train_total_steps,
        },
    }


@dataclasses.dataclass
class _ValidationRuntime:
    stack: contextlib.ExitStack = dataclasses.field(default_factory=contextlib.ExitStack)
    rng_state: dict[str, Any] | None = None
    modes: list[tuple[torch.nn.Module, bool]] = dataclasses.field(default_factory=list)
    roots: dict[str, torch.nn.Module] = dataclasses.field(default_factory=dict)
    iterator: Any = None
    cuda_device: torch.device | None = None


@dataclasses.dataclass(kw_only=True)
class ObjectiveValidationRunner:
    """Run a finite held-out source through one bound objective."""

    model: FlowMatchingPolicy
    objective: Objective
    source: DatasetPreferenceEnv
    state: ValidationState
    batch_size: int
    max_samples: int | None
    seed: int
    device: Callable[[], torch.device]

    def source_identity(self) -> PreferenceSourceIdentity:
        return self.source.preference_source_identity()

    def _prepare_state(self, runtime: _ValidationRuntime) -> None:
        from cosmos_framework.posttrain.checkpoint import collect_rng_state

        model_device = next(self.model.net.parameters()).device
        runtime.cuda_device = model_device if model_device.type == "cuda" else None
        runtime.rng_state = collect_rng_state(cuda_device=runtime.cuda_device)
        validation_roots, validation_modules = validation_module_inventory(self.model)
        runtime.roots = validation_roots
        runtime.modes = [(module, bool(module.training)) for module in validation_modules]
        validation_context = getattr(self.model, "validation_context", None)
        if callable(validation_context):
            runtime.stack.enter_context(validation_context())
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        torch.random.default_generator.manual_seed(self.seed)
        if runtime.cuda_device is not None:
            with torch.cuda.device(runtime.cuda_device):
                torch.cuda.manual_seed(self.seed)
        for root in validation_roots.values():
            root.eval()

    @staticmethod
    def _fetch_batch(iterator: Any, expected_index: int) -> Any:
        batch = next(iterator)
        if batch.batch_index != expected_index:
            raise RuntimeError(f"validation source returned batch_index={batch.batch_index}, expected {expected_index}")
        return batch

    def _annotate_seeds(self, trajectories: list[Trajectory]) -> None:
        for trajectory in trajectories:
            seed_material = f"{self.seed}|{self.source.source_fingerprint}|{trajectory.id}".encode()
            trajectory.transitions[0].algo_extra["validation_noise_seed"] = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                "big",
            )

    def _batch_report(
        self,
        raw_metrics: dict[str, Any],
        *,
        logical_count: int,
    ) -> MetricReport:
        raw_weight = as_scalar(raw_metrics.get(OBJECTIVE_WEIGHT_SUM))
        if raw_weight is None or not math.isclose(
            raw_weight,
            float(logical_count),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "held-out objective weight does not match the source's real-row mask: "
                f"objective={raw_weight!r}, source={logical_count}"
            )
        public_metrics = self.objective.adapt_metrics(raw_metrics, "val")
        report: MetricReport = {}
        for key, value in public_metrics.items():
            if not key.startswith("val/"):
                raise ValueError(f"held-out Objective emitted non-validation public key {key!r}")
            scalar = as_scalar(value, allow_bool=False)
            if scalar is None:
                raise TypeError(f"held-out metric {key!r} is not a numeric scalar")
            report[key] = MetricContribution.weighted_mean(scalar, float(logical_count))
        return report

    @staticmethod
    def _require_exhausted(iterator: Any, expected_batches: int) -> None:
        try:
            next(iterator)
        except StopIteration:
            return
        raise RuntimeError(f"validation source produced more than {expected_batches} physical batches")

    @staticmethod
    def _restore_state(runtime: _ValidationRuntime, primary_error: BaseException | None) -> None:
        from cosmos_framework.posttrain.checkpoint import restore_rng_state

        failures: list[str] = []

        def cleanup(label: str, operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as error:  # noqa: BLE001 -- converge before peers leave cleanup
                failures.append(f"{label}: {type(error).__name__}: {error}")

        close_iterator = getattr(runtime.iterator, "close", None)
        if callable(close_iterator):
            cleanup("validation iterator close", close_iterator)
        cleanup("model validation context close", runtime.stack.close)
        cleanup("FSDP reshard", lambda: reshard_fsdp_modules(runtime.roots))
        for module_index, (module, was_training) in enumerate(runtime.modes):
            cleanup(
                f"module mode restore {module_index}",
                lambda module=module, was_training=was_training: setattr(module, "training", was_training),
            )
        if runtime.rng_state is not None:
            cleanup(
                "training RNG restore",
                lambda: restore_rng_state(runtime.rng_state, cuda_device=runtime.cuda_device),
            )
        cleanup_error = RuntimeError("; ".join(failures)) if failures else None
        try:
            raise_if_any_rank_failed(cleanup_error, "preference validation state restoration")
        except BaseException as restoration_error:  # noqa: BLE001 -- preserve the primary failure
            if primary_error is not None:
                raise primary_error from restoration_error
            raise

    def _finalize_report(
        self,
        local_report: MetricReport,
        *,
        local_pairs: int,
        expected_batches: int,
        started: float,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

        local_report = merge_metric_reports(
            (
                local_report,
                {
                    "val/pair_count": MetricContribution.sum(local_pairs),
                    "val/batch_count": MetricContribution.identical(expected_batches),
                    "timer/validation_s": MetricContribution.maximum(time.monotonic() - started),
                },
            )
        )
        reports_by_rank = gather_rank_states(
            local_report,
            context="preference validation metric report preparation",
        )
        finalized = finalize_metric_report(merge_metric_reports(tuple(reports_by_rank.values())))
        pair_count = finalized.get("val/pair_count")
        if type(pair_count) is not int or pair_count < 1:
            raise RuntimeError(f"invalid global validation pair count {pair_count!r}")
        duration_s = finalized.get("timer/validation_s")
        if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool):
            raise RuntimeError(f"invalid validation duration {duration_s!r}")
        return ValidationReport(
            model_step=model_step,
            sample_count=pair_count,
            metrics=finalized,
            source_fingerprint=self.source.source_fingerprint,
            trigger_reasons=trigger_reasons,
            duration_s=float(duration_s),
        )

    def run(
        self,
        *,
        model_step: int,
        iteration: int,
        accumulation: int,
        trigger_reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        """Evaluate a transaction-owned pending phase without training-state side effects."""
        if model_step != iteration:
            raise ValueError(f"validation model_step={model_step} does not match trainer iteration {iteration}")
        if accumulation != 0:
            raise RuntimeError(f"validation requires an optimizer boundary, got accum={accumulation}")
        if self.state.progress.pending_step != model_step:
            raise RuntimeError(
                f"validation transaction must mark model step {model_step} pending before evaluation; "
                f"current pending step is {self.state.progress.pending_step}"
            )
        effective_reasons = self.state.progress.pending_reasons | trigger_reasons
        runtime = _ValidationRuntime()
        try:
            _run_converged(
                lambda: self._prepare_state(runtime),
                "preference validation state setup",
            )

            from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

            expected_batches = _run_converged(
                lambda: self.source.validation_batch_count(self.batch_size, self.max_samples),
                "preference validation inventory",
            )
            inventory_by_rank = gather_rank_states(
                {
                    "source_fingerprint": self.source.source_fingerprint,
                    "physical_batch_count": expected_batches,
                },
                context="preference validation inventory preparation",
            )
            inventory_values = list(inventory_by_rank.values())
            if any(inventory != inventory_values[0] for inventory in inventory_values[1:]):
                raise RuntimeError(f"preference validation inventory differs across ranks: {inventory_by_rank}")
            runtime.iterator = _run_converged(
                lambda: iter(self.source.validation_batches(self.batch_size, self.max_samples)),
                "preference validation iterator construction",
            )
            local_report: MetricReport = {}
            local_pairs = 0
            started = time.monotonic()
            with torch.inference_mode():
                for expected_index in range(expected_batches):
                    batch = _run_converged(
                        lambda: self._fetch_batch(runtime.iterator, expected_index),
                        "preference validation batch fetch",
                    )
                    physical_sizes = gather_rank_states(
                        len(batch.trajectories),
                        context="preference validation physical batch preparation",
                    )
                    if len(set(physical_sizes.values())) != 1:
                        raise RuntimeError(f"validation physical batch size differs across ranks: {physical_sizes}")
                    prepared_batch: list[Trajectory] = _run_converged(
                        lambda: move_to_device(batch.trajectories, self.device()),
                        "validation batch device transfer",
                    )
                    _run_converged(
                        lambda: self._annotate_seeds(prepared_batch),
                        "validation sample seed preparation",
                    )
                    loss, raw_metrics = self.objective(self.model, prepared_batch)  # loss: []
                    if not all_ranks_tensor_finite(loss):
                        raise RuntimeError(
                            f"held-out objective returned non-finite loss at model step {model_step}, "
                            f"batch {expected_index}"
                        )
                    batch_report = _run_converged(
                        lambda: self._batch_report(raw_metrics, logical_count=batch.logical_count),
                        "preference validation metric preparation",
                    )
                    local_report = merge_metric_reports((local_report, batch_report))
                    local_pairs += batch.logical_count
            _run_converged(
                lambda: self._require_exhausted(runtime.iterator, expected_batches),
                "preference validation source exhaustion",
            )
            return self._finalize_report(
                local_report,
                local_pairs=local_pairs,
                expected_batches=expected_batches,
                started=started,
                model_step=model_step,
                trigger_reasons=effective_reasons,
            )
        finally:
            self._restore_state(runtime, sys.exc_info()[1])
