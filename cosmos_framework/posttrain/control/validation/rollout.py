# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Held-out rollout/reward evaluation and its checkpoint identity."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.roles.env.prompt import validate_dataset_source
from cosmos_framework.posttrain.roles.rollout.validation import (
    build_validation_context,
    normalize_validation_semantics_identity,
    resolve_rollout_validation_semantics,
    resolved_validation_sampling,
    run_validation_stage,
)
from cosmos_framework.posttrain.utils.batch_stats import (
    ValidationRewardStats,
    resolve_validation_reward_report,
)
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
from cosmos_framework.posttrain.utils.entry import replica_lead_values, resolve_all
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.rank_state import gather_rank_states
from cosmos_framework.posttrain.workers.worker_group import Scatter

if TYPE_CHECKING:
    from cosmos_framework.posttrain.algorithm.reward import RewardProviderCatalog
    from cosmos_framework.posttrain.embodied.window import EvaluationEpisode, WindowEnvironment
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

_EMPTY_REWARD_PROVIDER_IDENTITY_HASH = hashlib.sha256(b"{}").hexdigest()


def normalize_rollout_validation_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the rollout/reward identity stored beside model weights."""
    if not isinstance(identity, dict):
        raise TypeError(f"validation identity must be a dictionary, got {type(identity).__name__}")
    enabled = identity.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"validation identity enabled must be bool, got {enabled!r}")
    if not enabled:
        if set(identity) != {"enabled"}:
            raise ValueError("disabled validation identity must contain only enabled=false")
        return {"enabled": False}

    expected = {
        "enabled",
        "source_fingerprint",
        "rollout_semantics",
        "sampling",
        "reward",
        "reward_provider",
        "seed",
        "batch_size",
        "max_samples",
        "schedule",
    }
    actual_fields = set(identity)
    legacy_fields = expected - {"reward_provider"}
    if actual_fields == legacy_fields:
        legacy_reward = identity.get("reward")
        if not isinstance(legacy_reward, str) or not legacy_reward:
            raise ValueError("legacy validation reward must be a non-empty string")
        identity = {
            **identity,
            "reward_provider": {
                "name": legacy_reward,
                "revision": "builtin-v1",
                "catalog": "builtin",
                "identity_hash": _EMPTY_REWARD_PROVIDER_IDENTITY_HASH,
            },
        }
        actual_fields = set(identity)
    if actual_fields != expected:
        raise ValueError(
            f"enabled validation identity fields differ: missing={sorted(expected - actual_fields)}, "
            f"unknown={sorted(actual_fields - expected)}"
        )
    source_fingerprint = identity["source_fingerprint"]
    reward = identity["reward"]
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("validation source_fingerprint must be a non-empty string")
    if not isinstance(reward, str) or not reward:
        raise ValueError("validation reward must be a non-empty string")
    reward_provider = identity["reward_provider"]
    if isinstance(reward_provider, dict) and set(reward_provider) == {"name", "revision", "catalog"}:
        reward_provider = {
            **reward_provider,
            "identity_hash": _EMPTY_REWARD_PROVIDER_IDENTITY_HASH,
        }
        identity = {**identity, "reward_provider": reward_provider}
    reward_provider_fields = {"name", "revision", "catalog", "identity_hash"}
    actual_reward_provider_fields = set(reward_provider) if isinstance(reward_provider, dict) else set()
    if not isinstance(reward_provider, dict) or actual_reward_provider_fields != reward_provider_fields:
        raise ValueError(
            "validation reward_provider identity fields differ: "
            f"missing={sorted(reward_provider_fields - actual_reward_provider_fields)}, "
            f"unknown={sorted(actual_reward_provider_fields - reward_provider_fields)}"
        )
    if any(not isinstance(value, str) or not value for value in reward_provider.values()):
        raise ValueError("validation reward_provider identity values must be non-empty strings")
    identity_hash = reward_provider["identity_hash"]
    if len(identity_hash) != 64 or any(character not in "0123456789abcdef" for character in identity_hash):
        raise ValueError(
            f"validation reward_provider identity_hash must be a lowercase 64-character SHA-256, got {identity_hash!r}"
        )
    if reward_provider["name"] != reward:
        raise ValueError(
            f"validation reward_provider name {reward_provider['name']!r} does not match reward {reward!r}"
        )
    rollout_semantics = identity["rollout_semantics"]
    if not isinstance(rollout_semantics, dict) or not rollout_semantics:
        raise ValueError("validation rollout_semantics must be a non-empty dictionary")
    try:
        rollout_semantics = json.loads(
            json.dumps(rollout_semantics, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise TypeError("validation rollout_semantics must contain only finite JSON values") from error
    for field in ("seed", "batch_size"):
        if isinstance(identity[field], bool) or not isinstance(identity[field], int):
            raise ValueError(f"validation {field} must be an integer, got {identity[field]!r}")
    if identity["batch_size"] < 1:
        raise ValueError(f"validation batch_size must be >= 1, got {identity['batch_size']}")
    max_samples = identity["max_samples"]
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1
    ):
        raise ValueError(f"validation max_samples must be None or a positive integer, got {max_samples!r}")

    sampling = identity["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("validation sampling identity must be a dictionary")
    sampling_fields = {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "n_generation",
        "max_tokens",
    }
    if set(sampling) != sampling_fields:
        raise ValueError(
            f"validation sampling identity fields differ: missing={sorted(sampling_fields - set(sampling))}, "
            f"unknown={sorted(set(sampling) - sampling_fields)}"
        )
    for field in ("temperature", "repetition_penalty"):
        value = sampling[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"validation sampling {field} must be finite, got {value!r}")
    if float(sampling["temperature"]) < 0.0:
        raise ValueError(f"validation sampling temperature must be >= 0, got {sampling['temperature']!r}")
    if float(sampling["repetition_penalty"]) <= 0.0:
        raise ValueError(f"validation sampling repetition_penalty must be > 0, got {sampling['repetition_penalty']!r}")
    top_p = sampling["top_p"]
    if top_p is not None and (
        isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(float(top_p))
    ):
        raise ValueError(f"validation sampling top_p must be None or finite, got {top_p!r}")
    if top_p is not None and not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"validation sampling top_p must be in (0, 1], got {top_p!r}")
    for field in ("top_k", "n_generation", "max_tokens"):
        value = sampling[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"validation sampling {field} must be None or an integer, got {value!r}")
    if sampling["top_k"] is not None and sampling["top_k"] < 1:
        raise ValueError("validation sampling top_k must be >= 1 when set")
    if sampling["n_generation"] is None or sampling["n_generation"] < 1:
        raise ValueError(
            f"validation sampling n_generation must be a positive integer, got {sampling['n_generation']!r}"
        )
    if sampling["max_tokens"] is not None and sampling["max_tokens"] < 1:
        raise ValueError(f"validation sampling max_tokens must be positive when set, got {sampling['max_tokens']!r}")
    schedule = identity["schedule"]
    if not isinstance(schedule, dict) or set(schedule) != {
        "before_training",
        "every_steps",
        "train_total_steps",
    }:
        raise ValueError("validation schedule identity has an unsupported schema")
    if not isinstance(schedule["before_training"], bool):
        raise ValueError("validation schedule before_training must be bool")
    for field in ("every_steps", "train_total_steps"):
        value = schedule[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"validation schedule {field} must be a non-negative integer, got {value!r}")
    return {
        "enabled": True,
        "source_fingerprint": source_fingerprint,
        "rollout_semantics": rollout_semantics,
        "sampling": {
            "temperature": float(sampling["temperature"]),
            "top_p": None if top_p is None else float(top_p),
            "top_k": sampling["top_k"],
            "repetition_penalty": float(sampling["repetition_penalty"]),
            "n_generation": sampling["n_generation"],
            "max_tokens": sampling["max_tokens"],
        },
        "reward": reward,
        "reward_provider": {
            "name": reward_provider["name"],
            "revision": reward_provider["revision"],
            "catalog": reward_provider["catalog"],
            "identity_hash": reward_provider["identity_hash"],
        },
        "seed": identity["seed"],
        "batch_size": identity["batch_size"],
        "max_samples": max_samples,
        "schedule": dict(schedule),
    }


@dataclass(frozen=True)
class _SourceInventory:
    sample_counts: tuple[int, ...]
    fingerprint: str

    @property
    def total_samples(self) -> int:
        return sum(self.sample_counts)


@dataclass(frozen=True)
class RolloutRewardValidationRunner:
    """Run a complete finite held-out pass through data, rollout, and reward roles."""

    data: Any
    rollout: Any
    reward: Any
    validation: Any
    train_total_steps: int
    effective_max_tokens: int
    # Data-plane hook run after every batch, success or failure: releases producer-owned storage
    # (Ray/NIXL pool leases) that the rollout payload references. None when no such storage exists.
    release_payload: Callable[[Any], None] | None = None

    def _begin_source(self, batch_size: int) -> tuple[int, ...]:
        counts = tuple(int(value) for value in replica_lead_values(self.data.begin_validation(batch_size)))
        if not counts or any(count < 0 for count in counts) or sum(counts) < 1:
            raise RuntimeError(f"validation produced no global samples: sample_counts={list(counts)}")
        return counts

    def _source_fingerprint(self) -> str:
        values = tuple(str(value) for value in replica_lead_values(self.data.validation_source_fingerprint()))
        if not values or any(not value for value in values) or len(set(values)) != 1:
            raise RuntimeError(f"validation replicas disagree on source fingerprint: {list(values)}")
        return values[0]

    @staticmethod
    def _converge_inventory(inventory: _SourceInventory) -> None:
        by_rank = gather_rank_states(
            inventory,
            context="rollout/reward validation source inventory preparation",
        )
        if any(candidate != inventory for candidate in by_rank.values()):
            raise RuntimeError(f"validation source inventory differs across local ranks: {by_rank!r}")

    def _fetch_batch(
        self,
        inventory: _SourceInventory,
        batch_size: int,
        batch_index: int,
    ) -> Any:
        # Only integer inventory metadata is sent to each data replica; the prompt handle flows
        # directly into rollout, and every replica validates its own batch before publishing it.
        expected_counts = [
            min(batch_size, max(0, sample_count - batch_index * batch_size)) for sample_count in inventory.sample_counts
        ]
        return self.data.get_validation_samples(batch_size, expected_count=Scatter(expected_counts))

    def _score_batch(self, trajectories: Any, model_step: int) -> ValidationRewardStats:
        scored = self.reward.score_validation_with_report(
            trajectories,
            model_step=model_step,
            n_generation=int(self.validation.sampling.n_generation),
        )
        return resolve_validation_reward_report(self.reward.validation_batch_stats(), scored)

    def _discard(self, trajectories: Any, batch_index: int, *, failed: bool) -> None:
        label = "failed-batch cleanup" if failed else "payload cleanup"
        try:
            run_validation_stage(
                lambda: resolve_all(self.rollout.discard_payload(trajectories)),
                f"rollout/reward validation {label} {batch_index}",
            )
        finally:
            if self.release_payload is not None:
                run_validation_stage(
                    lambda: self.release_payload(trajectories),
                    f"rollout/reward validation lease release {batch_index}",
                )

    def _finalize(
        self,
        reports: list[ValidationRewardStats],
        inventory: _SourceInventory,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
        started: float,
    ) -> ValidationReport:
        expected_sample_ids = tuple(
            f"validation:{inventory.fingerprint}:{int(self.validation.seed)}:{sample_index}"
            for sample_index in range(inventory.total_samples)
        )
        merged = ValidationRewardStats.merge(reports, expected_sample_ids=expected_sample_ids)
        if merged.provider_provenance:
            logger.info(
                f"validation reward providers model_step={model_step} provenance={list(merged.provider_provenance)}"
            )
        expected_rollouts = inventory.total_samples * int(self.validation.sampling.n_generation)
        if merged.count != expected_rollouts:
            raise RuntimeError(
                f"validation rollout count {merged.count} != expected {expected_rollouts} "
                f"({inventory.total_samples} samples * {self.validation.sampling.n_generation} generations)"
            )
        return ValidationReport(
            model_step=model_step,
            sample_count=inventory.total_samples,
            metrics=merged.metrics(model_step=model_step, train_total_steps=self.train_total_steps),
            source_fingerprint=inventory.fingerprint,
            trigger_reasons=reasons,
            duration_s=time.monotonic() - started,
        )

    def run(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        """Evaluate one pinned model version and return a rank-consistent report."""
        report: ValidationReport | None = None
        validation_error: BaseException | None = None
        try:
            report = self._run_local(model_step, reasons)
        except BaseException as error:  # noqa: BLE001 -- every controller rank must leave together
            validation_error = error
        raise_if_any_rank_failed(validation_error, "rollout/reward held-out validation")
        if report is None:
            raise AssertionError("rollout/reward held-out validation converged without a report")
        return report

    def _run_local(
        self,
        model_step: int,
        reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        started = time.monotonic()
        batch_size = int(self.validation.batch_size)
        sample_counts = run_validation_stage(
            lambda: self._begin_source(batch_size),
            "rollout/reward validation source initialization",
        )
        fingerprint = run_validation_stage(
            self._source_fingerprint,
            "rollout/reward validation source fingerprint",
        )
        inventory = _SourceInventory(sample_counts=sample_counts, fingerprint=fingerprint)
        run_validation_stage(
            lambda: self._converge_inventory(inventory),
            "rollout/reward validation source inventory convergence",
        )
        context = run_validation_stage(
            lambda: build_validation_context(
                self.validation,
                model_step,
                effective_max_tokens=self.effective_max_tokens,
            ),
            "rollout/reward validation rollout context",
        )
        batch_count = max((count + batch_size - 1) // batch_size for count in sample_counts)
        reports: list[ValidationRewardStats] = []
        for batch_index in range(batch_count):
            prompts = run_validation_stage(
                lambda: self._fetch_batch(inventory, batch_size, batch_index),
                f"rollout/reward validation source batch {batch_index}",
            )
            trajectories = run_validation_stage(
                lambda: self.rollout.generate(prompts, validation=context),
                f"rollout/reward validation rollout batch {batch_index}",
            )
            try:
                report = run_validation_stage(
                    lambda: self._score_batch(trajectories, model_step),
                    f"rollout/reward validation reward batch {batch_index}",
                )
            except BaseException:
                try:
                    self._discard(trajectories, batch_index, failed=True)
                except BaseException:  # noqa: BLE001 -- preserve the primary validation failure
                    logger.exception("secondary failure while discarding a failed validation rollout batch")
                raise
            self._discard(trajectories, batch_index, failed=False)
            reports.append(report)
        report = run_validation_stage(
            lambda: self._finalize(reports, inventory, model_step, reasons, started),
            "rollout/reward validation report finalization",
        )
        return converge_validation_report(report)


@dataclass(frozen=True)
class RolloutRewardValidationPlan:
    """Resolved checkpoint identity and optional rollout/reward evaluator."""

    checkpoint_identity: dict[str, Any]
    runner: RolloutRewardValidationRunner | None
    sampling: dict[str, int | float | None] | None = None


def validate_reward_algorithm_compatibility(rl: Any, catalog: RewardProviderCatalog) -> None:
    """Reject score-producing providers whose training terms the selected objective cannot consume."""
    training_reward = str(rl.reward)
    training_spec = catalog.spec(training_reward)
    if getattr(rl, "distillation_mode", None) == "direct" and training_spec.is_remote:
        raise ValueError(
            f"direct OPD cannot use remote training reward provider {training_reward!r}: "
            "distillation_mode='direct' does not consume task reward advantage"
        )


def validate_rollout_reward_sources(rl: Any, catalog: RewardProviderCatalog | None = None) -> None:
    """Preflight selected providers and their sources before model/cluster construction.

    Constructing each distinct provider on the driver proves that the catalog's zero-argument
    factory returns the declared group-native online provider. Evaluation still belongs to the
    reward worker because preflight has no rollout group to score.
    """
    from cosmos_framework.posttrain.algorithm.reward import (
        resolve_reward_provider_catalog,
    )
    from cosmos_framework.posttrain.roles.env.prompt import prompt_source

    if catalog is None:
        catalog = resolve_reward_provider_catalog(str(getattr(rl, "reward_provider_catalog", "")))
    training_reward = str(rl.reward)
    validation = getattr(rl, "validation", None)
    validation_enabled = validation is not None and bool(getattr(validation, "enabled", False))
    validation_reward = str(getattr(validation, "reward", "") or rl.reward) if validation_enabled else training_reward
    validate_reward_algorithm_compatibility(rl, catalog)
    for provider_name in dict.fromkeys((training_reward, validation_reward)):
        try:
            catalog.build(provider_name)
        except Exception as error:  # noqa: BLE001 -- add the selected catalog key to arbitrary factory failures
            raise ValueError(
                f"reward provider {provider_name!r} failed driver preflight construction: {error}"
            ) from error

    validate_dataset_source(rl.data)
    _source, training_has_reference = prompt_source(rl)
    if "reference" in catalog.spec(training_reward).needs and not training_has_reference:
        raise ValueError(f"training reward {training_reward!r} requires ground truth; set rl.data.response_column")
    if not validation_enabled:
        return

    if "reference" in catalog.spec(validation_reward).needs and not validation.data.response_column:
        raise ValueError(
            f"rollout/reward validation reward {validation_reward!r} requires ground truth; "
            "set rl.validation.data.response_column"
        )
    validate_dataset_source(validation.data)


def rollout_validation_checkpoint_identity(
    validation: Any,
    *,
    training_reward: str,
    reward_provider_identity: dict[str, str],
    rollout_semantics: dict[str, Any] | None = None,
    source_fingerprint: str | None = None,
    effective_max_tokens: int | None = None,
    train_total_steps: int = 0,
) -> dict[str, Any]:
    """Build the canonical identity that makes a pending validation safe to resume."""
    if validation is None or not bool(getattr(validation, "enabled", False)):
        return {"enabled": False}
    if not source_fingerprint:
        raise ValueError("enabled rollout/reward validation requires a non-empty source fingerprint")
    expected_provider_fields = {"name", "revision", "catalog", "identity_hash"}
    if not isinstance(reward_provider_identity, dict) or set(reward_provider_identity) != expected_provider_fields:
        raise ValueError(
            "enabled rollout/reward validation requires reward_provider_identity with exact "
            f"fields {sorted(expected_provider_fields)}, got {reward_provider_identity!r}"
        )
    if any(not isinstance(value, str) or not value for value in reward_provider_identity.values()):
        raise ValueError(
            "enabled rollout/reward validation reward_provider_identity values must be non-empty strings, "
            f"got {reward_provider_identity!r}"
        )
    identity_hash = reward_provider_identity["identity_hash"]
    if len(identity_hash) != 64 or any(character not in "0123456789abcdef" for character in identity_hash):
        raise ValueError(
            "enabled rollout/reward validation reward_provider_identity identity_hash must be a lowercase "
            f"64-character SHA-256, got {identity_hash!r}"
        )
    if effective_max_tokens is None or effective_max_tokens < 1:
        raise ValueError("enabled rollout/reward validation requires a positive effective max_tokens")
    if not rollout_semantics:
        raise ValueError("enabled rollout/reward validation requires a non-empty rollout semantics identity")
    semantics = normalize_validation_semantics_identity(
        rollout_semantics,
        context="rollout/reward validation semantics",
    )
    sampling = resolved_validation_sampling(validation, effective_max_tokens=effective_max_tokens)
    return {
        "enabled": True,
        "source_fingerprint": source_fingerprint,
        "rollout_semantics": semantics,
        "sampling": sampling,
        "reward": str(getattr(validation, "reward", "") or training_reward),
        "reward_provider": dict(reward_provider_identity),
        "seed": int(validation.seed),
        "batch_size": int(validation.batch_size),
        "max_samples": None if validation.max_samples is None else int(validation.max_samples),
        "schedule": {
            "before_training": bool(validation.before_training),
            "every_steps": int(validation.every_steps),
            "train_total_steps": int(train_total_steps),
        },
    }


def prepare_rollout_validation(
    *,
    data: Any,
    rollout: Any,
    reward: Any,
    validation: Any,
    rollout_config: Any,
    training_reward: str,
    reward_provider_identity: dict[str, str],
    train_total_steps: int,
    release_payload: Callable[[Any], None] | None = None,
) -> RolloutRewardValidationPlan:
    """Resolve source, sampling, and rollout semantics before journaling validation."""
    if validation is None or not bool(getattr(validation, "enabled", False)):
        return RolloutRewardValidationPlan(checkpoint_identity={"enabled": False}, runner=None)

    setup_error: BaseException | None = None
    source_fingerprint: str | None = None
    effective_max_tokens: int | None = None
    sampling: dict[str, int | float | None] | None = None
    try:
        fingerprints = [str(value) for value in replica_lead_values(data.validation_source_fingerprint())]
        if not fingerprints or any(not value for value in fingerprints) or len(set(fingerprints)) != 1:
            raise RuntimeError(f"validation replicas disagree on source fingerprint: {fingerprints}")
        source_fingerprint = fingerprints[0]
        configured_max_tokens = validation.sampling.max_tokens
        effective_max_tokens = (
            int(rollout_config.max_tokens) if configured_max_tokens is None else int(configured_max_tokens)
        )
        sampling = resolved_validation_sampling(
            validation,
            effective_max_tokens=effective_max_tokens,
        )
        build_validation_context(validation, 0, effective_max_tokens=effective_max_tokens)
    except BaseException as error:  # noqa: BLE001 -- converge before rollout identity collectives
        setup_error = error
    raise_if_any_rank_failed(setup_error, "rollout/reward validation identity setup")
    if sampling is None or source_fingerprint is None or effective_max_tokens is None:
        raise AssertionError("rollout/reward validation identity setup converged without resolved semantics")
    rollout_semantics = resolve_rollout_validation_semantics(rollout)
    identity = rollout_validation_checkpoint_identity(
        validation,
        training_reward=training_reward,
        reward_provider_identity=reward_provider_identity,
        rollout_semantics=rollout_semantics,
        source_fingerprint=source_fingerprint,
        effective_max_tokens=effective_max_tokens,
        train_total_steps=train_total_steps,
    )
    return RolloutRewardValidationPlan(
        checkpoint_identity=identity,
        runner=RolloutRewardValidationRunner(
            data=data,
            rollout=rollout,
            reward=reward,
            validation=validation,
            train_total_steps=train_total_steps,
            effective_max_tokens=effective_max_tokens,
            release_payload=release_payload,
        ),
        sampling=sampling,
    )


def converge_validation_report(report: ValidationReport) -> ValidationReport:
    """Return one exact report across local torchrun ranks, keeping max duration."""
    reports_by_rank = gather_rank_states(report, context="rollout/reward validation report preparation")
    reports = list(reports_by_rank.values())
    first = reports[0]
    for rank, candidate in list(reports_by_rank.items())[1:]:
        identity = (
            candidate.model_step,
            candidate.sample_count,
            candidate.source_fingerprint,
            candidate.trigger_reasons,
            candidate.metrics.keys(),
        )
        expected_identity = (
            first.model_step,
            first.sample_count,
            first.source_fingerprint,
            first.trigger_reasons,
            first.metrics.keys(),
        )
        if identity != expected_identity:
            raise RuntimeError(f"rollout/reward validation report structure differs on rank {rank}: {candidate!r}")
        for key, expected in first.metrics.items():
            actual = candidate.metrics[key]
            if type(actual) is not type(expected):
                raise RuntimeError(
                    f"rollout/reward validation metric {key!r} type differs on rank {rank}: "
                    f"{type(actual).__name__} != {type(expected).__name__}"
                )
            equal = (
                math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
                if isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                raise RuntimeError(
                    f"rollout/reward validation metric {key!r} differs on rank {rank}: {actual!r} != {expected!r}"
                )
    return ValidationReport(
        model_step=first.model_step,
        sample_count=first.sample_count,
        metrics=dict(first.metrics),
        source_fingerprint=first.source_fingerprint,
        trigger_reasons=first.trigger_reasons,
        duration_s=max(float(candidate.duration_s) for candidate in reports),
    )


def episode_metrics(episodes: tuple[EvaluationEpisode, ...]) -> dict[str, int | float]:
    """Summarize environment-validated episodes with explicit numerators and denominators."""
    from cosmos_framework.posttrain.schema.metrics import WeightedMean

    if not episodes:
        raise ValueError("evaluation requires a nonempty episode inventory")
    metrics = {"val/episode_count": len(episodes)}
    for key, field in (("success_rate", "success"), ("episode_return", "reward"), ("episode_steps", "steps")):
        metrics[f"val/{key}"] = WeightedMean(sum(float(getattr(row, field)) for row in episodes), len(episodes)).value
    for group in sorted({row.group for row in episodes}):
        selected = [row for row in episodes if row.group == group]
        successes = sum(row.success for row in selected)
        metrics[f"val/{group}/success_count"] = successes
        metrics[f"val/{group}/success_rate"] = WeightedMean(successes, len(selected)).value
    return metrics


@dataclass
class WindowEvaluator:
    """Evaluate fixed environment cases without mutating learner or online collection state."""

    policy: WorkerGroup
    collector: WorkerGroup
    environment: WindowEnvironment
    replicas: int
    output: str
    simulator: WorkerGroup | None = None

    def __call__(self, step: int, reasons: frozenset[ValidationTrigger]) -> ValidationReport:
        from pathlib import Path

        from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
        from cosmos_framework.posttrain.control.loop.decision_window import evaluate_decision_cases
        from cosmos_framework.posttrain.utils.entry import resolve_identical

        started = time.monotonic()
        before = resolve_all(self.policy.evaluation_state())
        version = resolve_identical(self.policy.publication_version(), operation="evaluation publication")
        if self.simulator is None:
            raise RuntimeError("window evaluation requires a bound fixed-case simulator Role")
        shards = resolve_all(
            evaluate_decision_cases(
                self.collector,
                self.simulator,
                version=version,
                cases=Scatter(
                    [self.environment.evaluation_cases(replica, self.replicas) for replica in range(self.replicas)]
                ),
                seed=self.policy_validation_seed(),
                decisions=self.environment.evaluation_decisions,
            )
        )
        if any(
            shard["training_session_unchanged"] is not True or shard["policy_version"] != version for shard in shards
        ):
            raise RuntimeError("evaluation changed collection state or used a different publication")
        if resolve_all(self.policy.evaluation_state()) != before:
            raise RuntimeError("evaluation changed learner parameters, optimizer, scheduler, RNG or counters")
        episodes = self.environment.validate_evaluation([row for shard in shards for row in shard["episodes"]])
        metrics = episode_metrics(episodes)
        fingerprint = self.environment.evaluation_fingerprint()
        result = {
            "optimizer_step": step,
            "publication_version": version,
            "manifest": fingerprint,
            "training_state_unchanged": True,
            "learner_state": before,
            "metrics": metrics,
            "episodes": [dict(row.details) for row in episodes],
            "reasons": sorted(reason.value for reason in reasons),
        }
        directory = Path(self.output) / "evaluation"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(directory / f"step_{step:06d}.json"), result)
        return ValidationReport(step, len(episodes), metrics, fingerprint, reasons, time.monotonic() - started)

    def policy_validation_seed(self) -> int:
        from cosmos_framework.posttrain.utils.entry import resolve_identical

        return resolve_identical(self.collector.evaluation_seed(), operation="window evaluation seed")
