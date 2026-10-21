# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reward providers, combination, and group annotation for one worker."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from cosmos_framework.posttrain.algorithm.reward import (
    BoundRewardProvider,
    RewardCombiner,
    RewardContractError,
    RewardGroupRequest,
)
from cosmos_framework.posttrain.algorithm.reward.contract import _completion_action_token_count
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult
from cosmos_framework.posttrain.roles.reward.diagnostics import (
    ProviderDiagnosticRecord,
    ProviderMetricReport,
    provider_diagnostics_report,
)
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Trajectory, Transition
from cosmos_framework.posttrain.schema.metrics import MetricReport
from cosmos_framework.posttrain.utils.batch_stats import (
    IdleRewardReport,
    RewardBatchStats,
    ValidationRewardStats,
    reward_batch_stats,
)

IndexedGroup = tuple[str, tuple[tuple[int, Trajectory], ...]]
PreparedGroup = tuple[BoundRewardProvider, tuple[tuple[int, Trajectory], ...], RewardGroupRequest]
_RewardRequestEnricher = Callable[[RewardGroupRequest, tuple[Trajectory, ...]], RewardGroupRequest]
FailureConvergence = Callable[[Exception | None, str], None]


@dataclass(frozen=True)
class _TransitionRewardState:
    """The small, inline part of a transition that the reward stage is allowed to mutate."""

    transition: Transition
    reward: Any
    reward_values: dict[str, float] | None
    advantage: Any
    algo_extra: dict[str, Any]
    algo_extra_values: dict[str, Any]


def _snapshot_reward_state(trajectories: Sequence[Trajectory]) -> tuple[_TransitionRewardState, ...]:
    """Retain reward-owned containers and values without copying tensor or bulk payloads."""
    return tuple(
        _TransitionRewardState(
            transition=transition,
            reward=transition.reward,
            reward_values=None if transition.reward is None else dict(transition.reward),
            advantage=transition.advantage,
            algo_extra=transition.algo_extra,
            algo_extra_values=dict(transition.algo_extra),
        )
        for trajectory in trajectories
        for transition in trajectory.transitions
    )


def _restore_reward_state(states: Sequence[_TransitionRewardState]) -> None:
    """Restore a failed reward transaction while preserving original mapping identities."""
    for state in states:
        if state.reward is None:
            state.transition.reward = None
        elif not isinstance(state.reward, dict):
            # Failures before commit must also restore deliberately immutable
            # input mappings without assuming they implement dict mutation.
            state.transition.reward = state.reward
        else:
            state.reward.clear()
            state.reward.update(state.reward_values or {})
            state.transition.reward = state.reward
        state.transition.advantage = state.advantage
        state.algo_extra.clear()
        state.algo_extra.update(state.algo_extra_values)
        state.transition.algo_extra = state.algo_extra


@contextmanager
def reward_state_transaction(trajectories: Sequence[Trajectory]) -> Iterator[None]:
    """Roll back reward, advantage, and annotation metadata if reward-stage work fails.

    The snapshot is deliberately shallow. Reward dictionaries contain scalars, while advantage
    and ``algo_extra`` may refer to tensors or bulk handles that must not be cloned. Reward-stage
    annotators own assignment and framework-defined top-level metadata writes, so retaining those
    object references and mapping contents is sufficient to restore the schema fields they are
    allowed to mutate.
    """
    states = _snapshot_reward_state(trajectories)
    try:
        yield
    except BaseException:
        _restore_reward_state(states)
        raise


class RewardWorker:
    # Held-out scoring consumes inline metadata and leaves rollout-owned bulk tensors in place.
    metadata_only_input_methods: frozenset[str] = frozenset({"score_validation_with_report"})

    def __init__(
        self,
        providers: Sequence[BoundRewardProvider],
        combiner: RewardCombiner | None = None,
        annotators: Sequence[Callable[[list[Trajectory]], None]] = (),
        validation_providers: Sequence[BoundRewardProvider] | None = None,
        *,
        training_group_size: int,
        failure_convergence: FailureConvergence | None = None,
    ) -> None:
        """Bind training providers, whole-group annotators, and optional validation providers."""
        self.providers = self._validate_provider_inventory(providers, field="training providers")
        if validation_providers is not None and not validation_providers:
            raise RewardContractError("explicit validation_providers must not be empty")
        self.validation_providers = (
            self._validate_provider_inventory(validation_providers, field="validation providers")
            if validation_providers is not None
            else self.providers
        )
        self.combiner = combiner or RewardCombiner()  # identity by default
        self.annotators = tuple(annotators)
        if isinstance(training_group_size, bool) or not isinstance(training_group_size, int) or training_group_size < 1:
            raise RewardContractError(f"training_group_size must be a positive integer, got {training_group_size!r}")
        self.training_group_size = training_group_size
        self._trusted_request_enricher: _RewardRequestEnricher | None = None
        self.failure_convergence = failure_convergence
        self._last_batch_stats: RewardBatchStats | IdleRewardReport | None = None
        self._last_validation_stats: ValidationRewardStats | None = None
        self._last_provider_report: ProviderMetricReport | None = None

    @staticmethod
    def _validate_provider_inventory(
        providers: Sequence[BoundRewardProvider],
        *,
        field: str,
    ) -> list[BoundRewardProvider]:
        """Require one canonical catalog-bound provider inventory."""
        inventory = list(providers)
        if not inventory:
            raise RewardContractError(f"reward worker requires at least one {field[:-1]}")
        for index, provider in enumerate(inventory):
            if not isinstance(provider, BoundRewardProvider):
                raise RewardContractError(
                    f"reward worker {field} must contain catalog-bound BoundRewardProvider values; "
                    f"item {index} is {type(provider).__name__}"
                )
        provider_names: set[str] = set()
        for provider in inventory:
            if provider.provider_name in provider_names:
                raise RewardContractError(f"reward worker {field}: duplicate provider name {provider.provider_name!r}")
            provider_names.add(provider.provider_name)
        term_owner: dict[str, str] = {}
        for provider in inventory:
            for term_name in (*provider.term_names, *provider.token_term_names):
                previous_owner = term_owner.get(term_name)
                if previous_owner is not None:
                    raise RewardContractError(
                        f"reward worker {field}: duplicate optimized term {term_name!r} from "
                        f"providers {previous_owner!r} and {provider.provider_name!r}"
                    )
                term_owner[term_name] = provider.provider_name
        return inventory

    def score(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Score and annotate trajectories without producing a diagnostics report."""
        self._last_batch_stats = None
        self._last_validation_stats = None
        self._last_provider_report = None
        with reward_state_transaction(trajectories):
            scored, _provider_report, _provider_provenance = self._score(
                trajectories,
                self.providers,
                self.annotators,
                diagnostic_namespace=None,
                expected_group_size=self.training_group_size,
            )
        return scored

    def score_with_provider_report(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Score one local shard and retain provider diagnostics without group batch statistics.

        Distributed WFM owns a complete reward group only after gathering one scalar reward from
        every model rank. This path therefore publishes provider diagnostics and provenance without
        claiming the rank-local trajectory list is a complete group-reporting batch.
        """
        self._last_batch_stats = None
        self._last_validation_stats = None
        self._last_provider_report = None
        with reward_state_transaction(trajectories):
            scored, provider_report, provider_provenance = self._score(
                trajectories,
                self.providers,
                self.annotators,
                diagnostic_namespace="train",
                expected_group_size=self.training_group_size,
            )
            self._last_provider_report = ProviderMetricReport(provider_report, provider_provenance)
        return scored

    def configured_group_size(self) -> int:
        """Return the one training-group cardinality bound into this reward body."""
        return self.training_group_size

    def _prepare_provider_group(
        self,
        provider: BoundRewardProvider,
        members: tuple[tuple[int, Trajectory], ...],
    ) -> RewardGroupRequest:
        """Build the canonical request before optional framework-owned local enrichment."""
        trajectories = tuple(trajectory for _index, trajectory in members)
        request = provider.prepare_group(trajectories)
        if self._trusted_request_enricher is None:
            return request
        prepared = self._trusted_request_enricher(request, trajectories)
        if not isinstance(prepared, RewardGroupRequest) or type(prepared) is RewardGroupRequest:
            raise RewardContractError(
                f"trusted request enricher returned {type(prepared).__name__}; expected a RewardGroupRequest subclass"
            )
        if prepared.group_id != request.group_id:
            raise RewardContractError("trusted request enricher changed the canonical group request")
        if prepared.candidates != request.candidates or prepared.context != request.context:
            raise RewardContractError("trusted request enricher must retain canonical group fields")
        return prepared

    @staticmethod
    def _index_groups(trajectories: Sequence[Trajectory], expected_group_size: int) -> tuple[IndexedGroup, ...]:
        """Validate and index a complete group batch without invoking a provider."""
        if isinstance(expected_group_size, bool) or not isinstance(expected_group_size, int) or expected_group_size < 1:
            raise RewardContractError(f"expected group size must be a positive integer, got {expected_group_size!r}")

        grouped: dict[str, list[tuple[int, Trajectory]]] = {}
        trajectory_ids: set[str] = set()
        for index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory.id, str) or not trajectory.id.strip():
                raise RewardContractError(
                    f"online reward requires a non-empty trajectory id at batch index {index}, got {trajectory.id!r}"
                )
            if trajectory.id in trajectory_ids:
                raise RewardContractError(f"online reward found duplicate trajectory id {trajectory.id!r}")
            trajectory_ids.add(trajectory.id)

            group_id = trajectory.group_id
            if not isinstance(group_id, str) or not group_id.strip():
                raise RewardContractError(
                    f"online reward requires a non-empty group_id on trajectory {trajectory.id!r}, got {group_id!r}"
                )
            obs_id = trajectory.transitions[-1].obs.id
            if obs_id != group_id:
                raise RewardContractError(
                    f"online reward trajectory {trajectory.id!r} has group_id={group_id!r} "
                    f"but terminal obs.id={obs_id!r}"
                )
            grouped.setdefault(group_id, []).append((index, trajectory))

        for group_id, members in grouped.items():
            if len(members) != expected_group_size:
                raise RewardContractError(
                    f"online reward expected {expected_group_size} trajectories for group {group_id!r}, "
                    f"got {len(members)}"
                )
        return tuple((group_id, tuple(members)) for group_id, members in grouped.items())

    @staticmethod
    def _record_result(
        *,
        provider: BoundRewardProvider,
        trajectory: Trajectory,
        result: RewardProviderResult,
        components: dict[str, float],
        token_components: dict[str, tuple[float, ...]],
        diagnostic_records: list[ProviderDiagnosticRecord],
    ) -> None:
        """Stage one catalog-validated candidate result without mutating its trajectory."""
        existing_term_names = {*components, *token_components}
        incoming_term_names = {*result.terms, *result.token_terms}
        duplicate_terms = existing_term_names.intersection(incoming_term_names)
        if duplicate_terms:
            raise RewardContractError(
                f"reward providers emitted duplicate reward term(s) {sorted(duplicate_terms)} "
                f"across scalar/token namespaces on trajectory {trajectory.id!r}"
            )
        components.update(result.terms)  # named optimized components only (§1.5)
        if result.token_terms:
            action_token_count = _completion_action_token_count(trajectory)
            for name, values in result.token_terms.items():
                if len(values) != action_token_count:
                    raise RewardContractError(
                        f"reward token term {name!r} has width {len(values)} but trajectory "
                        f"{trajectory.id!r} has action token count {action_token_count}"
                    )
            token_components.update(result.token_terms)
        diagnostic_records.append(
            ProviderDiagnosticRecord(
                provider_name=provider.provider_name,
                provider_revision=provider.provider_revision,
                diagnostics=result.diagnostics,
            )
        )

    def _score(
        self,
        trajectories: list[Trajectory],
        providers: Sequence[BoundRewardProvider],
        annotators: Sequence[Callable[[list[Trajectory]], None]],
        *,
        diagnostic_namespace: str | None,
        expected_group_size: int,
    ) -> tuple[list[Trajectory], MetricReport, tuple[tuple[str, str], ...]]:
        planned: list[tuple[Trajectory, dict[str, float], dict[str, tuple[float, ...]], float]] | None = None
        provider_report: MetricReport | None = None
        provider_provenance: tuple[tuple[str, str], ...] | None = None
        local_error: Exception | None = None
        try:
            for trajectory in trajectories:
                if not trajectory.transitions:
                    raise RewardContractError(
                        f"reward worker cannot score trajectory {trajectory.id!r} without a terminal transition"
                    )

            indexed_groups = self._index_groups(trajectories, expected_group_size)
            prepared_groups: tuple[PreparedGroup, ...] = tuple(
                (
                    provider,
                    members,
                    self._prepare_provider_group(provider, members),
                )
                for provider in providers
                for _group_id, members in indexed_groups
            )

            components_by_trajectory: list[dict[str, float]] = [{} for _trajectory in trajectories]
            token_components_by_trajectory: list[dict[str, tuple[float, ...]]] = [{} for _trajectory in trajectories]
            diagnostic_records: list[ProviderDiagnosticRecord] = []
            for provider, members, request in prepared_groups:
                member_indices = {trajectory.id: index for index, trajectory in members}
                group_result = provider.evaluate_prepared_group(request)
                for item in group_result.items:
                    index = member_indices[item.trajectory_id]
                    self._record_result(
                        provider=provider,
                        trajectory=trajectories[index],
                        result=item.result,
                        components=components_by_trajectory[index],
                        token_components=token_components_by_trajectory[index],
                        diagnostic_records=diagnostic_records,
                    )

            planned = [
                (trajectory, components, token_components, self.combiner.combine(trajectory, components))
                for trajectory, components, token_components in zip(
                    trajectories,
                    components_by_trajectory,
                    token_components_by_trajectory,
                    strict=True,
                )
            ]
            if diagnostic_namespace is None:
                provider_report = {}
                provider_provenance = ()
            else:
                provider_report, provider_provenance = provider_diagnostics_report(
                    diagnostic_records,
                    namespace=diagnostic_namespace,
                )
        except Exception as error:  # noqa: BLE001 -- converge before distributed annotation
            local_error = error
        if self.failure_convergence is not None:
            self.failure_convergence(local_error, "reward provider and combiner planning")
        if local_error is not None:
            # A malformed callback that returns despite this rank's failure cannot publish rewards.
            raise local_error
        if planned is None or provider_report is None or provider_provenance is None:
            raise RuntimeError("reward planning converged without a complete result")
        # Provider/checker failures abort before any member of the batch receives a reward. This
        # keeps a malformed routed row from leaving a plausible partially-scored batch behind.
        for traj, components, token_components, final in planned:
            traj.transitions[-1].reward = {**components, **token_components, FINAL_REWARD_KEY: final}
        for annotate in annotators:
            # Group-relative annotations run after every final reward is available.
            annotate(trajectories)
        return trajectories, provider_report, provider_provenance

    def score_with_report(
        self, trajectories: list[Trajectory], max_previews: int = 2, *, idle: bool = False
    ) -> list[Trajectory]:
        """Score trajectories, or acknowledge an explicitly idle, empty refill cell.

        Idle cells skip providers and annotators, but still produce a one-shot control report.
        An empty active batch remains a contract error.
        """
        # Invalidate first so a failed provider/annotator/report can never expose the previous
        # batch's report. The Ray host is single-threaded and preserves this driver's actor-task
        # order: score_with_report completes before the following batch_stats call on every cell.
        self._last_batch_stats = None
        self._last_validation_stats = None
        self._last_provider_report = None
        if not isinstance(idle, bool):
            raise TypeError(f"idle must be a bool, got {idle!r}")
        if idle:
            if trajectories:
                raise RewardContractError("an idle reward cell must receive no trajectories")
            self._last_batch_stats = IdleRewardReport()
            return []
        with reward_state_transaction(trajectories):
            scored, provider_report, provider_provenance = self._score(
                trajectories,
                self.providers,
                self.annotators,
                diagnostic_namespace="train",
                expected_group_size=self.training_group_size,
            )
            self._last_batch_stats = reward_batch_stats(
                scored,
                max_previews=max_previews,
                provider_report=provider_report,
                provider_provenance=provider_provenance,
            )
        return scored

    def score_validation_with_report(
        self,
        trajectories: list[Trajectory],
        *,
        model_step: int,
        n_generation: int,
    ) -> None:
        """Score held-out trajectories and retain exact stats, without returning tensor payloads.

        The following :meth:`validation_batch_stats` RPC is the terminal, bounded validation
        result. Returning the trajectories here would re-externalize them into the reward worker's
        BulkStore and leak that copy because validation has no replay/trainer consumer.
        """
        self._last_batch_stats = None
        self._last_validation_stats = None
        self._last_provider_report = None
        with reward_state_transaction(trajectories):
            scored, provider_report, provider_provenance = self._score(
                trajectories,
                self.validation_providers,
                (),
                diagnostic_namespace="val",
                expected_group_size=n_generation,
            )
            self._last_validation_stats = ValidationRewardStats.from_trajectories(
                scored,
                model_step=model_step,
                n_generation=n_generation,
                provider_report=provider_report,
                provider_provenance=provider_provenance,
            )

    def batch_stats(self) -> RewardBatchStats | IdleRewardReport:
        """Consume the report from the most recent successful :meth:`score_with_report` call.

        Returns:
            Reward/group diagnostics, or an explicit successful idle-cell acknowledgement.

        Raises:
            RuntimeError: No report-producing score completed since construction or plain
                :meth:`score` invalidated the previous report.
        """
        if self._last_batch_stats is None:
            raise RuntimeError("no completed reward report; score_with_report must succeed first")
        report = self._last_batch_stats
        self._last_batch_stats = None
        return report

    def validation_batch_stats(self) -> ValidationRewardStats:
        """Consume the exact report from the most recent successful held-out score."""
        if self._last_validation_stats is None:
            raise RuntimeError("no completed validation reward report; score_validation_with_report must succeed first")
        report = self._last_validation_stats
        self._last_validation_stats = None
        return report

    def provider_report(self) -> ProviderMetricReport:
        """Consume provider-only diagnostics from the latest successful provider-report score."""
        if self._last_provider_report is None:
            raise RuntimeError("no completed provider report; score_with_provider_report must succeed first")
        report = self._last_provider_report
        self._last_provider_report = None
        return report

    def annotation_metrics(self) -> dict[str, int | float]:
        """Return collision-free metrics declared by the latest successful annotator pass."""
        merged: dict[str, int | float] = {}
        for annotator in self.annotators:
            metrics = getattr(annotator, "metrics", None)
            if not callable(metrics):
                continue
            values = metrics()
            if not isinstance(values, dict):
                raise TypeError(
                    f"reward annotator {type(annotator).__name__}.metrics() must return a dict, "
                    f"got {type(values).__name__}"
                )
            overlap = set(merged).intersection(values)
            if overlap:
                raise ValueError(f"reward annotator metric keys collide: {sorted(overlap)}")
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"reward annotator metric keys must be non-empty strings, got {key!r}")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"reward annotator metric {key!r} must be a numeric scalar, got {value!r}")
                merged[key] = value
        return merged


def _with_trusted_request_enricher(
    worker: RewardWorker,
    request_enricher: _RewardRequestEnricher,
) -> RewardWorker:
    """Install one framework-owned local request adapter without widening the Worker API."""
    if not isinstance(worker, RewardWorker):
        raise TypeError(f"worker must be RewardWorker, got {type(worker).__name__}")
    if not callable(request_enricher):
        raise TypeError(f"_trusted_request_enricher must be callable, got {request_enricher!r}")
    if worker._trusted_request_enricher is not None:
        raise RewardContractError("trusted local request enrichment is already installed")
    if any(provider.is_remote for provider in (*worker.providers, *worker.validation_providers)):
        raise RewardContractError("trusted local request enrichment cannot be used with remote reward providers")
    worker._trusted_request_enricher = request_enricher
    return worker
