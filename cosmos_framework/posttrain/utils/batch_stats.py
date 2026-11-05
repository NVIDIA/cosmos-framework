# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict reward-boundary reporting over complete, scored GRPO groups.

The reward role is the last place that provably holds each group intact. It therefore owns the
facts that explain whether a GRPO batch can teach anything: reward range, tied versus informative
groups, and degenerate all-zero/all-one groups. The controller receives this small immutable report;
it does not pull every scored trajectory merely to reconstruct the same facts.

These are synchronous validation diagnostics, not training inputs or accept/drop decisions. They
are nevertheless strict: a missing reward or group annotation is a broken scored batch, not a
cosmetic logging failure. In particular, genuine ``0.0`` remains distinguishable from an absent
value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isclose, isfinite
from numbers import Real

from cosmos_framework.posttrain.algorithm.annotators.advantage import (
    GROUP_REWARD_MEAN,
    GROUP_REWARD_SPREAD,
    is_informative,
)
from cosmos_framework.posttrain.algorithm.reward import RewardContractError
from cosmos_framework.posttrain.algorithm.reward.contract import _terminal_reward
from cosmos_framework.posttrain.comm.transfer.mover import BulkRef
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Trajectory
from cosmos_framework.posttrain.schema.metrics import (
    FinalMetricReport,
    MetricContribution,
    MetricReport,
    MomentOutputs,
    Moments,
    finalize_metric_report,
    merge_metric_reports,
)


@dataclass(frozen=True)
class IdleRewardReport:
    """Successful, deliberately idle reward cell; never a zero-sized reward batch."""


@dataclass(frozen=True)
class RewardSamplePreview:
    """One bounded, text-only sample carried with a reward validation report."""

    trajectory_id: str
    reward: float
    reference_tail: str
    completion: str


@dataclass(frozen=True)
class RewardBatchStats:
    """Mergeable sufficient statistics for one or more complete reward-role batches."""

    trajectories: int
    reward_sum: float
    reward_m2: float
    reward_min: float
    reward_max: float
    groups: int
    informative_groups: int
    tied_groups: int
    all_zero_groups: int
    all_one_groups: int
    previews: tuple[RewardSamplePreview, ...] = ()
    completion_length_moments: Moments | None = None
    advantage_moments: Moments | None = None
    provider_report: MetricReport = field(default_factory=dict)
    provider_provenance: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def _completion_length(trajectory: Trajectory) -> int | None:
        """Read the rollout-stamped token count without materializing lazy action bulk.

        Older/local producers may not stamp the lightweight metadata yet, so an inline sequence or
        one-dimensional tensor remains a safe fallback. A ``BulkRef`` is deliberately opaque here:
        the reward role does not request action bulk, and diagnostics must never pull or iterate it.
        """
        if not trajectory.transitions:
            return None
        policy_info = trajectory.transitions[0].policy_info
        if "completion_token_count" in policy_info:
            count = policy_info["completion_token_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise RewardContractError(
                    "GRPO rollout diagnostics need non-negative integer "
                    f"policy_info['completion_token_count'], got {count!r} on trajectory {trajectory.id!r}"
                )
            return count
        action = trajectory.transitions[0].action
        if action is None or isinstance(action, (str, bytes, BulkRef)):
            return None
        if isinstance(action, Sequence):
            return len(action)
        shape = getattr(action, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
        return None

    def __post_init__(self) -> None:
        # Constructing Moments is the single validation path for finite reward sufficient stats.
        # Counts/group invariants stay strict because this report crosses a process boundary and a
        # malformed summary cannot be repaired by its controller-side merge.
        _ = self.reward_moments
        group_counts = (
            self.groups,
            self.informative_groups,
            self.tied_groups,
            self.all_zero_groups,
            self.all_one_groups,
        )
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in group_counts):
            raise ValueError(f"reward group counts must be non-negative integers, got {group_counts!r}")
        if self.informative_groups + self.tied_groups != self.groups:
            raise ValueError(
                "informative_groups + tied_groups must equal groups, got "
                f"{self.informative_groups} + {self.tied_groups} != {self.groups}"
            )
        if self.all_zero_groups + self.all_one_groups > self.tied_groups:
            raise ValueError(
                "all-zero and all-one groups must be disjoint subsets of tied groups, got "
                f"{self.all_zero_groups} + {self.all_one_groups} > {self.tied_groups}"
            )
        object.__setattr__(self, "provider_report", merge_metric_reports([self.provider_report]))
        object.__setattr__(
            self,
            "provider_provenance",
            _normalized_provider_provenance(self.provider_provenance),
        )

    @property
    def reward_moments(self) -> Moments:
        """The exact mergeable reward distribution represented by this report."""
        if isinstance(self.trajectories, bool) or not isinstance(self.trajectories, int) or self.trajectories <= 0:
            raise ValueError("reward batch statistics require a positive integer trajectory count")
        return Moments(
            count=self.trajectories,
            mean=self.reward_sum / self.trajectories,
            m2=self.reward_m2,
            minimum=self.reward_min,
            maximum=self.reward_max,
        )

    @property
    def mean_reward(self) -> float:
        """Sample-weighted mean reward."""
        return self.reward_moments.mean

    @property
    def reward_std(self) -> float:
        """Population reward standard deviation (``ddof=0``)."""
        return self.reward_moments.population_std

    def metric_report(self) -> MetricReport:
        """Return typed schema-v2 contributions without finalizing the reward moments."""
        report = {
            "train/reward": MetricContribution.moments(self.reward_moments),
            "train/rollout_count": MetricContribution.sum(self.trajectories),
            "train/grpo_group_count": MetricContribution.sum(self.groups),
            "train/grpo_informative_group_count": MetricContribution.sum(self.informative_groups),
            "train/grpo_tied_group_count": MetricContribution.sum(self.tied_groups),
            "train/grpo_all_zero_group_count": MetricContribution.sum(self.all_zero_groups),
            "train/grpo_all_one_group_count": MetricContribution.sum(self.all_one_groups),
        }
        if self.completion_length_moments is not None:
            report["rollout/completion_length"] = MetricContribution.moments(self.completion_length_moments)
        if self.advantage_moments is not None:
            report["rollout/advantage"] = MetricContribution.moments(self.advantage_moments)
        return merge_metric_reports([report, self.provider_report])

    def metrics(self) -> FinalMetricReport:
        """Finalize canonical training reward and GRPO group diagnostics."""
        moment_outputs = {
            "train/reward": MomentOutputs(
                mean="train/reward_mean",
                std="train/reward_std",
                minimum="train/reward_min",
                maximum="train/reward_max",
            )
        }
        if self.completion_length_moments is not None:
            moment_outputs["rollout/completion_length"] = MomentOutputs(
                mean="rollout/completion_length_mean",
                std="rollout/completion_length_std",
                minimum="rollout/completion_length_min",
                maximum="rollout/completion_length_max",
            )
        if self.advantage_moments is not None:
            moment_outputs["rollout/advantage"] = MomentOutputs(
                mean="rollout/advantage_mean",
                std="rollout/advantage_std",
                minimum="rollout/advantage_min",
                maximum="rollout/advantage_max",
            )
        return finalize_metric_report(
            self.metric_report(),
            moment_outputs=moment_outputs,
        )

    @classmethod
    def merge(cls, reports: Sequence[RewardBatchStats], max_previews: int = 2) -> RewardBatchStats:
        """Merge per-replica reports without averaging replica means.

        Args:
            reports: Non-empty per-replica or per-microbatch reports.
            max_previews: Maximum previews retained across the merged report.

        Returns:
            One sample-weighted report spanning every input.

        Raises:
            ValueError: No reports were supplied or ``max_previews`` is negative.
        """
        if not reports:
            raise ValueError("cannot merge an empty reward report inventory")
        if max_previews < 0:
            raise ValueError(f"max_previews must be >= 0, got {max_previews}")
        if any(report.trajectories <= 0 for report in reports):
            raise ValueError("cannot merge a reward report with no trajectories")
        previews: list[RewardSamplePreview] = []
        preview_index = 0
        while len(previews) < max_previews:
            appended = False
            for report in reports:
                if preview_index < len(report.previews):
                    previews.append(report.previews[preview_index])
                    appended = True
                    if len(previews) == max_previews:
                        break
            if not appended:
                break
            preview_index += 1

        reward_moments = Moments.merge_many([report.reward_moments for report in reports])
        completion_length_moments = (
            Moments.merge_many(
                [report.completion_length_moments for report in reports if report.completion_length_moments is not None]
            )
            if all(report.completion_length_moments is not None for report in reports)
            else None
        )
        advantage_moments = (
            Moments.merge_many([report.advantage_moments for report in reports if report.advantage_moments is not None])
            if all(report.advantage_moments is not None for report in reports)
            else None
        )
        assert reward_moments.minimum is not None and reward_moments.maximum is not None
        return cls(
            trajectories=reward_moments.count,
            reward_sum=reward_moments.mean * reward_moments.count,
            reward_m2=reward_moments.m2,
            reward_min=reward_moments.minimum,
            reward_max=reward_moments.maximum,
            groups=sum(report.groups for report in reports),
            informative_groups=sum(report.informative_groups for report in reports),
            tied_groups=sum(report.tied_groups for report in reports),
            all_zero_groups=sum(report.all_zero_groups for report in reports),
            all_one_groups=sum(report.all_one_groups for report in reports),
            # Round-robin by report so a bounded controller log does not always show replica 0.
            previews=tuple(previews),
            completion_length_moments=completion_length_moments,
            advantage_moments=advantage_moments,
            provider_report=merge_metric_reports([report.provider_report for report in reports]),
            provider_provenance=_merge_provider_provenance([report.provider_provenance for report in reports]),
        )


def _normalized_provider_provenance(
    provenance: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Validate provider revisions and reject conflicts across reports."""
    normalized: set[tuple[str, str]] = set()
    revisions: dict[str, str] = {}
    for item in provenance:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
        ):
            raise ValueError(
                f"reward provider provenance must contain non-empty (provider, revision) strings, got {item!r}"
            )
        provider, revision = item
        previous = revisions.get(provider)
        if previous is not None and previous != revision:
            raise ValueError(f"reward provider {provider!r} changed revision from {previous!r} to {revision!r}")
        revisions[provider] = revision
        normalized.add(item)
    return tuple(sorted(normalized))


def _merge_provider_provenance(
    reports: Sequence[Sequence[tuple[str, str]]],
) -> tuple[tuple[str, str], ...]:
    """Union provider provenance across batches and replicas."""
    return _normalized_provider_provenance([item for report in reports for item in report])


def _finite_annotation(traj: Trajectory, key: str) -> float:
    """Read one finite numeric group annotation from the terminal transition."""
    value = traj.transitions[-1].algo_extra.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RewardContractError(
            f"GRPO reward report needs terminal algo_extra[{key!r}] from the group-relative estimator, "
            f"got {value!r} on trajectory {traj.id!r}"
        )
    result = float(value)
    if not isfinite(result):
        raise RewardContractError(
            f"GRPO reward report needs finite terminal algo_extra[{key!r}], got {result!r} on trajectory {traj.id!r}"
        )
    return result


def _preview(traj: Trajectory, reward: float) -> RewardSamplePreview:
    """Build a bounded preview without retaining tensors or the trajectory object."""
    # Preserve the original GRPO sample-log meaning for a future multi-transition trajectory:
    # preview the first model response and its source observation while reporting terminal reward.
    transition = traj.transitions[0]
    completion = transition.policy_info.get("completion_text", "")
    completion_text = completion if isinstance(completion, str) else repr(completion)
    reference = transition.obs.representations.get("reference", "")
    return RewardSamplePreview(
        trajectory_id=traj.id,
        reward=reward,
        reference_tail=str(reference)[-30:].replace("\n", " "),
        completion=completion_text[:220].replace("\n", " "),
    )


def reward_batch_stats(
    scored: list[Trajectory],
    max_previews: int = 2,
    *,
    provider_report: MetricReport | None = None,
    provider_provenance: Sequence[tuple[str, str]] = (),
) -> RewardBatchStats:
    """Validate and summarize a group-preserving scored batch at the reward boundary.

    Args:
        scored: Trajectories after reward providers and the group-relative annotator ran.
        max_previews: Maximum bounded sample previews returned to the controller.

    Returns:
        A mergeable reward report.

    Raises:
        RewardContractError: The batch is empty, a reward/group id/annotation is missing, or the
            stamped group facts disagree with the actual complete group.
        ValueError: ``max_previews`` is negative.
    """
    if not scored:
        raise RewardContractError("GRPO reward report needs a non-empty scored batch")
    if max_previews < 0:
        raise ValueError(f"max_previews must be >= 0, got {max_previews}")

    grouped: dict[str, list[tuple[Trajectory, float, float, float]]] = {}
    rewards: list[float] = []
    completion_lengths: list[float] = []
    token_advantages: list[float] = []
    token_metrics_available = True
    previews: list[RewardSamplePreview] = []
    for traj in scored:
        reward = _terminal_reward(traj)
        if traj.group_id is None:
            raise RewardContractError(
                f"GRPO reward report needs group_id on trajectory {traj.id!r}; "
                "the group-relative estimator should have rejected this batch"
            )
        spread = _finite_annotation(traj, GROUP_REWARD_SPREAD)
        mean = _finite_annotation(traj, GROUP_REWARD_MEAN)
        grouped.setdefault(traj.group_id, []).append((traj, reward, spread, mean))
        rewards.append(reward)
        completion_length = RewardBatchStats._completion_length(traj)
        if completion_length is None:
            token_metrics_available = False
        else:
            completion_lengths.append(float(completion_length))
            advantage = traj.transitions[0].advantage
            if advantage is None or not isfinite(float(advantage)):
                raise RewardContractError(
                    f"GRPO rollout diagnostics need a finite first-transition advantage on trajectory {traj.id!r}"
                )
            token_advantages.extend([float(advantage)] * completion_length)
        if len(previews) < max_previews:
            previews.append(_preview(traj, reward))

    informative = tied = all_zero = all_one = 0
    for group_id, rows in grouped.items():
        group_rewards = [reward for _traj, reward, _spread, _mean in rows]
        expected_spread = max(group_rewards) - min(group_rewards)
        expected_mean = sum(group_rewards) / len(group_rewards)
        if any(
            not isclose(spread, expected_spread, rel_tol=1e-12, abs_tol=1e-12)
            or not isclose(mean, expected_mean, rel_tol=1e-12, abs_tol=1e-12)
            for _traj, _reward, spread, mean in rows
        ):
            raise RewardContractError(
                f"GRPO reward report found inconsistent group annotations for group {group_id!r}; "
                "reporting must consume the complete batch produced by the reward estimator"
            )
        # The stamp was just verified against the complete group, so the declared gate is the one
        # definition of "informative" for this report and for the replay buffer alike.
        if is_informative(rows[0][0]):
            informative += 1
        else:
            tied += 1
        all_zero += int(all(reward == 0.0 for reward in group_rewards))
        all_one += int(all(reward == 1.0 for reward in group_rewards))

    reward_moments = Moments.from_values(rewards)
    completion_length_moments = Moments.from_values(completion_lengths) if token_metrics_available else None
    advantage_moments = Moments.from_values(token_advantages) if token_metrics_available and token_advantages else None
    assert reward_moments.minimum is not None and reward_moments.maximum is not None
    return RewardBatchStats(
        trajectories=reward_moments.count,
        reward_sum=reward_moments.mean * reward_moments.count,
        reward_m2=reward_moments.m2,
        reward_min=reward_moments.minimum,
        reward_max=reward_moments.maximum,
        groups=len(grouped),
        informative_groups=informative,
        tied_groups=tied,
        all_zero_groups=all_zero,
        all_one_groups=all_one,
        previews=tuple(previews),
        completion_length_moments=completion_length_moments,
        advantage_moments=advantage_moments,
        provider_report={} if provider_report is None else provider_report,
        provider_provenance=tuple(provider_provenance),
    )


def resolve_reward_report(
    report_result: object,
    scored_result: object,
    max_previews: int = 2,
    *,
    idle_cells: tuple[int, ...] = (),
) -> RewardBatchStats:
    """Resolve the report barrier while preserving a failed score's root cause.

    The report call is queued after ``score_with_report`` on each single-threaded reward actor. If
    scoring fails, report retrieval observes only the deliberately invalidated report state. Resolve
    the scored result far enough on that error path to surface the provider/annotator/contract
    failure instead of hiding it behind a secondary "no completed reward report" error. Successful
    steps never resolve or re-ingress the scored trajectory payload.

    Args:
        report_result: Per-replica result of ``RewardWorker.batch_stats``.
        scored_result: Original per-replica result of ``RewardWorker.score_with_report``.
        max_previews: Maximum previews retained after merging replica reports.
        idle_cells: Original replica indices explicitly excluded from this refill. Only these
            cells may return an idle report; at least one cell must report a scored batch.

    Returns:
        One merged reward report spanning all replica leaders.

    Raises:
        Exception: The original score failure, or the report-resolution failure when scoring itself
            succeeded.
    """
    from cosmos_framework.posttrain.utils.entry import replica_lead_values, resolve_all

    try:
        reports = replica_lead_values(report_result)
    except Exception as report_error:
        try:
            resolve_all(scored_result)
        except Exception as score_error:
            raise score_error from report_error
        raise
    idle = set(idle_cells)
    if len(idle) != len(idle_cells) or any(
        isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(reports) for cell in idle
    ):
        raise ValueError(f"invalid idle reward cell indices: {idle_cells!r}")
    active_reports: list[RewardBatchStats] = []
    for cell, report in enumerate(reports):
        if cell in idle:
            if not isinstance(report, IdleRewardReport):
                raise RewardContractError(f"idle reward cell {cell} did not return an idle report")
        elif isinstance(report, RewardBatchStats):
            active_reports.append(report)
        else:
            raise RewardContractError(f"active reward cell {cell} did not return reward batch statistics")
    return RewardBatchStats.merge(active_reports, max_previews=max_previews)


@dataclass(frozen=True)
class ValidationRewardStats:
    """Exact mergeable held-out reward moments plus a compact identity inventory."""

    reward_moments: Moments
    model_step: int
    n_generation: int
    sample_ids: tuple[str, ...] = ()
    trajectory_ids: tuple[str, ...] = ()
    component_moments: dict[str, Moments] = field(default_factory=dict)
    provider_report: MetricReport = field(default_factory=dict)
    provider_provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.model_step, bool) or not isinstance(self.model_step, int) or self.model_step < 0:
            raise ValueError(f"validation model_step must be a non-negative integer, got {self.model_step!r}")
        if isinstance(self.n_generation, bool) or not isinstance(self.n_generation, int) or self.n_generation < 1:
            raise ValueError(f"validation n_generation must be a positive integer, got {self.n_generation!r}")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("validation reward report contains duplicate sample ids")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("validation reward report contains duplicate trajectory ids")
        if self.reward_moments.count != len(self.trajectory_ids):
            raise ValueError(
                f"validation reward moment count {self.reward_moments.count} != "
                f"trajectory inventory {len(self.trajectory_ids)}"
            )
        if len(self.trajectory_ids) != len(self.sample_ids) * self.n_generation:
            raise ValueError(
                f"validation trajectory inventory {len(self.trajectory_ids)} != "
                f"{len(self.sample_ids)} samples * {self.n_generation} generations"
            )
        self._component_output_names(self.component_moments)
        object.__setattr__(self, "provider_report", merge_metric_reports([self.provider_report]))
        object.__setattr__(
            self,
            "provider_provenance",
            _normalized_provider_provenance(self.provider_provenance),
        )

    @staticmethod
    def _component_output_names(components: dict[str, Moments]) -> dict[str, tuple[str, str]]:
        """Map raw provider keys to mean/count outputs and reject schema collisions."""
        reserved = {
            "val/reward_avg",
            "val/reward_std",
            "val/reward_max",
            "val/reward_min",
            "val/rollout_count",
            "val/step",
            "val/train_total_steps",
        }
        outputs: dict[str, tuple[str, str]] = {}
        claimed = set(reserved)
        for raw_key in sorted(components):
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(f"validation reward component key must be a non-empty string, got {raw_key!r}")
            if raw_key == FINAL_REWARD_KEY:
                raise ValueError(f"validation component moments must exclude reserved {FINAL_REWARD_KEY!r}")
            names = (f"val/{raw_key}", f"val/{raw_key}_count")
            collision = claimed & set(names)
            if collision:
                raise ValueError(
                    f"validation reward component {raw_key!r} collides with metric key(s) {sorted(collision)}"
                )
            outputs[raw_key] = names
            claimed.update(names)
        return outputs

    @staticmethod
    def _component_family_key(raw_key: str) -> str:
        """Legal internal moment-family key removed by final projection.

        The typed reducer validates every family key as schema-v2 public input. A digest avoids
        allowing arbitrary provider punctuation to escape that schema, while the explicit
        ``MomentOutputs`` mapping still projects the family to the human-facing raw key.
        """
        digest = hashlib.sha256(raw_key.encode()).hexdigest()
        return f"val/_component_moments/{digest}"

    @property
    def count(self) -> int:
        """Number of scored completions represented by these moments."""
        return self.reward_moments.count

    @classmethod
    def from_trajectories(
        cls,
        trajectories: Sequence[Trajectory],
        *,
        model_step: int,
        n_generation: int,
        provider_report: MetricReport | None = None,
        provider_provenance: Sequence[tuple[str, str]] = (),
    ) -> ValidationRewardStats:
        """Validate and summarize one physical held-out shard exactly.

        The rollout contract is one trajectory per ``obs.id × generation_index``. Retaining only
        a count lets a duplicate replace a missing completion undetected, so this boundary keeps
        the finite string inventory while discarding all tensor-bearing trajectory payloads.
        """
        grouped: dict[str, set[str]] = {}
        trajectory_ids: list[str] = []
        rewards: list[float] = []
        component_values: dict[str, list[float]] = {}
        for trajectory in trajectories:
            if len(trajectory.transitions) != 1:
                raise RewardContractError(
                    f"held-out GRPO validation needs one transition per trajectory, got "
                    f"{len(trajectory.transitions)} on {trajectory.id!r}"
                )
            transition = trajectory.transitions[0]
            sample_id = trajectory.group_id
            if not isinstance(sample_id, str) or not sample_id:
                raise RewardContractError(
                    f"held-out GRPO validation needs a non-empty group/sample id on {trajectory.id!r}"
                )
            if transition.obs.id != sample_id:
                raise RewardContractError(
                    f"held-out trajectory {trajectory.id!r} has group_id={sample_id!r} but obs.id={transition.obs.id!r}"
                )
            for key, expected in (("validation_step", model_step), ("weight_version", model_step)):
                actual = transition.policy_info.get(key)
                if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                    raise RewardContractError(
                        f"held-out trajectory {trajectory.id!r} needs policy_info[{key!r}]={expected}, got {actual!r}"
                    )
            if not isinstance(trajectory.id, str) or not trajectory.id:
                raise RewardContractError("held-out GRPO validation needs a non-empty trajectory id")
            grouped.setdefault(sample_id, set()).add(trajectory.id)
            trajectory_ids.append(trajectory.id)
            rewards.append(_terminal_reward(trajectory))
            reward_data = transition.reward
            assert reward_data is not None  # _terminal_reward above validated the mapping and final key
            for raw_key, value in reward_data.items():
                if raw_key == FINAL_REWARD_KEY:
                    continue
                if not isinstance(raw_key, str) or not raw_key:
                    raise RewardContractError(
                        f"held-out trajectory {trajectory.id!r} has invalid reward component key {raw_key!r}"
                    )
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                    raise RewardContractError(
                        f"held-out trajectory {trajectory.id!r} has non-finite/non-numeric reward "
                        f"component {raw_key!r}={value!r}"
                    )
                component_values.setdefault(raw_key, []).append(float(value))
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise RewardContractError("held-out GRPO validation produced duplicate trajectory ids in one shard")
        for sample_id, actual_ids in grouped.items():
            expected_ids = {f"{sample_id}_k{generation_index}" for generation_index in range(n_generation)}
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)[:5]
                unexpected = sorted(actual_ids - expected_ids)[:5]
                raise RewardContractError(
                    f"held-out sample {sample_id!r} has an invalid generation inventory; "
                    f"missing={missing}, unexpected={unexpected}"
                )
        # Empty is a valid PHYSICAL shard on the final uneven batch. ``Moments()`` is the exact
        # neutral merge element; finalizing an empty GLOBAL report remains an error.
        return cls(
            reward_moments=Moments.from_values(rewards),
            model_step=model_step,
            n_generation=n_generation,
            sample_ids=tuple(sorted(grouped)),
            trajectory_ids=tuple(sorted(trajectory_ids)),
            component_moments={raw_key: Moments.from_values(values) for raw_key, values in component_values.items()},
            provider_report={} if provider_report is None else provider_report,
            provider_provenance=tuple(provider_provenance),
        )

    @classmethod
    def merge(
        cls,
        reports: Sequence[ValidationRewardStats],
        *,
        expected_sample_ids: Sequence[str] | None = None,
    ) -> ValidationRewardStats:
        """Merge batches/replicas and reject duplicate, missing, or unexpected identities."""
        if not reports:
            raise ValueError("cannot merge an empty validation reward report inventory")
        model_step = reports[0].model_step
        n_generation = reports[0].n_generation
        if any(report.model_step != model_step for report in reports):
            raise ValueError("cannot merge validation reward reports from different model steps")
        if any(report.n_generation != n_generation for report in reports):
            raise ValueError("cannot merge validation reward reports with different n_generation")
        sample_ids = tuple(sample_id for report in reports for sample_id in report.sample_ids)
        trajectory_ids = tuple(trajectory_id for report in reports for trajectory_id in report.trajectory_ids)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("validation reward report merge found duplicate sample ids")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("validation reward report merge found duplicate trajectory ids")
        component_keys = set().union(*(report.component_moments for report in reports))
        component_moments = {
            raw_key: Moments.merge_many(
                [report.component_moments[raw_key] for report in reports if raw_key in report.component_moments]
            )
            for raw_key in component_keys
        }
        if expected_sample_ids is not None:
            expected_samples = tuple(expected_sample_ids)
            if len(set(expected_samples)) != len(expected_samples):
                raise ValueError("expected validation sample inventory itself contains duplicate ids")
            actual_set = set(sample_ids)
            expected_set = set(expected_samples)
            if actual_set != expected_set:
                missing = sorted(expected_set - actual_set)[:5]
                unexpected = sorted(actual_set - expected_set)[:5]
                raise ValueError(f"validation sample inventory mismatch; missing={missing}, unexpected={unexpected}")
            expected_trajectories = {
                f"{sample_id}_k{generation_index}"
                for sample_id in expected_set
                for generation_index in range(n_generation)
            }
            actual_trajectories = set(trajectory_ids)
            if actual_trajectories != expected_trajectories:
                missing = sorted(expected_trajectories - actual_trajectories)[:5]
                unexpected = sorted(actual_trajectories - expected_trajectories)[:5]
                raise ValueError(
                    f"validation trajectory inventory mismatch; missing={missing}, unexpected={unexpected}"
                )
        return cls(
            reward_moments=Moments.merge_many([report.reward_moments for report in reports]),
            model_step=model_step,
            n_generation=n_generation,
            sample_ids=tuple(sorted(sample_ids)),
            trajectory_ids=tuple(sorted(trajectory_ids)),
            component_moments=component_moments,
            provider_report=merge_metric_reports([report.provider_report for report in reports]),
            provider_provenance=_merge_provider_provenance([report.provider_provenance for report in reports]),
        )

    def metric_report(self, model_step: int, train_total_steps: int) -> MetricReport:
        """Typed validation contributions; moments remain unfinalized across every merge layer."""
        if model_step != self.model_step:
            raise ValueError(f"validation report is for model_step={self.model_step}, got {model_step}")
        report: MetricReport = {
            "val/reward": MetricContribution.moments(self.reward_moments),
            "val/step": MetricContribution.identical(model_step),
            "val/train_total_steps": MetricContribution.identical(train_total_steps),
        }
        for raw_key, moments in self.component_moments.items():
            report[self._component_family_key(raw_key)] = MetricContribution.moments(moments)
        return merge_metric_reports([report, self.provider_report])

    def metrics(self, model_step: int, train_total_steps: int) -> FinalMetricReport:
        """Cosmos-RL-compatible held-out reward keys with exact population std."""
        moment_outputs = {
            "val/reward": MomentOutputs(
                count="val/rollout_count",
                mean="val/reward_avg",
                std="val/reward_std",
                minimum="val/reward_min",
                maximum="val/reward_max",
            )
        }
        for raw_key, (mean_key, count_key) in self._component_output_names(self.component_moments).items():
            moment_outputs[self._component_family_key(raw_key)] = MomentOutputs(
                count=count_key,
                mean=mean_key,
            )
        return finalize_metric_report(
            # Exercise the same public-key validation and typed merge boundary used when reports
            # cross ranks. Internal component-family keys are legal val/* names and disappear in
            # the explicit moment projection below.
            merge_metric_reports([self.metric_report(model_step, train_total_steps)]),
            moment_outputs=moment_outputs,
        )


def resolve_validation_reward_report(
    report_result: object,
    scored_result: object,
) -> ValidationRewardStats:
    """Resolve and exactly merge per-replica validation reports, preserving score failures."""
    from cosmos_framework.posttrain.utils.entry import replica_lead_values, resolve_all

    try:
        reports = replica_lead_values(report_result)
    except Exception as report_error:
        try:
            resolve_all(scored_result)
        except Exception as score_error:
            raise score_error from report_error
        raise
    return ValidationRewardStats.merge(reports)
