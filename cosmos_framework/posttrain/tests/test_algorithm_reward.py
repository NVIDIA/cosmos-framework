# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for built-in rule-reward contracts and RewardWorker integration (DESIGN §1.5)."""

from __future__ import annotations

import pickle

import pytest

import cosmos_framework.posttrain.roles.reward.worker as reward_worker_module
from cosmos_framework.posttrain.algorithm.reward import (
    DEFAULT_REWARD_PROVIDER_CATALOG,
    RULE_REWARDS,
    RewardCandidate,
    RewardCombiner,
    RewardContext,
    RewardContractError,
    RewardProviderResult,
    arithmetic_reward_fn,
    bind_candidate_provider,
    constant_reward_fn,
    exact_match_reward_fn,
    nonempty_reward_fn,
    rule_reward_spec,
)
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.schema.metrics import MetricContribution, Moments
from cosmos_framework.posttrain.utils.batch_stats import IdleRewardReport


def _traj(
    answer: object | None,
    completion: object | None,
    *,
    include_answer: bool = True,
    include_completion: bool = True,
) -> Trajectory:
    representations: dict[str, object] = {"text": "1 + 1 ="}
    if include_answer:
        representations["reference"] = answer
    policy_info: dict[str, object] = {}
    if include_completion:
        policy_info["completion_text"] = completion
    transition = Transition(obs=ObsBundle(id="g", representations=representations), policy_info=policy_info, done=True)
    return Trajectory(transitions=[transition], id="t", group_id="g", status=Status.COMPLETED)


def _candidate(completion: str) -> RewardCandidate:
    return RewardCandidate(trajectory_id="t", completion=completion)


def _context(reference: object | None = None) -> RewardContext:
    return RewardContext(reference=reference)


def _task_score(result: RewardProviderResult) -> float:
    return result.terms["task"]


@pytest.mark.L0
@pytest.mark.CPU
def test_idle_reward_cell_skips_providers_and_preserves_one_shot_report_contract() -> None:
    def fail_if_called(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        raise AssertionError("idle cells must not call the provider")

    worker = RewardWorker([bind_candidate_provider("unused", fail_if_called)], training_group_size=1)
    assert worker.score_with_report([], idle=True) == []
    assert isinstance(worker.batch_stats(), IdleRewardReport)
    with pytest.raises(RuntimeError, match="no completed reward report"):
        worker.batch_stats()
    worker.score_with_report([], idle=True)
    with pytest.raises(RewardContractError, match="idle reward cell must receive no trajectories"):
        worker.score_with_report([_traj(2, "2")], idle=True)
    with pytest.raises(RuntimeError, match="no completed reward report"):
        worker.batch_stats()
    worker.score_with_report([], idle=True)
    with pytest.raises(RewardContractError):
        worker.score_with_report([])
    with pytest.raises(RuntimeError, match="no completed reward report"):
        worker.batch_stats()


def _validation_trajectories(completions: list[str], *, model_step: int) -> list[Trajectory]:
    """Stamp the exact rollout identity/weight contract expected by validation reward reporting."""
    sample_id = "validation:stream:7:0"
    trajectories = [_traj(2, completion) for completion in completions]
    for generation_index, trajectory in enumerate(trajectories):
        trajectory.id = f"{sample_id}_k{generation_index}"
        trajectory.group_id = sample_id
        trajectory.transitions[0].obs.id = sample_id
        trajectory.transitions[0].policy_info.update(
            validation_step=model_step,
            weight_version=model_step,
        )
    return trajectories


@pytest.mark.L0
@pytest.mark.CPU
def test_arithmetic_reward_preserves_legitimate_zero() -> None:
    assert _task_score(arithmetic_reward_fn(_candidate("the answer is 2!"), _context(2))) == 1.0
    assert _task_score(arithmetic_reward_fn(_candidate("2"), _context(2.0))) == 1.0
    assert _task_score(arithmetic_reward_fn(_candidate("2"), _context("2"))) == 1.0
    assert _task_score(arithmetic_reward_fn(_candidate("5"), _context(2))) == 0.0
    assert _task_score(arithmetic_reward_fn(_candidate("no number"), _context(2))) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_answer_rewards_reject_malformed_values() -> None:
    for answer in (True, 2.5, "2.5", [2]):
        with pytest.raises(RewardContractError, match="needs an integer answer"):
            arithmetic_reward_fn(_candidate("2"), _context(answer))
    for answer in (True, ["Paris"], {"value": "Paris"}):
        with pytest.raises(RewardContractError, match="string or finite numeric answer"):
            exact_match_reward_fn(_candidate("Paris"), _context(answer))
    for answer in (float("nan"), float("inf")):
        with pytest.raises(RewardContractError, match="common input.*must be finite"):
            _context(answer)


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_match_and_nonempty_semantics() -> None:
    assert _task_score(exact_match_reward_fn(_candidate("  paris "), _context("Paris"))) == 1.0
    assert _task_score(exact_match_reward_fn(_candidate("2"), _context(2))) == 1.0
    assert _task_score(exact_match_reward_fn(_candidate("Lyon"), _context("Paris"))) == 0.0
    assert _task_score(nonempty_reward_fn(_candidate("anything"), _context())) == 1.0
    assert _task_score(nonempty_reward_fn(_candidate("   "), _context())) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_constant_reward_makes_pure_distillation_task_advantage_exactly_zero() -> None:
    """Empty and non-empty completions receive one literal finite value before normalization."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    trajectories = [
        _traj(None, "", include_answer=False),
        _traj(None, "non-empty", include_answer=False),
    ]
    trajectories[0].id = "t0"
    trajectories[1].id = "t1"
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("constant")],
        RewardCombiner(),
        annotators=(group_relative(group_size=2),),
        training_group_size=2,
    )

    scored = worker.score(trajectories)

    assert _task_score(constant_reward_fn(_candidate(""), _context())) == 0.0
    assert all(trajectory.final_reward == 0.0 for trajectory in scored)
    assert all(trajectory.transitions[-1].advantage == 0.0 for trajectory in scored)


@pytest.mark.L0
@pytest.mark.CPU
def test_rule_rewards_fail_on_missing_contract_inputs() -> None:
    for reward_name in ("arithmetic", "exact_match"):
        worker = RewardWorker(
            [DEFAULT_REWARD_PROVIDER_CATALOG.build(reward_name)],
            training_group_size=1,
        )
        with pytest.raises(RewardContractError, match=r"needs terminal obs.representations\['reference'\]"):
            worker.score([_traj(None, "2", include_answer=False)])
        with pytest.raises(RewardContractError, match=r"needs terminal obs.representations\['reference'\]"):
            worker.score([_traj(None, "2")])

    for reward_name in set(RULE_REWARDS) - {"constant"}:
        worker = RewardWorker(
            [DEFAULT_REWARD_PROVIDER_CATALOG.build(reward_name)],
            training_group_size=1,
        )
        empty = Trajectory(transitions=[], id="t", group_id="g", status=Status.COMPLETED)
        with pytest.raises(RewardContractError, match="without a terminal transition"):
            worker.score([empty])
        with pytest.raises(RewardContractError, match="completion_text"):
            worker.score([_traj(2, None, include_completion=False)])
        with pytest.raises(RewardContractError, match="completion_text to be a string"):
            worker.score([_traj(2, ["2"])])

    constant = rule_reward_spec("constant")
    assert constant.needs == ()
    assert _task_score(constant.fn(_candidate(""), _context())) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_group_request_reads_the_terminal_transition() -> None:
    first = _traj(2, "9").transitions[0]
    terminal = _traj(2, "2").transitions[0]
    trajectory = Trajectory(transitions=[first, terminal], id="t", group_id="g", status=Status.COMPLETED)
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        training_group_size=1,
    )
    assert worker.score([trajectory])[0].final_reward == 1.0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rule_reward_provider_rejects_nonfinite_output(value: float) -> None:
    provider = bind_candidate_provider(
        "nonfinite",
        lambda _candidate, _context: RewardProviderResult(terms={"task": value}),
    )
    worker = RewardWorker([provider], training_group_size=1)
    with pytest.raises(RewardContractError, match="non-finite"):
        worker.score([_traj(2, "2")])


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_writes_final_onto_terminal() -> None:
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        RewardCombiner(),
        training_group_size=1,
    )
    [trajectory] = worker.score([_traj(2, "2")])
    assert trajectory.transitions[-1].reward["final"] == 1.0
    assert trajectory.final_reward == 1.0  # the convenience property reads final, not final + components


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_rejects_empty_provider_inventories() -> None:
    with pytest.raises(RewardContractError, match="at least one training provider"):
        RewardWorker([], training_group_size=1)
    with pytest.raises(RewardContractError, match="validation_providers must not be empty"):
        RewardWorker(
            [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
            validation_providers=[],
            training_group_size=1,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_provider_result_owns_generic_immutable_diagnostics() -> None:
    diagnostics = {"accepted": MetricContribution.sum(1)}
    result = RewardProviderResult(terms={"task": 1}, diagnostics=diagnostics)

    diagnostics.clear()
    assert result.terms == {"task": 1.0}
    assert result.diagnostics == {"accepted": MetricContribution.sum(1)}

    with pytest.raises(RewardContractError, match="must match"):
        RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"train/accepted": MetricContribution.sum(1)},
        )
    with pytest.raises(RewardContractError, match="reserved"):
        RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"sample_count": MetricContribution.sum(1)},
        )
    with pytest.raises(RewardContractError, match="MOMENTS"):
        RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"distribution": MetricContribution.moments(Moments.from_values([1.0]))},
        )
    with pytest.raises(RewardContractError, match="IDENTICAL payload must be a non-bool number"):
        RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"label": MetricContribution.identical("text")},
        )
    with pytest.raises(RewardContractError, match="positive observation weight"):
        RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"rate": MetricContribution.weighted_mean(1.0, weight=0.0)},
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_reports_catalog_bound_generic_diagnostics() -> None:
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    def _score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        del context
        correct = candidate.completion == "2"
        diagnostics = {"confidence": MetricContribution.weighted_mean(0.75)} if correct else {}
        return RewardProviderResult(terms={"task": float(correct)}, diagnostics=diagnostics)

    trajectories = [_traj(2, "2"), _traj(2, "5")]
    for index, trajectory in enumerate(trajectories):
        trajectory.id = f"t{index}"
    provider = bind_candidate_provider(
        "custom",
        _score_candidate,
        diagnostic_names=("confidence",),
        revision="r1",
    )
    worker = RewardWorker(
        [provider],
        annotators=(group_relative(group_size=2),),
        training_group_size=2,
    )

    worker.score_with_report(trajectories)
    report = worker.batch_stats()
    metrics = report.metrics()

    assert metrics["train/reward_provider/custom/sample_count"] == 2
    assert metrics["train/reward_provider/custom/confidence"] == pytest.approx(0.75)
    assert report.provider_provenance == (("custom", "r1"),)


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_rejects_unbound_or_undeclared_providers_atomically() -> None:
    class _BareProvider:
        @staticmethod
        def evaluate(_request: object) -> object:
            raise AssertionError("an unbound provider must not execute")

    bare_trajectory = _traj(2, "2")
    with pytest.raises(RewardContractError, match="catalog-bound BoundRewardProvider"):
        RewardWorker([_BareProvider()], training_group_size=1)  # type: ignore[list-item]
    assert bare_trajectory.transitions[-1].reward is None

    def _score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        del context
        diagnostic_name = "accepted" if candidate.completion == "2" else "undeclared"
        return RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={diagnostic_name: MetricContribution.sum(1)},
        )

    trajectories = [_traj(2, "2"), _traj(2, "5")]
    trajectories[0].id = "t0"
    trajectories[1].id = "t1"
    provider = bind_candidate_provider(
        "custom",
        _score_candidate,
        diagnostic_names=("accepted",),
    )
    with pytest.raises(RewardContractError, match="undeclared diagnostic"):
        RewardWorker([provider], training_group_size=2).score(trajectories)
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_converges_before_reward_mutation_or_annotation() -> None:
    def _score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        del context
        if candidate.trajectory_id == "bad":
            raise RewardContractError("rank-local provider failure")
        return RewardProviderResult(terms={"task": 1.0})

    provider = bind_candidate_provider("custom", _score_candidate)
    annotations: list[int] = []

    def annotate(trajectories: list[Trajectory]) -> None:
        annotations.append(len(trajectories))

    convergence_calls: list[tuple[Exception | None, str]] = []

    def converge(local_error: Exception | None, context: str) -> None:
        convergence_calls.append((local_error, context))

    good = _traj(2, "2")
    bad = _traj(2, "2")
    good.id = "good"
    bad.id = "bad"
    worker = RewardWorker(
        [provider],
        annotators=(annotate,),
        training_group_size=2,
        failure_convergence=converge,
    )
    with pytest.raises(RewardContractError, match="rank-local provider failure"):
        worker.score([good, bad])

    assert convergence_calls and isinstance(convergence_calls[0][0], RewardContractError)
    assert convergence_calls[0][1] == "reward provider and combiner planning"
    assert annotations == []
    assert good.transitions[-1].reward is None
    assert bad.transitions[-1].reward is None

    def peer_failed(local_error: Exception | None, _context: str) -> None:
        assert local_error is None
        raise RuntimeError("peer rank failed")

    healthy = _traj(2, "2")
    worker = RewardWorker(
        [provider],
        annotators=(annotate,),
        training_group_size=1,
        failure_convergence=peer_failed,
    )
    with pytest.raises(RuntimeError, match="peer rank failed"):
        worker.score([healthy])
    assert healthy.transitions[-1].reward is None
    assert annotations == []


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_provider_report_is_serializable_and_single_use() -> None:
    def _score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        del candidate, context
        return RewardProviderResult(
            terms={"task": 1.0},
            diagnostics={"latency": MetricContribution.weighted_mean(0.25)},
        )

    provider = bind_candidate_provider(
        "custom",
        _score_candidate,
        diagnostic_names=("latency",),
        revision="r2",
    )
    worker = RewardWorker([provider], training_group_size=1)
    worker.score_with_provider_report([_traj(2, "2")])
    report = worker.provider_report()
    restored = pickle.loads(pickle.dumps(report))

    assert restored.provenance == (("custom", "r2"),)
    assert set(restored.metrics) == {
        "train/reward_provider/custom/latency",
        "train/reward_provider/custom/sample_count",
    }
    with pytest.raises(RuntimeError, match="score_with_provider_report must succeed"):
        worker.provider_report()


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_exposes_declared_annotation_metrics_without_collisions() -> None:
    class _Annotator:
        def __init__(self, metrics: dict[str, int | float]) -> None:
            self._metrics = metrics

        def __call__(self, _trajectories: list[Trajectory]) -> None:
            return None

        def metrics(self) -> dict[str, int | float]:
            return dict(self._metrics)

    provider = DEFAULT_REWARD_PROVIDER_CATALOG.build("constant")
    worker = RewardWorker(
        [provider],
        annotators=(_Annotator({"rollout/reward_group_size": 1}),),
        training_group_size=1,
    )
    assert worker.annotation_metrics() == {"rollout/reward_group_size": 1}

    colliding = RewardWorker(
        [provider],
        annotators=(
            _Annotator({"rollout/reward_group_size": 1}),
            _Annotator({"rollout/reward_group_size": 1}),
        ),
        training_group_size=1,
    )
    with pytest.raises(ValueError, match="metric keys collide"):
        colliding.annotation_metrics()

    malformed = RewardWorker(
        [provider],
        annotators=(_Annotator({"rollout/reward_group_size": True}),),
        training_group_size=1,
    )
    with pytest.raises(TypeError, match="must be a numeric scalar"):
        malformed.annotation_metrics()


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_rolls_back_partial_annotator_writes() -> None:
    marker = object()
    trajectories = [_traj(2, "2"), _traj(2, "5")]
    for index, trajectory in enumerate(trajectories):
        trajectory.id = f"t{index}"
        transition = trajectory.transitions[-1]
        transition.reward = {"previous": float(index)}
        transition.advantage = float(index + 10)
        transition.algo_extra.update(previous=index, bulk_handle=marker)
    original_rewards = [trajectory.transitions[-1].reward for trajectory in trajectories]
    original_algo_extra = [trajectory.transitions[-1].algo_extra for trajectory in trajectories]

    def _partially_write_then_fail(batch: list[Trajectory]) -> None:
        transition = batch[0].transitions[-1]
        transition.reward["partial"] = 99.0
        transition.advantage = 99.0
        transition.algo_extra["partial"] = True
        raise RuntimeError("annotator failed after a partial write")

    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        annotators=(_partially_write_then_fail,),
        training_group_size=2,
    )

    with pytest.raises(RuntimeError, match="partial write"):
        worker.score(trajectories)

    for index, trajectory in enumerate(trajectories):
        transition = trajectory.transitions[-1]
        assert transition.reward is original_rewards[index]
        assert transition.reward == {"previous": float(index)}
        assert transition.advantage == float(index + 10)
        assert transition.algo_extra is original_algo_extra[index]
        assert transition.algo_extra == {"previous": index, "bulk_handle": marker}
        assert transition.algo_extra["bulk_handle"] is marker


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("report_builder", ["provider_diagnostics_report", "reward_batch_stats"])
def test_reward_worker_rolls_back_when_report_building_fails(
    monkeypatch: pytest.MonkeyPatch,
    report_builder: str,
) -> None:
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    marker = object()
    trajectories = [_traj(2, "2"), _traj(2, "5")]
    for index, trajectory in enumerate(trajectories):
        trajectory.id = f"t{index}"
        transition = trajectory.transitions[-1]
        transition.advantage = float(index + 20)
        transition.algo_extra["bulk_handle"] = marker

    def _fail_report(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"{report_builder} failed")

    monkeypatch.setattr(reward_worker_module, report_builder, _fail_report)
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        annotators=(group_relative(group_size=2),),
        training_group_size=2,
    )

    with pytest.raises(RuntimeError, match=f"{report_builder} failed"):
        worker.score_with_report(trajectories)

    for index, trajectory in enumerate(trajectories):
        transition = trajectory.transitions[-1]
        assert transition.reward is None
        assert transition.advantage == float(index + 20)
        assert transition.algo_extra == {"bulk_handle": marker}
        assert transition.algo_extra["bulk_handle"] is marker
    with pytest.raises(RuntimeError, match="score_with_report must succeed"):
        worker.batch_stats()


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_reports_after_group_annotations() -> None:
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    trajectories = [_traj(2, "2"), _traj(2, "5")]
    for index, trajectory in enumerate(trajectories):
        trajectory.id = f"t{index}"
        trajectory.group_id = "g"
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        RewardCombiner(),
        annotators=(group_relative(group_size=2),),
        training_group_size=2,
    )

    scored = worker.score_with_report(trajectories)
    report = worker.batch_stats()

    assert report.mean_reward == 0.5
    assert report.informative_groups == 1

    with pytest.raises(RuntimeError, match="score_with_report must succeed"):
        worker.batch_stats()

    worker.score_with_report(trajectories)
    worker.score(trajectories)
    with pytest.raises(RuntimeError, match="score_with_report must succeed"):
        worker.batch_stats()


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_validation_is_reward_only_and_exactly_mergeable() -> None:
    """Validation scores every completion but never runs GRPO advantage/filter annotators."""
    annotated: list[int] = []

    def _training_only_annotator(trajectories: list[Trajectory]) -> None:
        annotated.append(len(trajectories))
        for trajectory in trajectories:
            trajectory.transitions[-1].advantage = 123.0

    trajectories = _validation_trajectories(["2", "5"], model_step=4)
    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        RewardCombiner(),
        annotators=(_training_only_annotator,),
        training_group_size=2,
    )
    scored = worker.score_validation_with_report(trajectories, model_step=4, n_generation=2)
    report = worker.validation_batch_stats()

    assert scored is None
    assert annotated == []
    assert all(t.transitions[-1].advantage is None for t in trajectories)
    assert report.count == 2
    assert report.reward_moments.mean == 0.5
    assert report.reward_moments.m2 == 0.5
    assert report.reward_moments.minimum == 0.0
    assert report.reward_moments.maximum == 1.0
    assert report.metrics(model_step=4, train_total_steps=9) == pytest.approx(
        {
            "val/reward_avg": 0.5,
            "val/reward_std": 0.5,
            "val/reward_max": 1.0,
            "val/reward_min": 0.0,
            "val/reward_provider/arithmetic/sample_count": 2,
            "val/rollout_count": 2,
            "val/step": 4,
            "val/train_total_steps": 9,
            "val/task": 0.5,
            "val/task_count": 2,
        }
    )
    with pytest.raises(RuntimeError, match="score_validation_with_report must succeed"):
        worker.validation_batch_stats()


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_worker_validation_can_use_an_independent_provider() -> None:
    """The reused RM role may select a held-out reward without changing training scoring."""
    worker = RewardWorker(
        [
            bind_candidate_provider(
                "training_zero",
                lambda _candidate, _context: RewardProviderResult(terms={"task": 0.0}),
            )
        ],
        RewardCombiner(),
        validation_providers=[
            bind_candidate_provider(
                "validation_constant",
                lambda _candidate, _context: RewardProviderResult(terms={"task": 0.75}),
            )
        ],
        training_group_size=1,
    )
    training = worker.score([_traj(2, "2")])[0].final_reward
    validation_trajectory = _validation_trajectories(["2"], model_step=0)
    validation = worker.score_validation_with_report(
        validation_trajectory,
        model_step=0,
        n_generation=1,
    )
    assert training == 0.0
    assert validation is None
    assert validation_trajectory[0].final_reward == 0.75


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_rejects_sparse_component_schema_atomically() -> None:
    """Every canonical provider has one fixed optimized-term schema across its group."""

    def _sparse_score(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        del context
        if candidate.completion == "2":
            return RewardProviderResult(terms={"task": 1.0, "confidence": 0.75})
        return RewardProviderResult(terms={"task": 0.0})

    trajectories = _validation_trajectories(["2", "5"], model_step=3)
    provider = bind_candidate_provider(
        "sparse",
        _sparse_score,
        term_names=("task", "confidence"),
    )
    worker = RewardWorker([provider], RewardCombiner(), training_group_size=2)
    with pytest.raises(RewardContractError, match="term schema differs.*confidence"):
        worker.score_validation_with_report(trajectories, model_step=3, n_generation=2)
    assert all(trajectory.transitions[-1].reward is None for trajectory in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_reward_report_rejects_stale_and_incomplete_identity_inventory() -> None:
    from cosmos_framework.posttrain.utils.batch_stats import ValidationRewardStats

    worker = RewardWorker(
        [DEFAULT_REWARD_PROVIDER_CATALOG.build("arithmetic")],
        RewardCombiner(),
        training_group_size=1,
    )
    stale = _validation_trajectories(["2"], model_step=2)
    stale[0].transitions[0].policy_info["weight_version"] = 1
    with pytest.raises(RewardContractError, match="weight_version.*2.*got 1"):
        worker.score_validation_with_report(stale, model_step=2, n_generation=1)

    complete = _validation_trajectories(["2"], model_step=2)
    worker.score_validation_with_report(complete, model_step=2, n_generation=1)
    report = worker.validation_batch_stats()
    with pytest.raises(ValueError, match="sample inventory mismatch.*missing"):
        ValidationRewardStats.merge(
            [report],
            expected_sample_ids=("validation:stream:7:0", "validation:stream:7:1"),
        )

    duplicate = _validation_trajectories(["2", "5"], model_step=2)
    duplicate[1].id = duplicate[0].id
    with pytest.raises(RewardContractError, match="invalid generation inventory|duplicate trajectory"):
        worker.score_validation_with_report(duplicate, model_step=2, n_generation=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_reward_name_is_actionable() -> None:
    with pytest.raises(KeyError, match="unknown reward 'nope'"):
        rule_reward_spec("nope")
