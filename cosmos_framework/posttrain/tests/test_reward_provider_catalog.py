# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""L0/CPU contracts for run-local reward provider catalogs."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
from cosmos_framework.posttrain.algorithm.reward import (
    DEFAULT_REWARD_PROVIDER_CATALOG,
    RULE_REWARDS,
    RewardCandidate,
    RewardCandidateResult,
    RewardCombiner,
    RewardContext,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderCatalog,
    RewardProviderResult,
    RewardProviderSpec,
    bind_candidate_provider,
    map_candidates,
    resolve_reward_provider_catalog,
)
from cosmos_framework.posttrain.config.rl import (
    DatasetConfig,
    GRPOConfig,
    GRPOValidationConfig,
    OPDConfig,
)
from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.validation.rollout import validate_rollout_reward_sources
from cosmos_framework.posttrain.entrypoints import grpo, opd
from cosmos_framework.posttrain.entrypoints.grpo import manifest
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.schema import MetricContribution, ObsBundle, Status, Trajectory, Transition

_CUSTOM_CATALOG_PATH = "cosmos_framework.posttrain.tests.helpers.reward_provider_catalog.custom_reward_provider_catalog"
_EXAMPLE_CATALOG_PATH = (
)


class _Provider:
    requests: list[RewardGroupRequest]
    _result: RewardGroupResult | RewardProviderResult | None

    def __init__(self, result: RewardGroupResult | RewardProviderResult | None = None) -> None:
        self.requests = []
        self._result = result

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        self.requests.append(request)
        if isinstance(self._result, RewardGroupResult):
            return self._result
        candidate_result = self._result or RewardProviderResult(terms={"task": 1.0})
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(trajectory_id=candidate.trajectory_id, result=candidate_result)
                for candidate in request.candidates
            ),
        )


def _factory_with_required_argument(required: object) -> _Provider:
    del required
    return _Provider()


class _MissingEvaluateProvider:
    pass


def _reward_group_result(
    group_id: str,
    trajectory_ids: tuple[str, ...],
    results: tuple[RewardProviderResult, ...],
) -> RewardGroupResult:
    return RewardGroupResult(
        group_id=group_id,
        items=tuple(
            RewardCandidateResult(trajectory_id=trajectory_id, result=result)
            for trajectory_id, result in zip(trajectory_ids, results, strict=True)
        ),
    )


def _group_trajectory(
    trajectory_id: str,
    completion: str,
    *,
    group_id: str = "prompt_0",
    prompt: object = "common prompt",
    reference: object = "common reference",
    completion_token_count: object = None,
) -> Trajectory:
    policy_info: dict[str, object] = {"completion_text": completion}
    if completion_token_count is not None:
        policy_info["completion_token_count"] = completion_token_count
    transition = Transition(
        obs=ObsBundle(
            representations={"prompt": prompt, "reference": reference},
            id=group_id,
        ),
        policy_info=policy_info,
        done=True,
    )
    return Trajectory(
        transitions=[transition],
        id=trajectory_id,
        group_id=group_id,
        status=Status.COMPLETED,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_default_catalog_contains_only_existing_rule_rewards() -> None:
    assert set(DEFAULT_REWARD_PROVIDER_CATALOG) == set(RULE_REWARDS)
    with pytest.raises(TypeError):
        DEFAULT_REWARD_PROVIDER_CATALOG["new"] = RewardProviderSpec(factory=_Provider)  # type: ignore[index]


@pytest.mark.L0
@pytest.mark.CPU
def test_catalog_validates_names_and_rejects_collisions() -> None:
    spec = RewardProviderSpec(factory=_Provider)
    with pytest.raises(ValueError, match="must match"):
        RewardProviderCatalog({"Bad-Name": spec})
    with pytest.raises(ValueError, match="override.*arithmetic"):
        DEFAULT_REWARD_PROVIDER_CATALOG.extend({"arithmetic": spec})


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_semantic_identity_is_finite_deeply_frozen_and_stably_hashed() -> None:
    author_identity = {
        "model": "judge-v1",
        "retry": {"attempts": 2, "backoff_s": [0.5, 1.0]},
    }
    spec = RewardProviderSpec(
        factory=_Provider,
        semantic_identity=author_identity,
        is_remote=True,
    )
    reordered = RewardProviderSpec(
        factory=_Provider,
        semantic_identity={
            "retry": {"backoff_s": [0.5, 1.0], "attempts": 2},
            "model": "judge-v1",
        },
        is_remote=True,
    )

    author_identity["model"] = "mutated"
    author_identity["retry"]["backoff_s"].append(2.0)  # type: ignore[index,union-attr]

    assert spec.semantic_identity["model"] == "judge-v1"
    assert spec.semantic_identity["retry"] == {"attempts": 2, "backoff_s": (0.5, 1.0)}
    assert spec.identity_hash == reordered.identity_hash
    assert len(spec.identity_hash) == 64
    assert spec.is_remote
    with pytest.raises(TypeError):
        spec.semantic_identity["model"] = "cannot mutate"  # type: ignore[index]
    with pytest.raises(ValueError, match="must be finite"):
        RewardProviderSpec(factory=_Provider, semantic_identity={"timeout": float("nan")})
    with pytest.raises(TypeError, match="JSON-compatible"):
        RewardProviderSpec(factory=_Provider, semantic_identity={"models": {"judge-v1"}})
    with pytest.raises(TypeError, match="is_remote must be a bool"):
        RewardProviderSpec(factory=_Provider, is_remote=1)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_group_provider_builds_typed_common_request_and_returns_aligned_results() -> None:
    def score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        assert context.prompt == "common prompt"
        return RewardProviderResult(terms={"task": float(candidate.trajectory_id[-1])})

    implementation = map_candidates(score_candidate)
    catalog = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=lambda: implementation,
                needs=("prompt", "reference"),
                revision="rubric-v1",
                semantic_identity={"judge_revision": "2026-08-27"},
                is_remote=True,
            )
        }
    )
    provider = catalog.build("rubric")
    trajectories = [
        _group_trajectory("trajectory_0", "first completion", completion_token_count=2),
        _group_trajectory("trajectory_1", "second completion"),
    ]

    request = provider.prepare_group(trajectories)
    result = provider.evaluate_prepared_group(request)

    assert request.group_id == "prompt_0"
    assert request.context.prompt == "common prompt"
    assert request.context.reference == "common reference"
    assert tuple(candidate.trajectory_id for candidate in request.candidates) == (
        "trajectory_0",
        "trajectory_1",
    )
    assert tuple(candidate.completion for candidate in request.candidates) == (
        "first completion",
        "second completion",
    )
    assert tuple(candidate.completion_token_count for candidate in request.candidates) == (2, None)
    assert tuple(item.result.terms["task"] for item in result.items) == (0.0, 1.0)
    assert provider.needs == ("prompt", "reference")
    assert provider.is_remote and provider.identity_hash == catalog.spec("rubric").identity_hash


@pytest.mark.L0
@pytest.mark.CPU
def test_map_candidates_preserves_order_identity_and_context() -> None:
    calls: list[tuple[str, str, object]] = []

    def score_candidate(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
        calls.append((candidate.trajectory_id, candidate.completion, context.reference))
        return RewardProviderResult(terms={"custom": float(len(candidate.completion))})

    provider = bind_candidate_provider(
        "mapped",
        score_candidate,
        needs=("reference",),
        term_names=("custom",),
    )
    trajectories = [
        _group_trajectory("trajectory_0", "a"),
        _group_trajectory("trajectory_1", "long"),
    ]

    result = provider.evaluate_prepared_group(provider.prepare_group(trajectories))

    assert calls == [
        ("trajectory_0", "a", "common reference"),
        ("trajectory_1", "long", "common reference"),
    ]
    assert tuple(item.trajectory_id for item in result.items) == ("trajectory_0", "trajectory_1")
    assert tuple(item.result.terms["custom"] for item in result.items) == (1.0, 4.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_group_request_deep_freezes_common_inputs_without_aliasing_trajectories() -> None:
    prompt = {"messages": [{"role": "user", "content": "original"}]}
    reference = {"criteria": [{"id": "c1", "description": "Follow the instruction."}]}
    provider = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=_Provider,
                needs=("prompt", "reference"),
            )
        }
    ).build("rubric")
    trajectories = [
        _group_trajectory("trajectory_0", "first", prompt=prompt, reference=reference),
        _group_trajectory("trajectory_1", "second", prompt=prompt, reference=reference),
    ]

    request = provider.prepare_group(trajectories)
    prompt["messages"][0]["content"] = "mutated after prepare"
    reference["criteria"][0]["description"] = "mutated after prepare"

    assert isinstance(request.context.prompt, Mapping)
    frozen_messages = request.context.prompt["messages"]
    assert isinstance(frozen_messages, tuple)
    assert isinstance(frozen_messages[0], Mapping)
    assert frozen_messages[0]["content"] == "original"
    assert isinstance(request.context.reference, Mapping)
    with pytest.raises(TypeError):
        request.context.reference["criteria"] = ()  # type: ignore[index]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("first_reference", "second_reference"),
    [
        (1, True),
        (1, 1.0),
        ({"criterion": 1}, {"criterion": True}),
    ],
    ids=("integer-bool", "integer-float", "nested-integer-bool"),
)
def test_group_request_common_inputs_use_type_strict_structural_identity(
    first_reference: object,
    second_reference: object,
) -> None:
    implementation = _Provider()
    provider = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=lambda: implementation,
                needs=("reference",),
            )
        }
    ).build("rubric")
    trajectories = [
        _group_trajectory("trajectory_0", "first", reference=first_reference),
        _group_trajectory("trajectory_1", "second", reference=second_reference),
    ]

    with pytest.raises(RewardContractError, match="reference.*must be identical"):
        provider.prepare_group(trajectories)

    assert implementation.requests == []


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_binding_requires_the_canonical_evaluate_method() -> None:
    missing_method = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=_MissingEvaluateProvider,  # type: ignore[arg-type]
            )
        }
    )
    with pytest.raises(TypeError, match=r"without evaluate\(\)"):
        missing_method.build("rubric")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda trajectories: setattr(trajectories[1], "group_id", "other_group"),
            "mixed group_ids",
        ),
        (
            lambda trajectories: setattr(trajectories[1], "id", trajectories[0].id),
            "trajectory_ids must be unique",
        ),
        (
            lambda trajectories: setattr(trajectories[1].transitions[-1].obs, "id", "other_group"),
            "obs.id.*does not match",
        ),
        (
            lambda trajectories: (
                trajectories[1].transitions[-1].obs.representations.__setitem__("prompt", "different prompt")
            ),
            "prompt.*must be identical",
        ),
        (
            lambda trajectories: setattr(trajectories[1], "status", Status.ABORTED),
            "must be completed",
        ),
        (
            lambda trajectories: (
                trajectories[1].transitions[-1].policy_info.__setitem__("completion_token_count", True)
            ),
            "completion_token_count.*non-negative integer",
        ),
    ],
    ids=("mixed-group", "duplicate-id", "obs-id", "common-prompt", "incomplete", "token-count"),
)
def test_group_request_rejects_malformed_trajectories_before_provider_execution(
    mutate: Callable[[list[Trajectory]], object],
    match: str,
) -> None:
    implementation = _Provider()
    provider = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=lambda: implementation,
                needs=("prompt", "reference"),
            )
        }
    ).build("rubric")
    trajectories = [
        _group_trajectory("trajectory_0", "first completion"),
        _group_trajectory("trajectory_1", "second completion"),
    ]
    mutate(trajectories)

    with pytest.raises(RewardContractError, match=match):
        provider.prepare_group(trajectories)
    assert implementation.requests == []


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("result", "match"),
    [
        (
            _reward_group_result(
                "other_group",
                ("trajectory_0", "trajectory_1"),
                (
                    RewardProviderResult(terms={"task": 0.0}),
                    RewardProviderResult(terms={"task": 1.0}),
                ),
            ),
            "returned group_id",
        ),
        (
            _reward_group_result(
                "prompt_0",
                ("trajectory_0",),
                (RewardProviderResult(terms={"task": 0.0}),),
            ),
            "returned 1 candidates; expected 2",
        ),
        (
            _reward_group_result(
                "prompt_0",
                ("trajectory_0", "unknown"),
                (
                    RewardProviderResult(terms={"task": 0.0}),
                    RewardProviderResult(terms={"task": 1.0}),
                ),
            ),
            "different trajectory identities",
        ),
        (
            _reward_group_result(
                "prompt_0",
                ("trajectory_1", "trajectory_0"),
                (
                    RewardProviderResult(terms={"task": 1.0}),
                    RewardProviderResult(terms={"task": 0.0}),
                ),
            ),
            "out of order",
        ),
        (
            _reward_group_result(
                "prompt_0",
                ("trajectory_0", "trajectory_1"),
                (
                    RewardProviderResult(terms={"task": 0.0}),
                    RewardProviderResult(terms={"undeclared": 1.0}),
                ),
            ),
            "term schema differs",
        ),
    ],
    ids=("group-id", "cardinality", "identity", "order", "term-schema"),
)
def test_bound_group_provider_rejects_result_identity_and_schema_drift(
    result: RewardGroupResult,
    match: str,
) -> None:
    provider = RewardProviderCatalog(
        {
            "rubric": RewardProviderSpec(
                factory=lambda: _Provider(result),
            )
        }
    ).build("rubric")
    trajectories = [
        _group_trajectory("trajectory_0", "first completion"),
        _group_trajectory("trajectory_1", "second completion"),
    ]

    with pytest.raises(RewardContractError, match=match):
        request = provider.prepare_group(trajectories)
        provider.evaluate_prepared_group(request)


@pytest.mark.L0
@pytest.mark.CPU
def test_group_result_contract_rejects_empty_duplicate_and_untyped_items() -> None:
    valid_result = RewardProviderResult(terms={"task": 1.0})
    with pytest.raises(RewardContractError, match="trajectory_ids must be unique"):
        RewardGroupResult(
            group_id="prompt_0",
            items=(
                RewardCandidateResult("trajectory_0", valid_result),
                RewardCandidateResult("trajectory_0", valid_result),
            ),
        )
    with pytest.raises(RewardContractError, match="must not be empty"):
        RewardGroupResult(
            group_id="prompt_0",
            items=(),
        )
    with pytest.raises(RewardContractError, match="must contain RewardCandidateResult"):
        RewardGroupResult(
            group_id="prompt_0",
            items=({"trajectory_0": valid_result},),  # type: ignore[arg-type]
        )
    with pytest.raises(RewardContractError, match="must be a RewardProviderResult"):
        RewardCandidateResult("trajectory_0", {"task": 1.0})  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_spec_preserves_legacy_positional_field_order() -> None:
    spec = RewardProviderSpec(
        _Provider,
        (),
        "r1",
        (),
        ("task",),
        {"kind": "legacy"},
        True,
    )

    assert spec.revision == "r1"
    assert spec.term_names == ("task",)
    assert spec.semantic_identity == {"kind": "legacy"}
    assert spec.is_remote is True
    assert spec.token_term_names == ()


@pytest.mark.L0
@pytest.mark.CPU
def test_dotted_catalog_factory_merges_and_builds_bound_provider() -> None:
    catalog = resolve_reward_provider_catalog(_CUSTOM_CATALOG_PATH)
    provider = catalog.build("custom_prompt")

    request = provider.prepare_group([_group_trajectory("trajectory_0", "completion")])
    result = provider.evaluate_prepared_group(request)

    assert set(catalog) == {*RULE_REWARDS, "custom_prompt"}
    assert provider.provider_name == "custom_prompt"
    assert provider.provider_revision == "test-v1"
    assert provider.diagnostic_names == ("accepted",)
    assert result.items[0].result.terms == {"custom": 0.25}
    assert set(result.items[0].result.diagnostics) == {"accepted"}


@pytest.mark.L0
@pytest.mark.CPU
def test_bound_provider_rejects_factory_and_output_schema_drift() -> None:
    wrong_terms = RewardProviderResult(terms={"other": 1.0})
    catalog = RewardProviderCatalog({"custom": RewardProviderSpec(factory=lambda: _Provider(wrong_terms))})
    provider = catalog.build("custom")
    request = provider.prepare_group([_group_trajectory("trajectory_0", "completion")])
    with pytest.raises(RewardContractError, match="term schema differs"):
        provider.evaluate_prepared_group(request)

    undeclared_diagnostic = RewardProviderResult(
        terms={"task": 1.0},
        diagnostics={"extra": MetricContribution.sum(1)},
    )
    catalog = RewardProviderCatalog({"custom": RewardProviderSpec(factory=lambda: _Provider(undeclared_diagnostic))})
    provider = catalog.build("custom")
    request = provider.prepare_group([_group_trajectory("trajectory_0", "completion")])
    with pytest.raises(RewardContractError, match="undeclared diagnostic"):
        provider.evaluate_prepared_group(request)


@pytest.mark.L0
@pytest.mark.CPU
def test_provider_specs_support_exact_scalar_token_and_mixed_schemas() -> None:
    token_result = RewardProviderResult(token_terms={"process": (0.25, 0.5)})
    token_spec = RewardProviderSpec(
        factory=lambda: _Provider(token_result),
        term_names=(),
        token_term_names=("process",),
    )
    token_provider = RewardProviderCatalog({"token_only": token_spec}).build("token_only")

    assert token_spec.term_names == ()
    assert token_spec.token_term_names == ("process",)
    assert token_provider.token_term_names == ("process",)
    token_request = token_provider.prepare_group([_group_trajectory("token_0", "completion")])
    assert token_provider.evaluate_prepared_group(token_request).items[0].result.token_terms == {"process": (0.25, 0.5)}

    mixed_result = RewardProviderResult(
        terms={"task": 1.0},
        token_terms={"process": (0.25,)},
    )
    mixed_spec = RewardProviderSpec(
        factory=lambda: _Provider(mixed_result),
        term_names=("task",),
        token_term_names=("process",),
    )
    mixed_provider = RewardProviderCatalog({"mixed": mixed_spec}).build("mixed")
    mixed_request = mixed_provider.prepare_group([_group_trajectory("mixed_0", "completion")])
    assert mixed_provider.evaluate_prepared_group(mixed_request).items[0].result == mixed_result

    with pytest.raises(ValueError, match="at least one scalar or token term"):
        RewardProviderSpec(factory=_Provider, term_names=(), token_term_names=())
    with pytest.raises(ValueError, match="reserved term 'final'"):
        RewardProviderSpec(factory=_Provider, term_names=(), token_term_names=("final",))
    with pytest.raises(ValueError, match="both scalar and token"):
        RewardProviderSpec(factory=_Provider, term_names=("score",), token_term_names=("score",))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("result", "term_names", "match"),
    [
        (RewardProviderResult(terms={"task": 1.0}), ("task",), "token term schema differs"),
        (RewardProviderResult(token_terms={"other": (0.25,)}), (), "token term schema differs"),
        (
            RewardProviderResult(
                terms={"unexpected": 1.0},
                token_terms={"process": (0.25,)},
            ),
            (),
            "term schema differs",
        ),
    ],
)
def test_bound_provider_rejects_scalar_or_token_output_schema_drift(
    result: RewardProviderResult,
    term_names: tuple[str, ...],
    match: str,
) -> None:
    spec = RewardProviderSpec(
        factory=lambda: _Provider(result),
        term_names=term_names,
        token_term_names=("process",),
    )
    provider = RewardProviderCatalog({"custom": spec}).build("custom")
    request = provider.prepare_group([_group_trajectory("trajectory_0", "completion")])
    with pytest.raises(RewardContractError, match=match):
        provider.evaluate_prepared_group(request)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("provider_name", "factory", "match"),
    [
        ("required_argument", _factory_with_required_argument, "required positional argument"),
        ("missing_evaluate", _MissingEvaluateProvider, "without evaluate"),
    ],
)
def test_driver_preflight_builds_provider_and_reports_factory_contract_errors(
    provider_name: str,
    factory: Callable[..., object],
    match: str,
) -> None:
    catalog = RewardProviderCatalog(
        {provider_name: RewardProviderSpec(factory=factory)}  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=rf"reward provider '{provider_name}'.*{match}"):
        validate_rollout_reward_sources(GRPOConfig(reward=provider_name), catalog)


@pytest.mark.L0
@pytest.mark.CPU
def test_driver_preflight_builds_distinct_validation_provider_and_deduplicates_inherited_reward(
    tmp_path: Path,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(json.dumps({"prompt": "validation"}) + "\n", encoding="utf-8")
    constructions: list[str] = []

    def training_factory() -> _Provider:
        constructions.append("training")
        return _Provider()

    def validation_factory() -> _Provider:
        constructions.append("validation")
        return _Provider()

    catalog = RewardProviderCatalog(
        {
            "training": RewardProviderSpec(factory=training_factory),
            "validation": RewardProviderSpec(factory=validation_factory),
        }
    )
    rl = GRPOConfig(
        reward="training",
        validation=GRPOValidationConfig(
            enabled=True,
            data=DatasetConfig(path=str(validation_path), prompt_column="prompt"),
            reward="validation",
        ),
    )

    validate_rollout_reward_sources(rl, catalog)
    assert constructions == ["training", "validation"]

    constructions.clear()
    rl.validation.reward = ""
    validate_rollout_reward_sources(rl, catalog)
    assert constructions == ["training"]


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_opd_rejects_remote_training_reward_but_allows_remote_validation(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(json.dumps({"prompt": "validation"}) + "\n", encoding="utf-8")
    catalog = RewardProviderCatalog(
        {
            "local": RewardProviderSpec(factory=_Provider),
            "remote": RewardProviderSpec(
                factory=_Provider,
                semantic_identity={"endpoint": "https://judge.example.test/v1/chat/completions"},
                is_remote=True,
            ),
        }
    )

    with pytest.raises(ValueError, match="direct OPD cannot use remote training reward provider 'remote'"):
        validate_rollout_reward_sources(
            OPDConfig(reward="remote", distillation_mode="direct"),
            catalog,
        )

    evaluation_only = OPDConfig(
        reward="local",
        distillation_mode="direct",
        validation=GRPOValidationConfig(
            enabled=True,
            data=DatasetConfig(path=str(validation_path), prompt_column="prompt"),
            reward="remote",
        ),
    )
    validate_rollout_reward_sources(evaluation_only, catalog)


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_provider_uses_bounded_generic_manifest_contract() -> None:
    prompt_only = DatasetConfig(path="prompts.jsonl", prompt_column="prompt")
    rl = GRPOConfig(
        data=prompt_only,
        reward="custom_prompt",
        reward_provider_catalog=_CUSTOM_CATALOG_PATH,
    )

    recipe = manifest(rl)

    reward_role = next(role for role in recipe.roles if role.name == "rm")
    assert reward_role.impl_key == "grpo_reward_prompt"
    assert bind_check(recipe).ok

    catalog = RewardProviderCatalog({"custom_reference": RewardProviderSpec(factory=_Provider, needs=("reference",))})
    incompatible = GRPOConfig(data=prompt_only, reward="custom_reference")
    report = bind_check(manifest(incompatible, catalog))
    assert not report.ok
    assert any(error.category == "field" and "reference" in error.message for error in report.errors)


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_catalog_drives_source_preflight_and_validation(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(json.dumps({"prompt": "train"}) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps({"prompt": "validation"}) + "\n", encoding="utf-8")
    catalog = resolve_reward_provider_catalog(_CUSTOM_CATALOG_PATH).extend(
        {"custom_reference": RewardProviderSpec(factory=_Provider, needs=("reference",))}
    )
    rl = GRPOConfig(
        data=DatasetConfig(path=str(train_path), prompt_column="prompt"),
        reward="custom_prompt",
        reward_provider_catalog=_CUSTOM_CATALOG_PATH,
    )

    validate_rollout_reward_sources(rl, catalog)

    rl.validation.enabled = True
    rl.validation.data = DatasetConfig(path=str(validation_path), prompt_column="prompt")
    rl.validation.reward = "custom_reference"
    with pytest.raises(ValueError, match="validation reward 'custom_reference'.*requires ground truth"):
        validate_rollout_reward_sources(rl, catalog)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("entrypoint", "rl"),
    [
        (grpo, GRPOConfig(reward="custom_prompt", reward_provider_catalog=_CUSTOM_CATALOG_PATH, group_size=2)),
        (opd, OPDConfig(reward="custom_prompt", reward_provider_catalog=_CUSTOM_CATALOG_PATH, group_size=2)),
    ],
    ids=("grpo", "opd"),
)
def test_online_policy_setup_builds_custom_catalog_provider(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: ModuleType,
    rl: GRPOConfig,
) -> None:
    configured = SimpleNamespace(
        rl=rl,
        model_hf=SimpleNamespace(name_or_path="unused"),
        rollout=SimpleNamespace(name_or_path="unused"),
    )
    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = lambda *_args, **_kwargs: configured
    lazy_config_module.instantiate = lambda value: value
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)

    setup_kwargs = {"build_objective_fn": grpo.build_objective} if entrypoint is grpo else {}
    units = entrypoint.make_setup("unused.py", [], ["rm"], **setup_kwargs)()

    provider = units["rm"].providers[0]
    assert provider.provider_name == "custom_prompt"
    assert provider.provider_revision == "test-v1"
