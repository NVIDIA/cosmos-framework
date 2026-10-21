# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Group-native reward providers, composition, and run-bindable catalogs (DESIGN §1.5)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from math import isfinite
from types import MappingProxyType
from typing import Iterator, Protocol

from cosmos_framework.posttrain.algorithm.reward.arithmetic import arithmetic_reward_fn
from cosmos_framework.posttrain.algorithm.reward.contract import RewardContractError, _scalar_answer_text
from cosmos_framework.posttrain.algorithm.reward.countdown import countdown_reward_fn
from cosmos_framework.posttrain.algorithm.reward.group import (
    RewardCandidate,
    RewardCandidateResult,
    RewardContext,
    RewardGroupRequest,
    RewardGroupResult,
    build_reward_group_request,
)
from cosmos_framework.posttrain.algorithm.reward.gsm8k import gsm8k_reward_fn
from cosmos_framework.posttrain.algorithm.reward.result import RewardProviderResult
from cosmos_framework.posttrain.schema import Trajectory


class RewardProvider(Protocol):
    """The sole online reward runtime contract."""

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        """Return one explicitly correlated result for every candidate in the group."""
        ...


CandidateRewardFn = Callable[[RewardCandidate, RewardContext], RewardProviderResult]


class _MappedCandidateProvider:
    """Lift a pointwise scorer into the canonical group runtime contract."""

    _score_candidate: CandidateRewardFn

    def __init__(self, score_candidate: CandidateRewardFn) -> None:
        self._score_candidate = score_candidate

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        """Map the pointwise scorer over candidates while preserving explicit identity."""
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=self._score_candidate(candidate, request.context),
                )
                for candidate in request.candidates
            ),
        )


def map_candidates(score_candidate: CandidateRewardFn) -> RewardProvider:
    """Lift an immutable pointwise scorer into the canonical group provider contract."""
    if not callable(score_candidate):
        raise TypeError(f"candidate reward scorer must be callable, got {score_candidate!r}")
    return _MappedCandidateProvider(score_candidate)


class RewardCombiner:
    """Reduce named components → final reward (DESIGN §1.5).

    v0: weighted sum (identity if no weights). TODO: KL penalty term, shaping, etc.
    """

    weights: dict[str, float]

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def combine(self, traj: Trajectory, components: Mapping[str, float]) -> float:
        if not self.weights:
            result = sum(components.values())
        else:
            result = sum(self.weights.get(k, 1.0) * v for k, v in components.items())
        if not isfinite(result):
            raise RewardContractError(f"reward combiner returned non-finite final value {result!r}")
        return 0.0 if not components else result


def exact_match_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Whitespace/case-insensitive match of the completion against the ground-truth answer."""
    completion = candidate.completion.strip().lower()
    answer = _scalar_answer_text(context.reference).strip().lower()
    return RewardProviderResult(terms={"task": 1.0 if completion == answer else 0.0})


def nonempty_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Connectivity/smoke reward for a non-blank completion; not a general optimization objective."""
    del context
    return RewardProviderResult(terms={"task": 1.0 if candidate.completion.strip() else 0.0})


def constant_reward_fn(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Return zero without inspecting the candidate, including for an empty completion.

    Pure-distillation OPD needs the group-relative task term to be exactly zero. Using the
    connectivity-only ``nonempty`` reward almost achieves that, but one empty completion changes a
    group's reward distribution and injects task advantage into what the run reports as pure
    distillation. A literal finite zero makes every member equal by construction.
    """
    del candidate, context
    return RewardProviderResult(terms={"task": 0.0})


@dataclass(frozen=True)
class RuleRewardSpec:
    """A built-in rule reward and the observation fields it requires."""

    fn: CandidateRewardFn
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardProviderSpec:
    """A provider factory plus its run-bindable fixed output contract."""

    factory: Callable[[], RewardProvider]
    needs: tuple[str, ...] = ()
    revision: str = "builtin-v1"
    diagnostic_names: tuple[str, ...] = ()
    term_names: tuple[str, ...] = ("task",)
    semantic_identity: Mapping[str, object] = field(default_factory=dict)
    is_remote: bool = False
    token_term_names: tuple[str, ...] = ()
    identity_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError(f"reward provider factory must be callable, got {self.factory!r}")
        object.__setattr__(self, "needs", _validated_name_tuple(self.needs, field="needs", allow_empty=True))
        object.__setattr__(
            self,
            "diagnostic_names",
            _validated_name_tuple(self.diagnostic_names, field="diagnostic_names", allow_empty=True),
        )
        object.__setattr__(
            self,
            "term_names",
            _validated_name_tuple(self.term_names, field="term_names", allow_empty=True),
        )
        object.__setattr__(
            self,
            "token_term_names",
            _validated_name_tuple(self.token_term_names, field="token_term_names", allow_empty=True),
        )
        if not self.term_names and not self.token_term_names:
            raise ValueError("reward providers must declare at least one scalar or token term")
        if "final" in self.term_names:
            raise ValueError("reward provider term_names cannot contain reserved term 'final'")
        if "final" in self.token_term_names:
            raise ValueError("reward provider token_term_names cannot contain reserved term 'final'")
        collisions = sorted(set(self.term_names).intersection(self.token_term_names))
        if collisions:
            raise ValueError(f"reward providers cannot declare names as both scalar and token terms: {collisions}")
        if "sample_count" in self.diagnostic_names:
            raise ValueError("reward provider diagnostic name 'sample_count' is reserved for the framework")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError(f"reward provider revision must be a non-empty string, got {self.revision!r}")
        if not isinstance(self.is_remote, bool):
            raise TypeError(f"reward provider is_remote must be a bool, got {self.is_remote!r}")
        semantic_identity = _freeze_semantic_identity(self.semantic_identity)
        object.__setattr__(self, "semantic_identity", semantic_identity)
        object.__setattr__(self, "identity_hash", _semantic_identity_hash(semantic_identity))


RULE_REWARDS: Mapping[str, RuleRewardSpec] = MappingProxyType(
    {
        "arithmetic": RuleRewardSpec(arithmetic_reward_fn, needs=("reference",)),
        # Registered here rather than passed ad hoc so `rl.reward="gsm8k"` is selectable like any
        # other built-in and binds through the same declared-reference capability contract.
        "gsm8k": RuleRewardSpec(gsm8k_reward_fn, needs=("reference",)),
        # Countdown's reference is STRUCTURED ({'nums': [...], 'target': N}), unlike every other
        # built-in's scalar -- the verifier must check both which numbers were used and what they
        # reached. `needs` still names the one obs field, which is what bind_check reads; the SHAPE
        # is the verifier's contract (algorithm/reward/countdown.py::countdown_reference).
        "countdown": RuleRewardSpec(countdown_reward_fn, needs=("reference",)),
        "exact_match": RuleRewardSpec(exact_match_reward_fn, needs=("reference",)),
        "nonempty": RuleRewardSpec(nonempty_reward_fn),
        "constant": RuleRewardSpec(constant_reward_fn),
    }
)

_SAFE_PROVIDER_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _freeze_json_value(value: object, *, path: str, active: set[int]) -> object:
    """Deep-freeze one finite JSON-compatible value with an actionable path on failure."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"reward provider semantic_identity value at {path} must be finite, got {value!r}")
        return value

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"reward provider semantic_identity contains a cycle at {path}")
        invalid_key = next((key for key in value if not isinstance(key, str)), None)
        if invalid_key is not None:
            raise TypeError(
                f"reward provider semantic_identity object keys must be strings at {path}, got {invalid_key!r}"
            )
        active.add(marker)
        try:
            frozen = {key: _freeze_json_value(value[key], path=f"{path}.{key}", active=active) for key in sorted(value)}
        finally:
            active.remove(marker)
        return MappingProxyType(frozen)

    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError(f"reward provider semantic_identity contains a cycle at {path}")
        active.add(marker)
        try:
            return tuple(
                _freeze_json_value(item, path=f"{path}[{index}]", active=active) for index, item in enumerate(value)
            )
        finally:
            active.remove(marker)

    raise TypeError(
        f"reward provider semantic_identity values must be JSON-compatible; got {type(value).__name__} at {path}"
    )


def _freeze_semantic_identity(value: object) -> Mapping[str, object]:
    """Validate and deeply freeze the object-shaped semantic identity."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"reward provider semantic_identity must be a JSON-compatible mapping, got {type(value).__name__}"
        )
    frozen = _freeze_json_value(value, path="$", active=set())
    if not isinstance(frozen, Mapping):
        raise AssertionError("mapping semantic identity did not freeze to a mapping")
    return frozen


def _thaw_json_value(value: object) -> object:
    """Convert the frozen representation into standard containers for canonical JSON."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _semantic_identity_hash(semantic_identity: Mapping[str, object]) -> str:
    """Return a stable SHA-256 over canonical UTF-8 JSON identity bytes."""
    canonical = json.dumps(
        _thaw_json_value(semantic_identity),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_name_tuple(values: tuple[str, ...], *, field: str, allow_empty: bool) -> tuple[str, ...]:
    """Freeze and validate one provider-local name sequence."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"reward provider {field} must be a sequence of names, got {values!r}")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"reward provider {field} must be a sequence of names, got {values!r}") from error
    if not allow_empty and not normalized:
        raise ValueError(f"reward provider {field} must not be empty")
    invalid = [name for name in normalized if not isinstance(name, str) or _SAFE_PROVIDER_NAME.fullmatch(name) is None]
    if invalid:
        raise ValueError(f"reward provider {field} names must match '[a-z][a-z0-9_]*', got {invalid!r}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"reward provider {field} names must be unique, got {normalized!r}")
    return normalized


class BoundRewardProvider:
    """A catalog-built provider carrying fixed provenance and output-schema metadata."""

    _evaluate: Callable[[RewardGroupRequest], RewardGroupResult]
    _term_names: frozenset[str]
    _token_term_names: frozenset[str]
    _diagnostic_names: frozenset[str]
    provider_name: str
    provider_revision: str
    diagnostic_names: tuple[str, ...]
    term_names: tuple[str, ...]
    token_term_names: tuple[str, ...]
    identity_hash: str
    is_remote: bool
    needs: tuple[str, ...]
    semantic_identity: Mapping[str, object]

    def __init__(
        self,
        provider_name: str,
        spec: RewardProviderSpec,
        provider: RewardProvider,
    ) -> None:
        evaluate = getattr(provider, "evaluate", None)
        if not callable(evaluate):
            raise TypeError(
                f"reward provider factory for {provider_name!r} returned {type(provider).__name__} without evaluate()"
            )
        self._evaluate = evaluate
        self._term_names = frozenset(spec.term_names)
        self._token_term_names = frozenset(spec.token_term_names)
        self._diagnostic_names = frozenset(spec.diagnostic_names)
        self.provider_name = provider_name
        self.provider_revision = spec.revision
        self.diagnostic_names = spec.diagnostic_names
        self.term_names = spec.term_names
        self.token_term_names = spec.token_term_names
        self.identity_hash = spec.identity_hash
        self.is_remote = spec.is_remote
        self.needs = spec.needs
        self.semantic_identity = spec.semantic_identity

    def prepare_group(self, trajectories: Sequence[Trajectory]) -> RewardGroupRequest:
        """Validate complete trajectories and build one provider-facing request without executing it."""
        return build_reward_group_request(trajectories, needs=self.needs)

    def evaluate_prepared_group(self, request: RewardGroupRequest) -> RewardGroupResult:
        """Execute and validate a request previously prepared at the framework boundary."""
        if not isinstance(request, RewardGroupRequest):
            raise TypeError(f"group reward request must be RewardGroupRequest, got {type(request).__name__}")
        result = self._evaluate(request)
        if not isinstance(result, RewardGroupResult):
            raise RewardContractError(
                f"group reward provider {self.provider_name!r} returned {type(result).__name__}; "
                "expected RewardGroupResult"
            )

        expected_ids = tuple(candidate.trajectory_id for candidate in request.candidates)
        result_ids = tuple(item.trajectory_id for item in result.items)
        if result.group_id != request.group_id:
            raise RewardContractError(
                f"group reward provider {self.provider_name!r} returned group_id {result.group_id!r}; "
                f"expected {request.group_id!r}"
            )
        if len(result_ids) != len(expected_ids):
            raise RewardContractError(
                f"group reward provider {self.provider_name!r} returned {len(result_ids)} candidates; "
                f"expected {len(expected_ids)}"
            )
        if set(result_ids) != set(expected_ids):
            actual_ids = set(result_ids)
            expected_id_set = set(expected_ids)
            raise RewardContractError(
                f"group reward provider {self.provider_name!r} returned different trajectory identities: "
                f"missing={sorted(expected_id_set - actual_ids)}, extra={sorted(actual_ids - expected_id_set)}"
            )
        if result_ids != expected_ids:
            raise RewardContractError(
                f"group reward provider {self.provider_name!r} returned trajectory_ids out of order: "
                f"expected {expected_ids!r}, got {result_ids!r}"
            )
        for item in result.items:
            self._validate_result(item.result)
        return result

    def _validate_result(self, result: object) -> RewardProviderResult:
        """Enforce the catalog's fixed per-candidate term and diagnostic schemas."""
        if not isinstance(result, RewardProviderResult):
            raise RewardContractError(
                f"reward provider {self.provider_name!r} returned {type(result).__name__}; "
                "expected RewardProviderResult"
            )
        actual_terms = set(result.terms)
        if actual_terms != self._term_names:
            raise RewardContractError(
                f"reward provider {self.provider_name!r} term schema differs from catalog: "
                f"missing={sorted(self._term_names - actual_terms)}, "
                f"undeclared={sorted(actual_terms - self._term_names)}"
            )
        actual_token_terms = set(result.token_terms)
        if actual_token_terms != self._token_term_names:
            raise RewardContractError(
                f"reward provider {self.provider_name!r} token term schema differs from catalog: "
                f"missing={sorted(self._token_term_names - actual_token_terms)}, "
                f"undeclared={sorted(actual_token_terms - self._token_term_names)}"
            )
        undeclared_diagnostics = set(result.diagnostics).difference(self._diagnostic_names)
        if undeclared_diagnostics:
            raise RewardContractError(
                f"reward provider {self.provider_name!r} emitted undeclared diagnostic(s) "
                f"{sorted(undeclared_diagnostics)}; declared: {sorted(self._diagnostic_names)}"
            )
        return result


class RewardProviderCatalog(Mapping[str, RewardProviderSpec]):
    """An immutable provider mapping that can be extended only without name collisions."""

    _providers: Mapping[str, RewardProviderSpec]

    def __init__(self, providers: Mapping[str, RewardProviderSpec] | None = None) -> None:
        if providers is None:
            providers = {}
        if not isinstance(providers, Mapping):
            raise TypeError(f"reward provider catalog must be a mapping, got {type(providers).__name__}")
        normalized: dict[str, RewardProviderSpec] = {}
        for name, spec in providers.items():
            if not isinstance(name, str) or _SAFE_PROVIDER_NAME.fullmatch(name) is None:
                raise ValueError(f"reward provider names must match '[a-z][a-z0-9_]*', got {name!r}")
            if not isinstance(spec, RewardProviderSpec):
                raise TypeError(f"reward provider catalog entry {name!r} must be RewardProviderSpec, got {spec!r}")
            normalized[name] = spec
        self._providers = MappingProxyType(normalized)

    def __getitem__(self, name: str) -> RewardProviderSpec:
        return self._providers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._providers)

    def __len__(self) -> int:
        return len(self._providers)

    def extend(self, providers: Mapping[str, RewardProviderSpec]) -> RewardProviderCatalog:
        """Return a new catalog, rejecting overrides of any existing provider."""
        additions = RewardProviderCatalog(providers)
        collisions = sorted(set(self).intersection(additions))
        if collisions:
            raise ValueError(f"reward provider catalog cannot override existing provider(s): {collisions}")
        return RewardProviderCatalog({**self._providers, **additions._providers})

    def spec(self, name: str) -> RewardProviderSpec:
        """Resolve one provider spec with an actionable unknown-name failure."""
        if name not in self._providers:
            raise KeyError(f"unknown reward provider {name!r}; registered: {sorted(self._providers)}")
        return self._providers[name]

    def build(self, name: str) -> BoundRewardProvider:
        """Construct and bind one provider to its fixed catalog contract."""
        spec = self.spec(name)
        return BoundRewardProvider(name, spec, spec.factory())


DEFAULT_REWARD_PROVIDER_CATALOG = RewardProviderCatalog(
    {
        name: RewardProviderSpec(factory=partial(map_candidates, spec.fn), needs=spec.needs)
        for name, spec in RULE_REWARDS.items()
    }
)


def bind_candidate_provider(
    provider_name: str,
    score_candidate: CandidateRewardFn,
    *,
    needs: tuple[str, ...] = (),
    term_names: tuple[str, ...] = ("task",),
    token_term_names: tuple[str, ...] = (),
    diagnostic_names: tuple[str, ...] = (),
    revision: str = "inline-v1",
) -> BoundRewardProvider:
    """Bind one pointwise scorer without requiring a one-entry catalog at the call site."""
    return RewardProviderCatalog(
        {
            provider_name: RewardProviderSpec(
                factory=partial(map_candidates, score_candidate),
                needs=needs,
                revision=revision,
                diagnostic_names=diagnostic_names,
                term_names=term_names,
                token_term_names=token_term_names,
            )
        }
    ).build(provider_name)


def resolve_reward_provider_catalog(factory_path: str = "") -> RewardProviderCatalog:
    """Load one optional user catalog factory and merge it with immutable built-ins."""
    if not isinstance(factory_path, str):
        raise TypeError(f"reward_provider_catalog must be a dotted callable path string, got {factory_path!r}")
    if not factory_path:
        return DEFAULT_REWARD_PROVIDER_CATALOG

    # Keep the reward contract import-light. This utility imports torch, but only driver/actor setup
    # needs dynamic authoring; importing algorithm.reward for a manifest descriptor should not.
    from cosmos_framework.utils.serialization import load_callable

    try:
        factory = load_callable(factory_path)
    except (AssertionError, AttributeError, ImportError, ModuleNotFoundError) as error:
        raise ValueError(f"cannot load reward provider catalog factory {factory_path!r}: {error}") from error
    if not callable(factory):
        raise TypeError(f"reward provider catalog factory {factory_path!r} is not callable")
    custom = factory()
    if isinstance(custom, RewardProviderCatalog):
        additions: Mapping[str, RewardProviderSpec] = custom
    elif isinstance(custom, Mapping):
        additions = custom
    else:
        raise TypeError(
            f"reward provider catalog factory {factory_path!r} must return a mapping or RewardProviderCatalog, "
            f"got {type(custom).__name__}"
        )
    return DEFAULT_REWARD_PROVIDER_CATALOG.extend(additions)


def rule_reward_spec(name: str) -> RuleRewardSpec:
    """Resolve an immutable built-in rule reward; an unknown name lists the available ones."""
    if name not in RULE_REWARDS:
        raise KeyError(f"unknown reward {name!r}; registered: {sorted(RULE_REWARDS)}")
    return RULE_REWARDS[name]
