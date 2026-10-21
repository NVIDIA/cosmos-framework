# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Advantage estimators — the swappable part of a policy-gradient algorithm.

WHY THIS MODULE EXISTS. What distinguishes GRPO from PPO is ONLY the advantage estimator: GRPO
replaces PPO's learned critic + GAE with the group's own reward statistics, `(r - mean)/(std + eps)`.
The surrogate — ratio, epsilon-clip, dual-clip, KL, reduction — is PPO's and is shared by both
through ``algorithm/primitives/policy_gradient.py``.

WHY IT RUNS AT REWARD TIME, NOT IN THE OBJECTIVE. A group-relative advantage is only correct over
the WHOLE group. Computing it inside the objective made that a precondition on whatever batch a rank
happened to receive, which forced every layer downstream to preserve groups atomically: a startup
gate coupling `prompts_per_step` to the rank count, a runtime completeness check, and a dispatch
wrapper that could only round-robin GROUPS (so it refused any batch whose group count did not divide
the rank count). None of that is inherent to GRPO — it is the cost of computing the estimate too late.

At reward time a local placement holds the group intact BY CONSTRUCTION: one rollout call produces a
prompt's G completions, and the reward stage sees exactly that. Online WFM instead places one group
member on each model rank and gathers those reward scalars before normalization. In either shape the
advantage is computed over the complete group, then baked into each trajectory and carried with it,
so later dispatch can split the samples across ranks, replicas or steps. This is what makes cosmos-rl
structurally immune to the same problem (`dispatcher/algo/grpo.py`: advantage computed centrally at
reward time, shipped inside each `Rollout`).

The objective's job is then narrow and honest: read `transition.advantage`, apply the surrogate.

NOT THE ONLY ONE. `algorithm/annotators/prefix_filter.py::shared_prefix_ignore` is the same shape and moved for
the same reason — it compares completions and rewards WITHIN a group, so a rank holding half a group
computes a different loss mask than the group implies. It is the more dangerous of the two, because a
missing advantage raises (`require_advantages`) while a mis-derived prefix mask just quietly trains
on the wrong tokens. Build both through `config/build.py::build_reward_annotators`, which is the one
config -> reward-stage mapping, and hand them to `RewardWorker(annotators=...)`.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Protocol

from cosmos_framework.posttrain.algorithm.reward import RewardContractError
from cosmos_framework.posttrain.algorithm.reward.contract import _terminal_reward
from cosmos_framework.posttrain.schema import Trajectory

if TYPE_CHECKING:  # annotations only — see `advantage_to_tokens` on why torch is not imported here
    import torch

# `algo_extra` keys the estimator stamps onto every transition of a group it normalized. Named
# constants rather than literals because they are a CONTRACT between this module and its consumers
# (`utils/batch_stats.py`), and a typo'd string key fails as a silently-absent value.
GROUP_REWARD_SPREAD = "group_reward_spread"
"""max(rewards) - min(rewards) over the group. 0.0 ⇒ tied ⇒ zero advantage ⇒ no gradient."""

GROUP_REWARD_MEAN = "group_reward_mean"
"""The group's mean reward — the baseline each member's advantage was taken against."""

GROUP_REWARD_NONZERO_FRACTION = "group_reward_nonzero_fraction"
"""Fraction of the group's rewards that are exactly nonzero — reward-density telemetry."""

INFORMATIVE_GROUP_GATE: tuple[str, float] = (GROUP_REWARD_SPREAD, 0.0)
"""Declared ``(terminal algo_extra key, strict lower bound)`` under which a group is informative.

"Informative" is a reward-stage fact — the estimator stamps the complete group's spread — and this
pair is its only definition. `is_informative` reads it for the algorithm side (the reward report's
informative/tied group counts); a buffer evaluates the same pair as a `control.buffer.MetadataGate`
without reconstructing group semantics from rewards (DESIGN §5)."""


def is_informative(trajectory: Trajectory) -> bool:
    """Whether the reward stage judged this trajectory's group informative (reward spread > 0).

    Reads the stamp the estimator wrote while it held the complete group, so the verdict is the
    same on every member and independent of how the group was later split across cells.

    Raises:
        RewardContractError: the stamp is absent or non-numeric — the trajectory never passed the
            group-relative estimator, so no downstream stage may decide for it.
    """
    key, threshold = INFORMATIVE_GROUP_GATE
    value = trajectory.transitions[-1].algo_extra.get(key) if trajectory.transitions else None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RewardContractError(
            f"informative-group check needs terminal algo_extra[{key!r}] from the group-relative "
            f"estimator, got {value!r} on trajectory {trajectory.id!r}"
        )
    return float(value) > threshold


def _group_reward_facts(rewards: list[float]) -> dict[str, float | int]:
    """Return low-cardinality facts computed over one complete reward group."""
    if not rewards:
        raise ValueError("group reward facts require at least one reward")
    return _group_reward_facts_from_values(
        mean=sum(rewards) / len(rewards),
        spread=max(rewards) - min(rewards),
        nonzero_fraction=sum(reward != 0.0 for reward in rewards) / len(rewards),
        group_size=len(rewards),
    )


def _group_reward_facts_from_values(
    mean: float,
    spread: float,
    nonzero_fraction: float,
    group_size: int,
) -> dict[str, float | int]:
    """Build reporter fields from statistics evaluated in the estimator's arithmetic."""
    return {
        "rollout/reward_group_mean": mean,
        "rollout/reward_group_spread": spread,
        "rollout/reward_nonzero_fraction": nonzero_fraction,
        "rollout/reward_group_size": group_size,
        "rollout/reward_group_informative": int(spread > 0.0),
        "rollout/reward_group_all_zero": int(nonzero_fraction == 0.0),
    }


def _wfm_group_relative_values(
    rewards: torch.Tensor,
    *,
    eps: float,
    correction: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:  # rewards: [G], returns advantages: [G]
    """Normalize one complete WFM reward group with the pinned fp32 arithmetic."""
    import torch

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError(f"WFM advantage requires a non-empty reward vector, got shape={tuple(rewards.shape)}")
    if rewards.dtype != torch.float32:
        raise TypeError(f"WFM advantage requires fp32 rewards, got dtype={rewards.dtype}")
    denominator = rewards.numel() - correction
    if denominator <= 0:
        raise ValueError(
            f"advantage correction={correction} requires at least {correction + 1} members, got {rewards.numel()}"
        )

    mean = rewards.mean()  # []
    spread = rewards.max() - rewards.min()  # []
    std = rewards.std(correction=correction)  # []
    advantages = (rewards - mean) / (std + eps)  # [G]
    advantage_std = advantages.std(correction=correction)  # []
    advantage_abs_max = advantages.abs().max()  # []
    nonzero_count = torch.count_nonzero(rewards)  # []
    facts = _group_reward_facts_from_values(
        mean=float(mean.item()),
        spread=float(spread.item()),
        nonzero_fraction=float(nonzero_count.item()) / rewards.numel(),
        group_size=rewards.numel(),
    )
    facts.update(
        {
            "rollout/advantage_group_std": float(advantage_std.item()),
            "rollout/advantage_abs_max": float(advantage_abs_max.item()),
        }
    )
    return advantages, facts


def _apply_wfm_group_relative(
    members: list[Trajectory],
    *,
    eps: float,
    correction: int,
) -> dict[str, float | int]:
    """Apply the shared WFM fp32 normalization to one locally complete group."""
    import torch

    reward_values = [_terminal_reward(member) for member in members]
    rewards = torch.tensor(reward_values, dtype=torch.float32)  # [G]
    advantages, facts = _wfm_group_relative_values(rewards, eps=eps, correction=correction)  # [G], facts
    # The estimator deliberately keeps Cosmos-RL's fp32 arithmetic, while the generic
    # complete-group reporter validates annotations against the original Python scalars.
    _stamp_group_reward_facts(members, _group_reward_facts(reward_values))
    for member, advantage in zip(members, advantages.tolist(), strict=True):
        for transition in member.transitions:
            transition.advantage = float(advantage)
    return facts


def _stamp_group_reward_facts(members: list[Trajectory], facts: dict[str, float | int]) -> None:
    """Attach the complete-group reward facts required by downstream consumers."""
    mean = float(facts["rollout/reward_group_mean"])
    spread = float(facts["rollout/reward_group_spread"])
    nonzero_fraction = float(facts["rollout/reward_nonzero_fraction"])
    for member in members:
        for transition in member.transitions:
            transition.algo_extra[GROUP_REWARD_MEAN] = mean
            transition.algo_extra[GROUP_REWARD_SPREAD] = spread
            transition.algo_extra[GROUP_REWARD_NONZERO_FRACTION] = nonzero_fraction


class AdvantageEstimator(Protocol):
    """Writes `transition.advantage` for every transition of every trajectory it is given.

    Called once per reward batch, before the trajectories are dispatched to the trainer. The
    estimator is where an algorithm's identity lives; swapping it (group-relative -> GAE) is what
    turns GRPO into PPO, with the surrogate unchanged.
    """

    def __call__(self, trajectories: list[Trajectory]) -> None: ...


def group_relative(
    eps: float = 1e-6,
    unbiased: bool = False,
    group_size: int | None = None,
    correction: int = 0,
) -> Callable:
    """GRPO's estimator: normalize each sample's reward within its `group_id`.

    Two variants, matching cosmos-rl's `GRPO(unbiased=...)`:

    - `unbiased=False` (DEFAULT): `A = (r - mean) / (std + eps)` — standard GRPO.
    - `unbiased=True`: `A = r - mean` — mean-centred only, NO std division. Dividing by the group std
      rescales every group to unit variance, so a group that barely disagrees (one lucky success
      among near-identical failures, std ~ 0) yields an advantage as large as one that disagrees
      strongly. That inflates exactly the lowest-information groups — the Dr-GRPO critique.

    Both keep the two properties the surrogate depends on: advantages SUM TO ZERO within a group, and
    an all-equal group yields all-zero advantage (no gradient — intended, not a bug).

    Args:
        eps: denominator guard for the std division. Ignored when `unbiased=True`.
        unbiased: use the mean-centred variant.
        group_size: expected members per group. `None` disables the completeness check. Checking
            HERE is cheap and meaningful — the reward stage holds a whole group by construction, so a
            short group means the rollout or the reward edge lost samples, not that dispatch split
            them. This is the check that used to live at the objective, where it could only observe
            the consequence.
        correction: standard-deviation degrees-of-freedom correction. The GRPO default remains
            population standard deviation (`0`); Cosmos-RL WFM NFT uses sample standard deviation (`1`).

    Returns:
        An `AdvantageEstimator`.

    Raises:
        RewardContractError: any trajectory has `group_id is None`. Checked BEFORE grouping, because
            bucketing on None pools unrelated prompts into one group and the `group_size` check
            cannot see it -- exactly `group_size` id-less trajectories look like a valid group.
        ValueError: a group's member count != `group_size` (when `group_size` is given).
    """

    if correction not in (0, 1):
        raise ValueError(f"advantage correction must be 0 or 1, got {correction}")
    if unbiased and correction:
        raise ValueError("advantage correction is undefined when unbiased=True omits standard-deviation scaling")

    def estimate(trajectories: list[Trajectory]) -> None:
        # A MISSING group_id is rejected before anything is grouped. `Trajectory.group_id` defaults
        # to None, and bucketing on it would collapse every such trajectory into ONE group -- so G
        # completions of G DIFFERENT prompts would be normalized against each other, and the
        # advantage would measure question difficulty rather than completion quality. That is the
        # confound the cosmos-rl comparison had to neutralize by hand (docs/design/grpo-cosmos-rl-parity.md
        # section 2.1); producing it internally is worse.
        #
        # The `group_size` check below CANNOT catch this, which is what makes it dangerous: exactly
        # `group_size` id-less trajectories form one bucket of the right size, so the check passes
        # and reports nothing. A caller who set group_size reasonably believes they are covered.
        #
        # Rejected rather than inferred. There is no honest fallback -- position in the list is not
        # grouping (dispatch may reorder), and `obs.id` is what the rollout ALREADY derives group_id
        # from (`roles/rollout/base.py::group_ids_for_prompt`), so a producer that lost one lost the
        # other. Same standard as `_terminal_reward` above: this decides which rewards are
        # normalized together, so a missing value must raise, not be guessed.
        missing = [t.id or "<no id>" for t in trajectories if t.group_id is None]
        if missing:
            raise RewardContractError(
                f"{len(missing)} trajectory/ies reached the advantage estimator with group_id=None "
                f"(first: {missing[0]!r}). GRPO normalizes rewards WITHIN a group, so an unset "
                f"group_id would silently pool unrelated prompts into one. The rollout sets it from "
                f"the prompt's own id -- see `roles/rollout/base.py::group_ids_for_prompt`, which "
                f"both HFRollout and VLLMRollout use; a custom producer must do the same."
            )
        groups: dict[object, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.group_id, []).append(t)
        if group_size is not None:
            bad = {gid: len(members) for gid, members in groups.items() if len(members) != group_size}
            if bad:
                raise ValueError(
                    f"advantage estimator got incomplete group(s) {bad} (expected {group_size} members "
                    f"each). The group-relative estimate is only correct over the WHOLE group. At the "
                    f"reward stage a group arrives intact by construction (one rollout call produces "
                    f"its G completions), so a short group means the rollout or the reward edge "
                    f"dropped samples — not that dispatch split them."
                )
        for members in groups.values():
            rewards = [_terminal_reward(m) for m in members]
            # Stamp what this group's rewards were, while the group is in hand. The estimator is the
            # only place that legitimately sees a WHOLE group, so anything group-level a downstream
            # consumer needs has to be recorded here or re-derived from a batch that may since have
            # been split across ranks. The old `live_groups` reporting helper re-derived exactly
            # this (regrouping the batch on the driver and recomputing max != min) -- two
            # expressions of one predicate, in two modules, which is the drift this branch fixes.
            facts = _group_reward_facts(rewards)
            _stamp_group_reward_facts(members, facts)
            mean = float(facts["rollout/reward_group_mean"])
            if unbiased:
                for m, r in zip(members, rewards):
                    for tr in m.transitions:
                        tr.advantage = r - mean
                continue
            denominator = len(rewards) - correction
            if denominator <= 0:
                raise ValueError(
                    f"advantage correction={correction} requires at least {correction + 1} members, got {len(rewards)}"
                )
            var = sum((r - mean) ** 2 for r in rewards) / denominator
            std = var**0.5
            for m, r in zip(members, rewards):
                adv = (r - mean) / (std + eps)
                for tr in m.transitions:
                    tr.advantage = adv

    return estimate


class _DistributedGroupRelative:
    """Stateful WFM estimator over a local whole group or one member per FSDP rank."""

    def __init__(self, eps: float, group_size: int, correction: int) -> None:
        self.eps = eps
        self.group_size = group_size
        self.correction = correction
        self._last_metrics: dict[str, float | int] | None = None

    def metrics(self) -> dict[str, float | int]:
        """Return facts from the last complete global group, or nothing after invalidation."""
        return dict(self._last_metrics) if self._last_metrics is not None else {}

    def reset_metrics(self) -> None:
        """Invalidate telemetry before a new reward attempt can fail."""
        self._last_metrics = None

    def __call__(self, trajectories: list[Trajectory]) -> None:
        """Normalize one global group and cache only merge-safe telemetry."""
        import torch
        import torch.distributed as dist

        self.reset_metrics()
        if not dist.is_initialized() or dist.get_world_size() == 1:
            if len(trajectories) != self.group_size:
                raise ValueError(
                    f"advantage estimator got incomplete group with {len(trajectories)} members "
                    f"(expected {self.group_size})"
                )
            group_ids = {trajectory.group_id for trajectory in trajectories}
            if None in group_ids:
                raise ValueError("distributed advantage requires every trajectory to have group_id")
            if len(group_ids) != 1:
                raise ValueError(
                    "distributed advantage telemetry requires one group, got "
                    f"{sorted(repr(group_id) for group_id in group_ids)}"
                )
            self._last_metrics = _apply_wfm_group_relative(
                trajectories,
                eps=self.eps,
                correction=self.correction,
            )
            return

        from cosmos_framework.posttrain.utils.dist import collective_device, raise_if_any_rank_failed, rank_identical

        world_size = dist.get_world_size()
        local_error: Exception | None = None
        local_rewards: list[float] = []
        group_ids: set[str] = set()
        try:
            if self.group_size != world_size:
                raise ValueError(
                    "online NFT distributed advantage requires one member per rank: "
                    f"group_size={self.group_size} must equal world_size={world_size}"
                )
            if len(trajectories) != 1:
                raise ValueError(f"distributed advantage expected 1 local member, got {len(trajectories)}")
            for trajectory in trajectories:
                if trajectory.group_id is None:
                    raise ValueError("distributed advantage requires every trajectory to have group_id")
                group_ids.add(trajectory.group_id)
                local_rewards.append(_terminal_reward(trajectory))
        except Exception as error:  # noqa: BLE001 — converge before reward collectives
            local_error = error
        raise_if_any_rank_failed(local_error, "distributed group-relative advantage preparation")

        rank_identical(next(iter(group_ids)), "distributed advantage group id")

        # Cosmos-RL converts the gathered reward list to fp32 before sample standardization.
        reward_tensor = torch.tensor(local_rewards, dtype=torch.float32, device=collective_device())  # [G_local]
        gathered_rewards = [torch.empty_like(reward_tensor) for _ in range(world_size)]  # W * [G_local]
        dist.all_gather(gathered_rewards, reward_tensor)
        rewards = torch.cat(gathered_rewards, dim=0)  # [G]
        global_advantages, facts = _wfm_group_relative_values(  # [G], facts
            rewards,
            eps=self.eps,
            correction=self.correction,
        )
        rank = dist.get_rank()
        advantages = global_advantages[rank : rank + 1]  # [G_local]
        _stamp_group_reward_facts(trajectories, facts)
        for trajectory, advantage in zip(trajectories, advantages.tolist(), strict=True):
            for transition in trajectory.transitions:
                transition.advantage = float(advantage)
        self._last_metrics = facts


def distributed_group_relative(
    eps: float = 1e-4,
    *,
    group_size: int,
    correction: int = 1,
) -> _DistributedGroupRelative:
    """Normalize one WFM rollout group with Cosmos-RL-compatible fp32 arithmetic.

    Without an initialized process group (or with world size one), the caller supplies the complete
    ``group_size``-member group in one list. With a multi-rank process group, every rank supplies
    exactly one member and ``group_size`` must equal the world size.

    ``eps=1e-4`` and ``correction=1`` are the defaults used by the first Cosmos-RL-compatible
    online-WFM/NFT recipe. They remain configurable here and are intentionally distinct from the
    generic ``group_relative`` estimator's ``1e-6`` and population-standard-deviation defaults.

    ``metrics()`` reports the estimator's pinned fp32 facts. For a locally complete group,
    transition annotations retain the original reward-scalar arithmetic expected by
    ``reward_batch_stats``. In the one-member-per-rank shape those annotations describe the global
    group and must not be validated as if the rank-local singleton were a complete group.
    """
    if isinstance(eps, bool) or not isinstance(eps, Real) or not isfinite(float(eps)) or eps <= 0.0:
        raise ValueError(f"distributed advantage eps must be finite and > 0, got {eps}")
    if group_size < 2:
        raise ValueError(f"distributed advantage group_size must be >= 2, got {group_size}")
    if correction not in (0, 1):
        raise ValueError(f"distributed advantage correction must be 0 or 1, got {correction}")
    return _DistributedGroupRelative(eps=eps, group_size=group_size, correction=correction)


def _advantage_is_finite(value: object) -> bool:
    """True when `value` — a scalar OR a per-token tensor — is entirely finite.

    Duck-typed on `.isfinite()` rather than `isinstance(value, torch.Tensor)`: this runs in the
    reward stage, which has no other reason to import torch. The unconditional `float(value)` this
    replaces raised outright on a multi-element tensor, which is exactly the shape the widened
    advantage channel now allows (`schema/trajectory.py::Transition`).

    A 0-element tensor is vacuously finite. That is the empty-completion case, which the surrogate
    already contributes nothing for.

    Args:
        value: the advantage to check — a real number or a tensor.

    Returns:
        Whether every element of `value` is finite.

    Raises:
        TypeError: `value` is neither a number nor a tensor. The likely cause is an un-materialized
            `BulkRef`: a per-token advantage travels as bulk now, so a consumer that validates
            before `comm/codec.py::materialize_payload` sees the handle instead of the tensor.
    """
    isfinite_elementwise = getattr(value, "isfinite", None)
    if callable(isfinite_elementwise):
        return bool(isfinite_elementwise().all())
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"advantage must be a scalar or a per-token tensor, got {type(value).__name__} "
            f"({value!r}). A BulkRef here means the batch reached the objective un-materialized — "
            f"pull it through `comm/codec.py::materialize_payload` first."
        )
    return isfinite(float(value))


def require_advantages(trajectories: list[Trajectory]) -> None:
    """Fail loud, and TOGETHER, if any transition reaches the surrogate without a usable advantage.

    The objective no longer computes the estimate, so a bad advantage means the reward stage was not
    given an estimator (or a producer bypassed it). Treating `None` as 0.0 would train on a
    zero-gradient batch and report a healthy loss; a non-finite one poisons the whole batch's
    gradient. Both raise.

    Accepts either shape the channel carries — a scalar or a per-token tensor
    (`schema/trajectory.py::Transition`) — and checks finiteness elementwise for the latter.

    CONVERGED ACROSS RANKS, which is the part that is easy to get wrong. `select_chunk` hands
    each rank a DIFFERENT slice now, so one bad trajectory lands on exactly ONE rank. A local raise
    there aborts that rank while its peers walk into the next collective -- the forward-count
    all-reduce, the FSDP all-gathers, the token-count reduction -- and block forever. The run HANGS
    instead of reporting the error, which is strictly worse than the silent zero this check exists to
    prevent. `raise_if_any_rank_failed` makes every rank raise together and short-circuits to a plain
    raise when there is no process group. (This is the same reason the deleted
    `_require_complete_groups` converged; the requirement outlived the check.)

    Args:
        trajectories: the batch about to be scored. Every rank must call this, in the same place.

    Raises:
        ValueError: any transition has `advantage is None` (single-rank).
        RuntimeError: any RANK reported a bad advantage (multi-rank), naming the ranks.
    """
    # Counted, not raised inline. A malformed advantage lands on ONE rank under per-sample sharding,
    # so raising the moment it is seen would abort that rank while its peers walk into the next
    # collective -- the very desync `raise_if_any_rank_failed` below exists to prevent.
    missing = nonfinite = 0
    malformed: list[str] = []
    for t in trajectories:
        for tr in t.transitions:
            value = tr.advantage
            if value is None:
                missing += 1
                continue
            try:
                if not _advantage_is_finite(value):
                    nonfinite += 1
            except TypeError as exc:
                malformed.append(str(exc))
    local_error: Exception | None = None
    # The estimator is CONFIGURED (`config/build.py::build_reward_annotators` is the one config ->
    # reward-stage mapping), so these messages name that seam rather than hard-coding
    # `annotators/advantage.py` -- which is GRPO's estimator and would point every other algorithm's
    # user at a file they never touched.
    if malformed:
        local_error = ValueError(
            f"{len(malformed)} transition(s) reached the objective with an advantage that is "
            f"neither a scalar nor a per-token tensor. First: {malformed[0]}"
        )
    elif nonfinite:
        local_error = ValueError(
            f"{nonfinite} transition(s) reached the objective with a non-finite advantage. The "
            f"configured advantage estimator (see `config/build.py::build_reward_annotators`) "
            f"produced NaN/inf -- check the values it normalized over, and for the group-relative "
            f"estimator `eps` if the group std is zero."
        )
    elif missing:
        local_error = ValueError(
            f"{missing} transition(s) reached the objective with advantage=None. The advantage is "
            f"computed at REWARD time now, by the configured estimator, and travels with the "
            f"trajectory; pass the annotators to the reward worker, e.g. "
            f"RewardWorker(providers, combiner, annotators=build_reward_annotators(rl, name_or_path))."
        )
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    raise_if_any_rank_failed(local_error, "GRPO advantage validation")


def advantage_to_tokens(
    value: Any,
    n_tokens: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor":
    """Expand ONE transition's advantage into the `[n_tokens]` vector the surrogate multiplies.

    The advantage channel carries a scalar OR a per-token tensor
    (`schema/trajectory.py::Transition`), while the surrogate is per-token either way. This is the
    single place the two shapes meet: a scalar broadcasts over the completion, a per-token tensor is
    width-checked and moved/cast. Public because it is the shared expansion: any algorithm reusing the
    clipped policy-gradient surrogate (PPO and OPD's advantage objective) needs exactly this, and a
    second copy is how the two shapes drift apart.

    `dtype` defaults to float32 (spelled `None` so the default costs no import) to match
    `torch.tensor([...])` on a list of Python floats, which is what the scalar path has always
    built; the multiply against a bf16 ratio promotes exactly as before.

    THE `import torch` IS INSIDE THE FUNCTION ON PURPOSE — do not hoist it. This module is written
    to be importable without torch, because the reward stage runs the estimator above and has no
    other reason to pull it in (see `_advantage_is_finite`, which duck-types for the same reason).
    The scalar branch here has to CONSTRUCT a tensor, which cannot be duck-typed, so the dependency
    is real but is paid only by the objectives that call this — never by the reward worker.

    CALL IT FROM A CONVERGED BLOCK. The width check below is rank-local data validation, and under
    per-sample dp sharding a bad trajectory lands on ONE rank — raising it mid-forward aborts that
    rank while its peers block in FSDP all-gathers, which hangs the run instead of reporting the
    error. Both callers here run it during preparation, above the first forward.

    Args:
        value: the transition's `advantage` — a real scalar or a per-token tensor.
        n_tokens: the transition's completion-token count, i.e. what `logprob_masks[:-1]` selects.
        device: device of the returned vector.
        dtype: dtype of the returned vector.

    Returns:
        A `[n_tokens]` advantage vector.

    Raises:
        ValueError: a tensor advantage whose element count is not `n_tokens`. Not broadcast: a
            1-element tensor would silently spread over the whole completion, and an off-by-one one
            would misalign every token after the gap rather than fail.
    """
    import torch  # see the note above: deferred so importing this module stays torch-free

    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() != n_tokens:
            raise ValueError(
                f"per-token advantage has {flat.numel()} element(s) but its completion has "
                f"{n_tokens} scored token(s). A per-token advantage is aligned to the positions "
                f"`logprob_masks[:-1]` selects — one value per token the loss is taken over."
            )
        return flat.to(device=device, dtype=dtype or torch.float32)
    return torch.full((n_tokens,), float(value), device=device, dtype=dtype or torch.float32)
