# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared update-window runtime for PPO trainable roles.

PPO owns no optimizer engine of its own: both trainable roles are ``LLMTrainer`` engines driving an
injected PPO ``Objective``. What lives here is only what the policy and critic share and the engine
cannot own: cache validation and the nested drivers that turn the entry's one-call-per-batch
contract into engine ``train`` calls:

* `run_ppo_batch_update` expands one `PPOUpdatePlan` into this rank's epochs and global minibatches,
  from the role's own live model-DP coordinate and its own cached shard;
* `run_ppo_optimizer_window` turns one of those global minibatches into the engine's
  one-call-per-micro-batch ``train`` and closes the accumulation window exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch

from cosmos_framework.posttrain.algorithm.ppo.batch import PPOMinibatch, PPOUpdatePlan, ppo_minibatches
from cosmos_framework.posttrain.algorithm.ppo.objective import (
    WINDOW_SUMMED,
    WINDOW_WEIGHT_KEY,
    combine_window_metrics,
)
from cosmos_framework.posttrain.algorithm.ppo.records import PPORoleUpdateReceipt
from cosmos_framework.posttrain.model.ppo import prepare_deterministic_ppo_update
from cosmos_framework.posttrain.roles.trainer.batch_cache import BatchCache
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged


class PPOTrainableRole(LLMTrainer):
    """Shared PPO policy/critic hooks over the existing ``LLMTrainer`` engine."""

    _ppo_role: Literal["policy", "critic"]
    _parallel_dims: Any
    _ppo_cache: BatchCache[Trajectory]

    def _initialize_ppo_role(
        self,
        role: Literal["policy", "critic"],
        parallel_dims: Any,
        *,
        digest: Callable[[Trajectory], str],
    ) -> None:
        """Bind role-local cache and live model-DP coordinates after engine construction."""
        self._ppo_role = role
        self._parallel_dims = parallel_dims
        self._ppo_cache = BatchCache(role, digest=digest)

    def _prepare_update_model(self) -> None:
        """Select PPO's deterministic update forward instead of the engine's plain ``train()``."""
        prepare_deterministic_ppo_update(self.model)

    def _begin_update_attempt(self) -> None:
        """Converge gradient reset before the PPO objective's first collective."""
        begin = super()._begin_update_attempt
        run_rank_converged(begin, f"{self._ppo_role} update gradient reset")

    def _validate_post_optimizer_step(self) -> None:
        """Reject a mutation that produced non-finite weights or optimizer state."""
        validate_post_step_finite(self.model, self.opt)

    def _scheduler_clock_name(self) -> Literal["batch", "optimizer_step"]:
        """Advance the schedule once per PPO minibatch optimizer step."""
        return "optimizer_step"

    @property
    def optimizer_step(self) -> int:
        """Return the durable minibatch optimizer counter owned by the shared engine."""
        return self._iter


def validate_ppo_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return scalar PPO metrics after rejecting non-finite values."""
    values: dict[str, float] = {}
    for name, value in metrics.items():
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"PPO metric {name!r} is non-finite: {value!r}")
        values[name] = scalar
    return values


def validate_post_step_finite(model: Any, optimizer: torch.optim.Optimizer) -> None:
    """Collectively reject non-finite parameters or optimizer state after a step.

    This is the PPO engine's ``_validate_post_optimizer_step`` hook body: the shared trainer runs it
    inside its partial-update stage, so a failure here poisons the trainer instead of letting an
    already-mutated optimizer look retryable.
    """
    # This module is imported while the eager trainer package is initializing; defer the shared
    # primitive import until the role call to avoid that package cycle.
    from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_model_optimizer_finite

    net = getattr(model, "net", model)
    if not isinstance(net, torch.nn.Module):
        raise TypeError(f"PPO model must expose a torch.nn.Module, got {type(net).__name__}")
    parameters = list(net.parameters())
    if not parameters:
        raise ValueError("PPO post-step validation requires model parameters")
    if not all_ranks_model_optimizer_finite(net, optimizer):
        raise ValueError("PPO post-step model or optimizer state contains non-finite values")


def _window_scalar(value: Any, label: str) -> float | int:
    """Coerce one already-reduced engine metric into a plain scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"PPO window metric {label!r} must be scalar, got shape {tuple(value.shape)!r}")
        return float(value.detach().cpu())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"PPO window metric {label!r} must be a real scalar, got {value!r}")
    return value


def run_ppo_optimizer_window(
    trainer: Any,
    microbatches: Sequence[Sequence[Trajectory]],
    *,
    role: str,
    weighted: frozenset[str],
    weight_key: str = WINDOW_WEIGHT_KEY,
) -> dict[str, float | int]:
    """Drive exactly ONE shared-engine optimizer step over ``microbatches``.

    The controller's contract is one global minibatch per call; the engine's ``train`` is one
    MICRO-batch per call and closes its accumulation window on the last one. Setting ``grad_accum``
    to the micro-batch count is therefore what makes the window close exactly once here, and the
    engine owns everything inside it: finite checks, clipping, ``opt.step``, the scheduler, the
    post-step validation hook, and the durable optimizer counter.

    Every ``train`` call returns metrics that are already data-parallel reduced for ITS chunk, so the
    whole-window value is recovered by re-weighting those per-chunk means rather than by a second
    collective (`algorithm/ppo/objective.py::combine_window_metrics`). The gradient norm comes from
    the applied chunk, which is the only one that has it; the reported learning rate is read back
    AFTER the window closes, so it is the schedule's post-step value for this update rather than the
    pre-step value the engine stamps into its own per-chunk telemetry.

    The microbatch count must agree across ranks before the first collective. Local sample
    counts may differ: the objective pads its forwards with zero contribution and reduces by
    valid learning positions. Neither padding nor a transport fragment advances an optimizer.

    Args:
        trainer: the role, an ``LLMTrainer`` engine carrying the injected PPO objective.
        microbatches: this rank's disjoint micro-batches of prepared trajectories.
        role: ``"policy"`` or ``"critic"``; names the role-scoped controller metrics.
        weighted: public objective metric keys to recombine as token-weighted window means.

    Returns:
        The role's controller-facing metric schema plus ``f"{role}_optimizer_step"``.

    Raises:
        ValueError: no microbatches; globally empty work is rejected by the objective.
        RuntimeError: the rank schedules disagree, or the window did not close on one applied step.
    """
    if not microbatches:
        raise ValueError(f"{role} update received no microbatches")
    schedule = len(microbatches)
    rank_identical(schedule, f"{role} update schedule")
    trainer.grad_accum = len(microbatches)
    weight_map = trainer.training_metric_weight_map()
    weight_groups: dict[str, frozenset[str]] = {}
    for key in weighted:
        group = weight_map.get(key, weight_key)
        weight_groups[group] = weight_groups.get(group, frozenset()) | {key}
    needed = weighted | WINDOW_SUMMED | set(weight_groups)
    chunks: list[dict[str, float | int]] = []
    applied: Mapping[str, Any] = {}
    try:
        for microbatch in microbatches:
            applied = trainer.train(list(microbatch))
            chunks.append({key: _window_scalar(value, key) for key, value in applied.items() if key in needed})
        # The window closes on the LAST call, so only that one carries the applied step's optimizer
        # telemetry; the accumulating calls never reach the clip-and-step block that produces it.
        if "train/learning_rate" not in applied:
            raise RuntimeError(f"{role} update window did not close on an applied optimizer step")
    except BaseException:
        # Anything that escapes mid-window leaves accumulated gradients and an open attempt behind.
        # Without discarding them the NEXT update would skip its gradient reset, close early on a
        # mixture of two minibatches, and only then report a misleading "window did not close".
        trainer.abort_update_window()
        raise
    window = {}
    for group, names in weight_groups.items():
        if sum(chunk[group] for chunk in chunks) == 0:
            # A fixed episode's padded tail can occupy an entire minibatch. Its
            # masked diagnostics are zero while loss still divides by all slots.
            window.update(dict.fromkeys(names, 0.0))
        else:
            window.update(combine_window_metrics(chunks, weighted=names, weight_key=group))
    metrics = {key.removeprefix("train/"): value for key, value in window.items() if key in weighted}
    # POST-step learning rate: the controller's `optim/{role}_learning_rate` records the schedule
    # position this update LEFT BEHIND, which is what the next update will train at. The engine's
    # own `train/learning_rate` is captured before `opt.step`, so it cannot be reused here.
    metrics[f"{role}_lr"] = _window_scalar(trainer.opt.param_groups[0]["lr"], f"{role}_lr")
    if "train/grad_norm" in applied:
        metrics[f"{role}_grad_norm"] = _window_scalar(applied["train/grad_norm"], "train/grad_norm")
    return {
        **validate_ppo_metrics(metrics),
        f"{role}_optimizer_step": trainer.optimizer_step,
    }


def run_cached_ppo_minibatch(
    trainer: Any,
    cache: BatchCache[Trajectory],
    batch_id: str,
    microbatches: Sequence[Sequence[str]],
    *,
    role: str,
    requested_coefficients: Mapping[str, float],
    bound_coefficients: Mapping[str, float],
    weighted: frozenset[str],
    weight_key: str = WINDOW_WEIGHT_KEY,
) -> dict[str, float | int]:
    """Validate one ID-only minibatch and drive its shared trainer window."""

    def resolve() -> list[tuple[Trajectory, ...]]:
        if role not in ("policy", "critic"):
            raise ValueError(f"unsupported PPO trainable role {role!r}")
        if set(requested_coefficients) != set(bound_coefficients):
            raise ValueError(f"{role} update coefficient schema differs from the constructed objective")
        mismatched = {
            name: (requested, bound_coefficients[name])
            for name, requested in requested_coefficients.items()
            if float(requested) != float(bound_coefficients[name])
        }
        if mismatched:
            detail = ", ".join(
                f"{name}={requested!r} (objective has {bound!r})"
                for name, (requested, bound) in sorted(mismatched.items())
            )
            raise ValueError(f"{role} update coefficients differ from the constructed objective: {detail}")
        if not microbatches:
            raise ValueError(f"{role} update received no microbatches")
        selected: list[tuple[Trajectory, ...]] = []
        seen: set[str] = set()
        for episode_ids in microbatches:
            trajectories = cache.select(batch_id, episode_ids) if episode_ids else ()
            overlap = seen & {trajectory.id for trajectory in trajectories}
            if overlap:
                raise ValueError(f"{role} cached microbatches duplicate episode IDs {sorted(overlap)!r}")
            seen.update(trajectory.id for trajectory in trajectories)
            selected.append(trajectories)
        return selected

    resolved = run_rank_converged(resolve, f"{role} update request validation")
    return run_ppo_optimizer_window(trainer, resolved, role=role, weighted=weighted, weight_key=weight_key)


def model_dp_coordinate(parallel_dims: Any) -> tuple[int, int]:
    """This role rank's LIVE flattened model-DP coordinate ``(dp_rank, dp_world)``.

    The model's own ``ParallelDims`` is authoritative because it is the only object that knows how
    the role's K-rank torch world factors into ``dp_replicate x dp_shard`` — the same coordinate the
    ingress ``select`` tier resolves actor-side (`workers/ray.py::_resolve_dispatched`). Placement's
    ``(replica, rank)`` is deliberately NOT consulted: physical topology must not own data semantics.

    Without a ``ParallelDims`` — a direct torchrun/spawn construction, which is how every
    distributed PPO test builds these roles — the default process group IS the model world, so its
    rank and size are that coordinate. This is the same fallback the backend-less dispatch path
    already applies (`workers/worker_group.py::_local_select`).

    Args:
        parallel_dims: the role's live ``ParallelDims``, or ``None``.

    Returns:
        ``(dp_rank, dp_world)``; ``(0, 1)`` when the role owns no data-parallel axis at all.
    """
    if parallel_dims is not None:
        from cosmos_framework.posttrain.config.build import _dp_coords

        dp_rank, dp_world = _dp_coords(parallel_dims)
        if dp_world > 1:
            return int(dp_rank), int(dp_world)
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def run_ppo_batch_update(
    *,
    role: str,
    batch_id: str,
    plan: PPOUpdatePlan,
    global_episode_ids: Sequence[str],
    cached_episode_ids: Sequence[str],
    parallel_dims: Any,
    optimizer_step: Callable[[], int],
    update: Callable[[list[list[str]]], Mapping[str, Any]],
) -> PPORoleUpdateReceipt:
    """Run ONE trainable role's complete update window over one prepared batch.

    This is the body the PPO controller used to own. It moved here because every input it needs is
    role-private: the role's live model-DP coordinate, the contiguous shard of the global episode
    order its cache holds, and its own durable optimizer counter. A controller that split the work
    by minibatch INDEX over a shuffled global schedule could not see either of the first two, and
    with ``dp_world > 1`` it handed each rank episodes that lived in a different rank's cache.

    The order here is fail-closed by design. The DP coordinate is resolved, the shard it implies is
    checked against what the cache actually holds, and the schedule length is agreed across ranks
    BEFORE the first optimizer window opens — so a placement/ingress disagreement raises instead of
    stepping the optimizer on somebody else's episodes or deadlocking half the world.

    Args:
        role: ``"policy"`` or ``"critic"``.
        batch_id: the prepared batch this window consumes.
        plan: the controller's schedule parameters (see `algorithm/ppo/batch.py::PPOUpdatePlan`).
        global_episode_ids: the batch's global episode order, as this role recorded it.
        cached_episode_ids: what this rank's batch cache holds, in install order.
        parallel_dims: the role's live ``ParallelDims`` (may be ``None``).
        optimizer_step: reads the role's durable optimizer counter.
        update: applies one global minibatch from ID-only micro-batches and returns its metrics.

    Returns:
        The controller-facing `PPORoleUpdateReceipt` for the whole window.

    Raises:
        RuntimeError: the cache does not hold exactly the shard this DP coordinate owns, or the
            role's optimizer counter did not advance by exactly one per global minibatch.
        ValueError: the role returned a different metric schema across global minibatches, or a
            non-finite metric.
    """
    if role not in ("policy", "critic"):
        raise ValueError(f"unsupported PPO trainable role {role!r}")

    import torch.distributed as dist

    shard_sizes = None
    if dist.is_available() and dist.is_initialized():
        sizes: list[int] = [0] * dist.get_world_size()
        dist.all_gather_object(sizes, len(cached_episode_ids))
        shard_sizes = tuple(sizes)

    def derive() -> tuple[PPOMinibatch, ...]:
        from cosmos_framework.posttrain.workers.dispatch import select_model_dp

        if not isinstance(plan, PPOUpdatePlan):
            raise TypeError(f"{role} PPO update requires a PPOUpdatePlan, got {type(plan).__name__}")
        dp_rank, dp_world = model_dp_coordinate(parallel_dims)
        if shard_sizes is None:
            expected_local = tuple(select_model_dp(list(global_episode_ids), dp_rank=dp_rank, dp_world=dp_world))
        else:
            if len(shard_sizes) != dp_world or sum(shard_sizes) != len(global_episode_ids):
                raise ValueError("PPO cache shard sizes disagree with the global sample order")
            offset = sum(shard_sizes[:dp_rank])
            expected_local = tuple(global_episode_ids[offset : offset + shard_sizes[dp_rank]])
        if not expected_local:
            raise ValueError(f"{role} PPO update has an empty local model-DP shard")
        if len(set(global_episode_ids)) != len(global_episode_ids):
            raise ValueError(f"{role} PPO update global episode order contains duplicates")
        if tuple(cached_episode_ids) != expected_local:
            raise RuntimeError(
                f"{role} PPO cache holds {tuple(cached_episode_ids)!r}, but model-DP coordinate "
                f"{dp_rank}/{dp_world} of the global batch owns {expected_local!r}"
            )
        derived = ppo_minibatches(
            tuple(global_episode_ids),
            epochs=plan.epochs,
            num_minibatches=plan.num_minibatches,
            microbatch_size=plan.microbatch_size,
            seed=plan.seed,
            dp_rank=dp_rank,
            dp_world=dp_world,
            shard_sizes=shard_sizes,
            rank_seeded_torch_shuffle=plan.rank_seeded_torch_shuffle,
        )
        owned = [episode_id for entry in derived for microbatch in entry.microbatches for episode_id in microbatch]
        if sorted(owned) != sorted(expected_local * plan.epochs):
            raise RuntimeError(f"{role} PPO schedule does not consume its cached shard exactly once per epoch")
        return derived

    # CONVERGED, because a cache/coordinate disagreement is by nature asymmetric: it lands on the
    # one rank whose ingress and update selection diverged, and a bare raise there would abort it
    # while every peer walked into the agreement collective below and blocked forever.
    schedule = run_rank_converged(derive, f"{role} PPO update plan expansion")
    # The MINIBATCH COUNT must then agree before the first window opens. `run_ppo_optimizer_window`
    # agrees the micro-batch shape inside one window; it cannot see that a peer has a different
    # number of windows, and that rank would enter collectives everyone else had already left.
    rank_identical(len(schedule), f"{role} PPO update window count")

    metric_sums: dict[str, float] = {}
    expected_metric_names: set[str] | None = None
    counter_key = f"{role}_optimizer_step"
    expected_counter = optimizer_step()
    final_counter = expected_counter
    for index, entry in enumerate(schedule):
        result = update([list(microbatch) for microbatch in entry.microbatches])
        expected_counter += 1
        if not isinstance(result, Mapping):
            raise TypeError(f"{role} update must return an already-DP-reduced metric mapping")
        metric_names = set(result)
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError(f"{role} update returned different metric schemas across global minibatches")
        counter = result.get(counter_key)
        if isinstance(counter, bool) or not isinstance(counter, int) or counter != expected_counter:
            raise RuntimeError(
                f"{role} optimizer counter after global minibatch {index} is {counter!r}; expected {expected_counter}"
            )
        final_counter = counter
        for name, value in result.items():
            if name == counter_key:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{role} metric {name!r} must be a real scalar")
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"{role} metric {name!r} is non-finite")
            metric_sums[name] = metric_sums.get(name, 0.0) + scalar
    metrics = {name: total / len(schedule) for name, total in metric_sums.items()}
    metrics[counter_key] = float(final_counter)
    return PPORoleUpdateReceipt(
        role=role,
        batch_id=batch_id,
        optimizer_steps=len(schedule),
        optimizer_step=final_counter,
        metrics=metrics,
    )


__all__ = [
    "PPOTrainableRole",
    "model_dp_coordinate",
    "run_cached_ppo_minibatch",
    "run_ppo_batch_update",
    "run_ppo_optimizer_window",
    "validate_ppo_metrics",
    "validate_post_step_finite",
]
