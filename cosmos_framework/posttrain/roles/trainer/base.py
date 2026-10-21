# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trainer contracts and shared engine implementation (DESIGN §6).

A **trainer is an ENGINE body**, not an algorithm. Its job is fwd/bwd/step + sharding +
save/load for a model family (Diffusion / LLM / Local). The **algorithm is the `objective`
it drives**:

    train(batch) = objective(model, batch) -> (loss, metrics) -> backward -> step

So there is intentionally **no** `grpo_trainer` / `dpo_trainer` / `sft_trainer` (cosmos-rl
folds algorithm into the trainer; we don't). Swapping the algorithm = swapping the objective
(`algorithm/*.py`), not the trainer. Trainers differ only by engine; the matrix is
**N engines + M objectives**, not N×M trainer classes.

``TrainerProtocol`` is the structural control-plane contract. ``BaseTrainer`` owns the shared
finite-update and checkpoint-manager implementation used by the concrete ``DiffusionTrainer`` and
``LLMTrainer`` engines. ``LocalTrainer`` remains the structural toy implementation.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

import torch

from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint.store import CheckpointStore
    from cosmos_framework.posttrain.comm.weight_transfer import WeightPayload
    from cosmos_framework.posttrain.control.validation import ValidationReport, ValidationTrigger

# An Objective is the algorithm: (model/policy, batch) -> (loss, metrics).
Objective = Any
_ResultT = TypeVar("_ResultT")

_UpdateOutcome = Literal["accumulating", "applied", "skipped", "discarded"]
_NonFiniteReason = Literal["nonfinite_loss", "nonfinite_gradient"]
# How a rejected accumulation window is reported. "skip" discards the window and keeps training —
# the historical behaviour, and the right one when a single bad micro-batch is survivable. Some
# algorithms cannot survive it: an on-policy update whose optimizer counter is a durable contract
# with a controller has no way to express "this step did not happen", so silently continuing would
# advance training past a step the caller believes it performed.
_NonFiniteUpdatePolicy = Literal["skip", "fail_stop"]


class NonFiniteUpdateError(RuntimeError):
    """A window rejected under the ``fail_stop`` non-finite policy.

    Raised BEFORE any optimizer mutation, so unlike `PartialUpdateError` this is a clean rejection:
    no gradient has been applied and no counter has advanced. The trainer is not poisoned — the
    caller decides whether to abort or retry from a known state.
    """

    reason: _NonFiniteReason

    def __init__(self, reason: _NonFiniteReason) -> None:
        self.reason = reason
        super().__init__(f"update rejected before optimizer mutation: {reason}")


_PartialUpdateStage = Literal[
    "optimizer_step",
    "post_optimizer_step",
    "scheduler",
    "metric_reduction",
    "callback",
]


class PartialUpdateError(RuntimeError):
    """Report an update that may have mutated optimizer-owned state and is unsafe to retry.

    This typed receipt says only that mutation may have occurred. An exception raised before the
    optimizer step is intentionally not wrapped, but that does not make an interrupted trainer call
    automatically retry-safe: accumulation, attempt, or replay-cache state may already have moved.
    """

    stage: _PartialUpdateStage
    mutated: Literal[True]
    cause: BaseException
    optimizer_step: int
    publication_version: int

    def __init__(
        self,
        *,
        stage: _PartialUpdateStage,
        cause: BaseException,
        optimizer_step: int,
        publication_version: int,
    ) -> None:
        self.stage = stage
        self.mutated = True
        self.cause = cause
        self.optimizer_step = optimizer_step
        self.publication_version = publication_version
        super().__init__(
            f"trainer update failed after possible optimizer mutation at {stage}: "
            f"optimizer_step={optimizer_step}, publication_version={publication_version}; "
            f"retry is unsafe: {cause}"
        )

    def __reduce__(self) -> tuple[Any, tuple[_PartialUpdateStage, BaseException, int, int]]:
        """Preserve typed mutation metadata across Ray's pickle/cloudpickle boundary."""
        return (
            _reconstruct_partial_update_error,
            (self.stage, self.cause, self.optimizer_step, self.publication_version),
        )


def _reconstruct_partial_update_error(
    stage: _PartialUpdateStage,
    cause: BaseException,
    optimizer_step: int,
    publication_version: int,
) -> PartialUpdateError:
    error = PartialUpdateError(
        stage=stage,
        cause=cause,
        optimizer_step=optimizer_step,
        publication_version=publication_version,
    )
    error.__cause__ = cause
    return error


@dataclass(frozen=True)
class _UpdateControlState:
    """Validated optimizer-control state shared by every trainer engine."""

    optimizer_step: int
    accumulation_step: int
    attempt_id: int
    nonfinite_skips_total: int
    zero_grad_skips_total: int
    nonfinite_skips_consecutive: int
    publication_version: int | None


@dataclass(frozen=True)
class _PristineOptimizerGuard:
    """What a never-stepped optimizer looked like before a materializing state extraction."""

    pristine: bool
    group_keys: tuple[frozenset[str], ...]

    def sanitized(self, optim_state: dict[str, Any] | None) -> dict[str, Any] | None:
        """Strip everything the extraction's probe `step()` invented from an extracted state.

        The live optimizer is rolled back by the context manager, but the EXTRACTED dictionary is
        what gets written, so a pristine checkpoint would otherwise persist the probe's moments
        and any group key it created -- handing every later reader an optimizer that has stepped
        once. Group keys present before the probe are kept: they are this run's real
        hyperparameters, and restore needs them.
        """
        if not self.pristine or optim_state is None:
            return optim_state
        groups = [
            {key: value for key, value in group.items() if key == "params" or key in keys}
            for group, keys in zip(optim_state.get("param_groups", ()), self.group_keys)
        ]
        return {**optim_state, "state": {}, "param_groups": groups}


@runtime_checkable
class TrainerProtocol(Protocol):
    """Structural contract every engine trainer satisfies."""

    model: Any
    opt: Any
    objective: Objective

    def train(self, batch: list) -> dict:
        """One optimizer-cadence step: objective(model, batch) -> backward -> step; returns metrics."""
        ...

    def push_weights(self, target: Any = None) -> WeightPayload | None:
        """PRODUCER-DRIVEN weight push (design 2026-07-27 §2): the POLICY drives. ``target`` is the
        co-located rollout peer the grid routes in; the engine stages each bucket and pushes it to that
        rollout, which loads in place — so a streaming engine (``LLMTrainer``) returns **None** (nothing
        is returned for the consumer to pull). Also **None** when there is no bound backend / the two
        share a model instance (the SharedRef no-op, §8.2 #4), and for engines with no rollout edge
        (``DiffusionTrainer``/``LocalTrainer``). The ``WeightPayload`` arm of the union is the legacy
        whole-model batch return (``WeightSync.export``), retained for manual rigs."""
        ...

    def iteration(self) -> int:
        """Current training iteration — what the control loop resumes from."""
        ...

    def publication_version(self) -> int:
        """Current weight-publication version, independent when the role owns a separate clock."""
        ...

    def offload(self) -> float:
        """Move training-only GPU state to host memory; returns the MiB now held on the host."""
        ...

    def onload(self) -> float:
        """Move state released by ``offload`` back to the GPU; returns the MiB moved back."""
        ...

    def save(
        self,
        path: str,
        *,
        publish: bool = True,
        update_latest: bool = True,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
    ) -> None:
        """Write a component checkpoint into the iter dir ``path`` (shared checkpoint layout —
        unified across engines/backends; only state extraction is engine-specific). ``publish=False``
        keeps a pending-validation checkpoint local until its control state is atomically completed.
        ``update_latest=False`` is used for external-copy staging and non-resumable partial artifacts.
        ``components`` optionally narrows the manager's configured logical component plan."""
        ...

    def load(self, path: str) -> int:
        """Restore from an iter dir (or legacy single-file .pt); returns the restored iteration."""
        ...

    def finalize_checkpoints(self) -> None:
        """Drain asynchronous checkpoint work before this trainer's process group is destroyed."""
        ...

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        """Bind the exact directory generation read by setup for cross-role resume checks."""
        ...

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        """Return the exact directory generation read by setup, or None for non-directory starts."""
        ...


@runtime_checkable
class ObjectiveValidatingTrainerProtocol(TrainerProtocol, Protocol):
    """Optional capability for engines that evaluate a trainer-owned held-out Objective.

    Rollout/reward validation (GRPO and related RL shapes) stays on the existing data, rollout, and
    reward roles, so an engine can satisfy :class:`TrainerProtocol` without exposing this method.
    Diffusion DPO implements this capability; future SFT engines can opt in when they own a finite
    held-out target source.
    """

    def validate(
        self,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        """Run the trainer-owned held-out Objective for a transaction-marked model boundary."""
        ...

    def complete_ephemeral_validation(self, model_step: int) -> None:
        """Complete a transient in-memory phase after checkpoint-free initial validation."""
        ...


class BaseTrainer(ABC):
    """Shared finite-update state and checkpoint lifecycle for concrete trainer engines."""

    opt: torch.optim.Optimizer
    force_finite_grads: bool
    nonfinite_update: _NonFiniteUpdatePolicy
    require_gradients: bool
    checkpoint_manager: CheckpointManager
    grad_accum: int
    _iter: int
    _publication_version: int | None
    _accum: int
    _attempt_id: int
    _nonfinite_skips_total: int
    _zero_grad_skips_total: int
    _nonfinite_skips_consecutive: int
    _partial_update_error: PartialUpdateError | None
    _resumed_from_checkpoint: bool
    _loaded_checkpoint_provenance: dict[str, int | str] | None
    _denom_accum: float
    _skip_remaining: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        grad_accum: int = 1,
        force_finite_grads: bool = True,
        nonfinite_update: _NonFiniteUpdatePolicy = "skip",
        require_gradients: bool = False,
        checkpoint_format: str = "consolidated",
        save_async: bool = False,
        keep_optim_last: int | None = None,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        checkpoint_store: Mapping[str, Any] | CheckpointStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.opt = optimizer
        self.grad_accum = max(1, int(grad_accum))
        # TOKEN_SUM + grad_accum>1 is not refused here: `backward` (algorithm/primitives/objective.py)
        # recovers the chunk sum and normalizes once at the step via the objective's weight sum,
        # still raising NotImplementedError if absent — the guard moved.
        self._denom_accum = 0.0  # Σ of the objective's DP-mean divisors over the current window (§3.5)
        self._skip_remaining = 0  # chunks of a POISONED window still to be discarded (§5.2)
        if nonfinite_update not in ("skip", "fail_stop"):
            raise ValueError(f"nonfinite_update must be 'skip' or 'fail_stop', got {nonfinite_update!r}")
        self.nonfinite_update = nonfinite_update
        # An empty gradient inventory on any rank is a placement error, not a numerical one.
        # Engines whose every rank must contribute gradients opt in; the default keeps the
        # historical vacuous-finite result.
        self.require_gradients = bool(require_gradients)
        self.force_finite_grads = bool(force_finite_grads)
        if not self.force_finite_grads:
            from cosmos_framework.posttrain.utils.log import logger

            logger.warning(
                "force_finite_grads=False disables rank-consistent gradient rejection; "
                "non-finite loss rejection remains mandatory"
            )
        if checkpoint_manager is not None:
            legacy_checkpoint_options = (
                checkpoint_format != "consolidated"
                or save_async
                or keep_optim_last is not None
                or keep_model_last != 0
                or upload is not None
                or checkpoint_store is not None
            )
            if legacy_checkpoint_options:
                raise ValueError(
                    "checkpoint_manager is mutually exclusive with checkpoint_format, save_async, "
                    "keep_optim_last, keep_model_last, upload, and checkpoint_store"
                )
            self.checkpoint_manager = checkpoint_manager
        else:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_format=checkpoint_format,
                save_async=save_async,
                keep_optim_last=keep_optim_last,
                keep_model_last=keep_model_last,
                upload=upload,
                checkpoint_store=checkpoint_store,
            )
        self._initialize_update_state()

    def offload(self) -> float:
        """No-op default; engine trainers with GPU-resident state override it."""
        return 0.0

    def onload(self) -> float:
        """No-op default; engine trainers with GPU-resident state override it."""
        return 0.0

    @property
    def checkpoint_format(self) -> str:
        """Checkpoint format selected by the shared lifecycle manager."""
        return self.checkpoint_manager.checkpoint_format

    @checkpoint_format.setter
    def checkpoint_format(self, value: str) -> None:
        """Switch formats through the shared manager's policy boundary."""
        self.checkpoint_manager.set_checkpoint_format(value)

    @property
    def save_async(self) -> bool:
        """Whether immutable consolidated snapshots may use the background slot."""
        return self.checkpoint_manager.save_async

    @save_async.setter
    def save_async(self, value: bool) -> None:
        self.checkpoint_manager.save_async = bool(value)

    @property
    def keep_optim_last(self) -> int:
        """Number of newest local optimizer components retained."""
        return self.checkpoint_manager.keep_optim_last

    @property
    def keep_model_last(self) -> int:
        """Number of newest complete local checkpoint directories retained."""
        return self.checkpoint_manager.keep_model_last

    @property
    def upload(self) -> dict[str, Any] | None:
        """Optional object-store archive configuration."""
        return self.checkpoint_manager.upload

    def finalize_checkpoints(self) -> None:
        """Join any checkpoint writer owned by this process.

        The loop ends with the trainer offloaded; teardown work (checkpoint writers, process-group
        destruction) touches device memory, so it runs on a resident trainer.
        """
        self.onload()
        self.checkpoint_manager.finalize()

    def _save_dcp_components(
        self,
        path: str,
        dcp_components: Mapping[str, dict[str, Any] | None],
        control_components: Mapping[str, dict[str, Any] | None],
        live_control_state: Any,
        *,
        component_plan: CheckpointComponentPlan,
        publish: bool,
        update_latest: bool,
    ) -> str:
        """Commit engine-extracted DCP state through the shared manager lifecycle."""
        from cosmos_framework.posttrain.checkpoint import rank0_snapshot_is_immutable_cpu

        return self.checkpoint_manager.save_dcp(
            path,
            dcp_components,
            control_components,
            async_safe_snapshot=bool(control_components)
            and rank0_snapshot_is_immutable_cpu(
                control_components,
                live_control_state,
            ),
            component_plan=component_plan,
            publish=publish,
            update_latest=update_latest,
        )

    def _initialize_update_state(self) -> None:
        self._iter = 0
        # ``None`` preserves the historical lockstep view: publication follows the optimizer
        # clock until a role explicitly assumes ownership of the independent publication clock.
        self._publication_version = None
        self._accum = 0
        self._attempt_id = 0
        self._nonfinite_skips_total = 0
        # Attempts that reached the boundary with an exactly zero gradient and were skipped.
        # Counted apart from the non-finite skips because it is a NORMAL outcome under GRPO
        # (a tied group has zero advantage), not a numerical fault.
        self._zero_grad_skips_total = 0
        self._nonfinite_skips_consecutive = 0
        self._partial_update_error = None
        self._resumed_from_checkpoint = False
        self._loaded_checkpoint_provenance = None

    def iteration(self) -> int:
        """Current optimizer iteration."""
        return self._iter

    def publication_version(self) -> int:
        """Return the weight-publication clock, falling back to optimizer lockstep by default."""
        return self._iter if self._publication_version is None else self._publication_version

    def set_publication_version(self, version: int) -> None:
        """Advance the independent weight-publication clock without changing optimizer progress."""
        if type(version) is not int or version < 0:
            raise ValueError(f"publication version must be a non-negative exact integer, got {version!r}")
        current = self.publication_version()
        if version < current:
            raise ValueError(f"publication version cannot move backward from {current} to {version}")
        self._publication_version = version

    def weight_push_timing(self) -> dict[str, Any] | None:
        """Return this rank's last producer weight-stream timing; engines that stream no weights have none."""
        return None

    def resumed_from_checkpoint(self) -> bool:
        """Whether the trainer restored exact state, including iteration zero."""
        return self._resumed_from_checkpoint

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        """Bind the exact on-disk generation read by directory resume."""
        self._loaded_checkpoint_provenance = dict(provenance)

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        """Return the loaded directory generation, when available."""
        return None if self._loaded_checkpoint_provenance is None else dict(self._loaded_checkpoint_provenance)

    def _begin_update_attempt(self) -> None:
        if self._accum == 0:
            self.opt.zero_grad(set_to_none=True)
            self._attempt_id += 1

    def _completed_update_outcomes(self) -> int:
        """Attempts that reached a terminal outcome: an applied step or a counted skip."""
        return (
            self._iter
            + self._nonfinite_skips_total
            + (self._zero_grad_skips_total if bool(getattr(self, "skip_zero_grad", False)) else 0)
        )

    def _rewind_open_attempt(self) -> None:
        """Give back an attempt that was counted but never reached a completed outcome.

        ``_begin_update_attempt`` counts an attempt the moment a window opens, while
        ``_require_checkpoint_boundary`` requires ``attempt_id == optimizer_step + skips``. A window
        that is REJECTED (``fail_stop``) or abandoned produces neither a step nor a counted skip, so
        without this the attempt clock stays permanently ahead of every completed outcome and every
        later checkpoint save is refused. At most one attempt can be open, so the comparison against
        the completed-outcome clock is the whole rule.
        """
        if self._attempt_id > self._completed_update_outcomes():
            self._attempt_id -= 1

    def abort_update_window(self) -> None:
        """Discard a partially accumulated window so a failed update cannot contaminate the next one.

        An exception raised BETWEEN micro-batches leaves gradients accumulated, ``_accum`` non-zero,
        and the attempt open. The next window would then skip ``zero_grad``, close early on a mixture
        of the abandoned gradients and the new ones, and step the optimizer on data the caller never
        asked it to. Callers that own a multi-call accumulation window (see
        `roles/trainer/ppo_runtime.py::run_ppo_optimizer_window`) invoke this on any escape from that window.

        A POISONED trainer is left untouched: it is already fail-closed until a checkpoint restore
        resets every clock, and rewinding the attempt of a window that did mutate the optimizer would
        misreport what happened.

        Only the ACCOUNTING is reset; the stale gradients themselves are left where a completed step
        also leaves them, for ``_begin_update_attempt`` to clear when the next window opens. That
        keeps this method free of any call that can raise, because it runs from an exception handler
        and a failure here would replace -- and hide -- the failure that caused the abort.
        """
        if self._partial_update_error is not None:
            return
        self._accum = 0
        self._denom_accum = 0.0
        self._skip_remaining = 0
        self._rewind_open_attempt()

    def _restore_optimizer_step(self, optimizer_step: int) -> None:
        """Rewind every update clock to a saved, completed optimizer boundary.

        For engines whose durable checkpoint carries only the optimizer counter (the PPO roles),
        this is the counterpart of ``_apply_update_state``: a saved boundary is by definition a
        completed, quiescent one, so the accumulation window, the skip counters, and any
        ``PartialUpdateError`` poison are cleared with it. Clearing the poison is what makes the
        recovery path this class documents -- "restore a known checkpoint first" -- actually exist.
        """
        if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int) or optimizer_step < 0:
            raise ValueError(f"restored optimizer step must be a non-negative integer, got {optimizer_step!r}")
        self._iter = optimizer_step
        # Every saved boundary is a completed attempt, so the attempt clock follows the step clock.
        self._attempt_id = optimizer_step
        self._accum = 0
        self._denom_accum = 0.0
        self._skip_remaining = 0
        self._nonfinite_skips_total = 0
        self._nonfinite_skips_consecutive = 0
        self._zero_grad_skips_total = 0
        self._partial_update_error = None

    def _raise_if_partial_update_poisoned(self) -> None:
        """Fail closed until an explicit checkpoint restore replaces uncertain mutated state."""
        if self._partial_update_error is not None:
            raise self._partial_update_error from self._partial_update_error.cause

    def _record_partial_update_error(
        self,
        *,
        stage: _PartialUpdateStage,
        cause: BaseException,
    ) -> PartialUpdateError:
        """Poison this trainer and return its durable, non-retryable mutation receipt."""
        error = PartialUpdateError(
            stage=stage,
            cause=cause,
            optimizer_step=self._iter,
            publication_version=self.publication_version(),
        )
        self._partial_update_error = error
        return error

    def _run_partial_update_stage(
        self,
        *,
        stage: _PartialUpdateStage,
        action: Callable[[], _ResultT],
    ) -> _ResultT:
        """Converge a post-step phase, then poison every rank with one typed receipt on failure."""
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        try:
            return run_rank_converged(action, f"trainer partial update {stage}")
        except BaseException as cause:  # noqa: BLE001 -- optimizer mutation is already possible
            error = self._record_partial_update_error(stage=stage, cause=cause)
            raise error from cause

    def _commit_update(self) -> None:
        self._accum = 0
        self._iter += 1
        self._nonfinite_skips_consecutive = 0

    def _outcome_metrics(self, outcome: _UpdateOutcome) -> dict[str, Any]:
        # The consecutive counter belongs to the reusable trainer contract: direct or
        # future bounded-continuation callers can observe values >1. The shipped DPO
        # entry is fail-fast, so its external stream sees only 0 or 1.
        metrics: dict[str, Any] = {
            "optim/update_outcome": outcome,
            "optim/step_skipped": float(outcome == "skipped"),
            "optim/attempt_id": self._attempt_id,
            "optim/optimizer_step": self._iter,
            "optim/nonfinite_skips_total": self._nonfinite_skips_total,
            "optim/nonfinite_skips_consecutive": self._nonfinite_skips_consecutive,
            # Dense binary event indicators intentionally emit zero on healthy steps;
            # this supports direct skip-rate aggregation without sparse/missing values.
            "optim/nonfinite_loss": 0.0,
            "optim/nonfinite_gradient": 0.0,
            "optim/finite_gradient_check_enabled": float(self.force_finite_grads),
        }
        if bool(getattr(self, "skip_zero_grad", False)):
            metrics["optim/zero_grad_skips_total"] = self._zero_grad_skips_total
        return metrics

    def _skip_nonfinite(self, metrics: dict[str, Any], reason: _NonFiniteReason) -> dict[str, Any]:
        """Reject the current accumulation window without optimizer-side mutation.

        Under the default ``skip`` policy the window is discarded and training continues. Under
        ``fail_stop`` the same rejection is raised instead: the gradients are still dropped, but no
        skip counter advances, because a skip that the caller never learns about is exactly the
        silent divergence `fail_stop` exists to prevent. That rejection therefore has to give the
        consumed attempt back as well -- an attempt with NO completed outcome would leave
        ``_require_checkpoint_boundary`` permanently unsatisfiable, so the clean rejection this
        error documents would still cost the caller its ability to checkpoint.
        """
        self.opt.zero_grad(set_to_none=True)
        if self.nonfinite_update == "fail_stop":
            self._accum = 0
            self._denom_accum = 0.0
            self._rewind_open_attempt()
            raise NonFiniteUpdateError(reason)
        # Abandon the REST of this window: `train` returns before `_accum += 1` on a rejection, so at
        # grad_accum>1 the remaining chunks would otherwise accumulate into the fresh window (0, i.e.
        # today's behaviour, at grad_accum==1). ONLY `LLMTrainer.train` acts on this — under its
        # TOKEN_SUM a window is one mini-batch sharing one divisor, so a mixed window would step under
        # a divisor matching neither half. `DiffusionTrainer`/DPO use MEAN, where each micro-batch is
        # an independent `loss/grad_accum` draw, so a spanning window still averages correctly and the
        # field is inert. Revisit if a TOKEN_SUM diffusion objective appears.
        self._skip_remaining = max(0, self.grad_accum - self._accum - 1)
        self._accum = 0
        self._denom_accum = 0.0
        self._nonfinite_skips_total += 1
        self._nonfinite_skips_consecutive += 1
        # Objective metrics can themselves be NaN/Inf on a loss rejection. Reducing
        # them is intentional diagnostic propagation after the finite-status consensus:
        # the outcome/reason fields added below are authoritative, and the objective
        # values must not be sanitized into apparently healthy telemetry.
        metrics = self._reduce_metrics(metrics)
        metrics.update(self._outcome_metrics("skipped"))
        metrics["optim/skip_reason"] = reason
        metrics[f"optim/{reason}"] = 1.0
        return metrics

    @abstractmethod
    def _reduce_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Reduce engine-specific metrics before common outcome fields are added."""
        raise NotImplementedError

    def optimizer_is_pristine(self) -> bool:
        """Whether the optimizer has never stepped, by EVERY place a step leaves a trace.

        A stateful optimizer creates its moments and its own step counter on the first `step()`.
        Until then it is pristine, and its first real update must be the FIRST one it has ever
        seen -- AdamW's bias correction is a function of that counter. Checkpointing must not be
        able to change the answer, so the condition is asked before any state extraction and
        recorded in the checkpoint (`optimizer_pristine`) for restore to honor.

        `optimizer.state` alone does not answer it. `utils/fused_adam.py::FusedAdam`, which
        `config_ppo.py::_validate_optimizer` accepts, keeps ONE shared counter per param group
        and builds `param_groups_master` lazily inside `step()`. Reading only the per-parameter
        dictionary would call a stepped FusedAdam pristine.
        """
        if self.opt.state:
            return False
        if any("step" in group for group in self.opt.param_groups):
            return False
        return getattr(self.opt, "param_groups_master", None) is None

    @contextlib.contextmanager
    def _preserved_pristine_optimizer(self) -> Iterator[_PristineOptimizerGuard]:
        """Undo the optimizer initialization that state extraction performs as a side effect.

        `torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` materializes an
        untouched optimizer by running a zero-gradient `step()` at zero LR. Parameters do not
        move, so nothing in the model or in the trainer's own clocks records it -- but the
        optimizer's internal step advances, and the next real update is then bias-corrected as
        though it were the second. Saving a checkpoint may not change training numerics, so an
        optimizer that was pristine on entry is returned to pristine on exit.

        The rollback is defined over everything a `step()` can create -- per-parameter state,
        group-owned keys, and the lazily built master-weight copies -- because an implementation
        that keeps its counter on the group would otherwise survive the rollback, be seen as
        pristine by the NEXT save, and be advanced again.
        """
        guard = _PristineOptimizerGuard(
            pristine=self.optimizer_is_pristine(),
            group_keys=tuple(frozenset(group) for group in self.opt.param_groups),
        )
        master = getattr(self.opt, "param_groups_master", None)
        try:
            yield guard
        finally:
            if guard.pristine:
                self.opt.state.clear()
                for group, keys in zip(self.opt.param_groups, guard.group_keys):
                    for created in set(group) - keys:
                        del group[created]
                if hasattr(self.opt, "param_groups_master"):
                    self.opt.param_groups_master = master

    def reset_optimizer_to_pristine(self) -> None:
        """Return the optimizer to never-stepped, wherever it records having stepped.

        This is NOT the save-time rollback. There the optimizer was pristine on entry and only
        the extraction's side effect is undone; here the CHECKPOINT asserts pristine while the
        live optimizer may hold anything -- resume can load into a trainer that has already
        trained. It runs BEFORE a load target is shaped, so the target matches the group schema a
        pristine save wrote: DCP's planner is strict, and a target field the checkpoint omits
        fails the load before any later cleanup could run.
        """
        self.opt.state.clear()
        for group in self.opt.param_groups:
            # Group-owned counters belong to an optimizer that has stepped; a pristine one has
            # none (`utils/fused_adam.py::FusedAdam`).
            group.pop("step", None)
        if hasattr(self.opt, "param_groups_master"):
            self.opt.param_groups_master = None

    def _apply_saved_param_groups(self, optim_state: Mapping[str, Any] | None) -> None:
        """Apply saved group OPTIONS without creating moments or advancing the optimizer.

        An empty per-parameter state does not mean a pristine checkpoint has nothing to restore:
        `param_groups` still carries the learning rate, betas and weight decay. The scheduler's
        own `load_state_dict` restores its bookkeeping but does NOT write the learning rate back
        into the optimizer, so skipping this leaves the live optimizer on whatever it held.

        `params` is deliberately not copied: the saved entry identifies parameters by checkpoint
        key, while the live group holds this process's tensors.
        """
        saved = (optim_state or {}).get("param_groups") or ()
        if len(saved) != len(self.opt.param_groups):
            raise RuntimeError(
                f"checkpoint has {len(saved)} optimizer param groups but this run has "
                f"{len(self.opt.param_groups)}; resume requires the same optimizer grouping"
            )
        for group, values in zip(self.opt.param_groups, saved):
            for key, value in values.items():
                if key != "params":
                    group[key] = value

    def _update_state_dict(self) -> dict[str, int]:
        state = {
            # `iter` is the legacy checkpoint key; `optimizer_step` is the explicit
            # durable clock. They are intentionally identical and validated on restore.
            "iter": self._iter,
            "optimizer_step": self._iter,
            "accum": self._accum,
            # The batching shape this checkpoint was written under, so the driver can prove
            # its optimizer-step -> batch-step conversion is still valid on resume.
            "n_mini": int(getattr(self, "n_mini", 1)),
            "attempt_id": self._attempt_id,
            "nonfinite_skips_total": self._nonfinite_skips_total,
            "nonfinite_skips_consecutive": self._nonfinite_skips_consecutive,
        }
        if self.optimizer_is_pristine():
            # Restore cannot re-derive this: the optimizer state WRITTEN alongside it was
            # materialized by the extraction itself, so on disk a never-stepped optimizer is
            # indistinguishable from one that has stepped once. Absent means "not recorded",
            # which is exactly how every checkpoint written before this marker behaves.
            state["optimizer_pristine"] = 1
        if bool(getattr(self, "skip_zero_grad", False)):
            # The marker makes a behavior-changing checkpoint impossible to resume silently with
            # the legacy update rule. It is absent when disabled, preserving the old schema.
            state["skip_zero_grad"] = 1
            state["zero_grad_skips_total"] = self._zero_grad_skips_total
        if self._publication_version is not None:
            # Keep the legacy schema byte-for-byte until a role explicitly decouples publication.
            state["publication_version"] = self._publication_version
        return state

    def _require_checkpoint_boundary(self) -> None:
        """Converge invalid save boundaries before any rank enters checkpoint IO."""
        from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

        boundary_error: Exception | None = None
        if self._partial_update_error is not None:
            boundary_error = RuntimeError(
                "checkpoint save is unsafe after a partial optimizer update; restore a known checkpoint first"
            )
        elif self._accum != 0:
            boundary_error = RuntimeError(
                f"checkpoint save requires a completed update boundary, got accum={self._accum}"
            )
        elif self._attempt_id % max(1, int(getattr(self, "n_mini", 1))):
            # ATTEMPTS define the collected-batch boundary for every trainer: a non-finite or
            # zero-gradient skip still consumed its chunk. Optimizer commits cannot identify that
            # boundary after any rejected update and could allow a save in the middle of the next
            # batch instead.
            boundary_error = RuntimeError(
                "checkpoint save requires a completed collected-batch boundary: "
                f"attempts={self._attempt_id}, n_mini={getattr(self, 'n_mini', 1)}"
            )
        elif self._attempt_id != self._completed_update_outcomes():
            zero_grad_detail = (
                f", zero_grad_skips_total={self._zero_grad_skips_total}"
                if bool(getattr(self, "skip_zero_grad", False))
                else ""
            )
            boundary_error = RuntimeError(
                "checkpoint save requires every optimizer attempt to have a completed outcome: "
                f"attempt_id={self._attempt_id}, optimizer_step={self._iter}, "
                f"nonfinite_skips_total={self._nonfinite_skips_total}{zero_grad_detail}"
            )
        raise_if_any_rank_failed(boundary_error, "trainer checkpoint boundary")

    @staticmethod
    def _parse_update_state(trainer_state: dict[str, Any]) -> _UpdateControlState:
        """Validate checkpoint clocks without mutating a trainer."""
        if "iter" not in trainer_state and "optimizer_step" not in trainer_state:
            raise RuntimeError("training resume requires trainer iteration state")
        optimizer_step = trainer_state.get("optimizer_step", trainer_state.get("iter"))
        iteration = trainer_state.get("iter", optimizer_step)
        attempt_id = trainer_state.get("attempt_id", optimizer_step)
        accum = trainer_state.get("accum", 0)
        skips_total = trainer_state.get("nonfinite_skips_total", 0)
        zero_grad_skips_total = trainer_state.get("zero_grad_skips_total", 0)
        skips_consecutive = trainer_state.get("nonfinite_skips_consecutive", 0)
        publication_version = trainer_state.get("publication_version")
        control_values = {
            "optimizer_step": optimizer_step,
            "iter": iteration,
            "attempt_id": attempt_id,
            "accum": accum,
            "nonfinite_skips_total": skips_total,
            "zero_grad_skips_total": zero_grad_skips_total,
            "nonfinite_skips_consecutive": skips_consecutive,
        }
        for name, value in control_values.items():
            if type(value) is not int or value < 0:
                raise RuntimeError(f"trainer checkpoint {name} must be a non-negative integer, got {value!r}")
        if publication_version is not None and (type(publication_version) is not int or publication_version < 0):
            raise RuntimeError(
                "trainer checkpoint publication_version must be a non-negative integer when present, "
                f"got {publication_version!r}"
            )
        if iteration != optimizer_step:
            raise RuntimeError(
                f"trainer iteration {iteration} differs from optimizer_step {optimizer_step} in checkpoint"
            )
        if accum != 0:
            raise RuntimeError(f"checkpoint resume requires a quiescent accumulation boundary, got accum={accum}")
        if "zero_grad_skips_total" in trainer_state:
            if attempt_id != optimizer_step + skips_total + zero_grad_skips_total:
                raise RuntimeError(
                    "trainer attempt clock is inconsistent with completed outcomes: "
                    f"attempt_id={attempt_id}, optimizer_step={optimizer_step}, "
                    f"nonfinite_skips_total={skips_total}, zero_grad_skips_total={zero_grad_skips_total}"
                )
        elif attempt_id != optimizer_step + skips_total:
            # Preserve the legacy validation and its diagnostic exactly when the feature field is
            # absent. Older checkpoints must not acquire a new failure surface while loading.
            raise RuntimeError(
                "trainer attempt clock is inconsistent with completed outcomes: "
                f"attempt_id={attempt_id}, optimizer_step={optimizer_step}, "
                f"nonfinite_skips_total={skips_total}"
            )
        if skips_consecutive > skips_total:
            raise RuntimeError(
                f"trainer non-finite skip counters are invalid: total={skips_total}, consecutive={skips_consecutive}"
            )
        return _UpdateControlState(
            optimizer_step=optimizer_step,
            accumulation_step=accum,
            attempt_id=attempt_id,
            nonfinite_skips_total=skips_total,
            zero_grad_skips_total=zero_grad_skips_total,
            nonfinite_skips_consecutive=skips_consecutive,
            publication_version=publication_version,
        )

    def _apply_update_state(
        self,
        state: _UpdateControlState,
        *,
        rng_state: Any,
        cuda_device: Any = None,
    ) -> int:
        """Apply validated clocks and their rank-local RNG state."""
        from cosmos_framework.posttrain.checkpoint import restore_rng_state

        restore_rng_state(rng_state, cuda_device=cuda_device)  # no-op for legacy / world-size change
        self._iter = state.optimizer_step
        self._accum = state.accumulation_step
        self._attempt_id = state.attempt_id
        self._nonfinite_skips_total = state.nonfinite_skips_total
        self._zero_grad_skips_total = state.zero_grad_skips_total
        self._nonfinite_skips_consecutive = state.nonfinite_skips_consecutive
        self._publication_version = state.publication_version
        self._partial_update_error = None
        return self._iter
