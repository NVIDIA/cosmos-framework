# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Objective-driven trainer for flow-matching policies."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.algorithm.primitives.objective import backward as reduce_backward
from cosmos_framework.posttrain.checkpoint import CheckpointManager, rank0_snapshot_is_immutable_cpu
from cosmos_framework.posttrain.checkpoint.reference import ReferenceIdentity
from cosmos_framework.posttrain.control.validation import ValidationProgress, ValidationReport, ValidationTrigger
from cosmos_framework.posttrain.model.protocol import FlowMatchingPolicy
from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv, PreferenceSourceIdentity
from cosmos_framework.posttrain.roles.trainer.base import BaseTrainer
from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_gradients_finite, all_ranks_tensor_finite
from cosmos_framework.posttrain.roles.trainer.policy_state import (
    AppliedUpdateBoundary,
    DiffusionPolicyStateProgram,
    PolicyStateDescriptor,
    PolicyStateName,
)
from cosmos_framework.posttrain.roles.trainer.validation.preference import (
    ObjectiveValidationRunner,
    build_objective_validation_identity,
    move_to_device,
)
from cosmos_framework.posttrain.roles.trainer.validation.publication import PublicationJournal
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.utils.dist import (
    collective_device,
    is_dtensor_sharded,
    rank_identical,
    run_rank_converged,
)
from cosmos_framework.posttrain.utils.identity import (
    canonical_json_sha256,
    json_identity_equal,
    normalize_json_dict,
)
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.residency import HostOffload
from cosmos_framework.posttrain.utils.scheduler import scheduler_resume_identity

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint.store import CheckpointStore

_REF_REFRESH_TIMING = "after_applied_update_v1"
_OLD_REFRESH_TIMING = "after_applied_update_v1"
_OLD_POLICY_CHECKPOINT_VERSION = 1
_OLD_POLICY_DERIVED_MODE = "derived_from_policy"
_OLD_POLICY_MATERIALIZED_MODE = "materialized"


class TargetOnlyInventoryValidator(Protocol):
    """Model-owned validation for target-only copy-mode tensor schemas."""

    def __call__(
        self,
        *,
        source_keys: frozenset[str],
        target_only_keys: frozenset[str],
    ) -> None:
        """Accept the target-only schema or raise ``ValueError`` with the reason."""
        ...


@dataclass(frozen=True)
class CopyInventoryPolicy:
    """Explicitly permitted model-schema differences for model-only initialization."""

    source_only_keys: tuple[str, ...] = ()
    target_only_validator: TargetOnlyInventoryValidator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_only_keys, tuple) or any(
            not isinstance(key, str) or not key for key in self.source_only_keys
        ):
            raise TypeError("copy inventory source_only_keys must be a tuple of non-empty strings")
        normalized = tuple(key.removeprefix("net.") for key in self.source_only_keys)
        if len(normalized) != len(set(normalized)):
            raise ValueError("copy inventory source_only_keys must not contain duplicates")
        object.__setattr__(self, "source_only_keys", normalized)
        if self.target_only_validator is not None and not callable(self.target_only_validator):
            raise TypeError("copy inventory target_only_validator must be callable")


@dataclass(frozen=True)
class _CopyPlan:
    role: str
    load_keys: tuple[str, ...]


def _missing_resume_error(
    descriptor: PolicyStateDescriptor,
    path: str,
    *,
    copy_mode: bool = False,
    dcp: bool = True,
) -> FileNotFoundError | RuntimeError:
    """Build the stable load error for one typed policy-state role."""

    if descriptor.semantic_role == "policy":
        if copy_mode and dcp:
            message = f"no model component in DCP checkpoint {path!r}"
        elif dcp:
            message = f"training resume requires policy net state in DCP checkpoint {path!r}"
        else:
            message = f"checkpoint {path!r} has no policy net state"
        return FileNotFoundError(message)
    location = "DCP checkpoint" if dcp else "checkpoint"
    messages = {
        "old": f"moving-old-policy resume requires an old component in {location} {path!r}",
        "reference": f"moving-reference resume requires a ref component in {location} {path!r}",
        "ema": f"EMA-enabled resume requires net_ema state in {location} {path!r}",
    }
    return RuntimeError(messages[descriptor.semantic_role])


def _tensor_shapes(state: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Project a state dictionary into its mutation-free tensor-shape inventory."""
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"checkpoint state {key!r} is not tensor-shaped")
        shapes[key] = tuple(int(size) for size in shape)
    return shapes


def _build_copy_plan(
    source_shapes: dict[str, tuple[int, ...]],
    target_shapes: dict[str, tuple[int, ...]],
    policy: CopyInventoryPolicy,
    *,
    role: str,
) -> _CopyPlan:
    """Validate one target against the source before any target tensor is mutated."""
    source_keys = set(source_shapes)
    target_keys = set(target_shapes)
    configured_source_only = set(policy.source_only_keys)
    missing_configured = sorted(configured_source_only - source_keys)
    if missing_configured:
        raise RuntimeError(
            f"copy inventory for {role} declares source-only keys absent from the source: {missing_configured}"
        )
    configured_in_target = sorted(configured_source_only & target_keys)
    if configured_in_target:
        raise RuntimeError(
            f"copy inventory for {role} declares keys source-only but they exist in the target: {configured_in_target}"
        )

    actual_source_only = source_keys - target_keys
    unexpected_source_only = sorted(actual_source_only - configured_source_only)
    if unexpected_source_only:
        raise RuntimeError(f"copy inventory for {role} has unapproved source-only keys: {unexpected_source_only}")

    common_keys = source_keys & target_keys
    shape_drift = [
        (key, source_shapes[key], target_shapes[key])
        for key in sorted(common_keys)
        if source_shapes[key] != target_shapes[key]
    ]
    if shape_drift:
        raise RuntimeError(f"copy inventory for {role} has tensor shape drift: {shape_drift}")

    target_only = target_keys - source_keys
    if target_only:
        validator = policy.target_only_validator
        if validator is None:
            raise RuntimeError(f"copy inventory for {role} has unapproved target-only keys: {sorted(target_only)}")
        try:
            validator(
                source_keys=frozenset(source_keys),
                target_only_keys=frozenset(target_only),
            )
        except ValueError as error:
            raise RuntimeError(f"copy inventory for {role} {error}") from error

    return _CopyPlan(role=role, load_keys=tuple(sorted(common_keys)))


def _require_identical_copy_plans(plans: list[_CopyPlan]) -> None:
    """Prove that every rank will issue the same copy-mode checkpoint reads."""
    identity = [{"role": plan.role, "load_keys": list(plan.load_keys)} for plan in plans]
    digest = canonical_json_sha256(identity, context="copy-mode inventory plan")
    rank_identical(digest, "copy-mode inventory plan digest")


class DiffusionTrainer(BaseTrainer):
    """Model-agnostic trainer body: `train(batch)` = objective → backward → step.

    A policy that opts into EMA updates owns its complete persisted EMA state:
    ``update_ema(iteration)`` must update parameters and synchronize any persistent
    buffers that the model's EMA worker does not maintain itself.
    """

    reference_identity: ReferenceIdentity
    validation_source: DatasetPreferenceEnv | None
    validation_state: ValidationState
    validation_publication: PublicationJournal
    validation_runner: ObjectiveValidationRunner | None
    _policy_state: DiffusionPolicyStateProgram

    def __init__(
        self,
        model: FlowMatchingPolicy,
        objective: Objective,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        update_ema_after_iter: bool = False,
        grad_accum: int = 1,
        update_ref_every_iter: int = 0,
        update_old_every_iter: int = 0,
        device: torch.device | str | None = None,
        callbacks: Any = None,
        grad_clip_norm: float = 0.0,
        force_finite_grads: bool = True,
        checkpoint_format: str = "consolidated",
        save_async: bool = False,
        keep_optim_last: int | None = None,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        checkpoint_store: Mapping[str, Any] | CheckpointStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        validation_source: DatasetPreferenceEnv | None = None,
        validation_batch_size: int = 1,
        validation_max_samples: int | None = None,
        validation_seed: int = 0,
        validation_before_training: bool = False,
        validation_every_steps: int = 0,
        validation_train_total_steps: int = 0,
        defer_frozen_reference_identity: bool = False,
    ) -> None:
        from cosmos_framework.posttrain.callbacks.base import CallbackList

        self.model = model
        self.objective = objective
        # The rollout samples in this same process while the trainer is offloaded, so a whole-process
        # pause is not an option here; only update-side state moves.
        self._host_offload = HostOffload()
        super().__init__(
            optimizer,
            grad_accum=grad_accum,
            force_finite_grads=force_finite_grads,
            checkpoint_format=checkpoint_format,
            save_async=save_async,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            checkpoint_store=checkpoint_store,
            checkpoint_manager=checkpoint_manager,
        )
        self.sched = scheduler
        self.update_ema_after_iter = bool(update_ema_after_iter)
        self.update_old_every_iter = int(update_old_every_iter)
        self.update_ref_every_iter = int(update_ref_every_iter)
        self._policy_state = DiffusionPolicyStateProgram(
            self.model,
            update_ema_after_iter=self.update_ema_after_iter,
            update_old_every_iter=self.update_old_every_iter,
            update_ref_every_iter=self.update_ref_every_iter,
        )
        self._policy_state.validate_ema()
        if self._policy_state.ema_net is not None and not self.update_ema_after_iter:
            if os.environ.get("RANK", "0") == "0":
                logger.warning(
                    "policy exposes net_ema but EMA updates are disabled; the inactive EMA is not "
                    "initialized from rl.init_from, checkpointed, or restored"
                )
        # Diffusion objectives do not implement deferred token-sum normalization.
        if self.objective.reduction is Reduction.TOKEN_SUM and self.grad_accum > 1:
            raise ValueError("token_sum reduction requires grad_accum=1 (one full batch per step).")
        self._policy_state.validate_moving_roles()
        self._policy_state.initialize()
        self.grad_clip_norm = float(grad_clip_norm)
        self._device = torch.device(device) if device is not None else None
        self.callbacks = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        self.validation_source = validation_source
        self.validation_batch_size = int(validation_batch_size)
        self.validation_max_samples = None if validation_max_samples is None else int(validation_max_samples)
        self.validation_seed = int(validation_seed)
        self.validation_before_training = bool(validation_before_training)
        self.validation_every_steps = int(validation_every_steps)
        self.validation_train_total_steps = int(validation_train_total_steps)
        if self.validation_batch_size < 1:
            raise ValueError(f"validation_batch_size must be >= 1, got {self.validation_batch_size}")
        if self.validation_max_samples is not None and self.validation_max_samples < 1:
            raise ValueError(f"validation_max_samples must be >= 1 when set, got {self.validation_max_samples}")
        if self.validation_every_steps < 0 or self.validation_train_total_steps < 0:
            raise ValueError("validation schedule steps must be non-negative")
        self.reference_identity = ReferenceIdentity(
            self.model,
            refresh_every_iter=self.update_ref_every_iter,
            defer_frozen=defer_frozen_reference_identity,
        )
        self.validation_state = ValidationState(self._current_validation_identity())
        self.validation_publication = PublicationJournal(self.validation_state)
        self.validation_runner = (
            ObjectiveValidationRunner(
                model=self.model,
                objective=self.objective,
                source=self.validation_source,
                state=self.validation_state,
                batch_size=self.validation_batch_size,
                max_samples=self.validation_max_samples,
                seed=self.validation_seed,
                device=lambda: self.device,
            )
            if self.validation_source is not None
            else None
        )
        self.callbacks.on_init(self)

    def _checkpoint_nets(self) -> dict[PolicyStateName, torch.nn.Module]:
        """Adapt the typed policy-state inventory to state-dict APIs."""
        return self._policy_state.networks()

    def _validate_ema_resume_contract(self, trainer_state: dict[str, Any]) -> None:
        """Reject exact resume when saved and configured EMA-update semantics differ."""
        if "ema_updates_enabled" not in trainer_state:
            if self.update_ema_after_iter:
                raise RuntimeError(
                    "checkpoint predates the EMA-update contract; exact resume with EMA enabled is ambiguous "
                    "(use rl.init_from to start a new tuning run)"
                )
            return
        saved = trainer_state["ema_updates_enabled"]
        if type(saved) is not bool:
            raise RuntimeError("checkpoint EMA-update contract must be a boolean")
        if saved != self.update_ema_after_iter:
            raise RuntimeError(
                "checkpoint EMA-update contract differs from the current trainer: "
                f"saved={saved}, configured={self.update_ema_after_iter}"
            )

    def _validate_scheduler_resume_contract(self, trainer_state: dict[str, Any]) -> None:
        """Reject exact resume before mutable scheduler state can cross semantic versions."""
        configured = scheduler_resume_identity(self.sched)
        has_saved_identity = "scheduler_identity" in trainer_state
        if configured is None:
            if has_saved_identity:
                raise RuntimeError(
                    "checkpoint requires an identity-bearing scheduler, but the current trainer has no "
                    "compatible scheduler identity"
                )
            return
        if not has_saved_identity:
            raise RuntimeError(
                "checkpoint predates the WFM scheduler exact-resume identity; continuation under the "
                "configured schedule is ambiguous (use rl.init_from to start a new tuning run)"
            )
        if trainer_state.get("sched") is None:
            raise RuntimeError("checkpoint has a WFM scheduler identity but no scheduler state")
        try:
            restored = normalize_json_dict(
                trainer_state["scheduler_identity"],
                context="checkpoint scheduler exact-resume identity",
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("checkpoint scheduler exact-resume identity is malformed") from error
        if not json_identity_equal(restored, configured, context="scheduler exact-resume identity"):
            raise RuntimeError(
                "checkpoint scheduler exact-resume identity differs from the configured schedule: "
                f"saved={restored!r}, configured={configured!r}"
            )

    def _old_policy_checkpoint_record(self) -> dict[str, int | str] | None:
        """Describe whether the moving-old tensors are stored or derivable at this boundary."""
        if self.update_old_every_iter <= 0:
            return None
        mode = (
            _OLD_POLICY_DERIVED_MODE if self._iter % self.update_old_every_iter == 0 else _OLD_POLICY_MATERIALIZED_MODE
        )
        return {
            "version": _OLD_POLICY_CHECKPOINT_VERSION,
            "mode": mode,
            "optimizer_step": self._iter,
        }

    def _old_policy_is_derived(self, trainer_state: dict[str, Any]) -> bool:
        """Validate the versioned moving-old storage marker and return its derived mode."""
        record = trainer_state.get("old_policy_checkpoint")
        if record is None:
            return False
        if self.update_old_every_iter <= 0:
            raise RuntimeError("checkpoint declares moving-old state while the trainer has no moving-old cadence")
        if not isinstance(record, dict) or set(record) != {"version", "mode", "optimizer_step"}:
            raise RuntimeError("checkpoint moving-old storage marker is malformed")
        version = record["version"]
        mode = record["mode"]
        optimizer_step = record["optimizer_step"]
        if type(version) is not int or version != _OLD_POLICY_CHECKPOINT_VERSION:
            raise RuntimeError(f"unsupported moving-old storage marker version {version!r}")
        if not isinstance(mode, str) or mode not in (_OLD_POLICY_DERIVED_MODE, _OLD_POLICY_MATERIALIZED_MODE):
            raise RuntimeError(f"unsupported moving-old storage mode {mode!r}")
        saved_step = trainer_state.get("optimizer_step", trainer_state.get("iter"))
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or optimizer_step != saved_step
        ):
            raise RuntimeError(
                "checkpoint moving-old storage marker step differs from trainer state: "
                f"marker={optimizer_step!r}, trainer={saved_step!r}"
            )
        if mode == _OLD_POLICY_DERIVED_MODE and optimizer_step % self.update_old_every_iter:
            raise RuntimeError(
                "checkpoint claims policy-derived moving-old state away from its refresh boundary: "
                f"step={optimizer_step}, cadence={self.update_old_every_iter}"
            )
        return mode == _OLD_POLICY_DERIVED_MODE

    def _validate_old_resume_contract(self, trainer_state: dict[str, Any]) -> bool:
        """Reject an ambiguous moving-old continuation before network mutation."""
        if "old_refresh_every_iter" not in trainer_state:
            if self.update_old_every_iter > 0:
                raise RuntimeError("moving-old-policy resume requires an explicit refresh cadence")
            return False
        saved_cadence = trainer_state["old_refresh_every_iter"]
        if isinstance(saved_cadence, bool) or not isinstance(saved_cadence, int) or saved_cadence < 0:
            raise RuntimeError(f"old-policy refresh cadence must be a non-negative integer, got {saved_cadence!r}")
        if saved_cadence != self.update_old_every_iter:
            raise RuntimeError(
                f"old-policy refresh cadence {saved_cadence} differs from configured cadence "
                f"{self.update_old_every_iter}"
            )
        timing = trainer_state.get("old_refresh_timing")
        if self.update_old_every_iter > 0 and timing != _OLD_REFRESH_TIMING:
            raise RuntimeError(f"unsupported old-policy refresh timing {timing!r}")
        return self._old_policy_is_derived(trainer_state)

    @property
    def device(self) -> torch.device:
        """Return the explicit batch device or infer it from the policy."""
        if self._device is not None:
            return self._device
        try:
            return next(self.model.net.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def train(self, batch: list[Trajectory]) -> dict:
        self.model.net.train()  # net_ref is kept in eval by the model adapter
        self._begin_update_attempt()

        prepared_batch = run_rank_converged(
            lambda: move_to_device(batch, self.device),  # host→device data-fetch (§1.4)
            "trainer batch device transfer",
        )
        loss, metrics = self.objective(self.model, prepared_batch)
        # Consensus precedes FSDP backward so a bad rank cannot strand its peers.
        if not all_ranks_tensor_finite(loss):
            return self._skip_nonfinite(metrics, "nonfinite_loss")
        # The objective reduction owns gradient-accumulation normalization.
        reduce_backward(loss, self.objective.reduction, self.grad_accum)
        self._accum += 1

        if self._accum == self.grad_accum:
            # Reject the whole window rather than mutate the objective or optimizer moments.
            params = [p for p in self.model.net.parameters() if p.grad is not None]
            if self.force_finite_grads and not all_ranks_gradients_finite(params):
                return self._skip_nonfinite(metrics, "nonfinite_gradient")
            if self.grad_clip_norm > 0:
                # clip_grad_norm_ computes the cross-shard DTensor norm.
                total_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)  # []
                if not all_ranks_tensor_finite(total_norm):
                    return self._skip_nonfinite(metrics, "nonfinite_gradient")
                to_local = getattr(total_norm, "to_local", None)
                local_norm = to_local() if callable(to_local) else total_norm  # [] local scalar
                metrics["train/grad_norm"] = local_norm.detach()  # []
            update_learning_rate = float(self.opt.param_groups[0]["lr"])
            update_boundary = AppliedUpdateBoundary(self._iter, self._iter + 1)
            self.opt.step()
            if self.sched is not None:
                self.sched.step()
            # Match OmniMoT.on_before_zero_grad: EMA consumes the applied update's zero-based
            # pre-commit index. The first EMA update therefore receives iteration 0.
            self._policy_state.before_commit(update_boundary)
            self._commit_update()
            if self._iter != update_boundary.committed_step:
                raise RuntimeError(
                    "diffusion optimizer commit did not advance exactly one applied-update step: "
                    f"expected={update_boundary.committed_step}, actual={self._iter}"
                )
            # Rejected updates return before this point and cannot mutate either moving snapshot.
            state_events = self._policy_state.after_commit(update_boundary)
            metrics = self._reduce_metrics(metrics)  # consistent logged scalar across the shard group
            metrics["train/learning_rate"] = update_learning_rate
            metrics["optim/old_policy_refreshed"] = float(state_events.old_policy_refreshed)
            metrics["optim/reference_refreshed"] = float(state_events.reference_refreshed)
            metrics.update(self._outcome_metrics("applied"))
            self.callbacks.on_train_step(self, metrics)  # observers fire per optimizer step (§2.6)
        else:
            # Intermediate microbatches report accumulation without advancing the optimizer clock.
            metrics = self._reduce_metrics(metrics)
            metrics.update(self._outcome_metrics("accumulating"))
        return metrics

    def pending_validation_step(self) -> int | None:
        """Model step whose held-out pass must finish before training can continue."""
        return self.validation_state.progress.pending_step

    def pending_validation_metrics(self) -> dict[str, int | float | bool | str]:
        """Canonical train frame retained for an interrupted validation boundary."""
        return dict(self.validation_state.progress.pending_metrics)

    def pending_validation_publication_step(self) -> int | None:
        """Completed model-validation step whose checkpoint publication must be retried."""
        return self.validation_publication.pending_step

    def mark_validation_pending(
        self,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        """Persist the phase identity before evaluation or an emergency checkpoint."""
        self.validation_publication.assert_can_begin()
        self.validation_state.begin(model_step, trigger_reasons, metrics=metrics)

    def complete_validation(self, model_step: int) -> None:
        """Move a successful pass into the recoverable checkpoint-publication phase."""
        self.validation_publication.complete(model_step)

    def complete_ephemeral_validation(self, model_step: int) -> None:
        """Clear an initial pass without creating a checkpoint-publication phase."""
        if self.validation_publication.pending_step is not None:
            raise RuntimeError(
                f"validation step {self.validation_publication.pending_step} is already awaiting publication"
            )
        self.validation_state.progress.complete(model_step)

    def save_validation_progress(self, path: str) -> None:
        """Publish a completed held-out pass and clear its durable recovery marker."""
        self._require_checkpoint_boundary()
        self._refresh_validation_identity()
        self.validation_publication.publish(
            path,
            manager=self.checkpoint_manager,
            publication_identity={"reference_identity": self.reference_identity.state_dict()},
            trainer_state=self._trainer_state,
            on_checkpoint=lambda: self.callbacks.on_checkpoint(
                self,
                self.checkpoint_manager.iteration_location(path),
            ),
        )

    def validate(
        self,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
    ) -> ValidationReport:
        """Evaluate a transaction-owned pending phase over one finite preference inventory."""
        if self.validation_runner is None:
            raise RuntimeError("this DiffusionTrainer has no held-out objective validation source")
        self.validation_runner.objective = self.objective
        return self.validation_runner.run(
            model_step=model_step,
            iteration=self._iter,
            accumulation=self._accum,
            trigger_reasons=trigger_reasons,
        )

    def _reduce_metrics(self, metrics: dict) -> dict:
        """Reduce metric tensors so every replica reports the same public values."""
        import torch.distributed as dist

        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return self.objective.adapt_metrics(metrics, "train")

        def _local_metric(tensor: torch.Tensor) -> torch.Tensor:
            to_local = getattr(tensor, "to_local", None)
            return to_local() if callable(to_local) else tensor  # [*local metric]

        out = {key: value for key, value in metrics.items() if not isinstance(value, torch.Tensor)}
        scalar_items = sorted(
            ((key, value) for key, value in metrics.items() if isinstance(value, torch.Tensor) and value.numel() == 1),
            key=lambda item: item[0],
        )
        # Every rank must expose the same tensor metric keys and shapes. Sorting makes
        # collective order deterministic; it does not make rank-conditional schemas safe.
        device = collective_device()
        if scalar_items:
            packed = torch.stack(
                [_local_metric(value).detach().float().reshape(()).to(device) for _, value in scalar_items]
            )  # [M]
            dist.all_reduce(packed, op=dist.ReduceOp.AVG)
            for index, (key, _value) in enumerate(scalar_items):
                out[key] = packed[index]  # []
        for key, value in sorted(metrics.items()):
            if not isinstance(value, torch.Tensor) or value.numel() == 1:
                continue
            reduced = _local_metric(value).detach().float().to(device)  # [*local metric.shape]
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            out[key] = reduced
        return self.objective.adapt_metrics(out, "train")

    def offload(self) -> float:
        """Move update-only state to host memory while the shared policy net samples.

        The rollout samples on ``model.net``, so that net stays resident. The frozen old/reference
        snapshots, the EMA, the optimizer state and any accumulated gradients move; gradients move
        because online NFT interleaves a rollout between the accumulation rows of one window.
        Returns the MiB moved.
        """
        if self._accum == 0:
            # A completed window's gradients are dead: the next window starts by zeroing them.
            self.opt.zero_grad(set_to_none=True)
        snapshots = [descriptor.module for descriptor in self._policy_state.offload_descriptors()]
        return self._host_offload.offload(
            snapshots,
            optimizer=self.opt,
            grads_of=self.model.net.parameters(),
        )

    def onload(self) -> float:
        """Move state parked by ``offload`` back to the GPU. Returns the MiB moved."""
        return self._host_offload.onload()

    def push_weights(self, target=None) -> None:
        return None  # DPO: no rollout edge (§8.2 #4 seam)

    def abort_accumulation(self) -> bool:
        """Discard and rewind one partial, unapplied diffusion attempt.

        An upstream source, rollout, reward, or replay failure may arrive after
        several microbatches, but the attempt clock advances only once when the
        window opens. Remove exactly that one incomplete attempt.
        """
        completed_attempts = self._iter + self._nonfinite_skips_total
        if self._attempt_id == completed_attempts:
            if self._accum != 0:
                raise RuntimeError(
                    "quiescent diffusion attempt clock has a nonempty accumulation window: "
                    f"attempt_id={self._attempt_id}, optimizer_step={self._iter}, "
                    f"nonfinite_skips_total={self._nonfinite_skips_total}, accum={self._accum}"
                )
            return False
        if self._attempt_id != completed_attempts + 1:
            raise RuntimeError(
                "open diffusion accumulation has an invalid attempt clock: "
                f"attempt_id={self._attempt_id}, optimizer_step={self._iter}, "
                f"nonfinite_skips_total={self._nonfinite_skips_total}, accum={self._accum}"
            )
        self.opt.zero_grad(set_to_none=True)
        self._accum = 0
        self._attempt_id -= 1
        return True

    def shared_weight_identity(self) -> str | None:
        """Prove that a byte-free rollout edge shares this exact policy network."""
        backend = getattr(self, "_weight_backend", None)
        if backend is not None:
            from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

            if not isinstance(backend.mover, SharedMover):
                return None
        from cosmos_framework.posttrain.comm.weight_transfer import shared_model_identity

        return shared_model_identity(self.model.net)

    def _refresh_ref(self) -> None:
        self._policy_state.refresh_ref()

    def _refresh_old(self) -> None:
        """Refresh the behavior snapshot through the policy adapter's reshard-aware copy."""
        self._policy_state.refresh_old()

    # Checkpoint extraction is engine-specific; layout and publication are shared.
    def validation_source_identity(self) -> PreferenceSourceIdentity:
        """Return the held-out preference identity for the data-role leakage check."""
        if self.validation_runner is None:
            raise RuntimeError("preference validation is disabled")
        return self.validation_runner.source_identity()

    def _trainer_state(self) -> dict[str, Any]:
        from cosmos_framework.posttrain.checkpoint import collect_rng_state

        def _state_without_rng() -> dict[str, Any]:
            self._refresh_validation_identity()
            self.validation_publication.validate_checkpoint_state()
            prepared: dict[str, Any] = {
                **self._update_state_dict(),
                "ema_updates_enabled": self.update_ema_after_iter,
                "ref_refresh_timing": _REF_REFRESH_TIMING,
                "old_refresh_timing": _OLD_REFRESH_TIMING,
                "old_refresh_every_iter": self.update_old_every_iter,
                "reference_identity": self.reference_identity.state_dict(),
                "validation": self.validation_state.state_dict(
                    version=3,
                    extra={"publication": self.validation_publication.checkpoint_record()},
                ),
            }
            old_policy_checkpoint = self._old_policy_checkpoint_record()
            if old_policy_checkpoint is not None:
                prepared["old_policy_checkpoint"] = old_policy_checkpoint
            if self.sched is not None:
                prepared["sched"] = self.sched.state_dict()
            scheduler_identity = scheduler_resume_identity(self.sched)
            if scheduler_identity is not None:
                prepared["scheduler_identity"] = scheduler_identity
            return prepared

        state = run_rank_converged(_state_without_rng, "diffusion trainer state serialization preflight")
        state["rng"] = collect_rng_state(cuda_device=self.device if self.device.type == "cuda" else None)
        return state

    def save(
        self,
        path: str,
        data_state: dict[str, Any] | None = None,
        *,
        publish: bool = True,
        update_latest: bool = True,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
    ) -> None:
        """Save selected logical state.

        Consolidated state is portable; DCP state can reshard. Frozen references are rebuilt from
        base weights, while moving old/reference networks are checkpointed.
        """
        self._require_checkpoint_boundary()
        # State-dict extraction gathers over the device mesh, so offloaded shards must come back first.
        self.onload()
        component_plan = self.checkpoint_manager.component_plan(components)
        model_selected = component_plan.includes("model")
        optimizer_selected = component_plan.includes("optimizer")
        training_state_selected = component_plan.includes("training_state")
        moving_old = training_state_selected and self.update_old_every_iter > 0
        moving_ref = training_state_selected and self.update_ref_every_iter > 0
        descriptors = self._policy_state.descriptors()
        descriptors_by_name = {descriptor.name: descriptor for descriptor in descriptors}
        policy_descriptor = next(descriptor for descriptor in descriptors if descriptor.semantic_role == "policy")
        nets = (
            run_rank_converged(self._checkpoint_nets, "diffusion checkpoint network extraction")
            if model_selected or moving_old or moving_ref
            else {}
        )
        net_plan = rank_identical(tuple(nets), "diffusion checkpoint network plan")
        if model_selected and policy_descriptor.name not in net_plan:
            raise RuntimeError("model adapter exposes no `net` — a checkpoint without weights is never valid")
        old_policy_is_derived = bool(
            moving_old
            and rank_identical(self._iter, "diffusion old-policy checkpoint optimizer step")
            % self.update_old_every_iter
            == 0
        )
        persisted_descriptors = self._policy_state.checkpoint_descriptors(
            model_selected=model_selected,
            training_state_selected=training_state_selected,
            materialize_old=bool(moving_old and (not old_policy_is_derived or not model_selected)),
        )
        persisted_plan = rank_identical(
            tuple(descriptor.name for descriptor in persisted_descriptors),
            "diffusion checkpoint persisted network plan",
        )
        persisted_descriptors = tuple(descriptors_by_name[name] for name in persisted_plan)
        persisted_nets = {descriptor.name: nets[descriptor.name] for descriptor in persisted_descriptors}
        from cosmos_framework.posttrain.utils.rank_state import collect_rank_state

        data_component = (  # [*rank-local tensor] values
            collect_rank_state(data_state) if training_state_selected else None
        )

        def _training_state_component() -> dict[str, Any] | None:
            if not training_state_selected:
                return None
            trainer_state = self._trainer_state()
            if moving_old and old_policy_is_derived and not model_selected:
                # A full checkpoint can rebuild a refresh-aligned old policy from its model. A
                # training-state-only artifact must instead carry old bytes and a matching marker.
                trainer_state["old_policy_checkpoint"] = {
                    "version": _OLD_POLICY_CHECKPOINT_VERSION,
                    "mode": _OLD_POLICY_MATERIALIZED_MODE,
                    "optimizer_step": self._iter,
                }
            return trainer_state

        if self.checkpoint_format == "dcp":
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict

            component_states: dict[str, dict[str, Any]] = {"model": {}, "old": {}, "ref": {}}
            for descriptor in persisted_descriptors:
                module = descriptor.module
                destination = component_states[descriptor.checkpoint_component]
                module_state = run_rank_converged(  # [*sharded parameter] values
                    lambda module=module: get_model_state_dict(module),
                    f"diffusion DCP model state extraction ({descriptor.name})",
                )
                for key, value in module_state.items():  # SHARDED (DTensors under FSDP)
                    destination[f"{descriptor.name}.{key}"] = value
            optim_state = (
                run_rank_converged(  # [*sharded optimizer-state] values
                    lambda: get_optimizer_state_dict(self.model.net, self.opt),
                    "diffusion DCP optimizer state extraction",
                )
                if optimizer_selected
                else None
            )
            tail_components = (
                {"trainer": _training_state_component(), "data": data_component} if training_state_selected else {}
            )
            live_tail_state = (
                run_rank_converged(
                    lambda: (
                        self.opt.param_groups,
                        self.sched.state_dict() if self.sched is not None else {},
                        data_state,
                    ),
                    "diffusion DCP async-tail live state preparation",
                )
                if training_state_selected
                else ()
            )
            location = self._save_dcp_components(
                path,
                {
                    "model": component_states["model"],
                    "old": component_states["old"] or None,
                    "ref": component_states["ref"] or None,
                    "optim": optim_state,
                },
                tail_components,
                live_tail_state,
                component_plan=component_plan,
                publish=publish,
                update_latest=update_latest,
            )
        else:
            sharded = bool((persisted_nets or optimizer_selected) and is_dtensor_sharded(self.model.net))
            if sharded:
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_model_state_dict,
                    get_optimizer_state_dict,
                )

                opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                net_sds = {  # [*parameter] values
                    name: get_model_state_dict(m, options=opts) for name, m in persisted_nets.items()
                }
                opt_sd = (  # [*optimizer-state] values
                    get_optimizer_state_dict(self.model.net, self.opt, options=opts) if optimizer_selected else None
                )
            else:
                net_sds = {name: m.state_dict() for name, m in persisted_nets.items()}  # [*parameter] values
                opt_sd = self.opt.state_dict() if optimizer_selected else None  # [*optimizer-state] values
            component_sds: dict[str, dict[str, Any]] = {"model": {}, "old": {}, "ref": {}}
            for descriptor in persisted_descriptors:
                component_sds[descriptor.checkpoint_component][descriptor.name] = net_sds[descriptor.name]
            components = {
                "model": component_sds["model"] or None,
                "old": component_sds["old"] or None,
                "ref": component_sds["ref"] or None,
                "optim": opt_sd,
                "trainer": _training_state_component(),
                "data": data_component if training_state_selected else None,
            }
            async_safe_snapshot = bool(
                sharded
                and rank0_snapshot_is_immutable_cpu(
                    components,
                    (
                        tuple(  # [*parameter] values
                            parameter for module in persisted_nets.values() for parameter in module.parameters()
                        ),
                        tuple(buffer for module in persisted_nets.values() for buffer in module.buffers()),  # [*buffer]
                        self.opt.state if optimizer_selected else {},
                        self.opt.param_groups if optimizer_selected else (),
                        self.sched.state_dict() if training_state_selected and self.sched is not None else {},
                        data_state if training_state_selected else None,
                    ),
                )
            )
            location = self.checkpoint_manager.save_consolidated(
                path,
                components,
                # Plain state aliases live tensors; only offloaded FSDP snapshots are safe for async IO.
                async_safe_snapshot=async_safe_snapshot,
                component_plan=component_plan,
                publish=publish,
                update_latest=update_latest,
            )
        if publish:
            self.callbacks.on_checkpoint(self, location)

    def load(
        self,
        path: str,
        *,
        copy_mode: bool = False,
        strict: bool = True,
        copy_inventory_policy: CopyInventoryPolicy | None = None,
    ) -> int:
        """Load exact trainer state, or initialize every configured policy copy in copy mode."""
        if not copy_mode and copy_inventory_policy is not None:
            raise ValueError("copy_inventory_policy is valid only with copy_mode=True")
        if copy_mode and not strict:
            raise ValueError("frozen-reference copy_mode requires strict=True")
        inventory_policy = copy_inventory_policy or CopyInventoryPolicy()
        checkpoint_reader = (
            self.checkpoint_manager.reader_for_initialization(path) if copy_mode else self.checkpoint_manager
        )
        dcp_checkpoint = checkpoint_reader.has_dcp_component(path, "model")
        if dcp_checkpoint:
            resumed = self._load_dcp(
                path,
                copy_mode=copy_mode,
                copy_inventory_policy=inventory_policy,
                checkpoint_reader=checkpoint_reader,
            )
            if copy_mode:
                self.reference_identity.bind_after_copy()
            else:
                self._resumed_from_checkpoint = True
            return resumed

        if os.path.isfile(path):  # legacy single-file checkpoint (pre component-layout)
            # Trusted legacy pickle uses mmap to avoid materializing the full checkpoint per rank.
            ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            nets_sd = dict(ckpt["nets"])
            opt_sd = ckpt.get("opt")
            trainer_state = {k: ckpt[k] for k in ("iter", "accum", "sched") if k in ckpt}
        else:
            nets_sd = dict(checkpoint_reader.load_component(path, "model") or {})
            nets_sd.update(checkpoint_reader.load_component(path, "old") or {})
            nets_sd.update(checkpoint_reader.load_component(path, "ref") or {})
            opt_sd = checkpoint_reader.load_component(path, "optim")
            trainer_state = checkpoint_reader.load_component(path, "trainer") or {}
        descriptors = self._policy_state.descriptors()
        policy_descriptor = next(descriptor for descriptor in descriptors if descriptor.semantic_role == "policy")
        old_policy_is_derived = False
        if not copy_mode:
            run_rank_converged(
                lambda: self._validate_ema_resume_contract(trainer_state),
                "EMA resume contract validation",
            )
            run_rank_converged(
                lambda: self._validate_scheduler_resume_contract(trainer_state),
                "scheduler resume contract validation",
            )
            old_policy_is_derived = bool(
                run_rank_converged(
                    lambda: self._validate_old_resume_contract(trainer_state),
                    "moving-old-policy resume contract validation",
                )
            )
        if policy_descriptor.name not in nets_sd:
            raise _missing_resume_error(policy_descriptor, path, copy_mode=copy_mode, dcp=False)

        sharded = is_dtensor_sharded(self.model.net)
        set_model, set_opt = self._dist_setters() if sharded else (None, None)

        def _load_net(module: torch.nn.Module, sd: dict) -> None:
            if sharded:
                set_model(module, sd, options=self._full_opts(strict))
            else:
                module.load_state_dict(sd, strict=strict)

        if copy_mode:
            base = nets_sd[policy_descriptor.name]
            targets = self._policy_state.copy_descriptors()

            def _target_state(module: torch.nn.Module) -> dict[str, Any]:
                if not sharded:
                    return dict(module.state_dict())
                from torch.distributed.checkpoint.state_dict import get_model_state_dict

                return get_model_state_dict(module)

            source_shapes = _tensor_shapes(base)

            def _plan_all_targets() -> list[_CopyPlan]:
                return [
                    _build_copy_plan(
                        source_shapes,
                        _tensor_shapes(_target_state(descriptor.module)),
                        inventory_policy,
                        role=descriptor.semantic_role,
                    )
                    for descriptor in targets
                ]

            plans = run_rank_converged(_plan_all_targets, "diffusion copy-mode inventory validation")
            _require_identical_copy_plans(plans)
            for plan, descriptor in zip(plans, targets, strict=True):
                if plan.role != descriptor.semantic_role:
                    raise RuntimeError("copy-mode target inventory order changed between validation and loading")
                selected = {key: base[key] for key in plan.load_keys}
                if sharded:
                    set_model(descriptor.module, selected, options=self._full_opts(False))
                else:
                    descriptor.module.load_state_dict(selected, strict=False)
                if descriptor.semantic_role != "policy":
                    descriptor.module.eval()
            self.reference_identity.bind_after_copy()
            return 0
        if opt_sd is None:
            raise RuntimeError(
                f"checkpoint {path!r} has no optimizer state; set rl.init_from to this checkpoint "
                "for model-only initialization"
            )
        old_descriptors = tuple(descriptor for descriptor in descriptors if descriptor.persistence == "moving_old")
        if old_policy_is_derived and any(descriptor.name in nets_sd for descriptor in old_descriptors):
            raise RuntimeError("policy-derived moving-old checkpoint must not contain an old component")
        resume_descriptors = self._policy_state.exact_resume_descriptors(old_policy_is_derived=old_policy_is_derived)
        for descriptor in resume_descriptors:
            if descriptor.name not in nets_sd:
                raise _missing_resume_error(descriptor, path, dcp=False)
        for descriptor in resume_descriptors:
            _load_net(descriptor.module, nets_sd[descriptor.name])
        if opt_sd is not None:
            # Adam state is lazy, so optimizer restore is partial while model restore remains strict.
            if sharded:
                set_opt(self.model.net, self.opt, optim_state_dict=opt_sd, options=self._full_opts(False))
            else:
                self.opt.load_state_dict(opt_sd)
        resumed = self._restore_trainer_state(trainer_state)
        if old_policy_is_derived:
            self._refresh_old()
        self._resumed_from_checkpoint = True
        return resumed

    def _restore_trainer_state(self, trainer_state: dict[str, Any]) -> int:
        """Restore durable control state, converging rank-local failures before use."""
        restored_iteration, needs_legacy_ref_refresh = run_rank_converged(
            lambda: self._restore_trainer_state_local(trainer_state),
            "diffusion trainer state restore",
        )

        # State restore converges before a moving-reference collective.
        if needs_legacy_ref_refresh:
            self._refresh_ref()
        return restored_iteration

    def _restore_trainer_state_local(self, trainer_state: dict[str, Any]) -> tuple[int, bool]:
        """Apply rank-local trainer state; the public wrapper owns failure consensus."""
        self._validate_scheduler_resume_contract(trainer_state)
        update_state = self._parse_update_state(trainer_state)
        self._validate_old_resume_contract(trainer_state)
        refresh_timing = trainer_state.get("ref_refresh_timing")
        if self.update_ref_every_iter > 0 and refresh_timing not in (None, _REF_REFRESH_TIMING):
            raise RuntimeError(f"unsupported moving-reference refresh timing {refresh_timing!r}")
        if self.sched is not None and trainer_state.get("sched") is not None:
            self.sched.load_state_dict(trainer_state["sched"])
        self._refresh_validation_identity()
        validation_state = trainer_state.get("validation")
        if validation_state is None:
            if self.validation_source is not None:
                raise RuntimeError(
                    "checkpoint predates the validation identity journal; exact resume cannot enable "
                    "validation on an existing run (use rl.init_from to start a new tuning run)"
                )
            self.validation_state.progress = ValidationProgress()
            self.validation_publication.restore(None, iteration=update_state.optimizer_step)
        elif isinstance(validation_state, dict):
            extra = self.validation_state.load_state_dict(
                validation_state,
                extra_fields_by_version={
                    1: frozenset(),
                    3: frozenset({"publication"}),
                },
            )
            self.validation_publication.restore(
                extra.get("publication"),
                iteration=update_state.optimizer_step,
            )
        else:
            raise RuntimeError("trainer checkpoint validation state has an unsupported schema")
        self.reference_identity.validate_restored(trainer_state.get("reference_identity"))
        self._apply_update_state(
            update_state,
            rng_state=trainer_state.get("rng"),
            cuda_device=self.device if self.device.type == "cuda" else None,
        )
        needs_legacy_ref_refresh = bool(
            self.update_ref_every_iter > 0 and refresh_timing is None and self._iter % self.update_ref_every_iter == 0
        )
        # Convert legacy refresh timing only after every rank accepts the state.
        return self._iter, needs_legacy_ref_refresh

    def _refresh_validation_identity(self) -> None:
        """Bind checkpoint comparisons to the objective's current semantic identity."""
        self.validation_state.identity = self._current_validation_identity()

    def _current_validation_identity(self) -> dict[str, Any]:
        return build_objective_validation_identity(
            self.validation_source,
            batch_size=self.validation_batch_size,
            max_samples=self.validation_max_samples,
            seed=self.validation_seed,
            objective=self.objective,
            before_training=self.validation_before_training,
            every_steps=self.validation_every_steps,
            train_total_steps=self.validation_train_total_steps,
        )

    def _load_dcp(
        self,
        path: str,
        *,
        copy_mode: bool = False,
        copy_inventory_policy: CopyInventoryPolicy | None = None,
        checkpoint_reader: CheckpointManager | None = None,
    ) -> int:
        """Restore live sharded state without replacing an unsaved frozen anchor."""
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        inventory_policy = copy_inventory_policy or CopyInventoryPolicy()
        checkpoint_reader = checkpoint_reader or self.checkpoint_manager
        loaded_trainer_state = checkpoint_reader.load_component(path, "trainer")
        if loaded_trainer_state is None and not copy_mode:
            raise RuntimeError(f"training resume requires trainer state in DCP checkpoint {path!r}")
        trainer_state = loaded_trainer_state or {}
        descriptors = self._policy_state.descriptors()
        descriptors_by_name = {descriptor.name: descriptor for descriptor in descriptors}
        policy_descriptor = next(descriptor for descriptor in descriptors if descriptor.semantic_role == "policy")
        old_policy_is_derived = False
        if not copy_mode:
            run_rank_converged(
                lambda: self._validate_ema_resume_contract(trainer_state),
                "EMA resume contract validation",
            )
            run_rank_converged(
                lambda: self._validate_scheduler_resume_contract(trainer_state),
                "scheduler resume contract validation",
            )
            old_policy_is_derived = bool(
                run_rank_converged(
                    lambda: self._validate_old_resume_contract(trainer_state),
                    "moving-old-policy resume contract validation",
                )
            )
            old_descriptors = tuple(descriptor for descriptor in descriptors if descriptor.persistence == "moving_old")
            if old_policy_is_derived and any(
                checkpoint_reader.has_dcp_component(path, descriptor.checkpoint_component)
                for descriptor in old_descriptors
            ):
                raise RuntimeError("policy-derived moving-old checkpoint must not contain an old component")
        selected_descriptors = (
            self._policy_state.copy_descriptors()
            if copy_mode
            else self._policy_state.exact_resume_descriptors(old_policy_is_derived=old_policy_is_derived)
        )
        load_plan = rank_identical(
            tuple(descriptor.name for descriptor in selected_descriptors),
            "diffusion DCP component load plan",
        )
        load_descriptors = tuple(descriptors_by_name[name] for name in load_plan)

        copy_plans: list[_CopyPlan] = []

        def _validate_network_inventory() -> list[_CopyPlan]:
            required: list[tuple[str, str, PolicyStateDescriptor]] = []
            if copy_mode:
                required.extend(
                    (
                        policy_descriptor.checkpoint_component,
                        policy_descriptor.name,
                        descriptor,
                    )
                    for descriptor in load_descriptors
                )
            else:
                required.extend(
                    (descriptor.checkpoint_component, descriptor.name, descriptor) for descriptor in load_descriptors
                )
            plans: list[_CopyPlan] = []
            inventories: dict[str, dict[str, tuple[int, ...] | None]] = {}
            for component, saved_name, descriptor in required:
                if not checkpoint_reader.has_dcp_component(path, component):
                    raise _missing_resume_error(descriptor, path, copy_mode=copy_mode)
                if component not in inventories:
                    inventories[component] = checkpoint_reader.dcp_tensor_inventory(path, component)
                inventory = inventories[component]
                available = set(inventory)
                if copy_mode:
                    prefix = f"{saved_name}."
                    source_shapes = {
                        key[len(prefix) :]: shape
                        for key in sorted(available)
                        if (shape := inventory[key]) is not None
                        if key.startswith(prefix)
                    }
                    if not source_shapes:
                        raise _missing_resume_error(policy_descriptor, path, copy_mode=True)
                    target_state = get_model_state_dict(descriptor.module)
                    plans.append(
                        _build_copy_plan(
                            source_shapes,
                            _tensor_shapes(target_state),
                            inventory_policy,
                            role=descriptor.semantic_role,
                        )
                    )
                    continue
                target_keys = run_rank_converged(
                    lambda descriptor=descriptor: tuple(get_model_state_dict(descriptor.module)),
                    f"diffusion DCP inventory target extraction ({component}/{saved_name})",
                )
                expected = {f"{saved_name}.{key}" for key in target_keys}
                missing = sorted(expected - available)
                prefix = f"{saved_name}."
                unexpected = sorted(key for key in available - expected if key.startswith(prefix))
                if missing or unexpected:
                    problems: list[str] = []
                    if missing:
                        preview = ", ".join(missing[:5])
                        problems.append(f"missing {len(missing)} required tensors: {preview}")
                    if unexpected:
                        preview = ", ".join(unexpected[:5])
                        problems.append(f"found {len(unexpected)} unexpected tensors: {preview}")
                    raise RuntimeError(
                        f"DCP checkpoint {path!r} has an incompatible {saved_name} tensor inventory: "
                        + "; ".join(problems)
                    )
            return plans

        if copy_mode:
            copy_plans = run_rank_converged(
                _validate_network_inventory,
                "diffusion DCP network inventory validation",
            )
        else:
            copy_plans = _validate_network_inventory()
        if copy_mode:
            _require_identical_copy_plans(copy_plans)

        def _load_net_from(
            component: str,
            saved_name: str,
            module: torch.nn.Module,
            *,
            load_keys: tuple[str, ...] | None = None,
            set_eval: bool = False,
        ) -> bool:
            module_state = run_rank_converged(  # [*sharded parameter] values
                lambda: get_model_state_dict(module),
                f"diffusion DCP target extraction ({component}/{saved_name})",
            )
            selected_keys = tuple(module_state) if load_keys is None else load_keys
            target = run_rank_converged(  # [*sharded parameter] values
                lambda: {f"{saved_name}.{key}": module_state[key] for key in selected_keys},
                f"diffusion DCP target selection ({component}/{saved_name})",
            )
            loaded = checkpoint_reader.load_dcp_component(path, component, target)
            if not loaded:
                return False

            def _apply() -> None:
                # DCP copies into the destination-shaped state, so saved policy tensors are cast to
                # the target module's dtype (notably bf16 policy -> fp32 EMA) before installation.
                state = {key[len(saved_name) + 1 :]: value for key, value in target.items()}  # [*sharded parameter]
                if load_keys is None:
                    set_model_state_dict(module, state)  # [*sharded parameter] module values
                else:
                    from torch.distributed.checkpoint.state_dict import StateDictOptions

                    set_model_state_dict(  # [*sharded parameter] module values
                        module,
                        state,
                        options=StateDictOptions(strict=False),
                    )
                if set_eval:
                    module.eval()

            run_rank_converged(
                _apply,
                f"diffusion DCP state application ({component}/{saved_name})",
            )
            return True

        if copy_mode:
            if len(load_descriptors) != len(copy_plans):
                raise RuntimeError("copy-mode target inventory changed between validation and loading")
            for plan, descriptor in zip(copy_plans, load_descriptors, strict=True):
                if plan.role != descriptor.semantic_role:
                    raise RuntimeError("copy-mode target inventory order changed between validation and loading")
                if not _load_net_from(
                    policy_descriptor.checkpoint_component,
                    policy_descriptor.name,
                    descriptor.module,
                    load_keys=plan.load_keys,
                    set_eval=descriptor.semantic_role != "policy",
                ):
                    raise _missing_resume_error(policy_descriptor, path, copy_mode=True)
            return 0
        if not _load_net_from(
            policy_descriptor.checkpoint_component,
            policy_descriptor.name,
            policy_descriptor.module,
        ):
            raise _missing_resume_error(policy_descriptor, path)
        for descriptor in load_descriptors:
            if descriptor.semantic_role == "policy":
                continue
            loaded = _load_net_from(
                descriptor.checkpoint_component,
                descriptor.name,
                descriptor.module,
            )
            if not loaded:
                raise _missing_resume_error(descriptor, path)
        opt_state = run_rank_converged(  # [*sharded optimizer-state] values
            lambda: get_optimizer_state_dict(self.model.net, self.opt),
            "diffusion DCP optimizer target extraction",
        )
        # Never-gradded parameters legitimately have no saved Adam state.
        optim_loaded = checkpoint_reader.load_dcp_component(path, "optim", opt_state, allow_partial=True)
        if not optim_loaded:
            raise RuntimeError(
                f"training resume requires optimizer state in DCP checkpoint {path!r}; "
                "use rl.init_from for model-only initialization"
            )
        run_rank_converged(
            lambda: set_optimizer_state_dict(  # [*sharded optimizer-state] optimizer values
                self.model.net,
                self.opt,
                optim_state_dict=opt_state,
            ),
            "diffusion DCP optimizer state application",
        )
        resumed = self._restore_trainer_state(trainer_state)
        if old_policy_is_derived:
            run_rank_converged(self._refresh_old, "derived moving-old policy reconstruction")
        return resumed

    @staticmethod
    def _dist_setters():
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, set_optimizer_state_dict

        return set_model_state_dict, set_optimizer_state_dict

    @staticmethod
    def _full_opts(strict: bool):
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        return StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
